"""Tests for the TDI (classification) multitask arms.

The regression tests in `test_multitask.py` cover the shared-fold machinery itself,
which this path reuses unchanged. What is new here is the classification schema and
the classification-specific traps:

- **Probabilities must survive to the OOF frame.** TDI is ~21% positive and MCC is
  not optimized at a 0.5 cutoff, so a harness that recorded only hard labels would
  force a 25-fold refit to tune the threshold. `y_prob` is the column that prevents
  that, and several tests below pin it.
- **Both arms must be configured identically.** An arm trained for more epochs than
  its twin measures the epochs, not the multitask effect, so the single-task runner
  refuses settings it cannot apply rather than dropping them silently.
- **TFMs must never be fitted in-process.** That collision cost 14.5 silent hours
  once already (CLAUDE.md, "Environment traps"); the routing is asserted here so it
  cannot regress into a notebook's inline helper again.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cyp import cv, multitask

#: Same distinct-ring-system SMILES as the regression tests: acyclic molecules all
#: give Murcko scaffold "" and collapse into one bucket, making the splits degenerate.
_SMILES = {
    "m1": "c1ccccc1CCO",
    "m2": "c1ccncc1CC",
    "m3": "c1ccsc1CN",
    "m4": "C1CCCCC1CCl",
    "m5": "c1cc2ccccc2cc1CBr",
    "m6": "C1CCNCC1CF",
}


def _tdi_frame(names: list[str], labels: list[bool]) -> pl.DataFrame:
    """A frame in `data.tdi_training_frame`'s schema: no interval columns, boolean
    target. The absence of y_lower/y_upper is why the regression runners cannot be
    reused."""
    return pl.DataFrame(
        {
            "Molecule_Name": names,
            "SMILES": [_SMILES[n] for n in names],
            "y_true": labels,
        }
    )


@pytest.fixture
def toy_tdi() -> dict[str, pl.DataFrame]:
    """Two isoforms with partial overlap, mirroring the real 5.4% shared fraction --
    `m2` and `m3` are the compounds carrying both labels."""
    return {
        "CYP3A4": _tdi_frame(["m1", "m2", "m3", "m4"], [True, False, True, False]),
        "CYP2D6": _tdi_frame(["m2", "m3", "m5", "m6"], [False, True, True, False]),
    }


# ── schema ─────────────────────────────────────────────────────────────────────


def test_classification_oof_has_no_interval_columns(toy_tdi) -> None:
    """A TDI label carries no credible interval, and MCC does not use one. Emitting
    y_lower/y_upper here would invite scoring code meant for ST-RAE to run on it."""
    _, table = cv.shared_scaffold_folds(toy_tdi, n_outer=1, n_inner=2)
    oof = multitask.run_cv_multitask_classification(toy_tdi, "lgbm", n_bits=64, assignments=table)
    assert "y_lower" not in oof.columns
    assert "y_upper" not in oof.columns
    assert set(oof.columns) == {
        "method",
        "endpoint",
        "fold",
        "outer_fold",
        "inner_fold",
        "Molecule_Name",
        "y_true",
        "y_pred",
        "y_prob",
    }


def test_labels_are_boolean_not_float(toy_tdi) -> None:
    _, table = cv.shared_scaffold_folds(toy_tdi, n_outer=1, n_inner=2)
    oof = multitask.run_cv_multitask_classification(toy_tdi, "lgbm", n_bits=64, assignments=table)
    assert oof.schema["y_true"] == pl.Boolean
    assert oof.schema["y_pred"] == pl.Boolean


def test_probabilities_are_recorded_for_threshold_tuning(toy_tdi) -> None:
    """The column that makes a threshold sweep possible without refitting. MCC on a
    ~21%-positive label is not optimized at 0.5, so this is load-bearing."""
    _, table = cv.shared_scaffold_folds(toy_tdi, n_outer=1, n_inner=2)
    oof = multitask.run_cv_multitask_classification(toy_tdi, "lgbm", n_bits=64, assignments=table)
    assert oof.schema["y_prob"] == pl.Float64
    assert oof["y_prob"].min() >= 0.0
    assert oof["y_prob"].max() <= 1.0


def test_threshold_controls_the_recorded_label(toy_tdi) -> None:
    """y_pred must be y_prob thresholded, not an independent prediction -- otherwise
    a threshold sweep over y_prob would not reproduce the harness's own labels."""
    _, table = cv.shared_scaffold_folds(toy_tdi, n_outer=1, n_inner=2)
    oof = multitask.run_cv_multitask_classification(
        toy_tdi, "lgbm", n_bits=64, assignments=table, threshold=0.3
    )
    expected = oof["y_prob"].to_numpy() >= 0.3
    assert np.array_equal(oof["y_pred"].to_numpy(), expected)


def test_method_names_are_suffixed_by_arm(toy_tdi) -> None:
    """Both arms land in one frame for the paired comparison, so the suffix is the
    only thing keeping them apart."""
    _, table = cv.shared_scaffold_folds(toy_tdi, n_outer=1, n_inner=2)
    mt = multitask.run_cv_multitask_classification(toy_tdi, "lgbm", n_bits=64, assignments=table)
    st = multitask.run_cv_singletask_classification(toy_tdi, "lgbm", n_bits=64, assignments=table)
    assert mt["method"].unique().to_list() == ["lgbm_multitask"]
    assert st["method"].unique().to_list() == ["lgbm_singletask"]


# ── the arms must be comparable ────────────────────────────────────────────────


def test_both_arms_see_identical_test_rows(toy_tdi) -> None:
    """The property that makes `evaluation.paired_bootstrap` between the arms
    legitimate rather than an eyeball comparison."""
    _, table = cv.shared_scaffold_folds(toy_tdi, n_outer=1, n_inner=2)
    mt = multitask.run_cv_multitask_classification(toy_tdi, "lgbm", n_bits=64, assignments=table)
    st = multitask.run_cv_singletask_classification(toy_tdi, "lgbm", n_bits=64, assignments=table)
    key = ["endpoint", "fold", "Molecule_Name"]
    assert sorted(mt.select(key).rows()) == sorted(st.select(key).rows())


def test_single_task_rejects_settings_it_cannot_apply(toy_tdi) -> None:
    """A typo'd kwarg must fail loudly. Silently dropping it would leave the
    single-task arm configured differently from its multitask twin, and the
    difference would be attributed to multitask."""
    _, table = cv.shared_scaffold_folds(toy_tdi, n_outer=1, n_inner=2)
    with pytest.raises(TypeError, match="no attribute"):
        multitask.run_cv_singletask_classification(
            toy_tdi, "lgbm", n_bits=64, assignments=table, not_a_real_setting=3
        )


def test_single_task_refuses_kwargs_it_cannot_forward_to_a_tfm(toy_tdi) -> None:
    """The TFM branch goes through the subprocess runner, which takes explicit
    arguments -- so kwargs meant for the model cannot be honoured and must not be
    silently ignored."""
    _, table = cv.shared_scaffold_folds(toy_tdi, n_outer=1, n_inner=2)
    with pytest.raises(TypeError, match="not forwarded"):
        multitask.run_cv_singletask_classification(
            toy_tdi, "tabicl", n_bits=64, assignments=table, n_estimators=4
        )


# ── strategies ─────────────────────────────────────────────────────────────────


def test_macau_has_no_classification_strategy() -> None:
    """Macau factorizes a real-valued matrix; a boolean TDI label is not what it
    models, which is also why `models.CLASSIFIER_FACTORIES` omits it."""
    assert "macau" not in multitask.STRATEGY_CLASSIFICATION


def test_every_classification_strategy_has_a_factory() -> None:
    from cyp.models import CLASSIFIER_FACTORIES

    assert set(multitask.STRATEGY_CLASSIFICATION) <= set(CLASSIFIER_FACTORIES)


def test_graph_models_use_the_multitarget_strategy() -> None:
    """Chemprop has a real multi-head output; row-stacking it would throw that away."""
    assert multitask.STRATEGY_CLASSIFICATION["chemprop"] == "multitarget"
    assert multitask.STRATEGY_CLASSIFICATION["chemeleon"] == "multitarget"


def test_dispatch_rejects_unknown_method(toy_tdi) -> None:
    with pytest.raises(ValueError, match="No multitask classification strategy"):
        multitask.run_cv_multitask_classification(toy_tdi, "not_a_model", n_bits=64)


def test_stacked_rejects_graph_models(toy_tdi) -> None:
    """chemprop consumes SMILES, not a feature matrix -- routing it here would
    silently featurize the wrong thing."""
    _, table = cv.shared_scaffold_folds(toy_tdi, n_outer=1, n_inner=2)
    with pytest.raises(ValueError, match="run_cv_multitarget_classification"):
        multitask.run_cv_stacked_classification(
            toy_tdi, method="chemprop", n_bits=64, assignments=table
        )


# ── fold-major execution ───────────────────────────────────────────────────────


def test_folds_argument_restricts_to_those_folds(toy_tdi) -> None:
    """The mechanism behind the notebook's interleaved ordering: one method, one
    fold at a time, so a stalled run leaves partial results for every method rather
    than complete results for the first few."""
    _, table = cv.shared_scaffold_folds(toy_tdi, n_outer=1, n_inner=2)
    oof = multitask.run_cv_multitask_classification(
        toy_tdi, "lgbm", n_bits=64, assignments=table, folds=[1]
    )
    assert oof["fold"].unique().to_list() == [1]


def test_fold_slices_reassemble_into_the_whole_run(toy_tdi) -> None:
    """Running fold-by-fold and concatenating must equal running all folds at once.
    If it did not, interleaving would change the result rather than just its order."""
    _, table = cv.shared_scaffold_folds(toy_tdi, n_outer=1, n_inner=2)
    whole = multitask.run_cv_multitask_classification(toy_tdi, "lgbm", n_bits=64, assignments=table)
    pieces = pl.concat(
        [
            multitask.run_cv_multitask_classification(
                toy_tdi, "lgbm", n_bits=64, assignments=table, folds=[f]
            )
            for f in sorted(table["fold"].unique().to_list())
        ]
    )
    key = ["endpoint", "fold", "Molecule_Name"]
    assert sorted(whole.select(key).rows()) == sorted(pieces.select(key).rows())
    # And the predictions themselves, not just the row coverage.
    joined = whole.join(pieces, on=key, suffix="_piece")
    assert np.allclose(joined["y_prob"].to_numpy(), joined["y_prob_piece"].to_numpy())


def test_single_task_folds_argument_restricts_both_isoforms(toy_tdi) -> None:
    _, table = cv.shared_scaffold_folds(toy_tdi, n_outer=1, n_inner=2)
    oof = multitask.run_cv_singletask_classification(
        toy_tdi, "lgbm", n_bits=64, assignments=table, folds=[0]
    )
    assert oof["fold"].unique().to_list() == [0]
    assert set(oof["endpoint"].unique().to_list()) == set(toy_tdi)


# ── features ───────────────────────────────────────────────────────────────────


def test_stacked_rejects_misaligned_features(toy_tdi) -> None:
    _, table = cv.shared_scaffold_folds(toy_tdi, n_outer=1, n_inner=2)
    bad = {"CYP3A4": np.zeros((3, 8)), "CYP2D6": np.zeros((4, 8))}
    with pytest.raises(ValueError, match="row-aligned"):
        multitask.run_cv_stacked_classification(
            toy_tdi, method="lgbm", features=bad, assignments=table
        )


def test_single_task_rejects_misaligned_features(toy_tdi) -> None:
    _, table = cv.shared_scaffold_folds(toy_tdi, n_outer=1, n_inner=2)
    bad = {"CYP3A4": np.zeros((3, 8)), "CYP2D6": np.zeros((4, 8))}
    with pytest.raises(ValueError, match="row-aligned"):
        multitask.run_cv_singletask_classification(toy_tdi, "lgbm", features=bad, assignments=table)


def test_stacked_features_carry_endpoint_indicators(toy_tdi) -> None:
    """Without the one-hot columns the stacked model sees one compound with two
    different labels and no way to tell which isoform it is being asked about."""
    stacked, endpoints = multitask.endpoint_indicator_frame(toy_tdi)
    features = {name: np.zeros((frame.height, 5)) for name, frame in toy_tdi.items()}
    X = multitask._stacked_feature_matrix(stacked, toy_tdi, endpoints, features, "ecfp", 64)
    assert X.shape == (stacked.height, 5 + len(endpoints))
    # Every row is one-hot over the indicator block.
    assert np.all(X[:, 5:].sum(axis=1) == 1.0)


# ── progress reporting ─────────────────────────────────────────────────────────


def test_fold_callback_fires_once_per_fold(toy_tdi) -> None:
    _, table = cv.shared_scaffold_folds(toy_tdi, n_outer=1, n_inner=2)
    seen: list[int] = []
    multitask.run_cv_multitask_classification(
        toy_tdi,
        "lgbm",
        n_bits=64,
        assignments=table,
        on_fold=lambda fold, n: seen.append(fold),
    )
    assert sorted(seen) == sorted(table["fold"].unique().to_list())


def test_a_broken_progress_bar_cannot_kill_a_run(toy_tdi) -> None:
    """A multi-hour TDI run must not be discarded because a notebook widget raised.
    The fold's work is already done by the time the callback fires."""
    _, table = cv.shared_scaffold_folds(toy_tdi, n_outer=1, n_inner=2)

    def _explode(fold: int, n: int) -> None:
        raise RuntimeError("progress bar died")

    oof = multitask.run_cv_multitask_classification(
        toy_tdi, "lgbm", n_bits=64, assignments=table, on_fold=_explode
    )
    assert oof.height > 0
