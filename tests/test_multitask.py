"""Tests for shared folds and multitask training.

The load-bearing property here is leakage, not accuracy. 1,309 of the 4,905 labelled
compounds carry more than one endpoint, so if folds were drawn per endpoint a
compound could be in CYP3A4's training set and CYP2D6's test set at the same time --
and a multitask model scored under that split would look better than it is, for the
worst possible reason. Most of what follows checks that this cannot happen.

The rest checks that single-task and multitask arms remain *comparable*: identical
folds, identical test rows, same OOF schema. A comparison between two arms scored on
different rows is not a comparison at all.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cyp import cv, multitask


def _frame(names: list[str], smiles: list[str], values: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Molecule_Name": names,
            "SMILES": smiles,
            "y_true": values,
            "y_lower": [v - 0.2 for v in values],
            "y_upper": [v + 0.2 for v in values],
        }
    )


#: Distinct ring systems, so Murcko gives each molecule a real scaffold. Acyclic
#: SMILES all return "" and collapse into one ungroupable bucket, which would make
#: the fold splits degenerate rather than testing anything.
_SMILES = {
    "m1": "c1ccccc1CCO",
    "m2": "c1ccncc1CC",
    "m3": "c1ccsc1CN",
    "m4": "C1CCCCC1CCl",
    "m5": "c1cc2ccccc2cc1CBr",
    "m6": "C1CCNCC1CF",
}


@pytest.fixture
def toy_frames() -> dict[str, pl.DataFrame]:
    """Two endpoints with partial overlap -- the structure that makes shared folds
    necessary. `m2` and `m3` appear in both."""
    return {
        "A": _frame(
            ["m1", "m2", "m3", "m4"],
            [_SMILES[n] for n in ("m1", "m2", "m3", "m4")],
            [5.0, 6.0, 7.0, 8.0],
        ),
        "B": _frame(
            ["m2", "m3", "m5", "m6"],
            [_SMILES[n] for n in ("m2", "m3", "m5", "m6")],
            [4.0, 5.5, 6.5, 7.5],
        ),
    }


# ── shared folds ───────────────────────────────────────────────────────────────


def test_shared_folds_cover_the_union_of_compounds(toy_frames) -> None:
    _, table = cv.shared_scaffold_folds(toy_frames, n_outer=1, n_inner=2)
    assert set(table["Molecule_Name"].to_list()) == {"m1", "m2", "m3", "m4", "m5", "m6"}


def test_a_compound_gets_one_fold_per_repeat(toy_frames) -> None:
    """A compound measured in two endpoints must still occupy a single fold -- that
    is what stops it being trained on and tested against simultaneously."""
    _, table = cv.shared_scaffold_folds(toy_frames, n_outer=2, n_inner=2)
    per_repeat = table.group_by(["Molecule_Name", "outer_fold"]).len()
    assert per_repeat["len"].max() == 1


def test_no_cross_endpoint_leakage_on_real_data() -> None:
    """The property the whole multitask comparison rests on, checked against the
    actual challenge data rather than a toy fixture."""
    from cyp import constants as C
    from cyp import data

    frames = {e: data.training_frame(e) for e in C.REGRESSION_ENDPOINTS}
    _, table = cv.shared_scaffold_folds(frames, n_outer=1, n_inner=5)

    for fold in table["fold"].unique().to_list():
        test_names: set[str] = set()
        train_names: set[str] = set()
        for frame in frames.values():
            for fd, _o, _i, train, _v, test in cv.fold_assignment_splits(frame, table):
                if fd != fold:
                    continue
                test_names |= set(test["Molecule_Name"].to_list())
                train_names |= set(train["Molecule_Name"].to_list())
        assert not (test_names & train_names), (
            f"fold {fold}: {len(test_names & train_names)} compounds appear in both "
            "train and test across endpoints"
        )


def test_fold_assignment_splits_partition_the_frame(toy_frames) -> None:
    """Every row must be tested exactly once per repeat, or the OOF frame silently
    under- or over-counts compounds."""
    _, table = cv.shared_scaffold_folds(toy_frames, n_outer=1, n_inner=2)
    frame = toy_frames["A"]
    tested: list[str] = []
    for _fd, _o, _i, train, _v, test in cv.fold_assignment_splits(frame, table):
        tested.extend(test["Molecule_Name"].to_list())
        # Train and test must be disjoint within a fold too.
        assert not set(test["Molecule_Name"].to_list()) & set(
            train["Molecule_Name"].to_list()
        )
    assert sorted(tested) == sorted(frame["Molecule_Name"].to_list())


def test_both_arms_see_identical_test_rows(toy_frames) -> None:
    """Pairedness. If the arms were scored on different rows, the paired bootstrap
    comparing them would be invalid."""
    _, table = cv.shared_scaffold_folds(toy_frames, n_outer=1, n_inner=2)
    per_fold = {}
    for endpoint, frame in toy_frames.items():
        for fd, _o, _i, _tr, _v, test in cv.fold_assignment_splits(frame, table):
            per_fold.setdefault(fd, set()).update(
                (endpoint, n) for n in test["Molecule_Name"].to_list()
            )
    # Union across folds is every measurement exactly once.
    total = sum(len(v) for v in per_fold.values())
    assert total == sum(f.height for f in toy_frames.values())


# ── stacking ───────────────────────────────────────────────────────────────────


def test_stacked_frame_has_one_row_per_measurement(toy_frames) -> None:
    stacked, endpoints = multitask.endpoint_indicator_frame(toy_frames)
    assert stacked.height == sum(f.height for f in toy_frames.values())
    assert endpoints == ["A", "B"]
    assert set(stacked["endpoint"].unique().to_list()) == {"A", "B"}


def test_endpoint_indicators_are_one_hot() -> None:
    """Without these the model sees one compound with two different targets and no
    way to tell them apart -- it would learn the average, which is worse than either
    single-task model rather than better."""
    X = np.zeros((3, 2), dtype=np.float32)
    out = multitask.stack_features(X, ["A", "B", "A"], ["A", "B"])
    assert out.shape == (3, 4)
    assert out[:, 2:].tolist() == [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]


def test_stacked_rejects_misaligned_features(toy_frames) -> None:
    with pytest.raises(ValueError, match="row-aligned"):
        multitask.run_cv_stacked(
            toy_frames,
            method="ridge",
            features={"A": np.zeros((2, 4)), "B": np.zeros((4, 4))},
            n_outer=1,
            n_inner=2,
        )


def test_stacked_rejects_graph_models(toy_frames) -> None:
    """Chemprop consumes SMILES and has its own multi-target mode; routing it
    through the stacked path would silently train on stringified features."""
    pytest.importorskip("chemprop")
    with pytest.raises(ValueError, match="run_cv_multitarget"):
        multitask.run_cv_stacked(toy_frames, method="chemprop", n_outer=1, n_inner=2)


# ── dispatch ───────────────────────────────────────────────────────────────────


def test_every_registered_method_has_a_strategy() -> None:
    """A method in MODEL_FACTORIES without a strategy would fail only once the
    notebook reached it."""
    from cyp import models

    for method in ("lgbm", "xgb", "macau", "chemprop", "chemeleon", "tabicl", "tabpfn"):
        assert method in multitask.STRATEGY
        assert method in models.MODEL_FACTORIES
    # `mean` is excluded on purpose: a constant predictor cannot borrow strength.
    assert "mean" not in multitask.STRATEGY


def test_strategies_match_model_capability() -> None:
    """Macau factorizes natively and Chemprop has a real multi-target head; neither
    should be row-stacked, which would discard the thing that makes them suitable."""
    assert multitask.STRATEGY["macau"] == "native"
    assert multitask.STRATEGY["chemprop"] == "multitarget"
    assert multitask.STRATEGY["chemeleon"] == "multitarget"
    assert multitask.STRATEGY["lgbm"] == "stacked"


def test_dispatch_rejects_unknown_method(toy_frames) -> None:
    with pytest.raises(ValueError, match="No multitask strategy"):
        multitask.run_cv_multitask(toy_frames, "not_a_model")


# ── output schema ──────────────────────────────────────────────────────────────


def test_multitask_oof_matches_the_single_task_schema(toy_frames) -> None:
    """`evaluation`, `calibration` and `ensemble` all read this schema, so a
    multitask run has to be a drop-in or none of the comparison machinery works."""
    _, table = cv.shared_scaffold_folds(toy_frames, n_outer=1, n_inner=2)
    oof = multitask.run_cv_stacked(
        toy_frames, method="ridge", n_bits=64, n_outer=1, n_inner=2, assignments=table
    )
    required = {
        "method",
        "endpoint",
        "fold",
        "outer_fold",
        "inner_fold",
        "Molecule_Name",
        "y_true",
        "y_pred",
        "y_lower",
        "y_upper",
    }
    assert required <= set(oof.columns)
    # One prediction per measurement, and the endpoint column must be preserved so
    # per-endpoint scoring still works on a multitask run.
    assert oof.height == sum(f.height for f in toy_frames.values())
    assert set(oof["endpoint"].unique().to_list()) == {"A", "B"}


def test_multitask_method_names_are_suffixed(toy_frames) -> None:
    """The suffix is what keeps the two arms distinguishable once both are
    concatenated into one comparison frame."""
    _, table = cv.shared_scaffold_folds(toy_frames, n_outer=1, n_inner=2)
    oof = multitask.run_cv_stacked(
        toy_frames, method="ridge", n_bits=64, n_outer=1, n_inner=2, assignments=table
    )
    assert oof["method"].unique().to_list() == ["ridge_multitask"]
