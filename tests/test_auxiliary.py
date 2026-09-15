"""Tests for the auxiliary-data derivations and their CV harnesses.

The properties that matter here are about *what a model is allowed to see*, not
about accuracy. Three distinct ways to get this wrong are checked:

1. A weak label invented by `auxiliary.censored_labels` reaching a reported metric.
   That would reward a model for reproducing this repo's own guess, and because
   those rows carry a wide credible interval, ST-RAE would score most of them as
   zero error -- so the arm would look good precisely because its labels are made up.
2. An auxiliary target being scored as if it were an endpoint. The screen columns
   train the encoder and must never appear in an OOF frame.
3. A public compound that is structurally identical to a challenge compound staying
   in the pretraining corpus, which would let a fold's test label reach the encoder
   through a different lab's measurement of the same molecule.

The slow parts (anything that fits a Chemprop model) are not exercised here; they
are covered by running the notebook. What is tested is the data plumbing that
decides whether those fits are measuring what they claim to.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cyp import aux_training, auxiliary
from cyp import constants as C


def _wide(names: list[str], **columns: list[float | None]) -> pl.DataFrame:
    return pl.DataFrame({"Molecule_Name": names, **columns})


# ---------------------------------------------------------------------------
# Screen derivations
# ---------------------------------------------------------------------------


def test_screen_targets_are_dense_and_distinctly_named():
    """The screen has no missing cells, and its columns cannot be mistaken for endpoints.

    Density is what makes the auxiliary-head arm worth running -- a sparse auxiliary
    target would add little coverage over the labels it is meant to supplement. The
    naming check matters because a column ending in `_pIC50_direct_inhibition` would
    be picked up by the submission validator as a scored endpoint.
    """
    targets = auxiliary.screen_targets()

    aux_columns = [c for c in targets.columns if c not in ("Molecule_Name", "SMILES")]
    assert len(aux_columns) == len(C.ISOFORMS)
    for column in aux_columns:
        assert targets[column].is_null().sum() == 0, f"{column} has nulls"
        assert column not in C.REGRESSION_ENDPOINTS
        assert column.endswith("_screen_log2fc")

    assert targets["Molecule_Name"].n_unique() == targets.height


def test_censored_labels_only_cover_unlabelled_compounds():
    """A weak label must never be attached to a compound that has a real measurement.

    If it were, the real and invented labels for one compound would both enter
    training and one of them would be wrong.
    """
    from cyp import data

    for endpoint in C.REGRESSION_ENDPOINTS:
        weak = auxiliary.censored_labels(endpoint)
        real = data.training_frame(endpoint)
        overlap = set(weak["Molecule_Name"]) & set(real["Molecule_Name"])
        assert not overlap, f"{endpoint}: {len(overlap)} compounds labelled both ways"


def test_censored_labels_sit_below_the_screening_concentration():
    """The invented label is a ceiling, and the interval runs downward from it.

    A compound with no effect at the screening concentration cannot be more potent
    than that concentration, so `y_upper` is the censoring point and `y_true` may not
    exceed it. An interval running the other way would claim the compound is *at
    least* that potent, which is the opposite of what the measurement says.
    """
    for endpoint in C.REGRESSION_ENDPOINTS:
        weak = auxiliary.censored_labels(endpoint)
        if not weak.height:
            continue
        assert (weak["y_true"] <= auxiliary.SCREEN_CENSOR_PIC50 + 1e-9).all()
        assert (weak["y_upper"] <= auxiliary.SCREEN_CENSOR_PIC50 + 1e-9).all()
        assert (weak["y_lower"] < weak["y_upper"]).all()


def test_augmented_frame_preserves_the_real_rows_unchanged():
    """Augmentation adds rows; it must not alter or reorder the genuine ones.

    The real rows lead the frame so a downstream `head` or positional slice behaves
    the same whether or not augmentation was applied.
    """
    from cyp import data

    endpoint = C.REGRESSION_ENDPOINTS[0]
    real = data.training_frame(endpoint)
    augmented = auxiliary.augmented_training_frame(endpoint)

    assert augmented.height >= real.height
    assert augmented.columns == real.columns
    assert augmented.head(real.height).equals(real)


def test_triage_frame_covers_every_screened_compound():
    """Selection is modelled over the whole screen, not just the labelled part.

    The unlabelled compounds are the negative class. Restricting the frame to
    labelled compounds would leave a single-class target and nothing to learn.
    """
    for endpoint in C.REGRESSION_ENDPOINTS:
        triage = auxiliary.triage_frame(endpoint)
        assert triage["y_true"].dtype == pl.Boolean
        assert 0 < int(triage["y_true"].sum()) < triage.height


# ---------------------------------------------------------------------------
# Emax and the TDI shift
# ---------------------------------------------------------------------------


def test_emax_is_measured_rather_than_assumed_useless():
    """`emax_summary` reports spread and correlation for every isoform and condition.

    The verdict on Emax is snapshot-dependent, so the check is that the numbers are
    produced -- not what they currently say.
    """
    summary = auxiliary.emax_summary()
    assert summary.height == len(C.ISOFORMS) * 2
    assert set(summary["condition"]) == {"direct", "tdi"}
    assert summary["std"].is_not_null().all()


def test_tdi_shift_is_positive_for_time_dependent_inhibitors():
    """A TDI compound becomes more potent after preincubation, so its shift is positive.

    This is the mechanistic sanity check on the F6 side section: if the sign came out
    the other way, the two pIC50 columns would have been subtracted backwards.
    """
    summary = auxiliary.tdi_shift_summary()
    for row in summary.iter_rows(named=True):
        assert row["mean_shift_positive"] > row["mean_shift_negative"], row["isoform"]
        assert row["auc_shift_alone"] > 0.5, row["isoform"]


# ---------------------------------------------------------------------------
# Harness plumbing
# ---------------------------------------------------------------------------


def test_prediction_lookup_ignores_auxiliary_columns():
    """Auxiliary heads occupy trailing prediction columns and are never scored.

    This is the mechanism that lets the screen train the encoder without appearing in
    a metric, so it is checked directly rather than through a model fit.
    """
    test_wide = _wide(["a", "b"])
    test_long = pl.DataFrame(
        {
            "Molecule_Name": ["a", "b", "a"],
            "endpoint": ["e1", "e1", "e2"],
        }
    )
    # Two scored endpoints, one auxiliary column carrying deliberately absurd values.
    predictions = np.array([[1.0, 2.0, 999.0], [3.0, 4.0, 999.0]])

    result = aux_training._predictions_for(
        test_long, test_wide, predictions, ["e1", "e2", "aux_column"]
    )

    assert result.tolist() == [1.0, 3.0, 2.0]
    assert 999.0 not in result.tolist()


def test_inverse_propensity_weights_are_bounded_and_mean_normalised():
    """A vanishing propensity must not produce an unbounded weight.

    Without the clip, one compound the propensity model happens to score near zero
    would carry a weight of several hundred and dominate the loss -- adding far more
    variance than the bias it removes.
    """
    train_wide = _wide(["a", "b", "c"], e1=[1.0, 2.0, 3.0])
    propensity = {"e1": {"a": 1e-9, "b": 0.5, "c": 0.9}}

    weights = aux_training._inverse_propensity_weights(train_wide, ["e1"], propensity)

    assert weights.shape == (3,)
    assert (weights <= aux_training.WEIGHT_CLIP + 1e-9).all()
    assert (weights >= 1.0 / aux_training.WEIGHT_CLIP - 1e-9).all()
    # The rarely-selected compound still gets the largest weight -- clipping bounds
    # the correction without inverting its direction.
    assert weights[0] == weights.max()


def test_unmeasured_compounds_fall_back_to_unit_weight():
    """A compound with no propensity estimate keeps the unweighted default.

    Dropping it instead would silently shrink the training set, and the arm would
    then be measuring the smaller sample rather than the reweighting.
    """
    train_wide = _wide(["a", "b"], e1=[1.0, None])
    propensity = {"e1": {"a": 0.5}}

    weights = aux_training._inverse_propensity_weights(train_wide, ["e1"], propensity)

    assert weights.shape == (2,)
    assert np.isfinite(weights).all()


def test_weak_labels_are_endpoint_specific_not_global():
    """A screen negative for one isoform can be a real measurement for another.

    This is the subtlety that makes the "no weak label is ever scored" check a
    per-endpoint one. A compound can be inactive against CYP2D6 in the screen while
    carrying a genuine CYP3A4 dose-response curve, so its name legitimately appears
    in the OOF frame -- under CYP3A4 only. Asserting globally that no augmented
    compound appears anywhere in the results would flag that correct behaviour as a
    leak.
    """
    from cyp import data

    weak_names = {
        endpoint: set(auxiliary.censored_labels(endpoint)["Molecule_Name"])
        for endpoint in C.REGRESSION_ENDPOINTS
    }
    real_names = {
        endpoint: set(data.training_frame(endpoint)["Molecule_Name"])
        for endpoint in C.REGRESSION_ENDPOINTS
    }

    # Within an endpoint the two sets are disjoint, which is what scoring relies on.
    for endpoint in C.REGRESSION_ENDPOINTS:
        assert not (weak_names[endpoint] & real_names[endpoint])

    # Across endpoints they overlap, which is why the check must be per-endpoint.
    cross = any(
        weak_names[a] & real_names[b]
        for a in C.REGRESSION_ENDPOINTS
        for b in C.REGRESSION_ENDPOINTS
        if a != b
    )
    assert cross, "expected some compound to be weak for one endpoint and real for another"


def test_augmented_runner_rejects_mismatched_endpoint_sets():
    """Training and scoring frames must describe the same endpoints.

    A silent mismatch would train on one set of targets and score another, which
    fails deep inside chemprop with an unhelpful message rather than here.
    """
    frame = pl.DataFrame(
        {
            "Molecule_Name": ["a"],
            "SMILES": ["c1ccccc1"],
            "y_true": [5.0],
            "y_lower": [4.8],
            "y_upper": [5.2],
        }
    )
    with pytest.raises(ValueError, match="same endpoints"):
        aux_training.run_cv_augmented({"e1": frame}, {"e2": frame})


# ---------------------------------------------------------------------------
# Public data
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not C.available_external_snapshots(),
    reason="no external snapshot; run `make external-data`",
)
def test_public_pretraining_matrix_excludes_challenge_structures():
    """A challenge compound must not survive into the pretraining corpus.

    Its public measurement is a different lab's read on the same molecule, so leaving
    it in would let a fold's test compound inform the encoder before fine-tuning ever
    starts. Matching is on standardised structure, not on the raw SMILES string,
    because the two sources salt and write their molecules differently.
    """
    from cyp import data, external

    challenge = data.load_train_inhibition()["SMILES"].to_list()
    endpoints = list(C.REGRESSION_ENDPOINTS)

    kept = aux_training.public_pretraining_matrix(endpoints, exclude_smiles=challenge)

    excluded = {external.standardise_smiles(s) for s in challenge}
    excluded.discard(None)
    assert not (set(kept["smiles_std"]) & excluded)

    # The exclusion must remove the overlap, not the corpus.
    unfiltered = aux_training.public_pretraining_matrix(endpoints)
    assert kept.height > 0.9 * unfiltered.height


@pytest.mark.skipif(
    not C.available_external_snapshots(),
    reason="no external snapshot; run `make external-data`",
)
def test_public_pretraining_columns_match_the_finetuning_head():
    """Pretraining column order must equal the fine-tuned model's target order.

    Chemprop warm-starts the head by position, so a permuted column order would load
    CYP3A4's pretrained weights into CYP1A2's head -- a silent failure that trains
    without error and degrades every endpoint.
    """
    endpoints = list(C.REGRESSION_ENDPOINTS)
    matrix = aux_training.public_pretraining_matrix(endpoints)

    assert matrix.columns[0] == "smiles_std"
    assert matrix.columns[1:] == endpoints


@pytest.mark.skipif(
    not C.available_external_snapshots(),
    reason="no external snapshot; run `make external-data`",
)
def test_public_potencies_are_inhibition_only():
    """Activators must not contribute a pIC50.

    Their AC50 is on the same scale as an inhibitor's but means the opposite thing,
    so folding them in would teach the model to predict potent inhibition from
    compounds that do the reverse.
    """
    from cyp import external

    for isoform in C.PUBCHEM_AIDS:
        frame = external.load_pubchem(isoform)
        with_potency = frame.filter(pl.col("pIC50").is_not_null())
        assert with_potency["is_inhibitor"].all(), isoform
        # Inactives are kept as rows but never carry a fitted potency.
        assert frame.filter(pl.col("is_inactive"))["pIC50"].is_null().all(), isoform
