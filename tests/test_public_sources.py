"""Tests for the ChEMBL and Tox21 loaders and the multi-source union matrix.

These run against the committed snapshot rather than the network, so they pin the
parsing and the assembly rather than the download. The download itself is exercised
by running the modules; what breaks silently is a schema change downstream of it,
which is what these cover.
"""

from __future__ import annotations

import polars as pl
import pytest

from cyp import aux_training, chembl, data, external, tox21
from cyp import constants as C

pytestmark = pytest.mark.skipif(
    not C.available_external_snapshots(),
    reason="no external snapshot downloaded",
)


def _has(template: str, keys) -> bool:
    directory = C.external_snapshot_dir()
    return all((directory / template.format(isoform=k)).exists() for k in keys)


chembl_only = pytest.mark.skipif(
    not _has(C.CHEMBL_FILE_TEMPLATE, C.CHEMBL_TARGETS), reason="no ChEMBL extract"
)
tox21_only = pytest.mark.skipif(
    not _has(C.TOX21_FILE_TEMPLATE, list(C.TOX21_AIDS) + ["luciferase_counterscreen"]),
    reason="no Tox21 extract",
)


@chembl_only
def test_chembl_loads_with_the_documented_schema():
    frame = chembl.load_chembl("CYP2D6")

    assert {"SMILES", "pIC50", "n_records", "spread"} <= set(frame.columns)
    assert frame.height > 1000
    assert frame["SMILES"].n_unique() == frame.height, "one row per structure"
    assert frame["pIC50"].is_null().sum() == 0


@chembl_only
def test_chembl_carries_no_low_end_range():
    """The documented limitation, pinned so it is not quietly assumed away.

    ChEMBL assigns a pchembl only where a curve fitted, so it supplies new chemistry
    and not the weak-inhibitor range. A future change that appears to add low-end
    rows here is far more likely to be a censoring bug than a real improvement.
    """
    frame = chembl.load_chembl("CYP3A4")
    below = int((frame["pIC50"] < C.CHEMBL_PCHEMBL_FLOOR).sum())

    assert below / frame.height < 0.01


@tox21_only
def test_tox21_detects_inactives():
    """Regression test for the replicate-null bug.

    Only replicate 1 is populated for ~96% of compounds. Treating a null replicate as
    "not inactive" silently lost the inactive call for every singly-run compound,
    which is the entire reason this source is here.
    """
    frame = tox21.load_tox21("CYP2D6")

    assert int(frame["is_inactive"].sum()) > 1000
    # An inactive compound has no fitted inhibition curve, by construction.
    inactive = frame.filter(pl.col("is_inactive"))
    assert inactive["pIC50"].is_null().all()


@tox21_only
def test_tox21_removes_luciferase_artifacts():
    """The counter-screen call is honoured, not merely downloaded."""
    artifacts = tox21.luciferase_inhibitors()
    assert len(artifacts) > 100

    frame = tox21.load_tox21("CYP3A4")
    assert not set(frame["SMILES"].to_list()) & artifacts


@tox21_only
def test_tox21_has_no_cyp1a2():
    """The absence is carried rather than filled from another protocol."""
    assert "CYP1A2" not in C.TOX21_AIDS
    with pytest.raises(ValueError, match="no CYP1A2"):
        tox21.load_tox21("CYP1A2")


@chembl_only
@tox21_only
def test_full_union_excludes_blind_structures():
    """The leakage check that matters, on the real blind set.

    A public compound that is the same structure as a blind-set compound would put a
    test molecule's chemistry into the pretraining corpus. The standardisation is
    what makes this findable at all -- raw SMILES matching finds almost nothing.
    """
    endpoints = C.REGRESSION_ENDPOINTS
    test_smiles = data.load_test()["SMILES"].to_list()
    train_smiles = data.load_train_inhibition()["SMILES"].to_list()

    wide, columns = aux_training.full_union_matrix(
        endpoints, exclude_smiles=train_smiles + test_smiles
    )

    standardised = {external.standardise_smiles(s) for s in test_smiles}
    standardised.discard(None)
    assert not set(wide["smiles_std"].to_list()) & standardised


@chembl_only
@tox21_only
def test_full_union_is_a_superset_of_the_narrower_matrices():
    """Each arm adds heads to the one below it, so a comparison stays nested."""
    endpoints = C.REGRESSION_ENDPOINTS
    base = aux_training.public_pretraining_matrix(endpoints)
    _, union_columns = aux_training.public_union_matrix(endpoints)
    _, full_columns = aux_training.full_union_matrix(endpoints)

    assert set(base.columns) - {"smiles_std"} <= set(union_columns)
    assert set(union_columns) <= set(full_columns)
    assert len(full_columns) > len(union_columns) > len(endpoints)


def test_challenge_auxiliary_matrix_covers_the_unused_arms():
    """TDI-condition and Emax heads, with the CYP3A4 coverage gain that motivates them."""
    endpoints = C.REGRESSION_ENDPOINTS
    frame, columns = aux_training.challenge_auxiliary_matrix(endpoints)

    assert "CYP3A4_pIC50_TDI_condition" in columns
    assert "CYP2D6_EmaxVsPosCtrl_direct_inhibition" in columns

    # The reason the TDI arm is worth reading: it carries compounds the direct
    # inhibition table does not, and far more CYP3A4 rows.
    direct = int(data.load_train_inhibition()["CYP3A4_pIC50_direct_inhibition"].is_not_null().sum())
    tdi_condition = int(frame["CYP3A4_pIC50_TDI_condition"].is_not_null().sum())
    assert tdi_condition > direct


def test_emax_tracks_cyp2d6_specifically():
    """Emax is the one auxiliary readout with CYP2D6-specific signal.

    Measured at +0.77 Spearman against CYP2D6 potency and negative on every other
    isoform. CYP2D6's weak ordering is what caps the macro, so a regression here
    removes the strongest lever the challenge release offers for it.
    """
    inhibition = data.load_train_inhibition()
    emax = data.load_train_emax()
    joined = inhibition.join(emax.drop("SMILES"), on="Molecule_Name", how="inner")

    def rho(isoform: str) -> float:
        pair = joined.select(
            f"{isoform}_pIC50_direct_inhibition",
            f"{isoform}_EmaxVsPosCtrl_direct_inhibition",
        ).drop_nulls()
        return float(pair.select(pl.corr(*pair.columns, method="spearman")).item())

    assert rho("CYP2D6") > 0.6
    for isoform in ("CYP1A2", "CYP2C9", "CYP3A4"):
        assert rho(isoform) < 0


def test_inchikey_skeleton_ignores_stereochemistry_and_salts():
    """The property that makes it the right exclusion key.

    Canonical SMILES separates these; the connectivity block does not. That
    difference is the whole reason the exclusion was moved onto this key.
    """
    enantiomers = ("C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O")
    a, b = (external.inchikey_skeleton(s) for s in enantiomers)
    assert a is not None and a == b
    assert external.standardise_smiles(enantiomers[0]) != external.standardise_smiles(
        enantiomers[1]
    )

    # A hydrochloride reduces to its parent, because the largest fragment wins.
    assert external.inchikey_skeleton("CCN.Cl") == external.inchikey_skeleton("CCN")


def test_inchikey_skeleton_returns_none_rather_than_raising():
    assert external.inchikey_skeleton("not a molecule") is None
    assert external.inchikey_skeleton(None) is None
    assert external.skeleton_keys(["CCO", "garbage"]) == {external.inchikey_skeleton("CCO")}


@chembl_only
@tox21_only
def test_skeleton_exclusion_removes_more_than_smiles_exclusion():
    """Pins the measured gap between the two keys.

    On the current snapshots the skeleton key removes 184 compounds that canonical
    SMILES leaves in -- stereoisomers, tautomers and salt forms of challenge
    compounds. None were blind-set structures here, but 960 source records matched a
    *training* compound only under this key, so the corpus was pretraining on
    molecules it would later fine-tune on.
    """
    endpoints = C.REGRESSION_ENDPOINTS
    train_smiles = data.load_train_inhibition()["SMILES"].to_list()
    test_smiles = data.load_test()["SMILES"].to_list()

    wide, _ = aux_training.full_union_matrix(endpoints, exclude_smiles=train_smiles + test_smiles)
    corpus = external.skeleton_keys(wide["smiles_std"].to_list())

    # The exclusion holds under the stricter key, which is the point.
    assert not corpus & external.skeleton_keys(test_smiles)
    assert not corpus & external.skeleton_keys(train_smiles)


def test_pretrain_targets_must_lead_with_the_scored_endpoints():
    """Chemprop restores output heads positionally, so the order is load-bearing."""
    endpoints = C.REGRESSION_ENDPOINTS
    frames = {e: data.training_frame(e) for e in endpoints}
    pretraining = aux_training.public_pretraining_matrix(endpoints)

    with pytest.raises(ValueError, match="must begin with the scored endpoints"):
        aux_training.run_cv_pretrained(
            frames,
            pretraining=pretraining,
            pretrain_targets=list(reversed(endpoints)),
            folds=[],
        )


def test_pretrain_targets_rejects_columns_the_frame_lacks():
    endpoints = C.REGRESSION_ENDPOINTS
    frames = {e: data.training_frame(e) for e in endpoints}
    pretraining = aux_training.public_pretraining_matrix(endpoints)

    with pytest.raises(ValueError, match="missing target columns"):
        aux_training.run_cv_pretrained(
            frames,
            pretraining=pretraining,
            pretrain_targets=[*endpoints, "CYP3A4_max_response"],
            folds=[],
        )
