"""Tests for the CYP2D6 pharmacophore descriptors (notebook 09)."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from cyp import pharmacophore as ph

# Quinidine: a textbook CYP2D6 substrate/inhibitor. One basic tertiary amine, two
# aromatic rings (quinoline), and the amine several bonds from the ring system.
QUINIDINE = "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12"
# Nicotine: basic pyrrolidine N directly adjacent to pyridine -- the short-distance case.
NICOTINE = "CN1CCC[C@H]1c1cccnc1"
# Paracetamol: the amide-nitrogen case. Must NOT count as basic.
PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
BENZENE = "c1ccccc1"


def test_amide_nitrogen_is_not_basic():
    """An amide N is planar and non-basic at physiological pH.

    This is the exclusion that separates the CYP2D6 pharmacophore from ordinary
    carboxamides; without it, a large fraction of the library would be miscounted.
    """
    assert ph.basic_nitrogen_indices(Chem.MolFromSmiles(PARACETAMOL)) == []


def test_aliphatic_amines_are_basic():
    assert len(ph.basic_nitrogen_indices(Chem.MolFromSmiles(NICOTINE))) == 1
    assert len(ph.basic_nitrogen_indices(Chem.MolFromSmiles(QUINIDINE))) == 1


def test_aromatic_nitrogen_is_not_basic():
    """Pyridine's N is aromatic, not an sp3 amine -- nicotine has exactly one basic N."""
    mol = Chem.MolFromSmiles(NICOTINE)
    basic = ph.basic_nitrogen_indices(mol)
    assert all(not mol.GetAtomWithIdx(i).GetIsAromatic() for i in basic)


def test_absent_pharmacophore_gives_null_not_zero():
    """Zero is a real distance; a molecule with no basic N must not report it.

    The distinction matters because 59% of CYP2D6's compounds have no basic
    nitrogen, so a zero-fill would move the majority of the set to an extreme of the
    descriptor rather than marking them as unmeasured.
    """
    frame = ph.descriptors([BENZENE], include_3d=False)
    assert frame["topological_n_to_aromatic"].is_nan().all()
    assert frame["n_basic_nitrogen"][0] == 0


def test_spatial_distance_is_shorter_when_directly_attached():
    """Nicotine's N sits next to its ring; quinidine's is a chain away.

    A 3D descriptor that does not reproduce this ordering is not measuring geometry.
    """
    frame = ph.descriptors([NICOTINE, QUINIDINE], n_conformers=5)
    near, far = frame["spatial_n_to_aromatic_min"].to_list()
    assert near < far


def test_spatial_distance_is_physically_plausible():
    """Bonded heavy atoms sit ~1.4-1.6 A apart; nothing should be below that."""
    frame = ph.descriptors([NICOTINE, QUINIDINE], n_conformers=5)
    distances = [d for d in frame["spatial_n_to_aromatic_min"].to_list() if d == d]
    assert distances
    assert all(1.0 < d < 25.0 for d in distances)


def test_min_never_exceeds_mean_across_conformers():
    frame = ph.descriptors([QUINIDINE, NICOTINE], n_conformers=5)
    for lo, hi in zip(
        frame["spatial_n_to_aromatic_min"].to_list(),
        frame["spatial_n_to_aromatic_mean"].to_list(),
        strict=True,
    ):
        if lo == lo and hi == hi:
            assert lo <= hi + 1e-9


def test_unparseable_smiles_yields_nulls_not_an_exception():
    frame = ph.descriptors(["not_a_molecule"], include_3d=False)
    assert frame.height == 1
    assert frame["n_basic_nitrogen"][0] is None


def test_row_order_matches_input_order():
    """Every downstream join is positional, so a reordering would silently mislabel."""
    smiles = [QUINIDINE, BENZENE, NICOTINE, PARACETAMOL]
    frame = ph.descriptors(smiles, include_3d=False)
    assert frame["smiles"].to_list() == smiles


def test_spatial_distance_handles_total_embedding_failure(monkeypatch):
    """`AllChem.EmbedMultipleConfs` returns an RDKit vector of conformer ids, not a
    plain int -- `result == 0` compares the vector object to an int and is always
    False, even when embedding produced zero conformers. That let a total failure
    slip past the emptiness guard and crash deep in `np.min` on an empty sequence,
    on a real PubChem compound during a live run. Pinned by mocking the RDKit call
    directly, since a molecule that reliably fails ETKDG embedding is not a stable
    thing to depend on across RDKit versions.
    """
    from rdkit.Chem import AllChem

    original_embed = AllChem.EmbedMultipleConfs
    monkeypatch.setattr(AllChem, "EmbedMultipleConfs", lambda *a, **k: [])
    try:
        frame = ph.descriptors([NICOTINE], include_3d=True, n_conformers=5)
    finally:
        monkeypatch.setattr(AllChem, "EmbedMultipleConfs", original_embed)

    assert frame["spatial_n_to_aromatic_min"].is_nan().all()
    assert frame["spatial_n_to_aromatic_mean"].is_nan().all()


def test_matrix_has_no_nan_after_fill():
    """RDKit's sentinel is a float NaN, which `fill_null` alone does not catch.

    Pinned because the NaN reached the model matrix in the first implementation:
    Polars treats NaN and null as distinct, so the fill silently did nothing.
    """
    frame = ph.descriptors([QUINIDINE, BENZENE, PARACETAMOL, NICOTINE], n_conformers=3)
    matrix = ph.matrix(frame)
    assert not np.isnan(matrix).any()
    assert matrix.shape == (4, len(ph.DESCRIPTOR_COLUMNS))


@pytest.mark.parametrize("include_3d", [True, False])
def test_column_set_follows_include_3d(include_3d: bool):
    frame = ph.descriptors([NICOTINE], include_3d=include_3d, n_conformers=2)
    for column in ph.DESCRIPTOR_2D:
        assert column in frame.columns
    for column in ph.DESCRIPTOR_3D:
        assert (column in frame.columns) is include_3d


def test_parallel_matches_sequential():
    """n_jobs=-1 must produce identical results to n_jobs=1 -- parallelising a pure
    function over independent molecules should never change the answer, only the
    wall-clock time. Small n so the test stays fast; the actual speed benefit only
    shows up at hundreds-to-thousands of compounds (measured separately: 26.3s ->
    8.3s on 1,000 compounds after fixing a worker-pool-reuse bug in the first
    implementation)."""
    smiles = [QUINIDINE, BENZENE, PARACETAMOL, NICOTINE] * 5
    sequential = ph.matrix(ph.descriptors(smiles, n_conformers=3, n_jobs=1))
    parallel = ph.matrix(ph.descriptors(smiles, n_conformers=3, n_jobs=2))
    assert np.allclose(sequential, parallel, equal_nan=True)


def test_on_progress_reaches_total_for_sequential_and_parallel():
    """The progress callback exists because a multi-minute call with no signal is
    indistinguishable from a hang -- pinning that it actually fires and reaches
    the true total in both code paths, not just the sequential one."""
    smiles = [QUINIDINE, BENZENE, PARACETAMOL, NICOTINE] * 5

    seq_calls = []
    ph.descriptors(
        smiles, n_conformers=2, n_jobs=1, on_progress=lambda d, t: seq_calls.append((d, t))
    )
    assert seq_calls[-1] == (len(smiles), len(smiles))
    assert seq_calls[0] == (1, len(smiles))

    par_calls = []
    ph.descriptors(
        smiles, n_conformers=2, n_jobs=2, on_progress=lambda d, t: par_calls.append((d, t))
    )
    assert par_calls[-1] == (len(smiles), len(smiles))


def test_include_3d_false_ignores_n_jobs():
    """The 2D-only path is already sub-second, so n_jobs is documented as ignored
    there -- confirm it does not error or change behaviour when passed anyway,
    since a caller sweeping n_jobs across both 2D and 3D calls should not need a
    special case."""
    smiles = [QUINIDINE, BENZENE, NICOTINE]
    default = ph.descriptors(smiles, include_3d=False)
    with_jobs = ph.descriptors(smiles, include_3d=False, n_jobs=4)
    assert default.equals(with_jobs)
