"""Tests for the docking / PLIF / Uni-Mol stack (notebook 11).

Deliberately split by cost. The pure-logic tests -- PDB record parsing, receptor
metadata, frame alignment, pruning -- run everywhere and are what CI would keep.
Anything needing the smina binary, a network download or Uni-Mol's weights is
marked and skipped when unavailable, because those are environment facts rather
than code defects and a red suite for a missing brew formula teaches nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from cyp import docking, plif, unimol

# A PDB fragment with the traps that matter: a two-chain entry, a water, the heme,
# and an atom name that runs into the residue column under whitespace splitting.
# Columns are what PDB defines, so these strings are position-sensitive.
SAMPLE_PDB = """\
ATOM      1  N   MET A   1      10.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA  MET A   1      11.000  10.000  10.000  1.00 20.00           C
ATOM      3  N   LEU B   1      50.000  50.000  50.000  1.00 20.00           N
HETATM 4000 FE   HEM A 500      12.000  12.000  12.000  1.00 25.00          FE
HETATM 4100  O   HOH A 600      30.000  30.000  30.000  1.00 30.00           O
HETATM 4200  C1  P6U A 700      13.000  13.000  13.000  1.00 22.00           C
"""


def test_chain_filter_keeps_only_the_requested_chain():
    """Multi-chain entries must be split before docking.

    5TFT (the CYP2D6 receptor) has four copies. Docking into a merged receptor puts
    a second copy's atoms inside the first copy's search box, which produces poses
    rather than an error -- so this has to be caught by construction.
    """
    lines = SAMPLE_PDB.splitlines(keepends=True)
    assert len(docking._atom_records(lines, "A")) == 5
    assert len(docking._atom_records(lines, "B")) == 1


def test_record_parsing_is_column_based_not_whitespace():
    """Residue and chain come from fixed columns, not from `.split()`.

    PDB is a fixed-column format, and the fields run together on crowded records:
    a five-digit serial abuts the atom name, and a four-character atom name abuts
    the residue. Whitespace splitting then shifts every later field by one, so the
    chain filter reads a coordinate and silently keeps nothing.
    """
    crowded = "HETATM99999 FE   HEM A 500      12.000  12.000  12.000  1.00 25.00          FE\n"
    assert crowded[17:20].strip() == "HEM"
    assert crowded[21:22] == "A"
    # The naive parse merges the record type and serial, shifting every later
    # field left by one -- so the chain filter reads the residue number instead
    # of the chain and keeps nothing.
    normal = SAMPLE_PDB.splitlines(keepends=True)[0]
    assert normal.split()[4] == "A"  # chain, on a well-spaced record
    assert crowded.split()[4] == "500"  # same index, now the residue number

    # And the filter itself survives the crowded record.
    assert docking._atom_records([crowded], "A") == [crowded]
    assert docking._atom_records([crowded], "B") == []


def test_solvent_list_excludes_the_heme():
    """Waters go, the heme stays.

    Stripping all HETATM records is the reflex when cleaning a PDB and would remove
    the heme, leaving a cavity whose floor is missing so ligands dock too deep --
    a silent geometry error, not a crash.
    """
    assert "HOH" in docking._SOLVENT
    assert "HEM" not in docking._SOLVENT


def test_every_isoform_has_a_receptor_with_a_box_ligand():
    """Each scored isoform needs a holo structure; an apo one has no box."""
    from cyp import constants

    for isoform in constants.ISOFORMS:
        receptor = docking.RECEPTORS[isoform]
        assert receptor.isoform == isoform
        assert receptor.ligand_code, f"{isoform} has no box-defining ligand"
        assert receptor.chain
        assert receptor.resolution > 0


def test_score_frame_schema_on_empty_input(tmp_path):
    """An empty pose file yields an empty frame with the right columns.

    Docking can legitimately produce nothing (every ligand failed to embed), and the
    caller joins on this frame -- a bare empty DataFrame would lose the schema and
    break the join rather than contributing zero rows.
    """
    empty = tmp_path / "none.sdf"
    empty.write_text("")
    frame = docking.score_frame(empty)
    assert frame.columns == ["Molecule_Name", "docking_affinity"]


def test_plif_prune_drops_constant_and_singleton_bits():
    """Both tails are uninformative, for different reasons.

    An always-on bit carries no signal. A bit set in one compound out of many is
    closer to an identifier, and under scaffold CV those rare bits concentrate in
    single folds where a tree can memorise them.
    """
    n = 500
    frame = pl.DataFrame(
        {
            "Molecule_Name": [f"c{i}" for i in range(n)],
            "always": np.ones(n, dtype=np.uint8),
            "never": np.zeros(n, dtype=np.uint8),
            # 1/500 = 0.002, below the 0.01 floor.
            "singleton": np.array([1] + [0] * (n - 1), dtype=np.uint8),
            "informative": np.array([1] * 150 + [0] * (n - 150), dtype=np.uint8),
        }
    )
    kept = plif.prune(frame, min_frequency=0.01)
    assert kept.columns == ["Molecule_Name", "informative"]


def test_plif_matrix_zero_fills_compounds_without_a_pose():
    """A compound that failed to dock must stay in the matrix as all-zero.

    Dropping it would leave each arm on a different compound set, which silently
    invalidates the paired comparison the whole notebook rests on. All-zero is also
    the honest encoding: no pose means no contacts.
    """
    frame = pl.DataFrame(
        {
            "Molecule_Name": ["a", "c"],
            "PHE120.A|Hydrophobic": np.array([1, 1], dtype=np.uint8),
            "ASP301.A|Cationic": np.array([1, 0], dtype=np.uint8),
        }
    )
    matrix = plif.matrix(frame, ["a", "b", "c"])
    assert matrix.shape == (3, 2)
    assert matrix[0].tolist() == [1.0, 1.0]
    assert matrix[1].tolist() == [0.0, 0.0]  # "b" never docked
    assert matrix[2].tolist() == [1.0, 0.0]


def test_plif_summarise_splits_bit_names_into_residue_and_interaction():
    """The interpretable half: a winning arm must say *which* contact did the work."""
    frame = pl.DataFrame(
        {
            "Molecule_Name": ["a", "b"],
            "ASP301.A|Cationic": np.array([1, 1], dtype=np.uint8),
            "PHE120.A|PiStacking": np.array([1, 0], dtype=np.uint8),
        }
    )
    summary = plif.summarise(frame)
    assert summary["bit"][0] == "ASP301.A|Cationic"
    assert summary["residue"][0] == "ASP301.A"
    assert summary["interaction"][0] == "Cationic"
    assert summary["hit_rate"][0] == 1.0


def test_vdw_contact_is_excluded_from_the_interaction_set():
    """VdWContact fires on nearly every nearby residue.

    Including it turns the fingerprint into a proximity map that cannot distinguish
    a hydrogen bond from an incidental contact.
    """
    assert "VdWContact" not in plif.INTERACTIONS
    assert "HBDonor" in plif.INTERACTIONS


def test_unimol_align_zero_fills_and_preserves_order():
    """Pose coverage is a subset of the SMILES arm's; alignment keeps arms paired."""
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    aligned = unimol.align(embeddings, ["a", "c"], ["a", "b", "c"])
    assert aligned.shape == (3, 2)
    assert aligned[0].tolist() == [1.0, 2.0]
    assert aligned[1].tolist() == [0.0, 0.0]
    assert aligned[2].tolist() == [3.0, 4.0]


def test_unimol_embed_poses_rejects_misaligned_inputs():
    """Atoms and coordinates disagreeing means a caller lost a molecule somewhere."""
    with pytest.raises(ValueError, match="must align"):
        unimol.embed_poses([["C"], ["N"]], [np.zeros((1, 3), dtype=np.float32)])


@pytest.mark.skipif(not docking.smina_available(), reason="smina not installed")
def test_smina_binary_runs():
    """Guards the environment assumption the whole notebook rests on."""
    assert docking.smina_available()


def test_split_sdf_splits_on_record_boundaries(tmp_path):
    """Each chunk is a valid multi-molecule SDF, not a byte-count slice.

    Splitting on `$$$$` rather than a fixed byte offset is what keeps a chunk from
    cutting a molecule record in half.
    """
    sdf = tmp_path / "ligs.sdf"
    sdf.write_text("MOL1\n...\n$$$$\nMOL2\n...\n$$$$\nMOL3\n...\n$$$$\n")
    chunks = docking.split_sdf(sdf, chunk_size=2, chunk_dir=tmp_path / "chunks")
    assert len(chunks) == 2
    assert chunks[0].read_text().count("$$$$") == 2
    assert chunks[1].read_text().count("$$$$") == 1


def test_split_sdf_drops_trailing_empty_record(tmp_path):
    """A file ending in `$$$$\\n` splits to a trailing empty string; it must not
    become a chunk with zero molecules in it."""
    sdf = tmp_path / "ligs.sdf"
    sdf.write_text("MOL1\n...\n$$$$\n")
    chunks = docking.split_sdf(sdf, chunk_size=100, chunk_dir=tmp_path / "chunks")
    assert len(chunks) == 1
    assert chunks[0].read_text().count("$$$$") == 1


def test_dock_in_chunks_skips_cached_chunks_and_reports_zero_seconds(tmp_path, monkeypatch):
    """A chunk whose pose file already exists must not be re-docked, and `on_chunk`
    must be told it was cached (seconds == 0.0) rather than timed as free work."""
    calls = []

    def fake_one_call(rec, box, ligs, out, **kwargs):
        calls.append(ligs)
        out.write_text("POSE\n$$$$\n")
        return out

    monkeypatch.setattr(docking, "_dock_one_call", fake_one_call)

    ligands = tmp_path / "ligs.sdf"
    ligands.write_text("A\n$$$$\nB\n$$$$\n")
    out = tmp_path / "poses.sdf"

    # Pre-seed the first chunk's pose file so it looks already-docked.
    pose_dir = tmp_path / "poses_chunks" / "poses"
    pose_dir.mkdir(parents=True)
    (pose_dir / "poses_00000.sdf").write_text("CACHED\n$$$$\n")

    events = []
    docking.dock_in_chunks(
        Path("rec.pdbqt"),
        Path("box.sdf"),
        ligands,
        out,
        chunk_size=1,
        on_chunk=lambda i, n, s, p: events.append((i, n, s)),
    )

    assert len(calls) == 1  # only the second (uncached) chunk was docked
    assert events[0][2] == 0.0  # first chunk reported as cached
    assert events[1][2] > 0.0  # second chunk reported real (mocked, but nonzero) time
    assert out.exists()
    assert out.read_text().count("$$$$") == 2


# ── atom-level PLIF (notebook 11's chemprop --atom-features-path arm) ──────────


def test_heavy_atom_map_preserves_heavy_indices_and_maps_hydrogens():
    """A heavy atom must map to itself; a hydrogen must map to its one heavy
    neighbour -- the property the whole atom-features array depends on, since
    `Chem.AddHs` appends hydrogens after the existing atoms without renumbering."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles("CCO")  # C0, C1, O2 as heavy atoms
    mol_h = Chem.AddHs(mol)
    mapping = plif._heavy_atom_map(mol_h)

    heavy_indices = {a.GetIdx() for a in mol_h.GetAtoms() if a.GetSymbol() != "H"}
    assert heavy_indices == {0, 1, 2}
    for idx in heavy_indices:
        assert mapping[idx] == idx

    for atom in mol_h.GetAtoms():
        if atom.GetSymbol() == "H":
            neighbour = atom.GetNeighbors()[0]
            assert mapping[atom.GetIdx()] == mapping[neighbour.GetIdx()]


def test_prune_atom_interactions_drops_columns_that_never_fire():
    """The atom-level equivalent of `prune`: a column constant at zero across every
    atom of every molecule carries nothing, and keeping it hands chemprop a dead
    input dimension. Measured on real CYP2D6 data, only 5 of 11 `INTERACTIONS`
    ever fire -- this is the mechanism that drops the other 6."""
    features = {
        "mol_a": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32),
        "mol_b": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
    }
    interactions = ("Hydrophobic", "NeverFires", "AlsoNeverFires")
    pruned, kept = plif.prune_atom_interactions(features, interactions)
    assert kept == ("Hydrophobic",)
    assert pruned["mol_a"].shape == (2, 1)
    assert pruned["mol_b"].shape == (1, 1)
    assert np.array_equal(pruned["mol_a"][:, 0], [1.0, 0.0])


def test_prune_atom_interactions_keeps_everything_when_all_columns_fire():
    features = {"mol_a": np.array([[1.0, 1.0]], dtype=np.float32)}
    interactions = ("A", "B")
    pruned, kept = plif.prune_atom_interactions(features, interactions)
    assert kept == interactions
    assert pruned is features  # no copy needed when nothing was dropped


def test_prune_atom_interactions_handles_empty_input():
    pruned, kept = plif.prune_atom_interactions({}, ("A", "B"))
    assert pruned == {}
    assert kept == ("A", "B")


def test_atom_features_npz_uses_real_arrays_when_present(tmp_path):
    """A compound with a docked pose gets its real per-atom array, in the position
    matching its place in `smiles_order`/`names_order` -- the order chemprop's own
    CSV rows will be in, not the order `features` happens to store them."""
    features = {"b": np.array([[1.0, 0.0]], dtype=np.float32)}
    out = tmp_path / "feats.npz"
    plif.atom_features_npz(
        features,
        smiles_order=["CCO", "CC"],
        names_order=["a", "b"],
        n_interactions=2,
        out_path=out,
    )
    with np.load(out) as archive:
        assert len(archive.files) == 2
        # "a" has no pose: zero-filled at CCO's real heavy-atom count (3).
        assert archive["arr_0"].shape == (3, 2)
        assert (archive["arr_0"] == 0).all()
        # "b" has a pose: its real (possibly different-shaped) array is used as-is.
        assert np.array_equal(archive["arr_1"], features["b"])


def test_atom_features_npz_zero_fills_missing_compounds_at_the_right_width(tmp_path):
    """A compound entirely absent from `features` (never docked) still needs a
    correctly-shaped array -- an empty or wrongly-shaped one would desynchronise
    every later row against chemprop's positional reading of the file."""
    out = tmp_path / "feats.npz"
    plif.atom_features_npz(
        {},
        smiles_order=["CCO"],
        names_order=["missing"],
        n_interactions=4,
        out_path=out,
    )
    with np.load(out) as archive:
        assert archive["arr_0"].shape == (3, 4)  # CCO has 3 heavy atoms
        assert (archive["arr_0"] == 0).all()
