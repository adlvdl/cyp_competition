"""Protein-ligand interaction fingerprints from docked poses.

The arm that encodes what the *protein* sees. Its counterpart, Uni-Mol on a docked
pose, encodes the ligand's geometry; measured on six known CYP ligands the Uni-Mol
vector from a docked pose has cosine 0.80-1.00 against the same molecule's free
ETKDG conformer (median between-molecule cosine 0.72), so Uni-Mol largely cannot see
the pose at all. A PLIF has no such problem by construction: every bit *is* a
protein contact, and a molecule with no protein has no fingerprint.

## Why ProLIF

Three implementations were considered. PLIP produces per-complex reports meant for
human reading, which would need parsing back into bits. ODDT has classical IFP/SPLIF
but was last released in 2021 and pulls in OpenBabel's Python bindings. ProLIF is
pure Python over RDKit and MDAnalysis -- the same dependency class as
`pharmacophore.py`, no new binary -- and returns a labelled matrix where each column
is a named ``residue|interaction`` pair. That labelling is the reason it is worth
preferring: an unnamed 3D fingerprint that wins tells you 3D helped, while this one
tells you *which contact*.

## What it recovered on CYP2D6, and why that matters here

Notebook 09 tested the hypothesis that CYP2D6 binding runs on a basic-nitrogen /
aromatic pharmacophore anchored by Glu216 and Asp301, and concluded the hypothesis
was not expressible through ligand geometry -- explicit 3D pharmacophore descriptors
made every endpoint *worse*. This arm tests the same chemistry from the other side.
On six known CYP ligands docked into 5TFT, ProLIF returned contacts with **both
Glu216 and Asp301**, alongside the Phe cluster (Phe120, Phe112, Phe483) that lines
the site. Quinidine -- the textbook 2D6 substrate -- docked strongest and sat closest
to Asp301.

So the question 09 left open is not "does the ligand have the right geometry" but
"does it make the contact", and only a PLIF can ask that.

## Cost

Negligible next to docking: 12 ms per ligand measured, so all 1,493 CYP2D6 compounds
take ~18 seconds against tens of minutes for the docking that produced the poses.
The expensive input is already paid for by the time this runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

#: Interaction types requested from ProLIF. This is the default set minus
#: `VdWContact`, which fires on almost every nearby residue and would swamp the
#: specific interactions with a proximity map -- it inflates the bit count without
#: distinguishing a hydrogen bond from an incidental contact. The remaining types
#: are the ones with a mechanistic reading in a CYP site.
INTERACTIONS = (
    "Hydrophobic",
    "HBAcceptor",
    "HBDonor",
    "PiStacking",
    "Anionic",
    "Cationic",
    "CationPi",
    "PiCation",
    "XBAcceptor",
    "XBDonor",
    "MetalAcceptor",
)

#: Minimum fraction of ligands a bit must be set in to survive `prune`. A column
#: set in one compound out of 1,500 is a per-compound identifier, not a feature,
#: and tree models will happily split on it.
MIN_FREQUENCY = 0.01


def receptor_molecule(protonated_pdb: Path):
    """Load a prepared receptor for ProLIF.

    Takes the **protonated** protein written by `docking.prepare_receptor`, not the
    raw crystal PDB. MDAnalysis infers bond orders when converting to RDKit, and on
    an unprotonated structure that inference segfaults the interpreter outright
    (observed as exit 138 with no traceback). The protonation step is a correctness
    requirement, not a refinement.
    """
    import MDAnalysis as mda
    import prolif

    universe = mda.Universe(str(protonated_pdb))
    return prolif.Molecule.from_mda(universe.select_atoms("protein"), force=True)


def _run_fingerprint(poses_sdf: Path, protonated_pdb: Path, interactions: tuple[str, ...]):
    """Shared ProLIF run behind both `compute` (molecule-level) and
    `atom_features` (atom-level) -- the fingerprint pass is the expensive-ish step
    (still seconds, but no reason to pay it twice), and the two views are just
    different reductions of the same `fp.ifp` contact data.

    Returns `(names, mols_h, fp)`: pose names in file order, the corresponding
    explicit-H RDKit molecules (needed to map atom-level contacts back to heavy
    atoms), and the fitted ProLIF `Fingerprint` whose `.ifp[frame_index]` holds
    the raw per-atom contact records.
    """
    import prolif
    from rdkit import Chem

    from .docking import _supplier

    protein = receptor_molecule(protonated_pdb)

    names: list[str] = []
    mols_h: list = []
    ligands = []
    for mol in _supplier(poses_sdf, remove_hs=False):
        if mol is None or mol.GetNumConformers() == 0:
            continue
        names.append(mol.GetProp("_Name") if mol.HasProp("_Name") else None)
        mol_h = Chem.AddHs(mol, addCoords=True)
        mols_h.append(mol_h)
        ligands.append(prolif.Molecule.from_rdkit(mol_h))

    if not ligands:
        return names, mols_h, None

    fp = prolif.Fingerprint(list(interactions), count=False)
    fp.run_from_iterable(ligands, protein, progress=False)
    return names, mols_h, fp


def compute(
    poses_sdf: Path,
    protonated_pdb: Path,
    interactions: tuple[str, ...] = INTERACTIONS,
) -> pl.DataFrame:
    """Interaction fingerprints for every pose in `poses_sdf`.

    Returns a frame with ``Molecule_Name`` plus one uint8 column per observed
    ``residue|interaction`` pair. Columns are whatever the poses actually made
    contact with, so two isoforms return different column sets -- they are different
    proteins, and a shared column space would be meaningless.

    Hydrogens are added to each ligand with `addCoords=True`: ProLIF needs explicit
    hydrogens to assign donors, and coordinates for them to be placed geometrically
    rather than at the origin.
    """
    names, _, fp = _run_fingerprint(poses_sdf, protonated_pdb, interactions)
    if fp is None:
        return pl.DataFrame(schema={"Molecule_Name": pl.Utf8})

    frame = fp.to_dataframe()

    # ProLIF's columns are a (ligand, residue, interaction) MultiIndex. The ligand
    # level is a constant here (every pose is "UNL1"), so it is dropped -- keeping it
    # would prefix every feature name with the same token.
    columns = [f"{residue}|{interaction}" for _, residue, interaction in frame.columns]
    matrix = frame.to_numpy().astype(np.uint8)
    return pl.DataFrame(
        {"Molecule_Name": names, **{c: matrix[:, i] for i, c in enumerate(columns)}}
    )


def _heavy_atom_map(mol_h) -> dict[int, int]:
    """Map every atom index in an explicit-H molecule to its heavy-atom index.

    A heavy atom maps to itself (`Chem.AddHs` appends hydrogens after the existing
    atoms without renumbering them, verified directly against `Chem.MolFromSmiles`
    on the same SMILES). A hydrogen maps to its single heavy-atom neighbour, so a
    contact ProLIF assigns to an H -- routine for `HBDonor`/`HBAcceptor`, since the
    donor hydrogen is what is actually within range -- rolls up onto the atom
    chemprop's graph can represent, rather than being silently dropped.
    """
    heavy_index = 0
    mapping: dict[int, int] = {}
    hydrogens: list = []
    for atom in mol_h.GetAtoms():
        if atom.GetSymbol() == "H":
            hydrogens.append(atom)
            continue
        mapping[atom.GetIdx()] = heavy_index
        heavy_index += 1
    for atom in hydrogens:
        neighbours = atom.GetNeighbors()
        if neighbours:
            mapping[atom.GetIdx()] = mapping[neighbours[0].GetIdx()]
    return mapping


def atom_features(
    poses_sdf: Path,
    protonated_pdb: Path,
    interactions: tuple[str, ...] = INTERACTIONS,
) -> dict[str, np.ndarray]:
    """Per-heavy-atom, multi-hot interaction-type contacts for chemprop's graph.

    `compute` answers "did this ligand touch residue X"; this answers "which atoms
    of the ligand made which kind of contact" -- the resolution chemprop's
    `--atom-features-path` can actually use, since it attaches a feature vector to
    every node of the molecular graph rather than one vector to the whole molecule.

    Multi-hot, not one binary flag: measured on 200 CYP2D6 poses, 14.2% of heavy
    atoms carry any contact and 5.3% of *those* carry more than one interaction
    type simultaneously (most often `Hydrophobic` and `PiStacking` on the same
    aromatic carbon) -- collapsing to a single flag would discard exactly the
    per-type distinction that made the molecule-level PLIF arm interpretable.

    Returns ``{Molecule_Name: array of shape (n_heavy_atoms, len(interactions))}``,
    one row per heavy atom in the same order `Chem.MolFromSmiles` on that
    compound's own SMILES would produce -- the order chemprop's featurizer uses by
    default (`add_h=False`, `reorder_atoms=False`). This alignment is what makes the
    array usable at all: it depends on the docked pose being embedded from
    `Chem.MolFromSmiles(smiles)` with no atom reordering before `AddHs`
    (`docking.ligand_sdf` does this), not on any explicit atom-map matching.
    Compounds absent from the docking output are not included in the returned dict
    -- `atom_features_npz` fills them with an all-zero, correctly-shaped block from
    a caller-supplied heavy-atom count, since "no pose" has no atoms to report on.
    """
    names, mols_h, fp = _run_fingerprint(poses_sdf, protonated_pdb, interactions)
    if fp is None:
        return {}

    interaction_index = {name: i for i, name in enumerate(interactions)}
    result: dict[str, np.ndarray] = {}
    for frame_idx, (name, mol_h) in enumerate(zip(names, mols_h, strict=True)):
        if name is None or frame_idx not in fp.ifp:
            continue
        heavy_map = _heavy_atom_map(mol_h)
        n_heavy = len({v for v in heavy_map.values()})
        features = np.zeros((n_heavy, len(interactions)), dtype=np.float32)
        for (_lig_res, _prot_res), per_interaction in fp.ifp[frame_idx].items():
            for interaction_type, contacts in per_interaction.items():
                col = interaction_index.get(interaction_type)
                if col is None:
                    continue
                for contact in contacts:
                    for lig_atom_idx in contact["indices"]["ligand"]:
                        heavy_idx = heavy_map.get(lig_atom_idx)
                        if heavy_idx is not None:
                            features[heavy_idx, col] = 1.0
        result[name] = features
    return result


def prune_atom_interactions(
    features: dict[str, np.ndarray], interactions: tuple[str, ...] = INTERACTIONS
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    """Drop interaction-type columns that never fire across the whole compound set.

    The molecule-level `prune` drops near-constant *columns*; the equivalent
    failure mode here is a column that is constant (always zero) across every
    atom of every molecule -- measured on CYP2D6, only 5 of 11 `INTERACTIONS`
    ever appear (`Hydrophobic`, `PiStacking`, `HBDonor`, `HBAcceptor`, `XBDonor`),
    so keeping all 11 would hand chemprop six always-zero atom-feature columns.
    Kept separate from `prune` because a rare-but-real column is still worth
    keeping here -- pruning by per-atom frequency, as `prune` does, would discard
    the rare-but-informative long tail (e.g. `XBAcceptor` firing on a handful of
    halogenated compounds), which is exactly the kind of contact this arm exists
    to surface.
    """
    if not features:
        return features, interactions
    stacked = np.concatenate(list(features.values()), axis=0)
    keep_mask = stacked.sum(axis=0) > 0
    kept_interactions = tuple(
        name for name, keep in zip(interactions, keep_mask, strict=True) if keep
    )
    if len(kept_interactions) == len(interactions):
        return features, interactions
    pruned = {name: arr[:, keep_mask] for name, arr in features.items()}
    return pruned, kept_interactions


def atom_features_npz(
    features: dict[str, np.ndarray],
    smiles_order: list[str],
    names_order: list[str],
    n_interactions: int,
    out_path: Path,
) -> Path:
    """Write per-compound atom features to the `.npz` chemprop's `--atom-features-path`
    expects: positional ``arr_0, arr_1, ...`` keys, one 2D array per row of the
    training/validation/test CSV, in that CSV's row order.

    Args:
        features: `{Molecule_Name: (n_heavy_atoms, n_interactions) array}` from
            `atom_features`, already pruned if desired.
        smiles_order: SMILES in the exact order they will be written to chemprop's
            CSV -- used only to recover each compound's heavy-atom count when it
            has no pose, via a fresh `Chem.MolFromSmiles` parse. Passed alongside
            `names_order` rather than looked up from a frame, so this function has
            no dependency on any particular caller's DataFrame shape.
        names_order: `Molecule_Name` in the same order as `smiles_order` and as
            chemprop's CSV rows -- the key `atom_features_npz` looks up `features`
            by. Order, not content, is what has to match chemprop's CSV; a
            mismatch here silently attaches the wrong atoms' contacts to the wrong
            molecule.
        n_interactions: Width of each per-atom feature vector -- needed to build a
            correctly-shaped all-zero block for a compound with no docked pose,
            since "no pose" must still emit an array of the right heavy-atom count
            and column width, never an empty or wrongly-shaped one.
        out_path: Destination `.npz`.

    Returns:
        `out_path`.
    """
    from rdkit import Chem

    arrays = []
    for smiles, name in zip(smiles_order, names_order, strict=True):
        if name in features:
            arrays.append(features[name])
            continue
        # No docked pose for this compound (embedding or docking failure) -- an
        # all-zero block of the right shape, the same "no pose, no contacts"
        # convention `matrix` uses for the molecule-level arm.
        mol = Chem.MolFromSmiles(smiles)
        n_heavy = mol.GetNumAtoms() if mol is not None else 0
        arrays.append(np.zeros((n_heavy, n_interactions), dtype=np.float32))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, *arrays)
    return out_path


def prune(frame: pl.DataFrame, min_frequency: float = MIN_FREQUENCY) -> pl.DataFrame:
    """Drop interaction bits that are near-constant across the set.

    A bit set in every ligand carries no information; one set in a handful is closer
    to a compound identifier than a feature, and with scaffold CV those rare bits
    concentrate in single folds where a tree can memorise them. Both tails go.
    """
    feature_columns = [c for c in frame.columns if c != "Molecule_Name"]
    if not feature_columns:
        return frame
    n = frame.height
    keep = []
    for column in feature_columns:
        rate = float(frame[column].sum()) / n if n else 0.0
        if min_frequency <= rate <= (1.0 - min_frequency):
            keep.append(column)
    return frame.select(["Molecule_Name", *keep])


def matrix(frame: pl.DataFrame, names: list[str]) -> np.ndarray:
    """Align a PLIF frame to `names`, returning a dense array.

    Compounds absent from `frame` -- those that failed to embed or dock -- get an
    all-zero row. That is the honest encoding: "no pose, therefore no contacts",
    and it keeps the feature matrix aligned with the label vector rather than
    forcing the caller to drop rows and diverge from the other arms' compound sets.
    """
    feature_columns = [c for c in frame.columns if c != "Molecule_Name"]
    lookup = {
        row["Molecule_Name"]: np.array([row[c] for c in feature_columns], dtype=np.float32)
        for row in frame.iter_rows(named=True)
    }
    width = len(feature_columns)
    zero = np.zeros(width, dtype=np.float32)
    return (
        np.vstack([lookup.get(name, zero) for name in names])
        if width
        else np.zeros((len(names), 0), dtype=np.float32)
    )


def summarise(frame: pl.DataFrame) -> pl.DataFrame:
    """Per-bit hit rate, most frequent first -- the interpretable half of this arm.

    This is what makes a PLIF result readable as chemistry: if the arm helps, the
    top rows say which residues and which interaction types did the work.
    """
    feature_columns = [c for c in frame.columns if c != "Molecule_Name"]
    if not feature_columns:
        return pl.DataFrame(schema={"bit": pl.Utf8, "residue": pl.Utf8, "interaction": pl.Utf8})
    n = frame.height or 1
    rows = [
        {
            "bit": c,
            "residue": c.split("|")[0],
            "interaction": c.split("|")[1],
            "hit_rate": float(frame[c].sum()) / n,
        }
        for c in feature_columns
    ]
    return pl.DataFrame(rows).sort("hit_rate", descending=True)
