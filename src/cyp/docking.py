"""Structure-based docking of challenge compounds into the four CYP isoforms.

Notebook 11's input stage. Everything here produces *poses and scores*; turning a
pose into a model feature is `plif.py` (protein contacts) or `fingerprints`'
``unimol`` entries (ligand geometry).

## Why smina and not gnina

gnina was the first choice and cannot run on this machine. It is CUDA-only: the sole
v1.3.3 release asset is a 1.96 GB CUDA-12.8 Linux binary and every documented build
path requires `nvcc`, against an arm64 Mac with no NVIDIA GPU. smina is the Vina fork
gnina itself derives from, installs from `brewsci/bio`, and exposes the same
`--autobox_ligand` / scoring-term interface. What is lost is gnina's CNN rescoring --
a learned pose score, which is precisely the part this notebook is measuring by other
means anyway (`plif` and Uni-Mol are both "encode the pose" strategies).

## Why a co-crystal ligand defines the box

Every receptor here is a *holo* structure chosen so its bound ligand marks the active
site. `--autobox_ligand` then derives the search box from that ligand's extent, which
is both more reproducible and better centred than hand-typed coordinates. An apo
structure would need the box guessed, and a CYP active site is a buried cavity where
a misplaced box silently docks everything into a surface groove -- the failure would
look like poses, not like an error.

## The heme is part of the receptor

CYP catalysis runs through a heme iron, and inhibitor binding is frequently either
coordination to that iron or stacking directly above it. Stripping HETATM records
wholesale -- the reflex when cleaning a PDB -- removes the heme and leaves a cavity
whose floor is missing, so ligands dock a few angstroms too deep. `prepare_receptor`
keeps HEM and drops only waters.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from . import constants

#: Where downloaded PDBs and prepared receptors live. Structures are an external
#: download like `data/raw`, so they get the same dated-snapshot treatment: a PDB
#: entry can be re-released with different coordinates, and a pose set is only
#: reproducible against the structure it was docked into.
STRUCTURE_DIR = constants.DATA_DIR / "structures"


@dataclass(frozen=True)
class Receptor:
    """One CYP isoform's docking target.

    Attributes:
        isoform: Challenge isoform name, e.g. ``"CYP2D6"``.
        pdb_id: RCSB entry the receptor is built from.
        ligand_code: Residue name of the co-crystal ligand that defines the box.
        chain: Chain to keep. Several of these structures are multi-copy
            (5TFT has four chains); docking into a merged multi-chain receptor
            puts a second copy's atoms inside the first copy's box.
        resolution: Reported resolution in angstrom, carried for provenance.
        note: Why this entry rather than another.
    """

    isoform: str
    pdb_id: str
    ligand_code: str
    chain: str
    resolution: float
    note: str


#: One holo structure per isoform. Chosen for resolution, a drug-like co-crystal
#: ligand, and -- for 2D6 -- for where the ligand sits: the box is only as good as
#: the ligand that defines it.
RECEPTORS: dict[str, Receptor] = {
    "CYP1A2": Receptor(
        isoform="CYP1A2",
        pdb_id="2HI4",
        ligand_code="BHF",
        chain="A",
        resolution=1.95,
        note="alpha-naphthoflavone complex; the best-resolved human 1A2 structure.",
    ),
    "CYP2C9": Receptor(
        isoform="CYP2C9",
        pdb_id="1R9O",
        ligand_code="FLP",
        chain="A",
        resolution=2.00,
        note="flurbiprofen complex; substrate bound in the catalytic site, "
        "unlike 1OG5 where warfarin sits in a peripheral pocket.",
    ),
    "CYP2D6": Receptor(
        isoform="CYP2D6",
        pdb_id="5TFT",
        ligand_code="P6U",
        chain="A",
        resolution=2.71,
        note="BACE1-inhibitor complex. Preferred over 3QM4 (prinomastat, 2.85A) "
        "because the ligand is drug-like and occupies the catalytic site, so the "
        "autobox lands where challenge compounds bind.",
    ),
    "CYP3A4": Receptor(
        isoform="CYP3A4",
        pdb_id="4NY4",
        ligand_code="2QH",
        chain="A",
        resolution=2.95,
        note="inhibitor complex. Preferred over the higher-resolution 1TQN (2.05A), "
        "which is effectively ligand-free and would need a guessed box in 3A4's "
        "unusually large and plastic cavity.",
    ),
}

#: Residue names dropped from a receptor. Waters are removed because docking treats
#: them as rigid obstructions; the heme (HEM) is deliberately **not** in this list.
_SOLVENT = frozenset({"HOH", "WAT", "DOD"})


def _pdb_path(pdb_id: str, snapshot: str) -> Path:
    return STRUCTURE_DIR / snapshot / f"{pdb_id}.pdb"


def download_structure(pdb_id: str, snapshot: str, force: bool = False) -> Path:
    """Fetch one PDB entry into a dated snapshot directory.

    Returns the local path, downloading only if absent (or `force`).
    """
    path = _pdb_path(pdb_id, snapshot)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    subprocess.run(["curl", "-sSLf", url, "-o", str(path)], check=True)
    if path.stat().st_size == 0:
        path.unlink()
        raise RuntimeError(f"Empty download for {pdb_id} from {url}")
    return path


def _atom_records(lines: list[str], chain: str) -> list[str]:
    """ATOM/HETATM records for one chain.

    Parsed by column position rather than whitespace splitting: PDB is a
    fixed-column format and atom names routinely run into the residue name
    field, so `line.split()` mis-parses exactly the crowded records that matter.
    """
    keep = []
    for line in lines:
        if line[:6] not in ("ATOM  ", "HETATM"):
            continue
        if line[21:22] != chain:
            continue
        keep.append(line)
    return keep


def prepare_receptor(
    receptor: Receptor, snapshot: str, force: bool = False
) -> tuple[Path, Path, Path]:
    """Split a PDB entry into the three files docking and PLIF need.

    Produces, under ``data/structures/<snapshot>/prepared/``:

    - ``<pdb>_rec.pdbqt`` -- receptor for smina (protein + heme, no waters).
    - ``<pdb>_box.sdf`` -- the co-crystal ligand, as ``--autobox_ligand``.
    - ``<pdb>_rec_h.pdb`` -- protein protonated at pH 7.4, for ProLIF.

    The third file is separate for a specific reason: ProLIF reads the receptor
    through MDAnalysis, which infers bond orders, and inference on an
    unprotonated crystal structure **segfaults** (exit 139/138, no traceback,
    easy to mistake for this repo's OpenMP trap). A crystal PDB has no hydrogens,
    so it must be protonated before ProLIF ever sees it. smina, by contrast, adds
    its own polar hydrogens and wants the unprotonated file.

    Returns:
        `(receptor_pdbqt, box_ligand_sdf, protonated_protein_pdb)`.
    """
    pdb = download_structure(receptor.pdb_id, snapshot, force=force)
    out_dir = pdb.parent / "prepared"
    out_dir.mkdir(parents=True, exist_ok=True)

    rec_pdbqt = out_dir / f"{receptor.pdb_id}_rec.pdbqt"
    box_sdf = out_dir / f"{receptor.pdb_id}_box.sdf"
    rec_h = out_dir / f"{receptor.pdb_id}_rec_h.pdb"
    if rec_pdbqt.exists() and box_sdf.exists() and rec_h.exists() and not force:
        return rec_pdbqt, box_sdf, rec_h

    lines = pdb.read_text().splitlines(keepends=True)
    chain_records = _atom_records(lines, receptor.chain)
    if not chain_records:
        raise RuntimeError(
            f"{receptor.pdb_id} has no chain {receptor.chain!r}. "
            "Check the entry -- chain labels differ between depositions."
        )

    ligand, protein_with_heme, protein_only = [], [], []
    for line in chain_records:
        resname = line[17:20].strip()
        if resname == receptor.ligand_code:
            ligand.append(line)
        elif resname not in _SOLVENT:
            protein_with_heme.append(line)
            if line[:6] == "ATOM  ":
                protein_only.append(line)

    if not ligand:
        raise RuntimeError(
            f"No {receptor.ligand_code!r} residue in {receptor.pdb_id} chain "
            f"{receptor.chain}. Without it there is no box to dock into."
        )

    lig_pdb = out_dir / f"{receptor.pdb_id}_box.pdb"
    rec_pdb = out_dir / f"{receptor.pdb_id}_rec.pdb"
    prot_pdb = out_dir / f"{receptor.pdb_id}_prot.pdb"
    lig_pdb.write_text("".join(ligand))
    rec_pdb.write_text("".join(protein_with_heme))
    prot_pdb.write_text("".join(protein_only))

    _obabel(lig_pdb, box_sdf)
    _obabel(rec_pdb, rec_pdbqt, "-xr")
    # `-h -p 7.4` protonates for the PLIF topology; see the docstring.
    _obabel(prot_pdb, rec_h, "-h", "-p", "7.4")
    return rec_pdbqt, box_sdf, rec_h


def _obabel(src: Path, dst: Path, *flags: str) -> None:
    result = subprocess.run(
        ["obabel", str(src), "-O", str(dst), *flags],
        capture_output=True,
        text=True,
    )
    if not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError(f"obabel failed: {src} -> {dst}\n{result.stderr}")


def ligand_sdf(
    smiles: list[str],
    names: list[str],
    path: Path,
    seed: int = 0xF00D,
    timeout_s: float | None = 30.0,
) -> tuple[Path, list[str]]:
    """Embed and MMFF-minimise `smiles` into a multi-molecule SDF for docking.

    Docking needs a starting 3D conformer per ligand. Compounds that fail to embed
    are skipped and reported rather than written as 2D -- smina would accept a flat
    molecule and produce a pose from it, which is a silent correctness failure.

    Uses `pharmacophore._timeout` for the same reason notebook 09 needed it: ETKDG
    has no wall-clock bound and a handful of large flexible molecules can stall a
    batch indefinitely.

    Returns:
        `(path, written_names)` -- the SDF and the names actually embedded, in
        file order, so a caller can align poses back to compounds.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from .pharmacophore import ConformerTimeout, _timeout

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(path))
    written: list[str] = []
    try:
        for smi, name in zip(smiles, names, strict=True):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = seed
            try:
                with _timeout(timeout_s):
                    if AllChem.EmbedMolecule(mol, params) != 0:
                        continue
                    AllChem.MMFFOptimizeMolecule(mol)
            except (ConformerTimeout, ValueError):
                continue
            mol.SetProp("_Name", name)
            writer.write(mol)
            written.append(name)
    finally:
        writer.close()
    return path, written


def _dock_one_call(
    receptor_pdbqt: Path,
    box_sdf: Path,
    ligands_sdf: Path,
    out_sdf: Path,
    exhaustiveness: int,
    num_modes: int,
    autobox_add: float,
    seed: int,
    cpu: int,
) -> Path:
    """One smina invocation over a (possibly chunked) multi-molecule SDF.

    The primitive `dock_in_chunks` calls once per chunk. Not meant to be called
    directly on the full compound set -- see `dock_in_chunks` for why a single
    multi-hour call is the wrong shape for this notebook.
    """
    out_sdf.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_sdf.with_suffix(".partial.sdf")
    cmd = [
        "smina",
        "-r",
        str(receptor_pdbqt),
        "-l",
        str(ligands_sdf),
        "--autobox_ligand",
        str(box_sdf),
        "--autobox_add",
        str(autobox_add),
        "-o",
        str(tmp),
        "--seed",
        str(seed),
        "--exhaustiveness",
        str(exhaustiveness),
        "--num_modes",
        str(num_modes),
        "--cpu",
        str(cpu),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not tmp.exists():
        raise RuntimeError(f"smina failed ({result.returncode}):\n{result.stderr[-2000:]}")
    # Written to a partial file and renamed so an interrupted run never leaves a
    # truncated SDF that a resumability check would treat as complete.
    tmp.rename(out_sdf)
    return out_sdf


def split_sdf(ligands_sdf: Path, chunk_size: int, chunk_dir: Path) -> list[Path]:
    """Split a multi-molecule SDF into `chunk_size`-molecule files.

    Splitting on the raw `$$$$` record separator rather than reparsing through
    RDKit: this only needs to move bytes, and re-parsing molecules that already
    embedded successfully risks losing one to a `Chem.SDMolSupplier` edge case for
    no reason.
    """
    chunk_dir.mkdir(parents=True, exist_ok=True)
    records = ligands_sdf.read_text().split("$$$$\n")
    # A trailing split element after the final "$$$$\n" is an empty string; drop it
    # rather than write a chunk containing no molecules.
    records = [r for r in records if r.strip()]

    paths = []
    for start in range(0, len(records), chunk_size):
        chunk = records[start : start + chunk_size]
        path = chunk_dir / f"chunk_{start:05d}.sdf"
        path.write_text("$$$$\n".join(chunk) + "$$$$\n")
        paths.append(path)
    return paths


def dock_in_chunks(
    receptor_pdbqt: Path,
    box_sdf: Path,
    ligands_sdf: Path,
    out_sdf: Path,
    chunk_size: int = 100,
    exhaustiveness: int = 8,
    num_modes: int = 1,
    autobox_add: float = 4.0,
    seed: int = 42,
    cpu: int = 8,
    force: bool = False,
    on_chunk=None,
) -> Path:
    """Dock `ligands_sdf` in batches of `chunk_size`, concatenating poses into `out_sdf`.

    A single smina call over the full compound set (~1,400-2,300 per isoform) gives
    no signal until it finishes -- no per-compound progress, and on this machine
    exhaustiveness-8 timing has ranged 4x-100x compound-for-identical-compound
    across runs, purely from contention (see CLAUDE.md's `flag_contention`).
    Chunking surfaces real per-chunk timing early, lets a killed run resume at the
    chunk it reached instead of the isoform, and gives `on_chunk` a place to check
    system state (thermal budget, competing processes) and pause between chunks --
    the fan audibly running continuously during the first full-isoform attempt is
    exactly the situation this exists to make visible and interruptible.

    Skips the run if `out_sdf` already exists (`force` overrides). Chunk files are
    kept in `out_sdf`'s directory under a `<out_sdf.stem>_chunks/` subdirectory and
    are resumable individually: a chunk whose output already exists is not re-docked.

    Args:
        chunk_size: Molecules per smina call. 100 was chosen so a chunk finishes in
            single-digit minutes even at bad-case contention, which is short enough
            to check thermal state between chunks without losing much wall time to
            per-chunk fixed overhead (receptor loading, box computation).
        on_chunk: Optional `(chunk_index, n_chunks, seconds, poses_path) -> None`,
            called after each chunk completes and before the next one starts. A
            caller can use this to inspect CPU/thermal state and `time.sleep` a
            cooldown, or to raise and stop the run early.
    """
    if out_sdf.exists() and not force:
        return out_sdf

    out_sdf.parent.mkdir(parents=True, exist_ok=True)
    chunk_dir = out_sdf.parent / f"{out_sdf.stem}_chunks"
    ligand_chunks = split_sdf(ligands_sdf, chunk_size, chunk_dir / "ligands")
    pose_dir = chunk_dir / "poses"
    pose_dir.mkdir(parents=True, exist_ok=True)

    n_chunks = len(ligand_chunks)
    pose_paths = []
    for i, ligand_chunk in enumerate(ligand_chunks):
        pose_chunk = pose_dir / f"poses_{i:05d}.sdf"
        if not pose_chunk.exists():
            t0 = time.time()
            _dock_one_call(
                receptor_pdbqt,
                box_sdf,
                ligand_chunk,
                pose_chunk,
                exhaustiveness=exhaustiveness,
                num_modes=num_modes,
                autobox_add=autobox_add,
                seed=seed,
                cpu=cpu,
            )
            elapsed = time.time() - t0
        else:
            elapsed = 0.0  # cached; on_chunk still fires so a caller sees progress
        pose_paths.append(pose_chunk)
        if on_chunk is not None:
            on_chunk(i, n_chunks, elapsed, pose_chunk)

    # Concatenate chunk poses into the single file the rest of the pipeline expects.
    tmp = out_sdf.with_suffix(".partial.sdf")
    with tmp.open("w") as handle:
        for pose_chunk in pose_paths:
            text = pose_chunk.read_text()
            if text and not text.endswith("\n"):
                text += "\n"
            handle.write(text)
    tmp.rename(out_sdf)
    return out_sdf


def _supplier(poses_sdf: Path, remove_hs: bool):
    """`SDMolSupplier` that yields nothing instead of raising on an empty file.

    RDKit raises `OSError: Invalid input file` on an empty or truncated SDF rather
    than returning an empty iterator. Docking legitimately produces no poses -- an
    isoform where every ligand failed to embed, or a box nothing fits -- and an
    overnight four-isoform run must not die on the *reading* step after paying for
    the docking.
    """
    from rdkit import Chem

    if not poses_sdf.exists() or poses_sdf.stat().st_size == 0:
        return []
    try:
        return list(Chem.SDMolSupplier(str(poses_sdf), removeHs=remove_hs))
    except OSError:
        return []


def score_frame(poses_sdf: Path) -> pl.DataFrame:
    """Docking scores per compound -- the A3 arm's features.

    smina writes `minimizedAffinity` (kcal/mol) on each pose. That single number is
    the whole of what classical docking claims about a compound, so it is worth
    testing on its own: if the affinity alone carries signal, an elaborate pose
    encoding has to beat it to justify itself.
    """
    rows = []
    for mol in _supplier(poses_sdf, remove_hs=False):
        if mol is None:
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else None
        affinity = (
            float(mol.GetProp("minimizedAffinity"))
            if mol.HasProp("minimizedAffinity")
            else float("nan")
        )
        rows.append({"Molecule_Name": name, "docking_affinity": affinity})
    return (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(schema={"Molecule_Name": pl.Utf8, "docking_affinity": pl.Float64})
    )


def pose_coordinates(poses_sdf: Path, remove_hs: bool = True) -> tuple[list[str], list, list]:
    """Atoms and coordinates per docked pose, for Uni-Mol's custom-conformer input.

    Returns `(names, atoms_list, coords_list)` aligned by index -- the exact shape
    `unimol_tools.UniMolRepr.get_repr` accepts as ``{"atoms": ..., "coordinates": ...}``.

    Hydrogens are dropped by default to match the ``mol_pre_no_h`` Uni-Mol weights,
    which is the checkpoint `fingerprints.unimol_repr` loads.
    """
    names, atoms_list, coords_list = [], [], []
    for mol in _supplier(poses_sdf, remove_hs=remove_hs):
        if mol is None or mol.GetNumConformers() == 0:
            continue
        names.append(mol.GetProp("_Name") if mol.HasProp("_Name") else None)
        atoms_list.append([a.GetSymbol() for a in mol.GetAtoms()])
        coords_list.append(mol.GetConformer().GetPositions().astype(np.float32))
    return names, atoms_list, coords_list


def smina_available() -> bool:
    """Whether the smina binary is on PATH."""
    from shutil import which

    return which("smina") is not None
