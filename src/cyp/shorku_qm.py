"""Fetch and read the xTB conformer ensembles another challenge team published.

Source: Oleg Gromov (challenge user `Shorku`), "OpenADMET Blind Challenge: Predicting
CYP Inhibition", https://github.com/Shorku/OpenADMET_Blind_Challenge_Predicting_CYP_Inhibition,
MIT licence, pinned at commit `a98faaf` (2026-10-19, "add calculated data"). It is the
input data for their RhNet2TBequi model (Gromov, JCTC 2025, doi:10.1021/acs.jctc.5c00425).

## What it is

12,037 structures (6,862 InChIKey skeletons: tautomers and stereoisomers of one compound
are separate structures) covering every challenge training and blind compound, with
102,624 conformers. Their protocol: up to 50 RDKit conformers, MMFF94 where parameters
exist, deduplicate, xTB optimisation, drop conformers above a 10 kJ/mol window,
deduplicate again, then PTB orbitals. Three files:

- `index.csv`: `structure_id` (BLAKE2b of canonical SMILES), `SMILES`, `InChIKey`.
- `conf_props.zip`: per-conformer PTB dipole, PTB HOMO-LUMO gap, xTB total energy, volume,
  surface, radius of gyration, three ellipsoid axes.
- `geoms.zip`: one XMOL `.xyz` per conformer, the actual coordinates.

The density matrices their model trains on are **not** published, so this cannot
reproduce their model; what it offers is physically optimised conformer ensembles with
energies, which our 3D encoders (`cyp.encoders`) otherwise build from one random ETKDG
embedding.

## Why it lives outside the dated snapshot directories

`data/external/YYYYMMDD/` is resolved by `constants.external_snapshot_dir`, which picks
the newest dated folder by default. A dated folder holding only these files would become
"latest" for the PubChem, ChEMBL and Tox21 loaders and break them. So snapshots go to
`data/external/shorku_qm/YYYYMMDD/`, which the snapshot glob never matches.

`geoms.zip` is 134 MB, over GitHub's file limit, so it is gitignored and re-fetched (and
hash-checked) by `download`. The two small files are committed like every other
external snapshot.

    python -m cyp.shorku_qm                  # download to today's snapshot
    python -m cyp.shorku_qm --date 20261024  # write to a specific snapshot
"""

from __future__ import annotations

import argparse
import hashlib
import io
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from . import constants as C

SOURCE_DIR = C.EXTERNAL_DIR / "shorku_qm"

REPO = "Shorku/OpenADMET_Blind_Challenge_Predicting_CYP_Inhibition"
COMMIT = "a98faaf41e0630046d0c6f11bd11b6cfe55687fc"

#: `raw.githubusercontent.com` returns an LFS pointer for the zips; the media host
#: resolves the pointer to the blob. Hashes are the LFS oids, so a pointer fetched by
#: mistake (134 bytes) fails the check instead of failing later as a corrupt zip.
FILES: dict[str, tuple[str, str | None]] = {
    "index.csv": (f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/data/index.csv", None),
    "conf_props.zip": (
        f"https://media.githubusercontent.com/media/{REPO}/{COMMIT}/data/conf_props.zip",
        "4fa12cc4d944810ca31e0b989316bee5acdc0bf7cd8d1e7945b3e2aca9e76304",
    ),
    "geoms.zip": (
        f"https://media.githubusercontent.com/media/{REPO}/{COMMIT}/data/geoms.zip",
        "b1aabbca56f6b8a7213dd69af701df7ad6acb6e3617618595adc69899eaa4510",
    ),
}

#: Boltzmann temperature for conformer weights. Their energy window is 10 kJ/mol, so at
#: 298 K the highest-energy conformer kept still carries ~1.8% of the lowest's weight.
TEMPERATURE_K = 298.15
_HARTREE_KJ_MOL = 2625.4996
_R_KJ_MOL_K = 0.008314462618

_SOURCE_NOTE = f"""# xTB conformer ensembles from Shorku (Oleg Gromov)

Upstream: https://github.com/{REPO}/tree/{COMMIT}/data
Licence: MIT, Copyright (c) 2026 Oleg Gromov
Method: Gromov, J. Chem. Theory Comput. 2025, doi:10.1021/acs.jctc.5c00425

Fetched by `python -m cyp.shorku_qm`. See `src/cyp/shorku_qm.py` for what the files hold.
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(snapshot: str | None = None, force: bool = False) -> Path:
    """Fetch the three files at the pinned commit into a dated snapshot directory."""
    snapshot = snapshot or f"{date.today():%Y%m%d}"
    directory = SOURCE_DIR / snapshot
    directory.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in FILES.items():
        dest = directory / name
        if dest.exists() and not force:
            print(f"skip   {name}  already in {snapshot}")
        else:
            urllib.request.urlretrieve(url, dest)
            print(f"got    {name}  {dest.stat().st_size:,} bytes")
        if expected is not None and _sha256(dest) != expected:
            raise RuntimeError(
                f"{dest} does not match the upstream LFS oid {expected[:12]}...; "
                "delete it and re-fetch."
            )
    (directory / "SOURCE.md").write_text(_SOURCE_NOTE)
    return directory


def snapshot_dir(snapshot: str | None = None) -> Path:
    """The requested snapshot, or the newest one on disk."""
    available = sorted(p.name for p in SOURCE_DIR.glob(C.SNAPSHOT_GLOB) if p.is_dir())
    if not available:
        raise FileNotFoundError(f"No snapshots in {SOURCE_DIR}. Run `python -m cyp.shorku_qm`.")
    snapshot = snapshot or available[-1]
    if snapshot not in available:
        raise FileNotFoundError(f"Snapshot {snapshot!r} not in {SOURCE_DIR}: {available}")
    return SOURCE_DIR / snapshot


def load_index(snapshot: str | None = None) -> pl.DataFrame:
    """`structure_id`, `SMILES`, `InChIKey`, plus the `skeleton` key used for matching."""
    return pl.read_csv(snapshot_dir(snapshot) / "index.csv").with_columns(
        pl.col("InChIKey").str.split("-").list.first().alias("skeleton")
    )


def load_conformer_properties(snapshot: str | None = None) -> pl.DataFrame:
    """Per-conformer properties, with relative energy and Boltzmann weight added.

    `rel_energy_kj_mol` and `boltzmann_weight` are within one structure, so weights sum
    to 1 per `structure_id`. Comparing energies across tautomers is done separately in
    `match_structures`, since only same-formula structures share an energy zero.
    """
    with zipfile.ZipFile(snapshot_dir(snapshot) / "conf_props.zip") as archive:
        (member,) = [n for n in archive.namelist() if not n.endswith("/")]
        frame = pl.read_csv(io.BytesIO(archive.read(member)))
    rel = (pl.col("xTB_E_Eh") - pl.col("xTB_E_Eh").min().over("structure_id")) * _HARTREE_KJ_MOL
    frame = frame.with_columns(rel.alias("rel_energy_kj_mol"))
    weight = (-pl.col("rel_energy_kj_mol") / (_R_KJ_MOL_K * TEMPERATURE_K)).exp()
    return frame.with_columns(
        (weight / weight.sum().over("structure_id")).alias("boltzmann_weight")
    )


def match_structures(smiles: list[str], snapshot: str | None = None) -> pl.DataFrame:
    """Map each of our SMILES to one upstream structure.

    An exact canonical-SMILES match is used where it exists. Otherwise, among the
    structures sharing our compound's InChIKey skeleton, the one with the lowest xTB
    minimum energy, which is a fair tie-break between tautomers because they share a
    formula. Returns `SMILES`, `structure_id`, `structure_smiles`, `match` ("exact",
    "skeleton" or null when the compound is absent upstream).
    """
    from rdkit import Chem

    from .external import inchikey_skeleton

    index = load_index(snapshot)
    emin = (
        load_conformer_properties(snapshot)
        .group_by("structure_id")
        .agg(pl.col("xTB_E_Eh").min().alias("emin"))
    )
    index = index.join(emin, on="structure_id")
    by_smiles = dict(zip(index["SMILES"].to_list(), index["structure_id"].to_list(), strict=True))
    best_by_skeleton = index.sort("emin").unique("skeleton", keep="first")
    by_skeleton = dict(
        zip(
            best_by_skeleton["skeleton"].to_list(),
            best_by_skeleton["structure_id"].to_list(),
            strict=True,
        )
    )
    smiles_of = dict(zip(index["structure_id"].to_list(), index["SMILES"].to_list(), strict=True))

    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        canonical = Chem.MolToSmiles(mol) if mol is not None else None
        if canonical in by_smiles:
            sid, how = by_smiles[canonical], "exact"
        elif (key := inchikey_skeleton(smi)) in by_skeleton:
            sid, how = by_skeleton[key], "skeleton"
        else:
            sid, how = None, None
        rows.append((smi, sid, smiles_of.get(sid), how))
    return pl.DataFrame(
        rows, schema=["SMILES", "structure_id", "structure_smiles", "match"], orient="row"
    )


def structure_sets(smiles: list[str], snapshot: str | None = None) -> pl.DataFrame:
    """Every upstream structure that is a valid reading of each of our SMILES.

    40% of our compounds have undefined stereocentres, and upstream enumerated every
    stereoisomer of those. Picking one by xTB energy would be arbitrary (enantiomers are
    degenerate) and would hand a stereo-aware encoder a configuration the label never
    specified. So the set is every isomer consistent with whatever stereo our SMILES
    does define, to be averaged with equal weight: the measured sample's composition
    was set by synthesis, not by thermodynamics.

    Rules, recorded in the `rule` column:
      - "exact": our canonical SMILES is upstream; the set is that one structure.
      - "stereo": same skeleton, and our molecule is a chirality-aware substructure of
        the upstream one with the same heavy-atom count. Undefined centres in ours match
        either configuration; defined ones must agree. RDKit's chiral matching covers
        tetrahedral centres, not double-bond geometry.
      - "tautomer": same skeleton, but no structure passes the stereo test (a different
        tautomer or charge form). The lowest-xTB-energy structure, which is fair between
        forms with the same formula.
      - null: the compound is absent upstream; the caller falls back to its own conformer.

    Returns one row per (SMILES, structure): `SMILES`, `structure_id`, `structure_smiles`,
    `rule`, with null structure columns for absent compounds.
    """
    from rdkit import Chem

    from .external import inchikey_skeleton

    index = load_index(snapshot)
    emin = (
        load_conformer_properties(snapshot)
        .group_by("structure_id")
        .agg(pl.col("xTB_E_Eh").min().alias("emin"))
    )
    index = index.join(emin, on="structure_id").sort("emin")
    by_smiles = dict(zip(index["SMILES"].to_list(), index["structure_id"].to_list(), strict=True))
    groups: dict[str, list[tuple[str, str]]] = {}
    for sid, smi, key in index.select("structure_id", "SMILES", "skeleton").iter_rows():
        groups.setdefault(key, []).append((sid, smi))

    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        canonical = Chem.MolToSmiles(mol)
        if canonical in by_smiles:
            rows.append((smi, by_smiles[canonical], canonical, "exact"))
            continue
        group = groups.get(inchikey_skeleton(smi), [])
        if not group:
            rows.append((smi, None, None, None))
            continue
        consistent = []
        for sid, upstream in group:
            target = Chem.MolFromSmiles(upstream)
            if target.GetNumHeavyAtoms() == mol.GetNumHeavyAtoms() and target.HasSubstructMatch(
                mol, useChirality=True
            ):
                consistent.append((sid, upstream))
        if consistent:
            rows.extend((smi, sid, upstream, "stereo") for sid, upstream in consistent)
        else:
            # `group` is sorted by xTB minimum energy, so the head is the lowest form.
            rows.append((smi, group[0][0], group[0][1], "tautomer"))
    return pl.DataFrame(
        rows, schema=["SMILES", "structure_id", "structure_smiles", "rule"], orient="row"
    )


def read_conformers(
    structure_ids: list[str], snapshot: str | None = None
) -> dict[str, dict[int, tuple[list[str], np.ndarray]]]:
    """Coordinates for every conformer of the given structures, read from the zip.

    Returns `{structure_id: {conf_id: (elements, coords[n_atoms, 3])}}`. Reads members in
    place rather than extracting 102,624 small files to disk.
    """
    wanted = set(structure_ids)
    out: dict[str, dict[int, tuple[list[str], np.ndarray]]] = {}
    with zipfile.ZipFile(snapshot_dir(snapshot) / "geoms.zip") as archive:
        for name in archive.namelist():
            stem = Path(name).name
            if not stem.endswith(".xyz"):
                continue
            sid, conf, _ = stem.rsplit(".", 2)
            if sid not in wanted:
                continue
            lines = archive.read(name).decode().splitlines()
            n_atoms = int(lines[0])
            atoms = [line.split() for line in lines[2 : 2 + n_atoms]]
            elements = [a[0] for a in atoms]
            coords = np.array([[float(v) for v in a[1:4]] for a in atoms])
            out.setdefault(sid, {})[int(conf)] = (elements, coords)
    return out


def mol_with_coordinates(structure_smiles: str, elements: list[str], coords: np.ndarray):
    """RDKit molecule (explicit Hs) carrying an upstream conformer.

    The `.xyz` files carry no connectivity, so atom order is assumed to be RDKit's
    `AddHs(MolFromSmiles(structure_smiles))` order, which their RDKit-based embedding
    step produces. That assumption is checked, never trusted: every element must match
    and every bond must be under 1.3x the sum of its atoms' covalent radii. A flat
    length cutoff is wrong: a disulfide is 2.05 A and legitimate. On the full 82,056
    conformers used in `26` the longest bond is 1.08x, so 1.3x only catches a wrong
    atom order, which puts bonded atoms 2-4x apart. A mismatch raises.
    """
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    mol = Chem.AddHs(Chem.MolFromSmiles(structure_smiles))
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    if symbols != elements:
        raise ValueError(f"atom order mismatch for {structure_smiles}")
    table = Chem.GetPeriodicTable()
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        radii = table.GetRcovalent(bond.GetBeginAtom().GetAtomicNum()) + table.GetRcovalent(
            bond.GetEndAtom().GetAtomicNum()
        )
        if np.linalg.norm(coords[i] - coords[j]) > 1.3 * radii:
            raise ValueError(f"bond {i}-{j} is {np.linalg.norm(coords[i] - coords[j]):.2f} A")
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)
    return mol


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--date", default=None, help="snapshot date YYYYMMDD (default today)")
    parser.add_argument("--force", action="store_true", help="re-fetch files already present")
    args = parser.parse_args()
    download(args.date, args.force)
