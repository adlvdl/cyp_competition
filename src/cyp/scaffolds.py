"""Ring-based scaffold decomposition and scaffold-level coverage analysis.

Ported from the PXR repo's `1e_scaffold_analysis.py` (`decompose_scaffold_network`
and the counting cells that follow it).

This is deliberately *not* Bemis-Murcko. `cv.murcko_scaffold` gives one scaffold per
molecule and is the right thing for grouping CV folds; this module gives a molecule
several scaffolds at three nested levels of generality:

- ``ring_system`` -- one fused/bridged/spiro ring system, substituents stripped.
- ``linked_ring_systems`` -- two ring systems plus the linker between them.
- ``full_scaffold`` -- three or more connected ring systems with their linkers.

The reason to want all three: Murcko answers "do train and test share a scaffold?",
which for a 75-hits x ~10-analogs test set is usually no. The network answers the
more useful question of whether they share *ring systems* -- a test compound can be
Murcko-novel while every ring in it is well covered by the training set, and that is
a very different modelling situation from genuine novelty.
"""

from __future__ import annotations

import itertools
from collections import defaultdict, deque
from collections.abc import Sequence

import polars as pl
from rdkit import Chem

SCAFFOLD_TYPES = ("ring_system", "linked_ring_systems", "full_scaffold")


def _ring_systems(mol: Chem.Mol) -> list[frozenset[int]]:
    """Atom indices of each ring system.

    A ring system is the union of SSSR rings sharing at least one atom, so fused,
    bridged and spiro rings all merge into a single system rather than being
    reported as separate rings.
    """
    atom_rings = mol.GetRingInfo().AtomRings()
    if not atom_rings:
        return []

    adjacency: dict[int, set[int]] = defaultdict(set)
    for i, j in itertools.combinations(range(len(atom_rings)), 2):
        if set(atom_rings[i]) & set(atom_rings[j]):
            adjacency[i].add(j)
            adjacency[j].add(i)

    visited = [False] * len(atom_rings)
    systems: list[frozenset[int]] = []
    for start in range(len(atom_rings)):
        if visited[start]:
            continue
        component: list[int] = []
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if visited[node]:
                continue
            visited[node] = True
            component.append(node)
            queue.extend(adjacency[node])
        systems.append(frozenset(a for ring in component for a in atom_rings[ring]))
    return systems


def _linker_atoms(
    adjacency: dict[int, set[int]],
    first: frozenset[int],
    second: frozenset[int],
    non_ring: set[int],
) -> frozenset[int] | None:
    """Non-ring atoms on the shortest path between two ring systems.

    Returns an empty frozenset when the systems are directly bonded, and None when
    they are not connected at all (a disconnected structure, e.g. a salt).
    """
    for atom in first:
        if any(neighbour in second for neighbour in adjacency[atom]):
            return frozenset()

    previous: dict[int, int] = {}
    queue: deque[int] = deque()
    for atom in first:
        for neighbour in adjacency[atom]:
            if neighbour in non_ring and neighbour not in previous:
                previous[neighbour] = atom
                queue.append(neighbour)

    while queue:
        current = queue.popleft()
        if current in second:
            path: list[int] = []
            node = current
            while node not in first:
                if node in non_ring:
                    path.append(node)
                node = previous[node]
            return frozenset(path)
        for neighbour in adjacency[current]:
            if neighbour not in previous and (neighbour in non_ring or neighbour in second):
                previous[neighbour] = current
                queue.append(neighbour)
    return None


def _canonical_fragment(mol: Chem.Mol, atoms: frozenset[int]) -> str | None:
    """Canonical SMILES for an atom subset, or None when it cannot be re-parsed.

    The parent is kekulized first: an aromatic ring extracted as a fragment loses the
    ring context that made it aromatic, and RDKit will refuse to re-parse it.
    """
    if not atoms:
        return None
    try:
        kekulized = Chem.RWMol(mol)
        Chem.Kekulize(kekulized, clearAromaticFlags=False)
        smiles = Chem.MolFragmentToSmiles(kekulized, sorted(atoms), kekuleSmiles=True)
        if not smiles:
            return None
        parsed = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(parsed) if parsed is not None else None
    except Exception:
        return None


def decompose(smiles: Sequence[str]) -> pl.DataFrame:
    """Decompose each molecule into its ring-based scaffolds at all three levels.

    Acyclic molecules produce no rows; unparseable SMILES produce one row carrying
    `parse_error`.

    Args:
        smiles: SMILES strings, ideally deduplicated -- ring perception is the
            expensive step and there is no caching here.

    Returns:
        Long frame: SMILES, scaffold_smiles, scaffold_type, scaffold_heavy_atoms,
        parse_error. One row per (molecule, distinct scaffold).
    """
    input_smiles: list[str] = []
    scaffold_smiles: list[str | None] = []
    scaffold_types: list[str | None] = []
    heavy_atoms: list[int | None] = []
    errors: list[str | None] = []

    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            input_smiles.append(smi)
            scaffold_smiles.append(None)
            scaffold_types.append(None)
            heavy_atoms.append(None)
            errors.append(f"Invalid SMILES: {smi!r}")
            continue

        systems = _ring_systems(mol)
        if not systems:
            continue

        ring_atoms = {a for system in systems for a in system}
        non_ring = set(range(mol.GetNumAtoms())) - ring_atoms

        adjacency: dict[int, set[int]] = defaultdict(set)
        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adjacency[u].add(v)
            adjacency[v].add(u)

        # Keyed by canonical SMILES so a scaffold reached two ways is emitted once,
        # at the first (most specific) level that produced it. `mol` and `found` are
        # bound as defaults rather than captured, so the closure cannot read a later
        # iteration's molecule if it ever outlives this one.
        found: dict[str, str] = {}

        def emit(
            atoms: frozenset[int],
            scaffold_type: str,
            mol: Chem.Mol = mol,
            found: dict[str, str] = found,
        ) -> None:
            canonical = _canonical_fragment(mol, atoms)
            if canonical and canonical not in found:
                found[canonical] = scaffold_type

        for system in systems:
            emit(system, "ring_system")

        linkers: dict[tuple[int, int], frozenset[int] | None] = {}
        connected: dict[int, set[int]] = defaultdict(set)
        for i, first in enumerate(systems):
            for j, second in enumerate(systems):
                if j <= i:
                    continue
                linker = _linker_atoms(adjacency, first, second, non_ring)
                linkers[(i, j)] = linker
                if linker is not None:
                    connected[i].add(j)
                    connected[j].add(i)
                    emit(first | second | linker, "linked_ring_systems")

        if len(systems) >= 3:
            seen = [False] * len(systems)
            for start in range(len(systems)):
                if seen[start]:
                    continue
                component: list[int] = []
                queue = deque([start])
                while queue:
                    node = queue.popleft()
                    if seen[node]:
                        continue
                    seen[node] = True
                    component.append(node)
                    queue.extend(connected[node])

                if len(component) >= 3:
                    atoms = frozenset(a for idx in component for a in systems[idx])
                    for a, b in itertools.combinations(component, 2):
                        linker = linkers.get((min(a, b), max(a, b)))
                        if linker:
                            atoms = atoms | linker
                    emit(atoms, "full_scaffold")

        for canonical, scaffold_type in found.items():
            parsed = Chem.MolFromSmiles(canonical)
            input_smiles.append(smi)
            scaffold_smiles.append(canonical)
            scaffold_types.append(scaffold_type)
            heavy_atoms.append(parsed.GetNumHeavyAtoms() if parsed is not None else None)
            errors.append(None)

    return pl.DataFrame(
        {
            "SMILES": pl.Series(input_smiles, dtype=pl.Utf8),
            "scaffold_smiles": pl.Series(scaffold_smiles, dtype=pl.Utf8),
            "scaffold_type": pl.Series(scaffold_types, dtype=pl.Utf8),
            "scaffold_heavy_atoms": pl.Series(heavy_atoms, dtype=pl.Int32),
            "parse_error": pl.Series(errors, dtype=pl.Utf8),
        }
    )


def scaffold_counts(
    scaffold_long: pl.DataFrame,
    membership: pl.DataFrame,
    flag_columns: Sequence[str],
) -> pl.DataFrame:
    """Molecules per scaffold, broken down by dataset membership.

    A molecule can carry several membership flags at once, so the per-flag counts
    may sum to more than `n_total`. That is intended -- the interesting scaffolds are
    exactly the ones present in both training and test.

    Args:
        scaffold_long: Output of `decompose`.
        membership: Frame with SMILES and one boolean column per flag.
        flag_columns: Boolean membership columns to count.

    Returns:
        One row per scaffold with `n_total` and one `n_{flag}` column per flag.
    """
    joined = (
        scaffold_long.filter(pl.col("parse_error").is_null())
        .join(membership, on="SMILES", how="left")
        .unique(subset=["scaffold_smiles", "SMILES"])
    )

    aggregations = [
        pl.col("scaffold_heavy_atoms").first(),
        pl.col("SMILES").n_unique().alias("n_total"),
    ]
    aggregations += [
        pl.col(flag).filter(pl.col(flag)).len().alias(f"n_{flag}") for flag in flag_columns
    ]

    return (
        joined.group_by(["scaffold_smiles", "scaffold_type"])
        .agg(aggregations)
        .sort(["scaffold_type", "n_total"], descending=[False, True])
    )


def scaffold_activity(
    scaffold_long: pl.DataFrame,
    values: pl.DataFrame,
    endpoint: str,
    min_molecules: int = 3,
) -> pl.DataFrame:
    """Per-scaffold activity statistics for one endpoint.

    A scaffold whose members span a wide activity range is one where potency is
    driven by substituents rather than the core -- the situation a fingerprint model
    handles worst, and a useful complement to the pairwise cliff analysis because it
    does not depend on a similarity threshold.

    Args:
        scaffold_long: Output of `decompose`.
        values: Frame with SMILES and the endpoint column.
        endpoint: Activity column to summarise.
        min_molecules: Drop scaffolds with fewer labelled molecules than this; a
            spread computed on two compounds is not a spread.

    Returns:
        One row per scaffold: n_molecules, mean/min/max/spread of the endpoint.
    """
    labelled = values.filter(pl.col(endpoint).is_not_null()).select(["SMILES", endpoint])

    return (
        scaffold_long.filter(pl.col("parse_error").is_null())
        .join(labelled, on="SMILES", how="inner")
        .unique(subset=["scaffold_smiles", "SMILES"])
        .group_by(["scaffold_smiles", "scaffold_type"])
        .agg(
            pl.col("scaffold_heavy_atoms").first(),
            pl.len().alias("n_molecules"),
            pl.col(endpoint).mean().round(2).alias("mean_value"),
            pl.col(endpoint).min().round(2).alias("min_value"),
            pl.col(endpoint).max().round(2).alias("max_value"),
        )
        .with_columns((pl.col("max_value") - pl.col("min_value")).round(2).alias("spread"))
        .filter(pl.col("n_molecules") >= min_molecules)
        .sort("spread", descending=True)
    )
