"""Adapted from MolGpKa's `utils/ionization_group.py` (Xundrug/MolGpKa, MIT).

Finds candidate acidic and basic ionization sites by SMARTS match against
`smarts_pattern.tsv`, upstream's own curated table of 143 patterns. This part is
copied near-verbatim -- the only change is resolving the TSV path from this file's
own location rather than the process's current working directory, which was
upstream's issue #9 ("only works when run from ./MolGpKa/src").
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem

_SMARTS_FILE = Path(__file__).resolve().parent / "smarts_pattern.tsv"


def _split_acid_base_pattern() -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.read_csv(_SMARTS_FILE, sep="\t")
    table.columns = [c.strip() for c in table.columns]
    return table[table.Acid_or_base == "A"], table[table.Acid_or_base == "B"]


def _unique_matches(matches: list[list[int]]) -> list[list[int]]:
    """De-duplicate single-atom matches; keep multi-atom matches as-is.

    Upstream's own dedup logic -- a single-atom match found by more than one
    pattern should count once, but a two-atom match (index pair naming both the
    acidic proton and its parent, for the phosphonic/sulfonic patterns) should not
    be collapsed against a single-atom one for a different site.
    """
    singles = {m[0] for m in matches if len(m) == 1}
    doubles = [m for m in matches if len(m) == 2]
    return doubles + [[i] for i in singles]


def _match_acid(table: pd.DataFrame, mol: Chem.Mol) -> list[int]:
    matches: list[list[int]] = []
    for _, row in table.iterrows():
        pattern = Chem.MolFromSmarts(row.SMARTS.strip())
        hits = mol.GetSubstructMatches(pattern)
        if not hits:
            continue
        index = str(row.Index).strip()
        if "," in index:
            i, j = (int(x) for x in index.split(","))
            matches.extend([hit[i], hit[j]] for hit in hits)
        else:
            i = int(index)
            matches.extend([hit[i]] for hit in hits)
    return [atom for group in _unique_matches(matches) for atom in group]


def _match_base(table: pd.DataFrame, mol: Chem.Mol) -> list[int]:
    matches: list[list[int]] = []
    for _, row in table.iterrows():
        pattern = Chem.MolFromSmarts(row.SMARTS.strip())
        hits = mol.GetSubstructMatches(pattern)
        if not hits:
            continue
        for i in (int(x) for x in str(row.Index).strip().split(",")):
            matches.extend([hit[i]] for hit in hits)
    return [atom for group in _unique_matches(matches) for atom in group]


def ionization_sites(mol: Chem.Mol, acid_or_base: str) -> list[int]:
    """Atom indices of candidate acidic or basic ionization sites.

    Args:
        mol: An RDKit molecule with explicit hydrogens added (`Chem.AddHs`) --
            the SMARTS patterns match the ionizable hydrogen itself for the acid
            table, so a molecule without explicit Hs will find nothing there.
        acid_or_base: `"acid"` or `"base"`.
    """
    acid_table, base_table = _split_acid_base_pattern()
    if acid_or_base == "acid":
        return _match_acid(acid_table, mol)
    if acid_or_base == "base":
        return _match_base(base_table, mol)
    raise ValueError(f"acid_or_base must be 'acid' or 'base', got {acid_or_base!r}")
