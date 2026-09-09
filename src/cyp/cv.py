"""Nested cross-validation splits.

Ported from the PXR challenge repo (`marimo_notebooks/2_ml_baseline.py`), adapted to
Polars and to CYP's four sparse endpoints. The 5x5 nested protocol is what made the
PXR model comparisons trustworthy: 25 folds give enough repeats for a paired test
across methods, and the outer/inner structure is what leakage-free cross-fit
calibration needs (see `calibration.py`).

Split strategies, in order of how much you should trust them here:

- `scaffold_splits` -- default. The CYP test set is 75 hits x ~10 analogs, so
  scaffold-grouped folds are the closest analogue of the real evaluation.
- `random_splits` -- optimistic; near-duplicate analogs land on both sides. Kept
  because PXR found random/scaffold/temporal rankings nearly identical, so it is a
  useful cheap cross-check, not a headline number.

Every generator yields ``(fold, outer, inner, train, val, test)`` where `val` is None
unless `p_val > 0`. Keep that signature when adding a strategy -- the training loops
in `models.py` unpack it positionally.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

N_OUTER = 5
N_INNER = 5
SEED = 42


def split_random(
    df: pl.DataFrame, p_test: float = 0.2, seed: int = SEED
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Randomly split into (train, test)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(df.height)
    n_test = int(len(idx) * p_test)
    return df[idx[n_test:]].clone(), df[idx[:n_test]].clone()


def _group_kfold_shuffle(
    groups: np.ndarray, n_splits: int, seed: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """K-fold that keeps each group intact, with shuffling.

    sklearn's GroupKFold does not shuffle; this is the PXR `GroupKFoldShuffle`
    reduced to a generator. Groups are permuted then split into contiguous chunks.
    """
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    unique = rng.permutation(unique)
    for test_groups in np.array_split(unique, n_splits):
        test_mask = np.isin(groups, test_groups)
        yield np.where(~test_mask)[0], np.where(test_mask)[0]


def _nested(
    df: pl.DataFrame,
    groups: np.ndarray,
    n_outer: int,
    n_inner: int,
    seed: int,
    p_val: float,
) -> Iterator[tuple[int, int, int, pl.DataFrame, pl.DataFrame | None, pl.DataFrame]]:
    for i in range(n_outer):
        for j, (train_idx, test_idx) in enumerate(
            _group_kfold_shuffle(groups, n_inner, seed + i)
        ):
            fold = i * n_inner + j
            train, test = df[train_idx].clone(), df[test_idx].clone()
            val = None
            if p_val > 0:
                train, val = split_random(train, p_test=p_val, seed=seed + fold)
            yield fold, i, j, train, val, test


def random_splits(
    df: pl.DataFrame,
    n_outer: int = N_OUTER,
    n_inner: int = N_INNER,
    seed: int = SEED,
    p_val: float = 0.0,
) -> Iterator[tuple[int, int, int, pl.DataFrame, pl.DataFrame | None, pl.DataFrame]]:
    """Nested CV with each molecule as its own group (i.e. plain random folds)."""
    yield from _nested(df, np.arange(df.height), n_outer, n_inner, seed, p_val)


def murcko_scaffold(smiles: str) -> str:
    """Canonical Bemis-Murcko scaffold SMILES. Unparseable input and acyclic
    molecules both return "", pooling them into one "no scaffold" group."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol), canonical=True)


def scaffold_splits(
    df: pl.DataFrame,
    smiles_col: str = "SMILES",
    n_outer: int = N_OUTER,
    n_inner: int = N_INNER,
    seed: int = SEED,
    p_val: float = 0.0,
) -> Iterator[tuple[int, int, int, pl.DataFrame, pl.DataFrame | None, pl.DataFrame]]:
    """Nested CV grouped by Murcko scaffold -- no scaffold spans train and test."""
    scaffolds = np.array([murcko_scaffold(s) for s in df[smiles_col].to_list()])
    yield from _nested(df, scaffolds, n_outer, n_inner, seed, p_val)


def oof_frame(records: list[dict]) -> pl.DataFrame:
    """Assemble out-of-fold predictions into the long frame the rest of the pipeline
    expects: one row per (method, endpoint, fold, compound).

    Columns: method, endpoint, fold, outer_fold, inner_fold, Molecule_Name,
    y_true, y_pred. `calibration.crossfit_calibrate` and `evaluation.compare_methods`
    both read this schema, so build it with this helper rather than by hand.
    """
    frame = pl.DataFrame(records)
    required = {
        "method",
        "endpoint",
        "fold",
        "outer_fold",
        "Molecule_Name",
        "y_true",
        "y_pred",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"OOF frame missing columns: {sorted(missing)}")
    return frame
