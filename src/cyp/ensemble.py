"""Weighted ensembling and weight search.

Ported from the PXR repo's notebooks 3 and 4, where ensembling cut test MAE from
0.574 (best single model) to 0.507.

What PXR learned about *why* it helped, which shapes how to use it here: the gain
came from suppressing catastrophic predictions (errors > 1 log unit) rather than
from improving typical accuracy. That matters under ST-RAE, which is a ratio of
summed errors -- a handful of large misses dominates the numerator, so trimming the
tail is worth more than shaving the median.

The weight sweep also zeroed out whole models (Random Forest earned weight 0), so
expect a sparse solution and do not assume every candidate belongs in the blend.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import polars as pl

from .metrics import st_rae


def weighted_average(
    predictions: dict[str, np.ndarray], weights: dict[str, float]
) -> np.ndarray:
    """Normalized weighted average of per-model predictions.

    Weights need not sum to 1; they are normalized here. Models with weight 0 are
    dropped, and a model present in `weights` but missing from `predictions` raises
    rather than being silently skipped.
    """
    missing = {m for m, w in weights.items() if w != 0} - set(predictions)
    if missing:
        raise KeyError(f"No predictions for weighted models: {sorted(missing)}")

    total = sum(w for w in weights.values() if w != 0)
    if total <= 0:
        raise ValueError("Weights must include at least one positive entry")

    stacked = None
    for model, weight in weights.items():
        if weight == 0:
            continue
        contribution = (weight / total) * np.asarray(predictions[model], dtype=float)
        stacked = contribution if stacked is None else stacked + contribution
    return stacked


def oof_to_wide(oof: pl.DataFrame, endpoint: str, pred_col: str = "y_pred"):
    """Pivot long OOF rows into (per-model prediction matrix, y_true, ids) for one
    endpoint. Every model must cover the same compounds, which holds when the folds
    came from the same `cv` generator."""
    subset = oof.filter(pl.col("endpoint") == endpoint)
    wide = subset.pivot(
        values=pred_col, index=["Molecule_Name", "y_true"], on="method"
    ).sort("Molecule_Name")
    methods = [c for c in wide.columns if c not in ("Molecule_Name", "y_true")]
    preds = {m: wide[m].to_numpy() for m in methods}
    return preds, wide["y_true"].to_numpy(), wide["Molecule_Name"].to_list()


def sweep_weights(
    predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    grid: tuple[float, ...] = (0.0, 1 / 3, 1.0, 5.0),
    y_lower: np.ndarray | None = None,
    y_upper: np.ndarray | None = None,
    metric: str = "st_rae",
) -> pl.DataFrame:
    """Exhaustive weight search over `grid` for every model.

    PXR used exactly this coarse grid ({0, 1/3, 1, 5}), which is deliberate: a fine
    grid overfits the CV set, and the resulting differences are far below what the
    data can resolve. Keep it coarse.

    Cost is len(grid) ** n_models, so 4 values x 5 models = 1024 combinations. Beyond
    ~6 models, sample the grid instead.

    Returns every combination scored, best first.
    """
    models = sorted(predictions)
    rows = []
    for combo in product(grid, repeat=len(models)):
        if sum(combo) <= 0:
            continue
        weights = dict(zip(models, combo, strict=True))
        blended = weighted_average(predictions, weights)
        if metric == "st_rae":
            score = st_rae(y_true, blended, y_lower, y_upper)
        elif metric == "mae":
            score = float(np.mean(np.abs(y_true - blended)))
        else:
            raise ValueError(f"Unsupported metric: {metric!r}")
        rows.append(
            {"score": score, **{f"w_{m}": w for m, w in weights.items()}}
        )
    return pl.DataFrame(rows).sort("score")


def describe_weights(weights: dict[str, float]) -> str:
    """Compact label for a weight set, e.g. `lgbm5-xgb1-rf0`, for run naming."""
    return "-".join(
        f"{m}{w:g}" for m, w in sorted(weights.items()) if w != 0
    ) or "empty"


def catastrophic_rate(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 1.0
) -> float:
    """Fraction of predictions off by more than `threshold` log units.

    This is the quantity ensembling actually improved in PXR, so track it alongside
    ST-RAE when deciding whether a blend is worth its complexity.
    """
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) > threshold))
