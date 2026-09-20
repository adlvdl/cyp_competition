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

from collections import Counter
from collections.abc import Callable
from itertools import product

import numpy as np
import polars as pl

from .metrics import st_rae


def weighted_average(predictions: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
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
    wide = subset.pivot(values=pred_col, index=["Molecule_Name", "y_true"], on="method").sort(
        "Molecule_Name"
    )
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
        rows.append({"score": score, **{f"w_{m}": w for m, w in weights.items()}})
    return pl.DataFrame(rows).sort("score")


def describe_weights(weights: dict[str, float]) -> str:
    """Compact label for a weight set, e.g. `lgbm5-xgb1-rf0`, for run naming."""
    return "-".join(f"{m}{w:g}" for m, w in sorted(weights.items()) if w != 0) or "empty"


def catastrophic_rate(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 1.0) -> float:
    """Fraction of predictions off by more than `threshold` log units.

    This is the quantity ensembling actually improved in PXR, so track it alongside
    ST-RAE when deciding whether a blend is worth its complexity.
    """
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) > threshold))


def caruana_selection(
    predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    score: Callable[[np.ndarray, np.ndarray], float],
    *,
    maximize: bool = True,
    n_bags: int = 20,
    bag_frac: float = 0.5,
    max_iter: int = 50,
    seed: int = 42,
) -> dict[str, float]:
    """Caruana bagged ensemble selection (Caruana et al., ICML 2004).

    Forward stepwise selection **with replacement**: each round adds whichever
    candidate most improves the running average, and a strong candidate may be picked
    repeatedly, which is how it earns more weight without a continuous optimizer.
    Repeated over random subsets of the candidate library (bagging), with the final
    weights proportional to how often each candidate was selected across bags.

    Preferred over `sweep_weights` when the metric is not a mean of per-row errors.
    **MCC is the case that forces it here**: it is computed from a confusion matrix
    over the whole set, so it cannot be decomposed per row, and a weight grid scored
    on it has no structure for a coarse sweep to exploit. Caruana only ever asks
    "does adding this candidate improve the score", which any metric can answer.

    Also preferred when candidates are correlated. A continuous optimizer (NNLS and
    friends) can reallocate weight destructively between two near-duplicate models,
    fitting noise in their difference; selection with replacement cannot, since it
    only ever adds whole candidates.

    Args:
        predictions: Candidate name -> prediction vector, all the same length. For
            classification these are probabilities, not labels.
        y_true: Targets, aligned with every prediction vector.
        score: `(y_true, y_pred) -> float`. Takes the *blended* prediction, so a
            classification caller thresholds inside this function rather than passing
            labels in (see notebook 13 for the MCC-at-fixed-threshold pattern).
        maximize: True for a goodness metric (MCC, rho), False for an error metric.
        n_bags: Bagging rounds. Weights are the selection counts across all bags.
        bag_frac: Fraction of candidates visible in each bag.
        max_iter: Selection rounds per bag.
        seed: RNG seed for the bag subsets.

    Returns:
        Candidate name -> weight, summing to 1. Candidates never selected get 0.0,
        so read the result as a sparse solution rather than assuming every model is
        in the blend.
    """
    names = sorted(predictions)
    if not names:
        raise ValueError("no candidates to select from")
    matrix = np.vstack([np.asarray(predictions[n], dtype=float) for n in names]).T
    rng = np.random.default_rng(seed)
    bag_size = max(1, int(round(bag_frac * len(names))))
    counts = Counter()

    for _ in range(n_bags):
        subset = rng.choice(len(names), size=min(bag_size, len(names)), replace=False)
        # Start from the single best candidate in this bag, as the paper does --
        # starting from the full average makes the first step much harder to improve.
        individual = [score(y_true, matrix[:, i]) for i in subset]
        best_local = int(np.argmax(individual) if maximize else np.argmin(individual))
        selected = [subset[best_local]]
        running = matrix[:, selected[0]].copy()
        current = score(y_true, running)

        for _ in range(max_iter):
            best_idx, best_score = None, current
            for i in subset:
                trial = (running * len(selected) + matrix[:, i]) / (len(selected) + 1)
                s = score(y_true, trial)
                if (maximize and s > best_score) or (not maximize and s < best_score):
                    best_score, best_idx = s, i
            if best_idx is None:
                break
            selected.append(best_idx)
            running = (running * (len(selected) - 1) + matrix[:, best_idx]) / len(selected)
            current = best_score

        for i in selected:
            counts[names[i]] += 1

    total = sum(counts.values())
    return {n: counts.get(n, 0) / total for n in names}
