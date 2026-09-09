"""Model comparison with honest uncertainty.

The central lesson from the PXR challenge: a paired bootstrap with multiple-testing
correction showed *sixteen of seventeen* submissions were statistically
indistinguishable on the blind set. The author picked a submission by sorting three
models on a CV column that spanned 0.0039 MAE -- noise -- and lost five leaderboard
places to what was effectively a coin flip.

So: never rank methods by a point estimate alone. Use `compare_methods` to get
per-fold spreads and `paired_bootstrap` to ask whether a difference is resolvable at
all. When it is not, choose on grounds other than the score -- prior evidence,
robustness, or the fact that calibration is cheap insurance.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import matthews_corrcoef, mean_absolute_error, r2_score

from .metrics import st_rae


def _safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rho, returning 0.0 (no correlation) on degenerate input rather than
    NaN -- matching how the challenge backend handles constant predictions.

    The mean baseline predicts a constant, which is a legitimate case here rather
    than a problem, so scipy's ConstantInputWarning is suppressed instead of printed.
    """
    if len(y_true) < 2 or np.std(y_pred) == 0 or np.std(y_true) == 0:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, _ = spearmanr(y_true, y_pred)
    return 0.0 if np.isnan(rho) else float(rho)


def fold_metrics(
    oof: pl.DataFrame,
    pred_col: str = "y_pred",
    group_cols: tuple[str, ...] = ("method", "endpoint", "fold"),
) -> pl.DataFrame:
    """Per-fold regression metrics for every method/endpoint.

    Includes ST-RAE (the challenge's primary metric) when credible-interval columns
    `y_lower`/`y_upper` are present in the frame; otherwise ST-RAE degenerates to
    plain RAE, which is not what the leaderboard computes -- so carry those columns
    through your OOF frame if you want a comparable number.
    """
    rows: list[dict] = []
    for keys, group in oof.group_by(list(group_cols)):
        y_true = group["y_true"].to_numpy()
        y_pred = group[pred_col].to_numpy()
        record = dict(zip(group_cols, keys, strict=True))
        lower = group["y_lower"].to_numpy() if "y_lower" in group.columns else None
        upper = group["y_upper"].to_numpy() if "y_upper" in group.columns else None
        record |= {
            "n": len(y_true),
            "st_rae": st_rae(y_true, y_pred, lower, upper),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
            "rho": _safe_spearman(y_true, y_pred),
        }
        rows.append(record)
    return pl.DataFrame(rows)


def compare_methods(
    oof: pl.DataFrame, metric: str = "st_rae", pred_col: str = "y_pred"
) -> pl.DataFrame:
    """Mean +/- std of a metric across folds, per method and endpoint.

    The `std` column is the point of this table. If two methods' means differ by less
    than the fold-to-fold spread, treat them as tied and confirm with
    `paired_bootstrap` before acting on the ordering.
    """
    per_fold = fold_metrics(oof, pred_col=pred_col)
    return (
        per_fold.group_by(["method", "endpoint"])
        .agg(
            pl.col(metric).mean().alias(f"{metric}_mean"),
            pl.col(metric).std().alias(f"{metric}_std"),
            pl.len().alias("n_folds"),
        )
        .sort(f"{metric}_mean")
    )


def macro_averaged_fold_metrics(
    fold_scores_by_endpoint: dict[str, pl.DataFrame],
    metric_col: str = "st_rae",
    group_cols: tuple[str, ...] = ("method", "fold"),
) -> pl.DataFrame:
    """Macro-average a metric across endpoints, per fold -- the CV analogue of the
    leaderboard's "MA" pseudo-endpoint.

    This is what the rank in the competition is actually based on: the vendored
    backend (`vendor/cyp_challenge_tutorial/evaluation/evaluate_predictions.py`,
    `compute_macro_bootstrap_results`) scores every endpoint on the same bootstrap
    resample of the test set, then takes a **plain arithmetic mean of the metric
    across endpoints** for that resample -- not a per-endpoint score, and not a
    metric recomputed on pooled predictions. Everything in this repo's MCS
    heatmaps and CV summaries up to now was per-endpoint, which answers "which
    model is best at CYP3A4?" but not "which model would rank best on the
    leaderboard?" -- this function answers the second question.

    Caveat versus the real leaderboard: the backend's bootstrap resamples the same
    750 test compounds for every endpoint at once, so each resample is one coherent
    unit repeated across endpoints. Our CV folds are not that: each endpoint has its
    own labelled subset (see CLAUDE.md -- the four pIC50 endpoints share no fixed
    row count) and its own independent scaffold-grouped fold assignment, so "fold 3
    of CYP1A2" and "fold 3 of CYP2D6" are different compounds, not the same
    resample. This function still averages by matching fold *index* across
    endpoints (fold 0 with fold 0, fold 1 with fold 1, ...), which is the closest
    analogue available under nested 5x5 CV and is what makes the result usable as a
    repeated-measures unit for Tukey HSD -- but treat it as an approximation of the
    leaderboard macro-average, not a reproduction of it.

    Args:
        fold_scores_by_endpoint: Maps endpoint name to that endpoint's per-fold
            metric frame (e.g. one entry per `evaluation.fold_metrics` call). Every
            frame must share the same `fold` numbering and the same set of methods.
        metric_col: The metric to average (e.g. "st_rae" for regression, "mcc" for
            TDI) -- always the challenge's own metric, matching
            `ACTIVITY_METRICS`/`CLASSIFICATION_METRICS` in the vendored config.
        group_cols: Columns identifying one repeated-measures unit; defaults to
            (method, fold) so the result has exactly one row per method per fold,
            ready for `mcs.rm_tukey_hsd`.

    Returns:
        One row per (method, fold), with `metric_col` set to the mean of that
        metric across every endpoint's value at that (method, fold).
    """
    tagged = [
        frame.select(*group_cols, metric_col).with_columns(pl.lit(ep).alias("_endpoint"))
        for ep, frame in fold_scores_by_endpoint.items()
    ]
    combined = pl.concat(tagged)

    coverage = combined.group_by(list(group_cols)).agg(pl.len().alias("_n_endpoints"))
    n_endpoints = len(fold_scores_by_endpoint)
    incomplete = coverage.filter(pl.col("_n_endpoints") != n_endpoints)
    if incomplete.height > 0:
        raise ValueError(
            "Every (method, fold) must be scored on every endpoint to macro-average "
            f"correctly; {incomplete.height} group(s) are missing an endpoint -- "
            "check that every endpoint's CV used the same fold numbering and methods."
        )

    return (
        combined.group_by(list(group_cols))
        .agg(pl.col(metric_col).mean().alias(metric_col))
        .sort(list(group_cols))
    )


def fold_metrics_classification(
    oof: pl.DataFrame,
    pred_col: str = "y_pred",
    group_cols: tuple[str, ...] = ("method", "endpoint", "fold"),
) -> pl.DataFrame:
    """Per-fold MCC for TDI classifiers -- the classification analogue of
    `fold_metrics`. MCC (not accuracy) is the challenge metric because TDI labels
    are imbalanced (~21-22% positive); see `models.MajorityBaseline`."""
    rows: list[dict] = []
    for keys, group in oof.group_by(list(group_cols)):
        y_true = group["y_true"].to_numpy().astype(bool)
        y_pred = group[pred_col].to_numpy().astype(bool)
        record = dict(zip(group_cols, keys, strict=True))
        record |= {
            "n": len(y_true),
            "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else 0.0,
            "positive_rate_true": float(y_true.mean()),
            "positive_rate_pred": float(y_pred.mean()),
        }
        rows.append(record)
    return pl.DataFrame(rows)


def compare_methods_classification(
    oof: pl.DataFrame, pred_col: str = "y_pred"
) -> pl.DataFrame:
    """Mean +/- std MCC across folds, per method and endpoint -- sorted best
    (highest MCC) first, unlike `compare_methods` where lower ST-RAE is better."""
    per_fold = fold_metrics_classification(oof, pred_col=pred_col)
    return (
        per_fold.group_by(["method", "endpoint"])
        .agg(
            pl.col("mcc").mean().alias("mcc_mean"),
            pl.col("mcc").std().alias("mcc_std"),
            pl.len().alias("n_folds"),
        )
        .sort("mcc_mean", descending=True)
    )


def paired_bootstrap(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric: str = "mae",
    n_resamples: int = 10_000,
    seed: int = 42,
    y_lower: np.ndarray | None = None,
    y_upper: np.ndarray | None = None,
) -> dict[str, float]:
    """Is method A actually better than method B on these compounds?

    Resamples compounds with replacement, scoring both methods on the same resample
    each time (paired -- this is what gives the test its power). Returns the observed
    difference, a 95% interval on it, and a two-sided p-value.

    A p-value above ~0.05 means the data cannot resolve the two methods. In PXR that
    was true for nearly every pair, including the submitted model versus the best one
    (p = 0.262).

    Returns:
        `diff` is A minus B. For error metrics (mae, st_rae) negative favours A; for
        "goodness" metrics (rho, mcc) positive favours A.
    """
    is_classification = metric == "mcc"
    dtype = bool if is_classification else float
    y_true = np.asarray(y_true, dtype=dtype)
    a, b = np.asarray(y_pred_a, dtype=dtype), np.asarray(y_pred_b, dtype=dtype)

    def score(idx: np.ndarray, pred: np.ndarray) -> float:
        if metric == "mae":
            return float(mean_absolute_error(y_true[idx], pred[idx]))
        if metric == "st_rae":
            return st_rae(
                y_true[idx],
                pred[idx],
                None if y_lower is None else np.asarray(y_lower)[idx],
                None if y_upper is None else np.asarray(y_upper)[idx],
            )
        if metric == "rho":
            return _safe_spearman(y_true[idx], pred[idx])
        if metric == "mcc":
            sub_true = y_true[idx]
            return float(matthews_corrcoef(sub_true, pred[idx])) if len(set(sub_true)) > 1 else 0.0
        raise ValueError(f"Unsupported metric: {metric!r}")

    full = np.arange(len(y_true))
    observed = score(full, a) - score(full, b)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, len(y_true), len(y_true))
        diffs[i] = score(idx, a) - score(idx, b)

    # Two-sided p: how often the resampled difference crosses zero relative to the
    # observed direction.
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "diff": float(observed),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_value": float(min(p, 1.0)),
        "n": len(y_true),
    }


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> pl.DataFrame:
    """Holm-Bonferroni correction over a family of comparisons.

    Comparing many model variants against each other inflates false positives; PXR
    used this correction to establish that almost none of its seventeen submissions
    were distinguishable. Returns comparisons sorted by p with a `significant` flag.
    """
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    rows, still_significant = [], True
    for rank, (name, p) in enumerate(items):
        threshold = alpha / (m - rank)
        if p > threshold:
            still_significant = False
        rows.append(
            {
                "comparison": name,
                "p_value": p,
                "threshold": threshold,
                "significant": still_significant,
            }
        )
    return pl.DataFrame(rows)


def bias_by_potency_bin(
    oof: pl.DataFrame, pred_col: str = "y_pred", n_bins: int = 5
) -> pl.DataFrame:
    """Mean signed error per potency bin, per method.

    Regression to the mean was the dominant PXR failure: every model underpredicted
    potent compounds by up to a full log unit, and no amount of tuning fixed it. This
    table is how you see it. Under ST-RAE the low-potency bins are largely forgiven
    (wide credible intervals, plus explicit downweighting below pIC50 4), so
    shrinkage in the *top* bin is the error that actually costs.
    """
    return (
        oof.with_columns(
            (pl.col(pred_col) - pl.col("y_true")).alias("signed_error"),
            pl.col("y_true")
            .qcut(n_bins, labels=[f"bin{i}" for i in range(n_bins)])
            .alias("potency_bin"),
        )
        .group_by(["method", "endpoint", "potency_bin"])
        .agg(
            pl.col("signed_error").mean().alias("mean_bias"),
            pl.col("signed_error").abs().mean().alias("mae"),
            pl.col("y_true").mean().alias("mean_true"),
            pl.len().alias("n"),
        )
        .sort(["method", "endpoint", "potency_bin"])
    )
