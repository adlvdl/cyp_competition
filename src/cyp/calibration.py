"""Leakage-free cross-fit calibration.

Ported from the PXR repo (`marimo_notebooks/6_ml_optimization_3.py`).

Worth knowing before you decide whether to bother: in PXR, linear calibration
improved cross-validated MAE by 0.0001 and was dismissed as noise -- then turned out
to be the *best* variant on the Phase-2 blind set (0.4448 vs 0.4573 for the submitted
uncalibrated model). CV could not resolve the difference, but it was real. Two
lessons carried into this repo:

1. Calibration is cheap insurance against regression-to-the-mean, which was the
   dominant PXR failure mode and is likely worse here (ST-RAE forgives errors inside
   the credible interval, so systematic shrinkage of potent compounds is exactly the
   error that still costs).
2. Do not drop an intervention because CV says it is a rounding error. CV on ~1-2k
   compounds cannot resolve differences of that size.

Calibrators are always fit cross-fit -- on folds disjoint from the one being
transformed -- so calibrated OOF predictions stay honest for model comparison.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

CalibrationKind = str  # "raw" | "linear" | "isotonic"


def _fit_apply(
    kind: CalibrationKind, x_fit: np.ndarray, y_fit: np.ndarray, x_test: np.ndarray
) -> np.ndarray:
    if kind == "raw":
        return x_test
    if kind == "linear":
        model = LinearRegression().fit(x_fit.reshape(-1, 1), y_fit)
        return model.predict(x_test.reshape(-1, 1))
    if kind == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip").fit(x_fit, y_fit)
        return model.predict(x_test)
    raise ValueError(f"Unknown calibration kind: {kind!r}")


def crossfit_calibrate(
    oof: pl.DataFrame,
    kind: CalibrationKind,
    n_folds: int = 25,
    n_inner: int = 5,
) -> pl.DataFrame:
    """Add a `y_cal` column to one method+endpoint's OOF predictions.

    For each fold the calibrator is fit on the *other* inner folds of the same outer
    repeat -- disjoint compounds -- and applied to the held-out fold. Fitting on the
    same fold being transformed would leak and make calibration look far better than
    it is.

    Args:
        oof: Long OOF frame for a single method and endpoint (see `cv.oof_frame`).
        kind: "raw", "linear" or "isotonic".
        n_folds: Total folds (outer x inner).
        n_inner: Inner folds per outer repeat.
    """
    parts: list[pl.DataFrame] = []
    for fold in range(n_folds):
        outer = fold // n_inner
        fit = oof.filter((pl.col("outer_fold") == outer) & (pl.col("fold") != fold))
        test = oof.filter(pl.col("fold") == fold)
        if test.height == 0:
            continue
        if fit.height == 0:
            # Nothing disjoint to fit on: pass predictions through untouched rather
            # than silently fitting on the test fold.
            parts.append(test.with_columns(pl.col("y_pred").alias("y_cal")))
            continue
        y_cal = _fit_apply(
            kind,
            fit["y_pred"].to_numpy(),
            fit["y_true"].to_numpy(),
            test["y_pred"].to_numpy(),
        )
        parts.append(test.with_columns(pl.Series("y_cal", np.asarray(y_cal))))
    return pl.concat(parts) if parts else oof.with_columns(pl.lit(None).alias("y_cal"))


def calibrate_all(
    oof: pl.DataFrame, kinds: tuple[str, ...] = ("raw", "linear", "isotonic")
) -> pl.DataFrame:
    """Run every calibration variant across every (method, endpoint) group.

    Returns the OOF rows repeated once per variant, with `calibration` and `y_cal`
    columns added -- ready for `evaluation.compare_methods`.
    """
    out: list[pl.DataFrame] = []
    for kind in kinds:
        for (method, endpoint), group in oof.group_by(["method", "endpoint"]):
            calibrated = crossfit_calibrate(group.sort("fold"), kind)
            out.append(
                calibrated.with_columns(
                    pl.lit(kind).alias("calibration"),
                    pl.lit(method).alias("method"),
                    pl.lit(endpoint).alias("endpoint"),
                )
            )
    return pl.concat(out)


def fit_final_calibrator(y_true: np.ndarray, y_pred: np.ndarray, kind: CalibrationKind):
    """Fit a calibrator on *all* OOF predictions, for use on the blind test set.

    Cross-fitting is for honest evaluation; once a variant is chosen, the calibrator
    applied to the test set should use every training compound available. Returns an
    object with `.predict`, or None for "raw".
    """
    if kind == "raw":
        return None
    if kind == "linear":
        return LinearRegression().fit(np.asarray(y_pred).reshape(-1, 1), y_true)
    if kind == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(y_pred, y_true)
    raise ValueError(f"Unknown calibration kind: {kind!r}")


def apply_calibrator(calibrator, y_pred: np.ndarray) -> np.ndarray:
    """Apply a calibrator from `fit_final_calibrator` (None passes through)."""
    y_pred = np.asarray(y_pred, dtype=float)
    if calibrator is None:
        return y_pred
    if isinstance(calibrator, LinearRegression):
        return calibrator.predict(y_pred.reshape(-1, 1))
    return calibrator.predict(y_pred)
