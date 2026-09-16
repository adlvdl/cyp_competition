"""Tests for uncertainty calibration diagnostics (notebook 06)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cyp.evaluation import ence, miscalibration_area, uncertainty_calibration_report


@pytest.fixture
def calibrated_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predictions whose error variance genuinely matches the claimed variance."""
    rng = np.random.default_rng(0)
    n = 5000
    y_var = rng.uniform(0.1, 2.0, n)
    y_true = rng.normal(0, 1, n)
    noise = rng.normal(0, np.sqrt(y_var))
    y_pred = y_true - noise
    return y_true, y_pred, y_var


def test_ence_near_zero_when_variance_matches_error(calibrated_case) -> None:
    y_true, y_pred, y_var = calibrated_case
    assert ence(y_true, y_pred, y_var) < 0.05


def test_ence_large_when_variance_is_constant_but_error_is_not(calibrated_case) -> None:
    y_true, y_pred, _ = calibrated_case
    bad_var = np.full(len(y_true), 0.01)
    assert ence(y_true, y_pred, bad_var) > 1.0


def test_miscalibration_area_near_zero_when_calibrated(calibrated_case) -> None:
    y_true, y_pred, y_var = calibrated_case
    assert miscalibration_area(y_true, y_pred, y_var) < 0.05


def test_miscalibration_area_large_when_underconfident(calibrated_case) -> None:
    y_true, y_pred, _ = calibrated_case
    bad_var = np.full(len(y_true), 0.01)
    assert miscalibration_area(y_true, y_pred, bad_var) > 0.2


def test_ence_handles_zero_variance_bin_without_dividing_by_zero() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    y_var = np.array([0.0, 0.0, 0.0, 0.0])
    # all-zero variance means every bin is skipped; result is nan rather than a crash
    assert np.isnan(ence(y_true, y_pred, y_var, n_bins=2))


def test_uncertainty_calibration_report_groups_by_method_and_endpoint(calibrated_case) -> None:
    y_true, y_pred, y_var = calibrated_case
    oof = pl.concat(
        [
            pl.DataFrame(
                {
                    "method": ["mve"] * len(y_true),
                    "endpoint": ["CYP3A4_pIC50_direct_inhibition"] * len(y_true),
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "y_unc": y_var,
                }
            ),
            pl.DataFrame(
                {
                    "method": ["ensemble"] * len(y_true),
                    "endpoint": ["CYP3A4_pIC50_direct_inhibition"] * len(y_true),
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "y_unc": np.full(len(y_true), 0.01),
                }
            ),
        ]
    )
    report = uncertainty_calibration_report(oof)
    assert report.height == 2
    assert set(report["method"]) == {"mve", "ensemble"}
    mve_row = report.filter(pl.col("method") == "mve").row(0, named=True)
    ensemble_row = report.filter(pl.col("method") == "ensemble").row(0, named=True)
    assert mve_row["ence"] < ensemble_row["ence"]


def test_unc_error_spearman_is_positive_when_uncertainty_tracks_error() -> None:
    rng = np.random.default_rng(1)
    n = 500
    y_true = rng.normal(0, 1, n)
    y_var = rng.uniform(0.1, 3.0, n)
    y_pred = y_true - rng.normal(0, np.sqrt(y_var))
    oof = pl.DataFrame(
        {
            "method": ["mve"] * n,
            "endpoint": ["CYP3A4_pIC50_direct_inhibition"] * n,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_unc": y_var,
        }
    )
    report = uncertainty_calibration_report(oof)
    assert report.row(0, named=True)["unc_error_spearman"] > 0.1
