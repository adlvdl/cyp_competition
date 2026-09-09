"""Local scoring that mirrors the official leaderboard.

The scoring functions themselves are imported from the vendored tutorial code
(`vendor/cyp_challenge_tutorial/`), which is a port of the challenge backend -- this
module only wires the right columns into them and macro-averages. Never reimplement
ST-RAE here; if the leaderboard changes, re-vendor instead.

Caveat: the official leaderboard bootstraps and reports confidence intervals. These
helpers score a single point estimate, so treat small differences as noise -- see
`evaluation.paired_bootstrap` for the honest comparison.
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl
from sklearn.metrics import matthews_corrcoef, mean_absolute_error, r2_score

from . import constants as C

sys.path.insert(0, str(C.PROJECT_ROOT / "vendor" / "cyp_challenge_tutorial"))
from evaluation.custom_scoring_functions import (  # noqa: E402
    rae_soft_threshold_absolute_error,
)


def st_rae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: np.ndarray | None = None,
    y_upper: np.ndarray | None = None,
) -> float:
    """Soft-Threshold RAE: the challenge's primary regression metric.

    Lower is better; 1.0 means "no better than predicting the mean". Without bounds
    this degrades to plain RAE, which is *not* the leaderboard metric -- pass the
    credible intervals whenever you have them.
    """
    return float(
        rae_soft_threshold_absolute_error(
            np.asarray(y_true, dtype=float),
            np.asarray(y_pred, dtype=float),
            y_true_upper=None if y_upper is None else np.asarray(y_upper, dtype=float),
            y_true_lower=None if y_lower is None else np.asarray(y_lower, dtype=float),
        )
    )


def score_regression_endpoint(
    truth: pl.DataFrame, preds: pl.DataFrame, endpoint: str
) -> dict[str, float]:
    """Score one isoform. `truth` must carry the endpoint plus its `_conf_low`/
    `_conf_high` columns; rows with a missing label are dropped, as on the leaderboard."""
    merged = truth.join(
        preds.select("Molecule_Name", pl.col(endpoint).alias("_pred")),
        on="Molecule_Name",
        how="inner",
    ).filter(pl.col(endpoint).is_not_null() & pl.col("_pred").is_not_null())

    if merged.height == 0:
        return {"n": 0, "st_rae": float("nan"), "mae": float("nan"), "r2": float("nan")}

    y = merged[endpoint].to_numpy()
    p = merged["_pred"].to_numpy()
    lo_col, hi_col = endpoint + C.CONF_LOW_SUFFIX, endpoint + C.CONF_HIGH_SUFFIX
    lo = merged[lo_col].to_numpy() if lo_col in merged.columns else None
    hi = merged[hi_col].to_numpy() if hi_col in merged.columns else None

    return {
        "n": merged.height,
        "st_rae": st_rae(y, p, lo, hi),
        "mae": float(mean_absolute_error(y, p)),
        "r2": float(r2_score(y, p)),
        "rho": float(np.corrcoef(np.argsort(np.argsort(y)), np.argsort(np.argsort(p)))[0, 1]),
    }


def score_activity(truth: pl.DataFrame, preds: pl.DataFrame) -> pl.DataFrame:
    """Per-isoform scores plus the macro-averaged row (`MA`), the headline number
    for the direct-inhibition track."""
    rows = []
    for endpoint in C.REGRESSION_ENDPOINTS:
        rows.append({"endpoint": endpoint} | score_regression_endpoint(truth, preds, endpoint))
    table = pl.DataFrame(rows)
    macro = {"endpoint": "MA", "n": int(table["n"].sum())}
    for col in ("st_rae", "mae", "r2", "rho"):
        if col in table.columns:
            macro[col] = float(table[col].mean())
    return pl.concat([table, pl.DataFrame([macro])], how="diagonal")


def score_tdi(truth: pl.DataFrame, preds: pl.DataFrame) -> pl.DataFrame:
    """MCC per TDI isoform plus their macro-average. MCC is the primary TDI metric."""
    rows = []
    for cyp in C.TDI_ISOFORMS:
        col = f"{cyp}_is_TDI"
        merged = truth.join(
            preds.select("Molecule_Name", pl.col(col).alias("_pred")),
            on="Molecule_Name",
            how="inner",
        ).filter(pl.col(col).is_not_null() & pl.col("_pred").is_not_null())

        if merged.height == 0:
            rows.append({"endpoint": col, "n": 0, "mcc": float("nan")})
            continue
        y = merged[col].cast(pl.Boolean).to_numpy()
        p = merged["_pred"].cast(pl.Boolean).to_numpy()
        rows.append(
            {
                "endpoint": col,
                "n": merged.height,
                "mcc": float(matthews_corrcoef(y, p)),
                "positive_rate_true": float(y.mean()),
                "positive_rate_pred": float(p.mean()),
            }
        )
    table = pl.DataFrame(rows)
    macro = {"endpoint": "MA", "n": int(table["n"].sum()), "mcc": float(table["mcc"].mean())}
    return pl.concat([table, pl.DataFrame([macro])], how="diagonal")
