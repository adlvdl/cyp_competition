"""Lookup table and launcher shim for training chemprop against the ST-RAE interval.

The loss itself lives in `interval_launcher.py`, which runs inside the chemprop
subprocess. This module is the parent-process side: it builds the table of credible
band half-widths as a function of pIC50, and writes the executable shim that
`graph_models.CHEMPROP_BIN` is pointed at.

Idea credited in README.md ("Ideas from other entrants"): train against the credible
interval rather than the point estimate, as in the mokhalaf22 and doctawho42 reports.
"""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

LAUNCHER = Path(__file__).with_name("interval_launcher.py")

#: pIC50 grid the band offsets are tabulated on. Covers the observed label range
#: (about 1.9 to 7.9) with margin; lookups outside it clamp to the end values.
GRID = np.linspace(1.0, 9.0, 161)


def band_table(frames: dict[str, pl.DataFrame]) -> dict[str, np.ndarray]:
    """Band half-widths below and above the label, as isotonic functions of the label.

    Width falls as potency rises (tight intervals for actives, wide for inactives), so
    both offsets are fitted non-increasing in pIC50. Returns arrays ready for
    `write_band_table`, plus per-endpoint fit quality for the notebook to report.
    """
    targets = list(frames)
    lo = np.zeros((len(targets), len(GRID)))
    hi = np.zeros_like(lo)
    for k, endpoint in enumerate(targets):
        f = frames[endpoint]
        y = f["y_true"].to_numpy()
        for out, off in ((lo, y - f["y_lower"].to_numpy()), (hi, f["y_upper"].to_numpy() - y)):
            iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(y, off)
            out[k] = iso.predict(GRID)
    return {"targets": np.array(targets), "grid": GRID, "lo_off": lo, "hi_off": hi}


def band_fit_quality(frames: dict[str, pl.DataFrame], table: dict[str, np.ndarray]) -> pl.DataFrame:
    """How well the label-only band reproduces the real one (R^2 on each half-width)."""
    rows = []
    for k, endpoint in enumerate(table["targets"]):
        f = frames[str(endpoint)]
        y = f["y_true"].to_numpy()
        row = {"endpoint": str(endpoint), "n": f.height}
        for side, true in (
            ("lo", y - f["y_lower"].to_numpy()),
            ("hi", f["y_upper"].to_numpy() - y),
        ):
            fit = np.interp(y, table["grid"], table[f"{side}_off"][k])
            row[f"r2_{side}"] = float(
                1.0 - ((true - fit) ** 2).sum() / ((true - true.mean()) ** 2).sum()
            )
        rows.append(row)
    return pl.DataFrame(rows)


def write_band_table(table: dict[str, np.ndarray], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **table)
    return path


def write_shim(path: Path) -> Path:
    """Executable that stands in for the `chemprop` binary but registers the new losses.

    It uses the interpreter running this process, so the chemprop subprocess sees the
    same environment the normal `chemprop` console script would.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{LAUNCHER}" "$@"\n')
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path
