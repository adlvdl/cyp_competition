"""31b: does the interval loss survive blending? (the gap `30` left open)

    uv run --no-sync python notebooks/31b_interval_blend_check.py

Trains nothing. Takes the shipped blend (`22`'s weights, which are equal splits) and swaps
members for their interval-loss OOF from `30`, scoring macro ST-RAE per fold:

  A  pubchem -> interval pubchem                 all 25 folds
  B  A + pooled -> interval pooled               folds 0-4 only (all `30` has), CYP2D6 keeps
                                                 the MAE pooled member (interval was worse there)

Baseline is the blend as shipped (05's `pubchem`, 20's MAE pooled). The weights were fitted on
the baseline members' OOF, so they favour the baseline slightly: a fair-to-conservative test.
`pubchem` seeds differ between 05 and 30, so a small part of A is seed noise (30's `mse` arm
differs from 05's `pubchem` by 0.007 macro).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cyp  # noqa: E402,F401
from cyp import constants as C  # noqa: E402
from cyp import evaluation  # noqa: E402
from cyp.metrics import st_rae  # noqa: E402

EXP = PROJECT_ROOT / "experiments"
OUT = EXP / "31b_interval_blend_check"
W22 = PROJECT_ROOT / "submissions" / "22_pooled_seed_ensemble" / "20261023" / "weights.csv"
KEYS = ["endpoint", "fold", "Molecule_Name"]
MEMBERS = ["pubchem", "tabicl_monroe", "tabicl_atom_jepa", "tabicl_chemeleon", "pooled_workbench"]
SCORED_ENDPOINTS = list(C.REGRESSION_ENDPOINTS)


def load_21():
    spec = importlib.util.spec_from_file_location(
        "n21", PROJECT_ROOT / "notebooks" / "21_pooled_ensemble.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def attach(wide: pl.DataFrame, source: pl.DataFrame, name: str) -> pl.DataFrame:
    """Left-join another model's OOF predictions onto the wide frame as column `name`."""
    s = source.select(*KEYS, pl.col("y_pred").alias(name))
    return wide.join(s, on=KEYS, how="left")


def blend_scores(
    wide: pl.DataFrame, weights: dict[str, dict[str, float]], cols: dict[str, str]
) -> pl.DataFrame:
    """Per (endpoint, fold) ST-RAE of the weighted blend; `cols` maps member -> column used."""
    rows = []
    for e in SCORED_ENDPOINTS:
        g = wide.filter(pl.col("endpoint") == e)
        pred = sum(weights[e][m] * g[cols.get(m, m)].to_numpy() for m in MEMBERS)
        g = g.with_columns(pl.Series("blend", pred))
        for fold, f in g.group_by("fold"):
            rows.append(
                {
                    "endpoint": e,
                    "fold": fold[0],
                    "st_rae": st_rae(
                        f["y_true"].to_numpy(),
                        f["blend"].to_numpy(),
                        f["y_lower"].to_numpy(),
                        f["y_upper"].to_numpy(),
                    ),
                }
            )
    return pl.DataFrame(rows)


def macro(scores: pl.DataFrame, folds: list[int]) -> np.ndarray:
    s = scores.filter(pl.col("fold").is_in(folds))
    return s.group_by("fold").agg(pl.col("st_rae").mean()).sort("fold")["st_rae"].to_numpy()


def placement_check(wide: pl.DataFrame, weights: dict[str, dict[str, float]]) -> pl.DataFrame:
    """Macro ST-RAE per endpoint: baseline, swapped blend raw, swapped blend placed.

    "Placed" maps the swapped blend's OOF onto the baseline blend's mean and sd per endpoint,
    which is what shipping a fixed-moment placement does to the blind vector. Shows how much of
    the interval gain is just a shift in mean and spread.
    """
    rows = []
    for e in SCORED_ENDPOINTS:
        g = wide.filter(pl.col("endpoint") == e)
        base = sum(weights[e][m] * g[m].to_numpy() for m in MEMBERS)
        cand = sum(
            weights[e][m] * g["pubchem_interval" if m == "pubchem" else m].to_numpy()
            for m in MEMBERS
        )
        placed = (cand - cand.mean()) / cand.std() * base.std() + base.mean()
        for fold in sorted(g["fold"].unique().to_list()):
            i = np.flatnonzero(g["fold"].to_numpy() == fold)
            y, lo, hi = (g[c].to_numpy()[i] for c in ("y_true", "y_lower", "y_upper"))
            rows.append(
                {
                    "endpoint": e,
                    "fold": fold,
                    "baseline": st_rae(y, base[i], lo, hi),
                    "raw": st_rae(y, cand[i], lo, hi),
                    "placed": st_rae(y, placed[i], lo, hi),
                }
            )
    return pl.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wide = load_21().load_wide()
    w = pl.read_csv(W22)
    weights = {
        e: dict(zip(g["model"], g["weight"], strict=True)) for (e,), g in w.group_by("endpoint")
    }

    pub = pl.concat(
        [
            pl.read_parquet(f)
            for f in sorted(
                (EXP / "30_interval_loss_shipped" / "cache_pubchem").glob("interval_fold*.parquet")
            )
        ]
    )
    pool = pl.concat(
        [
            pl.read_parquet(f)
            for f in sorted(
                (EXP / "30_interval_loss_shipped" / "cache_pooled").glob("interval_fold*.parquet")
            )
        ]
    )
    wide = attach(wide, pub, "pubchem_interval")
    wide = attach(wide, pool, "pooled_interval")
    assert wide["pubchem_interval"].null_count() == 0, "interval pubchem OOF misaligned"
    folds_b = sorted(pool["fold"].unique().to_list())

    base = blend_scores(wide, weights, {})
    a = blend_scores(wide, weights, {"pubchem": "pubchem_interval"})
    # B: pooled swapped on three endpoints; CYP2D6 stays MAE
    wide_b = wide.with_columns(
        pl.when(pl.col("endpoint") == "CYP2D6_pIC50_direct_inhibition")
        .then(pl.col("pooled_workbench"))
        .otherwise(pl.col("pooled_interval"))
        .alias("pooled_b")
    )
    b = blend_scores(
        wide_b.filter(pl.col("fold").is_in(folds_b)),
        weights,
        {"pubchem": "pubchem_interval", "pooled_workbench": "pooled_b"},
    )

    rows = []
    for label, cand, folds in (
        ("A pubchem->interval", a, list(range(25))),
        ("B A + pooled->interval", b, folds_b),
    ):
        x, y = macro(cand, folds), macro(base, folds)
        res = evaluation.paired_fold_bootstrap(x, y)
        rows.append(
            {
                "arm": label,
                "n_folds": len(folds),
                "baseline": float(y.mean()),
                "candidate": float(x.mean()),
                **res,
            }
        )
        for e in SCORED_ENDPOINTS:
            ce = (
                cand.filter((pl.col("endpoint") == e) & pl.col("fold").is_in(folds))
                .sort("fold")["st_rae"]
                .to_numpy()
            )
            be = (
                base.filter((pl.col("endpoint") == e) & pl.col("fold").is_in(folds))
                .sort("fold")["st_rae"]
                .to_numpy()
            )
            rows.append(
                {
                    "arm": f"  {label[:1]} {e.split('_')[0]}",
                    "n_folds": len(folds),
                    "baseline": float(be.mean()),
                    "candidate": float(ce.mean()),
                    **evaluation.paired_fold_bootstrap(ce, be),
                }
            )
    out = pl.DataFrame(rows)
    out.write_csv(OUT / "blend_check.csv")
    pc = placement_check(wide, weights)
    pc.write_csv(OUT / "placement_check_by_fold.csv")
    pc.group_by("endpoint").agg(pl.col(["baseline", "raw", "placed"]).mean()).sort(
        "endpoint"
    ).write_csv(OUT / "placement_check.csv")
    with pl.Config(tbl_rows=20, tbl_width_chars=200):
        print(out)


if __name__ == "__main__":
    main()
