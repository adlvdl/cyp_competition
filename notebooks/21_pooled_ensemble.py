"""21: does 20's pooled workbench-recipe model improve the shipped pool-B blend?

Run as a script:

    uv run python notebooks/21_pooled_ensemble.py

Trains nothing -- reads 20's 25-fold OOF and 17's candidates. Weights are fitted on four
outer repeats and scored on the fifth, rotated, exactly as in 14 and 17, so every ensemble
number is out-of-sample with respect to the weights. Pool B is re-run here with the same
settings as 17 so the comparison is seed-for-seed rather than against a table.

Family for the Holm correction (two planned comparisons):
  1. caruana_BP vs caruana_B   -- does adding the pooled model help the shipped blend
  2. pooled     vs pubchem     -- is the pooled model competitive on its own
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cyp  # noqa: E402,F401
from cyp import constants as C  # noqa: E402
from cyp import ensemble, evaluation  # noqa: E402
from cyp.metrics import st_rae  # noqa: E402

EXP = PROJECT_ROOT / "experiments"
OUT = EXP / "21_pooled_ensemble"
KEYS = ["endpoint", "fold", "Molecule_Name"]
SRC = {
    "pubchem": (EXP / "05_auxiliary_data" / "oof.parquet", "pubchem"),
    "tabicl_monroe": (EXP / "16_encoders" / "oof.parquet", "tabicl_monroe"),
    "tabicl_atom_jepa": (EXP / "16_encoders" / "oof.parquet", "tabicl_atom_jepa"),
    "tabicl_chemeleon": (EXP / "16_encoders" / "oof.parquet", "tabicl_chemeleon"),
}
POOL_B = list(SRC)
POOL_BP = [*POOL_B, "pooled_workbench"]


def load_wide() -> pl.DataFrame:
    truth = (
        pl.read_parquet(SRC["pubchem"][0])
        .filter(pl.col("method") == "pubchem")
        .select(*KEYS, "outer_fold", "y_true", "y_lower", "y_upper")
        .sort(KEYS)
    )
    wide = truth
    for name, (path, method) in SRC.items():
        s = pl.read_parquet(path).filter(pl.col("method") == method).sort(KEYS)
        assert s.height == truth.height and (s.select(KEYS) == truth.select(KEYS)).to_numpy().all()
        wide = wide.with_columns(s["y_pred"].alias(name))
    cache = sorted((EXP / "20_pooled_workbench" / "cache").glob("*.parquet"))
    pooled = pl.concat([pl.read_parquet(f) for f in cache])
    pooled = pooled.sort(KEYS)
    assert pooled.height == truth.height, (pooled.height, truth.height)
    assert (pooled.select(KEYS) == truth.select(KEYS)).to_numpy().all(), "pooled rows misaligned"
    assert float((pooled["y_true"] - truth["y_true"]).abs().max()) < 1e-9
    return wide.with_columns(pooled["y_pred"].alias("pooled_workbench"))


def rotate(wide: pl.DataFrame, members: list[str], tag: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Caruana and uniform blends, weights fitted without the held-out outer repeat."""
    parts, weights = [], []
    for e in C.REGRESSION_ENDPOINTS:
        ep = wide.filter(pl.col("endpoint") == e)
        for held in sorted(ep["outer_fold"].unique().to_list()):
            fit, test = ep.filter(pl.col("outer_fold") != held), ep.filter(pl.col("outer_fold") == held)
            lo, hi = fit["y_lower"].to_numpy(), fit["y_upper"].to_numpy()
            w = ensemble.caruana_selection(
                {m: fit[m].to_numpy() for m in members},
                fit["y_true"].to_numpy(),
                lambda y, p, lo=lo, hi=hi: st_rae(y, p, lo, hi),
                maximize=False,
                n_bags=1,
                bag_frac=1.0,
                max_iter=50,
            )
            weights.extend(
                {"pool": tag, "endpoint": e, "held_out": held, "model": m, "weight": v}
                for m, v in w.items()
            )
            keep = test.select(*KEYS, "y_true", "y_lower", "y_upper")
            parts.append(
                keep.with_columns(
                    pl.Series("y_pred", sum(w[m] * test[m].to_numpy() for m in members)),
                    pl.lit(f"caruana_{tag}").alias("method"),
                )
            )
            parts.append(
                keep.with_columns(
                    pl.Series("y_pred", np.mean([test[m].to_numpy() for m in members], axis=0)),
                    pl.lit(f"uniform_{tag}").alias("method"),
                )
            )
    return pl.concat(parts), pl.DataFrame(weights)


def macro_by_fold(oof: pl.DataFrame) -> pl.DataFrame:
    f = evaluation.fold_metrics(oof)
    return evaluation.macro_averaged_fold_metrics(
        {e: f.filter(pl.col("endpoint") == e) for e in C.REGRESSION_ENDPOINTS}, metric_col="st_rae"
    )


def vec(m: pl.DataFrame, method: str) -> np.ndarray:
    return m.filter(pl.col("method") == method).sort("fold")["st_rae"].to_numpy()


def paired(a: np.ndarray, b: np.ndarray, n: int = 10_000, seed: int = 42) -> dict[str, float]:
    d = a - b
    r = d[np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    p = 2.0 * min((r <= 0).mean(), (r >= 0).mean())
    return {"diff": float(d.mean()), "ci_low": float(np.percentile(r, 2.5)),
            "ci_high": float(np.percentile(r, 97.5)), "p_value": float(min(p, 1.0)),
            "folds_better": int((d < 0).sum()), "n_folds": len(d)}


def main() -> None:
    wide = load_wide()
    base = wide.select(*KEYS, "y_true", "y_lower", "y_upper")
    singles = pl.concat(
        [base.with_columns(wide[m].alias("y_pred"), pl.lit(m).alias("method"))
         for m in ("pubchem", "pooled_workbench")]
    )
    ens_b, w_b = rotate(wide, POOL_B, "B")
    ens_bp, w_bp = rotate(wide, POOL_BP, "BP")
    cols = singles.columns
    allp = pl.concat([singles, ens_b.select(cols), ens_bp.select(cols)])
    allp.write_parquet(OUT / "oof.parquet")
    pl.concat([w_b, w_bp]).write_csv(OUT / "caruana_weights_by_rotation.csv")

    macro = macro_by_fold(allp)
    summary = (
        macro.group_by("method")
        .agg(pl.col("st_rae").mean().alias("macro_st_rae"), pl.col("st_rae").std().alias("std"))
        .sort("macro_st_rae")
    )
    summary.write_csv(OUT / "macro.csv")
    print(summary)

    cmp = {
        "caruana_BP vs caruana_B": paired(vec(macro, "caruana_BP"), vec(macro, "caruana_B")),
        "pooled vs pubchem": paired(vec(macro, "pooled_workbench"), vec(macro, "pubchem")),
    }
    holm = evaluation.holm_bonferroni({k: v["p_value"] for k, v in cmp.items()})
    table = pl.DataFrame([{"comparison": k, **v} for k, v in cmp.items()]).join(
        holm.select("comparison", "threshold", "significant"), on="comparison"
    )
    extra = {
        "uniform_BP vs uniform_B": paired(vec(macro, "uniform_BP"), vec(macro, "uniform_B")),
        "caruana_BP vs pubchem": paired(vec(macro, "caruana_BP"), vec(macro, "pubchem")),
    }  # descriptive, outside the Holm family
    table = pl.concat([table, pl.DataFrame([{"comparison": k, **v} for k, v in extra.items()],
                                           ).with_columns(pl.lit(None, pl.Float64).alias("threshold"),
                                                          pl.lit(None, pl.Boolean).alias("significant"))
                       .select(table.columns)])
    table.write_csv(OUT / "paired_macro.csv")
    with pl.Config(tbl_cols=20, tbl_width_chars=200):
        print(table)

    # Per endpoint, BP vs B: where does the gain live
    fm = evaluation.fold_metrics(allp)
    rows = []
    for e in C.REGRESSION_ENDPOINTS:
        g = fm.filter(pl.col("endpoint") == e)
        a = g.filter(pl.col("method") == "caruana_BP").sort("fold")["st_rae"].to_numpy()
        b = g.filter(pl.col("method") == "caruana_B").sort("fold")["st_rae"].to_numpy()
        rows.append({"endpoint": e.split("_")[0], "caruana_B": b.mean(), "caruana_BP": a.mean(),
                     **paired(a, b)})
    per = pl.DataFrame(rows)
    per = per.join(
        evaluation.holm_bonferroni({r["endpoint"]: r["p_value"] for r in rows}).select(
            pl.col("comparison").alias("endpoint"), pl.col("significant").alias("holm_significant")),
        on="endpoint")
    per.write_csv(OUT / "paired_per_endpoint.csv")
    with pl.Config(tbl_cols=20, tbl_width_chars=200):
        print(per)

    # Weight stability for BP, and member correlations
    ws = (pl.concat([w_bp]).group_by("endpoint", "model")
          .agg(pl.col("weight").mean().alias("mean"), pl.col("weight").min().alias("min"),
               pl.col("weight").max().alias("max")).sort("endpoint", "model"))
    ws.write_csv(OUT / "caruana_weight_stability.csv")
    with pl.Config(tbl_rows=40):
        print(ws)
    corr = []
    for e in C.REGRESSION_ENDPOINTS:
        g = wide.filter(pl.col("endpoint") == e)
        corr.append({"endpoint": e.split("_")[0],
                     **{m: float(np.corrcoef(g[m], g["pooled_workbench"])[0, 1]) for m in POOL_B}})
    pl.DataFrame(corr).write_csv(OUT / "pooled_correlation.csv")
    print(pl.DataFrame(corr))


if __name__ == "__main__":
    main()
