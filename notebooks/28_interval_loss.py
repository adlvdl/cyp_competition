"""28: does training chemprop against the credible interval beat training against the point?

Run as a script:

    uv run python notebooks/28_interval_loss.py          # full 5x5, ~1.5-2 h per arm
    uv run python notebooks/28_interval_loss.py --folds 0 1   # smoke test

The idea comes from two public reports on the live leaderboard (sources credited in
README.md under "Ideas from other entrants"). ST-RAE is zero for a prediction inside a
compound's credible interval and linear outside it, so the exact training analogue is the
hinge `relu(lo - pred) + relu(pred - hi)`. We have only ever trained on the point estimate
(chemprop's default MSE; the pooled model used MAE) and never touched `conf_low`/`conf_high`
in a loss.

Chemprop's CLI has no interval loss and the training file carries no bounds, so
`interval_launcher` registers one inside the chemprop subprocess. It rebuilds each cell's
band from its own pIC50 with an isotonic lookup (`interval_loss.band_table`; R^2 reported
below), which is valid because the band is nearly a function of the label alone.

Arms, all the 03 winner (from-scratch multitask chemprop, 50 epochs, 4 heads), same shared
folds as 03/05 (`cv.shared_scaffold_folds`, seed 42), same chemprop torch and data seed:
  mae          L1 on the point estimate. The control: isolates the interval from L1-vs-L2.
  interval     pure hinge. Validation early-stopping selection also uses the hinge.
  interval_tb  hinge + 0.1 * L1 toward the point estimate, so predictions inside the band
               are not left to drift (mokhalaf22 found Spearman held; this guards it).

Holm family on macro ST-RAE per fold (fold is the unit, see `15`): interval vs mae,
interval_tb vs mae. Macro Spearman is reported alongside, because a loss that is
indifferent inside the band could buy ST-RAE by giving up ordering.

Scope limit, stated now: this tests the from-scratch multitask model, not `pubchem`. The
pretrained incumbent restores a checkpoint scaler that the launcher does not read, so
extending the loss to it is a separate piece of work that only makes sense if this works.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cyp  # noqa: E402,F401  (must come first: claims the OpenMP slot, see CLAUDE.md)
from cyp import constants as C  # noqa: E402
from cyp import cv, data, evaluation, graph_models, interval_loss, multitask  # noqa: E402

os.environ.setdefault("CYP_CHEMPROP_DEVICE", "cpu")

SEED = 42
OUT = PROJECT_ROOT / "experiments" / "28_interval_loss"
CACHE = OUT / "cache"
ARMS = {"mae": "mae", "interval": "interval-mae", "interval_tb": "interval-mae-tb"}


def patched_cli(original):
    """`_run_chemprop_cli` that pins chemprop's own seeds on every `train` call."""

    def run(args: list[str]) -> None:
        if args and args[0] == "train":
            args = [*args, "--pytorch-seed", str(SEED), "--data-seed", str(SEED)]
        original(args)

    return run


def run_arms(folds: list[int] | None) -> pl.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = {e: data.training_frame(e) for e in C.REGRESSION_ENDPOINTS}
    _, assignments = cv.shared_scaffold_folds(frames, n_outer=5, n_inner=5, seed=SEED)
    n_folds = assignments["fold"].n_unique()

    table = interval_loss.band_table(frames)
    quality = interval_loss.band_fit_quality(frames, table)
    quality.write_csv(OUT / "band_fit_quality.csv")
    print(quality, flush=True)
    os.environ["CYP_INTERVAL_TABLE"] = str(
        interval_loss.write_band_table(table, OUT / "band_table.npz")
    )
    shim = interval_loss.write_shim(OUT / "bin" / "chemprop_interval")

    original_bin, original_cli = graph_models.CHEMPROP_BIN, graph_models._run_chemprop_cli
    graph_models.CHEMPROP_BIN = shim
    graph_models._run_chemprop_cli = patched_cli(original_cli)
    pieces = []
    try:
        for fold in range(n_folds) if folds is None else folds:  # fold-major
            for arm, loss in ARMS.items():
                path = CACHE / f"{arm}_fold{fold:02d}.parquet"
                if not path.exists():
                    part = multitask.run_cv_multitarget(
                        frames, assignments=assignments, folds=[fold], seed=SEED, loss_function=loss
                    ).with_columns(pl.lit(arm).alias("method"))
                    part.write_parquet(path)
                pieces.append(pl.read_parquet(path))
            print(f"fold {fold + 1}/{n_folds}", flush=True)
    finally:
        graph_models.CHEMPROP_BIN, graph_models._run_chemprop_cli = original_bin, original_cli
    return pl.concat(pieces)


def macro(folds: pl.DataFrame, metric: str) -> pl.DataFrame:
    return (
        folds.group_by(["method", "fold"])
        .agg(pl.col(metric).mean().alias(metric))
        .sort(["method", "fold"])
    )


def vec(m: pl.DataFrame, method: str, metric: str) -> np.ndarray:
    return m.filter(pl.col("method") == method).sort("fold")[metric].to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="*", default=None)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    oof = run_arms(args.folds)
    if args.folds is not None:
        print("smoke test only; not writing the analysis")
        folds = evaluation.fold_metrics(oof)
        print(
            folds.group_by(["method", "endpoint"])
            .agg(pl.col(["st_rae", "rho"]).mean())
            .sort(["endpoint", "method"])
        )
        return
    oof.write_parquet(OUT / "oof.parquet")

    folds = evaluation.fold_metrics(oof)
    folds.write_csv(OUT / "fold_metrics.csv")
    per_endpoint = folds.group_by(["method", "endpoint"]).agg(
        pl.col(["st_rae", "rho", "mae"]).mean()
    )
    macro_st = macro(folds, "st_rae")
    macro_rho = macro(folds, "rho")
    overall = (
        macro_st.group_by("method")
        .agg(pl.col("st_rae").mean(), pl.col("st_rae").std().alias("sd"))
        .join(
            macro_rho.group_by("method").agg(pl.col("rho").mean().alias("macro_rho")), on="method"
        )
        .sort("st_rae")
    )
    overall.write_csv(OUT / "summary.csv")
    per_endpoint.sort(["endpoint", "method"]).write_csv(OUT / "per_endpoint.csv")
    with pl.Config(tbl_rows=30, tbl_width_chars=160):
        print(overall)
        print(per_endpoint.sort(["endpoint", "method"]))

    rows = []
    for cand in ("interval", "interval_tb"):
        rows.append(
            {
                "scope": "macro",
                "metric": "st_rae",
                "candidate": cand,
                "vs": "mae",
                **evaluation.paired_fold_bootstrap(
                    vec(macro_st, cand, "st_rae"), vec(macro_st, "mae", "st_rae")
                ),
            }
        )
        rows.append(
            {
                "scope": "macro",
                "metric": "rho",
                "candidate": cand,
                "vs": "mae",
                **evaluation.paired_fold_bootstrap(
                    vec(macro_rho, cand, "rho"), vec(macro_rho, "mae", "rho")
                ),
            }
        )
        for e in C.REGRESSION_ENDPOINTS:
            ep = folds.filter(pl.col("endpoint") == e)
            rows.append(
                {
                    "scope": e,
                    "metric": "st_rae",
                    "candidate": cand,
                    "vs": "mae",
                    **evaluation.paired_fold_bootstrap(
                        vec(ep, cand, "st_rae"), vec(ep, "mae", "st_rae")
                    ),
                }
            )
    rows.append(
        {
            "scope": "macro",
            "metric": "st_rae",
            "candidate": "interval_tb",
            "vs": "interval",
            **evaluation.paired_fold_bootstrap(
                vec(macro_st, "interval_tb", "st_rae"), vec(macro_st, "interval", "st_rae")
            ),
        }
    )
    res = pl.DataFrame(rows).with_columns(
        (pl.col("candidate") + " vs " + pl.col("vs")).alias("comparison")
    )
    family = res.filter(
        (pl.col("scope") == "macro") & (pl.col("metric") == "st_rae") & (pl.col("vs") == "mae")
    )
    holm = evaluation.holm_bonferroni(
        dict(zip(family["comparison"], family["p_value"], strict=True))
    )
    res = res.join(
        holm.select("comparison", "threshold", "significant"), on="comparison", how="left"
    )
    res.write_csv(OUT / "paired.csv")
    with pl.Config(tbl_rows=40, tbl_width_chars=200, tbl_cols=14):
        print(res)


if __name__ == "__main__":
    main()
