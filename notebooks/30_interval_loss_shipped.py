"""30: does the interval loss still help the two chemprop models that actually ship?

Run as a script:

    uv run python notebooks/30_interval_loss_shipped.py pubchem [--folds 0 1] [--arms mse interval]
    uv run python notebooks/30_interval_loss_shipped.py pooled  [--folds 0 1 2 3 4]
    uv run python notebooks/30_interval_loss_shipped.py report

`28` showed the interval loss (see README, "Ideas from other entrants") lowers macro ST-RAE
by 0.023 for from-scratch multitask chemprop, with Spearman unchanged. That model does not
ship. Two that do are tested here, with the launcher extended for each:

pubchem  `05`'s incumbent: the PubChem-pretrained 4-head encoder, fine-tuned per fold. The
         checkpoint (`.chemprop_pretrain/pubchem`) is reused as is, so every arm fine-tunes
         from the same weights. Chemprop rebuilds a checkpointed model from the saved file
         and ignores `--loss-function`, so the launcher patches the criterion in after the
         load (see `interval_launcher`); targets are scaled by the checkpoint's own scaler.
         Arms, 25 shared folds, same torch/data seed:
           mse       05's exact configuration (chemprop's default loss) -- the thing shipped
           mae       L1, isolating the interval from L1-vs-L2
           interval  the hinge on the four scored heads

pooled   `20`/`21`'s workbench member: one from-scratch 17-head model on challenge plus
         public rows, MAE on every head. The interval loss applies to the four scored heads;
         the 13 auxiliary heads stay L1. One fold costs ~53 minutes (400 epochs, depth 5),
         so a full 5x5 arm is ~22 h and is not run. Instead: outer repeat 0 (folds 0-4, a
         complete 5-fold partition), the interval arm only. The control is `20`'s own cached
         MAE OOF for those folds, **not seed-matched**, so seed noise is inside the
         comparison. Read it as a thin check, not a Holm-corrected result.

Holm family for pubchem on macro ST-RAE per fold (fold is the unit, see `15`):
interval vs mse (the decision), interval vs mae (the mechanism).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cyp  # noqa: E402,F401  (must come first: claims the OpenMP slot, see CLAUDE.md)
from cyp import aux_training, cv, data, evaluation, graph_models, interval_loss  # noqa: E402
from cyp import constants as C  # noqa: E402

os.environ.setdefault("CYP_CHEMPROP_DEVICE", "cpu")

SEED = 42
OUT = PROJECT_ROOT / "experiments" / "30_interval_loss_shipped"
PUBCHEM_ARMS = {"mse": None, "mae": "mae", "interval": "interval-mae"}
PUBCHEM_SNAPSHOT = (
    "20260915"  # 05's external snapshot; only used for the pretraining-frame column check
)


def load_20():
    spec = importlib.util.spec_from_file_location(
        "nb20", PROJECT_ROOT / "notebooks" / "20_pooled_workbench.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Launcher:
    """Context that points graph_models at the interval shim and pins chemprop's seeds."""

    def __init__(self, frames: dict[str, pl.DataFrame]) -> None:
        table = interval_loss.band_table(frames)
        os.environ["CYP_INTERVAL_TABLE"] = str(
            interval_loss.write_band_table(table, OUT / "band_table.npz")
        )
        self.shim = interval_loss.write_shim(OUT / "bin" / "chemprop_interval")

    def __enter__(self) -> Launcher:
        self.saved = (graph_models.CHEMPROP_BIN, graph_models._run_chemprop_cli)
        original = self.saved[1]

        def run(args: list[str]) -> None:
            if args and args[0] == "train":
                args = [*args, "--pytorch-seed", str(SEED), "--data-seed", str(SEED)]
            original(args)

        graph_models.CHEMPROP_BIN, graph_models._run_chemprop_cli = self.shim, run
        return self

    def __exit__(self, *exc) -> None:
        graph_models.CHEMPROP_BIN, graph_models._run_chemprop_cli = self.saved


def run_pubchem(folds: list[int] | None, arms: list[str]) -> None:
    cache = OUT / "cache_pubchem"
    cache.mkdir(parents=True, exist_ok=True)
    frames = {e: data.training_frame(e) for e in C.REGRESSION_ENDPOINTS}
    _, assignments = cv.shared_scaffold_folds(frames, n_outer=5, n_inner=5, seed=SEED)
    n_folds = assignments["fold"].n_unique()
    pretraining = aux_training.public_pretraining_matrix(
        C.REGRESSION_ENDPOINTS,
        exclude_smiles=data.load_train_inhibition()["SMILES"].to_list(),
        snapshot=PUBCHEM_SNAPSHOT,
    )
    ckpt = PROJECT_ROOT / ".chemprop_pretrain" / "pubchem" / "model_0" / "best.pt"
    assert ckpt.exists(), "05's pubchem checkpoint is missing; refusing to retrain it here"
    with Launcher(frames):
        for fold in range(n_folds) if folds is None else folds:  # fold-major
            for arm in arms:
                path = cache / f"{arm}_fold{fold:02d}.parquet"
                if path.exists():
                    continue
                kwargs = {} if PUBCHEM_ARMS[arm] is None else {"loss_function": PUBCHEM_ARMS[arm]}
                part = aux_training.run_cv_pretrained(
                    frames,
                    pretraining,
                    method_name="pubchem",
                    pretrain_epochs=30,
                    assignments=assignments,
                    folds=[fold],
                    epochs=50,
                    **kwargs,
                ).with_columns(pl.lit(arm).alias("method"))
                part.write_parquet(path)
            print(f"pubchem fold {fold + 1}/{n_folds}", flush=True)


def run_pooled(folds: list[int]) -> None:
    cache = OUT / "cache_pooled"
    cache.mkdir(parents=True, exist_ok=True)
    nb20 = load_20()
    frames, pooled, heads, weights, assignments = nb20.build()
    print(f"{len(heads)} heads, {pooled.height:,} rows; folds {folds}", flush=True)
    with Launcher(frames):
        for fold in folds:
            path = cache / f"interval_fold{fold:02d}.parquet"
            if path.exists():
                continue
            part = aux_training.run_cv_pooled(
                frames,
                pooled,
                heads,
                nb20.METHOD,
                task_weights=weights,
                assignments=assignments,
                folds=[fold],
                **{**nb20.CHEMPROP, "loss_function": "interval-mae"},
            ).with_columns(pl.lit("interval").alias("method"))
            part.write_parquet(path)
            print(f"pooled fold {fold} done", flush=True)


def macro(folds: pl.DataFrame, metric: str) -> pl.DataFrame:
    return (
        folds.group_by(["method", "fold"])
        .agg(pl.col(metric).mean().alias(metric))
        .sort(["method", "fold"])
    )


def vec(m: pl.DataFrame, method: str, metric: str, only: list[int] | None = None) -> np.ndarray:
    g = m.filter(pl.col("method") == method)
    if only is not None:
        g = g.filter(pl.col("fold").is_in(only))
    return g.sort("fold")[metric].to_numpy()


def report() -> None:
    # ---- pubchem
    files = sorted((OUT / "cache_pubchem").glob("*_fold*.parquet"))
    oof = pl.concat([pl.read_parquet(f) for f in files])
    done = oof.group_by("method").agg(pl.col("fold").n_unique().alias("n")).sort("method")
    n_common = min(done["n"].to_list())
    common = sorted(
        set.intersection(
            *[
                set(oof.filter(pl.col("method") == m)["fold"].unique().to_list())
                for m in oof["method"].unique()
            ]
        )
    )
    oof = oof.filter(pl.col("fold").is_in(common))
    print(f"pubchem: {len(common)} folds common to all arms ({n_common} in the smallest)")
    folds = evaluation.fold_metrics(oof)
    folds.write_csv(OUT / "pubchem_fold_metrics.csv")
    ms, mr = macro(folds, "st_rae"), macro(folds, "rho")
    summary = (
        ms.group_by("method")
        .agg(pl.col("st_rae").mean(), pl.col("st_rae").std().alias("sd"))
        .join(mr.group_by("method").agg(pl.col("rho").mean().alias("macro_rho")), on="method")
        .sort("st_rae")
    )
    per_ep = folds.group_by(["method", "endpoint"]).agg(pl.col(["st_rae", "rho"]).mean())
    summary.write_csv(OUT / "pubchem_summary.csv")
    per_ep.sort(["endpoint", "method"]).write_csv(OUT / "pubchem_per_endpoint.csv")
    rows = []
    for a, b in (("interval", "mse"), ("interval", "mae"), ("mae", "mse")):
        for metric, m in (("st_rae", ms), ("rho", mr)):
            rows.append(
                {
                    "scope": "macro",
                    "metric": metric,
                    "candidate": a,
                    "vs": b,
                    **evaluation.paired_fold_bootstrap(vec(m, a, metric), vec(m, b, metric)),
                }
            )
        for e in C.REGRESSION_ENDPOINTS:
            ep = folds.filter(pl.col("endpoint") == e)
            rows.append(
                {
                    "scope": e,
                    "metric": "st_rae",
                    "candidate": a,
                    "vs": b,
                    **evaluation.paired_fold_bootstrap(vec(ep, a, "st_rae"), vec(ep, b, "st_rae")),
                }
            )
    res = pl.DataFrame(rows).with_columns(
        (pl.col("candidate") + " vs " + pl.col("vs")).alias("comparison")
    )
    fam = res.filter(
        (pl.col("scope") == "macro")
        & (pl.col("metric") == "st_rae")
        & pl.col("comparison").is_in(["interval vs mse", "interval vs mae"])
    )
    holm = evaluation.holm_bonferroni(dict(zip(fam["comparison"], fam["p_value"], strict=True)))
    res = res.join(
        holm.select("comparison", "threshold", "significant"), on="comparison", how="left"
    )
    res.write_csv(OUT / "pubchem_paired.csv")
    with pl.Config(tbl_rows=40, tbl_width_chars=200, tbl_cols=14):
        print(summary)
        print(per_ep.sort(["endpoint", "method"]))
        print(res.drop("comparison"))

    # ---- pooled (thin: interval arm vs 20's cached MAE control on the same folds)
    pooled_files = sorted((OUT / "cache_pooled").glob("interval_fold*.parquet"))
    if not pooled_files:
        return
    new = pl.concat([pl.read_parquet(f) for f in pooled_files])
    pf = sorted(new["fold"].unique().to_list())
    ctl = pl.concat(
        [
            pl.read_parquet(
                PROJECT_ROOT
                / "experiments"
                / "20_pooled_workbench"
                / "cache"
                / f"pooled_workbench_fold{f}.parquet"
            )
            for f in pf
        ]
    ).with_columns(pl.lit("mae_20").alias("method"))
    both = pl.concat([new.select(ctl.columns), ctl])
    pfm = evaluation.fold_metrics(both)
    pfm.write_csv(OUT / "pooled_fold_metrics.csv")
    pms, pmr = macro(pfm, "st_rae"), macro(pfm, "rho")
    per_fold = pms.pivot(values="st_rae", index="fold", on="method").with_columns(
        (pl.col("interval") - pl.col("mae_20")).alias("diff")
    )
    per_fold.write_csv(OUT / "pooled_per_fold.csv")
    pper = (
        pfm.group_by(["method", "endpoint"])
        .agg(pl.col(["st_rae", "rho"]).mean())
        .sort(["endpoint", "method"])
    )
    pper.write_csv(OUT / "pooled_per_endpoint.csv")
    pres = pl.DataFrame(
        [
            {
                "scope": "macro",
                "metric": m,
                "candidate": "interval",
                "vs": "mae_20",
                **evaluation.paired_fold_bootstrap(vec(x, "interval", m), vec(x, "mae_20", m)),
            }
            for m, x in (("st_rae", pms), ("rho", pmr))
        ]
    )
    pres.write_csv(OUT / "pooled_paired.csv")
    with pl.Config(tbl_rows=40, tbl_width_chars=200, tbl_cols=14):
        print(f"pooled: folds {pf}, control = 20's cached MAE OOF (not seed-matched)")
        print(per_fold)
        print(pper)
        print(pres)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["pubchem", "pooled", "report"])
    parser.add_argument("--folds", type=int, nargs="*", default=None)
    parser.add_argument("--arms", nargs="*", default=list(PUBCHEM_ARMS))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "pubchem":
        run_pubchem(args.folds, args.arms)
    elif args.cmd == "pooled":
        run_pooled(args.folds if args.folds is not None else [0, 1, 2, 3, 4])
    else:
        report()


if __name__ == "__main__":
    main()
