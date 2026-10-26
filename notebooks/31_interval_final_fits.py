"""31: refit the two chemprop members of the shipped blend with the interval loss.

Run as a script, under `caffeinate` (an idle-sleeping Mac pauses a long fit, see CLAUDE.md):

    caffeinate -ims uv run --no-sync python notebooks/31_interval_final_fits.py pubchem 0 1 2 3 4
    caffeinate -ims uv run --no-sync python notebooks/31_interval_final_fits.py pooled 0 1 2

Same recipes and seeds as `19_seed_spread` (pubchem) and `20_pooled_workbench final` (pooled),
fitted on 100% of the labelled rows, with one change: `--loss-function interval-mae` through
the `interval_launcher` shim (evidence: notebooks 28 and 30). One prediction file per seed, so a
crash between seeds repeats nothing. Output: `experiments/31_interval_final_fits/`.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cyp  # noqa: E402,F401  (must come first: claims the OpenMP slot, see CLAUDE.md)
from cyp import constants as C  # noqa: E402
from cyp import data, graph_models, interval_loss, multitask  # noqa: E402
from cyp.graph_models import ChempropMultitargetModel  # noqa: E402

os.environ.setdefault("CYP_CHEMPROP_DEVICE", "cpu")

OUT = PROJECT_ROOT / "experiments" / "31_interval_final_fits"
LOSS = "interval-mae"


class Seeded:
    """Point graph_models at the interval shim and pin chemprop's seeds for one fit."""

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def __enter__(self) -> None:
        self.saved = (graph_models.CHEMPROP_BIN, graph_models._run_chemprop_cli)
        original = self.saved[1]

        def run(args: list[str]) -> None:
            if args and args[0] == "train":
                args = [*args, "--pytorch-seed", str(self.seed), "--data-seed", str(self.seed)]
            original(args)

        graph_models.CHEMPROP_BIN = OUT / "bin" / "chemprop_interval"
        graph_models._run_chemprop_cli = run

    def __exit__(self, *exc: object) -> None:
        graph_models.CHEMPROP_BIN, graph_models._run_chemprop_cli = self.saved


def prepare(frames: dict[str, pl.DataFrame]) -> None:
    """Band table and shim, written once and pointed at through the environment."""
    table = interval_loss.band_table(frames)
    os.environ["CYP_INTERVAL_TABLE"] = str(
        interval_loss.write_band_table(table, OUT / "band_table.npz")
    )
    interval_loss.write_shim(OUT / "bin" / "chemprop_interval")


def run_pubchem(seeds: list[int]) -> None:
    cache = OUT / "pubchem"
    cache.mkdir(parents=True, exist_ok=True)
    frames = {e: data.training_frame(e) for e in C.REGRESSION_ENDPOINTS}
    test = data.load_test()
    pretrain_dir = C.PROJECT_ROOT / ".chemprop_pretrain" / "pubchem"
    if not (pretrain_dir / "model_0" / "best.pt").exists():
        raise FileNotFoundError(f"{pretrain_dir} has no checkpoint")
    prepare(frames)
    for seed in seeds:
        path = cache / f"pubchem_interval_seed{seed}.parquet"
        if path.exists():
            continue
        start = time.perf_counter()
        with Seeded(seed):
            preds = multitask.fit_predict_test_multitarget(
                frames,
                test["SMILES"].to_list(),
                pretrain_dir=pretrain_dir,
                pretrain_epochs=30,
                seed=seed,
                loss_function=LOSS,
            )
        pl.DataFrame({"Molecule_Name": test["Molecule_Name"], **preds}).write_parquet(path)
        print(f"pubchem seed {seed} done in {time.perf_counter() - start:.0f}s", flush=True)


def run_pooled(seeds: list[int]) -> None:
    cache = OUT / "pooled"
    cache.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location(
        "nb20", PROJECT_ROOT / "notebooks" / "20_pooled_workbench.py"
    )
    nb20 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nb20)
    frames, pooled, heads, weights, _ = nb20.build(exclude_blind=True)
    test = data.load_test()
    prepare(frames)
    print(f"{len(heads)} heads, {pooled.height:,} rows; final fit seeds {seeds}", flush=True)
    for seed in seeds:
        path = cache / f"pooled_interval_seed{seed}.parquet"
        if path.exists():
            continue
        start = time.perf_counter()
        # Same carve-out as 20: a seeded random 10% for early stopping only.
        order = np.random.default_rng(seed).permutation(pooled.height)
        n_val = max(1, int(0.1 * pooled.height))
        val, fit = pooled[order[:n_val]], pooled[order[n_val:]]
        with Seeded(seed):
            model = ChempropMultitargetModel(
                targets=heads, task_weights=weights, **{**nb20.CHEMPROP, "loss_function": LOSS}
            )
            model.fit(
                fit["SMILES"].to_list(),
                fit.select(heads).to_numpy(),
                smiles_val=val["SMILES"].to_list(),
                y_val=val.select(heads).to_numpy(),
            )
            preds = model.predict(test["SMILES"].to_list())
        out = pl.DataFrame({"Molecule_Name": test["Molecule_Name"]})
        for i, e in enumerate(C.REGRESSION_ENDPOINTS):
            out = out.with_columns(pl.Series(e, preds[:, i].astype(float)))
        assert out.select(C.REGRESSION_ENDPOINTS).null_count().sum_horizontal().item() == 0
        out.write_parquet(path)
        print(f"pooled seed {seed} done in {time.perf_counter() - start:.0f}s", flush=True)


if __name__ == "__main__":
    which, seed_list = sys.argv[1], [int(s) for s in sys.argv[2:]]
    OUT.mkdir(parents=True, exist_ok=True)
    {"pubchem": run_pubchem, "pooled": run_pooled}[which](seed_list)
