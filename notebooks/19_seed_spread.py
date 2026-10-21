"""A2a: how much does a final `pubchem` fit move with its random seed?

Run as a script:

    uv run python notebooks/19_seed_spread.py

Refits `05`'s shipped `pubchem` model (same frames, same PubChem checkpoint, same
epochs) on 100% of the labelled rows with several seeds, and compares the blind-set
predictions. The question is whether seed averaging can buy anything: if seeds agree
to Spearman > ~0.99 there is nothing to average away.

`multitask.fit_predict_test_multitarget`'s `seed` only controls the validation
carve-out. Chemprop's own torch and data-shuffle seeds are passed here by wrapping
`graph_models._run_chemprop_cli`, so a seed changes the split, the shuffle order and
the dropout masks together -- which is what a seed ensemble would vary.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cyp  # noqa: E402,F401  (must come first: claims the OpenMP slot, see CLAUDE.md)
from cyp import constants as C  # noqa: E402
from cyp import data, graph_models, multitask  # noqa: E402

os.environ.setdefault("CYP_CHEMPROP_DEVICE", "cpu")

SEEDS = (0, 1, 2, 3, 4)
PRETRAIN_EPOCHS = 30  # 05's full-run value, only used to locate the checkpoint config
SUB05 = PROJECT_ROOT / "submissions" / "05_auxiliary_data" / "20260915" / "activity.csv"
OUT = PROJECT_ROOT / "experiments" / "19_seed_spread"
CACHE = OUT / "cache"


def _seeded_cli(seed: int):
    """`_run_chemprop_cli` that pins chemprop's own seeds on every `train` call."""
    original = graph_models._run_chemprop_cli

    def run(args: list[str]) -> None:
        if args and args[0] == "train":
            args = [*args, "--pytorch-seed", str(seed), "--data-seed", str(seed)]
        original(args)

    return original, run


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = {e: data.training_frame(e) for e in C.REGRESSION_ENDPOINTS}
    test = data.load_test()
    pretrain_dir = C.PROJECT_ROOT / ".chemprop_pretrain" / "pubchem"
    if not (pretrain_dir / "model_0" / "best.pt").exists():
        raise FileNotFoundError(f"{pretrain_dir} has no checkpoint")

    for seed in SEEDS:
        path = CACHE / f"pubchem_seed{seed}.parquet"
        if path.exists():
            continue
        original, patched = _seeded_cli(seed)
        graph_models._run_chemprop_cli = patched
        try:
            preds = multitask.fit_predict_test_multitarget(
                frames,
                test["SMILES"].to_list(),
                pretrain_dir=pretrain_dir,
                pretrain_epochs=PRETRAIN_EPOCHS,
                seed=seed,
            )
        finally:
            graph_models._run_chemprop_cli = original
        pl.DataFrame({"Molecule_Name": test["Molecule_Name"], **preds}).write_parquet(path)
        print(f"seed {seed} done", flush=True)

    # Compare. Columns: seed vectors plus 05's shipped vector as a reference point.
    shipped = pl.read_csv(SUB05)
    rows = []
    for endpoint in C.REGRESSION_ENDPOINTS:
        vecs = {
            f"seed{s}": pl.read_parquet(CACHE / f"pubchem_seed{s}.parquet")[endpoint].to_numpy()
            for s in SEEDS
        }
        vecs["shipped05"] = shipped[endpoint].to_numpy()
        names = list(vecs)
        mat = np.vstack([vecs[n] for n in names])
        ranks = np.vstack([np.argsort(np.argsort(v)) for v in mat])
        pair = [
            (np.corrcoef(ranks[i], ranks[j])[0, 1], np.corrcoef(mat[i], mat[j])[0, 1])
            for i in range(len(SEEDS))
            for j in range(i + 1, len(SEEDS))
        ]
        mean_vec = mat[: len(SEEDS)].mean(axis=0)
        rows.append(
            {
                "endpoint": endpoint.split("_")[0],
                "spearman_seed_pairs_min": min(p[0] for p in pair),
                "spearman_seed_pairs_mean": float(np.mean([p[0] for p in pair])),
                "pearson_seed_pairs_mean": float(np.mean([p[1] for p in pair])),
                "mean_across_seeds": float(mat[: len(SEEDS)].mean()),
                "sd_single_seed": float(mat[: len(SEEDS)].std(axis=1).mean()),
                "sd_seed_average": float(mean_vec.std()),
                # Per-compound disagreement between seeds, in pIC50 units.
                "per_compound_seed_sd": float(mat[: len(SEEDS)].std(axis=0).mean()),
                "spearman_shipped_vs_seedmean": float(
                    np.corrcoef(np.argsort(np.argsort(mat[-1])), np.argsort(np.argsort(mean_vec)))[
                        0, 1
                    ]
                ),
                "shipped_mean": float(mat[-1].mean()),
            }
        )
    summary = pl.DataFrame(rows)
    summary.write_csv(OUT / "seed_spread.csv")
    with pl.Config(tbl_cols=20, tbl_width_chars=200):
        print(summary)


if __name__ == "__main__":
    main()
