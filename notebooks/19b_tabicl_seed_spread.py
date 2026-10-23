"""19b: seed spread of the TabICL ensemble members (Monroe, Atom-JEPA), final-fit setting.

Run as a script: uv run python notebooks/19b_tabicl_seed_spread.py

Same inputs as build_pool_b_submission.py (all labelled rows, blind-set embeddings), only the
TabICL seed varies. Seed 42 is the shipped one. Caches per (arm, endpoint, seed).
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cyp  # noqa: E402,F401
from cyp import constants as C  # noqa: E402
from cyp import data, encoders  # noqa: E402
from cyp.tabular_models import predict_subprocess  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

EXP = PROJECT_ROOT / "experiments"
OUT = EXP / "19_seed_spread"
CACHE = OUT / "tabicl_cache"
SEEDS = (42, 0, 1, 2, 3)
BLIND = EXP / "17_regression_ensemble" / "blind_embeddings"


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    test = data.load_test()
    endpoints = list(C.REGRESSION_ENDPOINTS)
    frames = {e: data.training_frame(e) for e in endpoints}
    union = sorted({s for f in frames.values() for s in f["SMILES"].to_list()})
    row_of = {s: i for i, s in enumerate(union)}
    cache16 = EXP / "16_encoders" / "cache"
    train = {
        "tabicl_monroe": encoders.embed_cached("monroe", union, cache16),
        "tabicl_atom_jepa": encoders.embed_cached("atom_jepa", union, cache16)[:, -1, :],
    }
    blind = {
        "tabicl_monroe": np.load(BLIND / "monroe.npy"),
        "tabicl_atom_jepa": np.load(BLIND / "atom_jepa.npy")[:, -1, :],
    }
    assert blind["tabicl_monroe"].shape[0] == test.height

    rows = []
    for arm in train:
        for e in endpoints:
            f = frames[e]
            x = train[arm][[row_of[s] for s in f["SMILES"].to_list()]]
            preds = {}
            for s in SEEDS:
                p = CACHE / f"{arm}_{e}_{s}.npy"
                if not p.exists():
                    pred, _, _ = predict_subprocess(
                        x, f["y_true"].to_numpy(), blind[arm], kind="tabicl", seed=s
                    )
                    np.save(p, np.asarray(pred, dtype=float))
                preds[s] = np.load(p)
                print(arm, e, s, flush=True)
            pairs = list(combinations(SEEDS, 2))
            sp = [spearmanr(preds[a], preds[b])[0] for a, b in pairs]
            mean = np.mean([preds[s] for s in SEEDS], axis=0)
            rows.append(
                {
                    "arm": arm,
                    "endpoint": e.split("_")[0],
                    "spearman_pairs_min": min(sp),
                    "spearman_pairs_mean": float(np.mean(sp)),
                    "per_compound_seed_sd": float(np.std([preds[s] for s in SEEDS], axis=0, ddof=1).mean()),
                    "sd_single_seed": float(preds[42].std()),
                    "sd_seed_average": float(mean.std()),
                    "spearman_shipped_vs_seedmean": float(spearmanr(preds[42], mean)[0]),
                }
            )
    out = pl.DataFrame(rows)
    out.write_csv(OUT / "tabicl_seed_spread.csv")
    with pl.Config(tbl_cols=20, tbl_width_chars=220):
        print(out)


if __name__ == "__main__":
    main()
