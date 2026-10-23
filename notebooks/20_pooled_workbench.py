"""20: one model on challenge + public labels (the workbench `union-p30` recipe).

Run as a script:

    uv run python notebooks/20_pooled_workbench.py probe     # time one short fit
    uv run python notebooks/20_pooled_workbench.py cv 0 1 2  # fold-major CV, cached
    uv run python notebooks/20_pooled_workbench.py final 0 1 2  # fit on all rows, predict blind

Replicates the regression member `cyp-reg-chemprop-union-p30` from
https://supercowpowers.github.io/workbench/blogs/cyp_challenge/ (code:
`ml_pipelines/OpenADMET/cyp/cyp_chemprop_union.py`):

- one Chemprop D-MPNN **from scratch**, public sources as extra *heads on extra rows*
  (not a pretrain stage), 4 scored pIC50 + 4 screen log2fc + ChEMBL pIC50 +
  Veith `max_response`
- task weights: inverse label count over the 4 scored heads, mean-normalised to 1;
  every other head 0.30 (= `--public-weight 0.30` x mean primary weight of 1)
- workbench chemprop template defaults: hidden 700, depth 5, dropout 0.1, FFN
  300 x 2, batch 64, warmup 2, lr 1e-4 / 1e-3 / 1e-4, MAE loss, max 400 epochs,
  patience 40

Deviations, all forced rather than chosen:

- public rows are structurally disjoint from every challenge compound (05's
  `exclude_smiles`), so a held-out compound has no public twin in training. Theirs
  merges the ~185 shared ChEMBL structures onto the challenge row.
- no CYP2C19 `max_response` (our Veith loader has the four scored isoforms), so 17
  heads rather than their 18.
- one model per CV fold, scored on our shared scaffold folds; theirs is a 5-model
  ensemble over the union's own scaffold folds.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cyp  # noqa: E402,F401  (must come first: OpenMP load order, see CLAUDE.md)
from cyp import auxiliary, aux_training, cv, data, timings  # noqa: E402
from cyp import constants as C  # noqa: E402

os.environ.setdefault("CYP_CHEMPROP_DEVICE", "cpu")

NAME = Path(__file__).stem
OUT = PROJECT_ROOT / "experiments" / NAME
CACHE = OUT / "cache"
METHOD = "pooled_workbench"

AUX_WEIGHT = 0.30  # workbench ASSAY_WEIGHT and --public-weight, each x mean primary (= 1)
CHEMPROP = dict(
    message_hidden_dim=700,
    depth=5,
    dropout=0.1,
    ffn_hidden_dim=300,
    ffn_num_layers=2,
    batch_size=64,
    init_lr=1e-4,
    max_lr=1e-3,
    final_lr=1e-4,
    warmup_epochs=2,
    loss_function="mae",
    patience=40,
    epochs=400,
)


def build(
    exclude_blind: bool = False,
) -> tuple[dict, pl.DataFrame, list[str], list[float], pl.DataFrame]:
    """Frames, the pooled table, its head order, task weights, and the fold table.

    `exclude_blind` also removes public structures matching a blind-set compound. CV
    leaves it off so the table is identical to the one the OOF was measured on; the
    final fit turns it on, since no public label for a scored compound should train.
    """
    endpoints = C.REGRESSION_ENDPOINTS
    frames = {e: data.training_frame(e) for e in endpoints}
    _, assignments = cv.shared_scaffold_folds(frames, n_outer=5, n_inner=5)

    exclude = data.load_train_inhibition()["SMILES"].to_list()
    if exclude_blind:
        exclude = exclude + data.load_test()["SMILES"].to_list()
    public, public_cols = aux_training.full_union_matrix(
        endpoints, exclude_smiles=exclude, include_tox21=False
    )
    # Drop the Veith pIC50 columns that carry the scored names (pretraining targets);
    # the workbench member uses Veith only through max_response.
    keep = [c for c in public_cols if c not in endpoints]
    screen = auxiliary.screen_targets()
    pooled, heads = aux_training.pooled_wide_frame(frames, public, keep, screen)

    counts = np.array([frames[e].height for e in endpoints], dtype=float)
    inv = 1.0 / counts
    primary = (inv / inv.mean()).tolist()  # workbench.compute_inverse_count_task_weights
    weights = primary + [AUX_WEIGHT] * (len(heads) - len(endpoints))
    return frames, pooled, heads, weights, assignments


def gradient_share(pooled: pl.DataFrame, heads: list[str], weights: list[float]) -> None:
    """Print each group's share of loss mass (label count x weight), as workbench does."""
    n = len(C.REGRESSION_ENDPOINTS)
    groups = {
        "scored": range(0, n),
        "screen": range(n, n + 4),
        "public": range(n + 4, len(heads)),
    }
    mass = {
        g: sum(int(pooled[heads[i]].is_not_null().sum()) * weights[i] for i in idx)
        for g, idx in groups.items()
    }
    total = sum(mass.values())
    for g, idx in groups.items():
        labels = sum(int(pooled[heads[i]].is_not_null().sum()) for i in idx)
        print(f"  {g:7s} {len(idx):3d} heads {labels:9,} labels {100 * mass[g] / total:5.1f}%")


def probe() -> None:
    """Time a few epochs on the biggest training table to decide what is affordable."""
    frames, pooled, heads, weights, assignments = build()
    print(f"{len(heads)} heads, {pooled.height:,} rows")
    gradient_share(pooled, heads, weights)
    kwargs = {**CHEMPROP, "epochs": 3, "patience": 3}
    start = time.perf_counter()
    aux_training.run_cv_pooled(
        frames, pooled, heads, METHOD, task_weights=weights, assignments=assignments,
        folds=[0], **kwargs,
    )
    print(f"3 epochs incl. startup + predict: {time.perf_counter() - start:.0f}s")


def run_cv(folds: list[int]) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    frames, pooled, heads, weights, assignments = build()
    print(f"{len(heads)} heads, {pooled.height:,} rows; folds {folds}")
    gradient_share(pooled, heads, weights)
    for fold in folds:
        path = CACHE / f"{METHOD}_fold{fold}.parquet"
        if path.exists():
            continue
        start = time.perf_counter()
        part = aux_training.run_cv_pooled(
            frames, pooled, heads, METHOD, task_weights=weights, assignments=assignments,
            folds=[fold], **CHEMPROP,
        )
        part.write_parquet(path)
        timings.record(OUT / "timings.csv", "pooled", METHOD, f"fold{fold}", "full",
                       time.perf_counter() - start)
        print(f"fold {fold} done in {time.perf_counter() - start:.0f}s", flush=True)


def run_final(seeds: list[int]) -> None:
    """Fit on every labelled challenge row plus all public rows; predict the blind set.

    Not cached against CV on purpose (CLAUDE.md): only the per-seed prediction file is
    kept, so a crash between seeds does not repeat the finished ones. Each seed pins
    chemprop's torch and shuffle seeds and the validation carve-out, the same way
    `19_seed_spread` did.
    """
    from cyp import graph_models
    from cyp.graph_models import ChempropMultitargetModel

    final_dir = OUT / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    frames, pooled, heads, weights, _ = build(exclude_blind=True)
    test = data.load_test()
    print(f"{len(heads)} heads, {pooled.height:,} rows; final fit seeds {seeds}")
    gradient_share(pooled, heads, weights)

    for seed in seeds:
        path = final_dir / f"pooled_seed{seed}.parquet"
        if path.exists():
            continue
        start = time.perf_counter()
        order = np.random.default_rng(seed).permutation(pooled.height)
        n_val = max(1, int(0.1 * pooled.height))
        val, fit = pooled[order[:n_val]], pooled[order[n_val:]]

        original = graph_models._run_chemprop_cli

        def patched(args: list[str], seed: int = seed) -> None:
            if args and args[0] == "train":
                args = [*args, "--pytorch-seed", str(seed), "--data-seed", str(seed)]
            original(args)

        graph_models._run_chemprop_cli = patched
        try:
            model = ChempropMultitargetModel(targets=heads, task_weights=weights, **CHEMPROP)
            model.fit(
                fit["SMILES"].to_list(),
                fit.select(heads).to_numpy(),
                smiles_val=val["SMILES"].to_list(),
                y_val=val.select(heads).to_numpy(),
            )
            preds = model.predict(test["SMILES"].to_list())
        finally:
            graph_models._run_chemprop_cli = original

        n = len(C.REGRESSION_ENDPOINTS)
        out = pl.DataFrame({"Molecule_Name": test["Molecule_Name"]})
        for i, e in enumerate(C.REGRESSION_ENDPOINTS):
            out = out.with_columns(pl.Series(e, preds[:, i].astype(float)))
        assert out.select(C.REGRESSION_ENDPOINTS).null_count().sum_horizontal().item() == 0
        out.write_parquet(path)
        print(f"seed {seed} done in {time.perf_counter() - start:.0f}s", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "probe":
        probe()
    elif cmd == "final":
        run_final([int(f) for f in sys.argv[2:]])
    elif cmd == "cv":
        run_cv([int(f) for f in sys.argv[2:]])
