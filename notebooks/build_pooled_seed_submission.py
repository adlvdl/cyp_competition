"""Build the seed-averaged pool B + pooled regression submission.

Run as a script, not a notebook:

    uv run python notebooks/build_pooled_seed_submission.py

## What this ships

Regression: a per-endpoint Caruana blend of five models, every member a seed mean --

- `pubchem`           5 seeds of `05`'s shipped recipe (`19_seed_spread`), not 05's single vector
- `tabicl_monroe`     5 TabICL seeds on frozen Monroe embeddings (`19_seed_spread/tabicl_cache`)
- `tabicl_atom_jepa`  5 TabICL seeds on frozen Atom-JEPA embeddings (last block)
- `tabicl_chemeleon`  TabICL seed 42 only: it gets weight 0 on every endpoint in CV
- `pooled_workbench`  3 seeds of `20`'s from-scratch pooled chemprop (workbench `union-p30` recipe)

TDI: `14`'s shipped ensemble, copied unchanged.

## Weights vs members

Weights come from `21`'s single-seed OOF, refit on all 25 folds (a submission uses every fold,
as it uses every compound). The blind members are seed means, which are slightly narrower than
the single-seed members the weights were fitted for. CV has no seed-averaged OOF, so that
mismatch is unmeasured. Placement remaps spread to fixed moments, which bounds its effect.

## Placement

Identical to `17`: each endpoint's blend is affine-mapped onto the mean and sd of the vector
`12` shipped (CYP1A2 placed, CYP2D6 `07`'s, CYP2C9 and CYP3A4 `05` unplaced).
"""

from __future__ import annotations

import csv
import importlib.util
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cyp  # noqa: E402,F401
from cyp import constants as C  # noqa: E402
from cyp import data, ensemble, evaluation, placement, submission  # noqa: E402
from cyp.metrics import st_rae  # noqa: E402

EXP = PROJECT_ROOT / "experiments"
SUB12 = PROJECT_ROOT / "submissions" / "12_final_placement" / "20260919" / "activity.csv"
SUB14_TDI = PROJECT_ROOT / "submissions" / "14_tdi_ensemble" / "20260920" / "tdi.csv"
SEED_CACHE = EXP / "19_seed_spread"
POOL = ("pubchem", "tabicl_monroe", "tabicl_atom_jepa", "tabicl_chemeleon", "pooled_workbench")
TABICL_SEEDS = (42, 0, 1, 2, 3)
PUBCHEM_SEEDS = range(5)
POOLED_SEEDS = range(3)


def _load_21():
    """21's `load_wide` is the OOF the weights are fitted on; reuse it rather than copy it."""
    spec = importlib.util.spec_from_file_location("n21", PROJECT_ROOT / "notebooks" / "21_pooled_ensemble.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _by_ids(frame: pl.DataFrame, endpoint: str, ids: list[str]) -> np.ndarray:
    """Align a Molecule_Name-keyed frame to the blind set's row order."""
    out = pl.DataFrame({"Molecule_Name": ids}).join(
        frame.select("Molecule_Name", endpoint), on="Molecule_Name", how="left"
    )[endpoint].to_numpy()
    assert np.isfinite(out).all() and len(out) == len(ids)
    return out.astype(float)


def main() -> None:
    n21 = _load_21()
    test = data.load_test()
    ids = test["Molecule_Name"].to_list()
    endpoints = list(C.REGRESSION_ENDPOINTS)
    short = {e: e.split("_")[0] for e in endpoints}

    # ── Blind-set members: seed means ──────────────────────────────────────────
    pub = [pl.read_parquet(SEED_CACHE / "cache" / f"pubchem_seed{s}.parquet") for s in PUBCHEM_SEEDS]
    pool = [pl.read_parquet(EXP / "20_pooled_workbench" / "final" / f"pooled_seed{s}.parquet") for s in POOLED_SEEDS]
    blind: dict[str, dict[str, np.ndarray]] = {a: {} for a in POOL}
    for e in endpoints:
        blind["pubchem"][e] = np.mean([_by_ids(f, e, ids) for f in pub], axis=0)
        blind["pooled_workbench"][e] = np.mean([_by_ids(f, e, ids) for f in pool], axis=0)
        for arm in ("tabicl_monroe", "tabicl_atom_jepa"):
            blind[arm][e] = np.mean(
                [np.load(SEED_CACHE / "tabicl_cache" / f"{arm}_{e}_{s}.npy") for s in TABICL_SEEDS], axis=0
            )
            assert len(blind[arm][e]) == len(ids)
    # chemeleon (weight 0 in CV): reuse 17's shipped vector via its build script's single fit
    sub17 = pl.read_csv(PROJECT_ROOT / "submissions" / "17_regression_ensemble" / "20261020" / "activity.csv")
    for e in endpoints:
        blind["tabicl_chemeleon"][e] = np.zeros(len(ids))  # placeholder, asserted weight 0 below

    print("Pearson r with `pubchem` on the blind set:")
    for arm in POOL[1:]:
        if arm == "tabicl_chemeleon":
            continue
        rs = {short[e]: np.corrcoef(blind[arm][e], blind["pubchem"][e])[0, 1] for e in endpoints}
        print(f"  {arm:18s} " + "  ".join(f"{k} {v:.3f}" for k, v in rs.items()))
        assert min(rs.values()) > 0.4, f"{arm}: blind predictions barely correlate with pubchem"

    # ── Weights: Caruana on all 25 folds ───────────────────────────────────────
    wide = n21.load_wide()
    rotation = pl.read_csv(EXP / "21_pooled_ensemble" / "caruana_weights_by_rotation.csv").filter(
        pl.col("pool") == "BP"
    )
    weights: dict[str, dict[str, float]] = {}
    print("\nCaruana weights (all 25 folds) vs 21's held-out mean:")
    for e in endpoints:
        g = wide.filter(pl.col("endpoint") == e)
        lo, hi, y = g["y_lower"].to_numpy(), g["y_upper"].to_numpy(), g["y_true"].to_numpy()
        w = ensemble.caruana_selection(
            {m: g[m].to_numpy() for m in POOL},
            y,
            lambda t, p, lo=lo, hi=hi: st_rae(t, p, lo, hi),
            maximize=False,
            n_bags=1,
            bag_frac=1.0,
            max_iter=50,
        )
        weights[e] = w
        held = rotation.filter(pl.col("endpoint") == e).group_by("model").agg(pl.col("weight").mean())
        held_d = dict(zip(held["model"], held["weight"], strict=True))
        print(f"  {short[e]}: " + "  ".join(f"{a} {w[a]:.3f} ({held_d.get(a, 0.0):.3f})" for a in POOL))
    assert all(weights[e]["tabicl_chemeleon"] == 0 for e in endpoints), (
        "chemeleon got weight; its blind vector is a placeholder here, fit it before shipping"
    )

    # ── Blend, then place onto 12's validated moments ──────────────────────────
    sub12 = pl.read_csv(SUB12).sort("Molecule_Name")
    ref12 = {e: _by_ids(sub12, e, ids) for e in endpoints}
    final: dict[str, np.ndarray] = {}
    plan_rows = []
    for e in endpoints:
        raw = sum(weights[e][a] * blind[a][e] for a in POOL)
        placed = placement.affine(raw, float(ref12[e].mean()), float(ref12[e].std()))
        ties = placement.check_rank_preserved(raw, placed)
        final[e] = placed
        rho17 = float(
            pl.DataFrame({"a": placed, "b": _by_ids(sub17, e, ids)}).select(pl.corr("a", "b", method="spearman")).item()
        )
        plan_rows.append(
            {
                "endpoint": e,
                "raw_mean": round(float(raw.mean()), 3),
                "raw_sd": round(float(raw.std()), 3),
                "placed_mean": round(float(placed.mean()), 3),
                "placed_sd": round(float(placed.std()), 3),
                "floor_ties": ties,
                "spearman_vs_17": round(rho17, 3),
            }
        )
    plan = pl.DataFrame(plan_rows)
    print("\nPlacement onto 12's moments:")
    print(plan)

    # ── Write + validate ───────────────────────────────────────────────────────
    dest = PROJECT_ROOT / "submissions" / "22_pooled_seed_ensemble" / date.today().strftime("%Y%m%d")
    if dest.exists():
        raise FileExistsError(f"{dest} exists; refusing to overwrite a submission folder")
    activity = submission.build_activity_submission(final, dest / "activity.csv")
    shutil.copy(SUB14_TDI, dest / "tdi.csv")
    ok_a = submission.check(activity, track="activity")
    ok_t = submission.check(dest / "tdi.csv", track="tdi")
    print(f"\nvalidation: activity {'PASSED' if ok_a else 'FAILED'}, tdi {'PASSED' if ok_t else 'FAILED'}")
    if not (ok_a and ok_t):
        raise SystemExit("validation failed; nothing should be submitted")

    with (dest / "weights.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["endpoint", "model", "weight"])
        writer.writeheader()
        writer.writerows({"endpoint": e, "model": a, "weight": round(weights[e][a], 4)} for e in endpoints for a in POOL)
    plan.write_csv(dest / "placement_plan.csv")

    # ── Expected performance (CV; placement-blind; single-seed members) ────────
    oof = pl.read_parquet(EXP / "21_pooled_ensemble" / "oof.parquet")
    labels = {
        "caruana_BP": "pool B + pooled (held-out weights)",
        "caruana_B": "pool B (17, held-out weights)",
        "pubchem": "pubchem (05)",
    }
    exp_rows = []
    for method, label in labels.items():
        folds = evaluation.fold_metrics(oof.filter(pl.col("method") == method))
        per = folds.group_by("endpoint").agg(pl.col("st_rae").mean())
        macro = evaluation.macro_averaged_fold_metrics(
            {e: folds.filter(pl.col("endpoint") == e) for e in endpoints}, metric_col="st_rae"
        )["st_rae"].mean()
        exp_rows += [{"model": label, "endpoint": r["endpoint"], "cv_st_rae": round(r["st_rae"], 4)} for r in per.iter_rows(named=True)]
        exp_rows.append({"model": label, "endpoint": "macro", "cv_st_rae": round(float(macro), 4)})
    expected = pl.DataFrame(exp_rows).pivot(on="model", index="endpoint", values="cv_st_rae")
    expected.write_csv(dest / "expected_performance.csv")
    print("\nExpected (CV):")
    print(expected)

    (dest / "PROVENANCE.md").write_text(_provenance(plan, expected, weights, endpoints))
    print(f"\nWrote {dest}")


def _provenance(plan: pl.DataFrame, expected: pl.DataFrame, weights, endpoints) -> str:
    def md(frame: pl.DataFrame) -> str:
        cols = frame.columns
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.iter_rows()]
        return "\n".join(lines)

    w_table = pl.DataFrame(
        [{"endpoint": e.split("_")[0], **{a: round(weights[e][a], 3) for a in POOL}} for e in endpoints]
    )
    return f"""# Seed-averaged pool B + pooled workbench regression ensemble + 14's TDI ensemble

- Generated: {date.today().isoformat()}
- Script: `notebooks/build_pooled_seed_submission.py`
- Evidence: notebooks `19` / `19b` (seed spread), `20` (pooled workbench-recipe model), `21` (pool B + pooled)
- Challenge data snapshot: `20261020`
- CV: nested scaffold 5x5, shared folds. Weights fitted on all 25 folds of single-seed OOF.
- **Not yet scored on the board.**

## Regression model

Per-endpoint Caruana blend of five models, each fitted on 100% of the labelled rows. Every
blind-set member is a seed mean:

- `pubchem`: 5 seeds of `05`'s recipe on the shared PubChem-pretrained checkpoint (pretraining seed fixed)
- `tabicl_monroe`, `tabicl_atom_jepa`: 5 TabICL seeds (42, 0-3), PCA to 128, 2 estimators
- `tabicl_chemeleon`: weight 0 on every endpoint, not fitted
- `pooled_workbench`: 3 seeds of a from-scratch multitask chemprop on challenge + public rows
  (17 heads, workbench `union-p30` task weights), public rows matching the blind set removed

Weights:

{md(w_table)}

## Placement

Same as `17`: each endpoint's blend is affine-mapped onto the mean and sd of the vector `12`
shipped. This moves no compound's rank. `spearman_vs_17` is the rank agreement with 17's file.

{md(plan)}

The live board scores half the test set. Placement moments were solved on that half and may not
transfer exactly to the other half (see CLAUDE.md, Placement).

## Expected performance (CV, placement-blind, single-seed members)

{md(expected)}

CV has no seed-averaged OOF, so seed-averaging's gain (estimated 0.003-0.005 macro) is not in
this table, and the weights were fitted on single-seed members. CV cannot see placement. Do not
quote a board score from these.

## TDI

`14`'s shipped ensemble (`submissions/14_tdi_ensemble/20260920/tdi.csv`), copied unchanged.
"""


if __name__ == "__main__":
    main()
