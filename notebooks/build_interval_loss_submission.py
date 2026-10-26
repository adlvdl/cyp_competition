"""Build the interval-loss regression submission (22's blend, chemprop members retrained).

Run as a script, after `31_interval_final_fits.py` has finished both members:

    uv run --no-sync python notebooks/build_interval_loss_submission.py

## What changes against 22

Same five-member per-endpoint blend, same weights (read from 22's `weights.csv`, equal splits
that 21 fitted on all 25 folds), same placement. Two members are swapped for their interval-loss
refits (idea credited in README, evidence in notebooks 28, 30, 31b):

- `pubchem`           5 seeds, interval loss (`31_interval_final_fits.py pubchem`)
- `pooled_workbench`  3 seeds, interval loss on CYP1A2, CYP2C9, CYP3A4. **CYP2D6 keeps `20`'s
                      MAE seeds**: the interval model was worse there in CV (+0.026, 5 folds).

TabICL members are `22`'s seed means, untouched. TDI is `24`'s file (the six-arm encoder blend,
built and validated, unscored), copied unchanged: the TDI track is independent of this change.
"""

from __future__ import annotations

import argparse
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
from cyp import data, placement, submission  # noqa: E402

EXP = PROJECT_ROOT / "experiments"
SUBS = PROJECT_ROOT / "submissions"
SUB12 = SUBS / "12_final_placement" / "20260919" / "activity.csv"
SUB22 = SUBS / "22_pooled_seed_ensemble" / "20261023"
SUB24_TDI = SUBS / "24_tdi_encoders" / "20261023" / "tdi.csv"
SEED_CACHE = EXP / "19_seed_spread"
FITS = EXP / "31_interval_final_fits"
POOL = ("pubchem", "tabicl_monroe", "tabicl_atom_jepa", "tabicl_chemeleon", "pooled_workbench")
TABICL_SEEDS = (42, 0, 1, 2, 3)
PUBCHEM_SEEDS = range(5)
POOLED_SEEDS = range(3)
MAE_POOLED_ENDPOINT = "CYP2D6_pIC50_direct_inhibition"
#: Endpoints `12` found were hurt by placement (gated off there); `--gate` leaves them raw.
GATED = ("CYP2C9_pIC50_direct_inhibition", "CYP3A4_pIC50_direct_inhibition")


def _by_ids(frame: pl.DataFrame, endpoint: str, ids: list[str]) -> np.ndarray:
    """Align a Molecule_Name-keyed frame to the blind set's row order."""
    out = (
        pl.DataFrame({"Molecule_Name": ids})
        .join(frame.select("Molecule_Name", endpoint), on="Molecule_Name", how="left")[endpoint]
        .to_numpy()
    )
    assert np.isfinite(out).all() and len(out) == len(ids)
    return out.astype(float)


def _mean_over(files: list[Path], endpoint: str, ids: list[str]) -> np.ndarray:
    for f in files:
        assert f.exists(), f"{f} is missing; run 31_interval_final_fits.py first"
    return np.mean([_by_ids(pl.read_parquet(f), endpoint, ids) for f in files], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true", help="leave CYP2C9 and CYP3A4 unplaced")
    gate = parser.parse_args().gate
    test = data.load_test()
    ids = test["Molecule_Name"].to_list()
    endpoints = list(C.REGRESSION_ENDPOINTS)

    blind: dict[str, dict[str, np.ndarray]] = {a: {} for a in POOL}
    old_pool = [
        EXP / "20_pooled_workbench" / "final" / f"pooled_seed{s}.parquet" for s in POOLED_SEEDS
    ]
    new_pool = [FITS / "pooled" / f"pooled_interval_seed{s}.parquet" for s in POOLED_SEEDS]
    new_pub = [FITS / "pubchem" / f"pubchem_interval_seed{s}.parquet" for s in PUBCHEM_SEEDS]
    for e in endpoints:
        blind["pubchem"][e] = _mean_over(new_pub, e, ids)
        blind["pooled_workbench"][e] = _mean_over(
            old_pool if e == MAE_POOLED_ENDPOINT else new_pool, e, ids
        )
        for arm in ("tabicl_monroe", "tabicl_atom_jepa"):
            blind[arm][e] = np.mean(
                [np.load(SEED_CACHE / "tabicl_cache" / f"{arm}_{e}_{s}.npy") for s in TABICL_SEEDS],
                axis=0,
            )
        blind["tabicl_chemeleon"][e] = np.zeros(len(ids))  # weight 0, asserted below

    w = pl.read_csv(SUB22 / "weights.csv")
    weights = {
        e: dict(zip(g["model"], g["weight"], strict=True)) for (e,), g in w.group_by("endpoint")
    }
    assert all(weights[e]["tabicl_chemeleon"] == 0 for e in endpoints)

    # Members must still agree with the 22 vectors they replace: a swap that decorrelates
    # them badly would mean the refit, not the loss, changed the model.
    sub22 = pl.read_csv(SUB22 / "activity.csv")
    sub12 = pl.read_csv(SUB12)
    final: dict[str, np.ndarray] = {}
    plan_rows = []
    for e in endpoints:
        raw = sum(weights[e][a] * blind[a][e] for a in POOL)
        ref = _by_ids(sub12, e, ids)
        if gate and e in GATED:
            placed, ties = raw, 0  # the blend's own mean and spread, no affine map
        else:
            placed = placement.affine(raw, float(ref.mean()), float(ref.std()))
            ties = placement.check_rank_preserved(raw, placed)
        final[e] = placed
        rho22 = float(
            pl.DataFrame({"a": placed, "b": _by_ids(sub22, e, ids)})
            .select(pl.corr("a", "b", method="spearman"))
            .item()
        )
        assert rho22 > 0.9, (
            f"{e}: Spearman vs 22 is {rho22:.3f}, the refit moved the ordering too far"
        )
        plan_rows.append(
            {
                "endpoint": e,
                "raw_mean": round(float(raw.mean()), 3),
                "raw_sd": round(float(raw.std()), 3),
                "placed_mean": round(float(placed.mean()), 3),
                "placed_sd": round(float(placed.std()), 3),
                "floor_ties": ties,
                "spearman_vs_22": round(rho22, 3),
            }
        )
    plan = pl.DataFrame(plan_rows)
    print(plan)

    dest = (
        SUBS / "31_interval_loss" / (date.today().strftime("%Y%m%d") + ("_gated" if gate else ""))
    )
    if dest.exists():
        raise FileExistsError(f"{dest} exists; refusing to overwrite a submission folder")
    activity = submission.build_activity_submission(final, dest / "activity.csv")
    shutil.copy(SUB24_TDI, dest / "tdi.csv")
    ok_a = submission.check(activity, track="activity")
    ok_t = submission.check(dest / "tdi.csv", track="tdi")
    print(
        f"validation: activity {'PASSED' if ok_a else 'FAILED'}, "
        f"tdi {'PASSED' if ok_t else 'FAILED'}"
    )
    if not (ok_a and ok_t):
        raise SystemExit("validation failed; nothing should be submitted")
    shutil.copy(SUB22 / "weights.csv", dest / "weights.csv")
    plan.write_csv(dest / "placement_plan.csv")

    check = pl.read_csv(EXP / "31b_interval_blend_check" / "blend_check.csv")
    pc = pl.read_csv(EXP / "31b_interval_blend_check" / "placement_check.csv")
    (dest / "PROVENANCE.md").write_text(_provenance(plan, check, pc, gate))
    print(f"Wrote {dest}")


def _provenance(plan: pl.DataFrame, check: pl.DataFrame, pc: pl.DataFrame, gate: bool) -> str:
    def md(frame: pl.DataFrame) -> str:
        cols = frame.columns
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.iter_rows()]
        return "\n".join(lines)

    shown = check.select(
        "arm", "n_folds", "baseline", "candidate", "diff", "ci_low", "ci_high"
    ).with_columns(pl.col(pl.Float64).round(4))
    placement_text = (
        "**Gated.** CYP1A2 and CYP2D6 are affine-mapped onto `12`'s moments, as in\n"
        "`22`/`17`. CYP2C9 and CYP3A4 are left at the blend's own mean and spread (`12` found\n"
        "placement cost those two endpoints 0.041 and 0.114 ST-RAE, and gating them kept their\n"
        "scores). Reason: mapping the OOF blend onto the baseline moments erases about 85% of the\n"
        "interval gain (table below)."
        if gate
        else "Identical to `22`/`17`. Affine onto the moments of `12`'s vector, no rank changes."
    )
    pc_shown = pc.with_columns(pl.col(pl.Float64).round(4))
    return f"""# 22's blend with interval-loss chemprop members + 24's TDI ensemble

- Generated: {date.today().isoformat()}
- Script: `notebooks/build_interval_loss_submission.py`
- Evidence: notebooks `28` (loss), `30` (on `pubchem` and pooled), `31_interval_final_fits`
  (refits), `31b` (blend)
- Idea source: mokhalaf22 and doctawho42 reports, see README "Ideas from other entrants".
  No code used.
- Challenge data snapshot: `20261020`
- **Not yet scored on the board.**

## Regression model

`22`'s five-member per-endpoint blend with the same weights (equal splits, `weights.csv`); placement
is described below. Two members are retrained with `--loss-function interval-mae`, which is zero
inside a compound's credible interval and linear outside it (ST-RAE's own shape):

- `pubchem`: 5 seeds, same pretrained checkpoint and recipe as `19`
- `pooled_workbench`: 3 seeds, CYP1A2 / CYP2C9 / CYP3A4. **CYP2D6 keeps the MAE seeds from `20`**,
  because the interval model was worse there in CV (+0.026 on 5 folds, untested for significance).
- TabICL members: `22`'s seed means, unchanged.

The loss rebuilds each cell's band from its own pIC50 (isotonic lookup, R2 0.84-0.98), because the
training file carries no bounds. Weights were not refit: they were fitted on the MAE members' OOF.

## Placement

{placement_text}

{md(plan)}

## Expected performance (CV, macro ST-RAE, 22's weights, members swapped)

{md(shown)}

Placement check (25 folds, `pubchem` swapped): the swapped blend scored raw and after being
mapped onto the baseline blend's mean and sd.

{md(pc_shown)}

Rows A cover all 25 folds with only `pubchem` swapped. Rows B cover folds 0-4, the only folds where
a pooled interval OOF exists. Both are against the blend as shipped in `22`. The weights were fitted
on the baseline members, which favours the baseline. CV cannot see placement, and these submissions
already apply an affine placement, so part of the CV gain may not appear on the board. Spearman did
not move in `28`/`30`: the gain is position relative to the bands, not ordering.

## TDI

`24`'s six-arm encoder blend (`submissions/24_tdi_encoders/20261023/tdi.csv`), copied unchanged. It
has not been scored; `14`'s and `23`'s CV gains both shrank on the board.
"""


if __name__ == "__main__":
    main()
