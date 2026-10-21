"""Build the pool-B regression submission from notebooks 16, 17 and 18.

Run as a script, not a notebook:

    uv run python notebooks/build_pool_b_submission.py

## What this ships

Regression: a Caruana blend of four models, per endpoint --

- `pubchem`           `05`'s Chemprop D-MPNN, PubChem-pretrained (the board-scored model)
- `tabicl_monroe`     TabICL on frozen Monroe embeddings
- `tabicl_atom_jepa`  TabICL on frozen Atom-JEPA embeddings (last block)
- `tabicl_chemeleon`  TabICL on frozen CheMeleon embeddings

TDI: `14`'s shipped ensemble, copied unchanged. Nothing in 16-18 touches the TDI track.

## Final fit, not CV

Each TabICL arm is fitted on 100% of the labelled rows for its endpoint and predicts the
750 blind compounds. `pubchem`'s blind predictions are `05`'s shipped, unplaced vector. The
Caruana weights are refit on all 25 folds of OOF (a submission must use every fold, as it
uses every compound); 17's held-out rotation weights are printed beside them so any drift is
visible. Deliberately not cached: the final fit is cheap next to CV and a stale cache has
already misled this repo once.

## Placement

CV is blind to placement (CLAUDE.md), and `05` unplaced scored CYP2D6 at ST-RAE 1.31 against
0.89 once placed. `12`'s per-endpoint plan was validated on the board three times, so this
script carries it over rather than re-solving: each endpoint's blend is affine-mapped onto
the mean and sd of the vector `12` actually shipped for that endpoint (CYP1A2 placed,
CYP2D6 `07`'s vector, CYP2C9 and CYP3A4 `05` unplaced). An affine map moves no compound's
rank, so the new ordering is the blend's. The assumption this carries is that the blend's
best spread is close to `pubchem`'s; the blend correlates slightly better, so its optimal
spread is slightly wider, and the effect is small.
"""

from __future__ import annotations

import csv
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cyp import constants as C  # noqa: E402
from cyp import data, encoders, ensemble, evaluation, placement, submission  # noqa: E402
from cyp.metrics import st_rae  # noqa: E402
from cyp.tabular_models import predict_subprocess  # noqa: E402

POOL = ("pubchem", "tabicl_monroe", "tabicl_atom_jepa", "tabicl_chemeleon")
KEYS = ["endpoint", "fold", "Molecule_Name"]

EXP = PROJECT_ROOT / "experiments"
SUB05 = PROJECT_ROOT / "submissions" / "05_auxiliary_data" / "20260915" / "activity.csv"
SUB12 = PROJECT_ROOT / "submissions" / "12_final_placement" / "20260919" / "activity.csv"
SUB14_TDI = PROJECT_ROOT / "submissions" / "14_tdi_ensemble" / "20260920" / "tdi.csv"
BLIND_CACHE = EXP / "17_regression_ensemble" / "blind_embeddings"


def _blind_embedding(kind: str, smiles: list[str]) -> np.ndarray:
    """Blind-set embedding. An input, not a fit, so caching it is safe."""
    BLIND_CACHE.mkdir(parents=True, exist_ok=True)
    path = BLIND_CACHE / f"{kind}.npy"
    if path.exists():
        out = np.load(path)
        assert out.shape[0] == len(smiles)
        return out
    if kind == "chemeleon":
        from cyp.graph_models import chemeleon_embed

        out, _ = chemeleon_embed(smiles, smiles[:1])
    else:
        out = encoders.embed(kind, smiles, log_path=BLIND_CACHE / f"{kind}.log")
    np.save(path, out)
    return out


def main() -> None:
    test = data.load_test()
    test_smiles = test["SMILES"].to_list()
    test_ids = test["Molecule_Name"].to_list()
    endpoints = list(C.REGRESSION_ENDPOINTS)
    frames = {e: data.training_frame(e) for e in endpoints}
    print(f"Fitting on all labelled rows; predicting {len(test_smiles)} blind compounds.")

    # ── Features ────────────────────────────────────────────────────────────────
    union = sorted({s for f in frames.values() for s in f["SMILES"].to_list()})
    row_of = {s: i for i, s in enumerate(union)}
    cache16 = EXP / "16_encoders" / "cache"
    train_tables = {
        "tabicl_monroe": encoders.embed_cached("monroe", union, cache16),
        "tabicl_atom_jepa": encoders.embed_cached("atom_jepa", union, cache16)[:, -1, :],
    }
    blind_feats = {
        "tabicl_monroe": _blind_embedding("monroe", test_smiles),
        "tabicl_atom_jepa": _blind_embedding("atom_jepa", test_smiles)[:, -1, :],
        "tabicl_chemeleon": _blind_embedding("chemeleon", test_smiles),
    }

    def train_features(arm: str, endpoint: str) -> np.ndarray:
        frame = frames[endpoint]
        if arm == "tabicl_chemeleon":
            x = np.load(EXP / "03_methods" / "cache" / f"chemeleon_{endpoint}.npy")
            assert x.shape[0] == frame.height
            return x
        return train_tables[arm][[row_of[s] for s in frame["SMILES"].to_list()]]

    # ── Final fits ──────────────────────────────────────────────────────────────
    blind: dict[str, dict[str, np.ndarray]] = {arm: {} for arm in POOL}
    sub05 = pl.read_csv(SUB05).sort("Molecule_Name")
    assert sub05["Molecule_Name"].to_list() == sorted(test_ids), "05 submission ids differ"
    for endpoint in endpoints:
        blind["pubchem"][endpoint] = (
            pl.DataFrame({"Molecule_Name": test_ids})
            .join(sub05.select("Molecule_Name", endpoint), on="Molecule_Name", how="left")[endpoint]
            .to_numpy()
        )

    for i, arm in enumerate(POOL[1:], start=1):
        print(f"\n[{i}/3] {arm}")
        for endpoint in endpoints:
            frame = frames[endpoint]
            pred, _, _ = predict_subprocess(
                train_features(arm, endpoint),
                frame["y_true"].to_numpy(),
                blind_feats[arm],
                kind="tabicl",
            )
            pred = np.asarray(pred, dtype=float)
            assert np.isfinite(pred).all(), f"{arm}/{endpoint}: non-finite predictions"
            blind[arm][endpoint] = pred
            print(f"  {endpoint}: n={frame.height}  mean {pred.mean():.3f}  sd {pred.std():.3f}")

    # ── Sanity: member agreement on the blind set vs in OOF ─────────────────────
    # A misaligned blind embedding (wrong row order, wrong encoder) shows up as near-zero
    # correlation with `pubchem`; OOF correlations are 0.7-0.9.
    print("\nPearson r with `pubchem` on the blind set:")
    for arm in POOL[1:]:
        rs = {
            e.split("_")[0]: np.corrcoef(blind[arm][e], blind["pubchem"][e])[0, 1]
            for e in endpoints
        }
        print(f"  {arm:18s} " + "  ".join(f"{k} {v:.3f}" for k, v in rs.items()))
        assert min(rs.values()) > 0.4, f"{arm}: blind predictions barely correlate with pubchem"

    # ── Weights: Caruana, refit on all 25 folds ─────────────────────────────────
    oof16 = pl.read_parquet(EXP / "16_encoders" / "oof.parquet")
    oof05 = pl.read_parquet(EXP / "05_auxiliary_data" / "oof.parquet").filter(
        pl.col("method") == "pubchem"
    )
    rotation = pl.read_csv(
        EXP / "17_regression_ensemble" / "caruana_weights_by_rotation.csv"
    ).filter(pl.col("pool") == "B")

    weights: dict[str, dict[str, float]] = {}
    print("\nCaruana weights (all 25 folds) vs 17's held-out mean:")
    for endpoint in endpoints:
        base = oof05.filter(pl.col("endpoint") == endpoint).sort(KEYS[1:])
        cands = {"pubchem": base["y_pred"].to_numpy()}
        for arm in POOL[1:]:
            part = oof16.filter((pl.col("method") == arm) & (pl.col("endpoint") == endpoint)).sort(
                KEYS[1:]
            )
            assert (part["Molecule_Name"] == base["Molecule_Name"]).all() and (
                part["fold"] == base["fold"]
            ).all()
            cands[arm] = part["y_pred"].to_numpy()
        lo, hi, y = (
            base["y_lower"].to_numpy(),
            base["y_upper"].to_numpy(),
            base["y_true"].to_numpy(),
        )
        w = ensemble.caruana_selection(
            cands,
            y,
            lambda t, p, lo=lo, hi=hi: st_rae(t, p, lo, hi),
            maximize=False,
            n_bags=1,
            bag_frac=1.0,
            max_iter=50,
        )
        weights[endpoint] = w
        held = (
            rotation.filter(pl.col("endpoint") == endpoint)
            .group_by("model")
            .agg(pl.col("weight").mean())
        )
        held_d = dict(zip(held["model"], held["weight"], strict=True))
        print(
            f"  {endpoint.split('_')[0]}: "
            + "  ".join(f"{a} {w[a]:.3f} ({held_d.get(a, 0.0):.3f})" for a in POOL)
        )

    # ── Blend, then place onto 12's validated moments ───────────────────────────
    sub12 = pl.read_csv(SUB12).sort("Molecule_Name")
    ref12 = {
        e: pl.DataFrame({"Molecule_Name": test_ids})
        .join(sub12.select("Molecule_Name", e), on="Molecule_Name", how="left")[e]
        .to_numpy()
        for e in endpoints
    }

    final: dict[str, np.ndarray] = {}
    plan_rows = []
    for endpoint in endpoints:
        raw = sum(weights[endpoint][a] * blind[a][endpoint] for a in POOL)
        target_mean, target_sd = float(ref12[endpoint].mean()), float(ref12[endpoint].std())
        placed = placement.affine(raw, target_mean, target_sd)
        ties = placement.check_rank_preserved(raw, placed)
        final[endpoint] = placed
        rho = float(
            pl.DataFrame({"a": placed, "b": ref12[endpoint]})
            .select(pl.corr("a", "b", method="spearman"))
            .item()
        )
        plan_rows.append(
            {
                "endpoint": endpoint,
                "raw_mean": round(float(raw.mean()), 3),
                "raw_sd": round(float(raw.std()), 3),
                "placed_mean": round(float(placed.mean()), 3),
                "placed_sd": round(float(placed.std()), 3),
                "floor_ties": ties,
                "spearman_vs_12": round(rho, 3),
            }
        )
    plan = pl.DataFrame(plan_rows)
    print("\nPlacement onto 12's moments:")
    print(plan)

    # ── Write + validate ────────────────────────────────────────────────────────
    dest = PROJECT_ROOT / "submissions" / "17_regression_ensemble" / date.today().strftime("%Y%m%d")
    if dest.exists():
        raise FileExistsError(f"{dest} exists; refusing to overwrite a submission folder")
    activity = submission.build_activity_submission(final, dest / "activity.csv")
    shutil.copy(SUB14_TDI, dest / "tdi.csv")
    ok_activity = submission.check(activity, track="activity")
    ok_tdi = submission.check(dest / "tdi.csv", track="tdi")
    verdict = {True: "PASSED", False: "FAILED"}
    print(f"\nvalidation: activity {verdict[ok_activity]}, tdi {verdict[ok_tdi]}")
    if not (ok_activity and ok_tdi):
        raise SystemExit("validation failed; nothing should be submitted")

    rows = [
        {"endpoint": e, "model": a, "weight": round(weights[e][a], 4)}
        for e in endpoints
        for a in POOL
    ]
    with (dest / "weights.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["endpoint", "model", "weight"])
        writer.writeheader()
        writer.writerows(rows)
    plan.write_csv(dest / "placement_plan.csv")

    # ── Expected performance (CV; placement-blind) ──────────────────────────────
    ens = pl.read_parquet(EXP / "17_regression_ensemble" / "ensemble_oof.parquet")
    exp_rows = []
    for label, frame in (
        ("pool B (held-out weights)", ens.filter(pl.col("method") == "caruana_B")),
        (
            "pubchem (05, shipped model)",
            oof05.select(*KEYS, "y_true", "y_lower", "y_upper", pl.col("y_pred")).with_columns(
                pl.lit("pubchem").alias("method")
            ),
        ),
    ):
        folds = evaluation.fold_metrics(frame.with_columns(pl.lit(label).alias("method")))
        per = folds.group_by("endpoint").agg(pl.col("st_rae").mean())
        macro = evaluation.macro_averaged_fold_metrics(
            {e: folds.filter(pl.col("endpoint") == e) for e in endpoints}, metric_col="st_rae"
        )["st_rae"].mean()
        for r in per.iter_rows(named=True):
            exp_rows.append(
                {"model": label, "endpoint": r["endpoint"], "cv_st_rae": round(r["st_rae"], 4)}
            )
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
        [
            {"endpoint": e.split("_")[0], **{a: round(weights[e][a], 3) for a in POOL}}
            for e in endpoints
        ]
    )
    return f"""# Pool B regression ensemble + 14's TDI ensemble

- Generated: {date.today().isoformat()}
- Script: `notebooks/build_pool_b_submission.py`
- Evidence: notebooks `16` (encoders), `17` (ensemble pools), `18` (shift check)
- Challenge data snapshot: `20261020` (byte-identical to `20260908` for every data file;
  only the dataset README changed, to add a separate structure track)
- CV: nested scaffold 5x5, shared folds, `experiments/16_encoders/oof.parquet` and
  `experiments/05_auxiliary_data/oof.parquet`
- **Not yet scored on the board.**

## Regression model

Per-endpoint Caruana blend of four models, each fitted on 100% of the labelled rows:

- `pubchem`: `05`'s PubChem-pretrained Chemprop (its shipped, unplaced blind predictions)
- `tabicl_monroe`, `tabicl_atom_jepa`, `tabicl_chemeleon`: TabICL (PCA to 128, 2 estimators,
  seed 42, the `03` readout) on frozen embeddings of Monroe, Atom-JEPA (last block) and CheMeleon

Weights are refit on all 25 folds of OOF:

{md(w_table)}

Pool B was chosen over pool A (2 models) and pool C (20 models) in `17` and `18`: C beat B
in plain CV (-0.009 macro) but not on the lowest-similarity slice of the OOF or after
reweighting toward the blind set's novelty and predicted potency, where C-vs-B held in only
69-73% of resamples. Pools A and B were picked after seeing `16`, so their absolute gain over
`pubchem` is mildly flattered.

## Placement

Each endpoint's blend is affine-mapped onto the mean and sd of the vector `12` shipped (and
the board scored) for that endpoint. This moves no compound's rank.

{md(plan)}

CYP1A2 and CYP2D6 carry `12`'s placement, CYP2C9 and CYP3A4 `05`'s unplaced moments (gated at
recoverable R2 < 0.05). The assumption: the blend's best spread is close to `pubchem`'s.
The blend correlates slightly better, so its optimal spread is slightly wider.

## Expected performance (CV, placement-blind)

{md(expected)}

CV cannot see placement, and the board-to-CV ratio ran 0.8-1.4 per endpoint on earlier
submissions (`07`). Do not quote a board score from these. Pool B's column uses 17's
held-out weights; the shipped weights use every fold.

## TDI

`14`'s shipped ensemble (`submissions/14_tdi_ensemble/20260920/tdi.csv`), copied unchanged.
"""


if __name__ == "__main__":
    main()
