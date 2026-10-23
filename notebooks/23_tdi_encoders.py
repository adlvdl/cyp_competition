"""23: do the regression-side gains (Monroe/Atom-JEPA encoders, decorrelated blend members) transfer to TDI?

Run as a script:

    uv run python notebooks/23_tdi_encoders.py

`16` found TabICL on frozen Monroe embeddings beats the CheMeleon readout on all four
regression endpoints, and `17`/`20`/`21` found that blend members help by being
decorrelated, not by being strong. `14`'s TDI ensemble is TabICL/CheMeleon plus three
Chemprop variants. Two testable transfers, nothing else changed:

  1. Encoder swap: TabICL (single-task, PCA 128, 2 estimators, seed 42 -- `04`'s readout)
     on Monroe and on Atom-JEPA instead of CheMeleon.
  2. Pool growth: do those two arms improve `14`'s Caruana blend when added to it.

Same folds as `13`/`14` (`cv.shared_scaffold_folds`, seed 42), threshold fixed at 0.25
as in `14`, weights fitted on four outer repeats and scored on the fifth, rotated.
`13`'s OOF is reused for the four existing arms; alignment is asserted.

Family for the Holm correction (three planned comparisons, MCC at 0.25):
  tabicl_monroe vs tabicl_singletask, tabicl_atom_jepa vs tabicl_singletask,
  caruana (6 arms) vs caruana (4 arms, = 14's result).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cyp  # noqa: E402,F401
from cyp import constants as C  # noqa: E402
from cyp import cv, data, encoders, ensemble, evaluation, multitask  # noqa: E402
from sklearn.metrics import matthews_corrcoef  # noqa: E402

EXP = PROJECT_ROOT / "experiments"
OUT = EXP / "23_tdi_encoders"
CACHE = OUT / "cache"
THRESHOLD = 0.25
KEYS = ["endpoint", "fold", "Molecule_Name"]
NEW_ARMS = ("tabicl_monroe", "tabicl_atom_jepa")
OLD_ARMS = ("tabicl_singletask", "chemprop_scratch", "chemprop_pubchem", "chemprop_pubchem_frozen")


def tdi_features(frames: dict[str, pl.DataFrame]) -> dict[str, dict[str, np.ndarray]]:
    """Per-endpoint Monroe / Atom-JEPA matrices, row-aligned to each TDI frame.

    Most TDI compounds are already in 16's regression-union cache; only the remainder
    is embedded here, so this costs minutes rather than the hour a full pass took.
    """
    reg_union = sorted(
        {s for e in C.REGRESSION_ENDPOINTS for s in data.training_frame(e)["SMILES"].to_list()}
    )
    cache16 = EXP / "16_encoders" / "cache"
    tdi_smiles = sorted({s for f in frames.values() for s in f["SMILES"].to_list()})
    missing = [s for s in tdi_smiles if s not in set(reg_union)]
    print(f"{len(tdi_smiles)} TDI compounds, {len(missing)} not in the regression cache", flush=True)

    out: dict[str, dict[str, np.ndarray]] = {}
    for kind, arm in (("monroe", "tabicl_monroe"), ("atom_jepa", "tabicl_atom_jepa")):
        reg_table = encoders.embed_cached(kind, reg_union, cache16)
        # Atom-JEPA raises on a molecule whose conformer generation fails. The regression
        # union embedded cleanly, but these 1,240 are new, so drop the offender named in
        # the error, retry, and give it the mean new-compound embedding below. Counted
        # and printed, never silent.
        bad_file = CACHE / f"{kind}_bad.json"
        bad: list[str] = json.loads(bad_file.read_text()) if bad_file.exists() else []
        while True:
            good = [s for s in missing if s not in bad]
            try:
                new_table = encoders.embed_cached(kind, good, CACHE) if good else None
                break
            except RuntimeError as err:
                hit = re.search(r"RuntimeError: '(.*)': embedded 0", str(err))
                if kind != "atom_jepa" or hit is None:
                    raise
                bad.append(hit.group(1))
                print(f"{kind}: conformer failure #{len(bad)}: {hit.group(1)[:60]}", flush=True)
        if bad:
            bad_file.write_text(json.dumps(bad))
        if kind == "atom_jepa":
            reg_table = reg_table[:, -1, :]
            new_table = new_table[:, -1, :] if new_table is not None else None
        print(f"{kind}: {len(bad)} compounds imputed with the mean embedding", flush=True)
        row = {s: ("reg", i) for i, s in enumerate(reg_union)}
        row.update({s: ("new", i) for i, s in enumerate(good)})
        impute = new_table.mean(axis=0) if new_table is not None else None
        out[arm] = {}
        for endpoint, frame in frames.items():
            rows = [
                impute
                if s in bad
                else (reg_table if row[s][0] == "reg" else new_table)[row[s][1]]
                for s in frame["SMILES"].to_list()
            ]
            out[arm][endpoint] = np.vstack(rows)
    return out


def run_new_arms() -> pl.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = {f"{iso}_is_TDI": data.tdi_training_frame(iso) for iso in C.TDI_ISOFORMS}
    _, assignments = cv.shared_scaffold_folds(frames, n_outer=5, n_inner=5, seed=42)
    n_folds = assignments["fold"].n_unique()
    feats = tdi_features(frames)
    pieces = []
    # Fold-major, cached per (arm, fold): a killed run leaves a usable paired comparison.
    for fold in range(n_folds):
        for arm in NEW_ARMS:
            path = CACHE / f"{arm}_fold{fold:02d}.parquet"
            if not path.exists():
                part = multitask.run_cv_singletask_classification(
                    frames,
                    method="tabicl",
                    features=feats[arm],
                    assignments=assignments,
                    folds=[fold],
                    seed=42,
                ).with_columns(pl.lit(arm).alias("method"))
                part.write_parquet(path)
            pieces.append(pl.read_parquet(path))
        print(f"fold {fold + 1}/{n_folds}", flush=True)
    return pl.concat(pieces)


def mcc(y: np.ndarray, p: np.ndarray) -> float:
    return float(matthews_corrcoef(y, p >= THRESHOLD))


def rotate(oof: pl.DataFrame, arms: list[str], tag: str) -> pl.DataFrame:
    """Caruana and uniform blends with weights fitted off the held-out outer repeat."""
    parts, weights = [], []
    for endpoint in sorted(oof["endpoint"].unique().to_list()):
        ep = oof.filter(pl.col("endpoint") == endpoint)
        for held in sorted(ep["outer_fold"].unique().to_list()):

            def probs(frame: pl.DataFrame) -> dict[str, np.ndarray]:
                return {
                    a: frame.filter(pl.col("method") == a).sort(KEYS)["y_prob"].to_numpy()
                    for a in arms
                }

            fit, test = ep.filter(pl.col("outer_fold") != held), ep.filter(pl.col("outer_fold") == held)
            y_fit = fit.filter(pl.col("method") == arms[0]).sort(KEYS)["y_true"].to_numpy()
            w = ensemble.caruana_selection(probs(fit), y_fit, mcc, maximize=True, seed=0)
            weights.extend(
                {"pool": tag, "endpoint": endpoint, "held_out": held, "model": a, "weight": w[a]}
                for a in arms
            )
            tp = probs(test)
            base = test.filter(pl.col("method") == arms[0]).sort(KEYS).select(
                *KEYS, "outer_fold", "y_true"
            )
            parts.append(
                base.with_columns(
                    pl.Series("y_prob", sum(w[a] * tp[a] for a in arms)),
                    pl.lit(f"caruana_{tag}").alias("method"),
                )
            )
            parts.append(
                base.with_columns(
                    pl.Series("y_prob", np.mean([tp[a] for a in arms], axis=0)),
                    pl.lit(f"uniform_{tag}").alias("method"),
                )
            )
    pl.DataFrame(weights).write_csv(OUT / f"caruana_weights_{tag}.csv")
    return pl.concat(parts)


def per_repeat_mcc(oof: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for (method, endpoint, held), g in oof.group_by(["method", "endpoint", "outer_fold"]):
        rows.append(
            {"method": method, "endpoint": endpoint, "outer_fold": held,
             "mcc": mcc(g["y_true"].to_numpy(), g["y_prob"].to_numpy())}
        )
    return pl.DataFrame(rows)


def stacked(oof: pl.DataFrame, method: str) -> tuple[np.ndarray, np.ndarray]:
    g = oof.filter(pl.col("method") == method).sort(KEYS)
    return g["y_true"].to_numpy(), g["y_prob"].to_numpy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    new = run_new_arms()
    old = pl.read_parquet(EXP / "13_tdi_pretrained" / "oof.parquet")
    cols = ["method", "endpoint", "fold", "outer_fold", "Molecule_Name", "y_true", "y_prob"]
    # Alignment: every arm must cover the same (endpoint, fold, compound) rows with the same truth.
    ref = old.filter(pl.col("method") == "tabicl_singletask").sort(KEYS)
    for arm in NEW_ARMS:
        g = new.filter(pl.col("method") == arm).sort(KEYS)
        assert g.height == ref.height, (arm, g.height, ref.height)
        assert (g.select(KEYS) == ref.select(KEYS)).to_numpy().all(), f"{arm}: rows misaligned"
        assert (g["y_true"].cast(pl.Int64) == ref["y_true"].cast(pl.Int64)).all(), f"{arm}: truth differs"
        assert g["y_prob"].is_nan().sum() == 0, f"{arm}: NaN probabilities (numpy/TabICL trap)"
    oof = pl.concat([old.select(cols), new.select(cols)])
    oof.write_parquet(OUT / "oof.parquet")

    # Correlation of predictions between members, per endpoint
    arms_all = [*OLD_ARMS, *NEW_ARMS]
    corr = []
    for e in sorted(oof["endpoint"].unique().to_list()):
        wide = {a: oof.filter((pl.col("method") == a) & (pl.col("endpoint") == e)).sort(KEYS)["y_prob"].to_numpy()
                for a in arms_all}
        m = np.corrcoef([wide[a] for a in arms_all])
        corr.append(pl.DataFrame({"endpoint": e, "arm": arms_all, **{a: m[:, i] for i, a in enumerate(arms_all)}}))
    pl.concat(corr).write_csv(OUT / "prediction_correlation.csv")

    pool0 = rotate(oof, list(OLD_ARMS), "P0")  # reproduces 14
    pool1 = rotate(oof, [*OLD_ARMS, *NEW_ARMS], "P1")
    pool2 = rotate(oof, ["tabicl_singletask", *NEW_ARMS, "chemprop_pubchem", "chemprop_pubchem_frozen"], "P2")
    singles = oof.select("method", "endpoint", "outer_fold", "Molecule_Name", "fold", "y_true", "y_prob")
    allp = pl.concat(
        [singles.select(pool0.columns), pool0, pool1.select(pool0.columns), pool2.select(pool0.columns)]
    )
    pr = per_repeat_mcc(allp)
    summary = (
        pr.group_by(["method", "endpoint"]).agg(pl.col("mcc").mean())
        .pivot(values="mcc", index="method", on="endpoint")
    )
    ends = [c for c in summary.columns if c != "method"]
    summary = summary.with_columns(pl.mean_horizontal(ends).alias("macro")).sort("macro", descending=True)
    summary.write_csv(OUT / "summary.csv")
    with pl.Config(tbl_rows=30, tbl_cols=10, tbl_width_chars=160):
        print(summary)

    # Paired comparisons (pooled held-out rows, MCC at THRESHOLD)
    def cmp(a: str, b: str) -> dict:
        ya, pa = stacked(allp, a)
        yb, pb = stacked(allp, b)
        assert (ya == yb).all()
        r = evaluation.paired_bootstrap(ya, pa >= THRESHOLD, pb >= THRESHOLD, metric="mcc", seed=0)
        return {"candidate": a, "vs": b, "diff": r["diff"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p": r["p_value"]}

    family = [
        cmp("tabicl_monroe", "tabicl_singletask"),
        cmp("tabicl_atom_jepa", "tabicl_singletask"),
        cmp("caruana_P1", "caruana_P0"),
    ]
    holm = evaluation.holm_bonferroni({f"{r['candidate']} vs {r['vs']}": r["p"] for r in family})
    fam = pl.DataFrame(family).with_columns(
        pl.Series("comparison", [f"{r['candidate']} vs {r['vs']}" for r in family])
    ).join(holm.select("comparison", "threshold", "significant"), on="comparison")
    extra = pl.DataFrame([cmp("caruana_P2", "caruana_P0"), cmp("uniform_P1", "uniform_P0"), cmp("caruana_P1", "tabicl_singletask")])
    fam.write_csv(OUT / "paired_family.csv")
    extra.write_csv(OUT / "paired_descriptive.csv")
    with pl.Config(tbl_cols=12, tbl_width_chars=200):
        print(fam)
        print(extra)

    # Threshold sweep for the new single arms (informational: is 0.25 right for them?)
    sweep = []
    for a in ("tabicl_singletask", *NEW_ARMS):
        for t in np.arange(0.05, 0.55, 0.05):
            vals = []
            for e in sorted(oof["endpoint"].unique().to_list()):
                y, p = stacked(oof.filter(pl.col("endpoint") == e), a)
                vals.append(float(matthews_corrcoef(y, p >= t)))
            sweep.append({"arm": a, "threshold": round(float(t), 2), "macro_mcc": float(np.mean(vals))})
    sw = pl.DataFrame(sweep)
    sw.write_csv(OUT / "threshold_sweep.csv")
    print(sw.sort("macro_mcc", descending=True).group_by("arm").head(1))


if __name__ == "__main__":
    main()
