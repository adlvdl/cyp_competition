"""29: does `27`'s A1 rule (train on both-arms rows only) survive the Caruana blend?

Run as a script:

    uv run python notebooks/29_tdi_blend_a1.py

`27` found that dropping the 1,249 no-direct-arm CYP3A4 rows (all TDI-negative) improves a
single TabICL/Monroe readout by +0.033 macro MCC on both-arms rows, the population the blind
set resembles. Idea credited in README.md ("Ideas from other entrants"). This notebook asks
whether that gain is still there once the readout is one member of the six-arm blend `23` built.

Pools, both scored on both-arms rows with the same folds as 13/14/23/27:
  PB  `23`'s six arms exactly as trained: TabICL on CheMeleon / Monroe / Atom-JEPA plus the
      three Chemprop variants, all fitted on every labelled row.
  PA  the three TabICL arms refitted with A1 (both-arms rows only, PCA fitted on those rows,
      128 PCs, 2 estimators, seed 42). The Monroe arm is `27`'s `a1`. The three Chemprop arms
      are `13`'s OOF unchanged: refitting them under A1 costs hours, so PA is a *mixed* pool
      and any gain is a lower bound on a full A1 rebuild.

Blends: Caruana (bagged forward selection) and a uniform mean. The objective is MCC at the
best threshold on a grid, and the threshold is chosen from the blended fit predictions, so
neither pool is tied to a fixed 0.25 that A1 moved. Weights and threshold are fitted on four
outer repeats and applied to the fifth, rotated. Fold is the unit of the bootstrap (see `15`).

Holm family: caruana_PA vs caruana_PB, uniform_PA vs uniform_PB.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sklearn.metrics import matthews_corrcoef, roc_auc_score  # noqa: E402

import cyp  # noqa: E402,F401  (must come first: claims the OpenMP slot, see CLAUDE.md)
from cyp import constants as C  # noqa: E402
from cyp import cv, data, ensemble, evaluation  # noqa: E402
from cyp.tabular_models import predict_subprocess  # noqa: E402

EXP = PROJECT_ROOT / "experiments"
OUT = EXP / "29_tdi_blend_a1"
CACHE = OUT / "cache"
KEYS = ["endpoint", "fold", "Molecule_Name"]
CHEMPROP_ARMS = ("chemprop_scratch", "chemprop_pubchem", "chemprop_pubchem_frozen")
TABICL_ARMS = (
    "tabicl_singletask",
    "tabicl_monroe",
    "tabicl_atom_jepa",
)  # CheMeleon, Monroe, Atom-JEPA
NEW_A1_ARMS = ("tabicl_singletask", "tabicl_atom_jepa")  # Monroe's a1 comes from 27


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "notebooks" / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nb27 = load("nb27", "27_tdi_prior_and_gate.py")
GRID = nb27.GRID


def mcc_grid(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """MCC at every threshold in GRID at once (vectorised; Caruana calls this thousands of times)."""
    pred = p[:, None] >= GRID[None, :]
    pos = y.astype(bool)[:, None]
    tp, fp = (pred & pos).sum(0), (pred & ~pos).sum(0)
    fn, tn = (~pred & pos).sum(0), (~pred & ~pos).sum(0)
    den = np.sqrt(float(1) * (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1), 0.0)


def best_mcc(y: np.ndarray, p: np.ndarray) -> float:
    return float(mcc_grid(y, p).max())


def run_new_a1_arms() -> pl.DataFrame:
    """TabICL on CheMeleon and Atom-JEPA under A1, fold-major, cached per (arm, endpoint, fold)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = {f"{iso}_is_TDI": data.tdi_training_frame(iso) for iso in C.TDI_ISOFORMS}
    _, assignments = cv.shared_scaffold_folds(frames, n_outer=5, n_inner=5, seed=nb27.SEED)
    n_folds = assignments["fold"].n_unique()
    atom = load("nb23", "23_tdi_encoders.py").tdi_features(frames)["tabicl_atom_jepa"]
    feats = {
        "tabicl_atom_jepa": atom,
        "tabicl_singletask": {
            e: np.load(EXP / "04_methods_tdi" / "cache" / f"chemeleon_tdi_{e.split('_')[0]}.npy")
            for e in frames
        },
    }
    for e, f in frames.items():
        assert feats["tabicl_singletask"][e].shape[0] == f.height, f"{e}: CheMeleon rows misaligned"
    y_direct = {e: nb27.direct_pic50(e.split("_")[0], f) for e, f in frames.items()}
    pieces = []
    for fold in range(n_folds):
        for endpoint, frame in frames.items():
            both = ~np.isnan(y_direct[endpoint])
            indexed = frame.with_row_index("_row")
            _, outer, _, train_all, _, test_all = next(
                s
                for s in cv.fold_assignment_splits(indexed, assignments, p_val=0.0, seed=nb27.SEED)
                if s[0] == fold
            )
            train_rows, test_rows = train_all["_row"].to_numpy(), test_all["_row"].to_numpy()
            fit_rows = train_rows[both[train_rows]]
            y = frame["y_true"].cast(pl.Int8).to_numpy()
            for arm in NEW_A1_ARMS:
                path = CACHE / f"{arm}_a1_{endpoint}_fold{fold:02d}.parquet"
                if not path.exists():
                    Z = nb27.fit_pcs(feats[arm][endpoint], fit_rows)
                    prob, _, _ = predict_subprocess(
                        Z[fit_rows],
                        y[fit_rows],
                        Z[test_rows],
                        kind="tabicl",
                        task="classification",
                        seed=nb27.SEED,
                        tabicl_max_features=Z.shape[1],
                    )
                    assert not np.isnan(prob).any(), f"{arm}: NaN probabilities (numpy/TabICL trap)"
                    pl.DataFrame(
                        {
                            "method": f"{arm}_a1",
                            "endpoint": endpoint,
                            "fold": fold,
                            "outer_fold": outer,
                            "Molecule_Name": test_all["Molecule_Name"],
                            "y_true": test_all["y_true"],
                            "y_prob": np.asarray(prob, dtype=float),
                        }
                    ).write_parquet(path)
                pieces.append(pl.read_parquet(path))
        print(f"fold {fold + 1}/{n_folds}", flush=True)
    return pl.concat(pieces)


def assemble() -> tuple[pl.DataFrame, list[str], list[str]]:
    """All arms on both-arms rows, one long frame, alignment asserted."""
    new = run_new_a1_arms()
    cols = ["method", "endpoint", "fold", "outer_fold", "Molecule_Name", "y_true", "y_prob"]
    old = pl.read_parquet(EXP / "23_tdi_encoders" / "oof.parquet")
    a1_monroe = (
        pl.read_parquet(OUT.parent / "27_tdi_prior_and_gate" / "oof_all_rows.parquet")
        .filter(pl.col("method") == "a1")
        .with_columns(pl.lit("tabicl_monroe_a1").alias("method"))
    )
    frames = {f"{iso}_is_TDI": data.tdi_training_frame(iso) for iso in C.TDI_ISOFORMS}
    keep = {
        e: set(
            f.filter(pl.Series(~np.isnan(nb27.direct_pic50(e.split("_")[0], f))))[
                "Molecule_Name"
            ].to_list()
        )
        for e, f in frames.items()
    }

    # The three sources were written by different notebooks with different integer widths.
    def norm(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.select(cols).with_columns(
            pl.col("fold").cast(pl.Int64),
            pl.col("outer_fold").cast(pl.Int64),
            pl.col("y_true").cast(pl.Boolean),
        )

    oof = pl.concat([norm(old), norm(new), norm(a1_monroe)])
    oof = pl.concat(
        [
            oof.filter((pl.col("endpoint") == e) & pl.col("Molecule_Name").is_in(sorted(n)))
            for e, n in keep.items()
        ]
    )
    pool_b = list(TABICL_ARMS) + list(CHEMPROP_ARMS)
    pool_a = [f"{a}_a1" for a in TABICL_ARMS] + list(CHEMPROP_ARMS)
    ref = oof.filter(pl.col("method") == "tabicl_monroe").sort(KEYS)
    for arm in {*pool_a, *pool_b}:
        g = oof.filter(pl.col("method") == arm).sort(KEYS)
        assert g.height == ref.height, (arm, g.height, ref.height)
        assert (g.select(KEYS) == ref.select(KEYS)).to_numpy().all(), f"{arm}: rows misaligned"
        assert (g["y_true"].cast(pl.Int64) == ref["y_true"].cast(pl.Int64)).all(), (
            f"{arm}: truth differs"
        )
    return oof, pool_b, pool_a


def blend(oof: pl.DataFrame, arms: list[str], tag: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Caruana and uniform blends; weights and threshold fitted off the held-out outer repeat."""
    parts, weights = [], []
    for endpoint in sorted(oof["endpoint"].unique().to_list()):
        ep = oof.filter((pl.col("endpoint") == endpoint) & pl.col("method").is_in(arms))
        for held in sorted(ep["outer_fold"].unique().to_list()):
            fit, test = (
                ep.filter(pl.col("outer_fold") != held),
                ep.filter(pl.col("outer_fold") == held),
            )

            def probs(frame: pl.DataFrame) -> dict[str, np.ndarray]:
                return {
                    a: frame.filter(pl.col("method") == a).sort(KEYS)["y_prob"].to_numpy()
                    for a in arms
                }

            y_fit = (
                fit.filter(pl.col("method") == arms[0]).sort(KEYS)["y_true"].to_numpy().astype(int)
            )
            w = ensemble.caruana_selection(
                probs(fit), y_fit, lambda y, p: best_mcc(y, p), maximize=True, seed=0
            )
            weights.extend(
                {"pool": tag, "endpoint": endpoint, "held_out": held, "model": a, "weight": w[a]}
                for a in arms
            )
            pf, pt = probs(fit), probs(test)
            base = (
                test.filter(pl.col("method") == arms[0])
                .sort(KEYS)
                .select(*KEYS, "outer_fold", "y_true")
            )
            for name, fit_p, test_p in (
                ("caruana", sum(w[a] * pf[a] for a in arms), sum(w[a] * pt[a] for a in arms)),
                (
                    "uniform",
                    np.mean([pf[a] for a in arms], axis=0),
                    np.mean([pt[a] for a in arms], axis=0),
                ),
            ):
                t_star = float(GRID[int(np.argmax(mcc_grid(y_fit, fit_p)))])
                parts.append(
                    base.with_columns(
                        pl.Series("y_prob", test_p),
                        pl.lit(f"{name}_{tag}").alias("method"),
                        pl.lit(t_star).alias("threshold"),
                    )
                )
    return pl.concat(parts), pl.DataFrame(weights)


def fold_scores(oof: pl.DataFrame, singles: list[str]) -> pl.DataFrame:
    """Per-(method, endpoint, fold) MCC and AUC. Blends carry their rotated threshold; singles get theirs here."""
    rows = []
    for (method, endpoint), g in oof.group_by(["method", "endpoint"], maintain_order=True):
        for held in sorted(g["outer_fold"].unique().to_list()):
            if method in singles:
                fit = g.filter(pl.col("outer_fold") != held)
                t = float(
                    GRID[
                        int(
                            np.argmax(
                                mcc_grid(
                                    fit["y_true"].to_numpy().astype(int), fit["y_prob"].to_numpy()
                                )
                            )
                        )
                    ]
                )
            else:
                t = float(g.filter(pl.col("outer_fold") == held)["threshold"][0])
            for fold in sorted(g.filter(pl.col("outer_fold") == held)["fold"].unique().to_list()):
                f = g.filter(pl.col("fold") == fold)
                y, p = f["y_true"].to_numpy().astype(int), f["y_prob"].to_numpy()
                rows.append(
                    {
                        "method": method,
                        "endpoint": endpoint,
                        "fold": fold,
                        "threshold": t,
                        "mcc": float(matthews_corrcoef(y, p >= t))
                        if len(set(y.tolist())) > 1
                        else 0.0,
                        "auc": float(roc_auc_score(y, p))
                        if len(set(y.tolist())) > 1
                        else float("nan"),
                    }
                )
    return pl.DataFrame(rows)


def macro(folds: pl.DataFrame, metric: str) -> pl.DataFrame:
    return (
        folds.group_by(["method", "fold"])
        .agg(pl.col(metric).mean().alias(metric))
        .sort(["method", "fold"])
    )


def vec(m: pl.DataFrame, method: str, metric: str) -> np.ndarray:
    return m.filter(pl.col("method") == method).sort("fold")[metric].to_numpy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # The vectorised MCC must agree with sklearn before anything is selected on it.
    rng = np.random.default_rng(0)
    y_chk, p_chk = rng.random(500) < 0.3, rng.random(500)
    for i in (0, 7, 15):
        assert abs(mcc_grid(y_chk, p_chk)[i] - matthews_corrcoef(y_chk, p_chk >= GRID[i])) < 1e-9

    oof, pool_b, pool_a = assemble()
    oof.write_parquet(OUT / "oof_both_arms.parquet")
    bl_b, w_b = blend(oof, pool_b, "PB")
    bl_a, w_a = blend(oof, pool_a, "PA")
    pl.concat([w_b, w_a]).write_csv(OUT / "caruana_weights.csv")
    blends = pl.concat([bl_b.select(bl_a.columns), bl_a])
    singles = sorted({*pool_a, *pool_b})
    folds = pl.concat(
        [fold_scores(oof.filter(pl.col("method").is_in(singles)), singles), fold_scores(blends, [])]
    )
    folds.write_csv(OUT / "fold_metrics.csv")

    summary = folds.group_by(["method", "endpoint"]).agg(pl.col(["mcc", "auc"]).mean())
    ma = (
        macro(folds, "mcc")
        .group_by("method")
        .agg(pl.col("mcc").mean())
        .join(macro(folds, "auc").group_by("method").agg(pl.col("auc").mean()), on="method")
        .with_columns(pl.lit("macro").alias("endpoint"))
    )
    summary = pl.concat([summary, ma.select(summary.columns)]).sort(
        ["endpoint", "mcc"], descending=[False, True]
    )
    summary.write_csv(OUT / "summary.csv")
    with pl.Config(tbl_rows=60, tbl_width_chars=160):
        print(summary.filter(pl.col("endpoint") == "macro"))
        print(
            summary.filter(pl.col("endpoint") != "macro").pivot(
                values="mcc", index="method", on="endpoint"
            )
        )

    pairs = [
        ("caruana_PA", "caruana_PB"),
        ("uniform_PA", "uniform_PB"),
        ("caruana_PA", "tabicl_monroe_a1"),
        ("caruana_PB", "tabicl_monroe"),
        ("caruana_PA", "uniform_PA"),
    ]
    rows = []
    for metric in ("mcc", "auc"):
        m = macro(folds, metric)
        for a, b in pairs:
            rows.append(
                {
                    "metric": metric,
                    "scope": "macro",
                    "candidate": a,
                    "vs": b,
                    **evaluation.paired_fold_bootstrap(vec(m, a, metric), vec(m, b, metric)),
                }
            )
        for e in sorted(folds["endpoint"].unique().to_list()):
            ep = folds.filter(pl.col("endpoint") == e)
            for a, b in pairs[:2]:
                rows.append(
                    {
                        "metric": metric,
                        "scope": e,
                        "candidate": a,
                        "vs": b,
                        **evaluation.paired_fold_bootstrap(vec(ep, a, metric), vec(ep, b, metric)),
                    }
                )
    res = pl.DataFrame(rows).with_columns(
        (pl.col("candidate") + " vs " + pl.col("vs")).alias("comparison")
    )
    fam = res.filter(
        (pl.col("metric") == "mcc")
        & (pl.col("scope") == "macro")
        & pl.col("comparison").is_in(["caruana_PA vs caruana_PB", "uniform_PA vs uniform_PB"])
    )
    holm = evaluation.holm_bonferroni(dict(zip(fam["comparison"], fam["p_value"], strict=True)))
    res = res.join(
        holm.select("comparison", "threshold", "significant"), on="comparison", how="left"
    )
    res.write_csv(OUT / "paired.csv")
    with pl.Config(tbl_rows=40, tbl_width_chars=200, tbl_cols=14):
        print(res.drop("comparison"))
        mw = (
            pl.concat([w_b, w_a])
            .group_by(["pool", "endpoint", "model"])
            .agg(pl.col("weight").mean())
            .sort(["pool", "endpoint", "weight"], descending=[False, False, True])
        )
        print(mw)


if __name__ == "__main__":
    main()
