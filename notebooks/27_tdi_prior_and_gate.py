"""27: two TDI ideas taken from other entrants' public reports, tested on one clean readout.

Run as a script:

    uv run python notebooks/27_tdi_prior_and_gate.py

Both come from reading the model-report links on the live leaderboard (2026-10-25);
sources are credited in README.md under "Ideas from other entrants".

A1. Train only on compounds that have a direct-inhibition arm (JacksonBurns's notes).
    1,249 of the 3,584 labelled CYP3A4 rows have no direct arm and are *all* TDI
    negative (verified on our snapshot), so the 21.3% training prior is diluted from
    32.7% among rows that look like the blind set, where every compound has both arms.
    We have always trained on all 3,584. The comparison is scored on the both-arms
    rows only, since those are the rows the blind set resembles.

A2. Feed the model a predicted direct pIC50 as a feature (Nomad-link-id's report).
    The TDI rule is "pIC50 shift > 0.301 given the compound inhibits at all", so direct
    potency is a gate. `13` closed the two-stage route for the *shift*, a difference of
    two noisy quantities. This is one predicted potency used as a feature, not a
    difference. Leak-free: inside each outer fold the feature for a training compound is
    an inner-fold out-of-fold prediction, and for a test compound it comes from a model
    fitted on the training compounds only.

Readout: TabICL single-task on frozen Monroe (`23`'s best single arm), PCA 128, 2
estimators, seed 42. Arms:
  ctl   all rows, 128 PCs (same pipeline as the others; checked against 23's arm)
  a1    both-arms rows only, 128 PCs
  a2    all rows, 128 PCs + gate
  a12   both-arms rows only, 128 PCs + gate

Scoring population: both-arms rows, every arm. Same folds as 13/14/23
(`cv.shared_scaffold_folds`, seed 42). The prior shifts under A1, so 0.25 is not
assumed right: the headline threshold is chosen on four outer repeats and applied to the
fifth, rotated. MCC at a fixed 0.25 and threshold-free AUC are reported beside it.
Fold is the unit of the bootstrap (see `15`). Holm family: a1 vs ctl, a2 vs ctl.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score  # noqa: E402

import cyp  # noqa: E402,F401  (must come first: claims the OpenMP slot, see CLAUDE.md)
from cyp import constants as C  # noqa: E402
from cyp import cv, data, evaluation  # noqa: E402
from cyp.tabular_models import predict_subprocess  # noqa: E402

EXP = PROJECT_ROOT / "experiments"
OUT = EXP / "27_tdi_prior_and_gate"
CACHE = OUT / "cache"
N_PCS = 128
SEED = 42
FIXED_THRESHOLD = 0.25
GRID = np.round(np.arange(0.05, 0.6001, 0.025), 3)
KEYS = ["endpoint", "fold", "Molecule_Name"]
ARMS = {  # name -> (both-arms rows only, gate feature)
    "ctl": (False, False),
    "a1": (True, False),
    "a2": (False, True),
    "a12": (True, True),
}


def load_notebook_23():
    """`23`'s Monroe feature builder, reused so the embeddings are byte-identical."""
    spec = importlib.util.spec_from_file_location(
        "nb23", PROJECT_ROOT / "notebooks" / "23_tdi_encoders.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def direct_pic50(isoform: str, frame: pl.DataFrame) -> np.ndarray:
    """Direct-arm pIC50 for each frame row, NaN where the compound has no direct arm."""
    col = f"{isoform}_pIC50_direct_inhibition"
    table = data.load_train_tdi().select("Molecule_Name", col)
    lookup = dict(zip(table["Molecule_Name"].to_list(), table[col].to_list(), strict=True))
    return np.array(
        [np.nan if lookup[n] is None else lookup[n] for n in frame["Molecule_Name"]], dtype=float
    )


def fit_pcs(X: np.ndarray, fit_rows: np.ndarray) -> np.ndarray:
    """128 principal components fitted on `fit_rows` only (unsupervised, so no label leak)."""
    return PCA(n_components=N_PCS, random_state=SEED).fit(X[fit_rows]).transform(X)


def gate_for_fold(
    endpoint: str,
    frame: pl.DataFrame,
    X: np.ndarray,
    y_direct: np.ndarray,
    assignments: pl.DataFrame,
    fold: int,
    outer: int,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
) -> np.ndarray:
    """Leak-free predicted direct pIC50 for every row of `frame` in one outer fold.

    Training rows get an inner-fold OOF prediction: the regression model never saw the
    compound's own label. Test rows get a prediction from a model fitted on all labelled
    training compounds. Compounds with no direct arm are never in a regression fit, so
    their prediction is leak-free by construction.
    """
    path = CACHE / f"gate_{endpoint}_fold{fold:02d}.npy"
    if path.exists():
        return np.load(path)
    Z = fit_pcs(X, train_rows)
    repeat = (
        assignments.filter(pl.col("outer_fold") == outer).select("Molecule_Name", "fold").unique()
    )
    fold_of = dict(zip(repeat["Molecule_Name"].to_list(), repeat["fold"].to_list(), strict=True))
    inner = np.array([fold_of[n] for n in frame["Molecule_Name"]])
    labelled = ~np.isnan(y_direct)
    gate = np.full(len(frame), np.nan)

    def regress(fit: np.ndarray, pred: np.ndarray) -> np.ndarray:
        out, _, _ = predict_subprocess(
            Z[fit],
            y_direct[fit],
            Z[pred],
            kind="tabicl",
            task="regression",
            seed=SEED,
            tabicl_max_features=N_PCS,
        )
        return np.asarray(out, dtype=float)

    for g in sorted(set(inner[train_rows])):
        in_g = train_rows[inner[train_rows] == g]
        fit = train_rows[(inner[train_rows] != g) & labelled[train_rows]]
        gate[in_g] = regress(fit, in_g)
    gate[test_rows] = regress(train_rows[labelled[train_rows]], test_rows)
    assert not np.isnan(gate).any()
    np.save(path, gate)
    return gate


def run_arms() -> pl.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    nb23 = load_notebook_23()
    frames = {f"{iso}_is_TDI": data.tdi_training_frame(iso) for iso in C.TDI_ISOFORMS}
    _, assignments = cv.shared_scaffold_folds(frames, n_outer=5, n_inner=5, seed=SEED)
    n_folds = assignments["fold"].n_unique()
    feats = nb23.tdi_features(frames)["tabicl_monroe"]
    y_direct = {e: direct_pic50(e.split("_")[0], f) for e, f in frames.items()}
    for e, f in frames.items():
        n_no = int(np.isnan(y_direct[e]).sum())
        pos = f["y_true"].cast(pl.Int8).to_numpy()
        print(
            f"{e}: {f.height} rows, {n_no} with no direct arm, positive rate "
            f"{pos.mean():.3f} overall / {pos[~np.isnan(y_direct[e])].mean():.3f} among both-arms rows",
            flush=True,
        )

    pieces = []
    for fold in range(n_folds):  # fold-major so an interrupted run leaves a usable paired set
        for endpoint, frame in frames.items():
            X, yd = feats[endpoint], y_direct[endpoint]
            indexed = frame.with_row_index("_row")
            both = ~np.isnan(yd)
            # Full-frame split for this fold: defines train/test rows for the gate and every arm.
            split = next(
                s
                for s in cv.fold_assignment_splits(indexed, assignments, p_val=0.0, seed=SEED)
                if s[0] == fold
            )
            _, outer, _, train_all, _, test_all = split
            train_rows, test_rows = train_all["_row"].to_numpy(), test_all["_row"].to_numpy()
            gate = None
            for arm, (both_only, use_gate) in ARMS.items():
                path = CACHE / f"{arm}_{endpoint}_fold{fold:02d}.parquet"
                if not path.exists():
                    if use_gate and gate is None:
                        gate = gate_for_fold(
                            endpoint, frame, X, yd, assignments, fold, outer, train_rows, test_rows
                        )
                    fit_rows = train_rows[both[train_rows]] if both_only else train_rows
                    Z = fit_pcs(X, fit_rows)
                    M = np.column_stack([Z, gate]) if use_gate else Z
                    prob, _, _ = predict_subprocess(
                        M[fit_rows],
                        frame["y_true"].cast(pl.Int8).to_numpy()[fit_rows],
                        M[test_rows],
                        kind="tabicl",
                        task="classification",
                        seed=SEED,
                        tabicl_max_features=M.shape[1],
                    )
                    assert not np.isnan(prob).any(), f"{arm}: NaN probabilities (numpy/TabICL trap)"
                    pl.DataFrame(
                        {
                            "method": arm,
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


def mcc_at(y: np.ndarray, p: np.ndarray, t: float) -> float:
    return float(matthews_corrcoef(y, p >= t)) if len(set(y.tolist())) > 1 else 0.0


def score(oof: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per-(method, endpoint, fold) metrics, and the rotated thresholds used."""
    rows, chosen = [], []
    for (method, endpoint), g in oof.group_by(["method", "endpoint"], maintain_order=True):
        for held in sorted(g["outer_fold"].unique().to_list()):
            fit = g.filter(pl.col("outer_fold") != held)
            y, p = fit["y_true"].to_numpy().astype(int), fit["y_prob"].to_numpy()
            t_star = float(GRID[int(np.argmax([mcc_at(y, p, t) for t in GRID]))])
            chosen.append(
                {"method": method, "endpoint": endpoint, "held_out": held, "threshold": t_star}
            )
            for fold in sorted(g.filter(pl.col("outer_fold") == held)["fold"].unique().to_list()):
                f = g.filter(pl.col("fold") == fold)
                yf, pf = f["y_true"].to_numpy().astype(int), f["y_prob"].to_numpy()
                rows.append(
                    {
                        "method": method,
                        "endpoint": endpoint,
                        "fold": fold,
                        "mcc_rotated": mcc_at(yf, pf, t_star),
                        "mcc_fixed": mcc_at(yf, pf, FIXED_THRESHOLD),
                        "auc": float(roc_auc_score(yf, pf))
                        if len(set(yf.tolist())) > 1
                        else float("nan"),
                        "ap": float(average_precision_score(yf, pf))
                        if len(set(yf.tolist())) > 1
                        else float("nan"),
                    }
                )
    return pl.DataFrame(rows), pl.DataFrame(chosen)


def macro_by_fold(folds: pl.DataFrame, metric: str) -> pl.DataFrame:
    return (
        folds.group_by(["method", "fold"])
        .agg(pl.col(metric).mean().alias(metric))
        .sort(["method", "fold"])
    )


def vec(macro: pl.DataFrame, method: str, metric: str) -> np.ndarray:
    return macro.filter(pl.col("method") == method).sort("fold")[metric].to_numpy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    oof_all = run_arms()

    # Score on both-arms rows only: the population the blind set resembles.
    frames = {f"{iso}_is_TDI": data.tdi_training_frame(iso) for iso in C.TDI_ISOFORMS}
    keep = {
        e: set(
            f.filter(pl.Series(~np.isnan(direct_pic50(e.split("_")[0], f))))[
                "Molecule_Name"
            ].to_list()
        )
        for e, f in frames.items()
    }
    oof = pl.concat(
        [
            oof_all.filter((pl.col("endpoint") == e) & pl.col("Molecule_Name").is_in(sorted(names)))
            for e, names in keep.items()
        ]
    )
    oof.write_parquet(OUT / "oof_both_arms.parquet")
    oof_all.write_parquet(OUT / "oof_all_rows.parquet")

    # Sanity: the control pipeline must reproduce 23's Monroe arm.
    ref = pl.read_parquet(EXP / "23_tdi_encoders" / "oof.parquet").filter(
        pl.col("method") == "tabicl_monroe"
    )
    j = oof_all.filter(pl.col("method") == "ctl").join(ref, on=KEYS, suffix="_23")
    r = float(np.corrcoef(j["y_prob"].to_numpy(), j["y_prob_23"].to_numpy())[0, 1])
    print(
        f"ctl vs 23's tabicl_monroe: {j.height} rows, probability correlation {r:.4f}", flush=True
    )
    assert j.height == oof_all.filter(pl.col("method") == "ctl").height and r > 0.99, (
        "control does not reproduce 23"
    )

    folds, chosen = score(oof)
    folds.write_csv(OUT / "fold_metrics.csv")
    chosen.write_csv(OUT / "rotated_thresholds.csv")
    summary = (
        folds.group_by(["method", "endpoint"])
        .agg(pl.col(["mcc_rotated", "mcc_fixed", "auc", "ap"]).mean())
        .sort(["endpoint", "method"])
    )
    macro_means = (
        folds.group_by(["method", "fold"])
        .agg(pl.col(["mcc_rotated", "mcc_fixed", "auc", "ap"]).mean())
        .group_by("method")
        .agg(pl.col(["mcc_rotated", "mcc_fixed", "auc", "ap"]).mean())
        .with_columns(pl.lit("macro").alias("endpoint"))
    )
    summary = pl.concat([summary, macro_means.select(summary.columns)]).sort(["endpoint", "method"])
    summary.write_csv(OUT / "summary.csv")
    with pl.Config(tbl_rows=40, tbl_width_chars=160):
        print(summary)
        print(
            chosen.group_by(["method", "endpoint"])
            .agg(pl.col("threshold").mean().alias("mean_threshold"))
            .sort(["endpoint", "method"])
        )

    # Paired fold-level comparisons.
    comparisons = [("a1", "ctl"), ("a2", "ctl"), ("a12", "ctl"), ("a12", "a1"), ("a12", "a2")]
    results = []
    for metric in ("mcc_rotated", "mcc_fixed", "auc"):
        macro = macro_by_fold(folds, metric)
        for a, b in comparisons:
            results.append(
                {
                    "metric": metric,
                    "scope": "macro",
                    "candidate": a,
                    "vs": b,
                    **evaluation.paired_fold_bootstrap(
                        vec(macro, a, metric), vec(macro, b, metric)
                    ),
                }
            )
        for e in sorted(folds["endpoint"].unique().to_list()):
            ep = folds.filter(pl.col("endpoint") == e)
            for a, b in comparisons:
                results.append(
                    {
                        "metric": metric,
                        "scope": e,
                        "candidate": a,
                        "vs": b,
                        **evaluation.paired_fold_bootstrap(vec(ep, a, metric), vec(ep, b, metric)),
                    }
                )
    res = pl.DataFrame(results)
    family = res.filter(
        (pl.col("metric") == "mcc_rotated")
        & (pl.col("scope") == "macro")
        & pl.col("candidate").is_in(["a1", "a2"])
        & (pl.col("vs") == "ctl")
    )
    holm = evaluation.holm_bonferroni(
        {f"{r['candidate']} vs {r['vs']}": r["p_value"] for r in family.iter_rows(named=True)}
    )
    res = res.with_columns((pl.col("candidate") + " vs " + pl.col("vs")).alias("comparison")).join(
        holm.select("comparison", "threshold", "significant"), on="comparison", how="left"
    )
    res.write_csv(OUT / "paired.csv")
    with pl.Config(tbl_rows=60, tbl_width_chars=200, tbl_cols=14):
        print(res.filter(pl.col("scope") == "macro").drop("comparison"))
        print(
            res.filter(pl.col("scope") != "macro")
            .filter(pl.col("metric") == "mcc_rotated")
            .select("scope", "candidate", "vs", "diff", "ci_low", "ci_high", "p_value")
        )


if __name__ == "__main__":
    main()
