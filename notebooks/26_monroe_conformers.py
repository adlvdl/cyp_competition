"""26: does Monroe gain from Shorku's xTB conformer ensembles? Four arms, both tracks.

Run as a script:

    uv run python notebooks/26_monroe_conformers.py

`25`'s probe found Monroe's embedding moves 1.1x seed noise when xTB geometry replaces
its random ETKDG conformer, and 0.40 (10x more) between Monroe's InChI route and the
coordinate route that external geometry requires. A direct etkdg-vs-xTB comparison would
therefore mix three changes. Four arms separate them; the readout is the one `16` and
`23` used (TabICL single-task, PCA 128, 2 estimators, seed 42), so only the embedding
differs:

  A  tabicl_monroe_etkdg            Monroe's InChI route, one random conformer. `16`'s
                                    (regression) and `23`'s (TDI) cached OOF, not refit.
  B  tabicl_monroe_sdf_etkdg        coordinate route, our SMILES, one seeded conformer.
  C  tabicl_monroe_sdf_etkdg_stereo coordinate route, uniform mean over every upstream
                                    stereoisomer consistent with our SMILES, one seeded
                                    conformer each (`shorku_qm.structure_sets`).
  D  tabicl_monroe_shorku_qm        as C, but each isomer is the Boltzmann-weighted mean
                                    over all its xTB conformers.

B-A is the route, C-B the stereo averaging, D-C the xTB geometry, D-A the total. Those
four are the Holm family on each track. Compounds absent upstream (27) take B's vector
in C and D. Statistics bootstrap the 25 per-fold macro differences (`15`'s protocol;
anti-conservative since repeats share compounds). TDI MCC at the fixed 0.25 threshold
of `14`/`23`.

Data: Shorku (Oleg Gromov) xTB conformers, MIT, README "Third-party data".
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cyp  # noqa: E402,F401  (claims LightGBM's OpenMP runtime first)
from cyp import constants as C  # noqa: E402
from cyp import cv, data, encoders, evaluation, multitask, timings  # noqa: E402
from cyp import shorku_qm as Q  # noqa: E402
from cyp.tabular_models import predict_subprocess  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from sklearn.metrics import matthews_corrcoef  # noqa: E402

RDLogger.DisableLog("rdApp.*")

EXP = PROJECT_ROOT / "experiments"
OUT = EXP / Path(__file__).stem
CACHE = OUT / "cache"
TIMING_LOG = OUT / "timings.csv"
THRESHOLD = 0.25
SEED = 1
CHUNK = 5000
KEYS = ["endpoint", "fold", "Molecule_Name"]
ARM_A, ARM_B, ARM_C, ARM_D = (
    "tabicl_monroe_etkdg",
    "tabicl_monroe_sdf_etkdg",
    "tabicl_monroe_sdf_etkdg_stereo",
    "tabicl_monroe_shorku_qm",
)
NEW_ARMS = (ARM_B, ARM_C, ARM_D)
FAMILY = [
    (ARM_B, ARM_A, "route"),
    (ARM_C, ARM_B, "stereo averaging"),
    (ARM_D, ARM_C, "xTB geometry"),
    (ARM_D, ARM_A, "total"),
]


def _conformer_block(smiles: str) -> tuple[str, bool]:
    """Seeded Monroe-recipe conformer as a mol block, for the process pool."""
    mol, flat = encoders.monroe_conformer_or_flat(smiles, SEED)
    return Chem.MolToMolBlock(mol), flat


def etkdg_mols(smiles: list[str]) -> tuple[list, int]:
    """Seeded conformers for many SMILES in parallel; returns mols and the flat count.

    RDKit only, no torch, so a fork pool after `import cyp` is safe (Monroe's own
    featurizer forks the same way). macOS defaults to spawn, which would re-import
    this script in every worker.
    """
    with ProcessPoolExecutor(max_workers=os.cpu_count(), mp_context=mp.get_context("fork")) as pool:
        out = list(pool.map(_conformer_block, smiles, chunksize=32))
    mols = [Chem.MolFromMolBlock(block, removeHs=False) for block, _ in out]
    return mols, sum(flat for _, flat in out)


def embed_chunked(name: str, keys: list[str], build) -> np.ndarray:
    """Monroe embeddings for `keys`, built and cached CHUNK at a time.

    `build(keys_slice) -> list[Mol]` makes the molecules only for chunks not yet cached,
    so a crash after 30 minutes of embedding loses one chunk, not the run. The cache
    name carries a hash of the key list, so a different compound set never reuses it.
    """
    tag = hashlib.sha256("\n".join(keys).encode()).hexdigest()[:10]
    parts = []
    for i, start in enumerate(range(0, len(keys), CHUNK)):
        path = CACHE / f"{name}_{tag}_chunk{i:03d}.npy"
        if not path.exists():
            t0 = time.time()
            np.save(path, encoders.embed_monroe_conformers(build(keys[start : start + CHUNK])))
            print(f"{name} chunk {i}: {time.time() - t0:.0f}s", flush=True)
        parts.append(np.load(path))
    return np.concatenate(parts)


def build_embeddings(universe: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """Per-compound Monroe vectors for arms B, C and D."""
    sets = Q.structure_sets(universe)
    sets.write_parquet(OUT / "structure_sets.parquet")
    present = sets.drop_nulls("structure_id")
    structures = sorted(set(present["structure_id"].to_list()))
    smiles_of = dict(
        zip(present["structure_id"].to_list(), present["structure_smiles"].to_list(), strict=False)
    )
    flats: dict[str, int] = {}

    def from_smiles(arm: str):
        def build(batch: list[str]) -> list:
            mols, flat = etkdg_mols(batch)
            flats[arm] = flats.get(arm, 0) + flat
            return mols

        return build

    t0 = time.time()
    b_table = embed_chunked("sdf_etkdg", universe, from_smiles("B"))
    timings.record(TIMING_LOG, "embed", "sdf_etkdg", "all", "full", time.time() - t0)

    t0 = time.time()
    stereo_smiles = [smiles_of[s] for s in structures]
    c_struct = embed_chunked("sdf_etkdg_stereo", stereo_smiles, from_smiles("C"))
    timings.record(TIMING_LOG, "embed", "sdf_etkdg_stereo", "all", "full", time.time() - t0)

    # D: every xTB conformer of every structure in any set.
    t0 = time.time()
    props = Q.load_conformer_properties().filter(pl.col("structure_id").is_in(structures))
    props = props.sort("structure_id", "conf_id")
    geoms = Q.read_conformers(structures)
    conf_keys = [f"{s}.{c}" for s, c in props.select("structure_id", "conf_id").iter_rows()]

    def from_xyz(batch: list[str]) -> list:
        out = []
        for key in batch:
            sid, conf = key.rsplit(".", 1)
            elements, xyz = geoms[sid][int(conf)]
            out.append(Q.mol_with_coordinates(smiles_of[sid], elements, xyz))
        return out

    d_conf = embed_chunked("shorku_qm", conf_keys, from_xyz)
    timings.record(TIMING_LOG, "embed", "shorku_qm", "all", "full", time.time() - t0)

    # Boltzmann mean per structure, then uniform mean per compound over its isomer set.
    weights = props["boltzmann_weight"].to_numpy()
    conf_rows: dict[str, list[int]] = {}
    for i, sid in enumerate(props["structure_id"].to_list()):
        conf_rows.setdefault(sid, []).append(i)
    d_struct = {
        sid: (weights[rows, None] * d_conf[rows]).sum(0) / weights[rows].sum()
        for sid, rows in conf_rows.items()
    }
    c_row = {s: i for i, s in enumerate(structures)}
    b_row = {s: i for i, s in enumerate(universe)}
    members: dict[str, list[str]] = {}
    for smi, sid in present.select("SMILES", "structure_id").iter_rows():
        members.setdefault(smi, []).append(sid)

    out = {ARM_B: {}, ARM_C: {}, ARM_D: {}}
    for smi in universe:
        b = b_table[b_row[smi]]
        out[ARM_B][smi] = b
        sids = members.get(smi)
        out[ARM_C][smi] = np.mean([c_struct[c_row[s]] for s in sids], axis=0) if sids else b
        out[ARM_D][smi] = np.mean([d_struct[s] for s in sids], axis=0) if sids else b

    per = sets.group_by("SMILES").agg(pl.col("rule").first(), pl.len().alias("n_structures"))
    diag = per.group_by("rule").agg(
        pl.len().alias("compounds"), pl.col("n_structures").sum().alias("structures")
    )
    diag = diag.with_columns(
        pl.lit(flats.get("B", 0)).alias("flat_fallback_B"),
        pl.lit(flats.get("C", 0)).alias("flat_fallback_C"),
        pl.lit(len(conf_keys)).alias("xtb_conformers"),
    )
    diag.write_csv(OUT / "embedding_diagnostics.csv")
    print(diag, flush=True)
    return out


def regression_cv(emb: dict[str, dict[str, np.ndarray]]) -> pl.DataFrame:
    """`16`'s loop for the three new arms, fold-major, cached per (arm, fold)."""
    frames = {e: data.training_frame(e) for e in C.REGRESSION_ENDPOINTS}
    _, assignments = cv.shared_scaffold_folds(frames)
    splits = {}
    for e, f in frames.items():
        indexed = f.with_row_index("_row")
        splits[e] = {
            fd: (tr, te)
            for fd, _o, _i, tr, _v, te in cv.fold_assignment_splits(indexed, assignments)
        }
    feats = {
        arm: {e: np.vstack([emb[arm][s] for s in f["SMILES"].to_list()]) for e, f in frames.items()}
        for arm in NEW_ARMS
    }
    parts = []
    for fd in sorted(assignments["fold"].unique().to_list()):
        for arm in NEW_ARMS:
            path = CACHE / f"reg_{arm}_fold{fd:02d}.parquet"
            if not path.exists():
                t0 = time.time()
                rows = []
                for e, by_fold in splits.items():
                    tr, te = by_fold[fd]
                    X = feats[arm][e]
                    # Child process: torch in the parent would break LightGBM (CLAUDE.md).
                    p, _, _ = predict_subprocess(
                        X[tr["_row"].to_numpy()],
                        tr["y_true"].to_numpy(),
                        X[te["_row"].to_numpy()],
                        kind="tabicl",
                    )
                    p = np.asarray(p, dtype=float)
                    assert np.isfinite(p).all(), f"{arm}/{e}/fold {fd}: non-finite predictions"
                    rows.append(
                        te.select(
                            pl.lit(arm).alias("method"),
                            pl.lit(e).alias("endpoint"),
                            pl.lit(fd).alias("fold"),
                            "Molecule_Name",
                            "y_true",
                            pl.Series("y_pred", p),
                            "y_lower",
                            "y_upper",
                        )
                    )
                pl.concat(rows).write_parquet(path)
                timings.record(
                    TIMING_LOG, "reg_cv", arm, "all_four", f"fold_{fd}", time.time() - t0
                )
            parts.append(pl.read_parquet(path))
        print(f"regression fold {fd + 1}/25", flush=True)
    new = pl.concat(parts)

    # Arm A is 16's OOF; it must sit on exactly these rows with exactly these labels.
    cols = ["method", "endpoint", "fold", "Molecule_Name", "y_true", "y_pred", "y_lower", "y_upper"]
    old = pl.read_parquet(EXP / "16_encoders" / "oof.parquet").filter(
        pl.col("method") == "tabicl_monroe"
    )
    old = old.with_columns(pl.lit(ARM_A).alias("method")).select(cols)
    ref = old.sort(KEYS)
    for arm in NEW_ARMS:
        g = new.filter(pl.col("method") == arm).sort(KEYS)
        assert g.height == ref.height, (arm, g.height, ref.height)
        assert (g.select(KEYS) == ref.select(KEYS)).to_numpy().all(), f"{arm}: rows misaligned"
        assert (g["y_true"] - ref["y_true"]).abs().max() < 1e-9, f"{arm}: y_true differs"
    oof = pl.concat([old, new.select(cols)], how="vertical_relaxed")
    oof.write_parquet(OUT / "oof_regression.parquet")
    return oof


def tdi_cv(emb: dict[str, dict[str, np.ndarray]]) -> pl.DataFrame:
    """`23`'s loop for the three new arms, fold-major, cached per (arm, fold)."""
    frames = {f"{iso}_is_TDI": data.tdi_training_frame(iso) for iso in C.TDI_ISOFORMS}
    _, assignments = cv.shared_scaffold_folds(frames, n_outer=5, n_inner=5, seed=42)
    feats = {
        arm: {e: np.vstack([emb[arm][s] for s in f["SMILES"].to_list()]) for e, f in frames.items()}
        for arm in NEW_ARMS
    }
    parts = []
    for fd in range(assignments["fold"].n_unique()):
        for arm in NEW_ARMS:
            path = CACHE / f"tdi_{arm}_fold{fd:02d}.parquet"
            if not path.exists():
                t0 = time.time()
                part = multitask.run_cv_singletask_classification(
                    frames,
                    method="tabicl",
                    features=feats[arm],
                    assignments=assignments,
                    folds=[fd],
                    seed=42,
                ).with_columns(pl.lit(arm).alias("method"))
                part.write_parquet(path)
                timings.record(TIMING_LOG, "tdi_cv", arm, "both", f"fold_{fd}", time.time() - t0)
            parts.append(pl.read_parquet(path))
        print(f"TDI fold {fd + 1}/25", flush=True)
    new = pl.concat(parts)

    cols = ["method", "endpoint", "fold", "outer_fold", "Molecule_Name", "y_true", "y_prob"]
    old = pl.read_parquet(EXP / "23_tdi_encoders" / "oof.parquet").filter(
        pl.col("method") == "tabicl_monroe"
    )
    old = old.with_columns(pl.lit(ARM_A).alias("method")).select(cols)
    ref = old.sort(KEYS)
    for arm in NEW_ARMS:
        g = new.filter(pl.col("method") == arm).sort(KEYS)
        assert g.height == ref.height, (arm, g.height, ref.height)
        assert (g.select(KEYS) == ref.select(KEYS)).to_numpy().all(), f"{arm}: rows misaligned"
        assert (g["y_true"].cast(pl.Int64) == ref["y_true"].cast(pl.Int64)).all(), arm
        # numpy 2.2 makes TabICL return all-NaN probabilities silently (CLAUDE.md).
        assert g["y_prob"].is_nan().sum() == 0, f"{arm}: NaN probabilities"
    oof = pl.concat([old, new.select(cols)], how="vertical_relaxed")
    oof.write_parquet(OUT / "oof_tdi.parquet")
    return oof


def paired_fold_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 10_000) -> dict[str, float]:
    """`15`'s test: bootstrap the mean of the 25 per-fold differences a - b."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(42)
    resampled = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(axis=1)
    p = 2.0 * min((resampled <= 0).mean(), (resampled >= 0).mean())
    return {
        "diff": float(diff.mean()),
        "ci_low": float(np.percentile(resampled, 2.5)),
        "ci_high": float(np.percentile(resampled, 97.5)),
        "p_value": float(min(p, 1.0)),
    }


def fold_scores(oof_reg: pl.DataFrame, oof_tdi: pl.DataFrame) -> pl.DataFrame:
    """Per (track, method, endpoint, fold) score; endpoint "macro" is the board's mean."""
    reg = evaluation.fold_metrics(oof_reg).select(
        "method", "endpoint", "fold", pl.col("st_rae").alias("score")
    )
    rows = []
    for (method, endpoint, fold), g in oof_tdi.group_by(["method", "endpoint", "fold"]):
        y, p = g["y_true"].cast(pl.Int64).to_numpy(), g["y_prob"].to_numpy()
        rows.append((method, endpoint, fold, float(matthews_corrcoef(y, p >= THRESHOLD))))
    tdi = pl.DataFrame(rows, schema=["method", "endpoint", "fold", "score"], orient="row")
    both = pl.concat(
        [
            reg.with_columns(pl.lit("regression").alias("track")),
            tdi.with_columns(pl.lit("tdi").alias("track")),
        ],
        how="diagonal",
    )
    macro = (
        both.group_by("track", "method", "fold")
        .agg(pl.col("score").mean())
        .with_columns(pl.lit("macro").alias("endpoint"))
    )
    return pl.concat([both, macro.select(both.columns)])


def compare(scores: pl.DataFrame) -> None:
    """Holm family per track on the macro, plus per-endpoint descriptives."""

    def vec(track: str, method: str, endpoint: str) -> np.ndarray:
        g = scores.filter(
            (pl.col("track") == track)
            & (pl.col("method") == method)
            & (pl.col("endpoint") == endpoint)
        )
        return g.sort("fold")["score"].to_numpy()

    family, descriptive = [], []
    for track in ("regression", "tdi"):
        # ST-RAE is lower-better, MCC higher-better: count folds where the candidate wins.
        better = (lambda d: d < 0) if track == "regression" else (lambda d: d > 0)
        endpoints = sorted(scores.filter(pl.col("track") == track)["endpoint"].unique().to_list())
        for a, b, label in FAMILY:
            for endpoint in endpoints:
                va, vb = vec(track, a, endpoint), vec(track, b, endpoint)
                r = paired_fold_bootstrap(va, vb)
                r["folds_candidate_better"] = int(better(va - vb).sum())
                row = {
                    "track": track,
                    "effect": label,
                    "candidate": a,
                    "vs": b,
                    "endpoint": endpoint,
                    **r,
                }
                (family if endpoint == "macro" else descriptive).append(row)
    fam = pl.DataFrame(family)
    verdicts = []
    for track in ("regression", "tdi"):
        sub = fam.filter(pl.col("track") == track)
        holm = evaluation.holm_bonferroni(dict(zip(sub["effect"], sub["p_value"], strict=True)))
        verdicts.append(
            holm.select(
                pl.lit(track).alias("track"),
                pl.col("comparison").alias("effect"),
                pl.col("threshold").alias("holm_threshold"),
                pl.col("significant").alias("holm_significant"),
            )
        )
    fam = fam.join(pl.concat(verdicts), on=["track", "effect"])
    fam.write_csv(OUT / "paired_family.csv")
    pl.DataFrame(descriptive).write_csv(OUT / "paired_per_endpoint.csv")

    summary = (
        scores.group_by("track", "method", "endpoint")
        .agg(pl.col("score").mean().round(4))
        .pivot(on="endpoint", index=["track", "method"], values="score")
        .sort("track", "method")
    )
    summary.write_csv(OUT / "summary.csv")
    with pl.Config(tbl_rows=40, tbl_cols=12, tbl_width_chars=220, fmt_str_lengths=40):
        print(summary)
        print(
            fam.select(
                "track",
                "effect",
                "diff",
                "ci_low",
                "ci_high",
                "p_value",
                "folds_candidate_better",
                "holm_significant",
            )
        )


def correlations(oof_reg: pl.DataFrame, oof_tdi: pl.DataFrame) -> None:
    """Pearson r between arms' OOF predictions per endpoint (17/21: errors decide blend value)."""
    rows = []
    arms = [ARM_A, *NEW_ARMS]
    for oof, col in ((oof_reg, "y_pred"), (oof_tdi, "y_prob")):
        for e in sorted(oof["endpoint"].unique().to_list()):
            vecs = {
                a: oof.filter((pl.col("method") == a) & (pl.col("endpoint") == e))
                .sort(KEYS)[col]
                .to_numpy()
                for a in arms
            }
            for i, a in enumerate(arms):
                for b in arms[i + 1 :]:
                    rows.append(
                        {
                            "endpoint": e,
                            "a": a,
                            "b": b,
                            "pearson_r": round(float(np.corrcoef(vecs[a], vecs[b])[0, 1]), 4),
                        }
                    )
    pl.DataFrame(rows).write_csv(OUT / "prediction_correlation.csv")


def main() -> None:
    assert tuple(int(x) for x in np.__version__.split(".")[:2]) >= (2, 4), "numpy<2.4 breaks TabICL"
    CACHE.mkdir(parents=True, exist_ok=True)
    reg_frames = [data.training_frame(e) for e in C.REGRESSION_ENDPOINTS]
    tdi_frames = [data.tdi_training_frame(iso) for iso in C.TDI_ISOFORMS]
    universe = sorted({s for f in reg_frames + tdi_frames for s in f["SMILES"].to_list()})
    print(f"{len(universe)} training compounds", flush=True)

    emb = build_embeddings(universe)
    # Per-compound tables are cheap to rebuild from the chunk caches; saved for reuse.
    for arm in NEW_ARMS:
        np.save(CACHE / f"{arm}.npy", np.vstack([emb[arm][s] for s in universe]))
    (CACHE / "universe.json").write_text(json.dumps(universe))

    oof_reg = regression_cv(emb)
    oof_tdi = tdi_cv(emb)
    scores = fold_scores(oof_reg, oof_tdi)
    scores.write_csv(OUT / "fold_scores.csv")
    compare(scores)
    correlations(oof_reg, oof_tdi)


if __name__ == "__main__":
    main()
