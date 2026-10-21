import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # 16 — Frozen 3D encoders: Monroe and Atom-JEPA vs CheMeleon

    `03` found that **representation moved the numbers and model choice did not**:
    four unrelated architectures on frozen CheMeleon landed within 0.012 of each other,
    while ECFP4 → Mordred → CheMeleon bought ~0.17. TabICL on CheMeleon was the best
    frozen-representation arm (macro ST-RAE 0.7599 on `03`'s per-endpoint folds, 0.7638
    on its shared folds). The binding constraint was the encoder, so this notebook
    changes only the encoder.

    | encoder | pretraining | geometry | width |
    |:--|:--|:--|--:|
    | **CheMeleon** (control) | 2D graph, Mordred-descriptor regression | none | 2048 |
    | **Monroe** | 81M PM6 properties, 1.56M PCBA assays, denoising | RDKit 3D | 720 |
    | **Atom-JEPA** | 19M conformers, masked latent prediction | RDKit 3D | 256 |

    ## Design

    - **Same readout as the control.** `tabicl`, single-task, through
      `tabular_models.predict_subprocess` with its defaults (PCA to 128, 2 estimators,
      seed 42), so only the embedding differs.
    - **The control is `03`'s cached `mt_tabicl_singletask` OOF**, not a rerun. It is
      aligned to this notebook's folds and `y_true` before anything is compared, and the
      notebook stops if it is not.
    - **Shared folds** (`cv.shared_scaffold_folds`), so every comparison is paired.
    - **Atom-JEPA reads its last block** (the package default). All 8 blocks are cached,
      so a layer sweep is free later; it is not run here.

    ## What a null result would mean

    The ICML 2026 study cited alongside Monroe reports frozen foundation-model
    embeddings losing to Morgan fingerprints on OpenADMET endpoints. Our `03` result
    says the opposite for CheMeleon. If neither new encoder beats the control, that is
    evidence the 3D pretraining corpora do not transfer to CYP inhibition, not that the
    readout failed: the readout is identical.
    """
    )
    return


@app.cell
def _():
    import os
    import sys
    import time
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import polars as pl

    # This notebook lives in notebooks/; the package lives in src/.
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from cyp import constants as C
    from cyp import cv, data, encoders, evaluation, mcs, timings
    from cyp import download as cyp_download

    return (
        C,
        Path,
        PROJECT_ROOT,
        cv,
        cyp_download,
        data,
        encoders,
        evaluation,
        mcs,
        mo,
        np,
        os,
        pl,
        time,
        timings,
    )


@app.cell
def _(Path, PROJECT_ROOT, os):
    NOTEBOOK_NAME = Path(__file__).stem
    OUT_DIR = PROJECT_ROOT / "experiments" / NOTEBOOK_NAME
    CACHE_DIR = OUT_DIR / "cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TIMING_LOG = OUT_DIR / "timings.csv"

    # Folds are the cache key (fold-major, per CLAUDE.md), so a run limited to the first
    # k folds is a valid smaller paired comparison and a later full run reuses it.
    N_FOLDS = int(os.environ.get("CYP_16_FOLDS", "25"))

    # The control: TabICL on frozen CheMeleon, single-task, 03's shared folds.
    CONTROL_OOF = (
        PROJECT_ROOT / "experiments" / "03_methods" / "cache" / "mt_tabicl_singletask_full.parquet"
    )
    CHEMELEON_CACHE = PROJECT_ROOT / "experiments" / "03_methods" / "cache"
    return CACHE_DIR, CHEMELEON_CACHE, CONTROL_OOF, N_FOLDS, NOTEBOOK_NAME, OUT_DIR, TIMING_LOG


@app.cell
def _(C, cv, data, mo, pl):
    # One fold assignment over the union of the four endpoints, as in 03. Recomputed
    # rather than loaded so the control's alignment is *checked* below, not assumed.
    frames = {_e: data.training_frame(_e) for _e in C.REGRESSION_ENDPOINTS}
    _fold_of, FOLD_ASSIGNMENTS = cv.shared_scaffold_folds(frames)

    _leaks = 0
    for _fold in FOLD_ASSIGNMENTS["fold"].unique().to_list():
        _test, _train = set(), set()
        for _e, _f in frames.items():
            for _fd, _o, _i, _tr, _v, _te in cv.fold_assignment_splits(_f, FOLD_ASSIGNMENTS):
                if _fd == _fold:
                    _test |= set(_te["Molecule_Name"].to_list())
                    _train |= set(_tr["Molecule_Name"].to_list())
        _leaks += len(_test & _train)
    assert _leaks == 0, (
        f"{_leaks} compounds leak across endpoints; the paired comparison is invalid"
    )

    mo.md(
        f"{FOLD_ASSIGNMENTS['Molecule_Name'].n_unique():,} compounds in "
        f"{FOLD_ASSIGNMENTS['fold'].n_unique()} folds. No cross-endpoint leakage."
    )
    return FOLD_ASSIGNMENTS, frames


@app.cell
def _(CACHE_DIR, CHEMELEON_CACHE, C, encoders, frames, mo, np, pl):
    # Embed the union of SMILES once (about an hour for both encoders on this CPU) and
    # index back per endpoint, so the 1,309 multi-endpoint compounds are not paid for
    # twice. Cached by a hash of the compound set, so a new data snapshot re-embeds.
    UNION = sorted({_s for _f in frames.values() for _s in _f["SMILES"].to_list()})
    _row_of = {_s: _i for _i, _s in enumerate(UNION)}

    _monroe = encoders.embed_cached("monroe", UNION, CACHE_DIR)
    _jepa = encoders.embed_cached("atom_jepa", UNION, CACHE_DIR)

    ARMS = ("chemeleon", "monroe", "atom_jepa")
    features = {"chemeleon": {}, "monroe": {}, "atom_jepa": {}}
    for _e, _f in frames.items():
        _idx = [_row_of[_s] for _s in _f["SMILES"].to_list()]
        features["monroe"][_e] = _monroe[_idx]
        # Last encoder block, the package default. Axis 1 is block 1..8.
        features["atom_jepa"][_e] = _jepa[_idx, -1, :]
        _chem = np.load(CHEMELEON_CACHE / f"chemeleon_{_e}.npy")
        assert _chem.shape[0] == _f.height, "CheMeleon cache is not row-aligned to the frame"
        features["chemeleon"][_e] = _chem

    pl.DataFrame(
        {
            "arm": list(ARMS),
            "n_features": [features[_a][C.REGRESSION_ENDPOINTS[0]].shape[1] for _a in ARMS],
            "mean_abs": [
                float(np.abs(features[_a][C.REGRESSION_ENDPOINTS[0]]).mean()) for _a in ARMS
            ],
        }
    )
    return ARMS, features


@app.cell
def _(CONTROL_OOF, FOLD_ASSIGNMENTS, N_FOLDS, frames, pl):
    # The control only means something if it is on exactly these folds and these labels.
    control = pl.read_parquet(CONTROL_OOF).with_columns(pl.lit("tabicl_chemeleon").alias("method"))
    _fold_ids = sorted(FOLD_ASSIGNMENTS["fold"].unique().to_list())[:N_FOLDS]
    control = control.filter(pl.col("fold").is_in(_fold_ids))

    # A compound appears once per outer repeat, so the key is (compound, fold). Every
    # control row must be a pair this notebook's assignment also contains.
    _expected = FOLD_ASSIGNMENTS.select("Molecule_Name", "fold")
    _orphans = control.join(_expected, on=["Molecule_Name", "fold"], how="anti")
    assert _orphans.height == 0, f"{_orphans.height} control rows are on different folds"

    for _e, _f in frames.items():
        _lab = _f.select("Molecule_Name", pl.col("y_true").alias("y_expected"))
        _c = control.filter(pl.col("endpoint") == _e).join(_lab, on="Molecule_Name", how="left")
        assert (_c["y_true"] - _c["y_expected"]).abs().max() < 1e-9, f"{_e}: control y_true differs"
    control.group_by("endpoint").len().sort("endpoint")
    return (control,)


@app.cell
def _(
    ARMS,
    CACHE_DIR,
    FOLD_ASSIGNMENTS,
    N_FOLDS,
    TIMING_LOG,
    cv,
    features,
    frames,
    mo,
    np,
    pl,
    time,
    timings,
):
    from cyp.tabular_models import predict_subprocess

    # Materialise the splits once per endpoint, keyed by fold, so the loop below can be
    # fold-major: a run killed at fold k leaves every arm complete through fold k.
    _splits = {}
    for _e, _f in frames.items():
        _indexed = _f.with_row_index("_row")
        _splits[_e] = {
            _fd: (_tr, _te)
            for _fd, _o, _i, _tr, _v, _te in cv.fold_assignment_splits(_indexed, FOLD_ASSIGNMENTS)
        }

    _folds = sorted(FOLD_ASSIGNMENTS["fold"].unique().to_list())[:N_FOLDS]
    _new_arms = [_a for _a in ARMS if _a != "chemeleon"]
    _parts = []
    with mo.status.progress_bar(total=len(_folds) * len(_new_arms), title="TabICL CV") as _bar:
        for _fd in _folds:
            for _arm in _new_arms:
                _cache = CACHE_DIR / f"oof_{_arm}_fold{_fd:02d}.parquet"
                if _cache.exists():
                    _parts.append(pl.read_parquet(_cache))
                    _bar.update()
                    continue
                _t0 = time.time()
                _rows = []
                for _e, _by_fold in _splits.items():
                    _tr, _te = _by_fold[_fd]
                    _X = features[_arm][_e]
                    # A child process, never in-process: torch in the parent would make
                    # a later LightGBM fit segfault (CLAUDE.md, environment traps).
                    _p, _, _ = predict_subprocess(
                        _X[_tr["_row"].to_numpy()],
                        _tr["y_true"].to_numpy(),
                        _X[_te["_row"].to_numpy()],
                        kind="tabicl",
                    )
                    _p = np.asarray(_p, dtype=float)
                    assert np.isfinite(_p).all(), f"{_arm}/{_e}/fold {_fd}: non-finite predictions"
                    _rows.append(
                        _te.select(
                            pl.lit(f"tabicl_{_arm}").alias("method"),
                            pl.lit(_e).alias("endpoint"),
                            pl.lit(_fd).alias("fold"),
                            "Molecule_Name",
                            "y_true",
                            pl.Series("y_pred", _p),
                            "y_lower",
                            "y_upper",
                        )
                    )
                _out = pl.concat(_rows)
                _out.write_parquet(_cache)
                timings.record(
                    TIMING_LOG,
                    stage="tabicl_cv",
                    method=_arm,
                    endpoint="all_four",
                    mode=f"fold_{_fd}",
                    seconds=time.time() - _t0,
                )
                _parts.append(_out)
                _bar.update()

    new_oof = pl.concat(_parts, how="vertical_relaxed")
    new_oof.group_by("method").len()
    return (new_oof,)


@app.cell
def _(C, OUT_DIR, control, evaluation, new_oof, pl):
    # Align schemas: the control carries fold bookkeeping columns the new arms do not.
    _cols = [
        "method",
        "endpoint",
        "fold",
        "Molecule_Name",
        "y_true",
        "y_pred",
        "y_lower",
        "y_upper",
    ]
    oof = pl.concat([control.select(_cols), new_oof.select(_cols)], how="vertical_relaxed")
    oof.write_parquet(OUT_DIR / "oof.parquet")

    fold_scores = evaluation.fold_metrics(oof)
    macro_folds = evaluation.macro_averaged_fold_metrics(
        {_e: fold_scores.filter(pl.col("endpoint") == _e) for _e in C.REGRESSION_ENDPOINTS},
        metric_col="st_rae",
    )
    macro_table = (
        macro_folds.group_by("method")
        .agg(
            pl.col("st_rae").mean().round(4).alias("macro_st_rae"),
            pl.col("st_rae").std().round(4).alias("std"),
            pl.len().alias("n_folds"),
        )
        .sort("macro_st_rae")
    )
    per_endpoint = (
        fold_scores.group_by("method", "endpoint")
        .agg(pl.col("st_rae").mean().round(4).alias("st_rae"))
        .pivot(on="endpoint", index="method", values="st_rae")
        .sort("method")
    )
    macro_table.write_csv(OUT_DIR / "macro_table.csv")
    per_endpoint.write_csv(OUT_DIR / "per_endpoint.csv")
    macro_table
    return macro_folds, macro_table, oof, per_endpoint


@app.cell
def _(per_endpoint):
    per_endpoint
    return


@app.cell
def _(C, OUT_DIR, evaluation, mo, oof, pl):
    # Paired bootstrap of each new encoder against the control, per endpoint, then Holm
    # over the whole family. Per-endpoint rather than pooled: the endpoints are not
    # exchangeable (02's scorecard), and the rank is a macro of per-endpoint scores.
    _ref = "tabicl_tabicl_chemeleon"
    _control = "tabicl_chemeleon"
    _rows = []
    for _arm in ("tabicl_monroe", "tabicl_atom_jepa"):
        for _e in C.REGRESSION_ENDPOINTS:
            _a = oof.filter((pl.col("method") == _arm) & (pl.col("endpoint") == _e))
            _b = oof.filter((pl.col("method") == "tabicl_chemeleon") & (pl.col("endpoint") == _e))
            _w = _a.select(
                "Molecule_Name",
                "fold",
                "y_true",
                "y_lower",
                "y_upper",
                pl.col("y_pred").alias("pa"),
            ).join(
                _b.select("Molecule_Name", "fold", pl.col("y_pred").alias("pb")),
                on=["Molecule_Name", "fold"],
                how="inner",
            )
            _r = evaluation.paired_bootstrap(
                _w["y_true"].to_numpy(),
                _w["pa"].to_numpy(),
                _w["pb"].to_numpy(),
                metric="st_rae",
                y_lower=_w["y_lower"].to_numpy(),
                y_upper=_w["y_upper"].to_numpy(),
            )
            _rows.append({"arm": _arm, "endpoint": _e, "n": _w.height, **_r})
    paired = pl.DataFrame(_rows)
    _holm = evaluation.holm_bonferroni(
        {f"{_r['arm']}|{_r['endpoint']}": _r["p_value"] for _r in _rows}
    )
    paired = (
        paired.with_columns((pl.col("arm") + "|" + pl.col("endpoint")).alias("comparison"))
        .join(_holm.select("comparison", "significant"), on="comparison")
        .drop("comparison")
    )
    paired.write_csv(OUT_DIR / "paired_vs_chemeleon.csv")
    mo.vstack(
        [
            mo.md(
                "**Each new encoder minus the CheMeleon control** (ST-RAE; negative favours "
                "the new encoder). `significant` is Holm-corrected over all eight comparisons."
            ),
            paired,
        ]
    )
    return (paired,)


@app.cell
def _(OUT_DIR, macro_folds, mcs):
    forest = mcs.reference_forest_plot(
        macro_folds,
        metric_col="st_rae",
        higher_is_better=False,
        reference="tabicl_chemeleon",
        title="Macro ST-RAE vs the CheMeleon TabICL control (5x5 scaffold CV)",
        save_path=OUT_DIR / "forest_vs_chemeleon.png",
    )
    forest
    return


@app.cell
def _(C, OUT_DIR, mo, np, oof, pl):
    # Pairwise correlation of OOF predictions. 14 showed an ensemble pays at r 0.30-0.62
    # and not at r ~0.94; this says whether either encoder is a candidate member at all
    # before anyone spends time on a blend.
    _wide = oof.pivot(
        on="method", index=["endpoint", "fold", "Molecule_Name"], values="y_pred"
    ).drop_nulls()
    _methods = [_c for _c in _wide.columns if _c.startswith("tabicl_")]
    _rows = []
    for _e in C.REGRESSION_ENDPOINTS:
        _sub = _wide.filter(pl.col("endpoint") == _e)
        for _i, _a in enumerate(_methods):
            for _b in _methods[_i + 1 :]:
                _rows.append(
                    {
                        "endpoint": _e,
                        "pair": f"{_a} vs {_b}",
                        "pearson_r": round(
                            float(np.corrcoef(_sub[_a].to_numpy(), _sub[_b].to_numpy())[0, 1]), 3
                        ),
                    }
                )
    correlation = pl.DataFrame(_rows)
    correlation.write_csv(OUT_DIR / "prediction_correlation.csv")
    mo.vstack(
        [
            mo.md("**Pearson r between arms' OOF predictions, per endpoint**"),
            correlation.pivot(on="endpoint", index="pair", values="pearson_r"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
