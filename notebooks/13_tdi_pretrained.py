import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # 13 — TDI: does the pretrained encoder transfer to the classification track?

    TDI has had one day of work against regression's ten. `04` shipped
    `tabicl_singletask` (macro MCC 0.2483 in CV) on 2026-09-15 and nothing has touched
    the track since. The live board says **0.2111, rank 94/113** — against a visible
    contender at **0.4025, rank 12**, whose TDI stack is architecturally what we
    already have (TabICL on frozen embeddings, threshold swept on OOF). That is the
    largest single deficit anywhere in this repo, and the cheapest to attack, because
    the thing their stack has that ours does not is the thing `05` already proved is
    the biggest lever here: **a CYP-pretrained encoder**.

    CLAUDE.md records the PubChem-pretrained chemprop encoder as the single largest
    gain on the regression track (0.7304 → 0.7160 macro ST-RAE, and pretraining helps
    only with the encoder *unfrozen*). It also records that it has **never been tried
    on TDI**. This notebook tries it.

    ## The arms

    | arm | encoder | what it isolates |
    |:--|:--|:--|
    | `tabicl_singletask` | frozen CheMeleon | `04`'s incumbent, the thing to beat |
    | `chemprop_scratch` | none, trained from scratch | chemprop's own floor on this track |
    | `chemprop_pubchem` | PubChem-pretrained, unfrozen | **the question** |
    | `chemprop_pubchem_frozen` | PubChem-pretrained, frozen | `05`'s freeze/unfreeze contrast |

    All on shared folds (`cv.shared_scaffold_folds`), so `evaluation.paired_bootstrap`
    between any two is a genuine paired test. `chemprop_scratch` is the control that
    separates "pretraining helped" from "chemprop is just better here" — without it a
    win by `chemprop_pubchem` over TabICL cannot be attributed.

    ## What the plumbing cost, and why it is worth writing down

    Two silent failures were found by direct probe while building this, both now
    pinned by tests in `tests/test_auxiliary.py`:

    1. **A regression pretrain checkpoint corrupts a classification fine-tune, without
       raising.** chemprop's `load_and_use_pretrained_model_scalers` applies the
       checkpoint's `UnscaleTransform` to the new targets whenever `--checkpoint` is
       passed, and there is no flag to skip it. With mismatched head counts it raises
       (`operands could not be broadcast together with shapes (250,2) (1,3)`). With
       matching head counts it is worse: training completes and `predict` returns
       values in the **pIC50 range** rather than probabilities — measured here as a
       "probability" of **−0.525** — which reads downstream as a model that learned
       nothing. The fix is to binarise the pretraining panel and pretrain as
       *classification*, so no `UnscaleTransform` is ever written
       (`aux_training.binarise_pretraining_matrix`).
    2. **The two stages must have the same number of heads**, for the same scaler
       reason. `run_cv_pretrained_classification` checks this before the fold
       assignment, because the mismatch otherwise surfaces at fine-tune time — after a
       pretrain that runs ~40 minutes on the full corpus.
    """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import sys
    from pathlib import Path

    import numpy as np
    import polars as pl

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from cyp import aux_training, cv, data, evaluation, multitask, timings
    from cyp import constants as C

    return (
        C,
        PROJECT_ROOT,
        Path,
        aux_training,
        cv,
        data,
        evaluation,
        multitask,
        np,
        pl,
        timings,
    )


@app.cell
def _(Path, PROJECT_ROOT):
    NOTEBOOK_NAME = Path(__file__).stem
    OUT_DIR = PROJECT_ROOT / "experiments" / NOTEBOOK_NAME
    CACHE_DIR = OUT_DIR / "cache"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR, NOTEBOOK_NAME, OUT_DIR


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Data, and the co-measurement gap that shapes the whole track

    `04` found multitask does not transfer to TDI and traced it to co-measurement:
    26.7% of regression compounds carry several endpoints against **5.4%** of TDI
    compounds. That number is worth re-deriving rather than trusting, because it is
    the reason this notebook runs a *multitarget* chemprop (two heads, one shared
    encoder) rather than expecting multitask magic — the shared encoder is doing the
    work, not the shared rows.
    """
    )
    return


@app.cell
def _(C, data, mo, pl):
    tdi_raw = data.load_train_tdi()

    # `data.tdi_training_frame` rather than an inline filter: it is what `04` used,
    # so this notebook's frames are row-for-row identical to the ones `04`'s cached
    # CheMeleon embeddings were computed against, and those embeddings can be reused
    # without recomputing or risking a silent row misalignment.
    frames = {f"{iso}_is_TDI": data.tdi_training_frame(iso) for iso in C.TDI_ISOFORMS}
    TDI_ENDPOINTS = list(frames)

    _both = tdi_raw.filter(
        pl.col("CYP3A4_is_TDI").is_not_null() & pl.col("CYP2D6_is_TDI").is_not_null()
    ).height
    _union = pl.concat([f.select("Molecule_Name") for f in frames.values()]).unique().height

    label_table = pl.DataFrame(
        [
            {
                "endpoint": endpoint,
                "n": frame.height,
                "positives": int(frame["y_true"].sum()),
                "rate": round(float(frame["y_true"].mean()), 4),
            }
            for endpoint, frame in frames.items()
        ]
    )

    mo.vstack(
        [
            mo.ui.table(label_table, selection=None),
            mo.md(
                f"**{_both} of {_union} compounds carry both TDI labels "
                f"({100 * _both / _union:.1f}%)** — `04`'s co-measurement gap, "
                "reproduced. Multitask across two tasks this weakly shared is not "
                "where a gain comes from; a pretrained encoder is."
            ),
        ]
    )
    return TDI_ENDPOINTS, frames, tdi_raw


@app.cell
def _(mo):
    mo.md(
        r"""
    ## The two-stage route is closed first, cheaply

    Before building anything: `is_TDI` is *defined* as a pIC50 shift between the
    direct and preincubated conditions, and both pIC50 columns have near-complete
    coverage on TDI-labelled compounds (1,497 of 1,497 on CYP2D6) where the label
    itself is what is scarce. That is exactly the shape of a two-stage opportunity —
    predict the shift, threshold it — so it has to be ruled out before the expensive
    arm runs.

    `08`'s rule is the test: **an auxiliary is usable as a two-stage feature only when
    it is more predictable from structure than the target is.** Correlation with the
    target is necessary and nowhere near sufficient.
    """
    )
    return


@app.cell
def _(C, OUT_DIR, cv, data, mo, np, pl, tdi_raw):
    import lightgbm as _lgb
    from scipy import stats as _stats
    from sklearn.metrics import matthews_corrcoef as _mcc

    from cyp import fingerprints as _fp

    _inh = data.load_train_inhibition()
    _rows = []
    for _iso in C.TDI_ISOFORMS:
        _lab = tdi_raw.filter(pl.col(f"{_iso}_is_TDI").is_not_null())
        _joined = _lab.join(
            _inh.select("Molecule_Name", f"{_iso}_pIC50_direct_inhibition"),
            on="Molecule_Name",
            how="left",
        )
        _d = _joined.filter(
            pl.col(f"{_iso}_pIC50_TDI_condition").is_not_null()
            & pl.col(f"{_iso}_pIC50_direct_inhibition").is_not_null()
        ).with_row_index("ridx")

        _shift = (
            _d[f"{_iso}_pIC50_TDI_condition"] - _d[f"{_iso}_pIC50_direct_inhibition"]
        ).to_numpy()
        _y = _d[f"{_iso}_is_TDI"].to_numpy().astype(bool)
        _X = _fp.compute(_d["SMILES"].to_list(), "ecfp", n_bits=2048)

        # The oracle: best MCC obtainable by thresholding the TRUE shift. This is the
        # ceiling the two-stage route would reach with a perfect stage 1.
        _oracle = max((_mcc(_y, _shift >= t), t) for t in np.arange(-0.5, 2.0, 0.01))

        _folds = [
            (tr["ridx"].to_numpy(), te["ridx"].to_numpy())
            for _f, _o, _i, tr, _v, te in cv.scaffold_splits(_d, n_outer=3, n_inner=5, seed=0)
        ]
        _oof_shift = np.full(len(_y), np.nan)
        _oof_direct = np.full(len(_y), np.nan)
        for _tr, _te in _folds:
            _r = _lgb.LGBMRegressor(n_estimators=300, random_state=0, verbose=-1)
            _r.fit(_X[_tr], _shift[_tr])
            _oof_shift[_te] = _r.predict(_X[_te])
            _c = _lgb.LGBMClassifier(
                n_estimators=300, random_state=0, verbose=-1, class_weight="balanced"
            )
            _c.fit(_X[_tr], _y[_tr])
            _oof_direct[_te] = _c.predict_proba(_X[_te])[:, 1]

        _two_stage = max((_mcc(_y, _oof_shift >= t), t) for t in np.arange(-0.5, 2.0, 0.01))
        _direct = max((_mcc(_y, _oof_direct >= t), t) for t in np.arange(0.05, 0.95, 0.01))

        _rows.append(
            {
                "endpoint": _iso,
                "n": len(_y),
                "shift_vs_label_r": round(float(_stats.pointbiserialr(_y, _shift).statistic), 3),
                "oracle_mcc": round(_oracle[0], 4),
                "shift_predictability": round(
                    float(_stats.spearmanr(_shift, _oof_shift).statistic), 4
                ),
                "two_stage_mcc": round(_two_stage[0], 4),
                "direct_mcc": round(_direct[0], 4),
                "two_stage_lift": round(_two_stage[0] - _direct[0], 4),
            }
        )

    two_stage = pl.DataFrame(_rows)
    two_stage.write_csv(OUT_DIR / "two_stage_check.csv")
    mo.ui.table(two_stage, selection=None)
    return (two_stage,)


@app.cell
def _(mo):
    mo.md(
        r"""
    **The two-stage route is dead, and it is `08`'s finding again in a new place.** The
    oracle is enormous — thresholding the *true* shift scores MCC 0.69 (CYP3A4) and
    0.90 (CYP2D6), against our board's 0.30 and 0.12 — so the quantity genuinely
    carries the label. But the shift is barely predictable from structure (Spearman
    0.12–0.22), and the two-stage arm **loses to a plain direct classifier at both
    isoforms** (−0.075 and −0.044).

    Same shape as `08`'s Emax result: a huge oracle, a near-useless stage 1, and a
    repair that is worse than doing nothing. Read the predictability column before the
    oracle column, every time.

    Note the difference from `08` though: there the cause was *coverage* (Emax is
    perfectly co-measured with pIC50, so stage 1 gains nothing). Here coverage is
    excellent and the cause is pure *predictability* — the shift is a difference of two
    noisy quantities, and differencing amplifies noise. Two distinct ways for the same
    route to fail.
    """
    )
    return


@app.cell
def _(C, PROJECT_ROOT, TDI_ENDPOINTS, frames, mo, np):
    # The incumbent's representation: frozen CheMeleon embeddings, exactly as `04`
    # built them. Reused from `04`'s cache rather than recomputed -- the frames are
    # row-for-row identical (both come from `data.tdi_training_frame`), verified
    # before wiring this up, and a CheMeleon embedding pass is minutes of GPU-less
    # torch work for a result that already exists on disk.
    _cache_04 = PROJECT_ROOT / "experiments" / "04_methods_tdi" / "cache"

    chemeleon_features = {}
    for _endpoint in TDI_ENDPOINTS:
        _iso = _endpoint.split("_")[0]
        _path = _cache_04 / f"chemeleon_tdi_{_iso}.npy"
        if _path.exists():
            _emb = np.load(_path)
        else:
            from cyp.graph_models import chemeleon_embed

            _smiles = frames[_endpoint]["SMILES"].to_list()
            _emb, _ = chemeleon_embed(_smiles, _smiles[:1])
            np.save(_path, _emb)
        # A row-count mismatch means the frames have drifted from `04`'s and the
        # embeddings are no longer aligned -- fail here rather than train on
        # mismatched rows, which no metric would reveal.
        assert _emb.shape[0] == frames[_endpoint].height, (
            f"{_endpoint}: {_emb.shape[0]} embedding rows against "
            f"{frames[_endpoint].height} frame rows"
        )
        chemeleon_features[_endpoint] = _emb

    mo.md(
        "Frozen CheMeleon embeddings for the incumbent arm: "
        + ", ".join(f"`{e.split('_')[0]}` {chemeleon_features[e].shape}" for e in TDI_ENDPOINTS)
    )
    return (chemeleon_features,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## The pretraining corpus

    The public panel has no TDI label anywhere — no public assay runs a preincubation
    arm at this scale — so the encoder cannot be pretrained on the target. What it
    *can* learn is CYP binding chemistry in general, which is the transfer `05`
    measured as the largest single lever in this repo.

    Binarised at pIC50 5.0, for the scaler reason in this notebook's header. Nulls stay
    null: an untested compound is not a confirmed inactive.
    """
    )
    return


@app.cell
def _(C, TDI_ENDPOINTS, aux_training, data, mo, pl):
    _inh_all = data.load_train_inhibition()

    # Head order matches `TDI_ENDPOINTS` positionally, and the count must match -- the
    # runner raises otherwise. CYP3A4 first, CYP2D6 second, same as `C.TDI_ISOFORMS`.
    pretraining_raw = aux_training.public_pretraining_matrix(
        [f"{iso}_pIC50_direct_inhibition" for iso in C.TDI_ISOFORMS],
        exclude_smiles=_inh_all["SMILES"].to_list(),
    )
    pretraining = aux_training.binarise_pretraining_matrix(pretraining_raw, threshold=5.0)

    _summary = pl.DataFrame(
        [
            {
                "head": c,
                "labelled": int(pretraining[c].is_not_null().sum()),
                "active_rate": round(float(pretraining[c].mean()), 4),
            }
            for c in pretraining.columns
            if c != "smiles_std"
        ]
    )

    mo.vstack(
        [
            mo.md(
                f"**{pretraining.height:,} compounds**, challenge structures excluded "
                f"via `exclude_smiles` (InChIKey skeleton keying, per CLAUDE.md). "
                f"{len(TDI_ENDPOINTS)} heads, matching the {len(TDI_ENDPOINTS)} "
                "fine-tuning endpoints."
            ),
            mo.ui.table(_summary, selection=None),
        ]
    )
    return (pretraining,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## The CV run

    Fold-major, cached per `(arm, fold)`, per CLAUDE.md's rule: a run killed at hour
    three leaves every arm complete through fold *k*, which is a smaller but fully
    usable paired comparison. Method-major caching would leave some arms complete and
    the rest missing, and no comparison at all.
    """
    )
    return


@app.cell
def _(
    CACHE_DIR,
    OUT_DIR,
    TDI_ENDPOINTS,
    aux_training,
    chemeleon_features,
    cv,
    frames,
    mo,
    multitask,
    pl,
    pretraining,
    timings,
):
    import time as _time

    N_OUTER, N_INNER = 5, 5
    _, ASSIGNMENTS = cv.shared_scaffold_folds(frames, n_outer=N_OUTER, n_inner=N_INNER, seed=42)
    N_FOLDS = ASSIGNMENTS["fold"].n_unique()

    # Chemprop settings shared by every chemprop arm. Both arms of any comparison get
    # identical settings -- an arm trained for more epochs than its twin measures the
    # epochs, not the treatment.
    CHEMPROP_KWARGS = {"epochs": 50, "pretrain_epochs": 30}

    def _run_arm(arm: str, fold: int) -> pl.DataFrame:
        if arm == "tabicl_singletask":
            # `features=` is not optional here, and omitting it does not fail loudly.
            # The default is ECFP-2048, but CLAUDE.md pins TabICL at
            # `TABICL_MAX_FEATURES=128` for memory reasons -- run it on 2048 raw bits
            # and it returns **all-NaN probabilities**, which score MCC 0.000 and read
            # as "the incumbent is terrible" rather than as a configuration error.
            # Measured directly: CYP3A4 came back 100% NaN across every fold, silently.
            # `04` passed frozen CheMeleon embeddings; so does this.
            return multitask.run_cv_singletask_classification(
                frames,
                method="tabicl",
                features=chemeleon_features,
                assignments=ASSIGNMENTS,
                folds=[fold],
                seed=42,
            )
        if arm == "chemprop_scratch":
            # `run_cv_multitarget_classification` derives its own method name
            # ("chemprop_multitask") rather than taking one, so it is relabelled here
            # -- the arm is the same model with no pretraining, and calling it what
            # this notebook calls it keeps every downstream group_by on one key.
            return multitask.run_cv_multitarget_classification(
                frames,
                assignments=ASSIGNMENTS,
                folds=[fold],
                seed=42,
                **{k: v for k, v in CHEMPROP_KWARGS.items() if k != "pretrain_epochs"},
            ).with_columns(pl.lit(arm).alias("method"))
        return aux_training.run_cv_pretrained_classification(
            frames,
            pretraining,
            method_name=arm,
            assignments=ASSIGNMENTS,
            folds=[fold],
            seed=42,
            freeze_encoder=(arm == "chemprop_pubchem_frozen"),
            **CHEMPROP_KWARGS,
        )

    ARMS = (
        "tabicl_singletask",
        "chemprop_scratch",
        "chemprop_pubchem",
        "chemprop_pubchem_frozen",
    )

    _pieces = []
    with mo.status.progress_bar(total=N_FOLDS * len(ARMS)) as _bar:
        for _fold in range(N_FOLDS):
            for _arm in ARMS:
                _cache = CACHE_DIR / f"{_arm}_fold{_fold}.parquet"
                if _cache.exists():
                    _pieces.append(pl.read_parquet(_cache))
                    _bar.update()
                    continue
                _t0 = _time.time()
                _piece = _run_arm(_arm, _fold)
                timings.record(
                    OUT_DIR / "timings.csv",
                    stage=f"fold{_fold}",
                    method=_arm,
                    endpoint="both",
                    mode="full",
                    seconds=_time.time() - _t0,
                )
                _piece.write_parquet(_cache)
                _pieces.append(_piece)
                _bar.update()

    oof = pl.concat(_pieces, how="diagonal_relaxed")
    oof.write_parquet(OUT_DIR / "oof.parquet")
    mo.md(f"**{oof.height:,} OOF rows** across {len(ARMS)} arms and {N_FOLDS} folds.")
    return ARMS, N_FOLDS, oof


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Threshold sweep, then the comparison

    `04`'s hardest-won TDI lesson: **the threshold matters more than the model, and 0.5
    is badly wrong** — every optimum there fell in 0.10–0.25, and at 0.5 both chemprop
    arms scored exactly 0.000, which is easy to misread as "the graph model learned
    nothing". The OOF frame carries `y_prob`, so the sweep reproduces any threshold
    without a refit.

    A threshold tuned on OOF is honest for *choosing* between arms and mildly
    optimistic as a *reported* score. Both arms are tuned the same way, so the
    comparison is fair even though the absolute number is not a clean forecast.
    """
    )
    return


@app.cell
def _(ARMS, OUT_DIR, TDI_ENDPOINTS, mo, np, oof, pl):
    from sklearn.metrics import matthews_corrcoef as _mcc2

    THRESHOLDS = np.arange(0.05, 0.96, 0.05)

    _rows = []
    for _arm in ARMS:
        for _t in THRESHOLDS:
            _per_endpoint = []
            for _endpoint in TDI_ENDPOINTS:
                _s = oof.filter((pl.col("method") == _arm) & (pl.col("endpoint") == _endpoint))
                if _s.height == 0:
                    continue
                _per_endpoint.append(_mcc2(_s["y_true"].to_numpy(), _s["y_prob"].to_numpy() >= _t))
            if _per_endpoint:
                _rows.append(
                    {
                        "arm": _arm,
                        "threshold": round(float(_t), 2),
                        "macro_mcc": round(float(np.mean(_per_endpoint)), 4),
                    }
                )

    sweep = pl.DataFrame(_rows)
    sweep.write_csv(OUT_DIR / "threshold_sweep.csv")

    best = (
        sweep.sort("macro_mcc", descending=True)
        .group_by("arm", maintain_order=True)
        .first()
        .sort("macro_mcc", descending=True)
    )
    best.write_csv(OUT_DIR / "best_thresholds.csv")
    mo.ui.table(best, selection=None)
    return best, sweep


@app.cell
def _(OUT_DIR, TDI_ENDPOINTS, best, mo, np, oof, pl):
    from sklearn.metrics import matthews_corrcoef as _mcc3

    _best_t = dict(zip(best["arm"].to_list(), best["threshold"].to_list(), strict=True))

    _rows = []
    for _arm, _t in _best_t.items():
        for _endpoint in TDI_ENDPOINTS:
            _s = oof.filter((pl.col("method") == _arm) & (pl.col("endpoint") == _endpoint))
            if _s.height == 0:
                continue
            _per_fold = [
                _mcc3(
                    _f["y_true"].to_numpy(),
                    _f["y_prob"].to_numpy() >= _t,
                )
                for _, _f in _s.group_by("fold")
            ]
            _rows.append(
                {
                    "arm": _arm,
                    "endpoint": _endpoint,
                    "threshold": _t,
                    "mcc": round(float(np.mean(_per_fold)), 4),
                    "std": round(float(np.std(_per_fold)), 4),
                }
            )

    per_endpoint = pl.DataFrame(_rows)
    per_endpoint.write_csv(OUT_DIR / "per_endpoint_mcc.csv")
    mo.ui.table(per_endpoint.pivot(values="mcc", index="arm", on="endpoint"), selection=None)
    return (per_endpoint,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Is the difference one the CV can resolve?

    CLAUDE.md's standing rule, and PXR's most expensive mistake: **do not rank models
    on differences the data cannot resolve.** Three PXR finalists spanned 0.0039 MAE,
    the ordering reversed between phases, and it cost five places. A point estimate is
    not a result until `paired_bootstrap` says it is.
    """
    )
    return


@app.cell
def _(ARMS, OUT_DIR, TDI_ENDPOINTS, best, evaluation, mo, np, oof, pl):
    # `paired_bootstrap` resamples *compounds*, scoring both arms on the same
    # resample -- which is what makes it paired, and why both arms must be aligned
    # row for row on (endpoint, fold, compound) before it is called. The arms share
    # folds by construction (`cv.shared_scaffold_folds`), so the join below should
    # drop nothing; the assert is there because a silent partial join would quietly
    # turn a paired test into an unpaired one.
    _best_t = dict(zip(best["arm"].to_list(), best["threshold"].to_list(), strict=True))
    INCUMBENT = "tabicl_singletask"

    def _labels(arm: str) -> pl.DataFrame:
        return (
            oof.filter(pl.col("method") == arm)
            .with_columns((pl.col("y_prob") >= _best_t[arm]).alias(arm))
            .select("endpoint", "fold", "Molecule_Name", "y_true", arm)
        )

    _base = _labels(INCUMBENT)
    _rows, _pvalues = [], {}
    for _arm in [a for a in ARMS if a != INCUMBENT]:
        _joined = _base.join(
            _labels(_arm).select("endpoint", "fold", "Molecule_Name", _arm),
            on=["endpoint", "fold", "Molecule_Name"],
            how="inner",
        )
        assert _joined.height == _base.height, (
            f"{_arm} does not align with {INCUMBENT} row for row -- "
            "the bootstrap below would not be paired"
        )
        _res = evaluation.paired_bootstrap(
            _joined["y_true"].to_numpy(),
            _joined[_arm].to_numpy(),
            _joined[INCUMBENT].to_numpy(),
            metric="mcc",
            seed=0,
        )
        _pvalues[_arm] = _res["p_value"]
        _rows.append(
            {
                "arm": _arm,
                "vs": INCUMBENT,
                "threshold": _best_t[_arm],
                # Positive favours the challenger: `mcc` is a goodness metric.
                "diff": round(_res["diff"], 4),
                "ci_low": round(_res["ci_low"], 4),
                "ci_high": round(_res["ci_high"], 4),
                "p_raw": round(_res["p_value"], 4),
                "n_paired": _joined.height,
            }
        )

    comparison = pl.DataFrame(_rows).join(
        evaluation.holm_bonferroni(_pvalues), left_on="arm", right_on="comparison", how="left"
    )
    comparison.write_csv(OUT_DIR / "paired_comparison.csv")
    mo.ui.table(comparison, selection=None)
    return (comparison,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## How to read this

    **The question is narrow and the control answers it.** If `chemprop_pubchem` beats
    `tabicl_singletask` but `chemprop_scratch` does too, the win is chemprop, not
    pretraining. If `chemprop_pubchem` beats `chemprop_scratch`, pretraining
    transferred — and `05`'s biggest regression lever works on TDI as well, which is a
    reusable finding rather than one submission's luck.

    **`chemprop_pubchem_frozen` is the mechanism check.** `05` found pretraining helps
    *only* with the encoder unfrozen (0.7160 unfrozen vs 0.7523 frozen, the frozen arm
    significant in the wrong direction). If that reproduces here, the gain is in the
    initialisation rather than the features, on both tracks.

    **Do not ship on a point estimate.** Read `paired_comparison.csv` before
    `best_thresholds.csv`. A gap the bootstrap cannot resolve is not a reason to change
    what we submit, and `04`'s incumbent is already validated and shipped.

    **If nothing beats the incumbent**, that is still worth having: it would say the
    TDI gap to the visible contender is not the encoder, which redirects the remaining
    45 days to the ensemble (their stack Caruana-combines two decorrelated candidates;
    ours ships one model) rather than to more pretraining.
    """
    )
    return


if __name__ == "__main__":
    app.run()
