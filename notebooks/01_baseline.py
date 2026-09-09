import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # 01 — Baseline models

    Fingerprint models across **both** challenge tracks, scored under nested
    scaffold cross-validation:

    - **Direct inhibition (regression)** — mean/1-NN baselines, Ridge, LightGBM,
      XGBoost across all four pIC50 endpoints, scored by ST-RAE.
    - **TDI (classification)** — majority/1-NN baselines, LightGBM, XGBoost for
      CYP3A4 and CYP2D6, scored by MCC.

    Purpose is to bank valid, non-trivial submissions for both tracks early — not
    to win. It follows the PXR challenge protocol: anchor against a trivial
    baseline, score under nested scaffold CV, check calibration (regression) or
    class balance (classification), and record everything needed to compare later
    runs honestly. See `../PLAN.md` and `../CLAUDE.md` for why each of these steps
    is here.

    Set `QUICK = True` below for a fast pass (1 outer CV repeat, 1024-bit
    fingerprints) while iterating on this notebook; the committed run uses the full
    5×5 protocol.

    **Caching.** CV is the slow part (5×5 folds × 5 methods × 4 endpoints for
    regression, plus the TDI track) — a cold run takes ~8 minutes, a warm rerun
    ~20 seconds. Each CV cell checks for a cached OOF parquet under
    `experiments/01_baseline/cache/` before training anything, keyed by mode (quick
    vs full) so the two never collide. Delete a cache file (or toggle `QUICK` to
    switch modes) to force a retrain — this mirrors the PXR repo's convention of
    committing slow-to-produce predictions so a fresh clone reproduces results in
    minutes, not hours.

    The final fit-on-everything-and-predict-the-test-set steps are deliberately
    *not* file-cached: they are cheap (one fit per endpoint/isoform, not 25 folds),
    and caching them separately would risk a submission silently going stale
    against a retrained CV cache. They always read whatever is in the OOF cache at
    the time they run.
    """
    )
    return


@app.cell
def _():
    import sys
    from datetime import date
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import polars as pl

    # This notebook lives in notebooks/; the package lives in src/.
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from cyp import calibration, data, evaluation, mcs, models, submission
    from cyp import constants as C
    from cyp import download as cyp_download

    return (
        C,
        Path,
        PROJECT_ROOT,
        calibration,
        cyp_download,
        data,
        date,
        evaluation,
        mcs,
        mo,
        models,
        np,
        pl,
        submission,
    )


@app.cell
def _(mo):
    QUICK = mo.ui.checkbox(label="Quick mode (1 outer repeat, 1024-bit fingerprints)")
    QUICK
    return (QUICK,)


@app.cell
def _(QUICK, mo):
    quick = QUICK.value
    n_outer = 1 if quick else 5
    n_bits = 1024 if quick else 2048
    mo.md(f"Running with `n_outer={n_outer}`, `n_bits={n_bits}`.")
    return n_bits, n_outer, quick


@app.cell
def _(Path, PROJECT_ROOT):
    # Derived from this file's own name so the convention holds for every future
    # notebook without hardcoding it twice: experiments/<notebook>/ and
    # submissions/<notebook>/<date>/. Defined early since plots are saved here too.
    NOTEBOOK_NAME = Path(__file__).stem
    OUT_DIR = PROJECT_ROOT / "experiments" / NOTEBOOK_NAME
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return NOTEBOOK_NAME, OUT_DIR


@app.cell
def _(OUT_DIR, quick):
    # Cache directory for CV results, keyed by mode so quick/full runs never read
    # each other's cache. This is what makes a rerun with nothing changed fast --
    # see the "Caching" note above.
    CACHE_DIR = OUT_DIR / "cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_SUFFIX = "quick" if quick else "full"
    return CACHE_DIR, CACHE_SUFFIX


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Data

    Download the challenge CSVs into a dated snapshot if none is present yet
    (`data/raw/YYYYMMDD/`). This is a no-op on every run after the first — see
    `cyp.download` and the "Data snapshots" section of `../README.md` for why
    downloads are dated rather than overwritten.
    """
    )
    return


@app.cell
def _(C, cyp_download, mo):
    if not C.available_snapshots():
        mo.output.replace(mo.md("No data snapshot found — downloading now..."))
        cyp_download.download()
    else:
        mo.output.replace(
            mo.md(f"Using existing snapshot `{C.latest_snapshot()}`.")
        )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Model zoo and the endpoint loop

    Two baselines anchor the comparison: `mean` (predict the training mean — scores
    ST-RAE = 1.0 by construction, the line every real model must clear) and `knn`
    (1-NN by Tanimoto similarity — a strong baseline on an analog-rich test set like
    this one). `ridge`, `lgbm` and `xgb` are the candidate models.

    For each endpoint: run nested scaffold CV for every method, summarize scores,
    then sweep calibration (raw / linear / isotonic) on the best non-baseline method.

    Each endpoint's OOF predictions are cached to `cache/oof_<endpoint>_<mode>.parquet`
    -- a rerun with nothing changed loads that file instead of retraining.
    """
    )
    return


@app.cell
def _():
    METHODS = ("mean", "knn", "ridge", "lgbm", "xgb")
    return (METHODS,)


@app.cell
def _(
    CACHE_DIR,
    CACHE_SUFFIX,
    C,
    METHODS,
    calibration,
    data,
    evaluation,
    mo,
    models,
    n_bits,
    n_outer,
    pl,
):
    oof_parts = []
    summaries = []
    calib_parts = []
    endpoint_reports = []
    fold_scores_by_endpoint = {}

    for endpoint in C.REGRESSION_ENDPOINTS:
        # One cache file per endpoint: an endpoint whose cache is missing (e.g. a
        # new endpoint added later) still gets computed without invalidating the
        # others.
        _cache_path = CACHE_DIR / f"oof_{endpoint}_{CACHE_SUFFIX}.parquet"

        frame = data.training_frame(endpoint)

        if _cache_path.exists():
            oof = pl.read_parquet(_cache_path)
        else:
            oof = models.run_cv(
                frame, endpoint, methods=METHODS, n_outer=n_outer, n_bits=n_bits
            )
            oof.write_parquet(_cache_path)
        oof_parts.append(oof)

        summary = evaluation.compare_methods(oof)
        summaries.append(summary)
        # Per-fold scores, kept per endpoint for the MCS heatmap below -- Tukey HSD
        # needs the fold-level values, not the mean/std that `summary` reduces to.
        fold_scores_by_endpoint[endpoint] = evaluation.fold_metrics(oof)

        # Calibration sweep on the best non-baseline method.
        best = (
            summary.filter(~pl.col("method").is_in(["mean", "knn"]))
            .sort("st_rae_mean")
            .row(0, named=True)["method"]
        )
        for kind in ("raw", "linear", "isotonic"):
            calibrated = calibration.crossfit_calibrate(
                oof.filter(pl.col("method") == best).sort("fold"), kind
            )
            score = evaluation.fold_metrics(
                calibrated.with_columns(pl.lit(f"{best}+{kind}").alias("method")),
                pred_col="y_cal",
            )
            calib_parts.append(
                score.group_by(["method", "endpoint"]).agg(
                    pl.col("st_rae").mean().alias("st_rae_mean"),
                    pl.col("st_rae").std().alias("st_rae_std"),
                )
            )

        endpoint_reports.append(
            mo.md(f"**{endpoint}** ({frame.height} labelled compounds) — "
                  f"best non-baseline method: `{best}`")
        )

    mo.vstack(endpoint_reports)
    return (
        best,
        calib_parts,
        endpoint,
        fold_scores_by_endpoint,
        frame,
        kind,
        oof,
        oof_parts,
        summaries,
        summary,
    )


@app.cell
def _(mo):
    mo.md("### CV summary (mean ± std ST-RAE across folds, lower is better)")
    return


@app.cell
def _(pl, summaries):
    all_summary = pl.concat(summaries)
    all_summary.sort(["endpoint", "st_rae_mean"])
    return (all_summary,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Which models are *actually* different? (MCS heatmaps)

    A mean ST-RAE table ranks methods, but PXR's own retrospective is the warning
    here: three finalists spanning 0.0039 MAE looked orderable and were not --
    sorting by a difference that small is sorting by noise, and it cost five
    leaderboard places.

    These heatmaps run a repeated-measures Tukey HSD across the 25 CV folds for
    every pair of methods, per endpoint. Cell = mean ST-RAE difference (row minus
    column; warm means the row method is better, since lower ST-RAE is better
    here); stars = significance after correcting for testing every pair at once
    (`*` p<0.05, `**` p<0.01, `***` p<0.001). A cell with no stars means those two
    methods are not proven different by this CV, however far apart their means
    look -- treat them as tied.

    **These four panels are per-endpoint diagnostics, not the competition metric.**
    The leaderboard ranks on the **macro-average across all four endpoints** (the
    challenge backend's "MA" pseudo-endpoint -- see
    `vendor/cyp_challenge_tutorial/evaluation/evaluate_predictions.py`,
    `compute_macro_bootstrap_results`: a plain mean of the metric across endpoints
    for each resample). A model can look best on one CYP and still lose on rank,
    so the panel below this one -- macro-averaged across endpoints, per fold -- is
    the one that actually mirrors how models get ranked.
    """
    )
    return


@app.cell
def _(OUT_DIR, fold_scores_by_endpoint, mcs):
    mcs_fig = mcs.make_mcs_grid(
        fold_scores_by_endpoint,
        metric_col="st_rae",
        higher_is_better={endpoint: False for endpoint in fold_scores_by_endpoint},
        save_path=OUT_DIR / "mcs_heatmap_regression.png",
    )
    mcs_fig
    return (mcs_fig,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### The metric that actually determines rank: macro-averaged ST-RAE

    For each CV fold, average ST-RAE across all four endpoints -- the CV analogue
    of the leaderboard's "MA" row, via `evaluation.macro_averaged_fold_metrics`
    (see its docstring for exactly how this approximates, but does not
    exactly reproduce, the backend's bootstrap-based macro-average). Then run the
    same Tukey HSD on that single macro series.

    This is the heatmap to actually decide a submission from -- a pair of models
    can each win on some endpoints and still be indistinguishable (or clearly
    ordered) once averaged the way the leaderboard averages them.
    """
    )
    return


@app.cell
def _(OUT_DIR, evaluation, fold_scores_by_endpoint, mcs):
    macro_fold_scores = evaluation.macro_averaged_fold_metrics(
        fold_scores_by_endpoint, metric_col="st_rae"
    )
    macro_mcs_fig = mcs.make_mcs_grid(
        {"MA (macro-average, all 4 endpoints)": macro_fold_scores},
        metric_col="st_rae",
        higher_is_better={"MA (macro-average, all 4 endpoints)": False},
        save_path=OUT_DIR / "mcs_heatmap_regression_macro.png",
    )
    macro_mcs_fig
    return macro_fold_scores, macro_mcs_fig


@app.cell
def _(mo):
    mo.md("### Calibration sweep")
    return


@app.cell
def _(calib_parts, pl):
    calib = pl.concat(calib_parts).sort(["endpoint", "st_rae_mean"])
    calib
    return (calib,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Regression-to-the-mean check

    The dominant PXR failure mode: models underpredict the most potent compounds.
    Under ST-RAE, shrinkage in the top potency bin is the error that actually costs
    — low-potency bins are largely forgiven by wide credible intervals and the
    explicit sub-pIC50-4 downweighting.
    """
    )
    return


@app.cell
def _(evaluation, oof_parts, pl):
    all_oof = pl.concat(oof_parts)
    bias = evaluation.bias_by_potency_bin(all_oof)
    bias
    return all_oof, bias


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Build the submission

    Train the submission method on the full labelled set for each endpoint, apply
    the final calibrator (fit on *all* OOF predictions — cross-fitting is for
    honest evaluation, not for the model that actually goes into the submission),
    and validate against the official checker before writing anything final.
    """
    )
    return


@app.cell
def _():
    SUBMIT_METHOD = "lgbm"
    SUBMIT_CALIBRATION = "linear"
    return SUBMIT_CALIBRATION, SUBMIT_METHOD


@app.cell
def _(
    C,
    SUBMIT_CALIBRATION,
    SUBMIT_METHOD,
    all_oof,
    calibration,
    data,
    models,
    n_bits,
    np,
    pl,
):
    # Not cached to a file: this step is one fit per endpoint (4 total), not the
    # 5x5=25-fold sweep the CV cells run, so it is cheap relative to CV -- and
    # caching it separately would let it silently go stale against a retrained
    # `all_oof` (the CV cache above is what determines the calibrator here). If
    # this step ever gets expensive too, cache it keyed on a hash of `all_oof`
    # itself, not just the run config, so a CV retrain can't be missed.
    test = data.load_test()
    predictions: dict[str, np.ndarray] = {}
    for _endpoint in C.REGRESSION_ENDPOINTS:
        _frame = data.training_frame(_endpoint)
        _raw = models.fit_predict_test(
            _frame, test["SMILES"].to_list(), method=SUBMIT_METHOD, n_bits=n_bits
        )
        _endpoint_oof = all_oof.filter(
            (pl.col("method") == SUBMIT_METHOD) & (pl.col("endpoint") == _endpoint)
        )
        _calibrator = calibration.fit_final_calibrator(
            _endpoint_oof["y_true"].to_numpy(),
            _endpoint_oof["y_pred"].to_numpy(),
            SUBMIT_CALIBRATION,
        )
        predictions[_endpoint] = calibration.apply_calibrator(_calibrator, _raw)
    return predictions, test


@app.cell
def _(mo):
    mo.md(
        r"""
    ## TDI classification track

    The other half of the competition: classify whether a compound shows
    time-dependent inhibition, for CYP3A4 and CYP2D6 only, scored by Matthews
    Correlation Coefficient (MCC) rather than ST-RAE.

    TDI labels are imbalanced (~21–22% positive), which is exactly why MCC is the
    metric and not accuracy — a classifier that always predicts "not TDI" scores
    ~78–79% accuracy while being useless. `majority` (predict the majority class)
    scores MCC = 0.0 by construction and makes that trap visible; `knn` is the same
    Tanimoto 1-NN idea as the regression baseline. `lgbm`/`xgb` use class-balanced
    weighting so the imbalance does not just get learned into the decision boundary.

    Same nested scaffold CV protocol as the regression track, same
    "do not trust a point estimate" discipline — just a different metric. Same
    caching too: OOF predictions land in `cache/tdi_oof_<isoform>_<mode>.parquet`.
    """
    )
    return


@app.cell
def _():
    TDI_METHODS = ("majority", "knn", "lgbm", "xgb")
    return (TDI_METHODS,)


@app.cell
def _(CACHE_DIR, CACHE_SUFFIX, C, TDI_METHODS, data, evaluation, mo, models, n_bits, n_outer, pl):
    tdi_oof_parts = []
    tdi_summaries = []
    tdi_reports = []
    tdi_fold_scores_by_isoform = {}

    for isoform in C.TDI_ISOFORMS:
        tdi_endpoint = f"{isoform}_is_TDI"
        _cache_path = CACHE_DIR / f"tdi_oof_{isoform}_{CACHE_SUFFIX}.parquet"

        tdi_frame = data.tdi_training_frame(isoform)

        if _cache_path.exists():
            tdi_oof = pl.read_parquet(_cache_path)
        else:
            tdi_oof = models.run_cv_classification(
                tdi_frame, tdi_endpoint, methods=TDI_METHODS, n_outer=n_outer, n_bits=n_bits
            )
            tdi_oof.write_parquet(_cache_path)
        tdi_oof_parts.append(tdi_oof)

        tdi_summary = evaluation.compare_methods_classification(tdi_oof)
        tdi_summaries.append(tdi_summary)
        tdi_fold_scores_by_isoform[isoform] = evaluation.fold_metrics_classification(tdi_oof)

        tdi_reports.append(
            mo.md(
                f"**{tdi_endpoint}** ({tdi_frame.height} labelled compounds, "
                f"{tdi_frame['y_true'].mean():.1%} positive)"
            )
        )

    mo.vstack(tdi_reports)
    return (
        isoform,
        tdi_endpoint,
        tdi_fold_scores_by_isoform,
        tdi_frame,
        tdi_oof,
        tdi_oof_parts,
        tdi_summaries,
        tdi_summary,
    )


@app.cell
def _(mo):
    mo.md("### TDI CV summary (mean ± std MCC across folds, higher is better)")
    return


@app.cell
def _(pl, tdi_summaries):
    all_tdi_summary = pl.concat(tdi_summaries)
    all_tdi_summary.sort(["endpoint", "mcc_mean"], descending=[False, True])
    return (all_tdi_summary,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Which TDI classifiers are *actually* different? (MCS heatmap)

    Same Tukey HSD discipline as the regression track, on MCC instead of ST-RAE.
    Cell = mean MCC difference (row minus column; warm means the row method is
    better, since higher MCC is better); stars = corrected significance.

    **Per-isoform again, not the competition metric** -- the TDI leaderboard also
    macro-averages MCC across both scored isoforms (CYP3A4, CYP2D6). The panel
    after this one is the macro-averaged version.
    """
    )
    return


@app.cell
def _(OUT_DIR, mcs, tdi_fold_scores_by_isoform):
    tdi_mcs_fig = mcs.make_mcs_grid(
        tdi_fold_scores_by_isoform,
        metric_col="mcc",
        higher_is_better={isoform: True for isoform in tdi_fold_scores_by_isoform},
        save_path=OUT_DIR / "mcs_heatmap_tdi.png",
    )
    tdi_mcs_fig
    return (tdi_mcs_fig,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### The metric that actually determines TDI rank: macro-averaged MCC

    Same construction as the regression macro-average, across CYP3A4 and CYP2D6.
    """
    )
    return


@app.cell
def _(OUT_DIR, evaluation, mcs, tdi_fold_scores_by_isoform):
    tdi_macro_fold_scores = evaluation.macro_averaged_fold_metrics(
        tdi_fold_scores_by_isoform, metric_col="mcc"
    )
    tdi_macro_mcs_fig = mcs.make_mcs_grid(
        {"MA (macro-average, both isoforms)": tdi_macro_fold_scores},
        metric_col="mcc",
        higher_is_better={"MA (macro-average, both isoforms)": True},
        save_path=OUT_DIR / "mcs_heatmap_tdi_macro.png",
    )
    tdi_macro_mcs_fig
    return tdi_macro_fold_scores, tdi_macro_mcs_fig


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Build the TDI submission

    Same discipline as the regression track: train the submission method on the
    full labelled set per isoform, predict hard boolean labels for the blind test
    set (MCC is computed on labels as submitted, not probabilities), and validate
    before writing anything final.
    """
    )
    return


@app.cell
def _():
    TDI_SUBMIT_METHOD = "lgbm"
    return (TDI_SUBMIT_METHOD,)


@app.cell
def _(C, TDI_SUBMIT_METHOD, data, models, n_bits, np, test):
    # Not cached to a file, same reasoning as the regression test-prediction cell:
    # one fit per isoform (2 total) is cheap next to the 25-fold CV above it.
    tdi_predictions: dict[str, np.ndarray] = {}
    for _isoform in C.TDI_ISOFORMS:
        _tdi_frame = data.tdi_training_frame(_isoform)
        tdi_predictions[f"{_isoform}_is_TDI"] = models.fit_predict_test_classification(
            _tdi_frame, test["SMILES"].to_list(), method=TDI_SUBMIT_METHOD, n_bits=n_bits
        )
    return (tdi_predictions,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Expected leaderboard performance

    A prediction of what the interim leaderboard should show for this exact
    submission, so the reveal is a check against a number rather than a surprise.
    Computed straight from the CV, for the precise configuration that was actually
    submitted (`SUBMIT_METHOD` + `SUBMIT_CALIBRATION` for regression,
    `TDI_SUBMIT_METHOD` for TDI) -- not from `calib`/`fold_scores_by_endpoint`
    above, which are keyed on whichever method happened to look best per endpoint
    and are not guaranteed to be the submitted one.

    Regression numbers are calibrated the same way the submission is: cross-fit
    (`calibration.crossfit_calibrate`) rather than fit-on-everything, so this
    stays a fair out-of-fold estimate and does not leak the way fitting on all the
    data before scoring it would. Macro-averages use
    `evaluation.macro_averaged_fold_metrics` -- the same construction as the
    leaderboard's own macro average (see CLAUDE.md's Non-negotiables). Expect the
    real leaderboard number to differ somewhat: it bootstraps the actual 750-compound
    test set rather than resampling training-set CV folds.
    """
    )
    return


@app.cell
def _(
    C,
    SUBMIT_CALIBRATION,
    SUBMIT_METHOD,
    TDI_SUBMIT_METHOD,
    all_oof,
    calibration,
    evaluation,
    pl,
    tdi_fold_scores_by_isoform,
):
    # Regression: calibrate the submitted method's OOF predictions cross-fit (same
    # rule as `crossfit_calibrate` everywhere else in this notebook), then take
    # fold-level ST-RAE per endpoint and macro-average across endpoints.
    _submit_oof = all_oof.filter(pl.col("method") == SUBMIT_METHOD)
    _calibrated_fold_scores_by_endpoint = {}
    for _endpoint in C.REGRESSION_ENDPOINTS:
        _endpoint_oof = _submit_oof.filter(pl.col("endpoint") == _endpoint).sort("fold")
        _calibrated = calibration.crossfit_calibrate(_endpoint_oof, SUBMIT_CALIBRATION)
        _calibrated_fold_scores_by_endpoint[_endpoint] = evaluation.fold_metrics(
            _calibrated, pred_col="y_cal"
        )

    expected_regression = pl.concat(
        [
            frame.group_by("endpoint")
            .agg(
                pl.col("st_rae").mean().alias("st_rae_mean"),
                pl.col("st_rae").std().alias("st_rae_std"),
            )
            .with_columns(pl.lit(ep.split("_")[0]).alias("endpoint"))
            for ep, frame in _calibrated_fold_scores_by_endpoint.items()
        ],
        how="diagonal",
    )
    expected_regression_macro = evaluation.macro_averaged_fold_metrics(
        _calibrated_fold_scores_by_endpoint, metric_col="st_rae", group_cols=("fold",)
    )
    expected_regression = pl.concat(
        [
            expected_regression,
            expected_regression_macro.select(
                pl.col("st_rae").mean().alias("st_rae_mean"),
                pl.col("st_rae").std().alias("st_rae_std"),
            ).with_columns(pl.lit("MA (macro-average)").alias("endpoint")),
        ],
        how="diagonal",
    ).select("endpoint", "st_rae_mean", "st_rae_std")

    # TDI: no calibration step in this track, so use the CV fold scores directly.
    _tdi_submit_fold_scores = {
        iso: frame.filter(pl.col("method") == TDI_SUBMIT_METHOD)
        for iso, frame in tdi_fold_scores_by_isoform.items()
    }
    expected_tdi = pl.concat(
        [
            frame.group_by("endpoint")
            .agg(pl.col("mcc").mean().alias("mcc_mean"), pl.col("mcc").std().alias("mcc_std"))
            .with_columns(pl.lit(iso).alias("endpoint"))
            for iso, frame in _tdi_submit_fold_scores.items()
        ],
        how="diagonal",
    )
    expected_tdi_macro = evaluation.macro_averaged_fold_metrics(
        _tdi_submit_fold_scores, metric_col="mcc", group_cols=("fold",)
    )
    expected_tdi = pl.concat(
        [
            expected_tdi,
            expected_tdi_macro.select(
                pl.col("mcc").mean().alias("mcc_mean"), pl.col("mcc").std().alias("mcc_std")
            ).with_columns(pl.lit("MA (macro-average)").alias("endpoint")),
        ],
        how="diagonal",
    ).select("endpoint", "mcc_mean", "mcc_std")

    return expected_regression, expected_tdi


@app.cell
def _(expected_regression):
    expected_regression
    return


@app.cell
def _(expected_tdi):
    expected_tdi
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Write outputs

    Everything lands under `experiments/01_baseline/` (OOF predictions, CV summaries,
    calibration table, bias diagnostic, MCS heatmap PNGs, for both tracks) and the
    submissions under `submissions/01_baseline/<date>/`, with a `PROVENANCE.md`
    recording exactly what produced them — model, calibration, CV protocol, and
    data snapshot. The MCS heatmaps save themselves as they are generated above:
    `mcs_heatmap_regression.png` / `mcs_heatmap_tdi.png` (per-endpoint diagnostics)
    and `mcs_heatmap_regression_macro.png` / `mcs_heatmap_tdi_macro.png` (macro-
    averaged across endpoints — the metric that actually determines rank). This
    cell handles the tabular outputs.
    """
    )
    return


@app.cell
def _(
    OUT_DIR,
    all_oof,
    all_summary,
    all_tdi_summary,
    bias,
    calib,
    expected_regression,
    expected_tdi,
    tdi_oof_parts,
    pl,
):
    all_oof.write_parquet(OUT_DIR / "oof.parquet")
    all_summary.write_csv(OUT_DIR / "cv_summary.csv")
    calib.write_csv(OUT_DIR / "calibration.csv")
    bias.write_csv(OUT_DIR / "bias_by_potency.csv")
    expected_regression.write_csv(OUT_DIR / "expected_performance_regression.csv")
    expected_tdi.write_csv(OUT_DIR / "expected_performance_tdi.csv")

    all_tdi_oof = pl.concat(tdi_oof_parts)
    all_tdi_oof.write_parquet(OUT_DIR / "tdi_oof.parquet")
    all_tdi_summary.write_csv(OUT_DIR / "tdi_cv_summary.csv")
    return (all_tdi_oof,)


@app.cell
def _(expected_regression, expected_tdi):
    def _markdown_table(frame, cols, headers) -> str:
        lines = [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join([":--"] + ["--:"] * (len(cols) - 1)) + "|",
        ]
        for row in frame.select(cols).iter_rows():
            cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    expected_regression_table = _markdown_table(
        expected_regression,
        ["endpoint", "st_rae_mean", "st_rae_std"],
        ["Endpoint", "Expected ST-RAE (mean)", "std across folds"],
    )
    expected_tdi_table = _markdown_table(
        expected_tdi,
        ["endpoint", "mcc_mean", "mcc_std"],
        ["Endpoint", "Expected MCC (mean)", "std across folds"],
    )
    return expected_regression_table, expected_tdi_table


@app.cell
def _(
    C,
    NOTEBOOK_NAME,
    SUBMIT_CALIBRATION,
    SUBMIT_METHOD,
    TDI_SUBMIT_METHOD,
    date,
    expected_regression_table,
    expected_tdi_table,
    n_bits,
    n_outer,
    predictions,
    submission,
    tdi_predictions,
):
    # submissions/<notebook name>/<date>/ -- ties every submission back to the
    # notebook that produced it. Both tracks share one dated subfolder since they
    # come from the same run; a notebook producing submissions from different runs
    # gets one such subfolder per run.
    out_dir = C.SUBMISSIONS_DIR / NOTEBOOK_NAME / f"{date.today():%Y%m%d}"

    submission_path = submission.build_activity_submission(
        predictions, out_dir / "activity.csv"
    )
    submission_ok = submission.check(submission_path, track="activity")

    tdi_submission_path = submission.build_tdi_submission(
        tdi_predictions, out_dir / "tdi.csv"
    )
    tdi_submission_ok = submission.check(tdi_submission_path, track="tdi")

    (out_dir / "PROVENANCE.md").write_text(
        f"""# Baseline submission

- Generated: {date.today():%Y-%m-%d}
- Notebook: `notebooks/{NOTEBOOK_NAME}.py`
- Data snapshot: `{C.latest_snapshot()}`
- CV: nested scaffold, {n_outer}x5 folds

## Direct inhibition (regression)

- Model: {SUBMIT_METHOD} on ECFP4 ({n_bits} bits), per-endpoint
- Calibration: {SUBMIT_CALIBRATION} (fit on all OOF predictions)
- Validation: {"PASSED" if submission_ok else "FAILED"}

**Expected performance (5x5 CV, cross-fit calibrated, lower ST-RAE is better;
1.0 = no better than the mean):**

{expected_regression_table}

Compare the `MA (macro-average)` row against the interim leaderboard's macro ST-RAE for
this submission -- that row, not any single endpoint, is what determines rank.
See `experiments/01_baseline/cv_summary.csv` for the full per-method comparison.

## TDI (classification)

- Model: {TDI_SUBMIT_METHOD} on ECFP4 ({n_bits} bits), per-isoform, class-balanced
- Validation: {"PASSED" if tdi_submission_ok else "FAILED"}

**Expected performance (5x5 CV, higher MCC is better; 0.0 = no better than the majority class):**

{expected_tdi_table}

Compare the `MA (macro-average)` row against the interim leaderboard's macro MCC.
See `experiments/01_baseline/tdi_cv_summary.csv` for the full per-method comparison.

## Caveat

These are CV estimates, not leaderboard guarantees -- the real leaderboard bootstraps
the actual 750-compound test set, while this resamples training-set CV folds (and,
for the macro-average, matches fold *index* across endpoints rather than the same
resampled compounds -- see `evaluation.macro_averaged_fold_metrics`). Expect the real
number to differ somewhat; a large gap is worth investigating (train/test distribution
shift, a scaffold split that was optimistic or pessimistic relative to the real split),
not just shrugged off.
"""
    )
    return out_dir, submission_ok, tdi_submission_ok


@app.cell
def _(mo, out_dir, submission_ok, tdi_submission_ok):
    mo.md(
        f"**Activity submission {'passed' if submission_ok else 'FAILED'} validation. "
        f"TDI submission {'passed' if tdi_submission_ok else 'FAILED'} validation.** "
        f"Written to `{out_dir}`."
    )
    return


if __name__ == "__main__":
    app.run()
