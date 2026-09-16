import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # 06 — Chemprop uncertainty estimation

    `05_auxiliary_data.py` shipped **pubchem**: a multi-target Chemprop D-MPNN whose
    encoder is pretrained on the PubChem qHTS panel then fine-tuned on all four
    challenge endpoints, macro ST-RAE 0.716 on the full 5x5 CV. It is a single point
    estimate per compound. This notebook asks whether Chemprop's own predictive
    uncertainty can be trusted, and if so, whether it can be put to work.

    ## What the submission format actually allows

    `vendor/cyp_challenge_tutorial/validation/activity_validation.py` requires
    exactly one pIC50 value per endpoint — `SMILES`, `Molecule_Name`, four numbers.
    There is no interval column. The `_conf_low`/`_conf_high` columns ST-RAE reads
    are the **ground truth's** assay uncertainty, supplied by the challenge backend,
    not something a submission can widen. So Chemprop uncertainty cannot change what
    gets submitted for a compound directly — it can only change *how a prediction is
    built*, inside the modelling pipeline.

    ## Plan

    1. **Screen** four uncertainty methods Chemprop 2.3.1 supports natively — MVE,
       evidential regression, a checkpoint ensemble, and MC dropout — on reduced CV,
       scored by how well the claimed uncertainty matches actual error (ENCE,
       miscalibration area, and a plain error/uncertainty rank correlation).
    2. **Validate** the best-calibrated method on the full 5x5, since a screen this
       reduced is a triage step, not the number to act on.
    3. **Put it to work**: if the ensemble method wins, its checkpoints are already
       an ensemble — replace the uniform average Chemprop's CLI produces with an
       inverse-variance-weighted average across checkpoints, and check whether that
       beats both the uniform ensemble average and the plain single-model `pubchem`
       baseline on identical folds.
    4. **Compare against the existing lever**: `05` and `01` both found linear
       calibration helps every endpoint. Check whether uncertainty-aware calibration
       (chemprop's own `--calibration-method`) adds anything on top of that, or is
       redundant with it.
    """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os
    import sys
    import time
    from datetime import date
    from pathlib import Path

    import numpy as np
    import polars as pl

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from cyp import (
        aux_training,
        calibration,
        cv,
        data,
        evaluation,
        external,
        submission,
        timings,
    )
    from cyp import constants as C
    from cyp import download as cyp_download

    return (
        C,
        Path,
        PROJECT_ROOT,
        aux_training,
        calibration,
        cv,
        cyp_download,
        data,
        date,
        evaluation,
        external,
        np,
        os,
        pl,
        submission,
        time,
        timings,
    )


@app.cell
def _(mo):
    QUICK = mo.ui.checkbox(value=False, label="Quick mode (5 folds, 10 epochs, small pretrain)")
    QUICK
    return (QUICK,)


@app.cell
def _(QUICK, mo, os):
    quick = QUICK.value
    chemprop_epochs = 10 if quick else 50
    pretrain_epochs = 5 if quick else 30

    # 04/05 measured this directly: chemprop on MPS climbs from ~100s to ~2000s per
    # fold and stays there, while CPU holds flat. Set here since this notebook is
    # almost entirely chemprop, rather than left to the caller's environment.
    os.environ.setdefault("CYP_CHEMPROP_DEVICE", "cpu")

    mo.md(
        f"Running with `chemprop_epochs={chemprop_epochs}`, `pretrain_epochs={pretrain_epochs}`, "
        f"`CYP_CHEMPROP_DEVICE={os.environ['CYP_CHEMPROP_DEVICE']}`."
    )
    return chemprop_epochs, pretrain_epochs, quick


@app.cell
def _(Path, PROJECT_ROOT):
    NOTEBOOK_NAME = Path(__file__).stem
    OUT_DIR = PROJECT_ROOT / "experiments" / NOTEBOOK_NAME
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return NOTEBOOK_NAME, OUT_DIR


@app.cell
def _(OUT_DIR, quick):
    CACHE_DIR = OUT_DIR / "cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_SUFFIX = "quick" if quick else "full"

    # Written per measurement, not at the end -- on every run after the first the
    # training branch is skipped, so a timer inside it would measure nothing.
    TIMING_LOG = OUT_DIR / "timings.csv"
    return CACHE_DIR, CACHE_SUFFIX, TIMING_LOG


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Data

    Same challenge snapshot and PubChem pretraining corpus `05` used, so the
    pretrained encoder this notebook fine-tunes from is comparable to the one that
    was actually shipped.
    """
    )
    return


@app.cell
def _(C, cyp_download, external, mo):
    if not C.available_snapshots():
        mo.output.append(mo.md("No challenge snapshot found — downloading now..."))
        cyp_download.download()
    CHALLENGE_SNAPSHOT = C.latest_snapshot()

    if not C.available_external_snapshots():
        mo.output.append(mo.md("No external snapshot found — downloading PubChem..."))
    EXTERNAL_SNAPSHOT = external.ensure_downloaded()

    mo.output.append(
        mo.md(
            f"Challenge snapshot `{CHALLENGE_SNAPSHOT}`, external snapshot `{EXTERNAL_SNAPSHOT}`."
        )
    )
    return CHALLENGE_SNAPSHOT, EXTERNAL_SNAPSHOT


@app.cell
def _(C, data, mo, pl):
    ENDPOINTS = list(C.REGRESSION_ENDPOINTS)
    frames = {endpoint: data.training_frame(endpoint) for endpoint in ENDPOINTS}

    mo.output.append(
        pl.DataFrame({"endpoint": ENDPOINTS, "n": [frames[e].height for e in ENDPOINTS]})
    )
    return ENDPOINTS, frames


@app.cell
def _(ENDPOINTS, EXTERNAL_SNAPSHOT, aux_training, data, frames, mo, pl, quick):
    _challenge = data.load_train_inhibition()["SMILES"].to_list()
    pretraining = aux_training.public_pretraining_matrix(
        ENDPOINTS, exclude_smiles=_challenge, snapshot=EXTERNAL_SNAPSHOT
    )
    if quick:
        pretraining = pretraining.head(2000)

    mo.output.append(
        mo.md(
            f"**{pretraining.height:,} public compounds** for pretraining, against "
            f"{sum(frames[e].height for e in ENDPOINTS):,} challenge rows across "
            f"{len(ENDPOINTS)} endpoints."
        )
    )
    return (pretraining,)


@app.cell
def _(frames, cv, mo):
    _, assignments = cv.shared_scaffold_folds(frames, n_outer=5, n_inner=5, seed=42)
    ALL_FOLDS = sorted(assignments["fold"].unique().to_list())
    SCREEN_FOLDS = ALL_FOLDS[:5]  # one outer repeat -- the cheap screen

    mo.md(
        f"`{len(ALL_FOLDS)}` total folds assigned; screening on the first "
        f"`{len(SCREEN_FOLDS)}` (`{SCREEN_FOLDS}`), validating the winner on all "
        f"`{len(ALL_FOLDS)}`."
    )
    return ALL_FOLDS, SCREEN_FOLDS, assignments


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Screen — four uncertainty methods, reduced CV

    Each method reuses the same PubChem-pretrained checkpoint `pubchem` warm-starts
    from (fine-tuning changes per method, since MVE and evidential change the FFN
    head and therefore need their own fine-tuning fold-by-fold same as any other
    arm — only the *pretrained encoder* is shared). Ensemble trains 4 checkpoints per
    fold, so it is roughly 4x the cost of the other three; screening on 5 folds
    rather than 25 keeps that affordable.

    Scored with three diagnostics, none of which is ST-RAE — this section asks
    "is the claimed uncertainty honest", not "does this method predict pIC50 well":

    - **ENCE** — expected normalized calibration error: do compounds Chemprop
      claims high variance for actually have higher squared error, on the right
      scale? 0 is perfect.
    - **Miscalibration area** — area between the observed and nominal coverage
      curves of the implied Gaussian interval. 0 is perfect.
    - **Spearman(|error|, variance)** — the weaker, more useful property for
      ensembling: even a badly-scaled uncertainty is useful for ranking predictions
      by trustworthiness as long as it is rank-correlated with actual error.
    """
    )
    return


@app.cell
def _(
    CACHE_DIR,
    CACHE_SUFFIX,
    SCREEN_FOLDS,
    TIMING_LOG,
    assignments,
    aux_training,
    chemprop_epochs,
    frames,
    mo,
    pl,
    pretrain_epochs,
    pretraining,
    time,
    timings,
):
    UNCERTAINTY_METHODS = ["mve", "evidential", "ensemble", "dropout"]
    ENSEMBLE_SIZE = 4

    def _run_unit(method: str, fold: int) -> pl.DataFrame:
        kwargs = dict(uncertainty_method=method)
        if method == "ensemble":
            kwargs["ensemble_size"] = ENSEMBLE_SIZE
        return aux_training.run_cv_pretrained(
            frames,
            pretraining,
            method_name=method,
            pretrain_epochs=pretrain_epochs,
            assignments=assignments,
            folds=[fold],
            epochs=chemprop_epochs,
            uncertainty=True,
            **kwargs,
        )

    _parts = []
    with mo.status.progress_bar(
        total=len(UNCERTAINTY_METHODS) * len(SCREEN_FOLDS), title="Uncertainty method screen"
    ) as _bar:
        for _fold in SCREEN_FOLDS:
            for _method in UNCERTAINTY_METHODS:
                _cache = CACHE_DIR / f"screen_{_method}_fold{_fold}_{CACHE_SUFFIX}.parquet"
                if _cache.exists():
                    _oof = pl.read_parquet(_cache)
                else:
                    _t0 = time.time()
                    _oof = _run_unit(_method, _fold)
                    _oof.write_parquet(_cache)
                    timings.record(
                        TIMING_LOG,
                        stage="screen",
                        method=_method,
                        endpoint=f"fold{_fold}",
                        mode=CACHE_SUFFIX,
                        seconds=time.time() - _t0,
                    )
                _parts.append(_oof)
                _bar.update()

    screen_oof = pl.concat(_parts, how="vertical_relaxed")
    screen_oof.group_by("method").len().sort("method")
    return ENSEMBLE_SIZE, UNCERTAINTY_METHODS, screen_oof


@app.cell
def _(evaluation, screen_oof):
    screen_calibration = evaluation.uncertainty_calibration_report(
        screen_oof, group_cols=("method",)
    ).sort("ence")
    screen_calibration
    return (screen_calibration,)


@app.cell
def _(mo, screen_calibration):
    _winner = screen_calibration.row(0, named=True)
    mo.md(
        f"""
    **`{_winner["method"]}` has the lowest ENCE on the screen**
    ({_winner["ence"]:.3f}, miscalibration area {_winner["miscalibration_area"]:.3f},
    error/uncertainty Spearman {_winner["unc_error_spearman"]:.3f}). Validated on the
    full 5x5 below before anything is built on it — 5 folds is a triage step, not a
    conclusion, and CLAUDE.md's own lesson from PXR is not to rank on a CV split this
    small.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Validation — the winner on the full 5x5

    If the screen winner is genuinely well calibrated, that should hold on the full
    fold set too, not just the 5 it was picked on.
    """
    )
    return


@app.cell
def _(
    ALL_FOLDS,
    CACHE_DIR,
    CACHE_SUFFIX,
    ENSEMBLE_SIZE,
    TIMING_LOG,
    assignments,
    aux_training,
    chemprop_epochs,
    frames,
    mo,
    pl,
    pretrain_epochs,
    pretraining,
    screen_calibration,
    time,
    timings,
):
    WINNER_METHOD = screen_calibration.row(0, named=True)["method"]

    def _run_winner(fold: int) -> pl.DataFrame:
        kwargs = dict(uncertainty_method=WINNER_METHOD)
        if WINNER_METHOD == "ensemble":
            kwargs["ensemble_size"] = ENSEMBLE_SIZE
        return aux_training.run_cv_pretrained(
            frames,
            pretraining,
            method_name=WINNER_METHOD,
            pretrain_epochs=pretrain_epochs,
            assignments=assignments,
            folds=[fold],
            epochs=chemprop_epochs,
            uncertainty=True,
            **kwargs,
        )

    _parts = []
    with mo.status.progress_bar(total=len(ALL_FOLDS), title=f"Validating {WINNER_METHOD}") as _bar:
        for _fold in ALL_FOLDS:
            _cache = CACHE_DIR / f"winner_{WINNER_METHOD}_fold{_fold}_{CACHE_SUFFIX}.parquet"
            if _cache.exists():
                _oof = pl.read_parquet(_cache)
            else:
                _t0 = time.time()
                _oof = _run_winner(_fold)
                _oof.write_parquet(_cache)
                timings.record(
                    TIMING_LOG,
                    stage="winner_validation",
                    method=WINNER_METHOD,
                    endpoint=f"fold{_fold}",
                    mode=CACHE_SUFFIX,
                    seconds=time.time() - _t0,
                )
            _parts.append(_oof)
            _bar.update()

    winner_oof = pl.concat(_parts, how="vertical_relaxed")
    return WINNER_METHOD, winner_oof


@app.cell
def _(OUT_DIR, winner_oof):
    winner_oof.write_parquet(OUT_DIR / "oof.parquet")
    return


@app.cell
def _(evaluation, winner_oof):
    winner_calibration_by_endpoint = evaluation.uncertainty_calibration_report(winner_oof)
    winner_calibration_overall = evaluation.uncertainty_calibration_report(
        winner_oof, group_cols=("method",)
    )
    winner_calibration_overall
    return winner_calibration_by_endpoint, winner_calibration_overall


@app.cell
def _(winner_calibration_by_endpoint):
    winner_calibration_by_endpoint
    return


@app.cell
def _(WINNER_METHOD, evaluation, mo, winner_oof):
    winner_st_rae = evaluation.compare_methods(winner_oof)
    _row = winner_st_rae.row(0, named=True)
    mo.md(
        f"""
    For reference, `{WINNER_METHOD}` scores **{_row["st_rae_mean"]:.4f} ± {_row["st_rae_std"]:.4f}**
    macro-pooled ST-RAE on the full 5x5 (pooled across endpoints in this table; see
    `evaluation.macro_averaged_fold_metrics` for the true per-fold macro-average used
    below). `pubchem` itself scored 0.716 in `05`, so this is the cost of asking the
    same architecture for calibrated uncertainty rather than only a point estimate.
    """
    )
    return (winner_st_rae,)


@app.cell
def _(WINNER_METHOD, mo):
    mo.md(
        f"""
    ## Putting it to work — inverse-variance ensembling

    `{WINNER_METHOD}` won the calibration screen. This section only applies when
    that method is `ensemble`: its checkpoints already form a genuine multi-model
    ensemble, so their disagreement can drive a weighted average instead of the
    uniform one Chemprop's CLI produces by default. MVE, evidential and dropout each
    describe *one* model's own uncertainty and have no second model to weight
    against here — a real use for their calibration exists (the comparison against
    linear calibration below), just not this one.
    """
    )
    return


@app.cell
def _(
    CACHE_DIR,
    CACHE_SUFFIX,
    ENSEMBLE_SIZE,
    TIMING_LOG,
    WINNER_METHOD,
    assignments,
    aux_training,
    chemprop_epochs,
    frames,
    mo,
    pl,
    pretrain_epochs,
    pretraining,
    time,
    timings,
):
    if WINNER_METHOD != "ensemble":
        ensemble_weighting_oof = None
        mo.output.append(
            mo.md(
                f"Skipped: the screen winner was `{WINNER_METHOD}`, not `ensemble` — "
                "nothing to weight."
            )
        )
    else:
        _all_folds = sorted(assignments["fold"].unique().to_list())

        _parts = []
        with mo.status.progress_bar(
            total=len(_all_folds), title="Inverse-variance ensemble weighting"
        ) as _bar:
            for _fold in _all_folds:
                _cache = CACHE_DIR / f"ivw_fold{_fold}_{CACHE_SUFFIX}.parquet"
                if _cache.exists():
                    _oof = pl.read_parquet(_cache)
                else:
                    _t0 = time.time()
                    _oof = aux_training.run_cv_ensemble_weighting(
                        frames,
                        pretraining,
                        ensemble_size=ENSEMBLE_SIZE,
                        pretrain_epochs=pretrain_epochs,
                        assignments=assignments,
                        folds=[_fold],
                        epochs=chemprop_epochs,
                    )
                    _oof.write_parquet(_cache)
                    timings.record(
                        TIMING_LOG,
                        stage="ensemble_weighting",
                        method="ivw_vs_uniform",
                        endpoint=f"fold{_fold}",
                        mode=CACHE_SUFFIX,
                        seconds=time.time() - _t0,
                    )
                _parts.append(_oof)
                _bar.update()

        ensemble_weighting_oof = pl.concat(_parts, how="vertical_relaxed")
    return (ensemble_weighting_oof,)


@app.cell
def _(WINNER_METHOD, ensemble_weighting_oof, evaluation, mo):
    if WINNER_METHOD == "ensemble":
        ensemble_weighting_compare = evaluation.compare_methods(ensemble_weighting_oof)
        mo.output.append(ensemble_weighting_compare)
    else:
        ensemble_weighting_compare = None
        mo.output.append(mo.md("N/A — see previous cell."))
    return (ensemble_weighting_compare,)


@app.cell
def _(WINNER_METHOD, ensemble_weighting_oof, evaluation, mo, pl):
    if WINNER_METHOD == "ensemble":
        _wide = ensemble_weighting_oof.pivot(
            values="y_pred",
            index=["Molecule_Name", "endpoint", "fold", "y_true", "y_lower", "y_upper"],
            on="method",
        )
        _bootstrap = evaluation.paired_bootstrap(
            _wide["y_true"].to_numpy(),
            _wide["ensemble_ivw"].to_numpy(),
            _wide["ensemble_uniform"].to_numpy(),
            metric="st_rae",
            y_lower=_wide["y_lower"].to_numpy(),
            y_upper=_wide["y_upper"].to_numpy(),
        )
        ivw_vs_uniform = pl.DataFrame([{"comparison": "ivw_vs_uniform", **_bootstrap}])
        mo.output.append(ivw_vs_uniform)
        mo.output.append(
            mo.md(
                f"""
    IVW vs. uniform ensemble averaging, paired bootstrap on identical checkpoints:
    diff (IVW − uniform) = **{_bootstrap["diff"]:.4f}**,
    95% CI [{_bootstrap["ci_low"]:.4f}, {_bootstrap["ci_high"]:.4f}], p={_bootstrap["p_value"]:.4f}.
    Negative and significant means inverse-variance weighting helps; a CI straddling
    zero means this CV cannot resolve a difference, in which case ship the simpler
    uniform average.
    """
            )
        )
    else:
        ivw_vs_uniform = None
        mo.output.append(mo.md("N/A — see previous cells."))
    return (ivw_vs_uniform,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Against the existing lever — linear calibration

    `01` and `05` both found plain linear calibration (`calibration.py`, cross-fit)
    helps every endpoint, cheaply. Chemprop's uncertainty gives a second, different
    kind of calibration: not "is the point prediction biased" but "is the claimed
    spread around it honest". They are not mutually exclusive, so the question is
    whether uncertainty-aware calibration finds something linear calibration missed,
    or is redundant with it once the point prediction is already corrected.

    Read as: apply `calibration.py`'s cross-fit linear calibrator to the winner's
    point predictions, then re-run the calibration diagnostics on the *residual*
    against the (now recentred) predictions. If ENCE/miscalibration area barely
    move, linear calibration already captured what mattered and the uncertainty
    estimate was mostly tracking the same bias. If they improve further, the
    uncertainty carries information calibration does not.
    """
    )
    return


@app.cell
def _(calibration, evaluation, pl, winner_oof):
    _parts = []
    for _endpoint, _group in winner_oof.group_by("endpoint"):
        _calibrated = calibration.crossfit_calibrate(_group.sort("fold"), "linear")
        _parts.append(_calibrated)
    winner_oof_calibrated = pl.concat(_parts)

    calibration_comparison = evaluation.uncertainty_calibration_report(
        winner_oof_calibrated, pred_col="y_cal", group_cols=("method",)
    )
    calibration_comparison
    return (winner_oof_calibrated,)


@app.cell
def _(calibration_comparison, mo, winner_calibration_overall):
    _raw = winner_calibration_overall.row(0, named=True)
    _cal = calibration_comparison.row(0, named=True)
    mo.md(
        f"""
    Raw-prediction ENCE **{_raw["ence"]:.3f}** vs. after-linear-calibration ENCE
    **{_cal["ence"]:.3f}** (miscalibration area {_raw["miscalibration_area"]:.3f} vs.
    {_cal["miscalibration_area"]:.3f}). Linear calibration recentres the *mean*; if
    `y_unc` was already tracking the true error around the raw mean rather than
    around a systematic bias, these numbers should barely move. A large drop instead
    would mean part of what `y_unc` was capturing was just the same bias linear
    calibration removes for free — worth knowing before treating chemprop's
    uncertainty as adding information beyond what calibration already gives.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Conclusion

    This notebook does not automatically produce a new submission. `05`'s `pubchem`
    remains the shipped model unless the sections above found a specific,
    CV-resolvable improvement:

    - If the IVW-vs-uniform paired bootstrap above showed a significant negative
      diff, that reweighting is worth carrying into a production ensemble fit — but
      note it changes the *architecture* (an ensemble of `ENSEMBLE_SIZE`
      checkpoints, `ENSEMBLE_SIZE`x the inference cost) relative to the single-model
      `pubchem` that was actually scored 0.716 in `05`. The comparison that would
      justify a resubmission — this notebook's ensemble arm vs. plain `pubchem`, both
      on identical folds — is not yet run above; add it before shipping anything,
      following the same `evaluation.paired_bootstrap` pattern used for
      IVW-vs-uniform.
    - If no comparison here beat `pubchem` at a resolvable margin, the honest
      conclusion is that Chemprop's native uncertainty is diagnostic (it tells you
      which compounds to trust less) but did not translate into a better point
      prediction on this CV — worth recording in CLAUDE.md as a measured result, not
      silently dropped.

    Given the submission format's constraint (Molecule_Name, SMILES, one pIC50 per
    endpoint — see the intro), the calibration diagnostics in this notebook are a
    deliverable in their own right for the Innovation-award angle CLAUDE.md flags,
    independent of whether they move a resubmission.
    """
    )
    return


if __name__ == "__main__":
    app.run()
