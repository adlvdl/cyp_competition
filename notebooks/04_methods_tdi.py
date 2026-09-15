import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # 04 — Single-task vs multitask on the TDI track

    `03_methods.py` ran the method comparison for the four regression endpoints and
    found that multitask won outright: one Chemprop D-MPNN with four masked output
    heads took the macro ST-RAE from 0.7599 (best alternative) to 0.7280, separable
    from every rival at p=0.0000. It then deliberately **declined** to ship a TDI
    submission on the strength of that result, because the mechanism behind the win
    does not obviously transfer.

    This notebook runs the measurement that decision was deferred to: the same
    single-task/multitask comparison, every method from 03 that has a classification
    formulation, on the two scored TDI isoforms.

    ## Why the multitask premise is weaker here

    The regression win rests on compounds carrying several endpoints, which gives a
    shared encoder real cross-endpoint signal. Measured on the current snapshot:

    | | regression | TDI |
    |:--|--:|--:|
    | compounds in the union | 4,905 | 4,822 |
    | carrying more than one endpoint | 1,309 (**26.7%**) | 259 (**5.4%**) |
    | endpoints | 4 | 2 |

    So the specific lever that worked in 03 is roughly five times scarcer here, and
    there are half as many tasks to share across. The honest prior is that multitask
    does *less* on TDI — possibly nothing, possibly harm. PXR found multitask hurting
    where it was expected to help, which is exactly why this repo runs paired arms
    instead of assuming the direction.

    That said, the mechanism is not only about co-measured compounds. The shared
    encoder still sees the whole union, so CYP2D6 (1,497 rows) could borrow
    *representation* from CYP3A4's 3,584 even where almost no compound carries both
    labels. Which of those two effects dominates is the question, and it is an
    empirical one.

    ## What is and is not run here

    **Only the single-task vs multitask comparison**, across every method 03 used.
    The fingerprint sweep, the interval diagnostics and the calibration study are not
    repeated: this notebook exists to answer one question, and 03 already answered
    the representation question for the regression track.

    `macau` is dropped. It is the one method from 03 with no classification
    formulation at all — it factorizes a real-valued matrix, and a boolean TDI label
    is not what it models. `models.CLASSIFIER_FACTORIES` omits it for the same reason.

    | method | 03 strategy | 04 strategy | note |
    |:--|:--|:--|:--|
    | `lgbm` | stacked | stacked | class-balanced weighting |
    | `xgb` | stacked | stacked | |
    | `tabicl` | stacked | stacked | subprocess, Apache-2.0 |
    | `tabpfn` | stacked | stacked | subprocess, needs a token |
    | `chemprop` | multitarget | multitarget | one D-MPNN, one head per isoform |
    | `macau` | native | — | **dropped**: no classification formulation |

    ## MCC, not accuracy — and not at a 0.5 threshold either

    TDI labels are ~21% positive on both isoforms, so a classifier that always
    predicts "not TDI" gets ~79% accuracy and MCC exactly 0.0. `models.MajorityBaseline`
    is carried through the comparison as that line.

    The subtler trap is the decision threshold, and on this data it is not a footnote
    — it dominates. MCC on a 21%-positive label is not optimized at 0.5, and 03's
    regression harness had no analogue of this problem. Every OOF row here records
    `y_prob` alongside the hard label so the cutoff can be tuned *after* the fits
    rather than by refitting 25 folds.

    **Measured on the quick run (1 repeat, 10 epochs), every method's optimum sat
    between 0.10 and 0.25 — none at 0.5** — and the ordering at 0.5 was not the
    ordering at the optimum. Two concrete consequences:

    - **Chemprop scores exactly MCC 0.0 at a 0.5 cutoff, in both arms.** Its
      probabilities are well calibrated to the base rate (mean ≈ 0.205 against a
      21.4% positive rate) but never cross 0.5, so thresholding there predicts zero
      positives and lands it on the majority-baseline line by construction. Tuned, it
      recovers to a real score. The tree models clear 0.5 only because
      `class_weight="balanced"` inflates their probabilities — an artefact of their
      configuration, not evidence they learned more.
    - **The ranking changes.** TabICL went from mid-table at 0.5 to first once tuned.

    So this notebook reports **both** cutoffs side by side and treats the tuned one as
    the honest comparison. A table at 0.5 alone would mostly be measuring which
    models happen to emit inflated probabilities.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Interleaving: why this notebook runs fold-major

    03 ran method-major — every fold of `chemprop` back to back, then every fold of
    the next method. That is the ordering that produced the failure documented in
    CLAUDE.md: Chemprop degraded across a long block of consecutive fits on MPS,
    from ~30s/fold early to ~290s/fold late, and the `chemeleon` arm spent 11.4 hours
    on a single endpoint without finishing.

    This notebook inverts the loop. The outer loop is over folds, the inner loop over
    (method, arm), so consecutive Chemprop fits are separated in wall-clock time by
    every other model's fit on that fold. Each Chemprop fit is already its own
    subprocess with its own MPS allocator (`graph_models._run_chemprop_cli` sets a
    real watermark ratio so the process gives memory back), and spacing them further
    gives the system room to actually reclaim between them.

    **The bigger win is the failure mode, not the speed.** Caching is keyed per
    `(method, arm, fold)` rather than per `(method, arm)`. Under method-major caching
    a run killed at hour six leaves complete results for the first two methods and
    nothing at all for the rest — so the comparison cannot be made, and the only
    recourse is to start again. Under fold-major caching with per-fold units, the
    same interruption leaves *every* method complete through fold *k*. That is a
    usable, if smaller, paired comparison — every arm still scored on identical rows,
    just fewer of them — and a resumed run picks up exactly where it stopped.

    The ordering cannot change the results, only when they arrive: folds are
    independent, `assignments` is fixed up front, and each unit is a complete fit on
    a predetermined split. `test_fold_slices_reassemble_into_the_whole_run` in
    `tests/test_multitask_tdi.py` pins that — running fold-by-fold and concatenating
    reproduces the all-folds run prediction for prediction.
    """
    )
    return


@app.cell
def _():
    # `date` and `cyp.submission` are imported but unused: this notebook writes no
    # submission yet, deliberately. They are what the submission cell needs once
    # there is a measured result to stand on -- see the Findings section.
    import os
    import sys
    import time
    from datetime import date
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import polars as pl

    # This notebook lives in notebooks/; the package lives in src/.
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from cyp import constants as C
    from cyp import (
        cv,
        data,
        evaluation,
        mcs,
        models,
        multitask,
        submission,
        timings,
    )
    from cyp import download as cyp_download

    return (
        C,
        Path,
        PROJECT_ROOT,
        cv,
        cyp_download,
        data,
        date,
        evaluation,
        mcs,
        mo,
        models,
        multitask,
        np,
        os,
        pl,
        submission,
        time,
        timings,
    )


@app.cell
def _(mo):
    QUICK = mo.ui.checkbox(
        value=False, label="Quick mode (1 outer repeat, 1024 bits, 10 epochs)"
    )
    QUICK
    return (QUICK,)


@app.cell
def _(QUICK, mo):
    quick = QUICK.value
    n_outer = 1 if quick else 5
    n_bits = 1024 if quick else 2048
    # Chemprop epochs are the single biggest lever on runtime here, and quick mode
    # exists to prove the pipeline end to end rather than to produce a number.
    chemprop_epochs = 10 if quick else 50
    mo.md(
        f"Running with `n_outer={n_outer}`, `n_bits={n_bits}`, "
        f"`chemprop_epochs={chemprop_epochs}`."
    )
    return chemprop_epochs, n_bits, n_outer, quick


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

    # Timings live in the experiment directory, not the cache: a small derived
    # summary worth committing, and it must survive a cache wipe. Written per
    # measurement so a run that dies mid-sweep still records what finished.
    TIMING_LOG = OUT_DIR / "timings.csv"
    return CACHE_DIR, CACHE_SUFFIX, TIMING_LOG


@app.cell
def _(C, cyp_download, mo):
    if not C.available_snapshots():
        mo.output.replace(mo.md("No data snapshot found — downloading now..."))
        cyp_download.download()
    else:
        mo.output.replace(mo.md(f"Using existing snapshot `{C.latest_snapshot()}`."))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## The data, and the overlap the whole comparison turns on

    Both isoforms, plus the co-measured fraction that decides whether a shared
    encoder has anything to share. This is recomputed on every run rather than quoted
    from CLAUDE.md, because the dataset has been amended mid-challenge before and the
    5.4% figure is load-bearing for how this notebook's result should be read.
    """
    )
    return


@app.cell
def _(C, data, mo, pl):
    tdi_frames = {iso: data.tdi_training_frame(iso) for iso in C.TDI_ISOFORMS}

    _names = {iso: set(f["Molecule_Name"].to_list()) for iso, f in tdi_frames.items()}
    _union = set().union(*_names.values())
    _both = set.intersection(*_names.values())

    tdi_overlap = {
        "union": len(_union),
        "both": len(_both),
        "pct": 100 * len(_both) / len(_union),
    }

    _counts = pl.DataFrame(
        {
            "isoform": list(tdi_frames),
            "n_compounds": [f.height for f in tdi_frames.values()],
            "positive_rate": [
                round(float(f["y_true"].mean()), 4) for f in tdi_frames.values()
            ],
        }
    )

    mo.vstack([
        mo.ui.table(_counts, page_size=5),
        mo.md(
            f"**{tdi_overlap['both']:,} of {tdi_overlap['union']:,} compounds "
            f"({tdi_overlap['pct']:.1f}%) carry both isoforms** — against 26.7% for "
            "the four regression endpoints. The multitask lever that won in 03 is "
            "roughly five times scarcer here; read every result below against that."
        ),
    ])
    return tdi_frames, tdi_overlap


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Shared folds, and the leakage check that justifies them

    Both arms run on one fold assignment over the **union** of compounds
    (`cv.shared_scaffold_folds`). This is correctness, not tidiness: with per-isoform
    splits a compound could sit in CYP3A4's training set and CYP2D6's test set at
    once, so a multitask model would be scored on a compound whose label it had
    partly seen.

    It is also what makes the two arms *paired* — identical test rows — so
    `evaluation.paired_bootstrap` between them is a real paired test rather than two
    independent runs eyeballed against each other.

    The check below runs every time rather than being trusted. It is cheap, and it is
    the assumption the entire notebook rests on.
    """
    )
    return


@app.cell
def _(cv, mo, n_outer, tdi_frames):
    _fold_of, FOLD_ASSIGNMENTS = cv.shared_scaffold_folds(tdi_frames, n_outer=n_outer)
    ALL_FOLDS = sorted(FOLD_ASSIGNMENTS["fold"].unique().to_list())

    # A compound held out for one isoform must not be in training for the other.
    _leaks = 0
    for _fold in ALL_FOLDS:
        _test_names, _train_names = set(), set()
        for _iso, _frame in tdi_frames.items():
            for _fd, _o, _i, _tr, _v, _te in cv.fold_assignment_splits(
                _frame, FOLD_ASSIGNMENTS
            ):
                if _fd != _fold:
                    continue
                _test_names |= set(_te["Molecule_Name"].to_list())
                _train_names |= set(_tr["Molecule_Name"].to_list())
        _leaks += len(_test_names & _train_names)

    mo.md(
        f"Assigned **{FOLD_ASSIGNMENTS['Molecule_Name'].n_unique():,} compounds** to "
        f"{len(ALL_FOLDS)} folds across "
        f"{FOLD_ASSIGNMENTS['outer_fold'].n_unique()} outer repeats.\n\n"
        + (
            f"✅ **No cross-isoform leakage** ({_leaks} compounds in both train and "
            "test across isoforms)."
            if _leaks == 0
            else f"🚨 **{_leaks} compounds leak across isoforms** — the comparison "
            "is invalid until this is fixed."
        )
    )
    return ALL_FOLDS, FOLD_ASSIGNMENTS


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Methods

    Every method from 03 that has a classification formulation, plus the two
    baselines. `majority` is the line a real classifier must clear — MCC = 0.0 by
    construction — and it is single-task only: predicting a constant cannot borrow
    strength across isoforms, so a "multitask" version of it would be the same model
    with extra steps.

    TabPFN is included only when a usable token is present. It is detected up front
    rather than partway through a multi-hour run; TabICL (Apache-2.0, no gate) carries
    the TFM comparison on its own otherwise.
    """
    )
    return


@app.cell
def _(mo, models, os):
    from cyp.tabular_models import load_env

    load_env()
    HAS_TABPFN_TOKEN = bool(os.environ.get("TABPFN_TOKEN"))

    # Methods with both arms. `macau` is absent -- no classification formulation.
    # `chemeleon` is absent for the runtime reason in 03: an identical architecture
    # to `chemprop` that spent 11.4 hours on one endpoint without finishing.
    PAIRED_METHODS = ["lgbm", "xgb", "tabicl", "chemprop"]
    if HAS_TABPFN_TOKEN:
        PAIRED_METHODS.insert(3, "tabpfn")

    # Single-task only: a constant predictor cannot share anything across isoforms.
    BASELINE_METHODS = ["majority"]

    _msg = (
        "TabPFN token found — including `tabpfn`."
        if HAS_TABPFN_TOKEN
        else (
            "**No `TABPFN_TOKEN` found — skipping `tabpfn`.** Put the API key "
            "(`tabpfn_sk_...`, from the account page at https://ux.priorlabs.ai) in "
            "`.env` at the project root. TabICL still runs."
        )
    )
    mo.md(
        f"{_msg}\n\nPaired arms: `{'`, `'.join(PAIRED_METHODS)}`. "
        f"Baseline (single-task only): `{'`, `'.join(BASELINE_METHODS)}`.\n\n"
        f"Registered classifiers: `{'`, `'.join(sorted(models.CLASSIFIER_FACTORIES))}`."
    )
    return BASELINE_METHODS, PAIRED_METHODS


@app.cell
def _(mo):
    mo.md(
        r"""
    ### CheMeleon embeddings

    Computed once per isoform and reused by the TFMs, exactly as in 03. The embedding
    is frozen and never sees a label, so embedding the whole labelled set in one call
    leaks nothing — there is no fit to leak through.

    Cached as plain `.npy`: large, purely derived from the data snapshot, no model fit
    involved, which is the case `.gitignore`'s `experiments/**/cache/` rule covers.
    """
    )
    return


@app.cell
def _(CACHE_DIR, mo, np, pl, tdi_frames):
    from cyp.graph_models import chemeleon_embed

    chemeleon_features = {}
    with mo.status.progress_bar(
        total=len(tdi_frames), title="CheMeleon embeddings"
    ) as _bar:
        for _iso, _frame in tdi_frames.items():
            _cache = CACHE_DIR / f"chemeleon_tdi_{_iso}.npy"
            if _cache.exists():
                chemeleon_features[_iso] = np.load(_cache)
            else:
                # The second argument takes a single throwaway SMILES because the
                # function embeds two lists; only the first is needed here.
                _emb, _ = chemeleon_embed(
                    _frame["SMILES"].to_list(), _frame["SMILES"].to_list()[:1]
                )
                np.save(_cache, _emb)
                chemeleon_features[_iso] = _emb
            _bar.update()

    pl.DataFrame(
        {
            "isoform": list(chemeleon_features),
            "n_compounds": [v.shape[0] for v in chemeleon_features.values()],
            "n_features": [v.shape[1] for v in chemeleon_features.values()],
        }
    )
    return (chemeleon_features,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## The run — fold-major, both arms interleaved

    One cache file per `(method, arm, fold)`. The loop order is fold → method → arm,
    so an interrupted run leaves every method complete through the last finished
    fold rather than a few methods complete and the rest missing entirely.

    Progress counts **work units**, where a unit is one (method, arm, fold) — the
    multitask arm is one fit per fold, the single-task arm is one fit per isoform per
    fold. A cached unit advances the bar by what it skipped, so a fully cached rerun
    still reaches 100%.

    Timings are recorded per unit as the work happens. On a rerun the training branch
    is skipped entirely, so a timer living inside it would measure nothing and the
    committed run — the one anyone reads later — would report an empty table.
    """
    )
    return


@app.cell
def _(
    ALL_FOLDS,
    BASELINE_METHODS,
    CACHE_DIR,
    CACHE_SUFFIX,
    FOLD_ASSIGNMENTS,
    PAIRED_METHODS,
    TIMING_LOG,
    chemeleon_features,
    chemprop_epochs,
    mo,
    multitask,
    n_bits,
    pl,
    tdi_frames,
    time,
    timings,
):
    def _run_unit(method, arm, fold):
        """One (method, arm, fold) fit. The atom of the interleaved loop."""
        # TFMs consume the frozen CheMeleon embedding, as in 03; the graph model
        # featurizes from SMILES itself and ignores whatever is passed.
        on_embedding = method in ("tabicl", "tabpfn")
        features = chemeleon_features if on_embedding else None
        # Both arms get the same epoch budget, or the comparison measures epochs
        # rather than multitask. The TFMs take their settings as explicit arguments
        # to the subprocess runner, so they get no model kwargs at all.
        kwargs = {"epochs": chemprop_epochs} if method == "chemprop" else {}

        if arm == "multitask":
            return multitask.run_cv_multitask_classification(
                tdi_frames,
                method,
                features=features,
                n_bits=n_bits,
                assignments=FOLD_ASSIGNMENTS,
                folds=[fold],
                **kwargs,
            )
        return multitask.run_cv_singletask_classification(
            tdi_frames,
            method,
            features=features,
            n_bits=n_bits,
            assignments=FOLD_ASSIGNMENTS,
            folds=[fold],
            **kwargs,
        )

    # Work units: every paired method in both arms, plus the baseline single-task.
    _units = [(m, a) for m in PAIRED_METHODS for a in ("singletask", "multitask")]
    _units += [(m, "singletask") for m in BASELINE_METHODS]

    _parts = []
    with mo.status.progress_bar(
        total=len(_units) * len(ALL_FOLDS), title="TDI: single-task vs multitask"
    ) as _bar:
        # Fold-major: the outer loop is the fold, so consecutive Chemprop fits are
        # separated by every other model's fit on the same fold.
        for _fold in ALL_FOLDS:
            for _method, _arm in _units:
                _cache = (
                    CACHE_DIR
                    / f"tdi_{_method}_{_arm}_fold{_fold}_{CACHE_SUFFIX}.parquet"
                )
                if _cache.exists():
                    # Timed when the cache was built; the log already has it.
                    _oof = pl.read_parquet(_cache)
                else:
                    _t0 = time.time()
                    _oof = _run_unit(_method, _arm, _fold)
                    _oof.write_parquet(_cache)
                    timings.record(
                        TIMING_LOG,
                        stage="tdi_multitask",
                        method=f"{_method}_{_arm}",
                        endpoint=f"fold{_fold}",
                        mode=CACHE_SUFFIX,
                        seconds=time.time() - _t0,
                    )
                _parts.append(_oof)
                _bar.update()

    tdi_oof = pl.concat(_parts, how="vertical_relaxed")
    tdi_oof.group_by("method").len().sort("method")
    return (tdi_oof,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Tuning the decision threshold, before anything is compared

    This runs *before* the comparison tables rather than after them, because on this
    data the cutoff matters more than the model choice and a table at 0.5 would
    mostly rank models by how inflated their probabilities happen to be.

    The sweep is a coarse grid rather than an analytic optimum: MCC as a function of
    threshold is a step function with one step per distinct probability, so a fine
    grid would claim more precision than the data supports. One threshold is chosen
    per method on the **macro-averaged** MCC, since that is what rank depends on.

    **This is tuned on the same OOF folds it is then scored on**, which is honest for
    *choosing* a cutoff but mildly optimistic as a *reported* number — the same
    caveat cross-fit calibration carries in `01_baseline`. Both the 0.5 and the tuned
    columns are carried forward so the size of that effect stays visible.
    """
    )
    return


@app.cell
def _(C, evaluation, np, pl, tdi_oof):
    def _macro_mcc(frame):
        """Macro-averaged MCC over both isoforms, per (method, fold)."""
        folds = evaluation.fold_metrics_classification(frame)
        return evaluation.macro_averaged_fold_metrics(
            {
                _iso: folds.filter(pl.col("endpoint") == _iso)
                for _iso in C.TDI_ISOFORMS
            },
            metric_col="mcc",
        )

    # 0.05-0.95: a threshold outside this range predicts a single class for every
    # method here, which scores MCC 0.0 whichever class it picks.
    THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)

    _rows = []
    for _method in sorted(tdi_oof["method"].unique().to_list()):
        _sub = tdi_oof.filter(pl.col("method") == _method)
        for _t in THRESHOLD_GRID:
            _macro = _macro_mcc(
                _sub.with_columns((pl.col("y_prob") >= _t).alias("y_pred"))
            )
            _rows.append(
                {
                    "method": _method,
                    "threshold": float(_t),
                    "macro_mcc": float(_macro["mcc"].mean()),
                }
            )

    threshold_sweep = pl.DataFrame(_rows)

    # One chosen cutoff per method. Ties break toward the *larger* threshold: a
    # higher cutoff predicts fewer positives, and on an imbalanced label the
    # conservative end of a plateau is the more defensible place to sit.
    best_thresholds = dict(
        threshold_sweep.sort(["macro_mcc", "threshold"], descending=[True, True])
        .group_by("method")
        .first()
        .select("method", "threshold")
        .iter_rows()
    )

    # Relabel every row at its method's tuned cutoff. Everything downstream reads
    # this frame, so the comparison tables and the MCS grids all reflect the tuned
    # threshold rather than the 0.5 default.
    tdi_oof_tuned = tdi_oof.with_columns(
        (
            pl.col("y_prob")
            >= pl.col("method").replace_strict(best_thresholds, return_dtype=pl.Float64)
        ).alias("y_pred")
    )

    threshold_table = (
        threshold_sweep.filter(pl.col("threshold") == 0.5)
        .select("method", pl.col("macro_mcc").round(4).alias("macro_mcc_at_0.5"))
        .join(
            pl.DataFrame(
                {
                    "method": list(best_thresholds),
                    "tuned_threshold": list(best_thresholds.values()),
                }
            ),
            on="method",
            how="left",
        )
        .join(
            _macro_mcc(tdi_oof_tuned)
            .group_by("method")
            .agg(pl.col("mcc").mean().round(4).alias("macro_mcc_tuned")),
            on="method",
            how="left",
        )
        .with_columns(
            (pl.col("macro_mcc_tuned") - pl.col("macro_mcc_at_0.5"))
            .round(4)
            .alias("gain")
        )
        .sort("macro_mcc_tuned", descending=True)
    )
    threshold_table
    return (
        THRESHOLD_GRID,
        best_thresholds,
        tdi_oof_tuned,
        threshold_sweep,
        threshold_table,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Per-isoform MCC, both arms

    `delta` is positive where multitask wins, since higher MCC is better — the
    opposite sign convention to 03's ST-RAE table, which is worth reading carefully
    when comparing the two notebooks.

    CYP2D6 is the column to watch. It is the smaller isoform (1,497 rows against
    3,584) and the weaker one in `01_baseline` (MCC 0.116 against 0.259), so it is
    where borrowing representation from CYP3A4 would show up if it happens at all.
    """
    )
    return


@app.cell
def _(evaluation, pl, tdi_oof_tuned):
    # Tuned labels, not the 0.5 default -- see the threshold section above for why
    # that choice is load-bearing rather than cosmetic on this data.
    tdi_fold_scores = evaluation.fold_metrics_classification(tdi_oof_tuned)

    _scores = tdi_fold_scores.group_by(["method", "endpoint"]).agg(
        pl.col("mcc").mean()
    )
    _split = _scores.with_columns(
        pl.col("method").str.replace(r"_(single|multi)task$", "").alias("model"),
        pl.when(pl.col("method").str.ends_with("_multitask"))
        .then(pl.lit("multitask"))
        .otherwise(pl.lit("singletask"))
        .alias("arm"),
    )
    # `majority` has no multitask arm, so the pivot leaves its `multitask` cell
    # null and `delta` null with it -- which is the honest rendering: there is no
    # comparison to make for a constant predictor, and a 0.0 would read as "no
    # difference measured" rather than "not applicable".
    tdi_by_isoform = (
        _split.pivot(values="mcc", index=["model", "endpoint"], on="arm")
        .with_columns(
            pl.col("singletask").round(4),
            pl.col("multitask").round(4),
        )
        .with_columns(
            (pl.col("multitask") - pl.col("singletask")).round(4).alias("delta")
        )
        .sort(["endpoint", "delta"], descending=[False, True], nulls_last=True)
    )
    tdi_by_isoform
    return tdi_by_isoform, tdi_fold_scores


@app.cell
def _(mo):
    mo.md(
        r"""
    ### The macro-average, which is what rank depends on

    The leaderboard sorts on the plain arithmetic mean of MCC across both scored
    isoforms, not on either one — see the non-negotiable in CLAUDE.md, and
    `compute_macro_bootstrap_results` in the vendored backend.
    `macro_averaged_fold_metrics` builds the CV analogue, one value per (method,
    fold), which is also the repeated-measures unit Tukey HSD needs.

    A method can win CYP3A4 and still lose rank. Read this table, not the one above,
    when choosing what to ship.
    """
    )
    return


@app.cell
def _(C, evaluation, pl, tdi_fold_scores):
    tdi_macro_folds = evaluation.macro_averaged_fold_metrics(
        {
            _iso: tdi_fold_scores.filter(pl.col("endpoint") == _iso)
            for _iso in C.TDI_ISOFORMS
        },
        metric_col="mcc",
    )
    tdi_macro = (
        tdi_macro_folds.group_by("method")
        .agg(
            pl.col("mcc").mean().round(4).alias("macro_mcc"),
            pl.col("mcc").std().round(4).alias("std"),
            pl.len().alias("n_folds"),
        )
        .sort("macro_mcc", descending=True)
    )
    tdi_macro
    return tdi_macro, tdi_macro_folds


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Paired bootstrap, per method

    Shared folds mean both arms saw identical test rows, so this is a legitimate
    paired comparison. Read the p-value, not the point estimate: PXR's costliest
    mistake was ranking on differences the CV could not resolve, and a p above ~0.05
    means the simpler single-task arm should ship.

    Because this is a family of comparisons rather than one, the Holm-corrected
    column is the one to believe.
    """
    )
    return


@app.cell
def _(PAIRED_METHODS, evaluation, mo, pl, tdi_oof_tuned):
    _rows = []
    for _method in PAIRED_METHODS:
        _a, _b = f"{_method}_multitask", f"{_method}_singletask"
        if tdi_oof_tuned.filter(pl.col("method") == _a).height == 0:
            continue
        try:
            # paired_bootstrap takes aligned arrays: join the arms on
            # (endpoint, fold, compound) so each resample compares the same
            # measurement under both.
            _wide = (
                tdi_oof_tuned.filter(pl.col("method") == _a)
                .select(
                    "endpoint", "fold", "Molecule_Name", "y_true",
                    pl.col("y_pred").alias("pred_a"),
                )
                .join(
                    tdi_oof_tuned.filter(pl.col("method") == _b).select(
                        "endpoint", "fold", "Molecule_Name",
                        pl.col("y_pred").alias("pred_b"),
                    ),
                    on=["endpoint", "fold", "Molecule_Name"],
                    how="inner",
                )
            )
            _result = evaluation.paired_bootstrap(
                _wide["y_true"].to_numpy(),
                _wide["pred_a"].to_numpy(),
                _wide["pred_b"].to_numpy(),
                metric="mcc",
            )
            _rows.append({"model": _method, "n": _wide.height, **_result})
        except Exception as _exc:  # noqa: BLE001 - surface, do not abort the sweep
            _rows.append({"model": _method, "error": str(_exc)[:80]})

    tdi_paired = pl.DataFrame(_rows) if _rows else pl.DataFrame()

    # Holm across the family, not per comparison read in isolation.
    tdi_holm = (
        evaluation.holm_bonferroni(
            {
                r["model"]: r["p_value"]
                for r in _rows
                if "p_value" in r and r["p_value"] is not None
            }
        )
        if any("p_value" in r for r in _rows)
        else pl.DataFrame()
    )

    mo.vstack([
        mo.md("**Paired bootstrap — multitask vs single-task, per model**"),
        mo.ui.table(tdi_paired, page_size=15),
        mo.md("**Holm-corrected across the family**"),
        mo.ui.table(tdi_holm, page_size=15),
    ])
    return tdi_holm, tdi_paired


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Does multitask help? The one figure that answers it

    This notebook exists to answer a single question, and it is a *paired* one:
    `lgbm_multitask` is only ever interesting against `lgbm_singletask`. An all-pairs
    heatmap spends most of its area on the 45 cross-model comparisons nobody asked
    about, and doubles the row count doing it.

    `mcs.paired_arm_plot` collapses that to one row per model — five instead of
    eleven — showing the per-fold difference between the arms with its confidence
    interval. Both arms are scored on identical folds, so the fold-wise difference is
    a genuine repeated measure and no Tukey correction is needed for comparisons that
    are not being made.

    Read the position of each interval relative to the dashed zero line: anything
    overlapping it is a model whose two arms the CV cannot separate.

    **Read the intervals, not the p-values.** These are uncorrected paired t-tests on
    fold differences — a different and more permissive test than the paired bootstrap
    used elsewhere in this notebook, which resamples compounds and is Holm-corrected
    across the family. A borderline p here routinely fails correction; the bootstrap
    table is what settles significance. What this figure is for is the *shape*: which
    arms differ, by how much, and in which direction.
    """
    )
    return


@app.cell
def _(OUT_DIR, mcs, tdi_macro_folds):
    tdi_paired_fig = mcs.paired_arm_plot(
        tdi_macro_folds,
        metric_col="mcc",
        higher_is_better=True,
        title="Does multitask help on TDI? (macro-averaged MCC, 5x5 CV)",
        save_path=OUT_DIR / "tdi_multitask_paired.png",
    )
    tdi_paired_fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Every method against the winner

    The ranking view, in the form used in the PXR challenge deck: effect size against
    one reference with its confidence interval, colour carrying the verdict. Unlike a
    heatmap this scales with the method count, because each method takes a row rather
    than a row *and* a column.

    `top_n` drops the majority baseline and the weakest methods. They are a quarter of
    a point of MCC away, so including them compresses every contender into the right
    edge of the axis — the differences that decide a submission become invisible next
    to a difference nobody is weighing.
    """
    )
    return


@app.cell
def _(OUT_DIR, mcs, tdi_macro_folds):
    tdi_forest_fig = mcs.reference_forest_plot(
        tdi_macro_folds,
        metric_col="mcc",
        higher_is_better=True,
        top_n=8,
        title="TDI — macro-averaged MCC against the best method",
        save_path=OUT_DIR / "tdi_forest_vs_best.png",
    )
    tdi_forest_fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### All-pairs view, trimmed to the contenders

    Kept as the complete pairwise record, but `top_n=6` holds it at a size where the
    cells are actually legible. An MCS grid with few stars is itself the finding — it
    says the CV cannot resolve a ranking — which is the PXR mistake this repo exists
    to avoid repeating. That reading only works if the grid can be read.
    """
    )
    return


@app.cell
def _(OUT_DIR, mcs, tdi_macro_folds):
    tdi_mcs_fig = mcs.make_mcs_grid(
        {"Macro-averaged MCC — top 6 methods (TDI)": tdi_macro_folds},
        metric_col="mcc",
        higher_is_better={"Macro-averaged MCC — top 6 methods (TDI)": True},
        top_n=6,
        save_path=OUT_DIR / "mcs_heatmap_tdi_multitask_macro.png",
    )
    tdi_mcs_fig
    return


@app.cell
def _(OUT_DIR, C, mcs, pl, tdi_fold_scores):
    # Per-isoform panels alongside the macro grid. 01_baseline found CYP2D6's methods
    # markedly less separable than CYP3A4's; this is where that would show again.
    tdi_mcs_per_isoform = mcs.make_mcs_grid(
        {
            _iso: tdi_fold_scores.filter(pl.col("endpoint") == _iso)
            for _iso in C.TDI_ISOFORMS
        },
        metric_col="mcc",
        higher_is_better={_iso: True for _iso in C.TDI_ISOFORMS},
        top_n=6,
        save_path=OUT_DIR / "mcs_heatmap_tdi_per_isoform.png",
    )
    tdi_mcs_per_isoform
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Threshold sweep, in full

    The chosen cutoffs above come from this curve. It is worth looking at rather than
    just taking the argmax, because the *shape* is the finding: if a method's curve is
    flat across a wide band, the exact cutoff does not matter much and the choice is
    robust; if it is a narrow spike, the tuned number is fragile and the honest read
    is that the threshold was fitted to these folds.

    Multitask arms are dashed, single-task solid, so each pair reads as a pair.
    """
    )
    return


@app.cell
def _(OUT_DIR, threshold_sweep):
    import matplotlib.pyplot as plt

    _fig, _ax = plt.subplots(figsize=(9.5, 5.5))

    _methods = sorted(threshold_sweep["method"].unique().to_list())
    # Colour by *model*, linestyle by arm, so a pair shares a colour and the
    # single-task/multitask comparison reads off one line against its own twin.
    # Matplotlib's default cycle carries only 10 colours and there are 11 series
    # here, so two unrelated methods would otherwise collide on the same blue.
    _models = sorted({_m.rsplit("_", 1)[0] for _m in _methods})
    _palette = plt.get_cmap("tab10")
    _colour = {_model: _palette(_i % 10) for _i, _model in enumerate(_models)}

    for _method in _methods:
        _sub = threshold_sweep.filter(threshold_sweep["method"] == _method).sort(
            "threshold"
        )
        _is_mt = _method.endswith("_multitask")
        _ax.plot(
            _sub["threshold"],
            _sub["macro_mcc"],
            marker="o",
            markersize=3,
            color=_colour[_method.rsplit("_", 1)[0]],
            linestyle="--" if _is_mt else "-",
            label=_method,
        )
    _ax.axvline(0.5, color="grey", linestyle=":", linewidth=1)
    _ax.annotate(
        "default 0.5\n(chemprop predicts\nno positives here)",
        xy=(0.5, _ax.get_ylim()[0]),
        xytext=(5, 6),
        textcoords="offset points",
        fontsize=7,
        color="grey",
    )
    _ax.set_xlabel("decision threshold")
    _ax.set_ylabel("macro-averaged MCC")
    _ax.set_title("MCC is not optimized at 0.5 on a ~21%-positive label")
    # Outside the axes: 11 series inside the plot covered the curves they describe.
    _ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    _fig.savefig(
        OUT_DIR / "threshold_sweep.png", dpi=300, bbox_inches="tight"
    )
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## What each method cost

    Read from `experiments/04_methods_tdi/timings.csv`, which accumulates across runs
    rather than being rebuilt. Rows are per (method, arm, fold), so the per-fold cost
    is directly visible — which is what the interleaving is meant to keep flat.

    **Check Chemprop's per-fold trend specifically.** The failure this notebook's
    ordering is designed around was monotonic degradation across consecutive fits
    (~30s/fold early, ~290s/fold late). If interleaving worked, the Chemprop rows
    should be roughly flat across folds rather than climbing.
    """
    )
    return


@app.cell
def _(CACHE_SUFFIX, TIMING_LOG, mo, n_outer, timings):
    _summary = timings.summary(TIMING_LOG, mode=CACHE_SUFFIX)
    if not _summary.height:
        _timing_view = mo.md(
            "_No timings recorded yet._ Every unit was served from cache, and "
            "nothing has been timed in this mode. Delete a cache file to re-time "
            "one, or run the other mode."
        )
    elif CACHE_SUFFIX == "quick":
        _timing_view = mo.vstack([
            mo.md("**Measured (quick mode)**"),
            mo.ui.table(_summary, page_size=20),
            mo.md(f"**Projected full 5×5 run** (×{5 // max(n_outer, 1)} folds)"),
            mo.ui.table(
                timings.extrapolate(
                    TIMING_LOG, from_mode="quick", from_n_outer=n_outer
                ),
                page_size=20,
            ),
        ])
    else:
        _timing_view = mo.vstack([
            mo.md("**Measured (full run)**"),
            mo.ui.table(_summary, page_size=20),
        ])
    _timing_view
    return


@app.cell
def _(OUT_DIR, TIMING_LOG, pl, timings):
    # Per-fold Chemprop cost against fold index -- the direct test of whether
    # interleaving kept the MPS degradation from accumulating.
    import matplotlib.pyplot as plt2

    _log = timings.load(TIMING_LOG) if TIMING_LOG.exists() else pl.DataFrame()
    if _log.height and "endpoint" in _log.columns:
        _cp = _log.filter(pl.col("method").str.starts_with("chemprop")).with_columns(
            pl.col("endpoint").str.replace("fold", "").cast(pl.Int64, strict=False).alias("fold")
        ).drop_nulls("fold")
    else:
        _cp = pl.DataFrame()

    if _cp.height:
        _fig2, _ax2 = plt2.subplots(figsize=(8, 4.5))
        for _arm in sorted(_cp["method"].unique().to_list()):
            _s = _cp.filter(pl.col("method") == _arm).sort("fold")
            _ax2.plot(_s["fold"], _s["seconds"], marker="o", label=_arm)
        _ax2.set_xlabel("fold index (execution order)")
        _ax2.set_ylabel("seconds per unit")
        _ax2.set_title("Chemprop cost across folds — flat means interleaving worked")
        _ax2.legend(fontsize=8)
        _fig2.savefig(
            OUT_DIR / "chemprop_fold_timings.png", dpi=300, bbox_inches="tight"
        )
        _timing_trend = _fig2
    else:
        _timing_trend = "No Chemprop timings recorded in this mode yet."
    _timing_trend
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Findings

    *Filled in after the first full run — leave the numbers out until they exist.*

    The questions this notebook is meant to answer, in the order they matter:

    1. **Does multitask help on TDI at all?** 03's win came from 26.7% of compounds
       carrying several endpoints; only 5.4% do here. If the macro MCC delta is
       indistinguishable from zero, that is a clean result and the single-task arm
       ships — it also confirms the mechanism behind 03's win rather than leaving it
       as a plausible story.
    2. **If it does help, is it the shared encoder or the shared compounds?** A gain
       on `chemprop_multitask` with none on the stacked tree arms would point at the
       learned encoder, matching 03's conclusion that sharing a *learned*
       representation is what pays and pooling rows of a frozen one is not.
    3. **Does CYP2D6 benefit more than CYP3A4?** It is the smaller and weaker
       isoform, so it has the most to borrow. A delta concentrated there would be the
       expected shape; a delta concentrated on CYP3A4 would need explaining.
    4. **How much does the threshold matter relative to the model choice?** The quick
       run already answers this one, and the answer is "more": tuning the cutoff moved
       macro MCC further than any architectural difference did, and changed the
       ranking. Confirm it holds on the full run, then treat the cutoff as a
       first-class parameter of a TDI submission rather than a default to inherit.

    **No submission is written from this notebook yet.** The honest output of a
    comparison is a ranking plus its uncertainty; a TDI submission follows once the
    MCS grid and the paired bootstrap agree, and per CLAUDE.md it goes in its own
    dated folder with a `PROVENANCE.md` carrying an expected-performance table
    computed for the exact method, arm and threshold shipped. `submissions/01_baseline/`
    holds the last validated TDI submission (LightGBM, CYP3A4 MCC 0.259 / CYP2D6
    0.116) if one is needed before then.
    """
    )
    return


if __name__ == "__main__":
    app.run()
