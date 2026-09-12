import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # 03 — Method exploration

    `01_baseline.py` established the floor and the problem: LightGBM on ECFP4 only
    genuinely learns on CYP3A4, while CYP2C9, CYP1A2 and CYP2D6 sit at or above the
    mean-predictor line (ST-RAE ≥ 1.0) before calibration. `PLAN.md` calls that a
    representation and data problem rather than a tuning one. This notebook attacks
    the representation half, on two axes at once:

    1. **Fingerprint sweep** — the same tree ensembles across every representation in
       `fingerprints.AVAILABLE`, to find out whether ECFP4 was simply the wrong
       descriptor for the three weak endpoints.
    2. **New model families** — Chemprop (D-MPNN from scratch), CheMeleon
       (fine-tuned foundation backbone), TabPFN and TabICL (tabular foundation
       models), and Macau (Bayesian matrix factorization with fingerprint side
       information).

    ## What the literature says to expect

    [arXiv 2604.16123](https://arxiv.org/abs/2604.16123) benchmarks exactly this
    combination — tabular foundation models × molecular representations — over 58
    Polaris/MoleculeACE tasks, and two of its findings shape the design here:

    - **TabPFN+CheMeleon won**, at an 86.2% win rate (avg rank 4.52), ahead of
      fine-tuned CheMeleon (41.4%), CatBoost-Mordred (19.0%) and XGBoost-Mordred
      (10.3%). So the TFM-on-embedding cell is the one to watch.
    - **Morgan/ECFP performed "substantially worse"** with TFMs than descriptors or
      learned embeddings. A TFM on raw ECFP4 is the configuration that paper says
      not to run — so it is included here only as the control that checks whether
      that finding reproduces on our data.

    This is why the two axes are in one notebook rather than two: representation and
    model family are not independent here, and the interesting cells are in the
    interior of the grid, not the margins.

    ## TabPFN vs TabICL

    Both are run, because they fail in different directions:

    | | TabPFN 8.5 | TabICL 2.2 |
    |:--|:--|:--|
    | feature cap | **500 hard** — ECFP4-2048 and CheMeleon-2048 need PCA first | ~2,000 native |
    | regression head | binned classification of the target | 999 quantiles, pinball loss |
    | licence | non-commercial, token-gated | Apache 2.0, no gate |
    | evidence | 86.2% win rate on molecular tasks | stronger on general tabular |

    TabPFN needs `TABPFN_TOKEN`. Put it in `.env` at the project root (gitignored)
    and `tabular_models.load_env` picks it up automatically — no need to source
    anything. It must be the **API key** from the account page at ux.priorlabs.ai,
    which looks like `tabpfn_sk_...`; a session JWT copied out of the browser gets a
    401 from the licence server, and the resulting error is indistinguishable from
    having no token at all. When no usable token is present, TabPFN is skipped rather
    than failing the run, and TabICL — Apache-2.0, no gate — carries the TFM
    comparison on its own.

    ## Uncertainty is the underexploited angle

    ST-RAE scores **zero error** for any prediction landing inside a compound's
    `_conf_low`/`_conf_high` band. Three of the five new models emit a predictive
    distribution natively — TabPFN and TabICL through quantiles, Macau through its
    Gibbs posterior — so the interval-quality cell below is not decoration: it is
    measuring the thing the metric actually rewards, and the Innovation award
    explicitly credits it.

    ## Caching

    CV is the slow part and it is cached per (endpoint, method-group, mode) under
    `experiments/03_methods/cache/`, exactly as in `01_baseline.py`. The graph models
    are *far* slower than the tree sweep, so they get their own cache files: a rerun
    that only adds a fingerprint must not retrain Chemprop 25 times. Delete a cache
    file or flip `QUICK` to force a retrain.

    Set `QUICK = True` while iterating. The full run is genuinely long — see the
    runtime note by the model-family cell before starting one.

    ## Memory — read this before starting a full run

Both tabular foundation models are memory-hungry, and they are hungry in
    *different* ways — so they get separate budgets rather than one shared setting.
    Measured on a 16 GB M-series machine at n=2335 (CYP3A4, our largest endpoint),
    PCA-reduced from the 2048-dim CheMeleon embedding:

    | model | features | n_est | peak RSS | time | corr |
    |:--|--:|--:|--:|--:|--:|
    | TabICL | 2048 | 8 | **froze the machine** | — | — |
    | TabICL | 128 | 8 | 10.96 GB | 35s | — |
    | **TabICL** | **128** | **2** | **3.89 GB** | **5s** | — |
    | TabPFN | 128 | 2 | 4.50 GB | 10s | 0.158 |
    | **TabPFN** | **192** | **2** | **6.58 GB** | **9s** | **0.232** |
    | TabPFN | 256 | 4 | 8.44 GB | 22s | 0.240 |
    | TabPFN | 500 | 4 | 10.83 GB | 140s | 0.236 |

    **TabICL** attends over the feature axis and materializes its whole context at
    once, so memory scales hard with *ensemble size* — 2→8 estimators roughly tripled
    it. It does not fail cleanly: there is no `MemoryError` to catch, the machine
    starts swapping and the desktop locks up.

    **TabPFN** barely moves with ensemble size (+0.7 GB from 2 to 8) but scales with
    features — and unlike TabICL it genuinely *uses* them, with accuracy climbing
    from 0.158 at 128 features to 0.232 by 192 before plateauing. Cutting it to
    TabICL's 128 would discard real signal to save memory it does not need; pushing
    it to its 500-feature model limit costs 10.8 GB and 140s for no gain over 192.

    Hence `TABICL_MAX_FEATURES=128`/`TABICL_N_ESTIMATORS=2` and
    `TABPFN_MAX_FEATURES=192`/`TABPFN_N_ESTIMATORS=2`, with `check_tabicl_budget` and
    `check_tabpfn_budget` refusing an over-budget config before anything is
    allocated. Treat every figure as ±1–2 GB — allocator reuse makes them
    non-monotonic across runs — and leave headroom.

    Chemprop is comparatively well behaved at ~2 GB per fit, and the tree models sit
    under 1 GB.

    If you raise either limit, watch actual RSS while it runs (`ps -o rss= -p <pid>`)
    rather than trusting it to fail safely.

    ## One hazard worth knowing about

    PyTorch ships its own OpenMP runtime, and once torch is imported into a process a
    later LightGBM fit **segfaults** — a hard crash that takes the kernel down, not an
    exception you can catch. This notebook fits both, so every torch model here runs
    in a child process: `chemeleon_embed`, the Chemprop CLI, `graph_models.device`,
    and the TFMs via `tabular_models.predict_subprocess`. Do not swap those for the
    in-process classes without moving the tree models to a separate notebook.
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

    from cyp import (
        calibration,
        cv,
        data,
        evaluation,
        fingerprints,
        mcs,
        models,
        multitask,
        timings,
    )
    from cyp import constants as C
    from cyp import download as cyp_download

    return (
        C,
        Path,
        PROJECT_ROOT,
        calibration,
        cv,
        cyp_download,
        data,
        evaluation,
        fingerprints,
        mcs,
        mo,
        models,
        multitask,
        np,
        os,
        pl,
        time,
        timings,
    )


@app.cell
def _(mo):
    QUICK = mo.ui.checkbox(value=True, label="Quick mode (1 outer repeat, 1024 bits)")
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
    NOTEBOOK_NAME = Path(__file__).stem
    OUT_DIR = PROJECT_ROOT / "experiments" / NOTEBOOK_NAME
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return NOTEBOOK_NAME, OUT_DIR


@app.cell
def _(OUT_DIR, quick):
    CACHE_DIR = OUT_DIR / "cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_SUFFIX = "quick" if quick else "full"

    # Timings live in the experiment directory, not the cache: they are a small
    # derived summary worth committing, and they must survive a cache wipe. Written
    # per measurement so a run that dies mid-sweep still records what finished, and
    # read back on cached runs so the committed run does not report an empty table.
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
    ## Part 1 — Fingerprint sweep for tree ensembles

    The question: was ECFP4 the wrong representation for the three weak endpoints, or
    are they weak regardless of descriptor?

    LightGBM and XGBoost across every fingerprint, all four endpoints. `mean` is
    carried along as the anchor — under ST-RAE it scores exactly 1.0, so any cell
    above 1.0 is a model doing worse than a constant.

    Mordred is the one to watch besides ECFP: PXR found it suited XGBoost
    specifically, and the TFM paper ranks 2D descriptors alongside learned embeddings
    and well ahead of Morgan. Conformer-requiring fingerprints are excluded — they
    need an ETKDG embedding per molecule, which costs far more than the rest of this
    sweep combined and belongs in the 3D work `PLAN.md` lists separately.
    """
    )
    return


@app.cell
def _(quick):
    # Fingerprints to sweep. Cheap 2D descriptors only; see the note above on why
    # conformer-dependent ones are out of scope here.
    FP_SWEEP = ("ecfp", "maccs", "rdkit", "atompair", "torsion", "avalon", "mqn", "mordred")
    # Quick mode drops the slowest featurizers so the loop stays interactive --
    # mordred computes ~1600 descriptors per molecule.
    if quick:
        FP_SWEEP = ("ecfp", "maccs", "atompair")

    TREE_METHODS = ("mean", "lgbm", "xgb")
    return FP_SWEEP, TREE_METHODS


@app.cell
def _(
    C,
    CACHE_DIR,
    CACHE_SUFFIX,
    FP_SWEEP,
    TIMING_LOG,
    TREE_METHODS,
    data,
    evaluation,
    mo,
    models,
    n_bits,
    n_outer,
    pl,
    time,
    timings,
):
    # One cache file per (fingerprint, endpoint): adding a fingerprint later
    # recomputes only that fingerprint's folds, not the whole sweep.
    _fp_oof_parts = []

    # Total is folds, not (fingerprint x endpoint): a 5x5 run is 25 fits per cell,
    # so a per-cell bar sits still for minutes at a time and looks like a hang.
    _fp_total = len(FP_SWEEP) * len(C.REGRESSION_ENDPOINTS) * n_outer * 5
    with mo.status.progress_bar(total=_fp_total, title="Fingerprint sweep") as _bar:
        for _fp in FP_SWEEP:
            for _endpoint in C.REGRESSION_ENDPOINTS:
                _cache = CACHE_DIR / f"fp_{_fp}_{_endpoint}_{CACHE_SUFFIX}.parquet"
                if _cache.exists():
                    # Nothing is timed here on purpose: the measurement that matters
                    # was taken when this cache was built and is already in the log.
                    _oof = pl.read_parquet(_cache)
                    # Advance by the folds this cell would have run, so the bar still
                    # reaches 100% on a fully cached rerun.
                    _bar.update(n_outer * 5)
                else:
                    _t0 = time.time()
                    _oof = models.run_cv(
                        data.training_frame(_endpoint),
                        _endpoint,
                        methods=TREE_METHODS,
                        fingerprint=_fp,
                        n_bits=n_bits,
                        n_outer=n_outer,
                        # One tick per completed fold rather than per cell.
                        on_fold=lambda _f, _n: _bar.update(),
                    )
                    # Tag with the representation so every fingerprint's rows can
                    # live in one frame and be compared as if they were methods.
                    _oof = _oof.with_columns(pl.lit(_fp).alias("fingerprint"))
                    _oof.write_parquet(_cache)
                    timings.record(
                        TIMING_LOG,
                        stage="fingerprint_sweep",
                        method=_fp,
                        endpoint=_endpoint,
                        mode=CACHE_SUFFIX,
                        seconds=time.time() - _t0,
                    )
                _fp_oof_parts.append(_oof)

    fp_oof = pl.concat(_fp_oof_parts, how="vertical_relaxed")
    # "method × fingerprint" becomes the comparison unit: `lgbm_mordred` competes
    # against `lgbm_ecfp` on equal footing, which is the question Part 1 asks.
    fp_oof = fp_oof.with_columns(
        (pl.col("method") + "_" + pl.col("fingerprint")).alias("method_fp")
    )
    fp_fold_scores = evaluation.fold_metrics(
        fp_oof.drop("method").rename({"method_fp": "method"})
    )
    return fp_fold_scores, fp_oof


@app.cell
def _(C, evaluation, fp_oof, pl):
    # Wide table: rows are fingerprints, columns are endpoints, cells are mean
    # ST-RAE for the best tree model on that combination. This is the "did the
    # representation matter" view.
    _summary = evaluation.compare_methods(
        fp_oof.drop("method").rename({"method_fp": "method"})
    )
    fp_table = (
        _summary.with_columns(
            pl.col("method").str.split("_").list.get(0).alias("model"),
            pl.col("method").str.splitn("_", 2).struct.field("field_1").alias("fingerprint"),
        )
        .filter(pl.col("model") != "mean")
        .pivot(values="st_rae_mean", index=["model", "fingerprint"], on="endpoint")
        .with_columns(
            # The macro-average is what the leaderboard ranks on, so it is the
            # column to sort by -- a representation can win an endpoint and still
            # lose here. See CLAUDE.md's non-negotiable on macro-averaging.
            pl.mean_horizontal([pl.col(e) for e in C.REGRESSION_ENDPOINTS]).alias("macro")
        )
        .sort("macro")
    )
    fp_table
    return (fp_table,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Macro-averaged fingerprint comparison

    Per-endpoint tables invite the mistake CLAUDE.md warns about: a representation
    can win on CYP3A4 and still lose the rank, because the leaderboard sorts on the
    plain arithmetic mean across all four endpoints. `macro_averaged_fold_metrics`
    builds the CV analogue of that, one value per (method, fold), which is also the
    repeated-measures unit Tukey HSD needs.
    """
    )
    return


@app.cell
def _(C, evaluation, fp_fold_scores, pl):
    fp_macro_folds = evaluation.macro_averaged_fold_metrics(
        {
            _e: fp_fold_scores.filter(pl.col("endpoint") == _e)
            for _e in C.REGRESSION_ENDPOINTS
        },
        metric_col="st_rae",
    )
    fp_macro_summary = (
        fp_macro_folds.group_by("method")
        .agg(
            pl.col("st_rae").mean().alias("macro_st_rae"),
            pl.col("st_rae").std().alias("std"),
            pl.len().alias("n_folds"),
        )
        .sort("macro_st_rae")
    )
    fp_macro_summary
    return fp_macro_folds, fp_macro_summary


@app.cell
def _(OUT_DIR, fp_macro_folds, mcs):
    # An MCS grid with few stars is itself the finding: it says the CV cannot
    # resolve a ranking between these representations, which is exactly the PXR
    # mistake (three finalists within 0.0039 MAE, ordering reversed on the blind
    # set) that this repo exists to avoid repeating.
    fp_mcs_fig = mcs.make_mcs_grid(
        {"Macro-averaged ST-RAE — fingerprint × tree model": fp_macro_folds},
        metric_col="st_rae",
        higher_is_better={"Macro-averaged ST-RAE — fingerprint × tree model": False},
        save_path=OUT_DIR / "mcs_heatmap_fingerprints_macro.png",
    )
    fp_mcs_fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Part 2 — New model families

    Five additions, in rough order of expected value:

    | method | what it is | representation |
    |:--|:--|:--|
    | `chemeleon` | Chemprop D-MPNN fine-tuned from the CheMeleon backbone | molecular graph |
    | `chemprop` | Chemprop D-MPNN trained from scratch | molecular graph |
    | `tabicl` | TabICL in-context regressor | any matrix |
    | `tabpfn` | TabPFN in-context regressor (PCA→500) | any matrix |
    | `macau` | Bayesian matrix factorization, fingerprints as side info | any matrix |

    **Runtime.** Measured on this machine (CYP3A4, n=1868 train): Chemprop is ~18s
    for 10 epochs, so ~90s per fold at the default 50 epochs. A full 5×5 run is
    25 folds × 4 endpoints ≈ **2.5 h per graph model**, or ~5 h for Chemprop plus
    CheMeleon. The TFMs and Macau are minutes by comparison.

    Memory is *not* the constraint for the graph models — one fit at a time, ~2 GB —
    but time is. Run quick mode first to confirm the pipeline end to end, then launch
    the full run when you can leave it. Each method caches per endpoint, so an
    interrupted run resumes rather than restarting.

    **Validation split.** Chemprop needs a held-out set for early stopping, so this
    part passes `p_val=0.1` to `run_cv`. That carves the stopping set out of each
    training fold *within* the scaffold grouping, rather than letting the model take
    a random 10% internally — which would leak analogs from the training scaffolds
    into the stopping criterion.
    """
    )
    return


@app.cell
def _(mo, models, os):
    # TabPFN is gated behind a one-time licence acceptance and cannot prompt for it
    # from a notebook. Rather than failing a multi-hour run partway through, detect
    # it up front and drop the method -- TabICL needs no token and covers the same
    # question.
    # Reads .env if the variable is not already exported, so a token stored there
    # works without sourcing anything first.
    from cyp.tabular_models import load_env

    load_env()
    HAS_TABPFN_TOKEN = bool(os.environ.get("TABPFN_TOKEN"))

    NEW_METHODS = ["macau", "tabicl", "chemprop", "chemeleon"]
    if HAS_TABPFN_TOKEN:
        NEW_METHODS.insert(2, "tabpfn")

    _msg = (
        "TabPFN token found — including `tabpfn`."
        if HAS_TABPFN_TOKEN
        else (
            "**No `TABPFN_TOKEN` found — skipping `tabpfn`.** Put the API key "
            "(`tabpfn_sk_...`, from the account page at https://ux.priorlabs.ai) in "
            "`.env` at the project root. TabICL (Apache-2.0, no gate) still runs."
        )
    )
    mo.md(
        f"{_msg}\n\nMethods this run: `{'`, `'.join(NEW_METHODS)}` "
        f"(+ `mean` and `lgbm` as anchors).\n\n"
        f"All require the `deep` extra: `uv sync --all-extras`. "
        f"Registered: `{'`, `'.join(sorted(models.DEEP_METHODS))}`."
    )
    return (NEW_METHODS,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### CheMeleon embeddings

    Computed once for every compound and reused by the matrix-consuming models
    (`macau`, `tabicl`, `tabpfn`). This is the representation the TFM paper found
    best, and it is far too slow to regenerate per method.

    The embedding is frozen and never sees a label, so embedding the whole labelled
    set in one call leaks nothing — there is no fit to leak through.
    """
    )
    return


@app.cell
def _(C, CACHE_DIR, data, mo, np, pl):
    from cyp.graph_models import chemeleon_embed
    from cyp.tabular_models import run_cv_subprocess as run_cv_tfm

    # Cached as a plain .npy per endpoint: large, purely derived from the data
    # snapshot, no model fit involved -- which is exactly the case .gitignore's
    # experiments/**/cache/ rule covers.
    chemeleon_features = {}
    with mo.status.progress_bar(
        total=len(C.REGRESSION_ENDPOINTS), title="CheMeleon embeddings"
    ) as _bar:
        for _endpoint in C.REGRESSION_ENDPOINTS:
            _cache = CACHE_DIR / f"chemeleon_{_endpoint}.npy"
            _frame = data.training_frame(_endpoint)
            if _cache.exists():
                chemeleon_features[_endpoint] = np.load(_cache)
            else:
                # Only the train half is needed here; the second argument takes a
                # single throwaway SMILES because the function embeds two lists.
                _emb, _ = chemeleon_embed(
                    _frame["SMILES"].to_list(), _frame["SMILES"].to_list()[:1]
                )
                np.save(_cache, _emb)
                chemeleon_features[_endpoint] = _emb
            _bar.update()

    pl.DataFrame(
        {
            "endpoint": list(chemeleon_features),
            "n_compounds": [v.shape[0] for v in chemeleon_features.values()],
            "n_features": [v.shape[1] for v in chemeleon_features.values()],
        }
    )
    return chemeleon_features, run_cv_tfm


@app.cell
def _(
    C,
    CACHE_DIR,
    CACHE_SUFFIX,
    NEW_METHODS,
    TIMING_LOG,
    chemeleon_features,
    data,
    mo,
    models,
    n_bits,
    n_outer,
    pl,
    run_cv_tfm,
    time,
    timings,
):
    # Each method gets its own cache file, because they differ by orders of
    # magnitude in cost: re-running after adding TabICL must not retrain Chemprop.
    _new_oof_parts = []

    # Folds, not cells. Chemprop is ~90s per fold at 50 epochs, so a per-cell bar
    # would sit still for ~37 minutes per endpoint -- indistinguishable from a hang.
    _new_total = len(NEW_METHODS) * len(C.REGRESSION_ENDPOINTS) * n_outer * 5
    with mo.status.progress_bar(total=_new_total, title="New model families") as _bar:
        for _method in NEW_METHODS:
            # Matrix models run on CheMeleon embeddings (the TFM paper's
            # recommendation); graph models featurize from SMILES themselves and
            # ignore whatever is passed as features.
            _on_embedding = _method in ("macau", "tabicl", "tabpfn")
            _is_tfm = _method in ("tabicl", "tabpfn")

            for _endpoint in C.REGRESSION_ENDPOINTS:
                _cache = CACHE_DIR / f"new_{_method}_{_endpoint}_{CACHE_SUFFIX}.parquet"
                if _cache.exists():
                    # Timed when the cache was built; the log already has it.
                    _oof = pl.read_parquet(_cache)
                    _bar.update(n_outer * 5)
                else:
                    _t0 = time.time()
                    if _is_tfm:
                        # TFMs go through the subprocess runner, never the in-process
                        # class: `run_cv` would import torch here and every later
                        # LightGBM fit in this kernel would segfault.
                        _oof = run_cv_tfm(
                            data.training_frame(_endpoint),
                            _endpoint,
                            kind=_method,
                            features=chemeleon_features[_endpoint],
                            n_outer=n_outer,
                            on_fold=lambda _f, _n: _bar.update(),
                        )
                    else:
                        _oof = models.run_cv(
                            data.training_frame(_endpoint),
                            _endpoint,
                            methods=(_method,),
                            features=(
                                chemeleon_features[_endpoint] if _on_embedding else None
                            ),
                            n_bits=n_bits,
                            n_outer=n_outer,
                            # Graph models need a scaffold-respecting early-stopping set.
                            p_val=0.1 if _method in ("chemprop", "chemeleon") else 0.0,
                            on_fold=lambda _f, _n: _bar.update(),
                        )
                    _oof = _oof.with_columns(
                        pl.lit("chemeleon_emb" if _on_embedding else "graph").alias(
                            "representation"
                        )
                    )
                    _oof.write_parquet(_cache)
                    timings.record(
                        TIMING_LOG,
                        stage="new_models",
                        method=_method,
                        endpoint=_endpoint,
                        mode=CACHE_SUFFIX,
                        seconds=time.time() - _t0,
                    )
                _new_oof_parts.append(_oof)

    new_oof = pl.concat(_new_oof_parts, how="vertical_relaxed")
    new_oof.select("method", "endpoint").unique().sort(["method", "endpoint"])
    return (new_oof,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Part 2b — Multitask: one model for all four endpoints

    Everything above trains **one model per endpoint**, which is the right default
    given that the endpoints do not share rows. This part adds the controlled
    alternative, so "does sharing across endpoints help?" gets measured rather than
    assumed.

    **Why it might help.** 1,309 of the 4,905 labelled compounds carry more than one
    endpoint. Three endpoints sit at or above the mean-predictor line while CYP3A4
    has 2,335 rows and genuinely learns, so there is a plausible route for the weak
    endpoints to borrow strength from the strong one — a data lever that costs no new
    data.

    **Why it might not.** PXR's retrospective records multitask and auxiliary-assay
    training *hurting* there, despite top teams extracting value from the same
    sources. Hence a control arm, not a replacement.

    ### Folds must be shared, or the comparison is worthless

    Both arms use `cv.shared_scaffold_folds`, which assigns folds over the **union**
    of compounds rather than per endpoint. This is not tidiness — it is correctness.
    With per-endpoint splits a compound could sit in CYP3A4's training set and
    CYP2D6's test set simultaneously, so a multitask model would be scored on a
    compound whose label it had already partly seen. Verified on the current
    snapshot: zero cross-endpoint overlap between train and test in every fold.

    Sharing folds is also what makes the two arms *paired* — identical test rows, so
    `evaluation.paired_bootstrap` is legitimate rather than an eyeball comparison.

    ### "Multitask" means three different things here

    | strategy | methods | what it does |
    |:--|:--|:--|
    | `stacked` | trees, TFMs | one model, all 6,525 rows, endpoint one-hot encoded |
    | `native` | macau | sparse (n_compounds × 4) factorization, missing entries absent |
    | `multitarget` | chemprop, chemeleon | one D-MPNN, four heads, missing targets masked |

    A single-output regressor can only be row-stacked. Macau's whole formulation *is*
    a sparse matrix, so it needs no encoding trick. Chemprop has a real multi-target
    head that masks unmeasured cells — which matters, because only 41 of 4,905
    compounds have all four endpoints, so dropping incomplete rows would leave almost
    nothing.
    """
    )
    return


@app.cell
def _(C, cv, data, mo, pl):
    # One fold assignment, reused by both arms and every method. Computed once here
    # rather than per method so nothing can drift between runs.
    mt_frames = {_e: data.training_frame(_e) for _e in C.REGRESSION_ENDPOINTS}
    _fold_of, FOLD_ASSIGNMENTS = cv.shared_scaffold_folds(mt_frames)

    # Leakage check, run every time rather than trusted: a compound held out for one
    # endpoint must not be in training for another. This is the assumption the whole
    # multitask comparison rests on, and it is cheap to verify.
    _leaks = 0
    for _fold in FOLD_ASSIGNMENTS["fold"].unique().to_list():
        _test_names, _train_names = set(), set()
        for _e, _f in mt_frames.items():
            for _fd, _o, _i, _tr, _v, _te in cv.fold_assignment_splits(
                _f, FOLD_ASSIGNMENTS
            ):
                if _fd != _fold:
                    continue
                _test_names |= set(_te["Molecule_Name"].to_list())
                _train_names |= set(_tr["Molecule_Name"].to_list())
        _leaks += len(_test_names & _train_names)

    mo.md(
        f"Assigned **{FOLD_ASSIGNMENTS['Molecule_Name'].n_unique():,} compounds** to "
        f"{FOLD_ASSIGNMENTS['fold'].n_unique()} folds across "
        f"{FOLD_ASSIGNMENTS['outer_fold'].n_unique()} outer repeats.\n\n"
        + (
            f"✅ **No cross-endpoint leakage** ({_leaks} compounds in both train and "
            "test across endpoints)."
            if _leaks == 0
            else f"🚨 **{_leaks} compounds leak across endpoints** — the multitask "
            "comparison is invalid until this is fixed."
        )
    )
    return FOLD_ASSIGNMENTS, mt_frames


@app.cell
def _(mo):
    mo.md(
        r"""
    Both arms now run on those folds. The single-task arm is re-run here rather than
    reused from Part 1/2, because those used per-endpoint splits — the point of this
    section is that both arms see *identical* test rows.
    """
    )
    return


@app.cell
def _(NEW_METHODS):
    # Every method that has a multitask formulation, plus the tree baselines. `mean`
    # is excluded: predicting a constant cannot borrow strength, so a multitask
    # version of it would be the same model with extra steps.
    MT_METHODS = ["lgbm", "xgb", *NEW_METHODS]
    return (MT_METHODS,)


@app.cell
def _(
    C,
    CACHE_DIR,
    CACHE_SUFFIX,
    FOLD_ASSIGNMENTS,
    MT_METHODS,
    TIMING_LOG,
    chemeleon_features,
    cv,
    mo,
    models,
    mt_frames,
    multitask,
    n_bits,
    n_outer,
    np,
    pl,
    time,
    timings,
):
    def _single_task_on_shared_folds(method, endpoint, frame, features, on_fold):
        """Single-task CV driven by the shared fold assignment.

        `models.run_cv` draws its own per-endpoint scaffold splits, which is correct
        for Part 1 but wrong here -- the multitask arm must be scored on the same
        rows, so the folds have to come from FOLD_ASSIGNMENTS instead.
        """
        indexed = frame.with_row_index("_row")
        _X = (
            np.asarray(features)
            if features is not None
            else models.fingerprints.compute(
                frame["SMILES"].to_list(), "ecfp", fp_size=n_bits
            )
        )
        _rows = []
        for _fd, _o, _i, _tr, _v, _te in cv.fold_assignment_splits(
            indexed, FOLD_ASSIGNMENTS, p_val=0.1 if method in ("chemprop", "chemeleon") else 0.0
        ):
            _model = models.MODEL_FACTORIES[method]()
            if models._needs_smiles(_model):
                _kw = (
                    {"smiles_val": _v["SMILES"].to_list(), "y_val": _v["y_true"].to_numpy()}
                    if _v is not None
                    else {}
                )
                _model.fit(_tr["SMILES"].to_list(), _tr["y_true"].to_numpy(), **_kw)
                _p = np.asarray(_model.predict(_te["SMILES"].to_list()), dtype=float)
            else:
                _model.fit(_X[_tr["_row"].to_numpy()], _tr["y_true"].to_numpy())
                _p = np.asarray(_model.predict(_X[_te["_row"].to_numpy()]), dtype=float)
            for _k in range(_te.height):
                _rows.append(
                    {
                        "method": f"{method}_singletask",
                        "endpoint": endpoint,
                        "fold": _fd,
                        "outer_fold": _o,
                        "inner_fold": _i,
                        "Molecule_Name": _te["Molecule_Name"][_k],
                        "y_true": float(_te["y_true"][_k]),
                        "y_pred": float(_p[_k]),
                        "y_lower": _te["y_lower"][_k],
                        "y_upper": _te["y_upper"][_k],
                    }
                )
            on_fold()
        return pl.DataFrame(_rows)

    # The worst offender before this change: 12 ticks for the entire section, so a
    # single-task tick hid 25 folds x 4 endpoints = 100 model fits. Count folds
    # instead -- the multitask arm is one pass over the folds, the single-task arm
    # is one pass per endpoint.
    _n_fold = n_outer * 5
    _mt_total = len(MT_METHODS) * _n_fold * (1 + len(C.REGRESSION_ENDPOINTS))
    _mt_parts = []
    with mo.status.progress_bar(
        total=_mt_total, title="Multitask vs single-task"
    ) as _bar:
        for _method in MT_METHODS:
            _on_emb = _method in ("macau", "tabicl", "tabpfn")
            _feat = chemeleon_features if _on_emb else None

            for _arm in ("singletask", "multitask"):
                _cache = CACHE_DIR / f"mt_{_method}_{_arm}_{CACHE_SUFFIX}.parquet"
                if _cache.exists():
                    _oof = pl.read_parquet(_cache)
                    # The multitask arm is one pass over the folds; the single-task
                    # arm is one pass per endpoint.
                    _bar.update(
                        _n_fold
                        if _arm == "multitask"
                        else _n_fold * len(C.REGRESSION_ENDPOINTS)
                    )
                else:
                    _t0 = time.time()
                    if _arm == "multitask":
                        _oof = multitask.run_cv_multitask(
                            mt_frames,
                            _method,
                            features=_feat,
                            n_bits=n_bits,
                            n_outer=n_outer,
                            assignments=FOLD_ASSIGNMENTS,
                            on_fold=lambda _f, _n: _bar.update(),
                        )
                    else:
                        _oof = pl.concat(
                            [
                                _single_task_on_shared_folds(
                                    _method,
                                    _e,
                                    _f,
                                    _feat[_e] if _feat is not None else None,
                                    lambda: _bar.update(),
                                )
                                for _e, _f in mt_frames.items()
                            ],
                            how="vertical_relaxed",
                        )
                    _oof.write_parquet(_cache)
                    timings.record(
                        TIMING_LOG,
                        stage="multitask",
                        method=f"{_method}_{_arm}",
                        endpoint="all",
                        mode=CACHE_SUFFIX,
                        seconds=time.time() - _t0,
                    )
                _mt_parts.append(_oof)

    mt_oof = pl.concat(_mt_parts, how="vertical_relaxed")
    mt_oof.group_by("method").len().sort("method")
    return (mt_oof,)


@app.cell
def _(C, evaluation, mt_oof, pl):
    # Per-endpoint, both arms side by side. `delta` is negative where multitask wins,
    # since lower ST-RAE is better.
    _scores = (
        evaluation.fold_metrics(mt_oof)
        .group_by(["method", "endpoint"])
        .agg(pl.col("st_rae").mean())
    )
    _split = _scores.with_columns(
        pl.col("method").str.replace(r"_(single|multi)task$", "").alias("model"),
        pl.when(pl.col("method").str.ends_with("_multitask"))
        .then(pl.lit("multitask"))
        .otherwise(pl.lit("singletask"))
        .alias("arm"),
    )
    mt_by_endpoint = (
        _split.pivot(values="st_rae", index=["model", "endpoint"], on="arm")
        .with_columns((pl.col("multitask") - pl.col("singletask")).round(3).alias("delta"))
        .with_columns(
            pl.col("singletask").round(3), pl.col("multitask").round(3)
        )
        .sort(["endpoint", "delta"])
    )
    mt_by_endpoint
    return (mt_by_endpoint,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### The macro-average, which is what rank depends on

    Per-endpoint deltas can point in opposite directions — the expected pattern is
    that multitask helps the data-poor endpoints and does nothing for CYP3A4, which
    has enough rows of its own. The leaderboard sorts on the plain mean across all
    four, so that is the number that decides whether multitask is worth shipping.
    """
    )
    return


@app.cell
def _(C, evaluation, mt_oof, pl):
    _folds = evaluation.fold_metrics(mt_oof)
    mt_macro_folds = evaluation.macro_averaged_fold_metrics(
        {
            _e: _folds.filter(pl.col("endpoint") == _e)
            for _e in C.REGRESSION_ENDPOINTS
        },
        metric_col="st_rae",
    )
    mt_macro = (
        mt_macro_folds.group_by("method")
        .agg(
            pl.col("st_rae").mean().round(4).alias("macro_st_rae"),
            pl.col("st_rae").std().round(4).alias("std"),
            pl.len().alias("n_folds"),
        )
        .sort("macro_st_rae")
    )
    mt_macro
    return mt_macro, mt_macro_folds


@app.cell
def _(MT_METHODS, evaluation, mo, mt_oof, pl):
    # The paired test. Shared folds mean both arms saw identical test rows, so this
    # is a legitimate paired comparison rather than two independent runs eyeballed
    # against each other. Read the p-value, not the point estimate: PXR's costliest
    # mistake was ranking on differences the CV could not resolve.
    _rows = []
    for _method in MT_METHODS:
        _a, _b = f"{_method}_multitask", f"{_method}_singletask"
        if mt_oof.filter(pl.col("method") == _a).height == 0:
            continue
        try:
            _result = evaluation.paired_bootstrap(
                mt_oof, method_a=_a, method_b=_b, metric="st_rae"
            )
            _rows.append({"model": _method, **_result})
        except Exception as _exc:  # noqa: BLE001 - surface, do not abort the sweep
            _rows.append({"model": _method, "error": str(_exc)[:80]})

    mt_paired = pl.DataFrame(_rows) if _rows else pl.DataFrame()
    mo.vstack([
        mo.md("**Paired bootstrap — multitask vs single-task, per model**"),
        mo.ui.table(mt_paired, page_size=15),
        mo.md(
            "A p-value above ~0.05 means the CV cannot resolve the two arms for that "
            "model, and the simpler single-task version should ship. Apply "
            "`evaluation.holm_bonferroni` across this column before believing any "
            "individual result — this is a family of comparisons, not one."
        ),
    ])
    return (mt_paired,)


@app.cell
def _(OUT_DIR, mcs, mt_macro_folds):
    mt_mcs_fig = mcs.make_mcs_grid(
        {"Macro-averaged ST-RAE — multitask vs single-task": mt_macro_folds},
        metric_col="st_rae",
        higher_is_better={"Macro-averaged ST-RAE — multitask vs single-task": False},
        save_path=OUT_DIR / "mcs_heatmap_multitask_macro.png",
        figsize=(11, 9),
    )
    mt_mcs_fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### What each method cost

    Read from `experiments/03_methods/timings.csv`, which accumulates across runs
    rather than being rebuilt each time. This matters because CV is cached: on a
    rerun the training branch is skipped entirely, so a timer inside it would record
    nothing and the committed run — the one anyone reads later — would show an empty
    table. Each measurement is written when the work actually happens and read back
    afterwards.

    The projection multiplies quick-mode time by the fold ratio. CV cost is close to
    linear in fold count, so it is a fair order-of-magnitude guide, but it ignores
    per-fold fixed costs (subprocess launch, checkpoint load) that are amortized in a
    long run — so it over-estimates slightly for the fast methods. Use it to decide
    what is affordable before starting a five-hour run, not as a promise.
    """
    )
    return


@app.cell
def _(CACHE_SUFFIX, TIMING_LOG, mo, n_outer, timings):
    _summary = timings.summary(TIMING_LOG, mode=CACHE_SUFFIX)
    if not _summary.height:
        _timing_view = mo.md(
            "_No timings recorded yet._ Every method was served from cache, and "
            "nothing has been timed in this mode. Delete a cache file to re-time "
            "one, or run the other mode."
        )
    elif CACHE_SUFFIX == "quick":
        _timing_view = mo.vstack([
            mo.md("**Measured (quick mode)**"),
            mo.ui.table(_summary, page_size=20),
            mo.md(f"**Projected full 5×5 run** (×{5 // max(n_outer, 1)} folds)"),
            mo.ui.table(
                timings.extrapolate(TIMING_LOG, from_mode="quick", from_n_outer=n_outer),
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
def _(mo):
    mo.md(
        r"""
    ## Part 3 — Everything against everything

    The comparison that matters. Tree baselines on their best fingerprint, plus all
    five new families, scored on the macro-averaged ST-RAE the leaderboard actually
    ranks on.

    `mean` anchors the table at 1.0 by construction. Read the `std` column before the
    ordering: if two methods differ by less than the fold-to-fold spread, they are
    tied regardless of which sorts higher, and the MCS grid below is what settles it.
    """
    )
    return


@app.cell
def _(evaluation, fp_oof, new_oof, pl):
    # Fold the fingerprint sweep and the new families into one frame. The tree rows
    # keep their fingerprint tag so `lgbm_mordred` stays distinguishable from
    # `lgbm_ecfp`; the new families are already unambiguous.
    _trees = (
        fp_oof.drop("method")
        .rename({"method_fp": "method"})
        .drop("fingerprint")
        .with_columns(pl.lit("fingerprint").alias("representation"))
    )
    combined_oof = pl.concat([_trees, new_oof], how="vertical_relaxed")
    combined_fold_scores = evaluation.fold_metrics(combined_oof)
    return combined_fold_scores, combined_oof


@app.cell
def _(C, combined_fold_scores, evaluation, pl):
    combined_macro_folds = evaluation.macro_averaged_fold_metrics(
        {
            _e: combined_fold_scores.filter(pl.col("endpoint") == _e)
            for _e in C.REGRESSION_ENDPOINTS
        },
        metric_col="st_rae",
    )
    combined_macro_summary = (
        combined_macro_folds.group_by("method")
        .agg(
            pl.col("st_rae").mean().alias("macro_st_rae"),
            pl.col("st_rae").std().alias("std"),
            pl.len().alias("n_folds"),
        )
        .sort("macro_st_rae")
    )
    combined_macro_summary
    return combined_macro_folds, combined_macro_summary


@app.cell
def _(C, combined_fold_scores, mo, pl):
    # Per-endpoint view, kept as a cross-check rather than the headline. CYP2D6 is
    # the endpoint to read carefully: 01_baseline left it *above* 1.0 (worse than a
    # constant), and PLAN.md's hypothesis is that its basic-nitrogen/aromatic
    # pharmacophore is what ECFP represents poorly -- so if graph models or
    # embeddings fix anything, this is the column where it should show.
    per_endpoint = (
        combined_fold_scores.group_by(["method", "endpoint"])
        .agg(pl.col("st_rae").mean().alias("st_rae"))
        .pivot(values="st_rae", index="method", on="endpoint")
        .with_columns(
            pl.mean_horizontal([pl.col(e) for e in C.REGRESSION_ENDPOINTS]).alias("macro")
        )
        .sort("macro")
    )
    mo.ui.table(per_endpoint, page_size=30)
    return (per_endpoint,)


@app.cell
def _(OUT_DIR, combined_macro_folds, mcs):
    combined_mcs_fig = mcs.make_mcs_grid(
        {"Macro-averaged ST-RAE — all methods": combined_macro_folds},
        metric_col="st_rae",
        higher_is_better={"Macro-averaged ST-RAE — all methods": False},
        save_path=OUT_DIR / "mcs_heatmap_all_methods_macro.png",
        figsize=(11, 9),
    )
    combined_mcs_fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Is the winner actually better than LightGBM?

    The MCS grid gives the all-pairs view; this is the direct pairwise test on the
    one comparison that decides whether any of this work changes the submission.
    PXR's costliest lesson was ranking models on differences the data could not
    resolve — three finalists within 0.0039 MAE, ordering reversed on the blind set,
    five leaderboard places lost. A high p-value here means "keep the simpler model",
    not "try harder to see a difference".
    """
    )
    return


@app.cell
def _(combined_macro_summary, combined_oof, evaluation, mo, pl):
    _ranked = combined_macro_summary.filter(pl.col("method") != "mean")
    _best = _ranked.row(0, named=True)["method"]
    # The incumbent: whichever LightGBM-on-a-fingerprint variant did best. That is
    # the model a submission would otherwise use, so it is the thing to beat.
    _lgbm_rows = _ranked.filter(pl.col("method").str.starts_with("lgbm"))
    _incumbent = _lgbm_rows.row(0, named=True)["method"] if _lgbm_rows.height else "mean"

    if _best == _incumbent:
        _verdict = mo.md(
            f"The best method **is** the incumbent (`{_incumbent}`) — nothing here "
            "displaces it, so no pairwise test is needed."
        )
    else:
        _p = evaluation.paired_bootstrap(
            combined_oof, method_a=_best, method_b=_incumbent, metric="st_rae"
        )
        _verdict = mo.md(
            f"**`{_best}` vs `{_incumbent}`** (macro-ranked best vs best LightGBM):\n\n"
            f"```\n{_p}\n```\n\n"
            "Read the p-value, not the point estimate. Above ~0.05 the CV cannot "
            "resolve these two, and the simpler/faster model should ship."
        )
    _verdict
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Part 4 — Predictive intervals

    The angle `01_baseline.py` could not touch. LightGBM emits a point estimate;
    TabPFN, TabICL and Macau emit a distribution. ST-RAE forgives any error landing
    inside a compound's credible interval, so a model that knows which compounds it
    is unsure about has something directly exploitable here — and CLAUDE.md notes the
    Innovation award rewards exactly this.

    Two things are measured, on a single scaffold-held-out split rather than the full
    CV (this is a diagnostic, not a ranking):

    - **Coverage** — does a nominal 80% interval actually contain ~80% of the true
      values? The TFM literature reports these are well calibrated out of the box;
      that claim is worth checking on our own data rather than inherited.
    - **Width** — a wide interval trivially achieves coverage and says nothing. Both
      numbers only mean something together.
    """
    )
    return


@app.cell
def _(C, chemeleon_features, data, mo, np, pl):
    from cyp.cv import scaffold_splits
    from cyp.matrix_factorization import MacauModel
    from cyp.tabular_models import predict_subprocess

    _interval_rows = []
    with mo.status.progress_bar(
        total=len(C.REGRESSION_ENDPOINTS), title="Interval diagnostics"
    ) as _bar:
        for _endpoint in C.REGRESSION_ENDPOINTS:
            _frame = data.training_frame(_endpoint).with_row_index("_row")
            _X = chemeleon_features[_endpoint]

            # One scaffold-grouped split, not the full 5x5: this cell answers "are
            # these intervals honest", which does not need 25 folds.
            _fold, _, _, _train, _, _test = next(
                iter(scaffold_splits(_frame, n_outer=1, n_inner=5))
            )
            _Xtr, _Xte = _X[_train["_row"].to_numpy()], _X[_test["_row"].to_numpy()]
            _ytr, _yte = _train["y_true"].to_numpy(), _test["y_true"].to_numpy()

            for _name in ("tabicl", "macau"):
                if _name == "macau":
                    # Macau is pure C++/OpenMP, no torch, so it runs in-process.
                    # Its posterior spread is a std rather than a quantile; the
                    # normal approximation makes it comparable to the TFM interval.
                    _model = MacauModel(num_latent=16, burnin=50, nsamples=100)
                    _model.fit(_Xtr, _ytr)
                    _mu = _model.predict(_Xte)
                    _sd = _model.predict_std(_Xte)
                    _lo, _hi = _mu - 1.2816 * _sd, _mu + 1.2816 * _sd
                else:
                    # Subprocess, for the OpenMP reason in the header.
                    _, _lo, _hi = predict_subprocess(
                        _Xtr, _ytr, _Xte, kind=_name, want_interval=True
                    )

                _interval_rows.append(
                    {
                        "endpoint": _endpoint,
                        "method": _name,
                        "n_test": len(_yte),
                        # Nominal 80%: the gap between this and 0.80 is the
                        # miscalibration, in either direction.
                        "coverage_80": float(np.mean((_yte >= _lo) & (_yte <= _hi))),
                        "mean_width": float(np.mean(_hi - _lo)),
                        # The challenge's own intervals, for scale: an interval much
                        # wider than the assay's is not useful under ST-RAE.
                        "challenge_ci_width": float(
                            np.nanmean(
                                _test["y_upper"].to_numpy() - _test["y_lower"].to_numpy()
                            )
                        ),
                    }
                )
            _bar.update()

    interval_table = pl.DataFrame(_interval_rows).sort(["endpoint", "method"])
    interval_table
    return (interval_table,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Part 5 — Calibration and bias

    Two checks carried forward from `01_baseline.py`, because they are the two
    failure modes this repo has already paid for once.

    **Calibration.** Linear calibration improved *every* endpoint in the baseline
    (CYP2D6: 1.063 → 0.947). It was dismissed in PXR CV at 0.0001 MAE and then won
    the blind set. Cross-fit so the evaluation stays honest.

    **Regression to the mean.** PXR's dominant failure mode, already visible here —
    LightGBM underpredicts potent CYP3A4 compounds by ~1 log unit. A model whose bias
    profile is flatter across potency bins is worth more than its headline ST-RAE
    suggests, because the potent tail is where a blind set will punish you.
    """
    )
    return


@app.cell
def _(C, calibration, combined_macro_summary, combined_oof, evaluation, pl):
    _top = (
        combined_macro_summary.filter(pl.col("method") != "mean")
        .head(4)["method"]
        .to_list()
    )

    _calib_rows = []
    for _method in _top:
        _sub = combined_oof.filter(pl.col("method") == _method)
        for _kind in ("raw", "linear", "isotonic"):
            _cal = calibration.crossfit_calibrate(_sub, kind=_kind)
            _scores = evaluation.fold_metrics(_cal, pred_col="y_pred_cal")
            _macro = evaluation.macro_averaged_fold_metrics(
                {
                    _e: _scores.filter(pl.col("endpoint") == _e)
                    for _e in C.REGRESSION_ENDPOINTS
                },
                metric_col="st_rae",
            )
            _calib_rows.append(
                {
                    "method": _method,
                    "calibration": _kind,
                    "macro_st_rae": float(_macro["st_rae"].mean()),
                }
            )

    calibration_table = (
        pl.DataFrame(_calib_rows)
        .pivot(values="macro_st_rae", index="method", on="calibration")
        .sort("linear")
    )
    calibration_table
    return (calibration_table,)


@app.cell
def _(combined_macro_summary, combined_oof, evaluation, pl):
    _best = (
        combined_macro_summary.filter(pl.col("method") != "mean").row(0, named=True)["method"]
    )
    bias_table = evaluation.bias_by_potency_bin(
        combined_oof.filter(pl.col("method").is_in([_best, "mean"]))
    )
    bias_table
    return (bias_table,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Findings

    *Filled in after the first full run — leave the numbers out until they exist.*

    Questions this notebook is meant to answer, in the order they matter:

    1. **Does any representation rescue CYP2C9 / CYP1A2 / CYP2D6?** All three sat at
       or above ST-RAE 1.0 on ECFP4. If Mordred or CheMeleon pulls them below 1.0,
       the baseline's weakness was the descriptor. If nothing does, it is the data,
       and `PLAN.md` item 3 (public data) becomes the priority over further modelling.
    2. **Do graph models beat trees on the macro-average?** CheMeleon won every PXR
       CV comparison. If it does not here, that is a genuine difference between the
       two challenges worth understanding before spending more time on it.
    3. **Does TabPFN+CheMeleon reproduce its 86.2% win rate on our endpoints?** If
       the TFM result holds, it is the cheapest strong model available — no training
       at all. If TabICL matches it, prefer TabICL: Apache-2.0 versus a
       non-commercial licence, on a public challenge submission.
    4. **Are the predictive intervals honest?** Coverage near 0.80 at a width no
       wider than the assay's own credible intervals would make an uncertainty-aware
       submission viable, which is both an ST-RAE lever and the Innovation angle.
    5. **Does Macau's multitask structure help?** It is the one model here that can
       use all four endpoints' rows at once without a dense matrix. PXR found
       multitask *hurt*, so the single-task Macau row is the control that says
       whether that repeats.

    **No submission is written from this notebook.** It is a comparison, and the
    honest output of a comparison is a ranking plus its uncertainty. A submission
    follows once the MCS grid and the paired bootstrap agree on a winner — and per
    CLAUDE.md, it goes in its own dated folder with a `PROVENANCE.md` carrying an
    expected-performance table computed for the exact method and calibration shipped.
    """
    )
    return


if __name__ == "__main__":
    app.run()
