import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # 05 — Auxiliary and public data

    `03_methods.py` established that **representation moved the numbers far more than
    model choice**: ECFP4 → Mordred → CheMeleon bought ~0.17 on the macro ST-RAE,
    while swapping models on a frozen CheMeleon bought ~0.012. What broke past the
    frozen-representation ceiling was Chemprop learning its own encoder and sharing it
    across four masked output heads, for a macro of **0.7280**.

    That result exhausted the levers available inside the dose-response table. This
    notebook opens the two that are left: **data the challenge shipped but no notebook
    has touched**, and **public data**.

    ## What has not been used yet

    Two of the five challenge files carry no scored endpoint, and notebooks 01–04
    ignored both:

    | file | rows | what it is |
    |:--|--:|:--|
    | `single-concentration-TRAIN.csv` | 17,504 | primary screen, 4,376 cpds × 4 isoforms |
    | `TRAIN_Emax.csv` | 6,145 | max effect vs positive control, both conditions |

    The screen turns out to matter a great deal and Emax almost not at all — both
    measured below rather than assumed.

    ## The constraint that shapes every arm

    The blinded test set is `Molecule_Name,SMILES` and nothing else. No screen
    measurement, no Emax, no public assay record, and **zero** of its 750 compounds
    appear in the screen. So none of this data can ever be a model *feature*:
    whatever a model reads at training time has to exist at prediction time too.

    That leaves exactly three shapes, and the four arms below are built from them:

    | arm | shape | what it tests |
    |:--|:--|:--|
    | **A1** `aux_screen` | auxiliary **target** | 4 extra heads predicting log2fc, never scored |
    | **A2** `augmented` | extra **rows** | screen negatives as censored weak labels |
    | **A3** `pubchem` | **pretraining** | encoder pretrained on the NCGC qHTS panel |
    | **A4** `selection` | **reweighting** | inverse-propensity correction for the triage |

    Anything that looks like a fourth shape is usually feature leakage wearing a hat.

    ## The prior, and why every arm is paired

    PXR's retrospective is unambiguous: *"multitask/pretraining on auxiliary assay
    data hurt in PXR despite helping top teams"*. That is a strong prior that the
    direction of these effects is not predictable from the setup. So every arm runs
    against `chemprop_multitask` — the 03 winner, re-run here on identical folds — and
    nothing is believed on the strength of a point estimate.
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
        auxiliary,
        calibration,
        cv,
        data,
        evaluation,
        external,
        mcs,
        multitask,
        submission,
        timings,
    )
    from cyp import constants as C
    from cyp import download as cyp_download

    return (
        C,
        Path,
        PROJECT_ROOT,
        auxiliary,
        aux_training,
        calibration,
        cv,
        cyp_download,
        data,
        date,
        evaluation,
        external,
        mcs,
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
        value=False, label="Quick mode (1 outer repeat, 10 epochs, small pretrain)"
    )
    QUICK
    return (QUICK,)


@app.cell
def _(QUICK, mo, os):
    quick = QUICK.value
    n_outer = 1 if quick else 5
    chemprop_epochs = 10 if quick else 50
    pretrain_epochs = 5 if quick else 30

    # 04 measured this directly: on MPS, chemprop climbed from ~100s to ~2000s per
    # fold and stayed there, while CPU folds held flat at ~76s. A 408K-parameter
    # model at batch 64 never gives the GPU enough work to amortise launch overhead,
    # so CPU is both stable and faster outright. Set here rather than left to the
    # caller's environment, because this notebook is almost entirely chemprop.
    os.environ.setdefault("CYP_CHEMPROP_DEVICE", "cpu")

    mo.md(
        f"Running with `n_outer={n_outer}`, `chemprop_epochs={chemprop_epochs}`, "
        f"`pretrain_epochs={pretrain_epochs}`, "
        f"`CYP_CHEMPROP_DEVICE={os.environ['CYP_CHEMPROP_DEVICE']}`."
    )
    return chemprop_epochs, n_outer, pretrain_epochs, quick


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

    # Written per measurement, not at the end: on every run after the first the
    # training branch is skipped, so a timer inside it would measure nothing and the
    # committed run would report an empty table.
    TIMING_LOG = OUT_DIR / "timings.csv"
    return CACHE_DIR, CACHE_SUFFIX, TIMING_LOG


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Data — challenge and public

    Both checked and downloaded if absent, so a fresh clone runs this notebook
    without a separate setup step. The two snapshot trees are kept apart
    (`data/raw/` and `data/external/`) because they move on independent schedules:
    OpenADMET has amended its release mid-challenge before, and PubChem changes on
    its own. Conflating them would make it impossible to say which one moved under a
    result.
    """
    )
    return


@app.cell
def _(C, cyp_download, external, mo):
    if not C.available_snapshots():
        mo.output.append(mo.md("No challenge snapshot found — downloading now..."))
        cyp_download.download()
    CHALLENGE_SNAPSHOT = C.latest_snapshot()

    # Same check-then-fetch contract for the public data. Returns the snapshot that
    # actually has every assay on disk, downloading only if none does.
    if not C.available_external_snapshots():
        mo.output.append(mo.md("No external snapshot found — downloading PubChem..."))
    EXTERNAL_SNAPSHOT = external.ensure_downloaded()

    mo.output.append(
        mo.md(
            f"Challenge snapshot `{CHALLENGE_SNAPSHOT}`, "
            f"external snapshot `{EXTERNAL_SNAPSHOT}`."
        )
    )
    return CHALLENGE_SNAPSHOT, EXTERNAL_SNAPSHOT


@app.cell
def _(mo):
    mo.md(
        r"""
    ### What the single-concentration screen actually contains

    Every compound tested against all four isoforms at one concentration (49.5 µM),
    one measurement each, no replicates. The readout `log2fc_estimate` is the log2
    fold change in enzyme activity against control, so **more negative means more
    inhibition**.

    Two properties make it interesting. It is **dense** — 4,376 compounds × 4
    isoforms with no missing cells, where the pIC50 table is sparse and the four
    endpoints share almost no rows. And it is **strongly predictive** of the thing
    being scored.
    """
    )
    return


@app.cell
def _(auxiliary):
    screen_agreement = auxiliary.screen_label_agreement()
    screen_agreement
    return (screen_agreement,)


@app.cell
def _(mo, screen_agreement, pl):
    _worst = screen_agreement.sort("r2_log2fc_only").row(0, named=True)
    _best = screen_agreement.sort("r2_log2fc_only", descending=True).row(0, named=True)
    mo.md(
        f"""
    A **single scalar** explains between
    {100 * _worst['r2_log2fc_only']:.0f}% ({_worst['endpoint'].split('_')[0]}) and
    {100 * _best['r2_log2fc_only']:.0f}% ({_best['endpoint'].split('_')[0]}) of the
    variance in pIC50. For comparison, the entire modelling effort in `03_methods`
    took the macro ST-RAE from 0.93 to 0.73.

    The screen also covers
    **{screen_agreement['n_screened_unlabelled'].sum():,}** compound/isoform pairs
    that have no dose-response label at all — roughly twice the labelled count. That
    is the coverage A1 and A2 are after.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### The screen is the triage rule that created the sparsity

    This is the finding that reframes the whole CYP2D6 problem. A compound has a
    pIC50 **only because it hit in this screen** — so the labelled set is not a sample
    of chemical space, it is a truncated selection, and the truncation is visible.

    Read the `separation` column as `unlabelled_min − labelled_max`: a value near
    zero means the triage was close to a hard threshold in screen space.
    """
    )
    return


@app.cell
def _(auxiliary):
    triage = auxiliary.triage_separation()
    triage
    return (triage,)


@app.cell
def _(mo):
    mo.md(
        r"""
    **CYP2D6 is the extreme case and it is not subtle.** Its unlabelled compounds
    have a *positive* median log2fc — no inhibition at all — while every labelled
    compound sits below −0.59. The two groups barely touch. So the CYP2D6 model is
    asked to predict a potency range it was essentially never shown the bottom of,
    which is a structural explanation for why CYP2D6 has resisted every method tried
    in 01, 03 and 04 and why CLAUDE.md has carried it as "the central problem" since
    the baseline.

    **CYP3A4 is the opposite**, and it is the one endpoint that already works. Its
    *unlabelled* compounds are more inhibitory on average than its labelled ones, so
    whatever decided which CYP3A4 compounds got a curve was not this screen alone.

    That contrast is the hypothesis behind A4: if truncated selection is what breaks
    CYP2D6, then correcting for the selection should move CYP2D6 and leave CYP3A4
    roughly alone.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### The public corpus

    The NCGC qHTS cytochrome panel, one assay per isoform, fetched from PubChem
    BioAssay. **ChEMBL was deliberately not used** despite carrying more CYP IC50
    rows in absolute terms (13,887 for CYP3A4 against 13,076 compounds here): those
    rows are pooled over hundreds of unrelated protocols, substrates and labs, so two
    measurements of the same compound routinely disagree by more than a log unit.
    PXR's finding was that heterogeneous auxiliary data hurt — restricting to a single
    protocol is what makes a result here attributable to the transfer rather than to
    the noise.

    Activators are dropped (their AC50 is on the same scale as an inhibitor's and
    means the opposite thing). Inactives are kept, labelled at the assay's weakest
    measurable potency, because dropping them would reproduce exactly the truncation
    problem the challenge labels already have.
    """
    )
    return


@app.cell
def _(EXTERNAL_SNAPSHOT, external):
    public_summary = external.summarise(EXTERNAL_SNAPSHOT)
    public_summary
    return (public_summary,)


@app.cell
def _(C, EXTERNAL_SNAPSHOT, data, external, mo, pl):
    _challenge_smiles = data.load_train_inhibition()["SMILES"].to_list()
    _rows = []
    for _iso in C.PUBCHEM_AIDS:
        _frame = external.pretraining_frame(_iso, snapshot=EXTERNAL_SNAPSHOT)
        _rows.append(
            {"isoform": _iso, **external.overlap_with_challenge(_frame, _challenge_smiles)}
        )
    public_overlap = pl.DataFrame(_rows)

    mo.output.append(public_overlap)
    mo.output.append(
        mo.md(
            f"""
    Structural overlap with the challenge set is small — **{public_overlap['n_shared'].min()}
    to {public_overlap['n_shared'].max()}** compounds per isoform against
    {public_overlap['n_public'].min():,}–{public_overlap['n_public'].max():,} public
    ones. Small, but removed anyway before pretraining: a public measurement of a
    challenge compound is a different lab's read on the same molecule, and leaving it
    in would let a fold's test label reach the encoder before fine-tuning starts.
    """
        )
    )
    return (public_overlap,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Side section — Emax, and the quantity it was supposed to be

    Emax is the other unused file, and it is worth a short section rather than an arm.
    The verdict is recomputed here rather than asserted, since the dataset has been
    amended before.
    """
    )
    return


@app.cell
def _(auxiliary):
    emax = auxiliary.emax_summary()
    emax
    return (emax,)


@app.cell
def _(emax, mo):
    mo.md(
        f"""
    **Emax is saturated.** Every value sits within ~0.1 of −1.0, with standard
    deviations between {emax['std'].min():.3f} and {emax['std'].max():.3f}, and
    correlations with pIC50 spanning {emax['corr_with_pic50'].min():.2f} to
    {emax['corr_with_pic50'].max():.2f}. Only CYP2D6 shows anything non-trivial. It
    is a readout that pinned at its ceiling, and there is no arm worth spending a
    chemprop run on here.

    The quantity Emax was *expected* to provide — how much a compound's behaviour
    changes between assay conditions — is already available, and in a much better
    form: the **difference between pIC50 in the TDI (NADPH-preincubation) condition
    and the direct condition**. Both columns are in `cyp-challenge-TRAIN_TDI.csv`,
    which notebook 04 loaded without ever computing their difference.
    """
    )
    return


@app.cell
def _(auxiliary):
    tdi_shift = auxiliary.tdi_shift_summary()
    tdi_shift
    return (tdi_shift,)


@app.cell
def _(mo, pl, tdi_shift):
    _d6 = tdi_shift.filter(pl.col("isoform") == "CYP2D6").row(0, named=True)
    _a4 = tdi_shift.filter(pl.col("isoform") == "CYP3A4").row(0, named=True)
    mo.md(
        f"""
    **The shift alone separates the TDI classes almost perfectly.** Used as a raw
    score with no model at all, it reaches **AUC {_d6['auc_shift_alone']:.3f}** on
    CYP2D6 and **{_a4['auc_shift_alone']:.3f}** on CYP3A4. Mechanistically this is
    exactly right: a time-dependent inhibitor becomes more potent after preincubation,
    so a positive shift *is* the signature of the thing `is_TDI` labels.

    ### What this does and does not mean

    It is **not** a feature. Neither pIC50 column exists for the blinded test set, so
    a model reading the shift would have nothing to read at prediction time — the same
    constraint the screen data carries.

    What it is, is a **ceiling**. Notebook 04's best macro MCC was 0.248
    (`tabicl_singletask` at threshold 0.25), reached by predicting the TDI label
    directly from structure. This says that a two-stage model — predict the shift as a
    *regression* target from structure, then threshold the predicted shift — is
    attacking a target that is nearly a deterministic function of the label. The
    entire difficulty collapses into how well the shift itself can be predicted.

    That reframes the TDI track from a hard classification problem into a regression
    problem with a trivial decision rule attached, and it is the most promising
    unexplored direction for a future notebook. It is flagged rather than run here
    because this notebook's arms are all on the regression track, and a TDI
    architecture change deserves its own paired comparison against 04's result rather
    than a section at the end of someone else's run.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## The arms

    Every arm is one Chemprop D-MPNN with four masked output heads — architecturally
    the 03 winner — differing only in what data it is allowed to learn from. Holding
    the architecture fixed is what makes a difference attributable to the *data*
    rather than to the model.

    | arm | training rows | extra targets | encoder init | weights |
    |:--|:--|:--|:--|:--|
    | `multitask` (reference) | real labels | 4 pIC50 | random | uniform |
    | `aux_screen` | real labels | 4 pIC50 + 4 log2fc | random | uniform |
    | `augmented` | real + censored | 4 pIC50 | random | uniform |
    | `pubchem` | real labels | 4 pIC50 | **PubChem-pretrained** | uniform |
    | `pubchem_frozen` | real labels | 4 pIC50 | PubChem-pretrained | uniform, encoder frozen |
    | `selection` | real labels | 4 pIC50 | random | **inverse propensity** |

    Two notes on correctness that are easy to get wrong and are tested in
    `tests/test_auxiliary.py`:

    **A1's auxiliary heads are never scored.** The model predicts eight columns; the
    OOF frame reads four. The screen shapes the encoder and never touches the
    inference path.

    **A2 trains on the weak labels but scores only on real ones.** Scoring them would
    reward the model for reproducing a label this repo invented — and because those
    rows carry a wide credible interval, ST-RAE would score most of them as zero error
    and flatter the arm by construction. The split between the training frame and the
    scoring frame is what prevents that.
    """
    )
    return


@app.cell
def _(C, auxiliary, data, mo, pl):
    ENDPOINTS = list(C.REGRESSION_ENDPOINTS)

    # Real dose-response labels. These drive scoring for every arm without exception.
    frames = {endpoint: data.training_frame(endpoint) for endpoint in ENDPOINTS}

    # Real labels plus censored screen negatives. Drives A2's training only.
    augmented_frames = {
        endpoint: auxiliary.augmented_training_frame(endpoint) for endpoint in ENDPOINTS
    }

    mo.output.append(
        pl.DataFrame(
            {
                "endpoint": ENDPOINTS,
                "real_rows": [frames[e].height for e in ENDPOINTS],
                "augmented_rows": [augmented_frames[e].height for e in ENDPOINTS],
                "weak_added": [
                    augmented_frames[e].height - frames[e].height for e in ENDPOINTS
                ],
            }
        )
    )
    mo.output.append(
        mo.md(
            f"""
    The weak labels are added where the screen says a compound showed no inhibition
    *and* the effect was not significant at FDR 0.05. That double condition is
    conservative on purpose, and the yield varies sharply: the three weak endpoints
    gain {augmented_frames[ENDPOINTS[0]].height - frames[ENDPOINTS[0]].height:,}
    ({ENDPOINTS[0].split('_')[0]}) to
    {augmented_frames[ENDPOINTS[2]].height - frames[ENDPOINTS[2]].height:,}
    ({ENDPOINTS[2].split('_')[0]}) rows, while CYP3A4 — the endpoint that already
    works — gains almost nothing. That asymmetry says this arm is aimed squarely at
    the weak endpoints.

    The censoring point is pIC50 {auxiliary.SCREEN_CENSOR_PIC50:.2f}, the screening
    concentration itself: a compound with no effect at 49.5 µM cannot be more potent
    than that. Checked against the labelled compounds that pass the same filter,
    whose true pIC50 averages 2.5–3.3 — genuinely inactive, which is why the interval
    runs two log units downward rather than hugging the censoring point.
    """
        )
    )
    return ENDPOINTS, augmented_frames, frames


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Folds

    Assigned once, over the union of every compound any arm trains on — including the
    censored rows. This is a correctness requirement, not tidiness: with per-endpoint
    splits a compound could sit in one endpoint's training set and another's test set
    at the same time, and a multi-head model would then be scored on a compound whose
    label it had partly seen. Assigning over the union also makes the arms genuinely
    *paired*, which is what licenses `evaluation.paired_bootstrap` between them.

    The augmented union is used rather than the real one so that a weak row cannot
    land on the opposite side of the split from its own scaffold. As it happens the
    two unions are identical — every screen negative already appears in the
    inhibition table carrying a label for some *other* endpoint — so A2 adds
    endpoint coverage within compounds the model already sees, not new molecules.
    That is worth knowing when reading its result: the arm tests whether filling in
    a compound's missing endpoints helps, not whether more compounds help.
    """
    )
    return


@app.cell
def _(augmented_frames, cv, mo, n_outer, pl):
    _, assignments = cv.shared_scaffold_folds(
        augmented_frames, n_outer=n_outer, n_inner=5, seed=42
    )
    ALL_FOLDS = sorted(assignments["fold"].unique().to_list())

    mo.output.append(
        mo.md(
            f"{assignments.height:,} compounds assigned across **{len(ALL_FOLDS)} folds** "
            f"({n_outer} outer × 5 inner)."
        )
    )
    mo.output.append(assignments.group_by("fold").len().sort("fold").head(5))
    return ALL_FOLDS, assignments


@app.cell
def _(mo):
    mo.md(
        r"""
    ### The pretraining corpus

    Assembled once: all four PubChem assays joined into one wide multi-target frame,
    with challenge structures removed. Column order matches the fine-tuned model's
    target order exactly — chemprop warm-starts the head by position, so a permuted
    order would load CYP3A4's pretrained weights into CYP1A2's head and degrade every
    endpoint without raising anything.
    """
    )
    return


@app.cell
def _(ENDPOINTS, EXTERNAL_SNAPSHOT, aux_training, data, frames, mo, pl, quick):
    _challenge = data.load_train_inhibition()["SMILES"].to_list()
    pretraining = aux_training.public_pretraining_matrix(
        ENDPOINTS, exclude_smiles=_challenge, snapshot=EXTERNAL_SNAPSHOT
    )
    if quick:
        pretraining = pretraining.head(2000)

    mo.output.append(
        pl.DataFrame(
            {
                "endpoint": ENDPOINTS,
                "public_labels": [
                    int(pretraining[e].is_not_null().sum()) for e in ENDPOINTS
                ],
            }
        )
    )
    mo.output.append(
        mo.md(
            f"**{pretraining.height:,} public compounds**, "
            f"{sum(int(pretraining[e].is_not_null().sum()) for e in ENDPOINTS):,} "
            f"labelled compound/isoform pairs — against "
            f"{sum(frames[e].height for e in ENDPOINTS):,} in the challenge's own "
            f"dose-response table."
        )
    )
    return (pretraining,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## The run — fold-major

    One cache file per `(arm, fold)`, with the fold as the **outer** loop. 03 ran
    method-major and that is the ordering that produced its MPS degradation; more
    importantly, a method-major run killed at hour six leaves a few arms complete and
    the rest missing entirely, so no comparison can be made at all. Fold-major leaves
    every arm complete through fold *k* — a smaller but fully usable paired
    comparison, with every arm scored on identical rows.

    Ordering cannot change results, only when they arrive: folds are independent and
    each unit is a complete fit on a predetermined split.

    The PubChem pretrain runs **once**, before the fold loop, and every fold
    fine-tunes from that same checkpoint. That is legitimate here in a way it would
    not be for challenge data — the corpus contains no challenge label, so no fold's
    test measurement can reach the encoder through it. It is also the only affordable
    option: a per-fold pretrain would multiply the most expensive step in the repo
    by 25.
    """
    )
    return


@app.cell
def _(
    ALL_FOLDS,
    CACHE_DIR,
    CACHE_SUFFIX,
    TIMING_LOG,
    assignments,
    augmented_frames,
    aux_training,
    chemprop_epochs,
    frames,
    mo,
    multitask,
    pl,
    pretraining,
    pretrain_epochs,
    time,
    timings,
):
    def _run_unit(arm: str, fold: int) -> pl.DataFrame:
        """One arm, one fold. Every call shares `assignments`, so arms stay paired."""
        common = dict(assignments=assignments, folds=[fold], epochs=chemprop_epochs)

        if arm == "multitask":
            # The 03 winner, re-run here rather than quoted: its 0.7280 came from a
            # different fold assignment (real labels only), so it is not comparable
            # to these arms without refitting on the folds they all share.
            return multitask.run_cv_multitarget(frames, **common)
        if arm == "aux_screen":
            return aux_training.run_cv_auxiliary_heads(
                frames, method_name="aux_screen", **common
            )
        if arm == "augmented":
            return aux_training.run_cv_augmented(
                frames, augmented_frames, method_name="augmented", **common
            )
        if arm == "pubchem":
            return aux_training.run_cv_pretrained(
                frames,
                pretraining,
                method_name="pubchem",
                pretrain_epochs=pretrain_epochs,
                **common,
            )
        if arm == "pubchem_frozen":
            return aux_training.run_cv_pretrained(
                frames,
                pretraining,
                method_name="pubchem_frozen",
                freeze_encoder=True,
                pretrain_epochs=pretrain_epochs,
                **common,
            )
        if arm == "selection":
            return aux_training.run_cv_selection_corrected(
                frames, method_name="selection", **common
            )
        raise ValueError(f"unknown arm {arm!r}")

    ARMS = [
        "multitask",
        "aux_screen",
        "augmented",
        "pubchem",
        "pubchem_frozen",
        "selection",
    ]

    _parts = []
    with mo.status.progress_bar(
        total=len(ARMS) * len(ALL_FOLDS), title="Auxiliary-data arms"
    ) as _bar:
        for _fold in ALL_FOLDS:
            for _arm in ARMS:
                _cache = CACHE_DIR / f"aux_{_arm}_fold{_fold}_{CACHE_SUFFIX}.parquet"
                if _cache.exists():
                    _oof = pl.read_parquet(_cache)
                else:
                    _t0 = time.time()
                    _oof = _run_unit(_arm, _fold)
                    _oof.write_parquet(_cache)
                    timings.record(
                        TIMING_LOG,
                        stage="aux_arms",
                        method=_arm,
                        endpoint=f"fold{_fold}",
                        mode=CACHE_SUFFIX,
                        seconds=time.time() - _t0,
                    )
                _parts.append(_oof)
                _bar.update()

    aux_oof = pl.concat(_parts, how="vertical_relaxed")
    aux_oof.group_by("method").len().sort("method")
    return ARMS, aux_oof


@app.cell
def _(OUT_DIR, aux_oof):
    aux_oof.write_parquet(OUT_DIR / "oof.parquet")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Results — the macro-average, which is what rank depends on

    The leaderboard sorts on a plain arithmetic mean of ST-RAE across all four
    regression endpoints, not on any single one. A model can win on individual
    endpoints and still lose on the metric that decides rank, so the macro table
    comes first and the per-endpoint breakdown after it.
    """
    )
    return


@app.cell
def _(C, aux_oof, evaluation, pl):
    def _macro(frame: pl.DataFrame) -> pl.DataFrame:
        """Macro-averaged ST-RAE per (method, fold)."""
        folds = evaluation.fold_metrics(frame)
        return evaluation.macro_averaged_fold_metrics(
            {
                endpoint: folds.filter(pl.col("endpoint") == endpoint)
                for endpoint in C.REGRESSION_ENDPOINTS
            },
            metric_col="st_rae",
        )

    macro_folds = _macro(aux_oof)
    macro_table = (
        macro_folds.group_by("method")
        .agg(
            pl.col("st_rae").mean().alias("macro_st_rae"),
            pl.col("st_rae").std().alias("std"),
            pl.len().alias("n_folds"),
        )
        .sort("macro_st_rae")
    )
    macro_table
    return macro_folds, macro_table


@app.cell
def _(aux_oof, evaluation, pl):
    per_endpoint = (
        evaluation.fold_metrics(aux_oof)
        .group_by("method", "endpoint")
        .agg(pl.col("st_rae").mean().alias("st_rae"))
        .pivot(values="st_rae", index="method", on="endpoint")
        .sort("method")
    )
    per_endpoint
    return (per_endpoint,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Is any of this separable from the reference?

    A point estimate is not a result. PXR cost five leaderboard places by ranking on
    differences the data could not resolve — three finalists spanned 0.0039 MAE and
    the ordering reversed completely between phases.

    So every arm gets a paired bootstrap against `multitask`, Holm-corrected across
    the family of comparisons. The forest plot beside it carries the same question
    visually: effect size on the x-axis with its confidence interval, colour carrying
    the verdict, one row per arm.
    """
    )
    return


@app.cell
def _(aux_oof, evaluation, pl):
    REFERENCE = "chemprop_multitask"

    _p_values = {}
    _rows = []
    for _arm in sorted(aux_oof["method"].unique().to_list()):
        if _arm == REFERENCE:
            continue
        # paired_bootstrap takes aligned arrays, not frames: join the arm against the
        # reference on (endpoint, fold, compound) so every resample compares two
        # predictions of the same measurement. The join is inner, so a fold missing
        # from one arm silently shrinks n rather than misaligning rows.
        _wide = (
            aux_oof.filter(pl.col("method") == _arm)
            .select(
                "endpoint",
                "fold",
                "Molecule_Name",
                "y_true",
                "y_lower",
                "y_upper",
                pl.col("y_pred").alias("pred_arm"),
            )
            .join(
                aux_oof.filter(pl.col("method") == REFERENCE).select(
                    "endpoint",
                    "fold",
                    "Molecule_Name",
                    pl.col("y_pred").alias("pred_ref"),
                ),
                on=["endpoint", "fold", "Molecule_Name"],
                how="inner",
            )
        )
        # diff is arm minus reference, and ST-RAE is an error metric, so a negative
        # diff means the arm beat the reference.
        _result = evaluation.paired_bootstrap(
            _wide["y_true"].to_numpy(),
            _wide["pred_arm"].to_numpy(),
            _wide["pred_ref"].to_numpy(),
            metric="st_rae",
            y_lower=_wide["y_lower"].to_numpy(),
            y_upper=_wide["y_upper"].to_numpy(),
        )
        _p_values[_arm] = _result["p_value"]
        _rows.append({"arm": _arm, **_result})

    bootstrap_table = pl.DataFrame(_rows).sort("p_value")
    holm = evaluation.holm_bonferroni(_p_values)
    bootstrap_table
    return REFERENCE, bootstrap_table, holm


@app.cell
def _(holm):
    holm
    return


@app.cell
def _(OUT_DIR, REFERENCE, macro_folds, mcs):
    forest = mcs.reference_forest_plot(
        macro_folds,
        metric_col="st_rae",
        higher_is_better=False,
        reference=REFERENCE,
        title="Auxiliary-data arms vs the 03 winner (macro ST-RAE)",
        save_path=OUT_DIR / "forest_macro.png",
    )
    forest
    return (forest,)


@app.cell
def _(OUT_DIR, macro_folds, mcs):
    _title = "Macro-averaged ST-RAE"
    grid = mcs.make_mcs_grid(
        {_title: macro_folds},
        metric_col="st_rae",
        higher_is_better={_title: False},
        save_path=OUT_DIR / "mcs_heatmap_macro.png",
    )
    grid
    return (grid,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Where the effect lands, per endpoint

    The macro number hides direction. A1 and A2 add most of their data to the weak
    endpoints and almost none to CYP3A4, and A4's hypothesis is specifically about
    CYP2D6's truncated selection — so an arm that moves the macro by moving only
    CYP2D6 is a different finding from one that moves everything a little.
    """
    )
    return


@app.cell
def _(OUT_DIR, aux_oof, evaluation, mcs, pl):
    _folds = evaluation.fold_metrics(aux_oof)
    _panels = {
        endpoint.split("_")[0]: _folds.filter(pl.col("endpoint") == endpoint)
        for endpoint in sorted(_folds["endpoint"].unique().to_list())
    }
    endpoint_grid = mcs.make_mcs_grid(
        _panels,
        metric_col="st_rae",
        higher_is_better={title: False for title in _panels},
        save_path=OUT_DIR / "mcs_heatmap_per_endpoint.png",
    )
    endpoint_grid
    return (endpoint_grid,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Regression to the mean

    PXR's dominant failure mode, and no amount of tuning fixed it there. It is
    already visible in this repo's baseline (LightGBM underpredicting potent CYP3A4
    compounds by ~1 log unit). A2 and A4 both add or upweight low-activity
    information, so they are the arms most likely to *worsen* it — a model given
    thousands of weak labels has every incentive to regress harder toward the middle.
    """
    )
    return


@app.cell
def _(aux_oof, evaluation, pl):
    bias = pl.concat(
        [
            evaluation.bias_by_potency_bin(
                aux_oof.filter(pl.col("method") == method)
            ).with_columns(pl.lit(method).alias("method"))
            for method in sorted(aux_oof["method"].unique().to_list())
        ]
    )
    bias
    return (bias,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Calibration

    `01_baseline` found linear calibration helping every endpoint, unlike PXR where
    it looked like noise, and CLAUDE.md's rule is to never ship uncalibrated without
    checking this table. Cross-fit for evaluation; a final calibrator fitted on all
    OOF data is what the submission would use.
    """
    )
    return


@app.cell
def _(aux_oof, calibration, evaluation, pl):
    _rows = []
    for _method in sorted(aux_oof["method"].unique().to_list()):
        _sub = aux_oof.filter(pl.col("method") == _method)
        for _kind in ("none", "linear", "isotonic"):
            _frame = (
                _sub
                if _kind == "none"
                else calibration.crossfit_calibrate(_sub, kind=_kind)
            )
            _macro = (
                evaluation.fold_metrics(_frame)
                .group_by("endpoint")
                .agg(pl.col("st_rae").mean())["st_rae"]
                .mean()
            )
            _rows.append(
                {"method": _method, "calibration": _kind, "macro_st_rae": _macro}
            )

    calibration_table = (
        pl.DataFrame(_rows)
        .pivot(values="macro_st_rae", index="method", on="calibration")
        .sort("method")
    )
    calibration_table
    return (calibration_table,)


@app.cell
def _(
    OUT_DIR,
    bias,
    bootstrap_table,
    calibration_table,
    emax,
    holm,
    macro_table,
    per_endpoint,
    public_summary,
    screen_agreement,
    tdi_shift,
    triage,
):
    # Derived summaries are committed like the figures, so a reader can check a claim
    # in this notebook without re-running a multi-hour sweep.
    for _name, _frame in [
        ("macro_table", macro_table),
        ("per_endpoint", per_endpoint),
        ("bootstrap_vs_reference", bootstrap_table),
        ("holm_correction", holm),
        ("calibration", calibration_table),
        ("bias_by_potency", bias),
        ("screen_agreement", screen_agreement),
        ("triage_separation", triage),
        ("emax_summary", emax),
        ("tdi_shift_summary", tdi_shift),
        ("public_summary", public_summary),
    ]:
        _frame.write_csv(OUT_DIR / f"{_name}.csv")
    return


@app.cell
def _(TIMING_LOG, timings):
    timings.summary(TIMING_LOG)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Reading the result

    *Filled in after the run, from the tables above.*

    The decision this notebook feeds is whether any auxiliary source earns a place in
    a submission. The bar is not "the point estimate improved" — it is the paired
    bootstrap surviving Holm correction, since PXR's five lost places came from
    ranking on differences the CV could not resolve.

    ## Submission

    **Deliberately not written yet.** Following 03 and 04, the submission section is
    added after the numbers exist and the choice is made from them rather than
    written speculatively alongside the run. `03_methods` shipped on its result and
    `04_methods_tdi` declined to, both correctly.

    When it is added it gets `submissions/05_auxiliary_data/<date>/` with a
    `PROVENANCE.md` recording model, calibration, CV protocol, **both** data
    snapshots (challenge and external — this is the first notebook where those are
    two different things), and an expected-performance table computed fresh from CV
    for the exact submitted method and calibration.
    """
    )
    return


if __name__ == "__main__":
    app.run()
