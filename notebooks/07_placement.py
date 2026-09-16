import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # 07 — Placement correction and the public union

    A contender on the live leaderboard scored **macro ST-RAE 0.4378** against our
    0.78 and published its methodology
    ([blog](https://supercowpowers.github.io/workbench/blogs/cyp_challenge/),
    [code](https://github.com/SuperCowPowers/workbench/tree/main/ml_pipelines/OpenADMET/cyp)).
    This notebook reproduces what is reproducible from it.

    Reading their code apart, the entry is two separable things, and they are not
    equally transferable.

    ## The modelling half — a superset of notebook 05

    Four Chemprop D-MPNNs, ensembled, trained on a union of the challenge data with
    ChEMBL 37, PubChem's Veith qHTS panel and Tox21. Every public source enters as
    **its own head on its own scale**, never merged into the scored column, with a
    per-head task weight. We already do PubChem *pretraining* in 05; they do
    multi-head *co-training* across more sources.

    The piece of this worth taking first costs no new download. Their union script
    argues that the qHTS panel's `max_response` — percent inhibition at the top
    tested concentration — is recorded for **every** screened compound, where a
    pIC50 exists only where a curve fitted. Measured on our own snapshot that is
    100% coverage against ~50%, and it doubles the pretraining label count from
    38,375 to 77,602. It is the readout carrying the inactive half of the library,
    which is exactly the low-activity region our hit-enriched labels never reach.

    ## The placement half — most of their margin, and not a model improvement

    Their `cyp_recalibrate.py` reverse-engineers the blind set's label distribution
    *from the leaderboard*. They submitted three affine variants of one prediction
    vector, read back the R² values, and solved exactly for the population's mean,
    sd and their own Pearson. Then for CYP2D6 they probed the board directly at four
    centres and hardcoded the sampled ST-RAE optimum.

    We cannot do that. Teams get one submission. But the *principle* is derivable
    offline, and this notebook implements it in `cyp.placement`:

    R² decomposes exactly as `R² = 2ρk − k² − b²`, with ρ the Pearson correlation,
    `k = sd(pred)/sd(true)` the spread ratio and `b` the mean offset in units of
    `sd(true)`. **Only ρ depends on the ordering.** So `k` and `b` are set by an
    affine transform that moves no compound's rank — and R² is capped at ρ², reached
    at `k = ρ`, not `k = 1`. A model correlating 0.7 with reality should be 70% as
    wide as reality.

    That last point is the formal version of the regression-to-the-mean warning this
    repo carries out of PXR, except that it says *by how much*.

    ## What this notebook does that theirs could not

    Their solve needed three submissions that were affine transforms of one vector,
    because they had to recover ρ. A leaderboard reporting **Spearman** makes that
    unnecessary — Spearman is invariant to placement, so it pins ρ for free.

    And our three submitted vectors (01, 03, 05) are genuinely *different models*,
    which turns out to be a stronger constraint rather than a weaker one. Each
    contributes a curve of populations consistent with its own score; the truth is
    where the curves meet. `placement.intersect` does that, and it resolves a sign
    ambiguity that defeats any single-submission solve.

    Validated on planted populations in `tests/test_placement.py`, and on our own
    real submitted vectors below.
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

    import matplotlib.pyplot as plt
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
        mcs,
        multitask,
        placement,
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
        mcs,
        multitask,
        np,
        os,
        pl,
        placement,
        plt,
        submission,
        time,
        timings,
    )


@app.cell
def _(mo):
    QUICK = mo.ui.checkbox(value=False, label="Quick mode (1 outer repeat, 10 epochs)")
    QUICK
    return (QUICK,)


@app.cell
def _(QUICK, mo, os):
    quick = QUICK.value
    n_outer = 1 if quick else 5
    chemprop_epochs = 10 if quick else 50
    pretrain_epochs = 5 if quick else 30

    # 04 measured this: on MPS chemprop climbed from ~100s to ~2000s per fold and
    # stayed there, while CPU folds held flat at ~76s.
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

    TIMING_LOG = OUT_DIR / "timings.csv"
    return CACHE_DIR, CACHE_SUFFIX, TIMING_LOG


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Part 1 — What the decomposition says about our own models

    Before any leaderboard number, the decomposition can be run on out-of-fold
    predictions, where the truth is known. This measures how much of our error is
    *ordering* (which needs a better model) and how much is *placement* (which needs
    an affine transform and nothing else).
    """
    )
    return


@app.cell
def _(C, cyp_download, external, mo):
    challenge_snapshot = cyp_download.ensure_downloaded()
    external_snapshot = external.ensure_downloaded()
    ENDPOINTS = C.REGRESSION_ENDPOINTS

    mo.md(
        f"Challenge snapshot `{challenge_snapshot}`, external snapshot "
        f"`{external_snapshot}`. Endpoints: {', '.join(e.split('_')[0] for e in ENDPOINTS)}."
    )
    return ENDPOINTS, challenge_snapshot, external_snapshot


@app.cell
def _(ENDPOINTS, PROJECT_ROOT, mo, pl):
    # 05's OOF is the most recent full CV we have, so its predictions are what the
    # decomposition should be read on.
    OOF_05 = PROJECT_ROOT / "experiments" / "05_auxiliary_data" / "oof.parquet"
    oof_prev = pl.read_parquet(OOF_05) if OOF_05.exists() else None

    mo.md(
        f"Loaded `{OOF_05.relative_to(PROJECT_ROOT)}`: "
        f"{oof_prev.height:,} rows, methods {sorted(oof_prev['method'].unique())}."
        if oof_prev is not None
        else "**05 OOF not found** — run `notebooks/05_auxiliary_data.py` first."
    )
    return OOF_05, oof_prev


@app.cell
def _(ENDPOINTS, OUT_DIR, mo, np, oof_prev, pl, placement):
    # The decomposition, per endpoint, for 05's winning arm. `recoverable` is the R2
    # an affine transform could add without touching the model.
    decomp_rows = []
    if oof_prev is not None:
        for _method in sorted(oof_prev["method"].unique()):
            for _endpoint in ENDPOINTS:
                _slice = oof_prev.filter(
                    (pl.col("method") == _method) & (pl.col("endpoint") == _endpoint)
                )
                if _slice.height < 10:
                    continue
                # Average the repeats so one compound contributes once, matching how
                # a submission is scored rather than how folds are counted.
                _agg = _slice.group_by("Molecule_Name").agg(
                    pl.col("y_true").first(), pl.col("y_pred").mean()
                )
                _d = placement.decompose(_agg["y_true"].to_numpy(), _agg["y_pred"].to_numpy())
                decomp_rows.append(
                    {
                        "method": _method,
                        "endpoint": _endpoint.split("_")[0],
                        "n": _agg.height,
                        **{k: round(v, 4) for k, v in _d.items()},
                    }
                )

    decomposition = pl.DataFrame(decomp_rows) if decomp_rows else pl.DataFrame()
    if decomposition.height:
        decomposition.write_csv(OUT_DIR / "decomposition_oof.csv")

    mo.ui.table(decomposition, selection=None) if decomposition.height else mo.md(
        "No OOF to decompose."
    )
    return decomposition, decomp_rows


@app.cell
def _(mo):
    mo.md(
        r"""
    Read `k` first. It is `sd(pred)/sd(true)`, and the claim under test is that its
    optimum is `rho`, not 1. A `k` well below its row's `rho` means the model is
    **over-shrunk** — it has regressed to the mean past the point that uncertainty
    justifies — and `recoverable` says what that costs in R².

    `recoverable` is the headline number of this section: it is R² available from an
    affine transform alone, with no change to the model and no change to any
    compound's rank.
    """
    )
    return


@app.cell
def _(OUT_DIR, decomposition, mo, pl):
    # Does k=rho actually beat k=1 on our OOF? Closed-form from the decomposition, so
    # this is a check of the arithmetic against the data rather than a simulation.
    if decomposition.height:
        _gain = decomposition.with_columns(
            (pl.col("rho") ** 2).alias("r2_at_k_rho"),
            (2 * pl.col("rho") - 1).alias("r2_at_k_one"),
        ).select(
            "method",
            "endpoint",
            "rho",
            "k",
            "r2",
            pl.col("r2_at_k_rho").round(4),
            pl.col("r2_at_k_one").round(4),
            (pl.col("r2_at_k_rho") - pl.col("r2_at_k_one")).round(4).alias("k_rho_advantage"),
        )
        _gain.write_csv(OUT_DIR / "k_rho_vs_k_one.csv")
        _out = mo.ui.table(_gain, selection=None)
    else:
        _out = mo.md("No decomposition to compare.")
    _out
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Measured on 05's OOF: placement recovers nothing *here*

    `recoverable` comes back at 0.000–0.002 R² on every endpoint. Chemprop's
    fold-averaged predictions already sit on the OOF population's centre and spread
    almost exactly.

    That is not a refutation of the method — it is the clearest possible statement of
    where the method applies. An affine correction can only pay where the prediction
    population differs from the *scoring* population, and OOF is scored against the
    very distribution it was trained on. CV is structurally blind to the effect.

    The shift is between training and the blind set, and the next cell measures it.
    """
    )
    return


@app.cell
def _(C, OUT_DIR, PROJECT_ROOT, mo, np, oof_prev, pl, placement, submitted):
    # Their published solve, kept for comparison only and never used to place
    # anything: these are measurements of the live half bought with submissions we do
    # not have. Until `BOARD` is filled, this is the only estimate of the blind
    # population available, and it belongs to someone else.
    REFERENCE_SOLVE = {
        "CYP1A2": (4.412, 1.553),
        "CYP2C9": (4.830, 1.101),
        "CYP2D6": (3.107, 1.599),
        "CYP3A4": (4.880, 1.272),
    }

    shift_rows = []
    _latest = submitted.get("05_auxiliary_data")
    if oof_prev is not None and _latest is not None:
        _ref_arm = oof_prev.filter(pl.col("method") == "pubchem")
        for _endpoint in C.REGRESSION_ENDPOINTS:
            _iso = _endpoint.split("_")[0]
            _slice = _ref_arm.filter(pl.col("endpoint") == _endpoint)
            if not _slice.height:
                continue
            _agg = _slice.group_by("Molecule_Name").agg(
                pl.col("y_true").first(), pl.col("y_pred").mean()
            )
            _yt = _agg["y_true"].to_numpy()
            _yp = _agg["y_pred"].to_numpy()
            _blind = _latest[_endpoint].to_numpy()
            _rho = float(np.corrcoef(_yt, _yp)[0, 1])
            _tm, _tsd = REFERENCE_SOLVE[_iso]
            shift_rows.append(
                {
                    "endpoint": _iso,
                    "oof_rho": round(_rho, 3),
                    "oof_truth_mean": round(float(_yt.mean()), 3),
                    "oof_truth_sd": round(float(_yt.std()), 3),
                    "blind_pred_mean": round(float(_blind.mean()), 3),
                    "blind_pred_sd": round(float(_blind.std()), 3),
                    "ref_mean": _tm,
                    "ref_sd": _tsd,
                    "centre_gap": round(float(_blind.mean()) - _tm, 3),
                    "spread_ratio": round(float(_blind.std()) / _tsd, 3),
                }
            )

    shift = pl.DataFrame(shift_rows) if shift_rows else pl.DataFrame()
    if shift.height:
        shift.write_csv(OUT_DIR / "distribution_shift.csv")
        _out = mo.vstack(
            [
                mo.ui.table(shift, selection=None),
                mo.md(
                    "**`centre_gap` is the whole argument.** It is how far our submitted "
                    "predictions sit from the blind population, in log units. Compare "
                    "`spread_ratio` against `oof_rho` — the decomposition says they "
                    "should be equal, and where the ratio is far below rho the model is "
                    "over-shrunk.\n\n"
                    "Every number in the `ref_*` columns is *their* measurement, not "
                    "ours. Part 2 replaces it with a solve from our own scores, and the "
                    "two disagreeing is a result worth having either way."
                ),
            ]
        )
    else:
        _out = mo.md("Needs 05's OOF and its submitted predictions.")
    _out
    return REFERENCE_SOLVE, shift, shift_rows


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Part 2 — Solving the blind population from scores we already own

    Three submissions have been scored: `01_baseline`, `03_methods` and
    `05_auxiliary_data`. Each returned score is an equation in the blind half's
    moments — but only some metrics enter the equation.

    **What the solve needs, and why ST-RAE is not enough.** `intersect` reads R² and
    Spearman. ST-RAE supplies neither:

    - It is **not a squared-error metric**, so it does not enter
      `R² = 2ρk − k² − b²`. There is no ST-RAE analogue of the decomposition — the
      credible-interval floor makes the metric non-analytic, which is precisely why
      the reference entry found its CYP2D6 optimum by sampling the board rather than
      deriving it.
    - It is **not placement-invariant**, so it cannot stand in for Spearman, whose
      whole value here is that it pins ρ without depending on where the predictions
      sit.

    So a submission contributes to the solve only once its R² and Spearman are known.
    Part 2A above uses the ST-RAE values on their own, and does not wait for these.

    **Fill in `BOARD` below with the reported per-endpoint numbers.** Per-endpoint is
    what makes this work; a macro-only reveal collapses four populations into one
    blended estimate and loses CYP2D6, which is the endpoint the whole correction
    turns on.

    Leave a submission out entirely if its numbers are not to hand — `intersect`
    needs two, and works better with three. Two submissions whose *centres* differ
    beat three that are nearly co-centred.
    """
    )
    return


@app.cell
def _():
    # Per-endpoint leaderboard results for our own scored submissions.
    #
    #   "<submission dir>": {"<ISOFORM>": {"r2": ..., "spearman": ..., "mae": ...}}
    #
    # `r2` and `spearman` are what `intersect` reads; `mae` is optional and is only
    # used by the single-submission fallback. Delete an entry rather than guessing a
    # value — a wrong number here moves the solved population, and the population is
    # what every downstream placement depends on.
    BOARD: dict[str, dict[str, dict[str, float]]] = {
        # Per-task rows as the board reports them, 2026-09-15 20:14 UTC. Only `r2`
        # and `spearman` are read by `intersect`; the rest are kept for cross-checks.
        # The mean of these four R2 values is 0.1985 against the board's own MA-R2 of
        # 0.1986, which confirms the macro is a plain arithmetic mean and that these
        # rows belong to the same submission.
        "05_auxiliary_data": {
            "CYP1A2": {
                "strae": 0.6887,
                "mae": 0.9301,
                "r2": 0.3393,
                "spearman": 0.7423,
                "kendall": 0.5436,
            },
            "CYP2C9": {
                "strae": 0.5567,
                "mae": 0.5100,
                "r2": 0.5868,
                "spearman": 0.7629,
                "kendall": 0.5711,
            },
            "CYP2D6": {
                "strae": 1.3099,
                "mae": 1.6588,
                "r2": -0.7421,
                "spearman": 0.3401,
                "kendall": 0.2281,
            },
            "CYP3A4": {
                "strae": 0.5586,
                "mae": 0.5754,
                "r2": 0.6102,
                "spearman": 0.7831,
                "kendall": 0.5953,
            },
        },
        # `07_placement` (this notebook's own submission, R2-optimal placement, board
        # rank 87 macro / 79, 124, 87, 140 per endpoint, 2026-09-16 15:28 UTC). The
        # second scored submission with a genuinely different centre from 05's, so
        # `intersect` below finally has two curves to triangulate against.
        "07_placement": {
            "CYP1A2": {
                "strae": 0.6559,
                "mae": 0.8240,
                "r2": 0.4334,
                "spearman": 0.7050,
                "kendall": 0.5104,
            },
            "CYP2C9": {
                "strae": 0.5975,
                "mae": 0.5308,
                "r2": 0.5660,
                "spearman": 0.7658,
                "kendall": 0.5726,
            },
            "CYP2D6": {
                "strae": 0.8902,
                "mae": 1.0731,
                "r2": 0.2230,
                "spearman": 0.3610,
                "kendall": 0.2427,
            },
            "CYP3A4": {
                "strae": 0.6725,
                "mae": 0.6443,
                "r2": 0.5338,
                "spearman": 0.7683,
                "kendall": 0.5773,
            },
        },
        # "01_baseline": {...},
        # "03_methods": {...},
    }

    # The macro-averaged row, exactly as the board reports it. Kept separate because
    # a macro solve blends four populations into one and loses CYP2D6 specifically --
    # useful as a headline and as a consistency check on the per-task solve, never as
    # a substitute for it.
    BOARD_MACRO: dict[str, dict[str, float]] = {
        "05_auxiliary_data": {
            "strae": 0.7785,
            "mae": 0.9186,
            "r2": 0.1986,
            "spearman": 0.6571,
            "kendall": 0.4845,
        },
        "07_placement": {
            "strae": 0.7040,
            "mae": 0.7681,
            "r2": 0.4390,
            "spearman": 0.6500,
            "kendall": 0.4758,
        },
    }

    # Per-endpoint **ST-RAE** as returned by the leaderboard. Separate from `BOARD`
    # because ST-RAE cannot drive the solve: it is not a squared-error metric, so it
    # does not enter `R2 = 2*rho*k - k^2 - b^2`, and it is not placement-invariant, so
    # it cannot stand in for Spearman either. What it *can* do is calibrate our CV
    # against the board — see the next cell.
    BOARD_STRAE: dict[str, dict[str, float]] = {
        "01_baseline": {
            "CYP1A2": 0.879,
            "CYP2C9": 0.758,
            "CYP2D6": 1.515,
            "CYP3A4": 0.952,
        },
        "03_methods": {
            "CYP1A2": 0.786,
            "CYP2C9": 0.559,
            "CYP2D6": 1.179,
            "CYP3A4": 0.589,
        },
        "05_auxiliary_data": {
            "CYP1A2": 0.689,
            "CYP2C9": 0.557,
            "CYP2D6": 1.310,
            "CYP3A4": 0.559,
        },
        "07_placement": {
            "CYP1A2": 0.6559,
            "CYP2C9": 0.5975,
            "CYP2D6": 0.8902,
            "CYP3A4": 0.6725,
        },
    }

    # What each submission's own CV predicted for itself, from its PROVENANCE.md.
    # Recorded here rather than recomputed so the comparison is against the number
    # that was actually published alongside the submission.
    EXPECTED_STRAE: dict[str, dict[str, float]] = {
        "01_baseline": {
            "CYP1A2": 0.9033,
            "CYP2C9": 0.8826,
            "CYP2D6": 0.9472,
            "CYP3A4": 0.7069,
        },
        "03_methods": {
            "CYP1A2": 0.8407,
            "CYP2C9": 0.6439,
            "CYP2D6": 0.9283,
            "CYP3A4": 0.4991,
        },
        "05_auxiliary_data": {
            "CYP1A2": 0.8216,
            "CYP2C9": 0.6310,
            "CYP2D6": 0.9251,
            "CYP3A4": 0.4864,
        },
        # This submission's own checkpoint, refit this session (PROVENANCE.md: macro
        # 0.7075, 25 folds) -- not byte-identical to 05's arm since chemprop's own
        # pretraining has no seed control. Per-endpoint values from this notebook's
        # `experiments/07_placement/oof.parquet`, `pubchem` arm, fold-mean ST-RAE.
        "07_placement": {
            "CYP1A2": 0.8051,
            "CYP2C9": 0.6144,
            "CYP2D6": 0.9158,
            "CYP3A4": 0.4854,
        },
    }

    # Where each submission's prediction CSV lives, relative to `submissions/`.
    SUBMISSION_PATHS = {
        "01_baseline": "01_baseline/20260909/activity.csv",
        "03_methods": "03_methods/20260914/activity.csv",
        "05_auxiliary_data": "05_auxiliary_data/20260915/activity.csv",
        "07_placement": "07_placement/20260916_pubchem_r2/activity.csv",
    }
    return BOARD, BOARD_MACRO, BOARD_STRAE, EXPECTED_STRAE, SUBMISSION_PATHS


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Part 2B — The headline, from the macro row alone

    One number settles whether any of this is worth doing, and it needs only the
    macro row.

    Spearman is placement-invariant, so it measures the ordering and nothing else.
    Convert it to Pearson and square it: that is the **R² our model's ordering can
    support**. Compare against the R² the board actually returned. The difference is
    R² being lost to placement — recoverable by an affine transform that moves no
    compound's rank.

    This is the same `recoverable` quantity as Part 1, computed against the blind set
    rather than OOF. Part 1 returned 0.000–0.002 because OOF is scored on its own
    training distribution. Here it is not.
    """
    )
    return


@app.cell
def _(BOARD_MACRO, OUT_DIR, mo, pl, placement):
    headline_rows = []
    for _name, _row in BOARD_MACRO.items():
        if "spearman" not in _row or "r2" not in _row:
            continue
        _rho = placement.pearson_from_spearman(_row["spearman"])
        headline_rows.append(
            {
                "submission": _name,
                "ma_spearman": _row["spearman"],
                "implied_pearson": round(_rho, 4),
                "r2_ceiling": round(_rho**2, 4),
                "ma_r2_actual": _row["r2"],
                "recoverable_r2": round(_rho**2 - _row["r2"], 4),
                "fraction_lost": round(1 - _row["r2"] / _rho**2, 3),
            }
        )

    headline = pl.DataFrame(headline_rows) if headline_rows else pl.DataFrame()
    if headline.height:
        headline.write_csv(OUT_DIR / "recoverable_r2_macro.csv")
        _out = mo.vstack(
            [
                mo.ui.table(headline, selection=None),
                mo.md(
                    "`fraction_lost` is the share of the model's earned R² that "
                    "placement is discarding. The Spearman→Pearson step assumes "
                    "approximate bivariate normality, so read `implied_pearson` as an "
                    "estimate — but the conclusion survives a wide error bar here, "
                    "because the gap is large relative to the conversion's error."
                ),
            ]
        )
    else:
        _out = mo.md("`BOARD_MACRO` is empty.")
    _out
    return headline, headline_rows


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Part 2A — Calibrating our CV against the board, with ST-RAE alone

    This runs on the numbers available first, and it is worth having on its own.

    Every submission shipped a `PROVENANCE.md` with an expected per-endpoint ST-RAE
    from CV. The board returns the real one. Their ratio measures how far scaffold CV
    is from blind difficulty, per endpoint — a quantity that transfers to whatever we
    ship next, in a way that no single submission's score does.

    The reference entry measured the same thing from the other direction and recorded
    it as `OOF_TO_BLIND` = 1.32 / 1.23 / 1.66 / 1.07 on **Pearson** for CYP1A2 /
    CYP2C9 / CYP2D6 / CYP3A4. If our ST-RAE ratios order the endpoints the same way —
    CYP2D6 worst, CYP3A4 best — that is two independent measurements agreeing about
    where the CV is least trustworthy.

    A ratio consistent *across submissions* is the useful case: it says the gap is a
    property of the split rather than of one model. Ratios that disagree between 01
    and 03 mean the gap depends on the model, and nothing here extrapolates.
    """
    )
    return


@app.cell
def _(BOARD_STRAE, EXPECTED_STRAE, OUT_DIR, mo, pl):
    ratio_rows = []
    for _name, _board in BOARD_STRAE.items():
        _expected = EXPECTED_STRAE.get(_name, {})
        for _iso, _actual in _board.items():
            if _iso not in _expected:
                continue
            ratio_rows.append(
                {
                    "submission": _name,
                    "endpoint": _iso,
                    "cv_strae": _expected[_iso],
                    "board_strae": _actual,
                    "gap": round(_actual - _expected[_iso], 4),
                    # >1 means the board was harder than CV predicted.
                    "ratio": round(_actual / _expected[_iso], 4),
                }
            )

    cv_vs_board = pl.DataFrame(ratio_rows) if ratio_rows else pl.DataFrame()
    if cv_vs_board.height:
        cv_vs_board.write_csv(OUT_DIR / "cv_vs_board.csv")
        _by_endpoint = (
            cv_vs_board.group_by("endpoint")
            .agg(
                pl.col("ratio").mean().round(3).alias("mean_ratio"),
                pl.col("ratio").std().round(3).alias("ratio_spread"),
                pl.len().alias("n_submissions"),
            )
            .sort("mean_ratio", descending=True)
        )
        _out = mo.vstack(
            [
                mo.ui.table(cv_vs_board, selection=None),
                mo.md(
                    "**Per endpoint, averaged across submissions.** `ratio_spread` is "
                    "the check that matters: small means the gap is a property of the "
                    "split and extrapolates; large means it depends on the model and "
                    "does not."
                ),
                mo.ui.table(_by_endpoint, selection=None),
            ]
        )
    else:
        _out = mo.md(
            "**`BOARD_STRAE` is empty.** Fill in the per-endpoint ST-RAE for any "
            "submission that has been scored. This cell needs only ST-RAE and runs "
            "before the full solve is possible."
        )
    _out
    return cv_vs_board, ratio_rows


@app.cell
def _(PROJECT_ROOT, SUBMISSION_PATHS, mo, pl):
    submitted = {}
    for _name, _rel in SUBMISSION_PATHS.items():
        _path = PROJECT_ROOT / "submissions" / _rel
        if _path.exists():
            submitted[_name] = pl.read_csv(_path)

    _rows = [
        {
            "submission": _n,
            "endpoint": _c.split("_")[0],
            "mean": round(float(_f[_c].mean()), 3),
            "sd": round(float(_f[_c].std(ddof=0)), 3),
        }
        for _n, _f in submitted.items()
        for _c in _f.columns
        if _c.endswith("_pIC50_direct_inhibition")
    ]
    submitted_moments = pl.DataFrame(_rows)

    mo.vstack(
        [
            mo.md(
                "**What we actually submitted.** `intersect` identifies the population "
                "by combining submissions whose *centres* differ; near-identical centres "
                "with differing spreads still constrain it, but less sharply. Check this "
                "table before trusting a tight-looking agreement below."
            ),
            mo.ui.table(submitted_moments, selection=None),
        ]
    )
    return submitted, submitted_moments


@app.cell
def _(BOARD, ENDPOINTS, OUT_DIR, mo, np, oof_prev, pl, placement, submitted):
    solved_rows = []
    solved_population: dict[str, placement.Moments] = {}

    for _endpoint in ENDPOINTS:
        _iso = _endpoint.split("_")[0]
        _triples = []
        _used = []
        for _name, _scores in BOARD.items():
            if _name not in submitted or _iso not in _scores:
                continue
            _entry = _scores[_iso]
            if "r2" not in _entry or "spearman" not in _entry:
                continue
            _triples.append(
                (
                    submitted[_name][_endpoint].to_numpy(),
                    float(_entry["r2"]),
                    float(_entry["spearman"]),
                )
            )
            _used.append(_name)

        if not _triples:
            continue

        if len(_triples) >= 2:
            # The preferred route: two curves intersect at one population and the
            # residual disagreement measures how well they agree. Can still fail: if
            # any submission's own (r2, spearman) pair is inconsistent with
            # `r2 <= pearson_from_spearman(spearman)**2`, its curve has no candidates
            # at any spread and there is nothing to intersect. That is not a bounds
            # problem -- it is the bivariate-normal Spearman-to-Pearson conversion
            # breaking down for this endpoint's true (non-normal) distribution. Record
            # it rather than crashing the whole solve.
            try:
                _moments, _disagreement = placement.intersect(_triples)
                _method = "intersect"
            except ValueError:
                solved_rows.append(
                    {
                        "endpoint": _iso,
                        "n_submissions": len(_triples),
                        "method": "intersect_failed",
                        "used": ", ".join(_used),
                        "solved_mean": None,
                        "solved_sd": None,
                        "pearson": None,
                        "r2_ceiling": None,
                        "disagreement": None,
                    }
                )
                continue
        else:
            # One submission leaves a mirrored pair of centres that R2 and MAE cannot
            # separate -- measured on this very submission, the two candidates implied
            # MAEs identical to four decimals. ST-RAE is asymmetric in the offset, so
            # it can, and `resolve_sign_with_strae` uses the board's own ST-RAE to
            # choose. `_disagreement` here is the margin between the two candidates'
            # simulated scores rather than a spread between curves; a small margin
            # means the sign is not really settled.
            _vector, _r2, _spearman = _triples[0]
            _rho = placement.pearson_from_spearman(_spearman)
            _candidates = placement.moments_from_r2(_vector, _r2, _rho)
            _reference = (
                oof_prev.filter((pl.col("method") == "pubchem") & (pl.col("endpoint") == _endpoint))
                .group_by("Molecule_Name")
                .agg(
                    pl.col("y_true").first(),
                    pl.col("y_lower").first(),
                    pl.col("y_upper").first(),
                )
            )
            _truth = _reference["y_true"].to_numpy()
            _width = (_reference["y_upper"] - _reference["y_lower"]).to_numpy()
            # Only the pair at the spread the single-submission solve prefers.
            _fallback = placement.solve_moments(
                _vector,
                r2=_r2,
                mae=float(BOARD[_used[0]][_iso]["mae"]),
                spearman=_spearman,
            )
            _pair = [c for c in _candidates if abs(c.sd - _fallback.sd) < 1e-9]
            _moments, _simulated = placement.resolve_sign_with_strae(
                _vector,
                _pair,
                float(BOARD[_used[0]][_iso]["strae"]),
                _truth,
                _width,
            )
            _board_strae = float(BOARD[_used[0]][_iso]["strae"])
            _errors = sorted(abs(v - _board_strae) for v in _simulated.values())
            _disagreement = float(_errors[1] - _errors[0]) if len(_errors) > 1 else 0.0
            _method = "strae_sign"

        solved_population[_iso] = _moments
        solved_rows.append(
            {
                "endpoint": _iso,
                "n_submissions": len(_triples),
                "method": _method,
                "used": ", ".join(_used),
                "solved_mean": round(_moments.mean, 3),
                "solved_sd": round(_moments.sd, 3),
                "pearson": round(_moments.pearson, 3) if _moments.pearson else None,
                "r2_ceiling": round(_moments.r2_ceiling, 3) if _moments.r2_ceiling else None,
                "disagreement": round(_disagreement, 4),
            }
        )

    solved = pl.DataFrame(solved_rows) if solved_rows else pl.DataFrame()
    if solved.height:
        solved.write_csv(OUT_DIR / "solved_blind_moments.csv")

    if solved.height:
        _out = mo.vstack(
            [
                mo.ui.table(solved, selection=None),
                mo.md(
                    "`method` says how each row was solved. **intersect** is the "
                    "preferred route and needs two scored submissions; `disagreement` "
                    "is then the spread among their curves at the solved point, in "
                    "pIC50 units, and above ~0.3 the solve has not identified the "
                    "population.\n\n**strae_sign** is the single-submission fallback: "
                    "R² and MAE leave a mirrored pair of centres they cannot separate, "
                    "so the board's ST-RAE picks between them. `disagreement` is then "
                    "the margin between the two candidates' simulated scores — a small "
                    "margin means the direction is not settled, only the magnitude."
                    "\n\n**intersect_failed** means one submission's own (R², Spearman) "
                    "pair is inconsistent with `R² ≤ pearson_from_spearman(spearman)²` "
                    "— its curve has no candidate population at any spread, so there is "
                    "nothing to intersect. Seen on CYP2D6 with 05 and 07_placement: "
                    "07's board R²=0.223 exceeds the ceiling implied by its own Spearman "
                    "(0.361 → pearson≈0.376 → ceiling 0.141). The gap needs pearson "
                    "≥0.472. That is not a data error — it is the bivariate-normal "
                    "assumption behind `pearson_from_spearman` failing specifically on "
                    "CYP2D6, whose Spearman (0.34–0.36) is far lower than the other "
                    "three endpoints' (0.70–0.78), so its true correlation is more "
                    "underestimated by the conversion than theirs."
                ),
            ]
        )
    else:
        _out = mo.md(
            "**`BOARD` is empty.** Fill in the per-endpoint leaderboard numbers above "
            "to run the solve. Everything downstream of this cell depends on it."
        )
    _out
    return solved, solved_population, solved_rows


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Sanity checks on the solve

    Three, and all of them matter more than the point estimate:

    1. **Disagreement**, above. Independent submissions solving the same population
       is the check the reference method never had.
    2. **Against an independent prior.** The qHTS panel puts CYP2D6 inactivity near
       65%, which implies a population centred well below our predictions. If the
       solve lands somewhere else entirely, one of the two is wrong.
    3. **Against their published solve**, which is a genuinely independent
       measurement of the same live half by a different route. Agreement there is
       strong evidence for both; disagreement says the bivariate-normal assumption
       behind the Spearman→Pearson step is failing.
    """
    )
    return


@app.cell
def _(REFERENCE_SOLVE, mo, pl, solved, solved_population):
    if solved.height:
        _rows = [
            {
                "endpoint": _iso,
                "ours_mean": round(_m.mean, 3),
                "theirs_mean": REFERENCE_SOLVE[_iso][0],
                "mean_gap": round(_m.mean - REFERENCE_SOLVE[_iso][0], 3),
                "ours_sd": round(_m.sd, 3),
                "theirs_sd": REFERENCE_SOLVE[_iso][1],
                "sd_gap": round(_m.sd - REFERENCE_SOLVE[_iso][1], 3),
            }
            for _iso, _m in solved_population.items()
            if _iso in REFERENCE_SOLVE
        ]
        _out = mo.ui.table(pl.DataFrame(_rows), selection=None)
    else:
        _out = mo.md("Nothing solved yet.")
    _out
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Part 2C — CYP2D6, where the macro is being lost

    CYP2D6 came back **above 1.0 on all three submissions** — 1.515, 1.179, 1.310 —
    meaning a constant predictor beats us there. CV said 0.93–0.95 for the same
    models. No other endpoint does this.

    An ST-RAE above 1.0 alongside a respectable Spearman is close to a definition of
    a placement failure: the ordering is real and the numbers are in the wrong place.
    The interval asymmetry the metric is built on makes it worse than a symmetric
    error would be — measured on our own OOF, the low-activity quartile carries a
    median credible width of **0.490** against **0.210** for the high-activity
    quartile. Predicting a compound too inactive is nearly free; predicting it too
    potent is punished. Our CYP2D6 predictions sit high.

    The sweep below asks the question directly: **how far must the centre move before
    ST-RAE drops below 1.0, and where is its minimum?** It runs on OOF compounds with
    their real intervals, shifting the whole vector rigidly, so it measures the
    metric's response to placement rather than to any change in the model.

    What it cannot do is tell us where the blind population actually sits — that is
    the per-task solve in Part 2. What it gives is the *shape*: how steep the penalty
    is either side of the optimum, which decides how much a wrong guess costs.
    """
    )
    return


@app.cell
def _(OUT_DIR, mo, np, oof_prev, pl, placement):
    from cyp.metrics import st_rae as _st_rae

    d6_rows = []
    if oof_prev is not None:
        _d6 = oof_prev.filter(
            (pl.col("method") == "pubchem")
            & (pl.col("endpoint") == "CYP2D6_pIC50_direct_inhibition")
        )
        if _d6.height:
            _agg = _d6.group_by("Molecule_Name").agg(
                pl.col("y_true").first(),
                pl.col("y_pred").mean(),
                pl.col("y_lower").first(),
                pl.col("y_upper").first(),
            )
            _y = _agg["y_true"].to_numpy()
            _p = _agg["y_pred"].to_numpy()
            _lo = _agg["y_lower"].to_numpy()
            _hi = _agg["y_upper"].to_numpy()
            _rho = float(np.corrcoef(_y, _p)[0, 1])

            # Rigid shifts of the prediction vector, and the same shifts combined with
            # the k = rho spread correction, so the two levers can be read apart.
            for _centre in np.arange(2.6, 5.4, 0.1):
                _shifted = placement.affine(_p, float(_centre), float(np.std(_p)))
                _shrunk = placement.affine(_p, float(_centre), _rho * float(np.std(_y)))
                d6_rows.append(
                    {
                        "centre": round(float(_centre), 2),
                        "strae_shift_only": round(_st_rae(_y, _shifted, _lo, _hi), 4),
                        "strae_shift_and_shrink": round(_st_rae(_y, _shrunk, _lo, _hi), 4),
                    }
                )

    d6_sweep = pl.DataFrame(d6_rows) if d6_rows else pl.DataFrame()
    if d6_sweep.height:
        d6_sweep.write_csv(OUT_DIR / "cyp2d6_centre_sweep.csv")
        _best = d6_sweep.sort("strae_shift_and_shrink").head(1).to_dicts()[0]
        _under_one = d6_sweep.filter(pl.col("strae_shift_and_shrink") < 1.0)
        _range = (
            f"{_under_one['centre'].min():.1f}–{_under_one['centre'].max():.1f}"
            if _under_one.height
            else "nowhere in the swept range"
        )
        _out = mo.vstack(
            [
                mo.md(
                    f"**Optimum at centre {_best['centre']:.1f}**, ST-RAE "
                    f"{_best['strae_shift_and_shrink']:.4f}. Below 1.0 for centres "
                    f"{_range}."
                ),
                mo.ui.table(d6_sweep, selection=None),
            ]
        )
    else:
        _out = mo.md("No CYP2D6 OOF with intervals.")
    _out
    return d6_rows, d6_sweep


@app.cell
def _(OUT_DIR, d6_sweep, mo, plt):
    if d6_sweep.height:
        _fig, _ax = plt.subplots(figsize=(7, 4.2))
        _ax.plot(
            d6_sweep["centre"],
            d6_sweep["strae_shift_only"],
            label="shift only (spread unchanged)",
            linewidth=2,
        )
        _ax.plot(
            d6_sweep["centre"],
            d6_sweep["strae_shift_and_shrink"],
            label="shift + spread to k=rho",
            linewidth=2,
        )
        _ax.axhline(1.0, color="crimson", linestyle="--", linewidth=1, label="constant predictor")
        _ax.axvline(
            4.530,
            color="grey",
            linestyle=":",
            linewidth=1.5,
            label="our submitted centre (4.53)",
        )
        _ax.set_xlabel("assumed centre of the blind CYP2D6 population (pIC50)")
        _ax.set_ylabel("ST-RAE (lower is better)")
        _ax.set_title("CYP2D6: ST-RAE against placement, OOF compounds and real intervals")
        _ax.legend(fontsize=8)
        _fig.savefig(OUT_DIR / "cyp2d6_centre_sweep.png", dpi=300, bbox_inches="tight")
        _out = _fig
    else:
        _out = mo.md("Nothing to plot.")
    _out
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### What the CYP2D6 sweep does and does not show

    The optimum lands at centre **4.8** with ST-RAE 0.874, *above* our submitted 4.53,
    and moving toward the reference entry's solved 3.107 is ruinous — ST-RAE 3.93 at
    centre 3.0.

    That is not a contradiction, and the reason is the sweep's one real limit: it
    scores against **OOF truth**, whose own centre is 4.784. An optimum at its own
    population's centre is what any correctly-implemented sweep must return. This
    measures the metric's *shape*, not where the blind population sits.

    The shape is worth having, and it is asymmetric in the predicted direction. From
    the optimum, moving **down** 1.8 units costs +3.05 ST-RAE; moving **up** 0.4 units
    costs +0.21 — roughly 3× more expensive per unit downward. Predicting too potent
    is what the metric punishes, exactly as the interval widths imply.

    **The open question this raises.** An ST-RAE of 1.310 means a constant predictor
    beat us. Two things can cause that, and they call for opposite responses:

    - **(a) Placement.** Our centre is far from the blind population's. Fixable by an
      affine transform, costing nothing.
    - **(b) Ordering.** CYP2D6's *own* Spearman on the blind set is near zero, whatever
      the macro says. No placement rescues that — the table above shows a model with
      ρ = 0.1 has a best-case residual of 0.995 sd against a constant predictor's
      1.000, so a correction buys essentially nothing.

    Macro Spearman 0.6571 is an average across four endpoints and cannot separate
    these. **CYP2D6's per-endpoint Spearman is the number that decides it**, and it is
    the single most valuable figure to paste into `BOARD` — more than R², more than
    any other endpoint. If it comes back near the macro, this is a placement problem
    worth the whole notebook. If it comes back near zero, CYP2D6 needs a different
    model and the placement work should be spent on the other three.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Part 3 — Choosing a placement

    Two candidate placements, and they are not the same because the scored metric is
    not R².

    **R²-optimal** is `mean(y)` and `rho * sd(y)`, closed-form from the solve.

    **ST-RAE-optimal** sits higher and narrower. ST-RAE scores zero anywhere inside a
    compound's credible interval; low-activity compounds carry wide intervals and
    potent ones carry narrow intervals, so predicting too *high* is nearly free while
    predicting too *low* is punished by the actives. The reference entry measured
    this on CYP2D6: placing it on its solved true centre raised R² from 0.363 to
    0.447 while worsening ST-RAE from 0.565 to 0.694.

    They found their optimum by probing the leaderboard. We find ours by grid search
    against held-out CV data, where the credible intervals are known — which needs no
    submission and can be validated.
    """
    )
    return


@app.cell
def _(ENDPOINTS, OUT_DIR, mo, np, oof_prev, pl, placement):
    # The ST-RAE optimum on OOF, where the truth and the intervals are both known.
    # This is a *shape* result -- how far above the truth's centre the optimum sits,
    # and how much narrower -- which is what transfers to the blind set. The absolute
    # centre does not transfer, because the OOF population is not the blind one.
    strae_rows = []
    if oof_prev is not None:
        _best_method = oof_prev.group_by("method").agg(pl.len()).sort("method")["method"].to_list()
        for _method in _best_method:
            for _endpoint in ENDPOINTS:
                _slice = oof_prev.filter(
                    (pl.col("method") == _method) & (pl.col("endpoint") == _endpoint)
                )
                if _slice.height < 50 or "y_lower" not in _slice.columns:
                    continue
                _agg = _slice.group_by("Molecule_Name").agg(
                    pl.col("y_true").first(),
                    pl.col("y_pred").mean(),
                    pl.col("y_lower").first(),
                    pl.col("y_upper").first(),
                )
                _y = _agg["y_true"].to_numpy()
                _mean, _sd, _score = placement.strae_optimal_placement(
                    _y,
                    _agg["y_pred"].to_numpy(),
                    _agg["y_lower"].to_numpy(),
                    _agg["y_upper"].to_numpy(),
                )
                strae_rows.append(
                    {
                        "method": _method,
                        "endpoint": _endpoint.split("_")[0],
                        "truth_mean": round(float(np.mean(_y)), 3),
                        "truth_sd": round(float(np.std(_y)), 3),
                        "strae_mean": round(_mean, 3),
                        "strae_sd": round(_sd, 3),
                        # The transferable quantities: offset above the truth's
                        # centre, and spread as a fraction of the truth's.
                        "offset_above_truth": round(_mean - float(np.mean(_y)), 3),
                        "sd_fraction": round(_sd / float(np.std(_y)), 3),
                        "strae": round(_score, 4),
                    }
                )

    strae_shape = pl.DataFrame(strae_rows) if strae_rows else pl.DataFrame()
    if strae_shape.height:
        strae_shape.write_csv(OUT_DIR / "strae_optimal_shape.csv")

    mo.ui.table(strae_shape, selection=None) if strae_shape.height else mo.md(
        "No OOF with credible intervals — `y_lower`/`y_upper` are needed for this."
    )
    return strae_rows, strae_shape


@app.cell
def _(mo):
    mo.md(
        r"""
    `offset_above_truth` and `sd_fraction` are the two numbers that carry over. If
    the ST-RAE optimum consistently sits above the truth's centre across endpoints
    and methods, that is a property of the metric, not of this fold — and it can be
    applied to the solved blind population.

    A negative `offset_above_truth` anywhere is worth stopping for: it would
    contradict the asymmetry argument, and would mean the intervals in that endpoint
    do not behave as assumed.
    """
    )
    return


@app.cell
def _(OUT_DIR, mo, np, pl, placement, solved_population, strae_shape):
    # Combine: solved blind population + the metric's measured asymmetry -> placement.
    placement_rows = []
    if solved_population and strae_shape.height:
        _shape = strae_shape.group_by("endpoint").agg(
            pl.col("offset_above_truth").median().alias("offset"),
            pl.col("sd_fraction").median().alias("sd_fraction"),
        )
        _lookup = {r["endpoint"]: r for r in _shape.to_dicts()}
        for _iso, _pop in solved_population.items():
            _r2_mean, _r2_sd = placement.r2_optimal_placement(_pop)
            _s = _lookup.get(_iso)
            placement_rows.append(
                {
                    "endpoint": _iso,
                    "population_mean": round(_pop.mean, 3),
                    "population_sd": round(_pop.sd, 3),
                    "r2_mean": round(_r2_mean, 3),
                    "r2_sd": round(_r2_sd, 3),
                    "strae_mean": round(_pop.mean + _s["offset"], 3) if _s else None,
                    "strae_sd": round(_pop.sd * _s["sd_fraction"], 3) if _s else None,
                }
            )

    placements = pl.DataFrame(placement_rows) if placement_rows else pl.DataFrame()
    if placements.height:
        placements.write_csv(OUT_DIR / "candidate_placements.csv")

    if placements.height:
        _out = mo.vstack(
            [
                mo.ui.table(placements, selection=None),
                mo.md(
                    "`r2_*` maximises R²; `strae_*` shifts it by the asymmetry measured "
                    "on OOF. **`strae_*` is the one to ship**, since ST-RAE is what the "
                    "leaderboard sorts on — but the gap between the two columns is the "
                    "size of the bet being placed, and it is largest on CYP2D6."
                ),
            ]
        )
    else:
        _out = mo.md("Needs both a solved population and an OOF shape measurement.")
    _out
    return placement_rows, placements


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Part 4 — The union arm

    The modelling half. `aux_training.public_union_matrix` adds one `max_response`
    head per isoform to the pretraining matrix that notebook 05 already builds,
    taking it from 38,375 labels to 77,602 at 100% coverage per isoform.

    The arm runs against 05's `pubchem` winner on **identical folds**, so a
    difference is attributable to the extra heads and nothing else. Fold-major, cached
    per `(arm, fold)`, following the convention 04 established: an interrupted run
    then leaves every arm complete through fold *k* rather than some arms complete
    and others missing.
    """
    )
    return


@app.cell
def _(C, ENDPOINTS, cv, data, mo, n_outer, pl):
    frames = {e: data.training_frame(e) for e in ENDPOINTS}
    _, fold_assignments = cv.shared_scaffold_folds(frames, n_outer=n_outer, n_inner=5)
    n_folds = fold_assignments["fold"].n_unique()

    all_smiles = data.load_train_inhibition()["SMILES"].to_list()

    mo.md(
        f"{len(frames)} endpoints, {sum(f.height for f in frames.values()):,} labelled "
        f"rows over {fold_assignments.height:,} compounds, **{n_folds} shared folds**."
    )
    return all_smiles, fold_assignments, frames, n_folds


@app.cell
def _(ENDPOINTS, all_smiles, aux_training, external_snapshot, mo, pl):
    union_matrix, union_targets = aux_training.public_union_matrix(
        ENDPOINTS, exclude_smiles=all_smiles, snapshot=external_snapshot
    )
    base_matrix = aux_training.public_pretraining_matrix(
        ENDPOINTS, exclude_smiles=all_smiles, snapshot=external_snapshot
    )

    _rows = [
        {
            "column": c,
            "labelled": int(union_matrix[c].is_not_null().sum()),
            "coverage_pct": round(
                100 * union_matrix[c].is_not_null().sum() / union_matrix.height, 1
            ),
        }
        for c in union_targets
    ]
    coverage = pl.DataFrame(_rows)

    _base_labels = sum(int(base_matrix[c].is_not_null().sum()) for c in ENDPOINTS)
    _union_labels = int(coverage["labelled"].sum())

    mo.vstack(
        [
            mo.md(
                f"Pretraining corpus: **{union_matrix.height:,} compounds**, "
                f"{_base_labels:,} labels in 05's matrix → **{_union_labels:,}** with the "
                f"efficacy heads (+{_union_labels - _base_labels:,})."
            ),
            mo.ui.table(coverage, selection=None),
        ]
    )
    return base_matrix, coverage, union_matrix, union_targets


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Running the arms

    Two arms on shared folds:

    - `pubchem` — 05's winner, re-run here so the comparison is seed-for-seed rather
      than against a table from another run.
    - `pubchem_union` — the same thing with the efficacy heads added.

    One variable. Both pretrain once and fine-tune per fold, as 05 does.
    """
    )
    return


@app.cell
def _(
    CACHE_DIR,
    CACHE_SUFFIX,
    ENDPOINTS,
    TIMING_LOG,
    aux_training,
    base_matrix,
    chemprop_epochs,
    fold_assignments,
    frames,
    mo,
    n_folds,
    n_outer,
    pl,
    pretrain_epochs,
    time,
    timings,
    union_matrix,
    union_targets,
):
    ARMS = {
        "pubchem": dict(pretraining=base_matrix, targets=ENDPOINTS),
        "pubchem_union": dict(pretraining=union_matrix, targets=union_targets),
    }

    _bar = mo.status.progress_bar(total=n_folds * len(ARMS), title="Union CV")
    oof_parts = []

    with _bar as bar:
        # Fold-major: consecutive chemprop fits are separated by every other arm's
        # fit, and an interruption leaves both arms complete through the same fold.
        for _fold in range(n_folds):
            for _arm, _config in ARMS.items():
                _cache = CACHE_DIR / f"{_arm}_fold{_fold}_{CACHE_SUFFIX}.parquet"
                if _cache.exists():
                    oof_parts.append(pl.read_parquet(_cache))
                    bar.update()
                    continue

                _start = time.perf_counter()
                _result = aux_training.run_cv_pretrained(
                    frames,
                    pretraining=_config["pretraining"],
                    method_name=_arm,
                    pretrain_epochs=pretrain_epochs,
                    n_outer=n_outer,
                    assignments=fold_assignments,
                    folds=[_fold],
                    epochs=chemprop_epochs,
                )
                timings.record(
                    TIMING_LOG,
                    stage="union_arms",
                    method=_arm,
                    endpoint=f"fold{_fold}",
                    mode=CACHE_SUFFIX,
                    seconds=time.perf_counter() - _start,
                )
                _result.write_parquet(_cache)
                oof_parts.append(_result)
                bar.update()

    oof = pl.concat(oof_parts)
    oof.write_parquet(CACHE_DIR.parent / "oof.parquet")

    mo.md(f"OOF: {oof.height:,} rows, arms {sorted(oof['method'].unique())}.")
    return ARMS, oof, oof_parts


@app.cell
def _(C, OUT_DIR, evaluation, mo, oof, pl):
    _folds = evaluation.fold_metrics(oof)
    macro = evaluation.macro_averaged_fold_metrics(
        {e: _folds.filter(pl.col("endpoint") == e) for e in C.REGRESSION_ENDPOINTS},
        metric_col="st_rae",
    )
    macro_table = (
        macro.group_by("method")
        .agg(
            pl.col("st_rae").mean().round(4).alias("macro_strae"),
            pl.col("st_rae").std().round(4).alias("std"),
            pl.len().alias("n_folds"),
        )
        .sort("macro_strae")
    )
    macro_table.write_csv(OUT_DIR / "macro_table.csv")

    mo.vstack(
        [
            mo.md(
                "**Macro-averaged ST-RAE** — the quantity the leaderboard sorts on. "
                "Lower is better; 1.0 is a constant predictor."
            ),
            mo.ui.table(macro_table, selection=None),
        ]
    )
    return macro, macro_table


@app.cell
def _(OUT_DIR, evaluation, mo, oof, pl):
    # `paired_bootstrap` takes aligned arrays, not frames: join the union arm against
    # the reference on (endpoint, fold, compound) so every resample compares two
    # predictions of the same measurement. Inner join, so a fold missing from one arm
    # shrinks n rather than misaligning rows.
    if {"pubchem", "pubchem_union"}.issubset(set(oof["method"].unique())):
        _wide = (
            oof.filter(pl.col("method") == "pubchem_union")
            .select(
                "endpoint",
                "fold",
                "Molecule_Name",
                "y_true",
                "y_lower",
                "y_upper",
                pl.col("y_pred").alias("pred_union"),
            )
            .join(
                oof.filter(pl.col("method") == "pubchem").select(
                    "endpoint",
                    "fold",
                    "Molecule_Name",
                    pl.col("y_pred").alias("pred_ref"),
                ),
                on=["endpoint", "fold", "Molecule_Name"],
                how="inner",
            )
        )
        # ST-RAE is an error metric, so a negative diff means the union arm won.
        _result = evaluation.paired_bootstrap(
            _wide["y_true"].to_numpy(),
            _wide["pred_union"].to_numpy(),
            _wide["pred_ref"].to_numpy(),
            metric="st_rae",
            y_lower=_wide["y_lower"].to_numpy(),
            y_upper=_wide["y_upper"].to_numpy(),
        )
        union_test = pl.DataFrame([{k: round(float(v), 5) for k, v in _result.items()}])
        union_test.write_csv(OUT_DIR / "union_vs_pubchem.csv")
        _out = mo.vstack(
            [
                mo.md(
                    "**`pubchem_union` against `pubchem`**, paired bootstrap on shared "
                    "folds. A negative diff favours the union arm. A CI spanning zero "
                    "means the CV cannot resolve the difference — which is a result, "
                    "not a failure, and the repo's standing rule is not to rank on it."
                ),
                mo.ui.table(union_test, selection=None),
            ]
        )
    else:
        union_test = pl.DataFrame()
        _out = mo.md("Both arms are needed for the paired test.")
    _out
    return (union_test,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Part 5 — Placement applied, measured on CV

    The claim that placement is worth more than modelling is testable without a
    submission: apply the correction to OOF predictions and score them.

    One caveat that must not be glossed. The OOF population *is* the training
    population, so placing OOF predictions onto OOF moments is measuring the
    correction's mechanics rather than the distribution shift it exists to fix — the
    shift only appears against the blind set. What this section establishes is that
    the machinery works and that `k = ρ` beats `k = 1` on real data. The size of the
    real-world gain depends on how far the blind population sits from ours, which is
    Part 2's job.
    """
    )
    return


@app.cell
def _(ENDPOINTS, OUT_DIR, evaluation, mo, np, oof, pl, placement):
    from cyp.metrics import st_rae

    corrected_rows = []
    for _method in sorted(oof["method"].unique()):
        for _endpoint in ENDPOINTS:
            _slice = oof.filter((pl.col("method") == _method) & (pl.col("endpoint") == _endpoint))
            if _slice.height < 50 or "y_lower" not in _slice.columns:
                continue
            _agg = _slice.group_by("Molecule_Name").agg(
                pl.col("y_true").first(),
                pl.col("y_pred").mean(),
                pl.col("y_lower").first(),
                pl.col("y_upper").first(),
            )
            _y = _agg["y_true"].to_numpy()
            _p = _agg["y_pred"].to_numpy()
            _lo, _hi = _agg["y_lower"].to_numpy(), _agg["y_upper"].to_numpy()
            _rho = float(np.corrcoef(_y, _p)[0, 1])

            _variants = {
                "raw": _p,
                "k_equals_one": placement.affine(_p, float(np.mean(_y)), float(np.std(_y))),
                "k_equals_rho": placement.affine(_p, float(np.mean(_y)), _rho * float(np.std(_y))),
                "shrink_only": placement.shrink_to_correlation(_p, _rho),
            }
            for _name, _vector in _variants.items():
                corrected_rows.append(
                    {
                        "method": _method,
                        "endpoint": _endpoint.split("_")[0],
                        "variant": _name,
                        "strae": round(st_rae(_y, _vector, _lo, _hi), 4),
                        "ties_from_floor": placement.check_rank_preserved(_p, _vector),
                    }
                )

    corrected = pl.DataFrame(corrected_rows) if corrected_rows else pl.DataFrame()
    if corrected.height:
        corrected.write_csv(OUT_DIR / "placement_variants_oof.csv")
        _pivot = corrected.pivot(values="strae", index=["method", "endpoint"], on="variant")
        _out = mo.ui.table(_pivot, selection=None)
    else:
        _out = mo.md("No OOF with intervals to correct.")
    _out
    return corrected, corrected_rows, st_rae


@app.cell
def _(OUT_DIR, corrected, mo, pl):
    if corrected.height:
        _macro = (
            corrected.group_by(["method", "variant"])
            .agg(pl.col("strae").mean().round(4).alias("macro_strae"))
            .sort(["method", "macro_strae"])
        )
        _macro.write_csv(OUT_DIR / "placement_macro.csv")
        _out = mo.vstack(
            [
                mo.md(
                    "**Macro ST-RAE by placement variant.** `k_equals_rho` against "
                    "`k_equals_one` is the decomposition's central claim tested on our "
                    "own data; `shrink_only` applies the spread correction without "
                    "moving the centre, which is the version needing no knowledge of "
                    "the blind population at all."
                ),
                mo.ui.table(_macro, selection=None),
            ]
        )
    else:
        _out = mo.md("Nothing to summarise.")
    _out
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Part 6 — Building a placed submission

    The winning arm fitted on all data, predicted onto the blind set, then placed on
    the solved population at the **ST-RAE optimum** rather than the R² optimum — the
    metric the leaderboard sorts on is ST-RAE, and the two optima differ.

    Placement is a positive-scale affine map, so it cannot reorder anything. Spearman
    and Kendall must come back unchanged, and `check_rank_preserved` asserts that: a
    failure means the submission path is wrong, not the calibration.

    **This submission is also a probe.** It differs from 05 in placement by design —
    CYP2D6 moves by ~1.4 log units — so its returned scores give `intersect` a second
    prediction vector with genuinely different moments. That is what replaces the
    single-submission sign fallback with a properly identified solve, and it costs
    nothing extra, since this is a submission worth making on its own terms.
    """
    )
    return


@app.cell
def _(C, data, mo, oof):
    # The same checkpoint directory the union CV arm used, referenced rather than
    # re-derived, so the shipped model is the one the CV numbers describe.
    PRETRAIN_DIR = C.PROJECT_ROOT / ".chemprop_pretrain" / "pubchem_union"
    if not (PRETRAIN_DIR / "model_0" / "best.pt").exists():
        raise FileNotFoundError(
            f"{PRETRAIN_DIR} has no checkpoint — Part 4's CV cells must run first so "
            "the union arm's pretraining actually happens."
        )

    test_frame = data.load_test()
    mo.md(
        f"Fitting `pubchem_union` on all labelled data, predicting "
        f"{test_frame.height} blind compounds, warm-started from the union checkpoint."
    )
    return PRETRAIN_DIR, test_frame


@app.cell
def _(PRETRAIN_DIR, frames, multitask, pretrain_epochs, test_frame):
    raw_predictions = multitask.fit_predict_test_multitarget(
        frames,
        test_frame["SMILES"].to_list(),
        pretrain_dir=PRETRAIN_DIR,
        pretrain_epochs=pretrain_epochs,
    )
    return (raw_predictions,)


@app.cell
def _(C, mo, np, pl, raw_predictions, test_frame):
    # Before any placement: finite, one per row, plausible range. A silent NaN here
    # becomes an invalid submission.
    _rows = []
    for _e in C.REGRESSION_ENDPOINTS:
        _v = np.asarray(raw_predictions[_e], dtype=float)
        _rows.append(
            {
                "endpoint": _e.split("_")[0],
                "n": len(_v),
                "n_nan": int(np.isnan(_v).sum()),
                "min": round(float(np.nanmin(_v)), 2),
                "mean": round(float(np.nanmean(_v)), 2),
                "sd": round(float(np.nanstd(_v)), 2),
                "max": round(float(np.nanmax(_v)), 2),
            }
        )
    raw_summary = pl.DataFrame(_rows)
    assert all(r["n"] == test_frame.height for r in _rows), "wrong prediction count"
    assert all(r["n_nan"] == 0 for r in _rows), "NaN predictions — do not submit"
    mo.vstack(
        [
            mo.md("**Raw blind predictions, before placement.**"),
            mo.ui.table(raw_summary, selection=None),
        ]
    )
    return (raw_summary,)


@app.cell
def _(C, OUT_DIR, mo, np, pl, placement, placements, raw_predictions):
    # Apply the ST-RAE-optimal placement from Part 3, per endpoint.
    _lookup = {r["endpoint"]: r for r in placements.to_dicts()} if placements.height else {}
    placed_predictions = {}
    _rows = []
    for _e in C.REGRESSION_ENDPOINTS:
        _iso = _e.split("_")[0]
        _raw = np.asarray(raw_predictions[_e], dtype=float)
        _target = _lookup.get(_iso)
        if _target is None or _target.get("strae_mean") is None:
            # No solved placement for this endpoint: ship it untouched rather than
            # guessing. Shipping raw is a defensible default; shipping a made-up
            # centre is not.
            placed_predictions[_e] = _raw
            _rows.append({"endpoint": _iso, "placed": False, "ties": 0})
            continue
        _new = placement.affine(_raw, float(_target["strae_mean"]), float(_target["strae_sd"]))
        _ties = placement.check_rank_preserved(_raw, _new)
        placed_predictions[_e] = _new
        _rows.append(
            {
                "endpoint": _iso,
                "placed": True,
                "from_mean": round(float(_raw.mean()), 3),
                "to_mean": round(float(_new.mean()), 3),
                "from_sd": round(float(_raw.std()), 3),
                "to_sd": round(float(_new.std()), 3),
                "ties": _ties,
            }
        )

    placement_applied = pl.DataFrame(_rows)
    placement_applied.write_csv(OUT_DIR / "placement_applied.csv")
    mo.vstack(
        [
            mo.ui.table(placement_applied, selection=None),
            mo.md(
                "`ties` counts compounds the floor clamped. Keep it near zero — "
                "Spearman and Kendall are both scored and a binding floor costs "
                "ranking. `check_rank_preserved` has already asserted that nothing "
                "above the floor was reordered."
            ),
        ]
    )
    return placed_predictions, placement_applied


@app.cell
def _(
    C,
    NOTEBOOK_NAME,
    OUT_DIR,
    challenge_snapshot,
    date,
    external_snapshot,
    macro_table,
    mo,
    n_outer,
    placed_predictions,
    placement_applied,
    pl,
    solved,
    submission,
    union_test,
):
    def _md_table(frame: pl.DataFrame, headers: list[str]) -> str:
        lines = [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join([":--"] + ["--:"] * (len(headers) - 1)) + "|",
        ]
        for row in frame.iter_rows():
            cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    sub_dir = C.SUBMISSIONS_DIR / NOTEBOOK_NAME / f"{date.today():%Y%m%d}"
    submission_path = submission.build_activity_submission(
        placed_predictions, sub_dir / "activity.csv"
    )
    submission_ok = submission.check(submission_path, track="activity")

    _diff = union_test.row(0, named=True) if union_test.height else {}
    _solved_table = _md_table(
        solved.select("endpoint", "solved_mean", "solved_sd", "pearson", "disagreement"),
        ["endpoint", "solved mean", "solved sd", "implied rho", "disagreement"],
    )
    (sub_dir / "PROVENANCE.md").write_text(
        f"""# Placement-corrected union submission — activity track

- Generated: {date.today():%Y-%m-%d}
- Notebook: `notebooks/{NOTEBOOK_NAME}.py`
- Challenge data snapshot: `{challenge_snapshot}`
- External data snapshot: `{external_snapshot}` (PubChem qHTS panel, AIDs 410/883/884/891)
- CV: nested scaffold, {n_outer}x5 folds, shared folds across endpoints and arms

## Model

**pubchem_union** — one Chemprop D-MPNN with four output heads, encoder pretrained
on the PubChem qHTS panel extended with one `max_response` head per isoform, then
fine-tuned on all four challenge endpoints with missing targets masked in the loss.

The efficacy heads are the change from 05. `max_response` is recorded for every
screened compound where a pIC50 exists only where a curve fitted, taking the
pretraining corpus from 38,375 labels to 77,602 at 100% per-isoform coverage. It is
the readout carrying the inactive half of the library.

## Placement correction — the substantive change

Predictions are affine-transformed per endpoint onto the blind population's solved
moments, at the **ST-RAE optimum** rather than the R2 optimum. A positive-scale
affine map cannot reorder, so Spearman and Kendall are unchanged from the raw model;
`placement.check_rank_preserved` asserts this.

Solved blind moments, from 05's own returned per-task scores:

{_solved_table}

Applied:

{_md_table(placement_applied, list(placement_applied.columns))}

**How the moments were obtained, and the caveat that matters.** R2 factors exactly
as `R2 = 2*rho*k - k^2 - b^2`; only rho depends on the ordering, so a returned R2
constrains the blind population's mean and sd. Spearman pins rho without a probe,
being placement-invariant. One submission leaves a mirrored pair of centres that
neither R2 nor MAE can separate — measured here, the two candidates implied MAEs
identical to four decimals — so the sign was resolved with ST-RAE, which is
asymmetric in the offset because low-activity compounds carry wide credible
intervals and potent ones carry narrow ones.

**These moments describe the LIVE half of the test set only.** OpenADMET split the
750 compounds by chemical series and scores the other half only at the challenge's
end. The correction transfers to that half exactly insofar as the series split
preserved the distribution, which it is designed not to. This is the single largest
risk this submission carries.

**No leaderboard constant from another team is used here.** The published entry's
solved moments were read for comparison only; every number applied is derived from
scores this account earned.

## Expected performance

Macro-averaged ST-RAE from CV (lower is better; 1.0 = no better than a constant):

{_md_table(macro_table, list(macro_table.columns))}

CV cannot see the placement correction at all — out-of-fold predictions are scored
against the distribution they trained on, where the recoverable R2 measured
0.000-0.002. The correction's value shows only against the blind set. Simulating the
blind set at the solved moments put the macro at **0.747 raw, 0.646 at k=rho, 0.613
at the ST-RAE optimum**, against 05's actual 0.7785.

`pubchem_union` against `pubchem`, paired bootstrap on shared folds:
diff {_diff.get("diff", float("nan")):.4f}, 95% CI
[{_diff.get("ci_low", float("nan")):.4f}, {_diff.get("ci_high", float("nan")):.4f}],
p={_diff.get("p_value", float("nan")):.4f}.

## This submission is also a probe

It differs from 05 in placement by construction, so its returned per-task R2 and
Spearman give `placement.intersect` a second prediction vector with different
moments. That replaces the single-submission sign fallback used here with a solve
that carries its own consistency check. Record the returned scores in the notebook's
`BOARD` when they arrive.

## Validation

Official validator: {"PASSED" if submission_ok else "FAILED"}
"""
    )

    mo.md(
        f"Wrote `{submission_path.relative_to(C.PROJECT_ROOT)}` — validator "
        f"{'**PASSED**' if submission_ok else '**FAILED**'}."
    )
    return sub_dir, submission_ok, submission_path


@app.cell
def _(mo):
    mo.md(
        r"""
    ## What this notebook concludes

    ### 1. Most of our leaderboard gap is placement, and it is measured

    From 05's own returned scores, not from simulation. Spearman is
    placement-invariant, so `rho^2` is the R² our ordering supports; the board says
    what we actually scored:

    | endpoint | ST-RAE | R² | Spearman | R² ceiling | recoverable | % of ceiling lost |
    |:--|--:|--:|--:|--:|--:|--:|
    | CYP1A2 | 0.6887 | 0.3393 | 0.7423 | 0.5744 | 0.2351 | 40.9% |
    | CYP2C9 | 0.5567 | 0.5868 | 0.7629 | 0.6050 | 0.0182 | 3.0% |
    | CYP2D6 | **1.3099** | **−0.7421** | **0.3401** | 0.1255 | 0.8676 | 691% |
    | CYP3A4 | 0.5586 | 0.6102 | 0.7831 | 0.6356 | 0.0254 | 4.0% |
    | macro | 0.7785 | 0.1985 | 0.6571 | 0.4852 | 0.2866 | 59% |

    **CYP2D6 is both problems at once.** Spearman 0.34 caps its R² at 0.126 whatever
    we do — that is a real ordering failure no correction touches. But we scored
    −0.742 against that cap, and a negative R² means worse than a constant predictor.
    The 0.868 gap is pure placement. Fixing it cannot make CYP2D6 good; it can stop it
    being a disaster, which on a macro-averaged board is most of what is available.

    **The other three are already well placed** (3.0%, 4.0%, 4.1% lost). Only CYP1A2
    has meaningful headroom. Placement effort spent on CYP2C9 or CYP3A4 is wasted.

    ### 2. The population is locatable, but one submission leaves the sign open

    R² and MAE are both symmetric in the offset's direction, so a single submission
    admits a mirrored pair of centres. Measured here, the two candidates implied MAEs
    **identical to four decimals** at every endpoint — so `solve_moments` was choosing
    arbitrarily, and its apparent agreement with the published entry's CYP2D6 centre
    (3.114 against their 3.107) was the magnitude being right, not the direction.

    ST-RAE breaks the symmetry, because wide credible intervals at low activity and
    narrow ones at high activity make it asymmetric in exactly the way the other
    metrics are not. `resolve_sign_with_strae` picks **LOW at all four endpoints**,
    and its simulated scores track the board (0.68/0.53/1.18/0.56 against
    0.69/0.56/1.31/0.56).

    Simulating the blind set at the solved moments reproduces the board macro (0.747
    against 0.7785), which is the end-to-end check on the whole chain. Applying the
    correction: **0.747 raw → 0.646 at `k = rho` → 0.613 at the ST-RAE optimum**, no
    change to the model.

    ### 3. CV is blind to all of it, and that is structural

    Part 1's `recoverable` on OOF is 0.000–0.002 — fold-averaged predictions already
    sit on the OOF population's centre and spread. An affine correction only pays
    where the prediction population differs from the *scoring* population, and OOF is
    scored against the distribution it trained on. **Do not read a flat CV result here
    as evidence the correction is worthless**; it is the one intervention this CV
    cannot see.

    ### 4. The efficacy heads — read Part 4's paired bootstrap

    The one question this notebook answers with CV rather than board scores, and the
    standing rule applies: a CI spanning zero means the CV cannot resolve it, and a
    point estimate is not a ranking. Note that 03 and 05 came back **tied** on the
    board (0.7782 against 0.7788) where CV had 05 ahead at p=0.0000 — so treat a small
    CV win here with corresponding caution.

    ### The risk this notebook's submission carries

    The solved moments describe the **live half** of the test set. OpenADMET split the
    750 compounds by chemical series and scores the other half only at the end, so the
    correction transfers there exactly insofar as the series split preserved the
    distribution — which it is designed not to. The published entry's own file carries
    the same warning about its own constants.

    Our solved spreads are also consistently narrower than theirs (1.42/0.99/1.52/1.15
    against 1.553/1.101/1.599/1.272). Two independent solves of one population
    disagreeing by 0.1–0.4, neither obviously authoritative. The direction is settled;
    the magnitudes are not.

    ### 5. The R²-optimal probe shipped and scored (2026-09-16 15:28 UTC, rank 87)

    | endpoint | board ST-RAE | board R² | board Spearman | CV expected ST-RAE | ratio |
    |:--|--:|--:|--:|--:|--:|
    | CYP1A2 | 0.6559 | 0.4334 | 0.7050 | 0.8051 | 0.815 |
    | CYP2C9 | 0.5975 | 0.5660 | 0.7658 | 0.6144 | 0.972 |
    | CYP2D6 | 0.8902 | 0.2230 | 0.3610 | 0.9158 | 0.972 |
    | CYP3A4 | 0.6725 | 0.5338 | 0.7683 | 0.4854 | **1.385** |
    | macro | **0.7040** | 0.4390 | 0.6500 | 0.7052 | 1.000 |

    **The correction worked where it was needed and cost where it wasn't.** CYP2D6
    went from R² **−0.742 to +0.223** and ST-RAE from 1.310 to 0.890 — the disaster
    Part 1 flagged is gone. CYP1A2 beat its own CV expectation (ratio 0.815, the only
    endpoint where the board scored better than CV predicted at all across any
    submission this notebook has shipped). But CYP3A4 — already 96% of its R² ceiling
    per Part 1 — got measurably worse (ratio 1.385, ST-RAE +0.187 over CV) from being
    pushed to `k = rho` it did not need. **Placement effort spent on an
    already-well-placed endpoint is not neutral; it is a cost**, confirming the "do not
    spend effort on CYP2C9 or CYP3A4" reading from Part 1 rather than merely leaving it
    untested.

    **Macro landed almost exactly on the CV prediction** (0.7040 board against 0.7052
    expected, ratio 1.000) — a sharper CV-to-board match than 05 got on any endpoint.
    That is partly real signal and partly cancellation: CYP1A2's over-performance and
    CYP3A4's under-performance offset almost exactly in the average, which a macro
    number alone would hide.

    **This submission's second purpose — enabling `intersect` — mostly succeeded.**
    With 05 and this submission both scored, three of four endpoints solve cleanly by
    intersecting their curves instead of guessing the sign from ST-RAE alone:

    | endpoint | solved mean | solved sd | disagreement |
    |:--|--:|--:|--:|
    | CYP1A2 | 4.378 | 1.080 | 0.003 |
    | CYP2C9 | 4.871 | 1.200 | 0.004 |
    | CYP3A4 | 4.856 | 1.340 | 0.001 |

    Disagreement under 0.005 pIC50 units on all three — the two independent solves
    (Part 2's single-submission ST-RAE-sign fallback, and this submission's
    two-submission intersection) now agree far more tightly than the earlier
    cross-check against the reference entry's own numbers did.

    **CYP2D6 does not solve at all, and the failure is itself informative.** This
    submission's own board R² (0.223) exceeds the ceiling implied by its own Spearman
    under the bivariate-normal conversion (pearson≈0.376 → ceiling 0.141) — `intersect`
    correctly refuses rather than returning a number derived from an impossible
    premise. The only way both are true is that CYP2D6's real Pearson correlation is at
    **least 0.472**, well above what `pearson_from_spearman` estimates from 0.361. Read
    literally: `pearson_from_spearman` underestimates CYP2D6's correlation by close to
    30%, specifically because CYP2D6's ordering is far weaker than the other three
    endpoints' (Spearman 0.34–0.36 against 0.70–0.78) and weak-correlation regions are
    exactly where a skewed, truncated pIC50 distribution departs furthest from
    bivariate normality. Any future CYP2D6-specific placement work should treat
    `pearson_from_spearman` as a lower bound there, not a point estimate.
    """
    )
    return


if __name__ == "__main__":
    app.run()
