"""Derived views of the two challenge files that carry no scored endpoint.

`cyp-challenge-single-concentration-TRAIN.csv` and `cyp-challenge-TRAIN_Emax.csv`
ship with the challenge but were never used by notebooks 01-04. This module turns
them into the specific shapes the auxiliary-data arms of `05_auxiliary_data.py`
consume, and records what is actually in them.

## What the single-concentration screen is

Every one of 4,376 compounds was tested against all four isoforms at a single
concentration (49.5 uM), one measurement each, no replicates. The readout is
``log2fc_estimate``: log2 fold change in enzyme activity against control, so *more
negative means more inhibition*. It is dense -- no nulls anywhere -- which makes it
the only complete four-isoform matrix in the release.

Measured against the dose-response labels on the current snapshot:

| endpoint | n with both | corr(log2fc, pIC50) | R^2 from log2fc alone |
|:--|--:|--:|--:|
| CYP1A2 | 1,412 | -0.887 | 0.786 |
| CYP3A4 | 1,805 | -0.882 | 0.777 |
| CYP2C9 | 1,285 | -0.831 | 0.690 |
| CYP2D6 | 1,493 | -0.704 | 0.496 |

A single scalar explains half to four-fifths of the variance in the target.

## The catch, and why it shapes every arm

**The screen is not available at test time.** The blinded test set is
``Molecule_Name,SMILES`` only, and none of its 750 compounds appear in the screen.
So ``log2fc`` cannot be a model *feature* -- a model trained on it would have
nothing to read at prediction time. It can only be used as an auxiliary *target*
(predict it alongside pIC50, so the encoder learns from it) or to manufacture extra
labelled *rows*.

## The screen is the triage rule that created the sparsity

A compound got a dose-response curve if and only if it hit in this screen, which is
why the pIC50 columns are sparse and why that sparsity is not missing-at-random. How
much the two groups overlap in screen space differs sharply by isoform -- measured as
``unlabelled_min - labelled_max``, where a number near zero means the triage rule was
close to a hard threshold:

| isoform | labelled median | unlabelled median | overlap |
|:--|--:|--:|--:|
| CYP2D6 | -1.74 | **+0.38** | -0.28 |
| CYP2C9 | -0.81 | -0.60 | -1.97 |
| CYP1A2 | -1.55 | -0.20 | -4.48 |
| CYP3A4 | -1.44 | **-1.93** | -6.10 |

CYP2D6 is the extreme case: its unlabelled compounds have a *positive* median log2fc
(no inhibition at all) while every labelled compound is below -0.59, so its labelled
set is very nearly a hard-thresholded selection of the strongest inhibitors. That is a
structural explanation for why CYP2D6 has been the hardest endpoint in every run so
far -- the model is asked to predict a range it was barely shown the bottom of.

CYP3A4 is the opposite and worth noting, because it is the one endpoint that already
works: its *unlabelled* compounds are more inhibitory on average than its labelled
ones, so whatever decided which CYP3A4 compounds got a curve was not this screen
alone. `triage_frame` exposes selection as a modelling target of its own.

## Emax carries almost nothing

`emax_summary` computes it rather than asserting it, but the measured result on the
current snapshot is that every Emax value sits within roughly 0.1 of -1.0 (standard
deviations 0.03 to 0.09), correlates with pIC50 between -0.10 and 0.56, and differs
between TDI-positive and TDI-negative compounds by about 0.02 -- well inside one
standard deviation. It is a saturated readout.

The quantity Emax was expected to provide is available elsewhere: the *difference*
between pIC50 in the TDI (NADPH-preincubation) condition and the direct condition,
both of which are already columns in ``cyp-challenge-TRAIN_TDI.csv``. That shift
separates the TDI classes cleanly (CYP2D6: 0.68 against 0.15; CYP3A4: 0.58 against
0.22). `tdi_shift_frame` builds it.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from . import constants as C
from . import data

#: Auxiliary target column names, one per isoform. Named with a distinct suffix so a
#: stray auxiliary column can never be mistaken for a scored endpoint by the
#: submission validator or by `evaluation`.
SCREEN_TARGET_TEMPLATE = "{isoform}_screen_log2fc"

#: pIC50 at the single screening concentration. A compound showing no inhibition at
#: 49.5 uM cannot have an IC50 below it, so this is the ceiling on the potency of
#: every screen negative -- the censoring point for `censored_labels`.
SCREEN_CENSOR_PIC50 = -float(np.log10(C.SINGLE_CONC_MOLAR))


def screen_targets(snapshot: str | None = None) -> pl.DataFrame:
    """The screen pivoted wide: one row per compound, one log2fc column per isoform.

    This is the shape the auxiliary-head Chemprop arm needs -- four extra output
    columns that sit alongside the four pIC50 targets in one wide frame.

    Returns:
        Frame with ``Molecule_Name``, ``SMILES`` and one
        ``{isoform}_screen_log2fc`` column per isoform. Dense: the screen has no
        missing cells, so none of these columns carries a null.
    """
    screen = data.load_single_concentration(snapshot)
    wide = screen.pivot(
        values=C.SCREEN_EFFECT_COLUMN,
        index=["Molecule_Name", "SMILES"],
        on=C.SCREEN_ENZYME_COLUMN,
    )
    renames = {
        isoform: SCREEN_TARGET_TEMPLATE.format(isoform=isoform)
        for isoform in C.ISOFORMS
        if isoform in wide.columns
    }
    return wide.rename(renames).sort("Molecule_Name")


def screen_vs_label(endpoint: str, snapshot: str | None = None) -> pl.DataFrame:
    """Join one endpoint's pIC50 against that isoform's screen measurement.

    Args:
        endpoint: A regression endpoint, e.g. ``"CYP3A4_pIC50_direct_inhibition"``.
        snapshot: Data snapshot date.

    Returns:
        Frame of ``Molecule_Name``, ``SMILES``, ``log2fc``, ``fdr``, ``y_true`` and
        ``has_label`` covering every screened compound -- labelled or not. Keeping
        the unlabelled rows is the point: they are what `triage_frame` and
        `censored_labels` are built from.
    """
    isoform = endpoint.split("_")[0]
    screen = data.load_single_concentration(snapshot).filter(
        pl.col(C.SCREEN_ENZYME_COLUMN) == isoform
    )
    labels = data.load_train_inhibition(snapshot).select(
        "Molecule_Name", pl.col(endpoint).alias("y_true")
    )
    return (
        screen.select(
            "Molecule_Name",
            "SMILES",
            pl.col(C.SCREEN_EFFECT_COLUMN).alias("log2fc"),
            pl.col(C.SCREEN_FDR_COLUMN).alias("fdr"),
        )
        .join(labels, on="Molecule_Name", how="left")
        .with_columns(pl.col("y_true").is_not_null().alias("has_label"))
        .sort("Molecule_Name")
    )


def screen_label_agreement(snapshot: str | None = None) -> pl.DataFrame:
    """How well the screen predicts the dose-response label, per endpoint.

    Computed rather than quoted, because the dataset has been amended mid-challenge
    and the correlations in this module's docstring are what justify spending a run
    on the auxiliary arms at all.
    """
    rows = []
    for endpoint in C.REGRESSION_ENDPOINTS:
        joined = screen_vs_label(endpoint, snapshot)
        labelled = joined.filter(pl.col("has_label"))
        log2fc = labelled["log2fc"].to_numpy()
        y_true = labelled["y_true"].to_numpy()
        correlation = (
            float(np.corrcoef(log2fc, y_true)[0, 1]) if len(labelled) > 2 else float("nan")
        )
        unlabelled = joined.filter(pl.col("has_label").not_())
        rows.append(
            {
                "endpoint": endpoint,
                "n_labelled": labelled.height,
                "n_screened_unlabelled": unlabelled.height,
                "corr_log2fc_pic50": correlation,
                "r2_log2fc_only": correlation**2,
                "max_log2fc_labelled": float(log2fc.max()) if len(labelled) else float("nan"),
                "min_log2fc_unlabelled": float(unlabelled["log2fc"].min())
                if unlabelled.height
                else float("nan"),
            }
        )
    return pl.DataFrame(rows)


def triage_separation(snapshot: str | None = None) -> pl.DataFrame:
    """Where the screen-to-curve triage threshold sits for each isoform.

    A compound was progressed to a dose-response curve on the strength of its screen
    result, so the labelled and unlabelled log2fc distributions are separated. How
    cleanly they separate differs sharply by isoform, and a clean separation means
    the endpoint's labels are a truncated sample rather than a representative one.

    Returns:
        Per-endpoint frame with each group's log2fc quantiles and ``separation`` --
        the gap between the labelled maximum and the unlabelled minimum. A positive
        separation means the two groups do not overlap at all.
    """
    rows = []
    for endpoint in C.REGRESSION_ENDPOINTS:
        joined = screen_vs_label(endpoint, snapshot)
        labelled = joined.filter(pl.col("has_label"))["log2fc"]
        unlabelled = joined.filter(pl.col("has_label").not_())["log2fc"]
        if not labelled.len() or not unlabelled.len():
            continue
        rows.append(
            {
                "endpoint": endpoint,
                "n_labelled": labelled.len(),
                "n_unlabelled": unlabelled.len(),
                "labelled_median": float(labelled.median()),
                "labelled_max": float(labelled.max()),
                "unlabelled_median": float(unlabelled.median()),
                "unlabelled_min": float(unlabelled.min()),
                # Negative overlap, positive a clean gap.
                "separation": float(unlabelled.min()) - float(labelled.max()),
            }
        )
    return pl.DataFrame(rows)


def triage_frame(endpoint: str, snapshot: str | None = None) -> pl.DataFrame:
    """Every screened compound with a boolean "was progressed to a curve" label.

    This is the first stage of the selection-bias arm: a classifier for *which*
    compounds get measured, learned from structure alone. Its output is a propensity
    that the second stage can condition on, which is what turns "the labels are a
    biased sample" from a caveat into something the model can correct for.

    Returns:
        Frame with ``Molecule_Name``, ``SMILES`` and boolean ``y_true``
        (was progressed), one row per screened compound.
    """
    joined = screen_vs_label(endpoint, snapshot)
    return joined.select(
        "Molecule_Name",
        "SMILES",
        pl.col("has_label").alias("y_true"),
    )


def censored_labels(
    endpoint: str,
    snapshot: str | None = None,
    fdr_threshold: float = 0.05,
    log2fc_threshold: float = -0.5,
    interval_width: float = 2.0,
) -> pl.DataFrame:
    """Screen negatives recast as weak, censored pIC50 labels.

    A compound that showed no inhibition at 49.5 uM has an IC50 above that
    concentration, which is a genuine measurement of low potency rather than a
    missing value. Adding these as labels roughly triples the training rows for the
    three weak endpoints, and does it in precisely the region ST-RAE treats
    forgivingly -- a wide credible interval means a prediction anywhere inside it
    scores zero error, so a mislabelled weak compound is cheap.

    The filter is deliberately conservative, and the FDR term does most of the work:
    of CYP2D6's 2,883 unlabelled compounds, 2,823 have a small effect but only 974
    also fail significance. The yield therefore varies a lot by endpoint (CYP1A2
    +1,381 rows, CYP2D6 +974, CYP2C9 +770, CYP3A4 only +197), and CYP3A4 -- the one
    endpoint that already works -- gains almost nothing. That asymmetry is itself
    informative: it says this arm is aimed squarely at the weak endpoints.

    Args:
        endpoint: The regression endpoint these labels belong to.
        snapshot: Data snapshot date.
        fdr_threshold: A compound counts as a screen negative only if its effect is
            not significant at this FDR. Requiring statistical non-significance as
            well as a small effect keeps out compounds that genuinely inhibit weakly
            but were measured precisely. Checked against the labelled compounds that
            pass the same filter: their mean true pIC50 is 2.5-3.3, so the filter is
            selecting genuinely inactive compounds rather than missed hits.
        log2fc_threshold: Effect size above which (i.e. less negative than) a
            compound counts as showing no inhibition.
        interval_width: How far below the censoring point ``y_lower`` is placed. The
            interval runs from ``SCREEN_CENSOR_PIC50 - interval_width`` up to the
            censoring point, encoding "somewhere at or below this, we cannot say
            where" in the form the metric already understands.

            The default of 2.0 is calibrated, not arbitrary. Labelled compounds that
            would also pass this filter turn out to have true pIC50 around 2.5-3.3 --
            well below the 4.31 censoring point -- so a narrow interval hugging the
            censoring point would systematically overstate the potency of these rows.
            Two log units reaches down to 2.31 and covers the range those compounds
            actually occupy.

    Returns:
        A frame in `data.training_frame`'s schema -- ``Molecule_Name``, ``SMILES``,
        ``y_true``, ``y_lower``, ``y_upper`` -- holding only compounds that have no
        real dose-response label for this endpoint.
    """
    joined = screen_vs_label(endpoint, snapshot)
    negatives = joined.filter(
        pl.col("has_label").not_()
        & (pl.col("log2fc") > log2fc_threshold)
        & (pl.col("fdr") > fdr_threshold)
    )
    return negatives.select(
        "Molecule_Name",
        "SMILES",
        pl.lit(SCREEN_CENSOR_PIC50).alias("y_true"),
        pl.lit(SCREEN_CENSOR_PIC50 - interval_width).alias("y_lower"),
        pl.lit(SCREEN_CENSOR_PIC50).alias("y_upper"),
    )


def augmented_training_frame(
    endpoint: str,
    snapshot: str | None = None,
    **censor_kwargs,
) -> pl.DataFrame:
    """`data.training_frame` with screen-negative weak labels appended.

    The real dose-response rows come first and are untouched; the censored rows are
    added below them. Every downstream harness consumes this unchanged, because the
    schema is identical to the unaugmented frame.

    Note that the added rows change what a fold contains, so an augmented arm must be
    compared against its unaugmented twin on folds assigned over the *union* of
    compounds -- otherwise the two arms are scored on different test sets and the
    comparison is not paired. `cv.shared_scaffold_folds` over both frames does this.
    """
    real = data.training_frame(endpoint, snapshot)
    weak = censored_labels(endpoint, snapshot, **censor_kwargs)
    return pl.concat([real, weak], how="vertical")


def emax_summary(snapshot: str | None = None) -> pl.DataFrame:
    """What Emax actually carries, per isoform and condition.

    Returns spread and correlation with the corresponding pIC50. The reason this
    function exists rather than a flat "Emax is useless" note is that the claim is
    snapshot-dependent: if OpenADMET ever amends the file, this recomputes the
    verdict instead of preserving a stale one.
    """
    emax = data.load_train_emax(snapshot)
    inhibition = data.load_train_inhibition(snapshot)
    rows = []
    for isoform in C.ISOFORMS:
        for condition, template in (
            ("direct", C.EMAX_DIRECT_TEMPLATE),
            ("tdi", C.EMAX_TDI_TEMPLATE),
        ):
            column = template.format(isoform=isoform)
            if column not in emax.columns:
                continue
            pic50_column = C.PIC50_DIRECT_TEMPLATE.format(isoform=isoform)
            joined = (
                emax.select("Molecule_Name", pl.col(column).alias("emax"))
                .join(
                    inhibition.select("Molecule_Name", pl.col(pic50_column).alias("pic50")),
                    on="Molecule_Name",
                )
                .drop_nulls()
            )
            correlation = (
                float(np.corrcoef(joined["emax"].to_numpy(), joined["pic50"].to_numpy())[0, 1])
                if joined.height > 2
                else float("nan")
            )
            values = emax[column].drop_nulls()
            rows.append(
                {
                    "isoform": isoform,
                    "condition": condition,
                    "n": values.len(),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "range": float(values.max()) - float(values.min()),
                    "corr_with_pic50": correlation,
                }
            )
    return pl.DataFrame(rows)


def tdi_shift_frame(isoform: str, snapshot: str | None = None) -> pl.DataFrame:
    """pIC50 shift between the TDI and direct conditions, with the TDI label.

    Both pIC50 columns are already in the TDI file, so this costs nothing to compute
    and is the mechanistically correct quantity: a time-dependent inhibitor becomes
    more potent after NADPH preincubation, so a positive shift is the signature of
    the thing ``is_TDI`` labels.

    This is a *diagnostic*, not a feature. Neither pIC50 column exists for the
    blinded test set, so a model reading the shift at training time would have
    nothing to read at prediction time -- the same constraint the screen data has.
    What it can do is set an upper bound on how well the TDI track could possibly go
    if the shift were predicted first and thresholded second.

    Returns:
        Frame with ``Molecule_Name``, ``SMILES``, ``pic50_tdi``, ``pic50_direct``,
        ``shift`` and boolean ``y_true``.
    """
    if isoform not in C.TDI_ISOFORMS:
        raise ValueError(f"TDI is scored for {C.TDI_ISOFORMS}, not {isoform!r}")
    tdi = data.load_train_tdi(snapshot)
    return (
        tdi.select(
            "Molecule_Name",
            "SMILES",
            pl.col(C.PIC50_TDI_TEMPLATE.format(isoform=isoform)).alias("pic50_tdi"),
            pl.col(C.PIC50_DIRECT_TEMPLATE.format(isoform=isoform)).alias("pic50_direct"),
            pl.col(f"{isoform}_is_TDI").cast(pl.Boolean).alias("y_true"),
        )
        .drop_nulls()
        .with_columns((pl.col("pic50_tdi") - pl.col("pic50_direct")).alias("shift"))
    )


def tdi_shift_summary(snapshot: str | None = None) -> pl.DataFrame:
    """How well the TDI/direct pIC50 shift separates the TDI classes.

    Reports the per-class mean shift and the AUC of using the raw shift as a score,
    which is the ceiling a two-stage "predict the shift, then threshold" model could
    reach if the shift were predicted perfectly.
    """
    from sklearn.metrics import roc_auc_score

    rows = []
    for isoform in C.TDI_ISOFORMS:
        frame = tdi_shift_frame(isoform, snapshot)
        positive = frame.filter(pl.col("y_true"))["shift"]
        negative = frame.filter(pl.col("y_true").not_())["shift"]
        rows.append(
            {
                "isoform": isoform,
                "n": frame.height,
                "n_positive": positive.len(),
                "mean_shift_positive": float(positive.mean()),
                "mean_shift_negative": float(negative.mean()),
                "auc_shift_alone": float(
                    roc_auc_score(frame["y_true"].to_numpy(), frame["shift"].to_numpy())
                ),
            }
        )
    return pl.DataFrame(rows)
