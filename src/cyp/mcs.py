"""Multiple-comparison-of-means (MCS) heatmaps.

Ported from the PXR repo's `rm_tukey_hsd` / `mcs_plot` / `make_mcs_plot_grid`
(`marimo_notebooks/2_ml_baseline.py`). Answers a sharper question than a bar chart
of mean scores: *which* pairs of methods are actually distinguishable, given the
fold-to-fold spread, after correcting for testing every pair at once?

This is the same "do not trust a point estimate" discipline as
`evaluation.paired_bootstrap`, applied as a repeated-measures ANOVA + Tukey HSD
instead of a bootstrap -- appropriate here because CV folds are a natural repeated-
measures design (every method is evaluated on the same folds). PXR's own
retrospective is the reason this matters: three finalists spanning 0.0039 MAE
looked orderable but were not, and cost five leaderboard places. A heatmap with no
stars in it is telling you something real: none of these methods are proven
different, so pick on other grounds (see CLAUDE.md).
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import polars as pl
import seaborn as sns
from statsmodels.stats.libqsturng import psturng, qsturng


def rm_tukey_hsd(
    df: pl.DataFrame,
    metric: str,
    group_col: str,
    cycle_col: str = "fold",
    alpha: float = 0.05,
    higher_is_better: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Repeated-measures Tukey HSD across CV folds.

    Args:
        df: Long frame with one row per (cycle_col, group_col) pair carrying
            `metric` -- i.e. `evaluation.fold_metrics`/`fold_metrics_classification`
            output. Every method must be present in every fold (a balanced design;
            Tukey HSD assumes it).
        metric: Column to compare.
        group_col: Column identifying which method a row belongs to.
        cycle_col: Column identifying the repeated-measures unit (the CV fold).
        alpha: Significance level (only used by the caller's star thresholds; kept
            for parity with the PXR signature).
        higher_is_better: Sorts `df_means` (and everything derived from its index --
            `df_means_diff`, `pc`) from best to worst method, so `mcs_plot` renders
            the best method top-left and the worst bottom-right. False (default)
            sorts ascending, correct for an error metric like ST-RAE; True sorts
            descending, for a metric like MCC where higher is better.

    Returns:
        `(result_tab, df_means, df_means_diff, pc)`: pairwise comparisons, per-group
        means, the matrix of mean differences, and the matrix of Tukey-adjusted
        p-values -- everything `mcs_plot` needs, all ordered best-to-worst.
    """
    df_pd = df.to_pandas()
    df_means = df_pd.groupby(group_col).mean(numeric_only=True)
    df_means = df_means.sort_values(metric, ascending=not higher_is_better)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="divide by zero encountered"
        )
        aov = pg.rm_anova(dv=metric, within=group_col, subject=cycle_col, data=df_pd, detailed=True)
    mse = aov.loc[1, "MS"]
    df_resid = aov.loc[1, "DF"]

    methods = df_means.index
    n_groups = len(methods)
    n_per_group = df_pd[group_col].value_counts().mean()

    tukey_se = np.sqrt(2 * mse / n_per_group)
    q = qsturng(1 - alpha, n_groups, df_resid)

    num_comparisons = n_groups * (n_groups - 1) // 2
    result_tab = pd.DataFrame(
        index=range(num_comparisons),
        columns=["group1", "group2", "meandiff", "lower", "upper", "p-adj"],
    )
    df_means_diff = pd.DataFrame(index=methods, columns=methods, data=0.0)
    pc = pd.DataFrame(index=methods, columns=methods, data=1.0)

    row_idx = 0
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i >= j:
                continue
            group1 = df_pd.loc[df_pd[group_col] == method1, metric]
            group2 = df_pd.loc[df_pd[group_col] == method2, metric]
            mean_diff = group1.mean() - group2.mean()
            studentized_range = np.abs(mean_diff) / tukey_se
            adjusted_p = psturng(studentized_range * np.sqrt(2), n_groups, df_resid)
            if isinstance(adjusted_p, np.ndarray):
                adjusted_p = adjusted_p[0]
            lower = mean_diff - (q / np.sqrt(2) * tukey_se)
            upper = mean_diff + (q / np.sqrt(2) * tukey_se)
            result_tab.loc[row_idx] = [method1, method2, mean_diff, lower, upper, adjusted_p]
            pc.loc[method1, method2] = adjusted_p
            pc.loc[method2, method1] = adjusted_p
            df_means_diff.loc[method1, method2] = mean_diff
            df_means_diff.loc[method2, method1] = -mean_diff
            row_idx += 1

    df_means_diff = df_means_diff.astype(float)
    result_tab["group1_mean"] = result_tab["group1"].map(df_means[metric])
    result_tab["group2_mean"] = result_tab["group2"].map(df_means[metric])
    result_tab.index = result_tab["group1"] + " - " + result_tab["group2"]

    return result_tab, df_means, df_means_diff, pc


def mcs_plot(
    pc: pd.DataFrame,
    effect_size: pd.DataFrame,
    means: pd.Series,
    ax: plt.Axes,
    reverse_cmap: bool = False,
    vlim: float | None = None,
    cell_text_size: int | None = None,
    axis_text_size: int | None = None,
) -> plt.Axes:
    """One heatmap: row/column = method (best to worst, top-left to bottom-right,
    per `rm_tukey_hsd`'s `higher_is_better`), cell color+text = mean difference
    (row minus column), stars = Tukey-adjusted significance.

    Color convention (kept consistent across a grid that may mix ST-RAE and MCC
    panels): warm always means "the row method is better than the column method",
    regardless of whether the underlying metric is minimized or maximized. For a
    maximize metric (MCC) that means warm = positive row-minus-column difference,
    the default colormap direction; pass `reverse_cmap=True` for a minimize metric
    (ST-RAE) so a *negative* (row is lower, i.e. better) difference still renders
    warm. `make_mcs_grid` sets this from its `higher_is_better` argument -- do not
    call this directly with the wrong `reverse_cmap` or the colors will say the
    opposite of what the numbers say.

    `cell_text_size` and `axis_text_size` default to None, meaning "scale to the
    grid". Fixed sizes were fine at the ~5 methods this started with but collapse
    into unreadable overlapping text by ~12, and the failure is silent -- the PNG is
    written either way, so nobody notices until they open it. Pass explicit sizes
    only to override that scaling.
    """
    cmap = "coolwarm_r" if reverse_cmap else "coolwarm"

    # Size the text from the cell, not from the method count. A cell holds a string
    # like "-0.028**" -- about 8 characters -- and a character is roughly 0.6x the
    # font size in width, so the text needs ~5x the font size in points to fit. Going
    # via the actual axes width makes this hold whatever figure size the caller
    # chose, which the previous method-count formula did not: it returned 12pt for a
    # six-method grid whose cells were only ~1.2 inches wide, and the values collided.
    # `make_mcs_grid` sizes the figure so each cell is CELL_IN inches wide, so the
    # font can be derived from that directly. A cell holds up to ~9 characters
    # ("-0.028***") and a character is roughly 0.6x the font size wide, so the text
    # needs about 5.4x the font size in points; leave a margin and divide by 6.5.
    cell_text_size = cell_text_size or max(6, min(13, int(CELL_IN * 72.0 / 6.5)))
    axis_text_size = axis_text_size or max(6, min(11, int(CELL_IN * 72.0 / 9.0)))

    significance = pc.copy().astype(object)
    significance[(pc < 0.001) & (pc >= 0)] = "***"
    significance[(pc < 0.01) & (pc >= 0.001)] = "**"
    significance[(pc < 0.05) & (pc >= 0.01)] = "*"
    significance[pc >= 0.05] = ""
    diag_values = significance.to_numpy(copy=True)
    np.fill_diagonal(diag_values, "")
    significance = pd.DataFrame(diag_values, index=significance.index, columns=significance.columns)

    annotations = effect_size.round(3).astype(str) + significance

    hax = sns.heatmap(
        effect_size,
        cmap=cmap,
        annot=annotations,
        fmt="",
        cbar=True,
        ax=ax,
        annot_kws={"size": cell_text_size},
        vmin=-2 * vlim if vlim else None,
        vmax=2 * vlim if vlim else None,
    )

    label_list = list(means.index)
    # The mean goes on the y labels only. Repeating it under every x label doubled
    # the label block's height for no extra information -- the matrix is symmetric,
    # so the same method appears on both axes.
    x_labels = list(label_list)
    y_labels = [f"{m}  ({means.loc[m]:.3g})" for m in label_list]
    # Set the tick POSITIONS before the labels. Calling set_*ticklabels alone leaves
    # matplotlib's default 10 locators in place, which silently works while a grid
    # has ~5 methods and raises "FixedLocator locations (10) does not match the
    # number of labels (28)" once it has more -- as the combined Part 3 grid does.
    positions = np.arange(len(label_list)) + 0.5
    hax.set_xticks(positions)
    hax.set_yticks(positions)
    # Always rotate. Method names here run to ~20 characters
    # ("chemprop_singletask") against a cell barely an inch wide, so horizontal
    # labels overlap their neighbours at every grid size -- width buys cell area,
    # not label room. This was briefly gated on `n_methods > 6`, which made a
    # trimmed six-method grid render *worse* than the eleven-method one it replaced.
    rotate = True
    hax.set_xticklabels(
        x_labels,
        size=axis_text_size,
        ha="right" if rotate else "center",
        va="top",
        rotation=45 if rotate else 0,
        rotation_mode="anchor" if rotate else None,
    )
    hax.set_yticklabels(
        y_labels,
        size=axis_text_size,
        ha="right" if rotate else "center",
        va="center",
        rotation=0 if rotate else 90,
    )
    hax.set_xlabel("")
    hax.set_ylabel("")
    return hax


def make_mcs_grid(
    fold_scores: dict[str, pl.DataFrame],
    metric_col: str,
    higher_is_better: dict[str, bool],
    effect_lim: dict[str, float] | None = None,
    group_col: str = "method",
    cycle_col: str = "fold",
    figsize: tuple[float, float] | None = None,
    top_n: int | None = None,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Grid of MCS heatmaps, one per key in `fold_scores`.

    **Readability caps out around eight methods.** Every cell carries a value plus
    its significance stars, so past that the text collides no matter how large the
    figure -- and it fails silently, writing the PNG regardless. Use `top_n` to trim
    to the contenders, or prefer `reference_forest_plot` (one row per method, scales
    indefinitely) when the question is "what should I ship?" rather than "which pairs
    differ?".

    Args:
        fold_scores: Maps a panel title (e.g. an endpoint or isoform name) to a
            long per-fold-metrics frame for that panel (all methods, all folds).
        metric_col: The metric column within each frame to compare (e.g. "st_rae"
            or "mcc" -- always the challenge's own metric, not a proxy).
        higher_is_better: Maps panel title to whether higher is better for that
            panel's metric, so the colormap direction reads consistently ("warm =
            row method wins") across a grid that might mix ST-RAE and MCC panels.
        effect_lim: Optional per-panel colorbar half-range; auto-scaled from the
            data if omitted.
        top_n: Keep only the `top_n` best methods per panel. A long tail of weak
            baselines both crowds the grid and flattens its colour scale, since the
            range is set by the worst pair rather than by the contenders. Must be at
            least 2 -- a one-method panel has no pair to compare.
        figsize: Explicit figure size. Omit it and the panel is scaled to the number
            of methods being compared -- see the note below on why a fixed size is
            the wrong default here.
        save_path: If given, the figure is written here (PNG) before returning.

    Returns:
        The assembled figure.
    """
    if top_n is not None and top_n < 2:
        # Caught here rather than in the ANOVA, which fails with a division-by-zero
        # warning and an opaque KeyError several frames down.
        raise ValueError(f"top_n must be at least 2 to have a pair to compare, got {top_n}")

    titles = list(fold_scores)
    # Never allocate more columns than there are panels. `ncol` used to be a fixed
    # 2-or-3 with the surplus axes hidden, but a hidden axes still takes its share of
    # the width from subplots_adjust -- a single-panel grid was laid out in a third of
    # the figure, giving 0.24in cells in a figure sized for 1.15in ones.
    ncol = min(len(titles), 2 if len(titles) in (2, 4) else 3)
    nrow = math.ceil(len(titles) / ncol)

    # Each panel is an n_methods x n_methods matrix with a tick label per method and
    # a printed value per cell, so the space it needs grows with the method count --
    # a fixed default silently produces an unreadable smear of overlapping text once
    # the grid passes roughly eight methods, and still writes the PNG. (Both of
    # 03_methods' 12-method macro grids were rendered that way.) Scale with the
    # largest panel; the 7x6 floor keeps small grids looking as they always did.
    #
    # Size on the columns actually drawn, not on `ncol`: a single-panel grid still
    # allocates 3 columns and hides 2, so multiplying by `ncol` would treble the
    # width for panels that are never rendered.
    used_col = min(len(titles), ncol)
    n_groups = max(frame[group_col].n_unique() for frame in fold_scores.values())
    if top_n is not None:
        n_groups = min(n_groups, top_n)
    # Size from the cell outward rather than sizing the figure and hoping the cells
    # follow. Each panel gets n_groups * CELL_IN inches of matrix plus fixed
    # allowances for the rotated tick labels and the colorbar -- so a cell really is
    # CELL_IN wide, which is what `mcs_plot` assumes when it picks its font sizes.
    panel_w = n_groups * CELL_IN + LABEL_IN + CBAR_IN
    panel_h = n_groups * CELL_IN + LABEL_IN
    figsize = figsize or (panel_w * used_col, panel_h * nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)

    for i, title in enumerate(titles):
        row, col = divmod(i, ncol)
        frame = fold_scores[title]
        if top_n is not None:
            # Rank first, then re-run the test on the survivors: Tukey's correction
            # depends on how many groups are being compared, so trimming afterwards
            # would leave p-values corrected for comparisons no longer shown.
            ranked = (
                frame.group_by(group_col)
                .agg(pl.col(metric_col).mean().alias("_m"))
                .sort("_m", descending=higher_is_better[title])
            )
            frame = frame.filter(pl.col(group_col).is_in(ranked.head(top_n)[group_col]))
        _, df_means, df_means_diff, pc = rm_tukey_hsd(
            frame,
            metric_col,
            group_col,
            cycle_col,
            higher_is_better=higher_is_better[title],
        )

        limit = (effect_lim or {}).get(title) or float(df_means_diff.abs().to_numpy().max() or 1.0)
        hax = mcs_plot(
            pc,
            effect_size=df_means_diff,
            means=df_means[metric_col],
            ax=axes[row][col],
            reverse_cmap=not higher_is_better[title],
            vlim=limit,
        )
        hax.set_title(f"{title}\n({metric_col.upper().replace('_', '-')})", fontsize=13)
        # Cell geometry is controlled by the figure size and the subplot rectangle
        # set at the end of this function, not by an aspect constraint. Forcing
        # `set_aspect("equal")` here instead makes the axes shrink to satisfy
        # whichever dimension is tighter, which silently undoes the width the figure
        # was sized to provide -- measured at 0.24in per cell against a 1.15in
        # target.

    for i in range(len(titles), nrow * ncol):
        row, col = divmod(i, ncol)
        axes[row][col].set_visible(False)

    # Place the axes by inches rather than letting matplotlib pick fractions. The
    # figure was sized so that n_groups * CELL_IN inches of matrix sit beside fixed
    # label and colorbar allowances; expressing those same allowances as a subplot
    # rectangle is what actually delivers a CELL_IN-wide cell. Without this the axes
    # defaults leave the matrix ~40% narrower than sized for, and the cell values --
    # which do fit a real CELL_IN cell -- collide.
    left = LABEL_IN / figsize[0]
    bottom = LABEL_IN / figsize[1]
    fig.subplots_adjust(
        left=left,
        right=1.0 - CBAR_IN / figsize[0],
        bottom=bottom,
        top=0.92,
        wspace=0.45,
        hspace=0.45,
    )
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ── reference-comparison forest plots ──────────────────────────────────────────
#
# The MCS heatmap answers "which pairs of methods differ?" -- an N x N question that
# stays readable to roughly eight methods and then collapses, because every cell must
# hold a value plus its significance stars. Both `03_methods` (12 methods) and
# `04_methods_tdi` (11) are past that point.
#
# The question actually being asked is narrower: *what should I ship, and is it
# really better than the incumbent?* That is one row per method against one
# reference, which is a forest plot -- the form used in the PXR challenge deck
# (2026_OpenADMET_PXR_Challenge_deck.pdf, p.10) and the one this mirrors. Effect size
# on the x-axis with its confidence interval, methods on the y sorted by effect, and
# colour carrying the verdict. It scales to any number of methods because each gets a
# row rather than a column as well.
#
# Both views are driven by the same `rm_tukey_hsd` call, so they cannot disagree.

#: Inches per heatmap cell. Everything else -- figure size, cell font, label font --
#: is derived from this, so a cell is the same physical size whatever the grid's
#: dimensions and the text always fits. Raise it for a poster, not to fix crowding:
#: crowding past ~8 methods is the format failing, and `top_n` or
#: `reference_forest_plot` is the fix.
CELL_IN = 1.15
#: Inches reserved for the rotated tick labels along each axis.
LABEL_IN = 2.6
#: Inches reserved for the colorbar and its tick labels.
CBAR_IN = 1.2

#: Verdict colours, shared by every plot here so "red = worse than the reference"
#: means the same thing across figures.
VERDICT_COLOURS = {
    "reference": "#1f77b4",
    "worse": "#d62728",
    "better": "#2ca02c",
    "similar": "#8c8c8c",
}


def _verdict(diff: float, p: float, higher_is_better: bool, alpha: float = 0.05) -> str:
    """Classify one comparison against the reference.

    `diff` is always row-minus-reference in raw metric units, so its sign has to be
    read through `higher_is_better`: a negative ST-RAE difference is an improvement,
    a negative MCC difference is a regression.
    """
    if p >= alpha:
        return "similar"
    improved = diff > 0 if higher_is_better else diff < 0
    return "better" if improved else "worse"


def reference_forest_plot(
    fold_scores: pl.DataFrame,
    metric_col: str,
    higher_is_better: bool,
    reference: str | None = None,
    group_col: str = "method",
    cycle_col: str = "fold",
    alpha: float = 0.05,
    title: str | None = None,
    top_n: int | None = None,
    ax: plt.Axes | None = None,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Every method against one reference: effect size, CI, and significance verdict.

    Args:
        fold_scores: Long per-fold metrics frame (all methods, all folds) -- the same
            input `make_mcs_grid` takes for one panel.
        metric_col: Metric to compare, e.g. "st_rae" or "mcc".
        higher_is_better: True for MCC, False for ST-RAE. Sets both the sort order
            and how the sign of a difference is read.
        reference: Method to compare against. Defaults to the best-scoring one, which
            makes the plot read as "is anything else as good as the winner?". Pass the
            incumbent instead to ask "is the new thing better than what we ship?".
        group_col: Column naming the method.
        cycle_col: Repeated-measures unit; CV fold.
        alpha: Significance level for the verdict colours.
        title: Plot title; a sensible default is derived if omitted.
        top_n: Keep only the `top_n` best methods plus the reference. A long tail of
            hopeless baselines compresses the x-axis and hides the differences that
            matter among the contenders.
        ax: Draw into an existing axes instead of a new figure.
        save_path: If given, write the figure here (PNG) before returning.

    Returns:
        The figure containing the plot.
    """
    result_tab, df_means, _, pc = rm_tukey_hsd(
        fold_scores,
        metric_col,
        group_col,
        cycle_col,
        alpha=alpha,
        higher_is_better=higher_is_better,
    )
    # df_means is already sorted best-to-worst by rm_tukey_hsd.
    ordered = list(df_means.index)
    if reference is None:
        reference = ordered[0]
    if reference not in ordered:
        raise ValueError(f"reference {reference!r} is not among {ordered}")

    if top_n is not None:
        keep = ordered[:top_n]
        if reference not in keep:
            keep.append(reference)
        ordered = [m for m in ordered if m in keep]

    rows = []
    for method in ordered:
        if method == reference:
            rows.append(
                {
                    "method": method,
                    "diff": 0.0,
                    "lo": 0.0,
                    "hi": 0.0,
                    "p": 1.0,
                    "verdict": "reference",
                    "mean": float(df_means.loc[method, metric_col]),
                }
            )
            continue
        # result_tab holds each pair once, in whichever order rm_tukey_hsd emitted;
        # flip the sign when the reference was recorded as group1 so `diff` always
        # reads method-minus-reference.
        hit = result_tab[(result_tab["group1"] == method) & (result_tab["group2"] == reference)]
        flip = False
        if hit.empty:
            hit = result_tab[(result_tab["group1"] == reference) & (result_tab["group2"] == method)]
            flip = True
        row = hit.iloc[0]
        sign = -1.0 if flip else 1.0
        diff = sign * float(row["meandiff"])
        lo, hi = sorted([sign * float(row["lower"]), sign * float(row["upper"])])
        p = float(pc.loc[method, reference])
        rows.append(
            {
                "method": method,
                "diff": diff,
                "lo": lo,
                "hi": hi,
                "p": p,
                "verdict": _verdict(diff, p, higher_is_better, alpha),
                "mean": float(df_means.loc[method, metric_col]),
            }
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 0.52 * len(rows) + 2.2))
    else:
        fig = ax.figure

    # Best at the top: matplotlib's y-axis grows upward, so plot in reverse order.
    ys = range(len(rows))
    for y, r in zip(reversed(list(ys)), rows, strict=True):
        colour = VERDICT_COLOURS[r["verdict"]]
        if r["verdict"] != "reference":
            ax.plot(
                [r["lo"], r["hi"]], [y, y], color=colour, lw=1.6, solid_capstyle="butt", zorder=2
            )
            for x in (r["lo"], r["hi"]):
                ax.plot([x, x], [y - 0.16, y + 0.16], color=colour, lw=1.6, zorder=2)
        ax.plot([r["diff"]], [y], "o", color=colour, ms=7, zorder=3)

    ax.axvline(0.0, color="black", ls="--", lw=1, zorder=1)
    ax.set_yticks(list(reversed(list(ys))))
    ax.set_yticklabels([f"{r['method']}\n{r['mean']:.3g}" for r in rows], fontsize=9)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    direction = "higher is better" if higher_is_better else "lower is better"
    ax.set_xlabel(f"mean {metric_col} difference from {reference}  ({direction})")
    ax.set_title(title or f"{metric_col.upper().replace('_', '-')} vs {reference}", fontsize=12)

    present = [r["verdict"] for r in rows]
    labels = {
        "reference": "reference method",
        "worse": f"significantly worse (p<{alpha})",
        "better": f"significantly better (p<{alpha})",
        "similar": "indistinguishable",
    }
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=VERDICT_COLOURS[k], label=labels[k])
        for k in ("reference", "better", "similar", "worse")
        if k in present
    ]
    ax.legend(handles=handles, fontsize=8, loc="best", framealpha=0.9)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def paired_arm_plot(
    fold_scores: pl.DataFrame,
    metric_col: str,
    higher_is_better: bool,
    arm_suffixes: tuple[str, str] = ("_multitask", "_singletask"),
    group_col: str = "method",
    cycle_col: str = "fold",
    alpha: float = 0.05,
    title: str | None = None,
    ax: plt.Axes | None = None,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """One row per *model*, showing the gap between its two arms.

    The multitask comparison is naturally paired -- `lgbm_multitask` is only ever
    interesting against `lgbm_singletask` -- but an all-pairs heatmap spends its
    space on the 45 cross-model comparisons nobody asked about, and doubles the
    category count in the process. This collapses the same data to one row per
    model: five rows instead of eleven, each answering the question directly.

    The paired structure also makes this a cleaner test. Both arms are scored on
    identical folds (`cv.shared_scaffold_folds`), so the per-fold difference is a
    repeated measure and its spread across folds is the right uncertainty -- no
    Tukey correction for comparisons that are not being made.

    **The p-values here are uncorrected paired t-tests on fold differences**, which
    is a different test from `evaluation.paired_bootstrap` (resamples compounds) and
    from the Tukey-adjusted values in `make_mcs_grid` (corrects across all pairs).
    They will not agree exactly, and the t-test is the most permissive of the three:
    it treats each fold as one observation and corrects for nothing. Use it to see
    the *shape* of the comparison -- which arms differ, by how much, in which
    direction -- and run `evaluation.paired_bootstrap` with `holm_bonferroni` before
    claiming any individual result is significant. A borderline p here (0.02-0.10)
    routinely fails correction.

    Args:
        fold_scores: Long per-fold metrics frame containing both arms of each model.
        metric_col: Metric column, e.g. "mcc" or "st_rae".
        higher_is_better: True for MCC, False for ST-RAE.
        arm_suffixes: `(treatment, control)` suffixes. The plotted difference is
            treatment minus control.
        group_col: Column naming the method.
        cycle_col: Repeated-measures unit; CV fold.
        alpha: Significance level for the paired t-test on fold differences.
        title: Plot title; derived if omitted.
        ax: Draw into an existing axes.
        save_path: If given, write the figure here (PNG).

    Returns:
        The figure containing the plot.
    """
    from scipy import stats

    treat, control = arm_suffixes
    names = fold_scores[group_col].unique().to_list()
    models = sorted(
        {n.removesuffix(treat) for n in names if n.endswith(treat)}
        & {n.removesuffix(control) for n in names if n.endswith(control)}
    )
    if not models:
        raise ValueError(f"no model has both a {treat!r} and a {control!r} arm")

    rows = []
    for model in models:
        wide = (
            fold_scores.filter(pl.col(group_col) == f"{model}{treat}")
            .select(cycle_col, pl.col(metric_col).alias("t"))
            .join(
                fold_scores.filter(pl.col(group_col) == f"{model}{control}").select(
                    cycle_col, pl.col(metric_col).alias("c")
                ),
                on=cycle_col,
                how="inner",
            )
        )
        d = (wide["t"] - wide["c"]).to_numpy()
        n = len(d)
        mean = float(d.mean())
        # Paired t on the per-fold differences: the folds are the repeated measure.
        if n > 1 and d.std(ddof=1) > 0:
            se = float(d.std(ddof=1) / (n**0.5))
            p = float(stats.ttest_rel(wide["t"].to_numpy(), wide["c"].to_numpy()).pvalue)
            half = float(stats.t.ppf(1 - alpha / 2, n - 1)) * se
        else:
            p, half = 1.0, 0.0
        rows.append(
            {
                "model": model,
                "diff": mean,
                "lo": mean - half,
                "hi": mean + half,
                "p": p,
                "verdict": _verdict(mean, p, higher_is_better, alpha),
                "n": n,
                "t_mean": float(wide["t"].mean()),
                "c_mean": float(wide["c"].mean()),
            }
        )
    # Best effect first, reading "most improved by the treatment" at the top.
    rows.sort(key=lambda r: r["diff"], reverse=higher_is_better)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 0.62 * len(rows) + 2.4))
    else:
        fig = ax.figure

    for y, r in zip(reversed(range(len(rows))), rows, strict=True):
        colour = VERDICT_COLOURS[r["verdict"]]
        ax.plot([r["lo"], r["hi"]], [y, y], color=colour, lw=1.8, solid_capstyle="butt", zorder=2)
        for x in (r["lo"], r["hi"]):
            ax.plot([x, x], [y - 0.14, y + 0.14], color=colour, lw=1.8, zorder=2)
        ax.plot([r["diff"]], [y], "o", color=colour, ms=8, zorder=3)
        ax.annotate(
            f"p={r['p']:.3f}",
            xy=(r["hi"], y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color="#444444",
        )

    ax.axvline(0.0, color="black", ls="--", lw=1, zorder=1)
    ax.set_yticks(list(reversed(range(len(rows)))))
    ax.set_yticklabels(
        [f"{r['model']}\n{r['c_mean']:.3g} → {r['t_mean']:.3g}" for r in rows], fontsize=9
    )
    ax.set_ylim(-0.7, len(rows) - 0.3)
    better_side = "right" if higher_is_better else "left"
    ax.set_xlabel(
        f"{metric_col} difference: {treat.lstrip('_')} − {control.lstrip('_')}"
        f"   ({better_side} of 0 favours {treat.lstrip('_')})"
    )
    ax.set_title(title or f"Paired arm comparison — {metric_col.upper()}", fontsize=12)

    present = {r["verdict"] for r in rows}
    labels = {
        "worse": f"{treat.lstrip('_')} significantly worse",
        "better": f"{treat.lstrip('_')} significantly better",
        "similar": "indistinguishable",
    }
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=VERDICT_COLOURS[k], label=labels[k])
        for k in ("better", "similar", "worse")
        if k in present
    ]
    if handles:
        ax.legend(handles=handles, fontsize=8, loc="best", framealpha=0.9)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
