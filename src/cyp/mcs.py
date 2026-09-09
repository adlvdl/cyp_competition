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
    cell_text_size: int = 14,
    axis_text_size: int = 11,
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
    """
    cmap = "coolwarm_r" if reverse_cmap else "coolwarm"

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
    x_labels = [f"{m}\n{means.loc[m]:.3g}" for m in label_list]
    y_labels = [f"{m}\n{means.loc[m]:.3g}" for m in label_list]
    hax.set_xticklabels(x_labels, size=axis_text_size, ha="center", va="top", rotation=0)
    hax.set_yticklabels(y_labels, size=axis_text_size, ha="center", va="center", rotation=90)
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
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Grid of MCS heatmaps, one per key in `fold_scores`.

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
        save_path: If given, the figure is written here (PNG) before returning.

    Returns:
        The assembled figure.
    """
    titles = list(fold_scores)
    ncol = 2 if len(titles) in (2, 4) else 3
    nrow = math.ceil(len(titles) / ncol)
    figsize = figsize or (7 * ncol, 6 * nrow)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)

    for i, title in enumerate(titles):
        row, col = divmod(i, ncol)
        frame = fold_scores[title]
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

    for i in range(len(titles), nrow * ncol):
        row, col = divmod(i, ncol)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
