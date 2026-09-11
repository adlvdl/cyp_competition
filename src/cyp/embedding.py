"""2D embeddings of chemical space and the scatter plots built on them.

Ported from the PXR repo's `1b_chemical_space_and_mmp.py` (`add_tsne_columns`,
`add_umap_columns`, `generate_embedding_plot`) and `1c_activity_cliffs.py`
(`generate_embedding_plot_continuous`), merged into one entry point per concern and
rewritten against this repo's `fingerprints.compute` rather than re-dispatching
scikit-fingerprints locally.

Two things carried over deliberately:

- **Jaccard, not Euclidean, for UMAP on binary fingerprints.** Jaccard distance on a
  bit vector is 1 - Tanimoto, which is the similarity measure chemists actually
  reason about. Euclidean on a 4096-bit vector is dominated by heavy-atom count.
- **PCA before t-SNE.** t-SNE on raw high-dimensional fingerprints is both slow and
  noisy; 50 PCA components first is the standard remedy and is what PXR used.

Both embeddings are unsupervised and see no labels, so it is safe to fit them on
train and test together -- and necessary, since the whole point is to see where the
test compounds sit relative to training chemical space.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from . import fingerprints

# Red-yellow-green diverging ramp, used where a property has a "bad to good"
# reading (potency, similarity to training data). Matplotlib has no built-in that
# reads this way without also implying a meaningful midpoint.
RYG_COLORS = ["#d73027", "#fee08b", "#1a9850"]

SEED = 42


def ryg_cmap() -> LinearSegmentedColormap:
    """The red-yellow-green ramp as a matplotlib colormap."""
    return LinearSegmentedColormap.from_list("ryg", RYG_COLORS)


def _resolve_cmap(cmap: str):
    return ryg_cmap() if cmap == "ryg" else cmap


def fingerprint_matrix(
    df: pl.DataFrame,
    smiles_col: str = "SMILES",
    kind: str = "ecfp",
    **fp_kwargs,
) -> np.ndarray:
    """Fingerprint matrix for `df`, defaulting to the ECFP6 used for embeddings.

    PXR embedded on ECFP6 (radius 3, 4096 bits, count-based, chirality-aware) rather
    than the ECFP4 used for modelling: a larger radius resolves local structure
    better for visualisation, where there is no overfitting cost to more bits.
    """
    defaults = {"radius": 3, "fp_size": 4096, "include_chirality": True, "count": True}
    if kind in ("ecfp", "morgan"):
        fp_kwargs = {**defaults, **fp_kwargs}
    return fingerprints.compute(df[smiles_col].to_list(), kind, **fp_kwargs)


def add_tsne(
    df: pl.DataFrame,
    fps: np.ndarray | None = None,
    smiles_col: str = "SMILES",
    x_col: str = "TSNE_x",
    y_col: str = "TSNE_y",
    n_pca: int = 50,
    perplexity: float = 30.0,
    seed: int = SEED,
) -> pl.DataFrame:
    """Add t-SNE coordinates, PCA-reduced first.

    Args:
        df: Frame with a SMILES column.
        fps: Precomputed fingerprint matrix; computed from `smiles_col` if None.
        smiles_col: SMILES column name, used only when `fps` is None.
        x_col: Output column for the first t-SNE coordinate.
        y_col: Output column for the second t-SNE coordinate.
        n_pca: PCA components retained before t-SNE.
        perplexity: t-SNE perplexity, clamped to n_samples - 1.
        seed: Random seed for both PCA and t-SNE.

    Returns:
        `df` with the two coordinate columns appended.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    if fps is None:
        fps = fingerprint_matrix(df, smiles_col)
    n = fps.shape[0]
    if n <= 1:
        return df.with_columns(pl.lit(float("nan")).alias(x_col), pl.lit(float("nan")).alias(y_col))

    reduced = PCA(n_components=min(n_pca, n - 1), random_state=seed).fit_transform(fps)
    coords = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=min(perplexity, float(n - 1)),
        init="pca",
        learning_rate="auto",
    ).fit_transform(reduced)

    return df.with_columns(pl.Series(x_col, coords[:, 0]), pl.Series(y_col, coords[:, 1]))


def add_umap(
    df: pl.DataFrame,
    fps: np.ndarray | None = None,
    smiles_col: str = "SMILES",
    x_col: str = "UMAP_x",
    y_col: str = "UMAP_y",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "jaccard",
    seed: int = SEED,
) -> pl.DataFrame:
    """Add UMAP coordinates.

    Defaults to Jaccard distance, which on a binary fingerprint is 1 - Tanimoto --
    see the module docstring. Count-based fingerprints are binarised first, since
    Jaccard is only defined on presence/absence.

    Args:
        df: Frame with a SMILES column.
        fps: Precomputed fingerprint matrix; computed from `smiles_col` if None.
        smiles_col: SMILES column name, used only when `fps` is None.
        x_col: Output column for the first UMAP coordinate.
        y_col: Output column for the second UMAP coordinate.
        n_neighbors: Local/global structure trade-off, clamped to n_samples - 1.
        min_dist: How tightly points may pack together.
        metric: Distance metric; "jaccard" for bit vectors, "euclidean" for descriptors.
        seed: Random seed.

    Returns:
        `df` with the two coordinate columns appended.
    """
    from umap import UMAP

    if fps is None:
        fps = fingerprint_matrix(df, smiles_col)
    n = fps.shape[0]
    if n <= 1:
        return df.with_columns(pl.lit(float("nan")).alias(x_col), pl.lit(float("nan")).alias(y_col))

    if metric == "jaccard":
        fps = (fps > 0).astype(np.uint8)

    coords = UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, n - 1),
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    ).fit_transform(fps)

    return df.with_columns(pl.Series(x_col, coords[:, 0]), pl.Series(y_col, coords[:, 1]))


def embedding_scatter(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    color_legend: dict[str, str] | None = None,
    cutoff_value: float | None = None,
    title: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    colorbar_title: str | None = None,
    cmap: str = "viridis",
    point_size: float = 18.0,
    alpha: float = 0.8,
    legend_loc: str = "best",
    figsize: tuple[float, float] = (8.0, 7.0),
    dpi: int = 300,
    ax: "plt.Axes | None" = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Scatter an embedding, colouring points three possible ways.

    The colour mode is chosen from `color_col`'s dtype and `cutoff_value`:

    - **string column** -- values are literal hex colours, assigned by the caller.
      Pass `color_legend` as {hex: label} to get a legend, since the colours carry
      no scale of their own.
    - **numeric + `cutoff_value`** -- split into two classes above/below the cutoff.
    - **numeric alone** -- continuous colourbar.

    Args:
        df: Frame holding the coordinates and the colour column.
        x_col: Column with the x coordinate.
        y_col: Column with the y coordinate.
        color_col: Optional colour column; points are uniform blue if None.
        color_legend: {hex: label} legend entries for a literal-colour column.
        cutoff_value: Split a numeric colour column into two classes at this value.
        title: Axes title.
        x_title: x-axis label; defaults to `x_col`.
        y_title: y-axis label; defaults to `y_col`.
        colorbar_title: Colourbar label; defaults to `color_col`.
        cmap: Matplotlib colormap name, or "ryg" for the red-yellow-green ramp.
        point_size: Marker area in points^2.
        alpha: Marker opacity.
        legend_loc: Matplotlib legend location string.
        figsize: Figure size in inches, ignored when `ax` is given.
        dpi: Figure resolution.
        ax: Draw into this axes instead of creating a figure -- used to build the
            per-endpoint grids, where every panel shares one figure.
        save_path: Write the figure here when set.

    Returns:
        The figure drawn into.
    """
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()

    with plt.style.context("seaborn-v0_8-whitegrid"):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.get_figure()

        if color_col is None:
            ax.scatter(x, y, c="steelblue", s=point_size, alpha=alpha, linewidths=0)
        elif df[color_col].dtype == pl.Utf8:
            ax.scatter(x, y, c=df[color_col].to_numpy(), s=point_size, linewidths=0)
            if color_legend:
                ax.legend(
                    handles=[
                        mpatches.Patch(color=hex_, label=label)
                        for hex_, label in color_legend.items()
                    ],
                    loc=legend_loc,
                    frameon=True,
                    fontsize=9,
                )
        elif cutoff_value is not None:
            values = df[color_col].to_numpy().astype(float)
            above = values > cutoff_value
            ax.scatter(
                x[above],
                y[above],
                c="#d73027",
                s=point_size,
                alpha=alpha,
                linewidths=0,
                label=f"{color_col} > {cutoff_value}",
            )
            ax.scatter(
                x[~above],
                y[~above],
                c="steelblue",
                s=point_size,
                alpha=alpha,
                linewidths=0,
                label=f"{color_col} <= {cutoff_value}",
            )
            ax.legend(frameon=True, fontsize=9)
        else:
            values = df[color_col].to_numpy().astype(float)
            sc = ax.scatter(
                x,
                y,
                c=values,
                cmap=_resolve_cmap(cmap),
                s=point_size,
                alpha=alpha,
                linewidths=0,
            )
            cbar = fig.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label(colorbar_title or color_col, fontsize=10)
            cbar.ax.tick_params(labelsize=9)

        ax.set_xlabel(x_title or x_col, fontsize=12, labelpad=6)
        ax.set_ylabel(y_title or y_col, fontsize=12, labelpad=6)
        ax.tick_params(axis="both", labelsize=9)
        if title:
            ax.set_title(title, fontsize=13, pad=10)

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=dpi, bbox_inches="tight")
    return fig


def endpoint_grid(
    df: pl.DataFrame,
    endpoints: Sequence[str],
    x_col: str = "UMAP_x",
    y_col: str = "UMAP_y",
    titles: Sequence[str] | None = None,
    ncols: int = 2,
    cmap: str = "viridis",
    point_size: float = 10.0,
    panel_size: tuple[float, float] = (5.5, 4.8),
    dpi: int = 300,
    background_color: str = "#e8e8e8",
    save_path: str | Path | None = None,
) -> Figure:
    """One embedding panel per endpoint, each coloured by that endpoint's values.

    The four regression endpoints do not share rows (see CLAUDE.md), so a single
    scatter cannot show all of them. Each panel colours only the compounds labelled
    for that endpoint and draws the rest as grey background, which keeps every panel
    on the same coordinates -- the comparison across panels is the point.

    Args:
        df: Frame with the embedding coordinates and one column per endpoint.
        endpoints: Endpoint column names, one panel each.
        x_col: Column with the x coordinate.
        y_col: Column with the y coordinate.
        titles: Panel titles; defaults to the endpoint names.
        ncols: Panels per row.
        cmap: Colormap for the endpoint values.
        point_size: Marker area for the labelled points.
        panel_size: Size of a single panel in inches.
        dpi: Figure resolution.
        background_color: Colour for compounds unlabelled for that endpoint.
        save_path: Write the figure here when set.

    Returns:
        The figure holding the grid.
    """
    titles = list(titles) if titles is not None else list(endpoints)
    nrows = (len(endpoints) + ncols - 1) // ncols

    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
            dpi=dpi,
            squeeze=False,
        )
        flat = axes.ravel()

        x_all = df[x_col].to_numpy()
        y_all = df[y_col].to_numpy()

        for ax, endpoint, title in zip(flat, endpoints, titles, strict=False):
            labelled = df[endpoint].is_not_null().to_numpy()
            values = df[endpoint].to_numpy()

            # Unlabelled compounds stay visible as context so the panels are
            # visually comparable, but recede behind the labelled ones.
            ax.scatter(
                x_all[~labelled],
                y_all[~labelled],
                c=background_color,
                s=point_size * 0.6,
                alpha=0.5,
                linewidths=0,
            )
            sc = ax.scatter(
                x_all[labelled],
                y_all[labelled],
                c=values[labelled].astype(float),
                cmap=_resolve_cmap(cmap),
                s=point_size,
                alpha=0.85,
                linewidths=0,
            )
            cbar = fig.colorbar(sc, ax=ax, pad=0.02)
            cbar.ax.tick_params(labelsize=8)

            ax.set_title(f"{title}  (n={int(labelled.sum())})", fontsize=11, pad=8)
            ax.set_xlabel(x_col, fontsize=9)
            ax.set_ylabel(y_col, fontsize=9)
            ax.tick_params(axis="both", labelsize=8)

        for ax in flat[len(endpoints) :]:
            ax.set_visible(False)

        fig.tight_layout()

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=dpi, bbox_inches="tight")
    return fig
