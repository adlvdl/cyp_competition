"""Pairwise similarity, nearest-neighbour coverage, and activity cliffs.

Ported from the PXR repo's `1c_activity_cliffs.py` (`compute_pairwise_similarities`,
`plot_similarity_distributions`) and `1d_train_test_exploration.py` (the cross-set
nearest-neighbour block), generalised from PXR's single pEC50 to CYP's four
regression endpoints.

**A Tanimoto number is meaningless without its fingerprint.** PXR's 1c makes this
point with a density plot: ECFP6 similarities run visibly lower than MACCS ones on
the same compounds, so a single "similar compounds" threshold does not transfer.
The working thresholds carried over are `CLIFF_SIM_THRESHOLDS` below; check them
against `similarity_percentiles` on this dataset before trusting them, which is
exactly what the notebook does.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure
from rdkit import DataStructs

from . import fingerprints

# Similarity thresholds above which a pair counts as "structurally similar", per
# fingerprint. Rules of thumb carried from PXR -- they are not interchangeable, and
# the notebook re-checks them against this dataset's own distribution.
CLIFF_SIM_THRESHOLDS: dict[str, float] = {
    "maccs": 0.8,
    "ecfp4": 0.4,
    "ecfp6": 0.4,
}

# A pIC50 difference this large between structurally similar compounds is the
# conventional activity-cliff definition (two orders of magnitude in potency).
CLIFF_DELTA_THRESHOLD = 2.0

# Potency bands for describing a compound qualitatively. The cut at 4.0 is the
# assay resolution floor (`constants.INACTIVE_PIC50_FLOOR`); the rest are chosen
# against this dataset's own distribution, where pIC50 >= 6 covers only 1.8-9.8%
# of labelled compounds depending on isoform, so "potent" has to mean >= 6 to pick
# out a meaningful tail rather than a third of the data.
POTENCY_BANDS: tuple[tuple[float, str], ...] = (
    (6.0, "potent (>=6)"),
    (5.0, "moderate (5-6)"),
    (4.0, "weak (4-5)"),
    (float("-inf"), "inactive (<4)"),
)

# Band for a compound with no measurement on the track in question -- distinct from
# "inactive", which is a measured result.
NOT_TESTED_BAND = "not tested"

# Named fingerprint configurations, so a plot legend can say "ecfp4_1k" and mean
# something reproducible rather than "ecfp".
FP_CONFIGS: dict[str, tuple[str, dict]] = {
    "maccs": ("maccs", {}),
    "ecfp4_1k": ("ecfp", {"fp_size": 1024, "radius": 2, "include_chirality": True}),
    "ecfp4_4k": ("ecfp", {"fp_size": 4096, "radius": 2, "include_chirality": True}),
    "ecfp6": ("ecfp", {"fp_size": 4096, "radius": 3, "include_chirality": True}),
}

_BULK_METRICS = {
    "tanimoto": DataStructs.BulkTanimotoSimilarity,
    "dice": DataStructs.BulkDiceSimilarity,
    "cosine": DataStructs.BulkCosineSimilarity,
}


def _to_bitvects(fps: np.ndarray) -> list[DataStructs.ExplicitBitVect]:
    """Convert a fingerprint matrix to RDKit bit vectors for the bulk similarity ops.

    Counts are binarised: the bulk Tanimoto functions operate on presence/absence.
    """
    binary = (fps > 0).astype(np.uint8)
    bitvects = []
    for row in binary:
        bv = DataStructs.ExplicitBitVect(int(row.shape[0]))
        for bit in np.flatnonzero(row).tolist():
            bv.SetBit(int(bit))
        bitvects.append(bv)
    return bitvects


def bitvects_for(
    df: pl.DataFrame, fp_name: str, smiles_col: str = "SMILES"
) -> list[DataStructs.ExplicitBitVect]:
    """Bit vectors for a named configuration in `FP_CONFIGS`."""
    if fp_name not in FP_CONFIGS:
        raise ValueError(f"Unknown fingerprint {fp_name!r}. Available: {sorted(FP_CONFIGS)}")
    kind, kwargs = FP_CONFIGS[fp_name]
    return _to_bitvects(fingerprints.compute(df[smiles_col].to_list(), kind, **kwargs))


def pairwise_similarities(
    df: pl.DataFrame,
    id_col: str,
    fp_name: str,
    metric: str = "tanimoto",
    smiles_col: str = "SMILES",
) -> pl.DataFrame:
    """Similarity for every unique (i, j) pair with i < j.

    Quadratic in the number of compounds -- ~12M pairs for the 4,905-row inhibition
    table, which is tolerable but not free. Subset to one endpoint first where the
    question is about that endpoint.

    Args:
        df: Frame with an identifier and a SMILES column.
        id_col: Unique identifier column.
        fp_name: Key into `FP_CONFIGS`.
        metric: One of "tanimoto", "dice", "cosine".
        smiles_col: SMILES column name.

    Returns:
        Long frame with columns ID1, ID2, fingerprint, metric, similarity.
    """
    if metric not in _BULK_METRICS:
        raise ValueError(f"metric must be one of {sorted(_BULK_METRICS)}, got {metric!r}")

    bitvects = bitvects_for(df, fp_name, smiles_col)
    ids = df[id_col].cast(pl.Utf8).to_list()
    bulk = _BULK_METRICS[metric]

    id1, id2, sims = [], [], []
    for i in range(len(ids) - 1):
        row = bulk(bitvects[i], bitvects[i + 1 :])
        id1.extend([ids[i]] * len(row))
        id2.extend(ids[i + 1 :])
        sims.extend(row)

    return pl.DataFrame(
        {
            "ID1": pl.Series(id1, dtype=pl.Utf8),
            "ID2": pl.Series(id2, dtype=pl.Utf8),
            "fingerprint": pl.Series([fp_name] * len(sims), dtype=pl.Utf8),
            "metric": pl.Series([metric] * len(sims), dtype=pl.Utf8),
            "similarity": pl.Series(sims, dtype=pl.Float32),
        }
    )


def cross_similarities(
    left: pl.DataFrame,
    right: pl.DataFrame,
    fp_name: str = "ecfp4_1k",
    metric: str = "tanimoto",
    smiles_col: str = "SMILES",
) -> np.ndarray:
    """Full (len(left), len(right)) similarity matrix between two compound sets.

    Used for train-vs-test comparisons, where every cross pair matters and no pair is
    double-counted the way `pairwise_similarities` guards against within one set.
    """
    left_bvs = bitvects_for(left, fp_name, smiles_col)
    right_bvs = bitvects_for(right, fp_name, smiles_col)
    bulk = _BULK_METRICS[metric]
    return np.array([bulk(bv, right_bvs) for bv in left_bvs], dtype=np.float32)


def nearest_neighbours(
    query: pl.DataFrame,
    reference: pl.DataFrame,
    fp_name: str = "ecfp4_1k",
    metric: str = "tanimoto",
    smiles_col: str = "SMILES",
    id_col: str = "Molecule_Name",
    carry: Sequence[str] = (),
) -> pl.DataFrame:
    """For each `query` compound, its most similar `reference` compound.

    This is the "can the model possibly know this compound?" diagnostic: a test
    compound whose nearest training neighbour sits at Tanimoto 0.2 is an
    extrapolation, and no amount of tuning fixes that.

    Args:
        query: Compounds to find neighbours for (typically the test set).
        reference: Compounds to search in (typically a training endpoint subset).
        fp_name: Key into `FP_CONFIGS`.
        metric: One of "tanimoto", "dice", "cosine".
        smiles_col: SMILES column, present in both frames.
        id_col: Identifier column, present in both frames.
        carry: Extra `reference` columns to attach to the matched neighbour, each
            prefixed `nn_` -- e.g. the endpoint value of the nearest neighbour.

    Returns:
        `query` with `nn_similarity`, `nn_{id_col}`, `nn_SMILES` and one `nn_*`
        column per entry in `carry`.
    """
    sims = cross_similarities(query, reference, fp_name, metric, smiles_col)
    best = sims.argmax(axis=1)
    best_sim = sims[np.arange(sims.shape[0]), best]

    columns = [
        pl.Series("nn_similarity", best_sim, dtype=pl.Float32),
        pl.Series(f"nn_{id_col}", [reference[id_col][int(i)] for i in best]),
        pl.Series("nn_SMILES", [reference[smiles_col][int(i)] for i in best]),
    ]
    for column in carry:
        values = reference[column]
        columns.append(pl.Series(f"nn_{column}", [values[int(i)] for i in best]))

    return query.with_columns(columns)


def potency_band(value: float | None) -> str:
    """Qualitative label for one pIC50, using `POTENCY_BANDS`.

    Returns `NOT_TESTED_BAND` for a null. Note that Polars' `map_elements` skips
    nulls rather than passing them in, so a caller applying this through an
    expression must `fill_null(NOT_TESTED_BAND)` as well -- see
    `neighbour_label_profile`.
    """
    if value is None:
        return NOT_TESTED_BAND
    for threshold, label in POTENCY_BANDS:
        if value >= threshold:
            return label
    return POTENCY_BANDS[-1][1]


def neighbour_label_profile(
    query: pl.DataFrame,
    reference: pl.DataFrame,
    regression_endpoints: Sequence[str],
    classification_endpoints: Sequence[str] = (),
    fp_name: str = "ecfp4_1k",
    smiles_col: str = "SMILES",
    id_col: str = "Molecule_Name",
) -> pl.DataFrame:
    """For each `query` compound, its single nearest neighbour and what that
    neighbour is labelled for across every track.

    This is the "we found the closest analog -- now, what do we actually know about
    it?" question, and it is distinct from `nearest_neighbours` called per endpoint.
    There, each endpoint gets its own neighbour drawn from that endpoint's labelled
    compounds, so a test compound has four different analogs. Here there is one
    global neighbour per compound, and the interesting quantity is how many of the
    tracks that one analog carries: a neighbour measured on all six is a far more
    informative anchor than one measured only on CYP3A4, even at identical Tanimoto.

    Args:
        query: Compounds to profile (typically the test set).
        reference: Compounds to search, carrying every endpoint column.
        regression_endpoints: pIC50 columns to report, each with a potency band.
        classification_endpoints: Boolean TDI columns to report.
        fp_name: Key into `FP_CONFIGS`.
        smiles_col: SMILES column, present in both frames.
        id_col: Identifier column, present in both frames.

    Returns:
        `query` with `nn_similarity`, `nn_{id_col}`, `nn_SMILES`, one `nn_{endpoint}`
        value and `band_{endpoint}` label per regression endpoint, one
        `nn_{endpoint}` per classification endpoint, plus `n_tracks_measured` and
        `n_regression_tracks` counting how many tracks the neighbour carries.
    """
    endpoints = [*regression_endpoints, *classification_endpoints]
    profile = nearest_neighbours(
        query,
        reference,
        fp_name=fp_name,
        smiles_col=smiles_col,
        id_col=id_col,
        carry=endpoints,
    )

    # Potency band per regression endpoint, computed from the neighbour's value.
    # `fill_null` runs *before* map_elements deliberately: Polars skips null inputs
    # rather than passing them through, so a null would stay null and the
    # "not tested" band -- the most interesting one here, since it marks a neighbour
    # that carries no information on this track -- would silently disappear.
    profile = profile.with_columns(
        [
            pl.col(f"nn_{endpoint}")
            .map_elements(potency_band, return_dtype=pl.Utf8)
            .fill_null(NOT_TESTED_BAND)
            .alias(f"band_{endpoint}")
            for endpoint in regression_endpoints
        ]
    )

    # How many tracks the neighbour is measured on. This is the headline number:
    # it says how much context the closest available analog actually brings.
    return profile.with_columns(
        pl.sum_horizontal(
            [pl.col(f"nn_{e}").is_not_null().cast(pl.Int32) for e in endpoints]
        ).alias("n_tracks_measured"),
        pl.sum_horizontal(
            [pl.col(f"nn_{e}").is_not_null().cast(pl.Int32) for e in regression_endpoints]
        ).alias("n_regression_tracks"),
    )


def similarity_percentiles(sim_df: pl.DataFrame, group_col: str = "fingerprint") -> pl.DataFrame:
    """Percentile summary of each group's similarity distribution.

    The upper tail is what matters for cliff thresholds: if a fingerprint's p99 sits
    below the threshold, that threshold selects almost nothing and any cliff count
    from it is noise.
    """
    rows = []
    for group in sim_df[group_col].unique(maintain_order=True).to_list():
        values = sim_df.filter(pl.col(group_col) == group)["similarity"].to_numpy()
        rows.append(
            {
                group_col: group,
                "n_pairs": int(values.size),
                "median": round(float(np.median(values)), 3),
                "Q75": round(float(np.percentile(values, 75)), 3),
                "p95": round(float(np.percentile(values, 95)), 3),
                "p99": round(float(np.percentile(values, 99)), 3),
                "p99.9": round(float(np.percentile(values, 99.9)), 3),
                "max": round(float(values.max()), 3),
            }
        )
    return pl.DataFrame(rows)


def plot_similarity_distributions(
    sim_df: pl.DataFrame,
    group_col: str,
    group_order: Sequence[str],
    colors: Sequence[str],
    title: str,
    x_label: str = "Tanimoto similarity",
    figsize: tuple[float, float] = (6.5, 5.0),
    dpi: int = 300,
    ax: "plt.Axes | None" = None,
    save_path: str | Path | None = None,
) -> Figure:
    """Overlaid KDE curves, one per group of the similarity distribution.

    Args:
        sim_df: Frame with a `similarity` column and `group_col`.
        group_col: Column splitting the data into curves.
        group_order: Groups to draw, in legend order.
        colors: One colour per group.
        title: Axes title.
        x_label: x-axis label.
        figsize: Figure size in inches, ignored when `ax` is given.
        dpi: Figure resolution.
        ax: Draw into this axes instead of creating a figure.
        save_path: Write the figure here when set.

    Returns:
        The figure drawn into.
    """
    from scipy.stats import gaussian_kde

    grid = np.linspace(0, 1, 500)
    with plt.style.context("seaborn-v0_8-whitegrid"):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.get_figure()

        for group, color in zip(group_order, colors, strict=True):
            values = sim_df.filter(pl.col(group_col) == group)["similarity"].to_numpy()
            if values.size < 2:
                continue
            density = gaussian_kde(values, bw_method="scott")(grid)
            ax.plot(grid, density, color=color, linewidth=2, label=str(group))
            ax.fill_between(grid, density, alpha=0.15, color=color)

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9, frameon=True, framealpha=0.9, edgecolor="0.8")
        ax.set_title(title, fontsize=12)

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=dpi, bbox_inches="tight")
    return fig


def activity_cliffs(
    sim_df: pl.DataFrame,
    values: pl.DataFrame,
    endpoint: str,
    id_col: str = "Molecule_Name",
    sim_threshold: float | None = None,
    delta_threshold: float = CLIFF_DELTA_THRESHOLD,
) -> pl.DataFrame:
    """Pairs that are structurally similar but differ sharply in activity.

    Activity cliffs break the "similar structure, similar activity" assumption every
    fingerprint model rests on, so a high cliff rate on an endpoint is direct
    evidence that fingerprints will struggle there -- which is the open question on
    CYP2D6/1A2/2C9.

    Args:
        sim_df: Output of `pairwise_similarities` (one fingerprint).
        values: Frame carrying `id_col` and `endpoint`.
        endpoint: Activity column to compare across each pair.
        id_col: Identifier column, matching the IDs in `sim_df`.
        sim_threshold: Minimum similarity; defaults to the `CLIFF_SIM_THRESHOLDS`
            entry for the fingerprint named in `sim_df`.
        delta_threshold: Minimum absolute activity difference.

    Returns:
        Cliff pairs with both activity values and `delta`, most extreme first.
    """
    if sim_threshold is None:
        fp_name = sim_df["fingerprint"][0]
        # ecfp4_1k / ecfp4_4k both fall back to the shared "ecfp4" threshold.
        key = fp_name if fp_name in CLIFF_SIM_THRESHOLDS else fp_name.rsplit("_", 1)[0]
        sim_threshold = CLIFF_SIM_THRESHOLDS.get(key, 0.4)

    labelled = values.filter(pl.col(endpoint).is_not_null()).select([id_col, endpoint])

    return (
        sim_df.filter(pl.col("similarity") >= sim_threshold)
        .join(labelled.rename({id_col: "ID1", endpoint: "value_1"}), on="ID1", how="inner")
        .join(labelled.rename({id_col: "ID2", endpoint: "value_2"}), on="ID2", how="inner")
        .with_columns((pl.col("value_1") - pl.col("value_2")).abs().alias("delta"))
        .filter(pl.col("delta") >= delta_threshold)
        .sort("delta", descending=True)
    )


def cliff_summary(
    sim_df: pl.DataFrame,
    values: pl.DataFrame,
    endpoints: Sequence[str],
    id_col: str = "Molecule_Name",
    sim_threshold: float | None = None,
    delta_threshold: float = CLIFF_DELTA_THRESHOLD,
) -> pl.DataFrame:
    """Cliff counts and rates per endpoint, from one similarity table.

    The rate -- cliffs as a fraction of similar pairs that are labelled for that
    endpoint -- is the comparable number across endpoints, since the four endpoints
    have different numbers of labelled compounds and hence different pair counts.
    """
    rows = []
    for endpoint in endpoints:
        cliffs = activity_cliffs(sim_df, values, endpoint, id_col, sim_threshold, delta_threshold)
        labelled = values.filter(pl.col(endpoint).is_not_null()).select([id_col, endpoint])
        n_similar = (
            sim_df.filter(
                pl.col("similarity")
                >= (
                    sim_threshold
                    if sim_threshold is not None
                    else CLIFF_SIM_THRESHOLDS.get(sim_df["fingerprint"][0].rsplit("_", 1)[0], 0.4)
                )
            )
            .join(labelled.rename({id_col: "ID1"}), on="ID1", how="inner")
            .join(labelled.rename({id_col: "ID2"}), on="ID2", how="inner")
            .height
        )
        rows.append(
            {
                "endpoint": endpoint,
                "similar_pairs": n_similar,
                "cliffs": cliffs.height,
                "cliff_rate_pct": round(100 * cliffs.height / n_similar, 2) if n_similar else 0.0,
                "max_delta": round(float(cliffs["delta"].max()), 2) if cliffs.height else 0.0,
            }
        )
    return pl.DataFrame(rows)


def plot_cliff_venn(
    cliff_sets: dict[str, set[frozenset]],
    title: str,
    colors: Sequence[str] = ("#4e79a7", "#f28e2b"),
    figsize: tuple[float, float] = (5.5, 4.5),
    dpi: int = 300,
    save_path: str | Path | None = None,
) -> Figure:
    """Two-way Venn of cliff pairs found by different fingerprints.

    PXR's finding was that the overlap is small -- different fingerprints disagree
    about which pairs are even similar, so "activity cliff" is a fingerprint-relative
    label rather than a property of the chemistry.

    Args:
        cliff_sets: Exactly two {label: set of frozenset({id1, id2})} entries.
        title: Axes title.
        colors: One colour per set.
        figsize: Figure size in inches.
        dpi: Figure resolution.
        save_path: Write the figure here when set.

    Returns:
        The figure drawn into.
    """
    from matplotlib_venn import venn2

    if len(cliff_sets) != 2:
        raise ValueError(f"plot_cliff_venn needs exactly 2 sets, got {len(cliff_sets)}")

    labels = list(cliff_sets)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    venn2(
        subsets=(cliff_sets[labels[0]], cliff_sets[labels[1]]),
        set_labels=labels,
        set_colors=tuple(colors),
        alpha=0.6,
        ax=ax,
    )
    ax.set_title(title, fontsize=12)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=dpi, bbox_inches="tight")
    return fig


def cliff_pair_set(cliffs: pl.DataFrame) -> set[frozenset]:
    """Cliff pairs as unordered frozensets, so (A, B) and (B, A) collapse to one."""
    return {frozenset(pair) for pair in cliffs.select(["ID1", "ID2"]).rows()}
