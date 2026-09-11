"""Matched molecular pairs and the similarity network built from them.

Ported from the PXR repo's `1a_data_preprocessing.py` (the mmpdb fragment/index
shell-out and the CXSMILES canonicalisation that makes it work) and
`1b_chemical_space_and_mmp.py` (`_run_graph_layout`, `generate_similarity_network`).

An MMP is a pair of molecules differing by a single well-defined transformation --
the cheminformatics formalisation of "same compound, one change". MMPs give a
cleaner activity-cliff definition than a similarity threshold does, because the
change is explicit and chemically interpretable rather than an artifact of how a
particular fingerprint hashes substructures.

`mmpdb` is a CLI, so `build_mmp_table` shells out to it and caches both intermediate
files. On the ~5.6k unique CYP compounds the fragment step is the slow one (minutes);
it is skipped whenever its output already exists.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from rdkit import Chem

from .embedding import ryg_cmap

MMP_COLUMNS = ["smiles1", "smiles2", "ID1", "ID2", "transform", "core"]


def write_mmp_input(
    df: pl.DataFrame,
    path: str | Path,
    smiles_col: str = "SMILES",
    id_col: str = "Molecule_Name",
) -> Path:
    """Write the two-column `.smi` file `mmpdb fragment` expects.

    SMILES are re-canonicalised through RDKit first. This is not cosmetic: CXSMILES
    extensions (the ``|...|`` suffix some vendors emit) break mmpdb's whitespace-
    delimited parser, and round-tripping through `MolToSmiles` strips them. PXR hit
    exactly this and it is the single reason this helper exists.

    Args:
        df: Frame with SMILES and identifier columns.
        path: Output `.smi` path.
        smiles_col: SMILES column name.
        id_col: Identifier column name; becomes the ID in the MMP table.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    canonical = df.with_columns(
        pl.col(smiles_col)
        .map_elements(
            lambda s: Chem.MolToSmiles(mol) if (mol := Chem.MolFromSmiles(s)) is not None else None,
            return_dtype=pl.Utf8,
        )
        .alias("_canonical")
    ).filter(pl.col("_canonical").is_not_null())

    canonical.select(["_canonical", id_col]).write_csv(
        path, separator=" ", include_header=False, quote_style="never"
    )
    return path


def build_mmp_table(
    df: pl.DataFrame,
    cache_dir: str | Path,
    smiles_col: str = "SMILES",
    id_col: str = "Molecule_Name",
    stem: str = "compounds",
) -> pl.DataFrame:
    """Run `mmpdb fragment` then `mmpdb index`, caching both intermediates.

    Both steps are skipped when their output already exists, so re-running a
    notebook is cheap. Delete the cached files to force a rebuild.

    Args:
        df: Compounds to fragment.
        cache_dir: Directory for the `.smi`, `.frag` and `.mmp.csv.gz` files.
        smiles_col: SMILES column name.
        id_col: Identifier column name.
        stem: Basename shared by the three cached files.

    Returns:
        Raw MMP pairs with the columns in `MMP_COLUMNS`.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    smi_path = cache_dir / f"{stem}.smi"
    frag_path = cache_dir / f"{stem}.frag"
    mmp_path = cache_dir / f"{stem}.mmp.csv.gz"

    if not smi_path.exists():
        write_mmp_input(df, smi_path, smiles_col, id_col)

    if not frag_path.exists():
        result = subprocess.run(
            [sys.executable, "-m", "mmpdblib", "fragment", str(smi_path), "-o", str(frag_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            frag_path.unlink(missing_ok=True)
            raise RuntimeError(f"mmpdb fragment failed:\n{result.stderr}")

    if not mmp_path.exists():
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mmpdblib",
                "index",
                str(frag_path),
                "--out",
                "csv.gz",
                "-o",
                str(mmp_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            mmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"mmpdb index failed:\n{result.stderr}")

    return pl.read_csv(mmp_path, separator="\t", has_header=False, new_columns=MMP_COLUMNS)


def annotate_mmp_sizes(mmp: pl.DataFrame, max_core_ratio: float = 1.0) -> pl.DataFrame:
    """Add heavy-atom counts to MMP pairs and filter on the core/transform ratio.

    `core_transform_ratio` is the larger changing fragment over the shared core. Below
    1.0 the pair shares more than it changes, which is the loosest defensible reading
    of "matched pair". PXR used 1.0 on a diverse set and noted that a series-rich set
    warrants 0.2-0.5 -- the CYP test set is 75 hits x ~10 analogs, so tightening this
    is worth trying if the resulting network is too dense to read.

    Args:
        mmp: Raw table from `build_mmp_table`.
        max_core_ratio: Keep pairs strictly below this ratio.

    Returns:
        Filtered pairs with the heavy-atom and ratio columns added.
    """

    def heavy_atoms(smiles: str) -> int | None:
        mol = Chem.MolFromSmiles(smiles)
        return mol.GetNumHeavyAtoms() if mol is not None else None

    sizes = {"n_ha1": [], "n_ha2": [], "n_ha_core": [], "n_ha_frag1": [], "n_ha_frag2": []}
    for smi1, smi2, core, transform in mmp.select(
        ["smiles1", "smiles2", "core", "transform"]
    ).iter_rows():
        left, right = transform.split(">>")
        sizes["n_ha1"].append(heavy_atoms(smi1))
        sizes["n_ha2"].append(heavy_atoms(smi2))
        sizes["n_ha_core"].append(heavy_atoms(core))
        sizes["n_ha_frag1"].append(heavy_atoms(left))
        sizes["n_ha_frag2"].append(heavy_atoms(right))

    return (
        mmp.with_columns(
            [pl.Series(name, values, dtype=pl.Int32) for name, values in sizes.items()]
        )
        .with_columns(
            (pl.col("n_ha_frag1") - pl.col("n_ha_frag2")).abs().alias("size_diff_transform"),
            (pl.max_horizontal("n_ha_frag1", "n_ha_frag2") / pl.col("n_ha_core")).alias(
                "core_transform_ratio"
            ),
        )
        .filter(pl.col("core_transform_ratio") < max_core_ratio)
    )


def mmp_cliffs(
    mmp: pl.DataFrame,
    values: pl.DataFrame,
    endpoint: str,
    id_col: str = "Molecule_Name",
    delta_threshold: float = 2.0,
) -> pl.DataFrame:
    """MMP pairs whose two members differ sharply in activity.

    The transformation-based counterpart to `similarity.activity_cliffs`, and the
    more interpretable of the two: `transform` names the exact chemical change that
    moved potency, so the output reads as SAR rather than as a list of pair IDs.

    Args:
        mmp: Filtered pairs from `annotate_mmp_sizes`.
        values: Frame with `id_col` and the endpoint column.
        endpoint: Activity column to compare.
        id_col: Identifier column, matching the MMP table's ID1/ID2.
        delta_threshold: Minimum absolute activity difference.

    Returns:
        Cliff pairs with both values, `delta` and the transform, largest delta first.
    """
    labelled = values.filter(pl.col(endpoint).is_not_null()).select([id_col, endpoint])

    return (
        mmp.join(labelled.rename({id_col: "ID1", endpoint: "value_1"}), on="ID1", how="inner")
        .join(labelled.rename({id_col: "ID2", endpoint: "value_2"}), on="ID2", how="inner")
        .with_columns((pl.col("value_1") - pl.col("value_2")).abs().alias("delta"))
        .filter(pl.col("delta") >= delta_threshold)
        .sort("delta", descending=True)
    )


def _graph_layout(
    node_ids: Sequence[str],
    edges: pl.DataFrame,
    layout: str = "fruchterman_reingold",
    iterations: int = 50,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Force-directed 2D layout, packing disconnected components into a grid.

    An MMP graph is almost always many small components plus a few large ones.
    Laying the whole graph out at once pushes components apart by repulsion alone and
    wastes most of the canvas, so each component is laid out on its own, normalised,
    scaled by sqrt(size) and then packed into rows.
    """
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(node_ids)

    weight_col = "similarity" if "similarity" in edges.columns else None
    if weight_col is None and "CoreSize" in edges.columns:
        max_core = edges["CoreSize"].max()
        if max_core:
            edges = edges.with_columns((pl.col("CoreSize") / max_core).alias("_weight"))
            weight_col = "_weight"

    for row in edges.iter_rows(named=True):
        graph.add_edge(row["ID1"], row["ID2"], weight=float(row[weight_col]) if weight_col else 1.0)

    def layout_component(subgraph) -> dict:
        if subgraph.number_of_nodes() == 1:
            return {next(iter(subgraph)): np.array([0.0, 0.0])}
        if layout == "kamada_kawai":
            return nx.kamada_kawai_layout(subgraph, weight="weight")
        return nx.fruchterman_reingold_layout(
            subgraph, weight="weight", seed=seed, iterations=iterations
        )

    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    if len(components) == 1:
        return {node: np.asarray(pos) for node, pos in layout_component(graph).items()}

    packed: list[tuple[list[str], np.ndarray, float]] = []
    for component in components:
        positions = layout_component(graph.subgraph(component))
        coords = np.array(list(positions.values()), dtype=float)
        extent = np.ptp(coords, axis=0)
        coords = (
            (coords - coords.min(axis=0)) / extent if extent.min() > 0 else np.zeros_like(coords)
        )
        packed.append((list(positions), coords, float(np.sqrt(len(component)))))

    row_width = np.sqrt(sum(scale**2 for _, _, scale in packed))
    gap = 0.15

    result: dict[str, np.ndarray] = {}
    cursor_x = cursor_y = row_height = 0.0
    for nodes, coords, scale in packed:
        if cursor_x > 0 and cursor_x + scale > row_width:
            # Advance by the height of the row being closed, then reset it -- doing
            # this in the other order collapses every row onto the same band.
            cursor_y += row_height + gap
            cursor_x = row_height = 0.0
        offset = np.array([cursor_x, cursor_y])
        for node, xy in zip(nodes, coords * scale, strict=True):
            result[node] = xy + offset
        cursor_x += scale * (1 + gap)
        row_height = max(row_height, scale)
    return result


def plot_network(
    nodes: pl.DataFrame,
    edges: pl.DataFrame,
    id_col: str = "Molecule_Name",
    property_col: str | None = None,
    color_legend: dict[str, str] | None = None,
    property_title: str | None = None,
    layout: str = "fruchterman_reingold",
    layout_iterations: int = 50,
    node_size: float = 16.0,
    edge_opacity: float = 0.35,
    max_edges: int | None = 15000,
    title: str | None = None,
    legend_loc: str = "lower center",
    legend_ncols: int = 4,
    figsize: tuple[float, float] = (9.0, 8.0),
    dpi: int = 300,
    save_path: str | Path | None = None,
) -> Figure:
    """Draw a molecular network: nodes are compounds, edges are MMP relationships.

    Node colour follows the same three-mode convention as
    `embedding.embedding_scatter` -- a string column holds literal hex colours (pass
    `color_legend`), a numeric column gets a red-yellow-green colourbar.

    Args:
        nodes: Frame with `id_col` and optionally `property_col`.
        edges: Edge list with ID1/ID2; edges touching absent nodes are dropped.
        id_col: Node identifier column.
        property_col: Optional colour column.
        color_legend: {hex: label} entries for a literal-colour column.
        property_title: Colourbar label; defaults to `property_col`.
        layout: "fruchterman_reingold" or "kamada_kawai".
        layout_iterations: Iterations for the Fruchterman-Reingold layout.
        node_size: Marker area in points^2.
        edge_opacity: Edge opacity.
        max_edges: Render at most this many edges, keeping the heaviest.
        title: Axes title.
        legend_loc: Matplotlib legend location.
        legend_ncols: Legend columns.
        figsize: Figure size in inches.
        dpi: Figure resolution.
        save_path: Write the figure here when set.

    Returns:
        The figure drawn into.
    """
    node_ids = nodes[id_col].cast(pl.Utf8).to_list()
    known = set(node_ids)

    edges = edges.filter(
        pl.col("ID1").cast(pl.Utf8).is_in(known) & pl.col("ID2").cast(pl.Utf8).is_in(known)
    )
    if max_edges is not None and edges.height > max_edges:
        for column in ("similarity", "CoreSize", "n_ha_core"):
            if column in edges.columns:
                edges = edges.sort(column, descending=True)
                break
        edges = edges.head(max_edges)

    positions = _graph_layout(node_ids, edges, layout, layout_iterations)
    segments = [
        [tuple(positions[str(row["ID1"])]), tuple(positions[str(row["ID2"])])]
        for row in edges.iter_rows(named=True)
    ]
    xs = np.array([positions[nid][0] for nid in node_ids])
    ys = np.array([positions[nid][1] for nid in node_ids])

    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.add_collection(
            LineCollection(segments, colors=[(0.5, 0.5, 0.5, edge_opacity)], linewidths=0.8)
        )

        if property_col is None:
            ax.scatter(
                xs, ys, c="steelblue", s=node_size, linewidths=0.4, edgecolors="white", zorder=2
            )
        elif nodes[property_col].dtype == pl.Utf8:
            ax.scatter(
                xs,
                ys,
                c=nodes[property_col].to_numpy(),
                s=node_size,
                linewidths=0.4,
                edgecolors="white",
                zorder=2,
            )
            if color_legend:
                anchor = {
                    "lower center": (0.5, -0.06),
                    "upper center": (0.5, 1.06),
                }.get(legend_loc)
                ax.legend(
                    handles=[
                        mpatches.Patch(color=hex_, label=label)
                        for hex_, label in color_legend.items()
                    ],
                    loc=legend_loc,
                    bbox_to_anchor=anchor,
                    ncols=legend_ncols,
                    frameon=True,
                    fontsize=9,
                    borderaxespad=0.0,
                )
        else:
            sc = ax.scatter(
                xs,
                ys,
                c=nodes[property_col].to_numpy().astype(float),
                cmap=ryg_cmap(),
                s=node_size,
                linewidths=0.4,
                edgecolors="white",
                zorder=2,
            )
            cbar = fig.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label(property_title or property_col, fontsize=10)
            cbar.ax.tick_params(labelsize=9)

        ax.autoscale_view()
        ax.set_axis_off()
        if title:
            ax.set_title(title, fontsize=13, pad=10)
        fig.tight_layout()

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=dpi, bbox_inches="tight")
    return fig
