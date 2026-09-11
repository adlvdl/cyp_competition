"""Interactive Altair charts with RDKit structure panels, for marimo notebooks.

Ported from the recurring pattern across the PXR repo's `1b`-`1e` notebooks: an
Altair scatter with a mouseover selection, paired with an `mo.hstack` panel that
draws the hovered molecule. Factored out here because PXR repeated the same ~60
lines in four notebooks with small divergences.

Why hover rather than click: an activity-cliff or chemical-space plot is read by
sweeping across it, and a click-to-select interaction turns that sweep into dozens of
clicks. The cost is that the selection is transient, which is why the tooltip also
carries the numbers -- the panel is for structure, the tooltip for values.

**This module renders SVG and returns marimo objects, so it is notebook-facing.**
Nothing here is imported by the modelling path; keep it that way, so `import cyp`
stays free of a marimo dependency.
"""

from __future__ import annotations

from collections.abc import Sequence

import altair as alt
import polars as pl
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

# Colour used for whichever point the cursor is nearest, across every chart here.
HIGHLIGHT_COLOR = "#f5c518"


def mol_svg(smiles: str, width: int = 280, height: int = 220) -> str:
    """Render a SMILES as an inline SVG string, or "" when it will not parse."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return _strip_xml_declaration(drawer.GetDrawingText())


def pair_svg(smiles1: str, smiles2: str, width: int = 360, height: int = 200) -> str:
    """Render two molecules side by side in one SVG, for pair views (cliffs, MMPs)."""
    left = Chem.MolFromSmiles(smiles1)
    right = Chem.MolFromSmiles(smiles2)
    if left is None or right is None:
        return ""
    combined = Chem.CombineMols(left, right)
    rdDepictor.Compute2DCoords(combined)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.DrawMolecule(combined)
    drawer.FinishDrawing()
    return _strip_xml_declaration(drawer.GetDrawingText())


def _strip_xml_declaration(svg: str) -> str:
    """Drop the `<?xml ...?>` prolog so the SVG can be inlined into HTML."""
    return svg.split("?>", 1)[-1].strip() if "?>" in svg else svg


def hover_selection(fields: Sequence[str], name: str = "hover") -> alt.Parameter:
    """A mouseover selection keyed on `fields`, cleared when the cursor leaves."""
    return alt.selection_point(
        fields=list(fields),
        name=name,
        empty=False,
        on="mouseover",
        nearest=True,
        clear="mouseout",
    )


def scatter(
    df: pl.DataFrame,
    x: str,
    y: str,
    key_fields: Sequence[str],
    color: str | None = None,
    color_scheme: str | None = "viridis",
    color_title: str | None = None,
    literal_color: bool = False,
    tooltip: Sequence[tuple[str, str, str | None]] = (),
    title: str = "",
    x_title: str | None = None,
    y_title: str | None = None,
    width: int = 520,
    height: int = 440,
    point_size: int = 24,
    selection_name: str = "hover",
) -> alt.Chart:
    """An Altair scatter wired for hover selection.

    Args:
        df: Data to plot.
        x: Column for the x axis.
        y: Column for the y axis.
        key_fields: Columns uniquely identifying a point; the hover selection and
            any downstream panel lookup both key on these.
        color: Optional colour column.
        color_scheme: Vega colour scheme for a numeric `color` column.
        color_title: Legend title; defaults to `color`.
        literal_color: Treat `color` as precomputed hex strings rather than values to
            scale. Use this when the colours encode categories assigned in Python --
            it avoids shipping a Vega scale and keeps the chart spec small.
        tooltip: (column, label, format) triples; format None for non-numeric.
        title: Chart title.
        x_title: x-axis label; defaults to `x`.
        y_title: y-axis label; defaults to `y`.
        width: Chart width in pixels.
        height: Chart height in pixels.
        point_size: Unselected marker size.
        selection_name: Name of the Altair selection parameter.

    Returns:
        A configured `alt.Chart`; wrap in `mo.ui.altair_chart` to display it.
    """
    selection = hover_selection(key_fields, selection_name)

    if color is None:
        color_encoding = alt.value("steelblue")
    elif literal_color:
        color_encoding = alt.Color(f"{color}:N", scale=None, legend=None)
    else:
        color_encoding = alt.Color(
            f"{color}:Q",
            scale=alt.Scale(scheme=color_scheme),
            legend=alt.Legend(title=color_title or color),
        )

    return (
        alt.Chart(df)
        .mark_circle(opacity=0.8)
        .encode(
            x=alt.X(
                f"{x}:Q", title=x_title or x, axis=alt.Axis(titleFontSize=12, labelFontSize=10)
            ),
            y=alt.Y(
                f"{y}:Q", title=y_title or y, axis=alt.Axis(titleFontSize=12, labelFontSize=10)
            ),
            color=alt.condition(selection, alt.value(HIGHLIGHT_COLOR), color_encoding),
            size=alt.condition(selection, alt.value(point_size * 4), alt.value(point_size)),
            tooltip=[
                alt.Tooltip(
                    f"{col}:{'Q' if fmt else 'N'}",
                    title=label,
                    **({"format": fmt} if fmt else {}),
                )
                for col, label, fmt in tooltip
            ],
        )
        .add_params(selection)
        .properties(title=title, width=width, height=height)
        .configure_title(fontSize=12)
    )


def structure_panel(
    row: dict | None,
    smiles_col: str = "SMILES",
    name_col: str = "Molecule_Name",
    fields: Sequence[tuple[str, str, str]] = (),
    badges: Sequence[tuple[str, str, str]] = (),
    width: int = 300,
    placeholder: str = "Hover over a point to see the structure",
):
    """An HTML panel showing one hovered compound: badges, values, and structure.

    Args:
        row: The selected row as a dict, or None when nothing is hovered.
        smiles_col: Column holding the SMILES to draw.
        name_col: Column holding the display name.
        fields: (column, label, format) triples rendered as label/value lines; a
            field whose value is null is skipped rather than shown as "None".
        badges: (column, label, hex) triples rendered as a badge when the boolean
            column is true.
        width: Panel width in pixels.
        placeholder: Text shown when `row` is None.

    Returns:
        An `mo.Html` panel.
    """
    import marimo as mo

    if row is None:
        return mo.Html(
            f"<div style='width:{width}px; height:320px; display:flex; "
            "align-items:center; justify-content:center; color:grey; font-size:13px; "
            "border:1px dashed #ccc; border-radius:6px; text-align:center; padding:12px'>"
            f"{placeholder}</div>"
        )

    badge_html = "".join(
        f"<span style='display:inline-block; background:{hex_}; color:#fff; "
        "padding:1px 6px; border-radius:3px; font-size:10px; margin-right:4px'>"
        f"{label}</span>"
        for column, label, hex_ in badges
        if row.get(column)
    )

    lines = ""
    for column, label, fmt in fields:
        value = row.get(column)
        if value is None:
            continue
        lines += f"<b>{label}:</b> {format(value, fmt) if fmt else value}<br>"

    return mo.Html(
        f"""
        <div style='width:{width}px; font-family:monospace; font-size:11px'>
            <div style='padding:6px; background:#f5f5f5; border-radius:4px;
                        margin-bottom:4px; line-height:1.8'>
                <b>{row.get(name_col) or ""}</b><br>
                {badge_html}{"<br>" if badge_html else ""}
                {lines}
            </div>
            {mol_svg(row[smiles_col], width=width, height=int(width * 0.75))}
        </div>
        """
    )


def pair_panel(
    row: dict | None,
    smiles1_col: str = "smiles1",
    smiles2_col: str = "smiles2",
    name1_col: str = "ID1",
    name2_col: str = "ID2",
    value1_col: str = "value_1",
    value2_col: str = "value_2",
    extra: Sequence[tuple[str, str, str]] = (),
    width: int = 380,
    placeholder: str = "Hover over a point to see the pair of structures",
):
    """An HTML panel showing a hovered compound pair side by side.

    Args:
        row: The selected row as a dict, or None when nothing is hovered.
        smiles1_col: SMILES column for the first molecule.
        smiles2_col: SMILES column for the second molecule.
        name1_col: Name column for the first molecule.
        name2_col: Name column for the second molecule.
        value1_col: Activity column for the first molecule.
        value2_col: Activity column for the second molecule.
        extra: (column, label, format) triples appended below the two molecules.
        width: Panel width in pixels.
        placeholder: Text shown when `row` is None.

    Returns:
        An `mo.Html` panel.
    """
    import marimo as mo

    if row is None:
        return mo.Html(
            f"<div style='width:{width}px; height:320px; display:flex; "
            "align-items:center; justify-content:center; color:grey; font-size:13px; "
            "border:1px dashed #ccc; border-radius:6px; text-align:center; padding:12px'>"
            f"{placeholder}</div>"
        )

    extra_html = "".join(
        f"<b>{label}:</b> {format(row[column], fmt) if fmt else row[column]}<br>"
        for column, label, fmt in extra
        if row.get(column) is not None
    )

    return mo.Html(
        f"""
        <div style='width:{width}px; font-family:monospace; font-size:11px'>
            <div style='padding:6px; background:#f8f8f8; border-radius:4px;
                        margin-bottom:6px; line-height:1.8'>
                <b>1:</b> {row[name1_col]} &nbsp; {row[value1_col]:.2f}<br>
                <b>2:</b> {row[name2_col]} &nbsp; {row[value2_col]:.2f}<br>
                {extra_html}
            </div>
            {pair_svg(row[smiles1_col], row[smiles2_col], width=width, height=int(width * 0.55))}
        </div>
        """
    )


def selected_row(chart, df: pl.DataFrame, key_fields: Sequence[str]) -> dict | None:
    """The full `df` row matching the chart's current hover selection.

    The chart's own `.value` carries only the encoded columns, so anything the panel
    needs beyond x/y/colour (SMILES, above all) has to be looked up in `df`.

    Args:
        chart: An `mo.ui.altair_chart` wrapping a chart from `scatter`.
        df: The frame the chart was built from.
        key_fields: The same `key_fields` passed to `scatter`.

    Returns:
        The matching row as a dict, or None when nothing is selected.
    """
    selection = chart.value
    if selection is None or len(selection) == 0:
        return None

    first = selection.row(0, named=True) if hasattr(selection, "row") else dict(selection.iloc[0])
    match = df
    for field in key_fields:
        match = match.filter(pl.col(field) == first[field])
    return match.row(0, named=True) if match.height else None
