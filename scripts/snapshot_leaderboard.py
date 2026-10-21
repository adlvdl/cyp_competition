"""Snapshot the OpenADMET CYP challenge leaderboards into one dated, tidy CSV.

The Space is a Gradio app that ships every leaderboard table inside its `/config`
JSON, so the tables can be read without scraping rendered HTML. Run from the project
root:

    uv run python scripts/snapshot_leaderboard.py

This writes `data/leaderboard/YYYYMMDD/leaderboards.csv`. Like the challenge data
snapshots, a rerun on the same day overwrites that day's file but never an earlier
one, so the interim reveal (2026-09-25) and the final board can be compared later.

Usernames other than ours are deliberately not stored. The board is public, but
nothing downstream needs anyone else's handle, and a committed copy would republish
every participant's name for no analytical gain. Our own rows are flagged `is_ours`.

The output is long-format (one row per entry, board, scope and metric) because the
tabs carry different metric columns: the overall tabs report macro-averages ("MA-*"),
the per-endpoint tabs plain metrics, and the structure track its own scores.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from datetime import date
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_URL = "https://openadmet-cyp-challenge.hf.space/config"
OURS = "adlvdl"

#: Columns that describe the entry rather than a score. Anything else in a table is
#: treated as a metric, so a metric the organizers add later is picked up unchanged.
ENTRY_COLUMNS = {
    "Rank",
    "Username",
    "Submitted",
    "Model Report Link",
    "Significance (tiers)",
    "Proprietary Data",
    "Open Code",
}


def fetch_config(url: str = CONFIG_URL) -> dict:
    """Download the Gradio config that embeds every leaderboard table."""
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https URL
        return json.load(response)


def _strip_html(cell: object) -> str:
    """Username cells are `<a href=...>name</a>`; keep only the visible text."""
    return re.sub(r"<[^>]+>", "", str(cell)).strip()


def _clean_label(label: str) -> str:
    """Drop the emoji prefix from a tab label ('🏆 Overall' -> 'Overall')."""
    return re.sub(r"^[^\w]+", "", label).strip()


def _scope(label: str) -> str:
    """Reduce a tab label to the scope it scores: 'overall' or an isoform name."""
    cleaned = _clean_label(label)
    if cleaned.lower() == "overall":
        return "overall"
    match = re.match(r"(CYP\w+)", cleaned)
    return match.group(1) if match else cleaned


def _walk(node: dict, components: dict[int, dict], path: tuple[str, ...]):
    """Yield (tab-label path, component) for every leaderboard under `node`.

    The layout tree nests tabs inside tabs, and a leaderboard has no label of its
    own, so which board it belongs to is only recoverable from the chain of tab items
    above it.
    """
    component = components.get(node["id"])
    if component is not None and component["type"] == "tabitem":
        path = (*path, str(component["props"].get("label", "")))
    if component is not None and component["type"] == "leaderboard":
        yield path, component
    for child in node.get("children") or []:
        yield from _walk(child, components, path)


def tidy_leaderboards(config: dict) -> pl.DataFrame:
    """Flatten every leaderboard in a Gradio config into one long-format frame."""
    components = {c["id"]: c for c in config["components"]}
    rows: list[dict] = []
    for path, component in _walk(config["layout"], components, ()):
        value = component["props"].get("value")
        if not isinstance(value, dict) or not value.get("data"):
            continue

        # path is (board tab, track tab, scope tab); the structure track has no
        # scope tab, so pad rather than index blindly.
        board_label = _clean_label(path[0]) if path else ""
        track_label = _clean_label(path[1]) if len(path) > 1 else ""
        board = "interim" if "Interim" in board_label else "live"
        if "Regression" in track_label:
            track = "regression"
        elif "Classification" in track_label:
            track = "tdi"
        else:
            track = "structure"
        scope = _scope(path[2]) if len(path) > 2 else "overall"

        headers: list[str] = value["headers"]
        for position, record in enumerate(value["data"]):
            entry = dict(zip(headers, record, strict=True))
            base = {
                "board": board,
                "track": track,
                "scope": scope,
                # Row position within its table. Ranks can tie, so `rank` alone cannot
                # re-assemble the long format into one row per entry.
                "entry": position,
                "rank": int(entry["Rank"]),
                "tier": entry.get("Significance (tiers)"),
                "submitted": entry.get("Submitted"),
                "is_ours": _strip_html(entry["Username"]) == OURS,
                "open_code": entry.get("Open Code"),
                "proprietary_data": entry.get("Proprietary Data"),
            }
            for header in headers:
                if header in ENTRY_COLUMNS:
                    continue
                try:
                    metric_value = float(entry[header])
                except (TypeError, ValueError):
                    continue
                rows.append({**base, "metric": header, "value": metric_value})

    # `tier` is null on every per-endpoint tab, which come first, so inferring the
    # schema from the leading rows would type it as null and then fail on "Tier 1".
    return pl.DataFrame(rows, infer_schema_length=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default=CONFIG_URL)
    parser.add_argument(
        "--date",
        default=date.today().strftime("%Y%m%d"),
        help="Snapshot folder name (YYYYMMDD); defaults to today.",
    )
    args = parser.parse_args()

    frame = tidy_leaderboards(fetch_config(args.url))
    out_dir = PROJECT_ROOT / "data" / "leaderboard" / args.date
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "leaderboards.csv"
    frame.write_csv(path)

    summary = (
        frame.group_by("board", "track", "scope")
        .agg(pl.col("rank").n_unique().alias("entries"))
        .sort("board", "track", "scope")
    )
    print(summary)
    print(f"wrote {frame.height:,} rows to {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
