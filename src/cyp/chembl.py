"""Fetch CYP inhibition potency from ChEMBL, as its own auxiliary target family.

The third public source, after `cyp.external` (PubChem/Veith qHTS) and `cyp.tox21`.
It is the one the repo deliberately skipped when 05 was built, and the reasoning for
skipping it still holds -- it is just no longer a reason to leave it out entirely.

## Why this was avoided, and what changed

`cyp.external`'s docstring states the case against ChEMBL: its activities are pooled
across hundreds of unrelated protocols, substrates, readouts and labs, so two rows
for the same compound and isoform routinely disagree by more than a log unit. PXR
found heterogeneous auxiliary data actively harmful, and a single-protocol source was
the version of the experiment that could attribute a result.

None of that is wrong. What changed is the *shape* the data enters in. Notebook 05
used public data as a pretraining target whose scale had to be comparable to the
challenge's. Here every source gets **its own head on its own scale**, so the encoder
learns from ChEMBL's chemistry while the head absorbs its scale disagreement. That is
the layout the reference entry uses (see `notebooks/07_placement.py`), and it is what
makes a heterogeneous source usable rather than dangerous.

The thing ChEMBL supplies that nothing else does is **new chemistry**: ~24,900
skeletons, almost none of which appear in the challenge deck and none in the blind
set. Volume of new molecules, not new labels on molecules we already have.

## What it does not supply

Low-end range. ChEMBL records a `pchembl_value` only where a concentration-response
curve was fitted, so essentially nothing sits below pIC50 4.0 -- the same double
selection that makes the challenge labels hit-enriched. Weak-inhibitor coverage is
`cyp.tox21`'s job, and efficacy coverage of the inactive half is the Veith panel's
`max_response`.

    python -m cyp.chembl                  # download to today's snapshot
    python -m cyp.chembl --date 20260916  # write to a specific snapshot
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

import polars as pl

from . import constants as C

API = "https://www.ebi.ac.uk/chembl/api/data"

#: ChEMBL's API caps a page at 1,000 records. Larger values are silently truncated
#: rather than rejected, which would quietly lose most of a target's activities.
PAGE_SIZE = 1000

#: Seconds between requests. EBI does not publish a hard rate limit for this API, but
#: a full pull is ~90 requests and there is no reason to hammer a free service.
REQUEST_DELAY = 0.2

#: Assay types worth keeping. "B" is binding, "F" is functional; ADMET rows ("A") are
#: a different measurement and are left out rather than pooled in.
ASSAY_TYPES = ("B", "F")


def _get(url: str, timeout: int = 90, retries: int = 4) -> dict:
    """One API call, retrying on the transient failures a long pull will hit."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "cyp-challenge/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last = error
            # Linear backoff: the failures here are timeouts and transient 5xx, not
            # rate limiting, so an aggressive exponential wait only slows the pull.
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"ChEMBL request failed after {retries} attempts: {url}") from last


def fetch_isoform(isoform: str, target_id: str) -> pl.DataFrame:
    """Every pchembl-valued activity for one target, paged to completion.

    Returns one row per *activity*, not per compound -- a compound measured in five
    papers appears five times. Collapsing is `load_chembl`'s job, so that the raw
    snapshot keeps the disagreement between protocols visible rather than averaging
    it away at download time.
    """
    rows: list[dict] = []
    offset = 0
    total: int | None = None

    while True:
        url = (
            f"{API}/activity?target_chembl_id={target_id}"
            f"&pchembl_value__isnull=false&format=json"
            f"&limit={PAGE_SIZE}&offset={offset}"
        )
        payload = _get(url)
        if total is None:
            total = payload["page_meta"]["total_count"]

        activities = payload.get("activities", [])
        if not activities:
            break

        for activity in activities:
            rows.append(
                {
                    "molecule_chembl_id": activity.get("molecule_chembl_id"),
                    "canonical_smiles": activity.get("canonical_smiles"),
                    "pchembl_value": activity.get("pchembl_value"),
                    "standard_type": activity.get("standard_type"),
                    "standard_relation": activity.get("standard_relation"),
                    "assay_type": activity.get("assay_type"),
                    "assay_chembl_id": activity.get("assay_chembl_id"),
                    "document_chembl_id": activity.get("document_chembl_id"),
                }
            )

        offset += PAGE_SIZE
        if offset >= total:
            break
        time.sleep(REQUEST_DELAY)

    frame = pl.DataFrame(rows) if rows else pl.DataFrame(schema={"molecule_chembl_id": pl.Utf8})
    print(f"  {isoform:<8} {target_id:<14} {frame.height:>7,} activities")
    return frame


def download(snapshot: str | None = None, force: bool = False) -> Path:
    """Fetch all five isoforms into a dated snapshot directory."""
    snapshot = snapshot or f"{date.today():%Y%m%d}"
    directory = C.EXTERNAL_DIR / snapshot
    directory.mkdir(parents=True, exist_ok=True)

    print(f"Fetching ChEMBL CYP activities into {directory}")
    for isoform, target_id in C.CHEMBL_TARGETS.items():
        path = directory / C.CHEMBL_FILE_TEMPLATE.format(isoform=isoform)
        if path.exists() and not force:
            print(f"  {isoform:<8} already present, skipping")
            continue
        fetch_isoform(isoform, target_id).write_csv(path)
    return directory


def ensure_downloaded(snapshot: str | None = None) -> str:
    """Snapshot date holding a complete ChEMBL extract, downloading if absent."""
    snapshots = C.available_external_snapshots()
    for candidate in reversed(snapshots):
        directory = C.EXTERNAL_DIR / candidate
        if all(
            (directory / C.CHEMBL_FILE_TEMPLATE.format(isoform=i)).exists()
            for i in C.CHEMBL_TARGETS
        ):
            return candidate
    return download(snapshot).name


def load_chembl(isoform: str, snapshot: str | None = None) -> pl.DataFrame:
    """One isoform's ChEMBL potency, collapsed to one row per structure.

    Args:
        isoform: One of `constants.CHEMBL_TARGETS`.
        snapshot: External snapshot date; defaults to the most recent.

    Returns:
        Frame with ``SMILES``, ``pIC50`` (the median across every activity for that
        structure), ``n_records`` and ``spread`` (max minus min pchembl for the
        structure). The last two are the honest part: a compound whose ``spread`` is
        two log units is not one measurement, and a head trained on its median is
        absorbing a disagreement rather than learning a number.
    """
    if isoform not in C.CHEMBL_TARGETS:
        raise ValueError(f"No ChEMBL target for {isoform!r}. Known: {sorted(C.CHEMBL_TARGETS)}")

    path = C.external_snapshot_dir(snapshot) / C.CHEMBL_FILE_TEMPLATE.format(isoform=isoform)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m cyp.chembl` to fetch the ChEMBL extract."
        )

    raw = pl.read_csv(path, infer_schema_length=0, ignore_errors=True)
    if raw.height == 0 or "canonical_smiles" not in raw.columns:
        return pl.DataFrame(schema={"SMILES": pl.Utf8, "pIC50": pl.Float64})

    frame = raw.with_columns(
        pl.col("pchembl_value").cast(pl.Float64, strict=False).alias("pchembl")
    ).filter(
        pl.col("pchembl").is_not_null()
        & pl.col("canonical_smiles").is_not_null()
        # A ">" relation is a bound, not a measurement. Without a censored-loss model
        # to read it as one, keeping it would teach the head that the compound is
        # exactly as potent as the concentration the assay happened to stop at.
        & (pl.col("standard_relation") == "=").fill_null(True)
        & pl.col("assay_type").is_in(list(ASSAY_TYPES)).fill_null(True)
    )

    return (
        frame.group_by("canonical_smiles")
        .agg(
            pl.col("pchembl").median().alias("pIC50"),
            pl.len().alias("n_records"),
            (pl.col("pchembl").max() - pl.col("pchembl").min()).alias("spread"),
        )
        .rename({"canonical_smiles": "SMILES"})
        .sort("SMILES")
    )


def summarise(snapshot: str | None = None) -> pl.DataFrame:
    """Per-isoform row counts and cross-protocol disagreement.

    ``median_spread`` over compounds with more than one record is the number to read
    before trusting this source: it is how far apart two labs are on the same
    molecule, and it is the quantity `cyp.external` cites as the reason to prefer a
    single-protocol panel.
    """
    rows = []
    for isoform in C.CHEMBL_TARGETS:
        frame = load_chembl(isoform, snapshot)
        repeated = frame.filter(pl.col("n_records") > 1)
        rows.append(
            {
                "isoform": isoform,
                "compounds": frame.height,
                "with_repeats": repeated.height,
                "median_spread": (
                    round(float(repeated["spread"].median()), 3) if repeated.height else None
                ),
                "median_pIC50": round(float(frame["pIC50"].median()), 3) if frame.height else None,
                "below_4": int((frame["pIC50"] < C.CHEMBL_PCHEMBL_FLOOR).sum())
                if frame.height
                else 0,
            }
        )
    return pl.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None, help="Snapshot date (YYYYMMDD)")
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    arguments = parser.parse_args()

    directory = download(arguments.date, force=arguments.force)
    print(f"\nWrote {directory}")
    print(summarise(directory.name))
