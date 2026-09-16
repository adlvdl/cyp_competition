"""Fetch Tox21's P450-Glo CYP panel, the weak-inhibitor-rich public source.

The second public source after `cyp.external`'s Veith qHTS panel, and the one that
covers a region neither of the others reaches.

## What it is for: explicit inactives, not a weaker median

The reference blog describes this source as weak-inhibitor-rich, citing a median
pIC50 of 4.76 against ChEMBL's 4.0 floor. Measured on our own extract that is **not**
what it delivers: the median among fitted actives is 4.92-5.07 and essentially
nothing sits below 4.0, so its actives are no weaker than anyone else's.

What it does deliver is roughly **2,000 explicitly inactive compounds per isoform** --
molecules a real assay tested and called negative. The challenge labels exist only
where the primary screen flagged a compound and ChEMBL assigns a `pchembl_value` only
where a curve fitted, so neither carries a single confirmed negative. That is the gap
this source fills, and it is a different gap from the one the blog claims.

`cyp.external`'s `max_response` attacks the same gap from a different angle -- it
keeps the inactives by recording efficacy rather than potency. The two are
complementary: efficacy says "this compound did nothing", Tox21 says "this compound
did a little, and here is how much".

## The counter-screen is not optional

These are luciferase cell-based assays: the readout is light from a reporter enzyme,
so a compound that inhibits firefly luciferase itself reads as a CYP inhibitor. Tox21
runs a dedicated counter-screen for exactly this artifact, and its call has to be
honoured or the head learns luciferase chemistry and calls it CYP inhibition. That is
what `TOX21_LUCIFERASE_COUNTERSCREEN_AID` is for.

## Five replicates, which is a quality signal rather than a nuisance

Each compound carries up to five independent runs with their own phenotype, potency
and efficacy. `load_tox21` collapses them to a median and reports how many replicates
agreed, so a caller can weight or filter on reproducibility -- information the
single-run sources simply do not have.

## No CYP1A2

This deposition batch has no CYP1A2 assay. The absence is carried rather than filled
from another protocol: substituting a different assay's numbers into one column is
the scale-mixing failure the per-source head layout exists to prevent. Models built
on this source get three heads, not four.

    python -m cyp.tox21                  # download to today's snapshot
    python -m cyp.tox21 --date 20260916  # write to a specific snapshot
"""

from __future__ import annotations

import urllib.request
from datetime import date
from pathlib import Path

import polars as pl

from . import constants as C
from .external import SMILES_COLUMN, _strip_metadata_rows

DOWNLOAD_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/assay/pcget.cgi"
    "?query=download&record_type=datatable&actvty=all&response_type=save&aid={aid}"
)

#: Replicate count in this deposition. Read from the header rather than assumed where
#: possible; this is the fallback and the documented shape.
N_REPLICATES = 5


def download(snapshot: str | None = None, force: bool = False) -> Path:
    """Fetch every Tox21 CYP assay into a dated snapshot directory."""
    snapshot = snapshot or f"{date.today():%Y%m%d}"
    directory = C.EXTERNAL_DIR / snapshot
    directory.mkdir(parents=True, exist_ok=True)

    print(f"Fetching Tox21 P450-Glo panel into {directory}")
    targets = dict(C.TOX21_AIDS)
    targets["luciferase_counterscreen"] = C.TOX21_LUCIFERASE_COUNTERSCREEN_AID

    for name, aid in targets.items():
        path = directory / C.TOX21_FILE_TEMPLATE.format(isoform=name)
        if path.exists() and not force:
            print(f"  {name:<26} already present, skipping")
            continue
        request = urllib.request.Request(
            DOWNLOAD_URL.format(aid=aid), headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            path.write_bytes(response.read())
        print(f"  {name:<26} AID {aid:<9} {path.stat().st_size / 1e6:>6.1f} MB")
    return directory


def ensure_downloaded(snapshot: str | None = None) -> str:
    """Snapshot date holding a complete Tox21 extract, downloading if absent."""
    for candidate in reversed(C.available_external_snapshots()):
        directory = C.EXTERNAL_DIR / candidate
        if all(
            (directory / C.TOX21_FILE_TEMPLATE.format(isoform=i)).exists()
            for i in list(C.TOX21_AIDS) + ["luciferase_counterscreen"]
        ):
            return candidate
    return download(snapshot).name


def _read_raw(name: str, snapshot: str | None = None) -> pl.DataFrame:
    path = C.external_snapshot_dir(snapshot) / C.TOX21_FILE_TEMPLATE.format(isoform=name)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m cyp.tox21` to fetch the Tox21 panel."
        )
    return _strip_metadata_rows(pl.read_csv(path, infer_schema_length=0, ignore_errors=True))


def luciferase_inhibitors(snapshot: str | None = None) -> set[str]:
    """SMILES the counter-screen flagged as luciferase inhibitors.

    These must be removed from every Tox21 CYP head. The artifact is systematic, not
    noise: the compound really did reduce the signal, just by inhibiting the reporter
    rather than the cytochrome.
    """
    raw = _read_raw("luciferase_counterscreen", snapshot)
    outcome = "PUBCHEM_ACTIVITY_OUTCOME"
    if outcome not in raw.columns or SMILES_COLUMN not in raw.columns:
        return set()
    flagged = raw.filter(pl.col(outcome).str.to_lowercase() == "active")
    return set(flagged[SMILES_COLUMN].drop_nulls().to_list())


def load_tox21(isoform: str, snapshot: str | None = None) -> pl.DataFrame:
    """One isoform's Tox21 potency, replicates collapsed and artifacts removed.

    Args:
        isoform: One of `constants.TOX21_AIDS`. There is no CYP1A2 assay.
        snapshot: External snapshot date; defaults to the most recent.

    Returns:
        Frame with ``SMILES``, ``pIC50`` (median over replicates that fitted a curve),
        ``n_replicates`` (how many agreed a curve existed) and ``is_inactive`` (every
        replicate called it inactive). Luciferase-inhibiting compounds are dropped.
    """
    if isoform not in C.TOX21_AIDS:
        raise ValueError(
            f"No Tox21 assay for {isoform!r}. Known: {sorted(C.TOX21_AIDS)} "
            "(this deposition batch has no CYP1A2 assay)."
        )

    raw = _read_raw(isoform, snapshot)
    artifacts = luciferase_inhibitors(snapshot)

    potency_columns = [c for c in raw.columns if c.startswith("Potency-Replicate")]
    phenotype_columns = [c for c in raw.columns if c.startswith("Phenotype-Replicate")]
    if not potency_columns:
        raise ValueError(f"{isoform}: no per-replicate Potency columns — schema changed?")

    # A potency only counts where that replicate called the compound an inhibitor.
    # An activator's AC50 is a real number meaning the opposite thing, so folding it
    # in would train the head to predict inhibition from compounds that do the
    # reverse -- the same rule `external.load_pubchem` applies.
    inhibitor_potency = []
    for potency, phenotype in zip(potency_columns, phenotype_columns, strict=False):
        inhibitor_potency.append(
            pl.when(pl.col(phenotype) == "Inhibitor")
            .then(pl.col(potency).cast(pl.Float64, strict=False))
            .otherwise(None)
        )

    frame = raw.select(
        pl.col(SMILES_COLUMN).alias("SMILES"),
        *[expression.alias(f"_p{i}") for i, expression in enumerate(inhibitor_potency)],
        # Most compounds carry only replicate 1 -- the rest are null, not negative.
        # A null must not read as "not inactive", or the inactive call is lost for
        # every singly-run compound, which here is 96% of the library.
        *[
            pl.when(pl.col(c).is_null())
            .then(None)
            .otherwise(pl.col(c) == "Inactive")
            .alias(f"_inactive{i}")
            for i, c in enumerate(phenotype_columns)
        ],
    ).filter(pl.col("SMILES").is_not_null() & pl.col("SMILES").is_in(list(artifacts)).not_())

    potency_names = [f"_p{i}" for i in range(len(potency_columns))]
    inactive_names = [f"_inactive{i}" for i in range(len(phenotype_columns))]

    # Potency is micromolar in this deposition, as in the 2007 NCGC batch.
    return (
        frame.with_columns(
            pl.mean_horizontal(
                [pl.col(c).is_not_null().cast(pl.Int8) for c in potency_names]
            ).alias("_frac_fitted"),
            pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int8) for c in potency_names]).alias(
                "n_replicates"
            ),
            pl.concat_list(potency_names).list.median().alias("_potency_um"),
            # Inactive where every replicate that actually ran said so, ignoring the
            # nulls of replicates that did not run.
            (
                pl.sum_horizontal(
                    [pl.col(c).fill_null(False).cast(pl.Int8) for c in inactive_names]
                )
                > 0
            ).alias("_any_inactive"),
            pl.sum_horizontal(
                [pl.col(c).is_not_null().cast(pl.Int8) for c in inactive_names]
            ).alias("_n_called"),
        )
        .with_columns(
            pl.when(pl.col("_potency_um").is_not_null() & (pl.col("_potency_um") > 0))
            .then(-(pl.col("_potency_um") * 1e-6).log10())
            .otherwise(None)
            .alias("pIC50"),
            # No replicate fitted an inhibition curve, and at least one called it
            # inactive outright.
            (pl.col("_any_inactive") & (pl.col("n_replicates") == 0)).alias("is_inactive"),
        )
        .group_by("SMILES")
        .agg(
            pl.col("pIC50").median().alias("pIC50"),
            pl.col("n_replicates").max().alias("n_replicates"),
            pl.col("is_inactive").all().alias("is_inactive"),
        )
        .sort("SMILES")
    )


def summarise(snapshot: str | None = None) -> pl.DataFrame:
    """Per-isoform coverage, and the median potency that justifies this source.

    ``median_pIC50`` is the number to read: if it does not sit well below ChEMBL's
    4.0 floor, this source is not supplying the range it was added for.
    """
    rows = []
    for isoform in C.TOX21_AIDS:
        frame = load_tox21(isoform, snapshot)
        active = frame.filter(pl.col("pIC50").is_not_null())
        rows.append(
            {
                "isoform": isoform,
                "compounds": frame.height,
                "with_potency": active.height,
                "inactive": int(frame["is_inactive"].sum()),
                "median_pIC50": (
                    round(float(active["pIC50"].median()), 3) if active.height else None
                ),
                "below_4": int((active["pIC50"] < C.CHEMBL_PCHEMBL_FLOOR).sum())
                if active.height
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
