"""Fetch and normalise public CYP inhibition data from PubChem BioAssay.

PXR's retrospective names public data as the single biggest thing that challenge
missed, so this module exists to make the CYP equivalent available. It fetches the
NCGC qHTS cytochrome panel -- one assay per isoform -- into dated snapshots under
``data/external/``, mirroring `cyp.download`'s handling of the challenge release.

    python -m cyp.external                  # download to today's snapshot
    python -m cyp.external --date 20260915  # write to a specific snapshot
    python -m cyp.external --compare-only   # diff the two latest, download nothing

## Why PubChem and not ChEMBL

ChEMBL carries more CYP IC50 measurements in absolute terms (13,887 for CYP3A4
against PubChem's 13,076 compounds), but they are pooled across hundreds of
unrelated assay protocols, substrates, readouts and labs. Two ChEMBL rows for the
same compound and isoform routinely disagree by more than a log unit because they
were not measuring the same thing. The NCGC panel is one protocol run over one
compound library, so a potency is comparable across rows and an inactive call means
the same thing everywhere. PXR found heterogeneous auxiliary data actively hurting;
restricting to a single protocol is what makes a negative result here attributable
to the transfer rather than to the noise.

## The two schemas

The four assays were deposited in two batches and do not share a column layout,
which is a trap worth naming because both parse without error and only one has a
``Potency`` column:

- **AIDs 883 / 884 / 891** (CYP2C9, CYP3A4, CYP2D6) are the 2007 NCGC panel:
  ``Potency`` in micromolar, ``Phenotype`` in {Inhibitor, Activator, Inactive}.
- **AID 410** (CYP1A2) is an earlier deposition with no CYP1A2 twin in that batch:
  ``Qualified AC50`` in *molar*, and ``Activity Direction`` in {decreasing, inactive}
  standing in for the phenotype.

`load_pubchem` normalises both to the same frame, so downstream code never branches
on which isoform it is handling.

## Activators are dropped, inactives are kept

The assays measure both inhibition and activation of the isoform. An activator's
AC50 is a real number on the same scale as an inhibitor's but means the opposite
thing, so folding it into a pIC50 column would train the model to predict potent
inhibition from compounds that do the reverse. Inactives are kept and carried as a
censoring flag rather than a value: "no effect up to the top concentration" is real
information about the low-potency region, and it is the same kind of statement the
challenge's own screen negatives make (see `cyp.auxiliary`).
"""

from __future__ import annotations

import hashlib
import urllib.request
from datetime import date
from pathlib import Path

import polars as pl

from . import constants as C

# The PUG REST assay endpoint refuses any assay above 10,000 SIDs ("Too many SIDs"),
# which every assay here exceeds. The classic pcget download has no such cap and
# returns the full activity datatable, so it is the route used.
DOWNLOAD_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/assay/pcget.cgi"
    "?query=download&record_type=datatable&actvty=all&response_type=save&aid={aid}"
)

#: Rows PubChem prepends to describe the columns rather than to carry data. They are
#: identifiable by a null SID, which is more robust than skipping a fixed count.
_METADATA_TAGS = ("RESULT_TYPE", "RESULT_DESCR", "RESULT_UNIT")

SMILES_COLUMN = "PUBCHEM_EXT_DATASOURCE_SMILES"
OUTCOME_COLUMN = "PUBCHEM_ACTIVITY_OUTCOME"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(snapshot: str | None = None, force: bool = False) -> Path:
    """Download every PubChem qHTS assay into ``data/external/<snapshot>/``.

    Args:
        snapshot: Target date as ``YYYYMMDD``. Defaults to today.
        force: Re-fetch assays already present in that snapshot.

    Returns:
        The snapshot directory.
    """
    snapshot = snapshot or date.today().strftime("%Y%m%d")
    dest_dir = C.EXTERNAL_DIR / snapshot
    dest_dir.mkdir(parents=True, exist_ok=True)

    for isoform, aid in C.PUBCHEM_AIDS.items():
        dest = dest_dir / C.PUBCHEM_FILE_TEMPLATE.format(isoform=isoform)
        if dest.exists() and not force:
            print(f"skip   {isoform} (AID {aid})  already in {snapshot}")
            continue
        urllib.request.urlretrieve(DOWNLOAD_URL.format(aid=aid), dest)
        print(f"got    {isoform} (AID {aid})  {dest.stat().st_size:,} bytes")

    print(f"\nExternal snapshot written to {dest_dir}")
    return dest_dir


def ensure_downloaded(snapshot: str | None = None) -> str:
    """Return a snapshot date that has every assay on disk, downloading if needed.

    This is the entry point a notebook calls at the top of a run: it makes the
    external data behave like the challenge data, where the first cell checks and
    fetches rather than failing on a missing file halfway through.

    Args:
        snapshot: Pin a specific date. Defaults to the newest complete snapshot, or
            today's if none is complete.

    Returns:
        The snapshot date to pass to `load_pubchem`.
    """
    if snapshot is not None:
        if not _is_complete(C.EXTERNAL_DIR / snapshot):
            download(snapshot)
        return snapshot

    for candidate in reversed(C.available_external_snapshots()):
        if _is_complete(C.EXTERNAL_DIR / candidate):
            return candidate

    return download().name


def _is_complete(directory: Path) -> bool:
    """Whether `directory` holds a file for every isoform."""
    if not directory.is_dir():
        return False
    return all(
        (directory / C.PUBCHEM_FILE_TEMPLATE.format(isoform=isoform)).exists()
        for isoform in C.PUBCHEM_AIDS
    )


def _strip_metadata_rows(frame: pl.DataFrame) -> pl.DataFrame:
    """Drop PubChem's column-description preamble.

    Those rows carry the type, description and unit of each column in the data
    positions, so leaving them in makes every numeric column parse as a string and
    silently contributes three junk compounds.
    """
    return frame.filter(
        pl.col("PUBCHEM_RESULT_TAG").is_in(_METADATA_TAGS).not_()
        & pl.col(SMILES_COLUMN).is_not_null()
    )


def _read_raw(isoform: str, snapshot: str | None = None) -> pl.DataFrame:
    path = C.external_snapshot_dir(snapshot) / C.PUBCHEM_FILE_TEMPLATE.format(isoform=isoform)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m cyp.external` to fetch the PubChem "
            "qHTS panel, or call external.ensure_downloaded()."
        )
    # infer_schema_length=0 reads every column as a string: the metadata preamble
    # poisons type inference, and the numeric columns are cast explicitly below.
    return _strip_metadata_rows(pl.read_csv(path, infer_schema_length=0, ignore_errors=True))


def _max_response(raw: pl.DataFrame) -> pl.Series:
    """Efficacy at the top tested concentration, normalised across both schemas.

    The 2007 batch reports this outright as ``Max_Response``. AID 410 does not, so
    it is rebuilt from the readouts that are there, in descending order of how
    directly each measures the same quantity:

    1. the activity at the highest tested concentration -- the literal definition,
       present for 67% of rows;
    2. ``Hill Sinf``, the fitted asymptote, which correlates 0.973 with (1) where
       both exist and covers a different 56%;
    3. the most extreme activity observed anywhere on the curve, which is a weaker
       claim but is never missing.

    Coalescing the three reaches 100% coverage, which is the whole reason for
    preferring this readout over pIC50 -- a column that is null for the inactive
    half of the library would carry none of the signal this is here to supply.
    """
    if "Max_Response" in raw.columns:
        return raw["Max_Response"].cast(pl.Float64, strict=False)

    # Deposited in ascending concentration order, so the last column is the top dose.
    activity = [c for c in raw.columns if c.startswith("Activity at")]
    if not activity:
        raise ValueError("no Max_Response and no per-concentration activity columns")
    numeric = raw.select([pl.col(c).cast(pl.Float64, strict=False) for c in activity])
    return numeric.select(
        pl.coalesce(
            [
                pl.col(activity[-1]),
                raw["Hill Sinf"].cast(pl.Float64, strict=False)
                if "Hill Sinf" in raw.columns
                else pl.lit(None, dtype=pl.Float64),
                pl.min_horizontal(pl.all()),
            ]
        ).alias("max_response")
    ).to_series()


def load_pubchem(isoform: str, snapshot: str | None = None) -> pl.DataFrame:
    """One isoform's qHTS assay, normalised across the two deposition schemas.

    Args:
        isoform: One of `constants.PUBCHEM_AIDS`.
        snapshot: External snapshot date; defaults to the most recent.

    Returns:
        Frame with one row per tested compound and columns:

        - ``SMILES`` -- as deposited, not standardised. Run it through
          `standardise_smiles` before matching against challenge compounds.
        - ``pIC50`` -- potency converted from the assay's own units, null for
          compounds with no fitted curve.
        - ``is_inhibitor`` -- whether the fitted direction is inhibition. Activators
          and ambiguous rows are False.
        - ``is_inactive`` -- explicitly measured as having no effect up to the top
          concentration. These carry a null ``pIC50`` by construction and are the
          censored observations `auxiliary.censored_labels` consumes.
        - ``max_response`` -- efficacy at the top tested concentration, percent
          change from control, clipped to `constants.MAX_RESPONSE_CLIP`. Negative
          for inhibition. Unlike ``pIC50`` this is present for every screened
          compound, so it is the column that carries the inactive half of the
          library rather than dropping it.
    """
    if isoform not in C.PUBCHEM_AIDS:
        raise ValueError(
            f"No PubChem assay registered for {isoform!r}. Known: {sorted(C.PUBCHEM_AIDS)}"
        )

    raw = _read_raw(isoform, snapshot)

    if "Potency" in raw.columns:
        # 2007 panel (AIDs 883/884/891): Potency is micromolar.
        potency_molar = raw["Potency"].cast(pl.Float64, strict=False) * 1e-6
        phenotype = raw["Phenotype"]
        is_inhibitor = phenotype == "Inhibitor"
        is_inactive = phenotype == "Inactive"
    else:
        # Earlier deposition (AID 410): AC50 is already molar, and the phenotype is
        # split across a direction column instead.
        potency_molar = raw["Qualified AC50"].cast(pl.Float64, strict=False)
        direction = raw["Activity Direction"]
        is_inhibitor = direction == "decreasing"
        is_inactive = direction == "inactive"

    frame = raw.select(
        pl.col(SMILES_COLUMN).alias("SMILES"),
        pl.col(OUTCOME_COLUMN).alias("outcome"),
        potency_molar.alias("_potency_molar"),
        is_inhibitor.fill_null(False).alias("is_inhibitor"),
        is_inactive.fill_null(False).alias("is_inactive"),
        _max_response(raw).clip(*C.MAX_RESPONSE_CLIP).alias("max_response"),
    )

    # A potency is only meaningful where the curve describes inhibition, so the
    # pIC50 column is nulled elsewhere rather than carrying an activator's AC50.
    return frame.with_columns(
        pl.when(pl.col("is_inhibitor") & pl.col("_potency_molar").is_not_null())
        .then(-pl.col("_potency_molar").log10())
        .otherwise(None)
        .alias("pIC50")
    ).drop("_potency_molar")


def standardise_smiles(smiles: str) -> str | None:
    """Canonical SMILES for the largest organic fragment, or None if unparseable.

    PubChem deposits salts, mixtures and the occasional record RDKit cannot read,
    while the challenge SMILES are single neutral species. Matching the two on raw
    strings finds almost nothing; matching on this finds the real overlap. The
    largest-fragment rule is what strips a counterion without discarding the
    compound.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=False)
    if not fragments:
        return None
    largest = max(fragments, key=lambda fragment: fragment.GetNumHeavyAtoms())
    try:
        return Chem.MolToSmiles(largest)
    except Exception:
        return None


def inchikey_skeleton(smiles: str) -> str | None:
    """InChIKey connectivity block for the largest fragment, or None if unparseable.

    **The correct key for deciding whether two records are the same compound**, and
    the one every leakage check in this repo uses. Canonical SMILES is a weaker key:
    it distinguishes stereoisomers, tautomers and charge states that the InChIKey's
    first block treats as one structure. A blind-set compound deposited elsewhere as
    its enantiomer, its hydrochloride or a different tautomer is the *same molecule*
    for the purpose of "did the model already see this", and SMILES matching misses
    every one of those.

    The exposure is real rather than theoretical. Measured on the current snapshots,
    hundreds of records per source collapse when rekeyed -- 451 in the Veith CYP2D6
    arm, 140 in ChEMBL's, 47 in Tox21's. That none of them happened to be a blind-set
    compound on this particular snapshot is luck, not a property worth relying on;
    the dataset has been amended mid-challenge before.

    The full InChIKey is *not* used, because its second block encodes stereochemistry
    and protonation -- exactly the distinctions that should not separate two records
    of one compound here. The connectivity block alone is the right granularity.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if molecule is None:
        return None
    fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=False)
    if not fragments:
        return None
    largest = max(fragments, key=lambda fragment: fragment.GetNumHeavyAtoms())
    try:
        return Chem.MolToInchiKey(largest).split("-")[0]
    except Exception:
        # A record RDKit can parse but not serialise to InChI. Excluding it from a
        # leakage check would be the wrong failure direction, so the caller sees the
        # null and can fall back to the SMILES key for that row.
        return None


def skeleton_keys(smiles: list[str]) -> set[str]:
    """InChIKey skeletons for a list of SMILES, nulls dropped.

    The set to exclude against. Build it from the challenge structures and filter
    every public source through it.
    """
    keys = {inchikey_skeleton(s) for s in smiles}
    keys.discard(None)
    return keys


def add_skeleton(frame: pl.DataFrame, column: str = "SMILES") -> pl.DataFrame:
    """Add an ``inchikey_skeleton`` column, keeping rows RDKit cannot key.

    Unkeyable rows are retained with a null rather than dropped: they are a handful
    of malformed depositions, and silently removing public training data is a worse
    outcome than carrying a row that no exclusion can match.
    """
    keys = [inchikey_skeleton(s) for s in frame[column].to_list()]
    return frame.with_columns(pl.Series("inchikey_skeleton", keys))


def add_standard_smiles(frame: pl.DataFrame, column: str = "SMILES") -> pl.DataFrame:
    """Add a ``smiles_std`` column and drop rows RDKit cannot parse."""
    standard = [standardise_smiles(s) for s in frame[column].to_list()]
    return frame.with_columns(pl.Series("smiles_std", standard)).filter(
        pl.col("smiles_std").is_not_null()
    )


def pretraining_frame(
    isoform: str,
    snapshot: str | None = None,
    inactive_pic50: float | None = None,
    include_inactives: bool = True,
) -> pl.DataFrame:
    """Public pIC50 labels for one isoform, ready to pretrain on.

    Args:
        isoform: One of `constants.PUBCHEM_AIDS`.
        snapshot: External snapshot date.
        inactive_pic50: Value to assign explicitly-inactive compounds. Defaults to
            the pIC50 of the assay's top tested concentration, which is the strongest
            claim the measurement supports -- the compound is no more potent than
            this. Anything lower would be inventing precision the assay never had.
        include_inactives: Whether to keep those compounds at all. Dropping them
            makes the pretraining set an actives-only sample with the same truncation
            problem the challenge's own labels have, so the default keeps them.

    Returns:
        Frame with ``smiles_std``, ``y_true`` and ``is_censored``, one row per
        compound. Duplicate structures are collapsed to their median.
    """
    frame = add_standard_smiles(load_pubchem(isoform, snapshot))

    if inactive_pic50 is None:
        # Top concentration tested in the qHTS panel, ~57 uM for the 2007 batch.
        # Measured from the actives rather than hardcoded, so a schema change or a
        # different assay does not silently shift the floor.
        actives = frame.filter(pl.col("pIC50").is_not_null())["pIC50"]
        inactive_pic50 = float(actives.min()) if actives.len() else C.INACTIVE_PIC50_FLOOR

    active = frame.filter(pl.col("pIC50").is_not_null()).select(
        "smiles_std", pl.col("pIC50").alias("y_true"), pl.lit(False).alias("is_censored")
    )

    if include_inactives:
        inactive = frame.filter(pl.col("is_inactive") & pl.col("pIC50").is_null()).select(
            "smiles_std",
            pl.lit(inactive_pic50).alias("y_true"),
            pl.lit(True).alias("is_censored"),
        )
        combined = pl.concat([active, inactive])
    else:
        combined = active

    # A compound can appear on several plates or as several salts of one parent.
    # Median over the group is the robust summary; a compound measured both active
    # and inactive keeps the active call, since a real curve outranks a null result.
    return (
        combined.group_by("smiles_std")
        .agg(
            pl.col("y_true").median().alias("y_true"),
            pl.col("is_censored").all().alias("is_censored"),
        )
        .sort("smiles_std")
    )


def overlap_with_challenge(
    public: pl.DataFrame,
    challenge_smiles: list[str],
    column: str = "smiles_std",
) -> dict[str, int]:
    """How many public compounds are the same structure as a challenge compound.

    Matched on InChIKey connectivity block, so a compound deposited elsewhere as its
    enantiomer, a different tautomer or a salt still counts as shared. Canonical
    SMILES under-reports this: measured on the current snapshots, 451 Veith CYP2D6
    records, 140 ChEMBL and 47 Tox21 collapse onto another record when rekeyed.

    Exact-structure overlap is the leak that matters least (it is easy to remove)
    and the one worth reporting first, because a large number here means the public
    set is partly a copy of the training set rather than new information.
    """
    challenge_keys = skeleton_keys(challenge_smiles)
    public_keys = skeleton_keys(public[column].to_list())
    return {
        "n_public": len(public_keys),
        "n_challenge": len(challenge_keys),
        "n_shared": len(public_keys & challenge_keys),
        "n_public_only": len(public_keys - challenge_keys),
    }


def compare(old: str, new: str) -> dict[str, str]:
    """Compare two external snapshots file by file, by content hash."""
    old_dir, new_dir = C.external_snapshot_dir(old), C.external_snapshot_dir(new)
    result: dict[str, str] = {}
    for isoform in C.PUBCHEM_AIDS:
        name = C.PUBCHEM_FILE_TEMPLATE.format(isoform=isoform)
        old_file, new_file = old_dir / name, new_dir / name
        if not old_file.exists() and new_file.exists():
            result[name] = "added"
        elif old_file.exists() and not new_file.exists():
            result[name] = "removed"
        elif not old_file.exists() and not new_file.exists():
            continue
        else:
            result[name] = "unchanged" if _sha256(old_file) == _sha256(new_file) else "CHANGED"
    return result


def report_against_previous(snapshot: str | None = None) -> None:
    """Print a comparison of `snapshot` against the external snapshot before it."""
    snapshots = C.available_external_snapshots()
    snapshot = snapshot or (snapshots[-1] if snapshots else None)
    if snapshot is None:
        print("No external snapshots to compare.")
        return
    older = [s for s in snapshots if s < snapshot]
    if not older:
        print(f"{snapshot} is the only external snapshot -- nothing to compare against.")
        return

    previous = older[-1]
    print(f"\n{previous} -> {snapshot}")
    for name, status in compare(previous, snapshot).items():
        marker = " " if status == "unchanged" else "*"
        print(f" {marker} {status:<10} {name}")


def summarise(snapshot: str | None = None) -> pl.DataFrame:
    """Per-isoform row counts for a snapshot -- what was actually fetched."""
    rows = []
    for isoform in C.PUBCHEM_AIDS:
        frame = load_pubchem(isoform, snapshot)
        potency = frame["pIC50"].drop_nulls()
        rows.append(
            {
                "isoform": isoform,
                "aid": C.PUBCHEM_AIDS[isoform],
                "n_tested": frame.height,
                "n_inhibitor_pic50": int(frame["pIC50"].is_not_null().sum()),
                "n_inactive": int(frame["is_inactive"].sum()),
                "median_pic50": float(potency.median()) if potency.len() else float("nan"),
                "max_pic50": float(potency.max()) if potency.len() else float("nan"),
            }
        )
    return pl.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", help="snapshot date YYYYMMDD (default: today)")
    parser.add_argument("--force", action="store_true", help="re-fetch assays already present")
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="compare the two most recent external snapshots without downloading",
    )
    args = parser.parse_args()

    if args.compare_only:
        report_against_previous(args.date)
    else:
        download(args.date, force=args.force)
        report_against_previous(args.date)
        print()
        print(summarise(args.date))
