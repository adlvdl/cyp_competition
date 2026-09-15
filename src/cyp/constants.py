"""Canonical column names, paths, and challenge constants.

Endpoint names mirror `vendor/cyp_challenge_tutorial/evaluation/config.py`, which is
itself a port of the challenge scoring backend. Do not rename these without checking
that file first -- the leaderboard matches on exact column names.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Raw data lives in dated snapshot directories, data/raw/YYYYMMDD/, one per download.
# The dataset has been amended mid-challenge, so keeping snapshots side by side makes
# those changes diffable instead of silently overwriting them.
SNAPSHOT_GLOB = "20[0-9][0-9][0-1][0-9][0-3][0-9]"

# Filenames as published on Hugging Face (openadmet/cyp-challenge-train-test).
TRAIN_INHIBITION_FILE = "cyp-challenge-TRAIN_inhibition.csv"
TRAIN_TDI_FILE = "cyp-challenge-TRAIN_TDI.csv"
TRAIN_EMAX_FILE = "cyp-challenge-TRAIN_Emax.csv"
TRAIN_SINGLE_CONC_FILE = "cyp-challenge-single-concentration-TRAIN.csv"
TEST_BLINDED_FILE = "cyp-challenge-TEST-BLINDED.csv"
DATASET_CARD_FILE = "README.md"

DATA_FILES = [
    TRAIN_INHIBITION_FILE,
    TRAIN_TDI_FILE,
    TRAIN_EMAX_FILE,
    TRAIN_SINGLE_CONC_FILE,
    TEST_BLINDED_FILE,
    DATASET_CARD_FILE,
]


def available_snapshots() -> list[str]:
    """Snapshot dates present in data/raw/, oldest first."""
    if not RAW_DIR.exists():
        return []
    return sorted(p.name for p in RAW_DIR.glob(SNAPSHOT_GLOB) if p.is_dir())


def latest_snapshot() -> str:
    """Most recent snapshot date. Raises if no data has been downloaded."""
    snapshots = available_snapshots()
    if not snapshots:
        raise FileNotFoundError(
            f"No data snapshots in {RAW_DIR}. Run `make data` (or python -m cyp.download) "
            "to fetch the challenge datasets from Hugging Face."
        )
    return snapshots[-1]


def snapshot_dir(snapshot: str | None = None) -> Path:
    """Directory for a given snapshot date, defaulting to the most recent one.

    Pin a specific date when reproducing an old result; leave it None for current work.
    """
    if snapshot is None:
        snapshot = latest_snapshot()
    path = RAW_DIR / snapshot
    if not path.is_dir():
        raise FileNotFoundError(
            f"Snapshot {snapshot!r} not found in {RAW_DIR}. "
            f"Available: {available_snapshots() or 'none'}"
        )
    return path


# ---------------------------------------------------------------------------
# External (non-challenge) data. Kept in its own dated snapshot tree, data/external/,
# parallel to data/raw/ and for the same reason: a public database is a moving target,
# so a result stays reproducible only if the exact extract it used is still on disk.
# ---------------------------------------------------------------------------
EXTERNAL_DIR = DATA_DIR / "external"

# PubChem BioAssay AIDs for the NCGC qHTS cytochrome panel -- one assay per isoform,
# all four run on the same compound library under one protocol. That uniformity is
# why this source was chosen over ChEMBL: ChEMBL carries more CYP IC50s in total
# (13,887 for CYP3A4 against 13,076 CIDs here) but pools them over hundreds of
# unrelated assay protocols, substrates and labs, so a pIC50 there is not comparable
# across rows. PXR's lesson was that heterogeneous auxiliary data hurt; one protocol
# is the version of this experiment that can actually attribute a result.
PUBCHEM_AIDS: dict[str, int] = {
    "CYP1A2": 410,
    "CYP2C9": 883,
    "CYP2D6": 891,
    "CYP3A4": 884,
}

PUBCHEM_FILE_TEMPLATE = "pubchem-qhts-{isoform}.csv"


def external_snapshot_dir(snapshot: str | None = None) -> Path:
    """Directory for an external-data snapshot, defaulting to the most recent one.

    Mirrors `snapshot_dir` but over `data/external/`. Separate from the challenge
    snapshots because the two move independently: OpenADMET amends its release on its
    own schedule, PubChem on its own, and conflating them would make it impossible to
    say which one changed under a result.
    """
    if snapshot is None:
        snapshots = available_external_snapshots()
        if not snapshots:
            raise FileNotFoundError(
                f"No external snapshots in {EXTERNAL_DIR}. Run `make external-data` "
                "(or python -m cyp.external) to fetch the PubChem qHTS panel."
            )
        snapshot = snapshots[-1]
    path = EXTERNAL_DIR / snapshot
    if not path.is_dir():
        raise FileNotFoundError(
            f"External snapshot {snapshot!r} not found in {EXTERNAL_DIR}. "
            f"Available: {available_external_snapshots() or 'none'}"
        )
    return path


def available_external_snapshots() -> list[str]:
    """External snapshot dates present in data/external/, oldest first."""
    if not EXTERNAL_DIR.exists():
        return []
    return sorted(p.name for p in EXTERNAL_DIR.glob(SNAPSHOT_GLOB) if p.is_dir())


ID_COLUMNS = ["SMILES", "Molecule_Name"]

# All four isoforms are scored in the regression track.
ISOFORMS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
# Only these two are scored in the TDI track.
TDI_ISOFORMS = ("CYP3A4", "CYP2D6")

REGRESSION_ENDPOINTS = [f"{cyp}_pIC50_direct_inhibition" for cyp in ISOFORMS]
CLASSIFICATION_ENDPOINTS = [f"{cyp}_is_TDI" for cyp in TDI_ISOFORMS]

CONF_HIGH_SUFFIX = "_conf_high"
CONF_LOW_SUFFIX = "_conf_low"
STD_SUFFIX = "_std"

TEST_SET_SIZE = 750

# Below this pIC50 the assay cannot resolve potency; the official metric downweights
# these compounds. Useful as a floor when clipping predictions.
INACTIVE_PIC50_FLOOR = 4.0

SUBMISSION_ACTIVITY_COLUMNS = ID_COLUMNS + REGRESSION_ENDPOINTS
SUBMISSION_TDI_COLUMNS = ID_COLUMNS + [f"{cyp}_is_TDI" for cyp in TDI_ISOFORMS]

# The single-concentration primary screen was run at one concentration for every
# compound and isoform. Everything auxiliary.py derives from it -- the censoring
# floor, the triage model -- is anchored to this number, so it is named rather than
# repeated as a literal.
SINGLE_CONC_MOLAR = 4.95049505e-05

# Column names in the single-concentration long table.
SCREEN_EFFECT_COLUMN = "log2fc_estimate"
SCREEN_FDR_COLUMN = "log2fc_fdr"
SCREEN_ENZYME_COLUMN = "enzyme"

# Emax columns, both assay conditions. Kept here for completeness; see auxiliary.py
# for why these turn out to carry almost no usable signal.
EMAX_DIRECT_TEMPLATE = "{isoform}_EmaxVsPosCtrl_direct_inhibition"
EMAX_TDI_TEMPLATE = "{isoform}_EmaxVsPosCtrl_TDI_condition"

# pIC50 in the TDI (NADPH-preincubation) condition, and its direct-condition twin.
# The difference between them is the time-dependent shift -- the quantity Emax was
# expected to supply and does not.
PIC50_TDI_TEMPLATE = "{isoform}_pIC50_TDI_condition"
PIC50_DIRECT_TEMPLATE = "{isoform}_pIC50_direct_inhibition"
