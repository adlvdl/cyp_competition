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

# Efficacy at the top tested concentration, as a percent change from control, is
# recorded for *every* screened compound -- unlike pIC50, which only exists where a
# curve fitted (~50% of rows). That makes it the qHTS panel's only readout carrying
# the inactive half of the library, which is exactly the low-activity region the
# challenge's hit-enriched labels never reach. Inhibition is negative.
#
# The clip window is fixed rather than percentile-based, and deliberately so: the
# positive tail differs by two orders of magnitude across isoforms (CYP3A4 reaches
# +669, CYP2D6 only +106), so a percentile cut would keep CYP3A4's artifacts while
# removing CYP2D6's real signal. Complete inhibition is -100, so -150 leaves a noise
# margin; anything above +50 is a compound "activating" the enzyme, which at this
# scale is fluorescence interference rather than efficacy.
MAX_RESPONSE_CLIP = (-150.0, 50.0)

# Tox21's CYP panel, deposited by NCGC as luciferase cell-based P450-Glo assays.
# A different detection chemistry and a different compound library from the Veith
# qHTS panel above, so it carries its own scale and belongs in its own head rather
# than merged with Veith's numbers.
#
# It is here for *range*, not volume. Its actives are weak ones where the challenge
# set is hit-enriched and ChEMBL assigns a pchembl only where a curve fitted, so it
# covers the low-potency region both other sources miss.
#
# There is no CYP1A2 assay in this deposition batch. That absence is carried rather
# than substituted: filling it from a different protocol would put two incomparable
# scales in one column, which is the failure this per-source head layout exists to
# avoid.
TOX21_AIDS: dict[str, int] = {
    "CYP2C9": 1645842,
    "CYP2D6": 1645840,
    "CYP3A4": 1645841,
}

TOX21_FILE_TEMPLATE = "tox21-p450glo-{isoform}.csv"

# Bioluminescent CYP readouts score firefly-luciferase inhibitors as CYP inhibitors,
# because the reporter is the thing being inhibited. Tox21 runs a counter-screen for
# exactly this, and its call has to be honoured or the head learns luciferase
# chemistry. Rows whose phenotype is an activator are dropped for the same reason
# activators are dropped from the Veith panel.
TOX21_LUCIFERASE_COUNTERSCREEN_AID = 1224835

# ChEMBL target IDs for the five isoforms. CYP2C19 is not scored by the challenge but
# the assays measure it anyway, so it rides along as a correlated auxiliary task.
#
# ChEMBL is the heterogeneous source: these activities are pooled across hundreds of
# unrelated protocols, substrates and labs, and two rows for one compound routinely
# disagree by more than a log unit. PXR found heterogeneous auxiliary data actively
# harmful, which is why the qHTS panel was preferred for pretraining. ChEMBL earns a
# place here only as its own separate head -- the encoder sees the chemistry, and the
# head absorbs the scale disagreement rather than pushing it into a scored column.
#
# What it buys is new chemistry rather than new labels: ~24,900 skeletons with almost
# no overlap against the challenge deck and none against the blind set.
CHEMBL_TARGETS: dict[str, str] = {
    "CYP1A2": "CHEMBL3356",
    "CYP2C9": "CHEMBL3397",
    "CYP2D6": "CHEMBL289",
    "CYP3A4": "CHEMBL340",
    "CYP2C19": "CHEMBL3622",
}

CHEMBL_FILE_TEMPLATE = "chembl-{isoform}.csv"

#: ChEMBL assigns a `pchembl_value` only where a concentration-response curve was
#: fitted, so nothing sits below this -- the same double selection that makes the
#: challenge labels hit-enriched. Public potency buys ranking on new chemistry, not
#: low-end range; range is what Tox21 is for.
CHEMBL_PCHEMBL_FLOOR = 4.0


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
