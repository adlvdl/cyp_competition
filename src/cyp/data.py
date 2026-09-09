"""Loaders for the five challenge CSVs.

Polars throughout (see CLAUDE.md). Each loader returns the raw table with minimal
massaging so nothing is silently dropped. Targets are sparse: every isoform column is
null for most rows, because a compound only has a dose-response curve where the
primary screen flagged it. Filter per-endpoint at training time with
`labelled_subset`.

Every loader takes an optional `snapshot` date (``"YYYYMMDD"``) and defaults to the
most recent one in ``data/raw/``. Pin a date when reproducing an older result.
"""

from __future__ import annotations

import polars as pl

from . import constants as C


def _read(filename: str, snapshot: str | None = None) -> pl.DataFrame:
    path = C.snapshot_dir(snapshot) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `make data` (or python -m cyp.download) to fetch "
            "the challenge datasets from Hugging Face."
        )
    return pl.read_csv(path)


def load_train_inhibition(snapshot: str | None = None) -> pl.DataFrame:
    """Direct-inhibition dose-response training set (4905 rows).

    Columns: identifiers, one pIC50 per isoform, plus `_conf_low`/`_conf_high`
    (credible interval bounds used by the official ST-RAE metric) and `_std`.
    """
    return _read(C.TRAIN_INHIBITION_FILE, snapshot)


def load_train_tdi(snapshot: str | None = None) -> pl.DataFrame:
    """TDI training set (6145 rows).

    Carries the boolean `{CYP}_is_TDI` labels for CYP3A4/CYP2D6 alongside pIC50 in
    both the TDI (NADPH preincubation) and direct conditions, so the 2-fold shift
    can be recomputed or modelled directly.
    """
    return _read(C.TRAIN_TDI_FILE, snapshot)


def load_train_emax(snapshot: str | None = None) -> pl.DataFrame:
    """Emax (maximum effect vs positive control) for both assay conditions."""
    return _read(C.TRAIN_EMAX_FILE, snapshot)


def load_single_concentration(snapshot: str | None = None) -> pl.DataFrame:
    """Long-format primary screen (17504 rows, one row per compound x enzyme x plate).

    This is the widest-coverage data in the challenge and covers compounds that never
    got a dose-response curve -- useful as auxiliary signal.

    A caution from PXR: multitask and pretraining setups built on single-dose and
    counter-screen data consistently *hurt* there, even though top teams extracted
    value from the same sources. Treat this as promising but unproven, and validate
    against a model that ignores it.
    """
    return _read(C.TRAIN_SINGLE_CONC_FILE, snapshot)


def load_test(snapshot: str | None = None) -> pl.DataFrame:
    """Blinded test set: 750 rows, identifiers only."""
    test = _read(C.TEST_BLINDED_FILE, snapshot)
    if test.height != C.TEST_SET_SIZE:
        raise ValueError(f"Expected {C.TEST_SET_SIZE} test rows, got {test.height}")
    return test


def labelled_subset(df: pl.DataFrame, endpoint: str) -> pl.DataFrame:
    """Rows of `df` that actually have a value for `endpoint`.

    Also carries the credible-interval columns through when present, since ST-RAE
    needs them.
    """
    if endpoint not in df.columns:
        raise KeyError(f"{endpoint!r} not in dataframe; have {df.columns[:8]}...")
    return df.filter(pl.col(endpoint).is_not_null())


def endpoint_counts(df: pl.DataFrame, endpoints: list[str]) -> pl.DataFrame:
    """How many labelled rows exist per endpoint -- worth checking before training."""
    return pl.DataFrame(
        {
            "endpoint": [e for e in endpoints if e in df.columns],
            "n_labelled": [
                int(df[e].is_not_null().sum()) for e in endpoints if e in df.columns
            ],
        }
    )


def training_frame(endpoint: str, snapshot: str | None = None) -> pl.DataFrame:
    """Model-ready frame for one regression endpoint.

    Returns identifiers, the target as `y_true`, and the credible-interval bounds as
    `y_lower`/`y_upper` -- the schema `cv`, `evaluation` and `metrics` expect.
    """
    train = load_train_inhibition(snapshot)
    subset = labelled_subset(train, endpoint)
    return subset.select(
        pl.col("Molecule_Name"),
        pl.col("SMILES"),
        pl.col(endpoint).alias("y_true"),
        pl.col(endpoint + C.CONF_LOW_SUFFIX).alias("y_lower"),
        pl.col(endpoint + C.CONF_HIGH_SUFFIX).alias("y_upper"),
    )


def tdi_training_frame(isoform: str, snapshot: str | None = None) -> pl.DataFrame:
    """Model-ready frame for one TDI classification endpoint (CYP3A4 or CYP2D6)."""
    if isoform not in C.TDI_ISOFORMS:
        raise ValueError(f"TDI is scored for {C.TDI_ISOFORMS}, not {isoform!r}")
    column = f"{isoform}_is_TDI"
    tdi = labelled_subset(load_train_tdi(snapshot), column)
    return tdi.select(
        pl.col("Molecule_Name"),
        pl.col("SMILES"),
        pl.col(column).cast(pl.Boolean).alias("y_true"),
    )
