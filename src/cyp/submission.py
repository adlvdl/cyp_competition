"""Build and validate the two submission CSVs.

Polars internally; the vendored official validators read the written file with Pandas
themselves, so the boundary stays inside `vendor/`. A file that passes `check` is the
same file the challenge Space will accept.

Always validate before uploading -- the leaderboard allows one submission per team.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

from . import constants as C
from .data import load_test

sys.path.insert(0, str(C.PROJECT_ROOT / "vendor" / "cyp_challenge_tutorial"))
from validation.activity_validation import validate_activity_submission  # noqa: E402
from validation.tdi_validation import validate_tdi_submission  # noqa: E402


def _test_frame(snapshot: str | None = None) -> pl.DataFrame:
    return load_test(snapshot).select("SMILES", "Molecule_Name")


def _attach(
    base: pl.DataFrame,
    predictions: dict[str, np.ndarray] | pl.DataFrame,
    columns: list[str],
) -> pl.DataFrame:
    """Attach prediction columns to the test frame, by join or by position."""
    if isinstance(predictions, pl.DataFrame):
        return base.join(
            predictions.select("Molecule_Name", *columns), on="Molecule_Name", how="left"
        )

    out = base
    for column in columns:
        if column not in predictions:
            raise KeyError(f"Missing predictions for {column}")
        values = np.asarray(predictions[column])
        if len(values) != base.height:
            raise ValueError(
                f"{column}: got {len(values)} predictions, expected {base.height}"
            )
        out = out.with_columns(pl.Series(column, values))
    return out


def build_activity_submission(
    predictions: dict[str, np.ndarray] | pl.DataFrame,
    out_path: Path | str,
    snapshot: str | None = None,
) -> Path:
    """Write a direct-inhibition submission.

    `predictions` maps each `{CYP}_pIC50_direct_inhibition` endpoint to 750 values in
    the blinded test set's row order, or is a DataFrame keyed by `Molecule_Name`.
    """
    base = _test_frame(snapshot)
    merged = _attach(base, predictions, C.REGRESSION_ENDPOINTS)

    nulls = {c: merged[c].null_count() for c in C.REGRESSION_ENDPOINTS}
    if any(nulls.values()):
        raise ValueError(f"Submission has missing predictions: {nulls}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.select(C.SUBMISSION_ACTIVITY_COLUMNS).write_csv(out_path)
    return out_path


def build_tdi_submission(
    predictions: dict[str, np.ndarray] | pl.DataFrame,
    out_path: Path | str,
    snapshot: str | None = None,
) -> Path:
    """Write a TDI submission. Values are cast to bool -- submit hard labels, since
    MCC is computed on the labels as written."""
    base = _test_frame(snapshot)
    columns = [f"{cyp}_is_TDI" for cyp in C.TDI_ISOFORMS]
    merged = _attach(base, predictions, columns).with_columns(
        [pl.col(c).cast(pl.Boolean) for c in columns]
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.select(C.SUBMISSION_TDI_COLUMNS).write_csv(out_path)
    return out_path


def _expected_ids(snapshot: str | None = None) -> set[str]:
    return set(load_test(snapshot)["Molecule_Name"].cast(pl.Utf8).to_list())


def validate_activity(path: Path | str) -> tuple[bool, list[str]]:
    return validate_activity_submission(Path(path), expected_ids=_expected_ids())


def validate_tdi(path: Path | str) -> tuple[bool, list[str]]:
    return validate_tdi_submission(Path(path), expected_ids=_expected_ids())


def check(path: Path | str, track: str) -> bool:
    """Validate and print the result. `track` is "activity" or "tdi"."""
    ok, errors = (validate_activity if track == "activity" else validate_tdi)(path)
    if ok:
        print(f"PASS  {path} ({track})")
    else:
        print(f"FAIL  {path} ({track})")
        for err in errors:
            print(f"  - {err}")
    return ok


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate a submission CSV.")
    parser.add_argument("path")
    parser.add_argument("--track", choices=["activity", "tdi"], required=True)
    args = parser.parse_args()
    raise SystemExit(0 if check(args.path, args.track) else 1)
