"""Tests for the persistent timing log.

The bug this module exists to prevent is specific and easy to reintroduce: CV output
is cached, so on every run after the first the training branch is skipped and a timer
living inside it records nothing. The committed run is exactly the run that would
report an empty table. So the tests that matter here are about *persistence* across
runs, not about measuring elapsed time correctly.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from cyp import timings


def test_load_returns_typed_empty_frame_when_absent(tmp_path: Path) -> None:
    """An absent log must still have the right columns, so a caller can filter and
    group it without branching on emptiness."""
    frame = timings.load(tmp_path / "nothing.csv")
    assert frame.height == 0
    assert set(frame.columns) == set(timings.SCHEMA)


def test_record_then_load_roundtrip(tmp_path: Path) -> None:
    log = tmp_path / "t.csv"
    timings.record(log, "new_models", "chemprop", "CYP3A4", "full", 305.2)
    frame = timings.load(log)
    assert frame.height == 1
    row = frame.row(0, named=True)
    assert row["method"] == "chemprop"
    assert row["seconds"] == pytest.approx(305.2)


def test_timings_survive_a_cached_rerun(tmp_path: Path) -> None:
    """The whole point of the module. Simulates: run once (measures and records),
    then run again with the cache warm (measures nothing). The second run must still
    be able to report what the first one cost."""
    log = tmp_path / "t.csv"

    # First run: cache cold, work is done and timed.
    timings.record(log, "new_models", "chemprop", "CYP3A4", "full", 300.0)
    timings.record(log, "new_models", "tabicl", "CYP3A4", "full", 33.0)

    # Second run: cache warm, nothing is timed -- no record() calls at all.
    summary = timings.summary(log, mode="full")

    assert summary.height == 2
    methods = set(summary["method"].to_list())
    assert methods == {"chemprop", "tabicl"}


def test_retiming_replaces_rather_than_duplicates(tmp_path: Path) -> None:
    """Deleting a cache file and re-running must update that method's time, not
    append a second row -- duplicates would silently double the reported total."""
    log = tmp_path / "t.csv"
    timings.record(log, "new_models", "chemprop", "CYP3A4", "full", 300.0)
    timings.record(log, "new_models", "chemprop", "CYP3A4", "full", 250.0)

    frame = timings.load(log)
    assert frame.height == 1
    assert frame.row(0, named=True)["seconds"] == pytest.approx(250.0)


def test_modes_do_not_overwrite_each_other(tmp_path: Path) -> None:
    """A quick-mode time is not comparable to a full-run time, so the two must
    coexist rather than one clobbering the other."""
    log = tmp_path / "t.csv"
    timings.record(log, "new_models", "chemprop", "CYP3A4", "quick", 60.0)
    timings.record(log, "new_models", "chemprop", "CYP3A4", "full", 300.0)

    assert timings.load(log).height == 2
    assert timings.summary(log, mode="quick").row(0, named=True)[
        "total_seconds"
    ] == pytest.approx(60.0)
    assert timings.summary(log, mode="full").row(0, named=True)[
        "total_seconds"
    ] == pytest.approx(300.0)


def test_summary_totals_across_endpoints(tmp_path: Path) -> None:
    """A method's cost is the sum over endpoints -- that is the number that answers
    'can I afford to run this'."""
    log = tmp_path / "t.csv"
    for endpoint, seconds in [("CYP3A4", 300.0), ("CYP1A2", 200.0), ("CYP2D6", 100.0)]:
        timings.record(log, "new_models", "chemprop", endpoint, "full", seconds)

    row = timings.summary(log, mode="full").row(0, named=True)
    assert row["total_seconds"] == pytest.approx(600.0)
    assert row["mean_seconds"] == pytest.approx(200.0)
    assert row["n_endpoints"] == 3
    assert row["total_minutes"] == pytest.approx(10.0)


def test_summary_sorts_slowest_first(tmp_path: Path) -> None:
    """The expensive method is the one you need to see, so it goes on top."""
    log = tmp_path / "t.csv"
    timings.record(log, "fingerprint_sweep", "ecfp", "CYP3A4", "full", 10.0)
    timings.record(log, "new_models", "chemprop", "CYP3A4", "full", 300.0)

    assert timings.summary(log, mode="full")["method"].to_list()[0] == "chemprop"


def test_extrapolate_scales_by_fold_ratio(tmp_path: Path) -> None:
    """Quick mode runs 1 outer repeat; the full protocol runs 5. CV cost is close to
    linear in folds, so the projection is a 5x multiply."""
    log = tmp_path / "t.csv"
    timings.record(log, "new_models", "chemprop", "CYP3A4", "quick", 360.0)

    row = timings.extrapolate(log, from_mode="quick", to_n_outer=5, from_n_outer=1).row(
        0, named=True
    )
    assert row["projected_full_minutes"] == pytest.approx(30.0)
    assert row["projected_full_hours"] == pytest.approx(0.5)


def test_extrapolate_is_empty_without_quick_data(tmp_path: Path) -> None:
    """No quick-mode rows means no basis for a projection -- return empty rather
    than a confidently wrong zero."""
    log = tmp_path / "t.csv"
    timings.record(log, "new_models", "chemprop", "CYP3A4", "full", 300.0)
    assert timings.extrapolate(log, from_mode="quick").height == 0


def test_record_creates_parent_directories(tmp_path: Path) -> None:
    """A first run writes into an experiment directory that may not exist yet."""
    log = tmp_path / "experiments" / "03_methods" / "timings.csv"
    timings.record(log, "new_models", "tabicl", "CYP3A4", "full", 33.0)
    assert log.exists()


def test_summary_of_empty_log_is_empty(tmp_path: Path) -> None:
    summary = timings.summary(tmp_path / "nothing.csv")
    assert isinstance(summary, pl.DataFrame)
    assert summary.height == 0
