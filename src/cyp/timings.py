"""Persistent timing log for notebook runs.

Exists because of a specific failure in `notebooks/03_methods.py`: CV results are
cached, so on the second and every subsequent run the training branch is skipped
and any timer inside it records nothing. The committed run -- the one whose outputs
get read later -- is precisely the run that reports an empty timing table.

So timings are written to a CSV on disk as each measurement is taken, and the cached
path reads back what was recorded when that work was actually done. A method timed
once stays timed.

This matters beyond curiosity. `PLAN.md` budgets days against a deadline, and the
graph models cost hours per endpoint while the tree models cost seconds. Knowing
which is which is what makes "can I afford the full 5x5 on this?" answerable from
the record rather than from a guess.
"""

from __future__ import annotations

from datetime import UTC
from pathlib import Path

import polars as pl

#: Columns every timing row carries. `cached` marks rows replayed from a previous
#: run rather than measured in this one -- without it a reader cannot tell a fast
#: method from a skipped one.
SCHEMA = ("stage", "method", "endpoint", "mode", "seconds", "recorded")


def _empty() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "stage": pl.Utf8,
            "method": pl.Utf8,
            "endpoint": pl.Utf8,
            "mode": pl.Utf8,
            "seconds": pl.Float64,
            "recorded": pl.Utf8,
        }
    )


def load(path: Path | str) -> pl.DataFrame:
    """Read the timing log, or an empty frame with the right schema if absent."""
    path = Path(path)
    if not path.exists():
        return _empty()
    return pl.read_csv(path)


def record(
    path: Path | str,
    stage: str,
    method: str,
    endpoint: str,
    mode: str,
    seconds: float,
) -> None:
    """Append one timing measurement, replacing any previous row for the same key.

    Written immediately rather than accumulated in memory: a run that dies partway
    through -- which is the normal fate of a multi-hour graph-model sweep -- should
    still leave behind the timings for everything that did finish.

    Args:
        path: CSV to append to.
        stage: Which part of the notebook this came from, e.g. "fingerprint_sweep".
        method: Model or featurizer name.
        endpoint: Endpoint or isoform.
        mode: Run mode, e.g. "quick" or "full" -- a quick-mode time is not
            comparable to a full-run one, so it must not overwrite it.
        seconds: Wall-clock duration.
    """
    from datetime import datetime

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row = pl.DataFrame(
        {
            "stage": [stage],
            "method": [method],
            "endpoint": [endpoint],
            "mode": [mode],
            "seconds": [round(float(seconds), 1)],
            "recorded": [datetime.now(UTC).strftime("%Y-%m-%d")],
        }
    )

    existing = load(path)
    if existing.height:
        # Drop any prior row for this exact key so a re-timed method updates rather
        # than accumulating duplicates that would skew a later mean.
        existing = existing.filter(
            ~(
                (pl.col("stage") == stage)
                & (pl.col("method") == method)
                & (pl.col("endpoint") == endpoint)
                & (pl.col("mode") == mode)
            )
        )
        row = pl.concat([existing, row], how="vertical_relaxed")

    row.write_csv(path)


def summary(path: Path | str, mode: str | None = None) -> pl.DataFrame:
    """Per-method totals, slowest first -- the "what will the full run cost" table.

    Args:
        path: The timing CSV.
        mode: Restrict to one run mode. Mixing quick and full times would produce a
            meaningless total, so pass the mode you care about.

    Returns:
        One row per (stage, method): total and mean seconds, and how many endpoints
        contributed. Empty frame if nothing has been recorded yet.
    """
    frame = load(path)
    if not frame.height:
        return frame
    if mode is not None:
        frame = frame.filter(pl.col("mode") == mode)
    if not frame.height:
        return frame

    return (
        frame.group_by(["stage", "method"])
        .agg(
            pl.col("seconds").sum().round(1).alias("total_seconds"),
            pl.col("seconds").mean().round(1).alias("mean_seconds"),
            pl.len().alias("n_endpoints"),
        )
        .with_columns((pl.col("total_seconds") / 60).round(1).alias("total_minutes"))
        .sort("total_seconds", descending=True)
    )


def extrapolate(
    path: Path | str,
    from_mode: str = "quick",
    to_n_outer: int = 5,
    from_n_outer: int = 1,
) -> pl.DataFrame:
    """Project a quick-mode run's cost onto the full protocol.

    CV time is very close to linear in fold count, so a quick run (1 outer repeat)
    times 5 is a decent estimate of the full 5x5. This is the number worth having
    *before* starting a run that might take five hours -- PLAN.md's schedule depends
    on knowing which methods are affordable.

    Treat it as an order-of-magnitude guide: it ignores per-fold fixed costs
    (subprocess launch, checkpoint load) which inflate short runs and are amortized
    in long ones, so it tends to over-estimate slightly for the fast methods.
    """
    frame = summary(path, mode=from_mode)
    if not frame.height:
        return frame
    factor = to_n_outer / from_n_outer
    return frame.with_columns(
        (pl.col("total_seconds") * factor / 60).round(1).alias("projected_full_minutes"),
        (pl.col("total_seconds") * factor / 3600).round(2).alias("projected_full_hours"),
    ).select("stage", "method", "total_seconds", "projected_full_minutes", "projected_full_hours")


#: A unit above this many multiples of its own method's median is contention, not a
#: property of the method. `majority_singletask` -- literally `np.mean` on a boolean
#: array -- once took 3.6s against a 0.1-0.2s baseline (18-36x): a method that does
#: no real computation cannot have a genuine slow fold, so any multiple this large is
#: proof the machine was doing something else. Chosen from that evidence rather than
#: a round number: 3x sits well below every real contention episode seen so far
#: (18x-46x on the trivial baseline, 5x-32x on the timed models) but above the
#: fold-to-fold jitter a healthy run shows.
CONTENTION_MULTIPLE = 3.0

#: A unit need not be slow at all to be worth flagging -- a method with a genuinely
#: short baseline (a tree model at 2s) would trip `CONTENTION_MULTIPLE` on ordinary
#: jitter. This floor means "and it must also be slow in absolute terms", matching
#: the 200s threshold both prior investigations settled on by hand.
CONTENTION_FLOOR_SECONDS = 200.0


def flag_contention(
    path: Path | str,
    mode: str | None = None,
    multiple: float = CONTENTION_MULTIPLE,
    floor_seconds: float = CONTENTION_FLOOR_SECONDS,
) -> pl.DataFrame:
    """Which recorded units look like machine contention rather than real cost.

    Two prior sessions (see `experiments/04_methods_tdi/` and
    `experiments/05_auxiliary_data/`) diagnosed this by hand: watch `timings.csv` as
    it grows, and when a unit is anomalously slow, check `ps` for Spotlight/backup
    activity and `pmset -g therm` for thermal state at that moment. The clearest
    evidence for "this is environmental, not the method" was `majority_singletask` --
    plain `np.mean` on a boolean array -- taking 20-30x its own baseline; a method
    with no real computation cannot have a slow fold for a genuine reason.

    This is the classification half of that pattern, factored out so a fourth
    session does not reconstruct it from a scratchpad script. It answers "which rows
    are contention" from the log alone; live process/thermal snapshotting at
    detection time is `watch_contention.sh` alongside this module, since that part
    needs the shell and cannot be done from a CSV after the fact.

    **Known false positive: a method's one-time pretrain fold.**
    `aux_training.run_cv_pretrained` pays the pretraining cost once, on whichever
    fold runs first for that method, then every later fold is a fine-tune-only fit
    at a fraction of the cost. Nothing in this schema marks which row that was --
    `stage`/`method`/`endpoint`/`seconds` carries no such flag -- so with only a
    handful of folds recorded so far, that one legitimately-slow row inflates the
    25th-percentile baseline and can itself get flagged. It stops mattering once
    enough folds have run that one slow row no longer sits in the lower quartile.
    Read a flagged *fold 0* (or whichever fold ran first) with that in mind rather
    than assuming it is contention; every other flagged row is not subject to this.

    Args:
        path: The timing CSV.
        mode: Restrict to one run mode, as in `summary`.
        multiple: How many times a method's own baseline counts as an outlier.
        floor_seconds: A unit must also exceed this in absolute terms -- guards a
            fast method (a tree model at 2s) from tripping the multiple on ordinary
            jitter.

    Returns:
        One row per flagged unit, with its method's baseline for context and
        `ratio = seconds / baseline`. Empty if nothing was recorded or nothing
        qualifies -- an empty result is the common, healthy case, not a failure.
    """
    frame = load(path)
    if mode is not None:
        frame = frame.filter(pl.col("mode") == mode)
    if not frame.height:
        return frame.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("baseline_seconds"),
            pl.lit(None, dtype=pl.Float64).alias("ratio"),
        )

    # The 25th percentile, not the median, is the baseline. A method's very first
    # fold is a real, structural cost in the pretrained arms this was written for
    # (`aux_training.run_cv_pretrained` pays for the one-time pretrain on fold 0,
    # not the fold's own fit), and that pattern is one-sided: pretraining and
    # contention both make a unit slower than steady-state, never faster. A median
    # already has that slow outlier pulled into it once there are only a handful of
    # folds, which drags every OTHER fold's ratio down and can hide real contention.
    # The lower quartile is contaminated only once at least a quarter of a method's
    # folds are themselves slow -- by which point "contention" is the wrong word for
    # what is happening anyway.
    with_baseline = frame.with_columns(
        pl.col("seconds").quantile(0.25, interpolation="lower").over(["stage", "method"]).alias(
            "baseline_seconds"
        )
    )
    return (
        with_baseline.filter(
            (pl.col("seconds") >= floor_seconds)
            & (pl.col("seconds") >= multiple * pl.col("baseline_seconds"))
        )
        .with_columns((pl.col("seconds") / pl.col("baseline_seconds")).round(1).alias("ratio"))
        .sort("ratio", descending=True)
    )
