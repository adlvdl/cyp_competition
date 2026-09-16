#!/usr/bin/env bash
# Tail a notebook's timings.csv and snapshot system state the moment a unit looks
# like machine contention rather than real cost.
#
# Reconstructs, as a committed script rather than a scratchpad one-off, what two
# prior sessions each built by hand while watching notebooks/04_methods_tdi.py and
# notebooks/05_auxiliary_data.py: a live monitor over the timing log, and on an
# anomalous unit, `ps` for Spotlight/backup activity plus `pmset -g therm` for
# thermal state at that exact moment. Waiting until after the run to investigate
# does not work -- by the time a session asks "what caused that", the responsible
# process (if any) has usually already exited, and thermal budget logs roll over.
# The check has to fire while the anomaly is live.
#
# Written for bash 3.2 (macOS ships nothing newer, and this machine has no
# Homebrew bash either) -- no associative arrays, no `mapfile`. Per-method history
# lives in one file per key under a scratch directory instead.
#
# What "contention" means here is defined in `cyp.timings.flag_contention`: a unit
# taking more than CONTENTION_MULTIPLE (3x) its own method's 25th-percentile
# baseline, and at least CONTENTION_FLOOR_SECONDS (200s) in absolute terms. The
# floor exists so a fast tree model's ordinary jitter doesn't trip the multiple;
# the multiple exists so a genuinely slow method (chemprop) doesn't trip the floor
# on every fold. This script uses a running median (cheaper to keep updated as a
# growing file than a percentile) as a close approximation for the *live* check --
# `flag_contention` on the finished CSV is the version to trust for the record.
#
# Usage:
#   scripts/watch_contention.sh path/to/timings.csv [snapshot_dir]
#
# Runs until 30 minutes pass with no new line -- matched to the harness's own fold
# cadence rather than a fixed duration, since a run can take minutes or hours
# depending on the method.
#
# Each snapshot is a plain text file capturing, at the moment of detection:
#   - the flagged row itself (stage, method, fold, seconds, ratio to that method's
#     running median)
#   - top CPU consumers system-wide
#   - Spotlight/backup/photo-analysis process activity specifically, since these are
#     the two causes both prior investigations actually confirmed (see CLAUDE.md and
#     experiments/04_methods_tdi/, experiments/05_auxiliary_data/): mdworker CPU use
#     correlating with a spike in one session, and a depressed kernel thermal power
#     budget with no competing process at all in another. Neither cause is
#     universal, so both are always checked.
#   - `pmset -g therm`, for whatever thermal state the OS is willing to report
#     (frequently nothing -- macOS does not expose this reliably from userspace,
#     which is why the kernel log grep below is the more informative half)
#   - a kernel-log grep for `ApplePassthroughPPM.*ThermalPowerBudget` over the last
#     10 minutes, which is what actually caught the thermal case: a depressed
#     "Thermal Budget" number in that log line, even when `pmset -g therm` itself
#     reported nothing
#
# A snapshot is evidence to read afterward, not an action taken automatically --
# this script does not kill, renice, or throttle anything. Deciding whether a
# flagged unit should be excluded from a paired comparison is `flag_contention`'s
# job on the finished CSV; this script only makes sure the *cause* is not lost by
# the time anyone looks.

set -euo pipefail

TIMING_CSV="${1:?usage: watch_contention.sh path/to/timings.csv [snapshot_dir]}"
SNAPSHOT_DIR="${2:-$(dirname "$TIMING_CSV")/contention_snapshots}"
IDLE_TIMEOUT_S="${IDLE_TIMEOUT_S:-1800}"   # stop if this many seconds pass with no new row
CONTENTION_FLOOR_S=200
CONTENTION_MULTIPLE=3

mkdir -p "$SNAPSHOT_DIR"
HISTORY_DIR="$SNAPSHOT_DIR/.history"
mkdir -p "$HISTORY_DIR"

if [ ! -f "$TIMING_CSV" ]; then
    echo "waiting for $TIMING_CSV to exist..." >&2
    until [ -f "$TIMING_CSV" ]; do sleep 2; done
fi

last_size=$(wc -l < "$TIMING_CSV" | tr -d ' ')
last_change=$(date +%s)

snapshot() {
    stage="$1"; method="$2"; fold="$3"; seconds="$4"; median="$5"; ratio="$6"
    stamp=$(date +%Y%m%d_%H%M%S)
    out="$SNAPSHOT_DIR/slow_${method}_${fold}_${stamp}.txt"

    {
        echo "=== flagged unit ==="
        echo "stage=$stage method=$method fold=$fold seconds=$seconds median=$median ratio=${ratio}x"
        echo "detected: $(date)"
        echo
        echo "=== top CPU consumers ==="
        # || true matters here, not decoration: under `set -eo pipefail`, `head -15`
        # closing its read end early sends `ps` a SIGPIPE, and pipefail turns that
        # broken-pipe exit into the whole pipeline's status -- which then kills the
        # entire script on its very next flagged row, silently, with no error text.
        # This one line was the actual cause of two earlier silent mid-run deaths;
        # every other piped command in this function already had a guard, this
        # was the one that didn't.
        ps -Ao pid,ppid,%cpu,%mem,etime,comm -r 2>/dev/null | head -15 || true
        echo
        echo "=== spotlight / backup / photo-analysis activity ==="
        ps aux 2>/dev/null | grep -iE "mds|mdworker|backupd|photoanalysis" | grep -v grep \
            || echo "(none running)"
        echo
        echo "=== pmset thermal state ==="
        pmset -g therm 2>/dev/null || echo "(pmset -g therm returned nothing)"
        echo
        echo "=== kernel thermal power budget, last 10 min ==="
        echo "(a depressed 'Thermal Budget' number here caught a contention episode"
        echo " that showed no competing process and nothing from pmset -g therm)"
        log show --last 10m \
            --predicate 'eventMessage contains "ThermalPowerBudget" or eventMessage contains "Thermal Budget"' \
            2>/dev/null | tail -20 || echo "(log show unavailable or empty)"
        echo
        echo "=== memory pressure ==="
        memory_pressure 2>/dev/null | tail -3 || true
    } > "$out"

    echo "SLOW  $method $fold: ${seconds}s  (${ratio}x median ${median}s)  snapshot: $out"
}

echo "watching $TIMING_CSV (floor=${CONTENTION_FLOOR_S}s, multiple=${CONTENTION_MULTIPLE}x)" >&2

while true; do
    size=$(wc -l < "$TIMING_CSV" | tr -d ' ')
    if [ "$size" -gt "$last_size" ]; then
        last_change=$(date +%s)
        tail -n $((size - last_size)) "$TIMING_CSV" | while IFS=, read -r stage method endpoint mode seconds recorded; do
            [ "$stage" = "stage" ] && continue  # header, if tail caught it
            [ -z "$stage" ] && continue

            # One history file per (stage, method): a newline-separated list of every
            # `seconds` value seen for that key. Its running median is this script's
            # baseline. Bash 3.2 has no associative arrays, so a file is the portable
            # substitute for a hash keyed on "$stage|$method".
            key=$(echo "${stage}_${method}" | tr -c 'A-Za-z0-9_' '_')
            hist_file="$HISTORY_DIR/$key"

            # Median of everything seen BEFORE this row, not including it -- comparing
            # a value against a baseline that already contains itself can never flag
            # a first-ever measurement (900s against a median of exactly 900s is not
            # >= 3x itself), which would silently miss real contention on a method's
            # very first recorded fold.
            if [ -f "$hist_file" ]; then
                median=$(sort -n "$hist_file" | awk '
                    { a[NR]=$1 } END {
                        if (NR==0) { print 0; exit }
                        if (NR%2==1) print a[(NR+1)/2]; else print (a[NR/2]+a[NR/2+1])/2
                    }')
            else
                median=0
            fi
            echo "$seconds" >> "$hist_file"

            is_slow=$(awk -v s="$seconds" -v m="$median" -v floor="$CONTENTION_FLOOR_S" \
                -v mult="$CONTENTION_MULTIPLE" \
                'BEGIN { print (s>=floor && m>0 && s>=mult*m) ? 1 : 0 }')

            if [ "$is_slow" = "1" ]; then
                ratio=$(awk -v s="$seconds" -v m="$median" 'BEGIN { printf "%.1f", s/m }')
                snapshot "$stage" "$method" "$endpoint" "$seconds" "$median" "$ratio"
            fi
        done
        last_size=$size
    fi

    now=$(date +%s)
    if [ $((now - last_change)) -ge "$IDLE_TIMEOUT_S" ]; then
        echo "no new rows for ${IDLE_TIMEOUT_S}s, stopping" >&2
        break
    fi
    sleep 15
done
