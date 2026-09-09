"""Fetch the challenge datasets from Hugging Face into a dated snapshot directory.

Each run writes to ``data/raw/YYYYMMDD/``. OpenADMET has amended the dataset
mid-challenge, so snapshots are kept side by side rather than overwritten: after a
re-download, `compare` reports exactly which files changed against the previous
snapshot, and the loaders default to the most recent one.

    python -m cyp.download                  # download to today's date
    python -m cyp.download --date 20260401  # write to a specific snapshot
    python -m cyp.download --compare-only   # diff the two latest, download nothing
"""

from __future__ import annotations

import hashlib
import urllib.request
from datetime import date
from pathlib import Path

from . import constants as C

REPO = "openadmet/cyp-challenge-train-test"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(snapshot: str | None = None, force: bool = False) -> Path:
    """Download every dataset file into ``data/raw/<snapshot>/``.

    Args:
        snapshot: Target date as ``YYYYMMDD``. Defaults to today.
        force: Re-fetch files that are already present in that snapshot.

    Returns:
        The snapshot directory.
    """
    snapshot = snapshot or date.today().strftime("%Y%m%d")
    dest_dir = C.RAW_DIR / snapshot
    dest_dir.mkdir(parents=True, exist_ok=True)

    for name in C.DATA_FILES:
        dest = dest_dir / name
        if dest.exists() and not force:
            print(f"skip   {name}  (already in {snapshot})")
            continue
        urllib.request.urlretrieve(f"{BASE_URL}/{name}", dest)
        print(f"got    {name}  ({dest.stat().st_size:,} bytes)")

    print(f"\nSnapshot written to {dest_dir}")
    return dest_dir


def compare(old: str, new: str) -> dict[str, str]:
    """Compare two snapshots file by file, by content hash.

    Returns a mapping of filename to one of "unchanged", "CHANGED", "added",
    or "removed".
    """
    old_dir, new_dir = C.snapshot_dir(old), C.snapshot_dir(new)
    result: dict[str, str] = {}
    for name in C.DATA_FILES:
        old_file, new_file = old_dir / name, new_dir / name
        if not old_file.exists() and new_file.exists():
            result[name] = "added"
        elif old_file.exists() and not new_file.exists():
            result[name] = "removed"
        elif not old_file.exists() and not new_file.exists():
            continue
        else:
            same = _sha256(old_file) == _sha256(new_file)
            result[name] = "unchanged" if same else "CHANGED"
    return result


def report_against_previous(snapshot: str | None = None) -> None:
    """Print a comparison of `snapshot` against the snapshot immediately before it."""
    snapshots = C.available_snapshots()
    snapshot = snapshot or (snapshots[-1] if snapshots else None)
    if snapshot is None:
        print("No snapshots to compare.")
        return
    older = [s for s in snapshots if s < snapshot]
    if not older:
        print(f"{snapshot} is the only snapshot -- nothing to compare against.")
        return

    previous = older[-1]
    print(f"\n{previous} -> {snapshot}")
    changes = compare(previous, snapshot)
    for name, status in changes.items():
        marker = " " if status == "unchanged" else "*"
        print(f" {marker} {status:<10} {name}")
    if any(s != "unchanged" for s in changes.values()):
        print(
            "\nThe dataset changed. Re-run `make test` and check whether the "
            "row counts and endpoint names in README.md / constants.py still hold."
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", help="snapshot date YYYYMMDD (default: today)")
    parser.add_argument(
        "--force", action="store_true", help="re-fetch files already in the snapshot"
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="compare the two most recent snapshots without downloading",
    )
    args = parser.parse_args()

    if args.compare_only:
        report_against_previous(args.date)
    else:
        download(args.date, force=args.force)
        report_against_previous(args.date)
