"""Which challenge compounds appear, with CYP labels, in Monroe's PCBA pretraining corpus?

Monroe pretrains on 1.56M PCBA compounds, and that corpus contains the Veith CYP panel
(AIDs 410, 883, 884, 891, 899, 1851). A challenge compound present there has had its
inhibition label seen during pretraining, which would inflate any CV score on it.

Matching is on the InChIKey connectivity block (`external.inchikey_skeleton`), the repo's
leakage key. Writes experiments/16_encoders/monroe_pcba_overlap.csv: one row per
challenge compound found, plus the blind-set count on stdout.

    uv run python scripts/audit_monroe_pcba_overlap.py PCBA_1328_1564k.parquet

The parquet is in pcba_1328.zip at https://zenodo.org/records/8024997 (153 MB).
"""

import sys
from pathlib import Path

import polars as pl

import cyp  # noqa: F401  (claims LightGBM's OpenMP runtime first)
from cyp import constants as C
from cyp import data
from cyp.external import inchikey_skeleton

CYP_AIDS = [
    "assayID-410",
    "assayID-883",
    "assayID-884",
    "assayID-891",
    "assayID-899",
    "assayID-1851",
]


def _skeleton(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.col("SMILES").map_elements(inchikey_skeleton, return_dtype=pl.String).alias("sk")
    )


if __name__ == "__main__":
    pcba = pl.scan_parquet(sys.argv[1]).select(["SMILES", *CYP_AIDS]).collect()
    # Only rows carrying a CYP label matter; keying the other 1.5M is wasted work.
    pcba = _skeleton(
        pcba.filter(pl.any_horizontal([pl.col(c).is_not_null() for c in CYP_AIDS]))
    ).unique("sk")

    train = _skeleton(
        pl.concat(
            [
                data.training_frame(e).select("Molecule_Name", "SMILES")
                for e in C.REGRESSION_ENDPOINTS
            ]
        ).unique("Molecule_Name")
    )
    test = _skeleton(pl.read_csv(next(Path("data/raw").glob("*/cyp-challenge-TEST-BLINDED.csv"))))

    hit = train.join(pcba.drop("SMILES"), on="sk", how="inner")
    out = Path("experiments/16_encoders/monroe_pcba_overlap.csv")
    hit.drop("sk").write_csv(out)
    blind = test.filter(pl.col("sk").is_in(set(pcba["sk"].to_list()))).height
    print(f"train {hit.height}/{train.height} compounds carry a PCBA CYP label")
    print(f"blind {blind}/{test.height}")
    print(f"wrote {out}")
