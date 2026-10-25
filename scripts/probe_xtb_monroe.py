"""Does Monroe's embedding respond to Shorku's xTB conformers, beyond its own seed noise?

Monroe embeds one random ETKDG+MMFF94s conformer per molecule (`cyp.encoders`). The
xTB ensembles in `cyp.shorku_qm` would only be worth a full CV run if swapping in that
geometry moves the embedding by more than re-rolling the ETKDG seed does, or if the
seed noise itself is large enough that averaging over an ensemble could remove it.

Arms per compound, all built from the same upstream structure SMILES so only the
coordinates differ:

- `etkdg_a`, `etkdg_b`: Monroe's own recipe (ETKDGv3, random coords, MMFF94s, UFF
  fallback), seeds 1 and 2.
- `xtb_min`: the lowest-energy xTB conformer.
- `xtb_boltz`: Boltzmann-weighted mean embedding over the five highest-weight conformers.
- `cached` (training compounds only): `16`'s embedding, built by Monroe's InChI route.
  Compared against `etkdg_a` to check that the coordinate-injection route featurizes the
  same molecule the same way.

Distances are Euclidean after centring on the mean of all embeddings, reported in units
of the median distance between two *different* molecules' `etkdg_a` embeddings, so 1.0
means "as far apart as two unrelated compounds".

Decision rule, fixed before running: run the full CV (A2) if seed noise is at least
0.15 of the between-molecule distance (averaging has something to remove), or if the
xTB shift is at least 1.5x the seed noise (the geometry carries information ETKDG does
not). Otherwise stop.

Writes experiments/25_xtb_conformer_probe/{per_compound.csv,summary.csv}.

    uv run python scripts/probe_xtb_monroe.py
"""

from pathlib import Path

import numpy as np
import polars as pl
from rdkit import RDLogger

import cyp  # noqa: F401  (claims LightGBM's OpenMP runtime first)
from cyp import constants as C
from cyp import data, encoders
from cyp import shorku_qm as Q

RDLogger.DisableLog("rdApp.*")

OUT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "25_xtb_conformer_probe"
CACHED_MONROE = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "16_encoders"
    / "cache"
    / "monroe_584ae37899ec.npy"
)
N_PER_SET = 150
TOP_CONFORMERS = 5
SEED = 0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # The cache is keyed on this exact sorted union (see notebooks/16_encoders.py).
    union = sorted(
        {s for e in C.REGRESSION_ENDPOINTS for s in data.training_frame(e)["SMILES"].to_list()}
    )
    cached = np.load(CACHED_MONROE)
    assert cached.shape[0] == len(union)
    cached_row = {s: i for i, s in enumerate(union)}
    blind = pl.read_csv(C.snapshot_dir("20261020") / "cyp-challenge-TEST-BLINDED.csv")[
        "SMILES"
    ].to_list()

    picks = [("train", s) for s in rng.choice(union, N_PER_SET, replace=False)] + [
        ("blind", s) for s in rng.choice(blind, N_PER_SET, replace=False)
    ]
    matched = (
        Q.match_structures([s for _, s in picks])
        .with_columns(pl.Series("set", [p[0] for p in picks]))
        .drop_nulls("structure_id")
    )

    props = Q.load_conformer_properties().filter(
        pl.col("structure_id").is_in(matched["structure_id"].implode())
    )
    top = (
        props.sort("boltzmann_weight", descending=True)
        .group_by("structure_id")
        .head(TOP_CONFORMERS)
    )
    geoms = Q.read_conformers(matched["structure_id"].to_list())

    # One flat list of molecules for a single Monroe subprocess call; `records` says
    # which compound and arm each row belongs to.
    mols, records = [], []
    for row in matched.iter_rows(named=True):
        sid, smi = row["structure_id"], row["structure_smiles"]
        try:
            arms = [
                ("etkdg_a", encoders.monroe_etkdg_conformer(smi, 1), 1.0),
                ("etkdg_b", encoders.monroe_etkdg_conformer(smi, 2), 1.0),
            ]
        except ValueError:
            continue
        for conf in top.filter(pl.col("structure_id") == sid).iter_rows(named=True):
            elements, xyz = geoms[sid][conf["conf_id"]]
            arms.append(
                ("xtb", Q.mol_with_coordinates(smi, elements, xyz), conf["boltzmann_weight"])
            )
        for arm, mol, weight in arms:
            mols.append(mol)
            records.append((row["SMILES"], arm, weight))
    print(f"embedding {len(mols)} conformers for {matched.height} compounds")
    emb = encoders.embed_monroe_conformers(mols)

    # Collapse to one vector per (compound, arm). The first xtb row per compound is the
    # highest-weight conformer, which is the lowest-energy one.
    vectors: dict[str, dict[str, np.ndarray]] = {}
    xtb_rows: dict[str, list[tuple[np.ndarray, float]]] = {}
    for (smi, arm, weight), vec in zip(records, emb, strict=True):
        if arm == "xtb":
            xtb_rows.setdefault(smi, []).append((vec, weight))
        else:
            vectors.setdefault(smi, {})[arm] = vec
    for smi, rows in xtb_rows.items():
        weights = np.array([w for _, w in rows])
        stack = np.stack([v for v, _ in rows])
        vectors[smi]["xtb_min"] = stack[0]
        vectors[smi]["xtb_boltz"] = (weights[:, None] * stack).sum(0) / weights.sum()
        if smi in cached_row:
            vectors[smi]["cached"] = cached[cached_row[smi]]

    centre = np.mean([v for arms in vectors.values() for v in arms.values()], axis=0)
    names = list(vectors)
    a = np.stack([vectors[s]["etkdg_a"] for s in names]) - centre
    pair = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1)
    between = float(np.median(pair[np.triu_indices(len(names), 1)]))

    def dist(s: str, x: str, y: str) -> float | None:
        if x not in vectors[s] or y not in vectors[s]:
            return None
        return float(np.linalg.norm(vectors[s][x] - vectors[s][y]) / between)

    # Nearest-neighbour identity: is a compound's own etkdg_a the closest etkdg_a to
    # its xtb_min? Fails only if the geometry swap moves it into another compound's region.
    xm = np.stack([vectors[s]["xtb_min"] for s in names]) - centre
    own_nearest = np.argmin(
        np.linalg.norm(xm[:, None, :] - a[None, :, :], axis=-1), axis=1
    ) == np.arange(len(names))

    info = dict(
        zip(matched["SMILES"].to_list(), matched.select("set", "match").rows(), strict=True)
    )
    per = pl.DataFrame(
        {
            "SMILES": names,
            "set": [info[s][0] for s in names],
            "match": [info[s][1] for s in names],
            "seed_noise": [dist(s, "etkdg_a", "etkdg_b") for s in names],
            "xtb_shift": [dist(s, "xtb_min", "etkdg_a") for s in names],
            "boltz_vs_min": [dist(s, "xtb_boltz", "xtb_min") for s in names],
            "cached_vs_etkdg": [dist(s, "cached", "etkdg_a") for s in names],
            "own_nearest": own_nearest.tolist(),
        }
    )
    per.write_csv(OUT_DIR / "per_compound.csv")

    diff = np.stack([vectors[s]["xtb_min"] - vectors[s]["etkdg_a"] for s in names])
    systematic = float(np.sum(diff.mean(0) ** 2) / np.mean(np.sum(diff**2, axis=1)))
    summary = (
        per.group_by("set")
        .agg(
            pl.len().alias("n"),
            *[
                pl.col(c).median().round(3).alias(c)
                for c in ("seed_noise", "xtb_shift", "boltz_vs_min", "cached_vs_etkdg")
            ],
            pl.col("own_nearest").mean().round(3),
        )
        .sort("set")
    )
    summary.write_csv(OUT_DIR / "summary.csv")
    print(summary)
    print(per.group_by("match").agg(pl.col("seed_noise", "xtb_shift").median().round(3)))
    print(f"between-molecule median distance (centred): {between:.3f}")
    print(f"systematic fraction of the xtb-etkdg shift: {systematic:.3f}")
    noise, shift = per["seed_noise"].median(), per["xtb_shift"].median()
    go = noise >= 0.15 or shift >= 1.5 * noise
    print(
        f"seed noise {noise:.3f}, xtb shift {shift:.3f} ({shift / noise:.2f}x noise) "
        f"-> {'RUN A2' if go else 'STOP'}"
    )


if __name__ == "__main__":
    main()
