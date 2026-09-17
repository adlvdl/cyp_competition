"""Explicit CYP2D6 pharmacophore descriptors.

The interpretable arm of notebook 09. `fingerprints`' 3D entries (`e3fp`, `usrcat`,
...) are learned or hashed representations: if one of them wins, it says *that* 3D
helps without saying *which* geometry mattered. These descriptors are the control
that answers the second question, because each one is a named quantity.

## Why CYP2D6 specifically

Measured on the challenge's own labels, CYP2D6 is mechanistically unlike the other
three endpoints rather than merely harder:

| endpoint | basic N vs potency | logP vs potency | % with basic N |
|:--|--:|--:|--:|
| CYP1A2 | -0.132 | 0.233 | 16.6% |
| CYP2C9 | -0.160 | 0.485 | 17.6% |
| **CYP2D6** | **+0.284** | **0.034** | **40.9%** |
| CYP3A4 | -0.068 | 0.602 | 23.0% |

The other three track lipophilicity, which a circular fingerprint captures well as a
bulk property. CYP2D6 is the only endpoint where logP carries essentially nothing and
the only one where a basic nitrogen is *positively* associated with potency. That is
the literature's CYP2D6 pharmacophore -- a protonatable nitrogen at a characteristic
distance from an aromatic system, engaging Glu216/Asp301 -- and it is a *spatial*
relationship, which is what a 2D fingerprint cannot express.

## The one measurement that motivates the 3D versions

Topological distance from the basic nitrogen to the nearest aromatic atom correlates
-0.122 with CYP2D6 potency, with a monotonic trend across bins (mean pIC50 5.155 at
2-4 bonds, 4.993 at 4-6, 4.978 at 6-8). So the relationship is real but weakly
expressed through bond counts. Through-space distance over a conformer ensemble is
the direct test of whether the pharmacophore is genuinely geometric, and
`descriptors` computes both so the comparison is available in one frame.

A 3D distance that beats the topological one is evidence for geometry. One that does
not is evidence the hypothesis is wrong, which is equally worth having before
spending a dependency on a pretrained 3D model.
"""

from __future__ import annotations

import signal
from collections.abc import Sequence

import numpy as np
import polars as pl


class ConformerTimeout(Exception):
    """Raised when embedding a single molecule exceeds its wall-clock budget."""


class _timeout:
    """SIGALRM-based wall-clock guard for one embedding call.

    RDKit's `EmbedMultipleConfs` takes `maxAttempts` (a retry count) but no
    wall-clock bound, and a handful of large/flexible molecules -- observed
    directly on PubChem's pretraining corpus, see `descriptors`' docstring --
    can run far longer than the rest of the batch. `signal.alarm` runs in the
    calling process's main thread, which is exactly where this lands: in-process
    when `n_jobs=1`, or in a joblib/loky worker's main thread otherwise. POSIX
    only, matching this repo's darwin-only environment. `seconds <= 0` disables
    the guard.
    """

    def __init__(self, seconds: float | None):
        self.seconds = seconds

    def _raise(self, signum, frame):
        raise ConformerTimeout(f"conformer generation exceeded {self.seconds}s")

    def __enter__(self):
        if self.seconds and self.seconds > 0:
            signal.signal(signal.SIGALRM, self._raise)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.seconds and self.seconds > 0:
            signal.setitimer(signal.ITIMER_REAL, 0)
        return False

#: Columns `descriptors` returns, in order. Exposed so a caller can select the 2D
#: subset without generating conformers -- the 2D arm is seconds, the 3D arm minutes.
DESCRIPTOR_2D = (
    "n_basic_nitrogen",
    "n_aromatic_rings",
    "topological_n_to_aromatic",
    "mol_logp",
    "tpsa",
)
DESCRIPTOR_3D = (
    "spatial_n_to_aromatic_min",
    "spatial_n_to_aromatic_mean",
)
DESCRIPTOR_COLUMNS = DESCRIPTOR_2D + DESCRIPTOR_3D


def basic_nitrogen_indices(mol) -> list[int]:
    """Atom indices of likely-protonated nitrogens.

    "Basic" here means sp3, non-aromatic, and not an amide nitrogen. The amide
    exclusion is the one that matters: an amide N is planar and effectively
    non-basic at physiological pH, so counting it would put a large number of
    ordinary carboxamides into the same bin as the amines the CYP2D6 pharmacophore
    is actually about.

    This is a structural heuristic, not a pKa calculation. It is deliberately cheap
    and deliberately transparent -- a descriptor whose failures can be read off the
    structure is worth more here than a black-box pKa predictor whose errors would be
    invisible inside an already-uncertain result.
    """
    from rdkit import Chem

    indices = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "N" or atom.GetIsAromatic():
            continue
        if atom.GetHybridization() != Chem.HybridizationType.SP3:
            continue
        is_amide = any(
            neighbour.GetSymbol() == "C"
            and any(
                bond.GetBondType() == Chem.BondType.DOUBLE
                and bond.GetOtherAtom(neighbour).GetSymbol() == "O"
                for bond in neighbour.GetBonds()
            )
            for neighbour in atom.GetNeighbors()
        )
        if not is_amide:
            indices.append(atom.GetIdx())
    return indices


def _topological_distance(mol) -> float:
    """Shortest bond-path distance from any basic N to any aromatic atom."""
    from rdkit import Chem

    nitrogens = basic_nitrogen_indices(mol)
    aromatics = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
    if not nitrogens or not aromatics:
        return float("nan")
    matrix = Chem.GetDistanceMatrix(mol)
    return float(min(matrix[i][j] for i in nitrogens for j in aromatics))


def _spatial_distances(
    mol, n_conformers: int, timeout_s: float | None = None
) -> tuple[float, float]:
    """Through-space basic-N-to-aromatic distance, over a conformer ensemble.

    Returns `(min, mean)` across conformers of each conformer's own shortest
    N-to-aromatic distance. Both are kept because they answer different questions:
    the minimum is the closest approach the molecule can adopt (what matters if
    binding selects a conformer), the mean is what an unbiased ensemble looks like.

    Conformers are generated here rather than taken from the caller because the
    descriptor is meaningless without them, and a silent fallback to 2D would be the
    worst outcome -- it would look like a null result for geometry.

    `timeout_s` bounds the embedding call's wall-clock time; a molecule that runs
    over yields NaN, the same as any other embedding failure, rather than stalling
    the batch. `None` or non-positive disables the bound.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    nitrogens = basic_nitrogen_indices(mol)
    aromatics = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
    if not nitrogens or not aromatics:
        return float("nan"), float("nan")

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    try:
        with _timeout(timeout_s):
            # `EmbedMultipleConfs` returns an RDKit vector of conformer ids, not a
            # plain int -- `== 0` compares the vector object to an int and is never
            # True, so a total embedding failure (empty vector) silently passed this
            # guard and crashed later in `np.min` on an empty sequence instead of
            # being caught here. `len(...) == 0` is the correct emptiness check
            # regardless of the vector-like type RDKit returns.
            n_embedded = len(AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params))
    except ConformerTimeout:
        return float("nan"), float("nan")
    if n_embedded == 0:
        return float("nan"), float("nan")

    per_conformer = []
    for conformer in mol.GetConformers():
        positions = conformer.GetPositions()
        per_conformer.append(
            min(
                float(np.linalg.norm(positions[i] - positions[j]))
                for i in nitrogens
                for j in aromatics
            )
        )
    return float(np.min(per_conformer)), float(np.mean(per_conformer))


def _descriptor_row(
    smi: str, include_3d: bool, n_conformers: int, timeout_s: float | None = None
) -> dict:
    """One molecule's descriptor row -- factored out so it can run under
    `joblib.Parallel`, which needs a picklable top-level function rather than a
    closure over `descriptors`' loop state."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {"smiles": smi, **{c: None for c in DESCRIPTOR_COLUMNS}}

    row = {
        "smiles": smi,
        "n_basic_nitrogen": len(basic_nitrogen_indices(mol)),
        "n_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "topological_n_to_aromatic": _topological_distance(mol),
        "mol_logp": Descriptors.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
    }
    if include_3d:
        minimum, mean = _spatial_distances(mol, n_conformers, timeout_s)
        row["spatial_n_to_aromatic_min"] = minimum
        row["spatial_n_to_aromatic_mean"] = mean
    return row


def descriptors(
    smiles: Sequence[str],
    include_3d: bool = True,
    n_conformers: int = 10,
    n_jobs: int = 1,
    timeout_s: float | None = None,
    on_progress=None,
) -> pl.DataFrame:
    """Pharmacophore descriptors for `smiles`, one row per input in input order.

    Args:
        smiles: SMILES strings.
        include_3d: Generate conformers and compute the spatial distances. Costs
            roughly a minute per thousand compounds against under a second for the
            2D columns, so a caller comparing only the 2D arm should pass False.
        n_conformers: Ensemble size per molecule when `include_3d`.
        timeout_s: Wall-clock bound per molecule's embedding call. A molecule that
            runs over gets NaN spatial columns instead of stalling the batch --
            same failure mode as an outright embedding failure. `None` (default)
            or non-positive disables the bound. Ignored when `include_3d=False`.
        n_jobs: Worker processes for `include_3d`. Conformer generation is pure
            RDKit/CPU work with no shared state between molecules, so it
            parallelises with no correctness cost -- unlike a model fit, there is
            no OpenMP runtime involved and no result to keep in sync across
            workers. 1 (the default) runs in-process, matching every prior call
            site; pass -1 for all cores on a large corpus. Ignored when
            `include_3d=False`, since the 2D columns are already sub-second even
            at these sizes.

            **Do not submit work in manually-sized chunks** (a fixed-size list per
            `Parallel(...)` call, one call per chunk) -- found the hard way on
            PubChem's 12,719-compound pretraining matrix, whose molecules average
            larger and more flexible than the challenge's own curated CYP2D6 set,
            so a handful are much slower to embed than the rest. Chunking forces
            every worker to finish its share of one chunk before the next chunk
            can be dispatched, so one slow molecule anywhere in a chunk stalls
            every other worker in that chunk until it finishes -- observed
            directly as 9 of 10 workers sitting idle while 1 ground through a
            single molecule, for over two hours, on a corpus this implementation
            should clear in minutes. `return_as="generator"` (below) submits the
            whole job at once and lets joblib's own scheduler pull the next task
            onto any worker the instant it frees up, so a slow molecule costs
            exactly its own time and nothing else's.
        on_progress: Optional `on_progress(done, total)` callback, called after
            each molecule (or each completed parallel batch) -- see `cv.FoldCallback`
            for the same pattern applied to CV folds, for the same reason: a
            multi-minute step with no progress signal is indistinguishable from a
            hang, which is exactly what made this parameter necessary.

    Returns:
        A frame with `DESCRIPTOR_COLUMNS` (2D only when `include_3d=False`), plus
        `smiles`. Unparseable molecules and molecules lacking either pharmacophore
        feature get nulls rather than zeros -- a zero would be read as "distance 0",
        a genuine and very different measurement.
    """
    smiles = list(smiles)
    if not include_3d or n_jobs == 1:
        records = []
        for i, smi in enumerate(smiles):
            records.append(_descriptor_row(smi, include_3d, n_conformers, timeout_s))
            if on_progress is not None:
                on_progress(i + 1, len(smiles))
    else:
        from joblib import Parallel, delayed

        # `return_as="generator"` streams each result the moment it is ready, in
        # submission order (joblib's own guarantee -- verified directly: a
        # deliberately slow task submitted early still yields in its original
        # position). That is what gives continuous work-stealing across all
        # workers, unlike the earlier chunked-dispatch version this replaced.
        records = []
        with Parallel(n_jobs=n_jobs, return_as="generator") as parallel:
            jobs = (
                delayed(_descriptor_row)(smi, include_3d, n_conformers, timeout_s)
                for smi in smiles
            )
            for i, record in enumerate(parallel(jobs)):
                records.append(record)
                if on_progress is not None:
                    on_progress(i + 1, len(smiles))

    frame = pl.DataFrame(records, infer_schema_length=None)
    columns = ["smiles", *DESCRIPTOR_2D] + (list(DESCRIPTOR_3D) if include_3d else [])
    return frame.select([c for c in columns if c in frame.columns])


def matrix(frame: pl.DataFrame, columns: Sequence[str] | None = None) -> np.ndarray:
    """`descriptors` output as a float matrix, nulls filled with the column median.

    Median rather than zero for the same reason `descriptors` returns nulls: zero is
    a meaningful distance. Molecules with no basic nitrogen are the bulk of the nulls
    here (59% of CYP2D6's compounds), so the fill value is load-bearing and a
    column-median fill keeps them at the population centre rather than at an extreme.
    """
    columns = list(columns or [c for c in frame.columns if c != "smiles"])
    # RDKit's absent-feature sentinel arrives as float NaN, not as a Polars null, so
    # `fill_null` alone silently leaves it in place and the model receives NaN. Map
    # NaN to null first, then fill -- the two are distinct in Polars and only the
    # second is what `fill_null` acts on.
    filled = frame.select(
        [
            pl.col(c)
            .cast(pl.Float64)
            .fill_nan(None)
            .fill_null(pl.col(c).cast(pl.Float64).fill_nan(None).median())
            for c in columns
        ]
    )
    return filled.to_numpy().astype(float)
