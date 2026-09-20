"""Uni-Mol 3D molecular representations, from free conformers or docked poses.

A 512-dimensional CLS embedding per molecule, from the ``mol_pre_no_h_220816``
checkpoint. Two entry points that differ only in where the coordinates come from:

- `embed_smiles` -- Uni-Mol generates its own ETKDG conformer. The **control**.
- `embed_poses` -- coordinates supplied from a docked pose.

## The control is not optional

Uni-Mol was pretrained on isolated-ligand conformers, so it encodes a molecule's own
geometry and has never seen a protein. Measured on six known CYP ligands docked into
CYP1A2, the docked-pose embedding against the same molecule's free ETKDG embedding
gave cosine 0.80-1.00 (caffeine 1.0000, paracetamol 0.9998, quinidine 0.8046),
against a median *between-molecule* cosine of 0.72. Rigid compounds are essentially
unchanged by docking; even the flexible ones only approach the between-molecule
floor.

So a docked-pose arm that beats a 2D baseline is evidence for Uni-Mol as a 3D ligand
featurizer, **not** evidence that docking helped. Only the difference between the two
arms supports a docking claim. This is the same interpretation trap notebook 09 hit
from the other direction, where `ecfp+pharmacophore_3d` lifted every endpoint but
lifted CYP2D6 least -- the reverse of what the hypothesis predicted.

## Why every call goes through a subprocess

Two independent reasons, either sufficient:

1. **OpenMP.** Uni-Mol is torch, and this repo's hard environment constraint is that
   a torch fit in the same process as LightGBM segfaults on macOS (see
   `tabular_models`' module docstring). Every torch model here runs in a child.
2. **unimol_tools forks.** It spawns worker processes internally, so a caller that
   invokes it at module scope fork-bombs: many Python processes at ~100% CPU and the
   real error ("An attempt has been made to start a new process before the current
   process has finished its bootstrapping phase") buried in output. The child script
   carries the `if __name__ == "__main__"` guard that prevents it.

The second failure looks like the first but is distinguishable by CPU: the OpenMP
deadlock sits at 0% in `kmp_flag_64::wait`, this one runs hot.

## `unimol_tools` is not installed by default -- set `CYP_UNIMOL_PYTHON`

`unimol_tools` pins ``numpy<2.3.0``, and numpy 2.2.x makes TabICL return all-NaN
probabilities on a ~2,400-row classification fit while raising nothing, which
silently zeroes the whole TDI track (see CLAUDE.md's environment-traps section, and
`pyproject.toml`'s ``numpy>=2.4``). The two cannot share one environment, so
`unimol_tools` is deliberately absent from every extra.

Because the child process is already separate, it does not have to be *this*
interpreter. Set ``CYP_UNIMOL_PYTHON`` to a Python that has `unimol_tools`
installed and both entry points work unchanged:

    uv venv /tmp/unimol
    VIRTUAL_ENV=/tmp/unimol uv pip install unimol_tools
    CYP_UNIMOL_PYTHON=/tmp/unimol/bin/python uv run python notebooks/11_docking_3d.py

Unset, it falls back to `sys.executable`, where the import simply fails -- so
`available()` returns False rather than the call dying mid-notebook.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

#: Dimensionality of the CLS embedding from the no-hydrogen checkpoint.
EMBEDDING_DIM = 512

#: Child script. Kept as a string, like `tabular_models._TFM_SCRIPT`, so the
#: subprocess boundary is visible at the call site rather than hidden in a
#: separate importable module that could be imported into the parent by mistake.
_UNIMOL_SCRIPT = """
import json
import sys

import numpy as np


def main():
    payload = json.loads(sys.argv[1])

    # Imported inside main() so the fork-safety guard is in force before
    # unimol_tools spawns anything.
    from unimol_tools import UniMolRepr

    clf = UniMolRepr(data_type="molecule", remove_hs=True, use_gpu=False)

    if payload["mode"] == "smiles":
        with open(payload["smiles"]) as handle:
            smiles = json.load(handle)
        result = clf.get_repr(smiles)
    else:
        atoms = json.load(open(payload["atoms"]))
        coords = [np.load(f"{payload['coords']}/{i}.npy") for i in range(len(atoms))]
        result = clf.get_repr({"atoms": atoms, "coordinates": coords})

    cls = result["cls_repr"] if isinstance(result, dict) else result
    np.save(payload["out"], np.asarray(cls, dtype=np.float32))


if __name__ == "__main__":
    main()
"""


def _python() -> str:
    """Interpreter for the Uni-Mol child.

    `CYP_UNIMOL_PYTHON` when set, else this process's own. See the module docstring:
    `unimol_tools` cannot be installed alongside the numpy this repo needs, so the
    only supported way to run it is a separate environment pointed at by that
    variable.
    """
    return os.environ.get("CYP_UNIMOL_PYTHON") or sys.executable


def _run_child(payload: dict, tmp: Path) -> np.ndarray:
    script = tmp / "unimol_embed.py"
    script.write_text(_UNIMOL_SCRIPT)
    result = subprocess.run(
        [_python(), str(script), json.dumps(payload)],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    out = Path(payload["out"])
    if result.returncode != 0 or not out.exists():
        raise RuntimeError(
            f"Uni-Mol child failed ({result.returncode}).\nstderr tail:\n{result.stderr[-3000:]}"
        )
    return np.load(out)


def embed_smiles(smiles: list[str], batch_size: int = 512) -> np.ndarray:
    """Uni-Mol embeddings from SMILES, using its own generated conformers.

    The A1 control arm. Batched because Uni-Mol holds every conformer in memory and
    the full challenge set at once is a needless peak on a 16 GB machine.

    Returns:
        Array of shape `(len(smiles), EMBEDDING_DIM)`.
    """
    blocks = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for start in range(0, len(smiles), batch_size):
            chunk = list(smiles[start : start + batch_size])
            smiles_path = tmp / "smiles.json"
            smiles_path.write_text(json.dumps(chunk))
            blocks.append(
                _run_child(
                    {
                        "mode": "smiles",
                        "smiles": str(smiles_path),
                        "out": str(tmp / "out.npy"),
                    },
                    tmp,
                )
            )
    return np.vstack(blocks) if blocks else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)


def embed_poses(
    atoms: list[list[str]], coordinates: list[np.ndarray], batch_size: int = 512
) -> np.ndarray:
    """Uni-Mol embeddings from supplied 3D coordinates -- the docked-pose arm.

    `atoms` and `coordinates` come from `docking.pose_coordinates`, which already
    strips hydrogens to match the ``mol_pre_no_h`` checkpoint.

    Returns:
        Array of shape `(len(atoms), EMBEDDING_DIM)`.
    """
    if len(atoms) != len(coordinates):
        raise ValueError(f"atoms and coordinates must align: {len(atoms)} vs {len(coordinates)}")
    blocks = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for start in range(0, len(atoms), batch_size):
            chunk_atoms = atoms[start : start + batch_size]
            chunk_coords = coordinates[start : start + batch_size]
            coord_dir = tmp / "coords"
            coord_dir.mkdir(exist_ok=True)
            for i, block in enumerate(chunk_coords):
                np.save(coord_dir / f"{i}.npy", np.asarray(block, dtype=np.float32))
            atoms_path = tmp / "atoms.json"
            atoms_path.write_text(json.dumps([list(a) for a in chunk_atoms]))
            blocks.append(
                _run_child(
                    {
                        "mode": "poses",
                        "atoms": str(atoms_path),
                        "coords": str(coord_dir),
                        "out": str(tmp / "out.npy"),
                    },
                    tmp,
                )
            )
    return np.vstack(blocks) if blocks else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)


def align(embeddings: np.ndarray, embedded_names: list[str], names: list[str]) -> np.ndarray:
    """Reorder `embeddings` onto `names`, zero-filling compounds without a pose.

    Docking loses compounds (embedding failures, molecules too large for the box),
    so the pose arm covers fewer compounds than the SMILES arm. Zero-filling keeps
    every arm on the same compound set, which is what makes a paired comparison
    legitimate -- dropping rows per arm would compare different populations.
    """
    lookup = {name: embeddings[i] for i, name in enumerate(embedded_names)}
    zero = np.zeros(embeddings.shape[1] if embeddings.size else EMBEDDING_DIM, dtype=np.float32)
    return np.vstack([lookup.get(name, zero) for name in names])


def available() -> bool:
    """Whether `unimol_tools` can be imported in the child interpreter (`_python`)."""
    try:
        result = subprocess.run(
            # `import unimol_tools` alone is not a sufficient probe: uninstalling
            # leaves a `unimol_tools/` directory behind holding downloaded model
            # weights, and Python imports that empty shell as a *namespace package*
            # with `__file__ = None`. The import succeeds, `available()` returns
            # True, and the arm then dies on the first real attribute access.
            # Requiring `__file__` distinguishes a real package from the leftovers.
            [_python(), "-c", "import unimol_tools; assert unimol_tools.__file__"],
            capture_output=True,
            text=True,
        )
    except OSError:
        # A `CYP_UNIMOL_PYTHON` pointing at a path that does not exist is the most
        # likely way this is misconfigured, and a probe named `available` must answer
        # False rather than raise -- callers use it to decide whether to run the arm
        # at all, so raising here turns a skippable arm into a dead notebook.
        return False
    return result.returncode == 0
