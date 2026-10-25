"""Frozen 3D molecular encoders: Atom-JEPA and Monroe.

Both are used the way `graph_models.chemeleon_embed` uses CheMeleon: the encoder is
frozen, no label ever enters the computation, and the embedding is read out by a
tabular foundation model downstream. Both import torch, so every call runs in a child
process -- `import cyp` claims LightGBM's OpenMP runtime first, and a torch model fitted
in the parent afterwards segfaults or deadlocks (see CLAUDE.md, environment traps).

Both build a conformer per molecule from SMILES (ETKDGv3 + MMFF94 for Atom-JEPA, ETKDG
+ MMFF94s/UFF for Monroe), so the embedding depends on RDKit's random conformer. That
is fixed by caching: embed once, never recompute inside a CV loop.

Install with the `encoders` extra. Monroe's 300 MB weights are a Git LFS file that a
plain `git clone` leaves as a 134-byte pointer; `fetch_monroe_weights` downloads the real
file into `.encoders/monroe/`.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENCODER_DIR = PROJECT_ROOT / ".encoders"
MONROE_DIR = ENCODER_DIR / "monroe"

#: Where the 300 MB checkpoint lives upstream. `media.githubusercontent.com` resolves
#: an LFS pointer to the blob; `raw.githubusercontent.com` returns the pointer itself.
_MONROE_WEIGHTS_URL = (
    "https://media.githubusercontent.com/media/blazejba/monroe/main/checkpoint/weights.pt"
)
_MONROE_WEIGHTS_SHA256 = "df996e8e98b12a3cda5fe5acbef3e1e9a4ea31946d3736461cd07e53a5bf205a"
_MONROE_CONFIG_URL = "https://raw.githubusercontent.com/blazejba/monroe/main/checkpoint/config.json"

KINDS = ("atom_jepa", "monroe")

#: Atom-JEPA embedding blocks (the encoder has 8) and width.
ATOM_JEPA_LAYERS = 8
ATOM_JEPA_DIM = 256
MONROE_DIM = 720

# Each script is a template run as `python -c`-style file in a child process. The
# `__main__` guard is not optional: Monroe featurizes through a fork pool and
# unimol_tools-style libraries re-import the script, so an unguarded script fork-bombs.
_ATOM_JEPA_SCRIPT = """
import json, sys
import numpy as np

def main(smi_file, out_file, batch_size):
    from atom_jepa import AtomJEPA
    smiles = json.load(open(smi_file))
    model = AtomJEPA.from_pretrained("molecules")
    chunks = []
    for start in range(0, len(smiles), batch_size):
        # layers="all" is [n, 8, 256]: one read of the encoder yields every block, so
        # a later layer sweep costs nothing. Structure features are the atom mean.
        chunks.append(np.asarray(model.embed(smiles[start:start + batch_size], layers="all")))
        print(f"{min(start + batch_size, len(smiles))}/{len(smiles)}", file=sys.stderr, flush=True)
    np.save(out_file, np.concatenate(chunks).astype(np.float32))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
"""

_MONROE_SCRIPT = """
import json, os, sys
import numpy as np

def main(smi_file, out_file, ckpt_dir, batch_size):
    import torch
    from torch_geometric.data import Batch
    from monroe.eval.embed import featurize_smiles, to_pyg
    from monroe.model.ckpt import _build_encoder, _load_config_and_weights, _resolve_ckpt_dir

    smiles = json.load(open(smi_file))

    # Load by hand rather than through `monroe.model.ckpt.load_ckpt`: that call uses
    # load_state_dict(strict=False), so a key mismatch silently leaves layers at their
    # random initialisation and every downstream score is a score of noise.
    # The checkpoint was saved on a GPU; without map_location torch.load raises on a Mac.
    hp, state = _load_config_and_weights(
        _resolve_ckpt_dir(ckpt_dir), device=torch.device("cpu"), use_ema=False
    )
    encoder = _build_encoder(hp)
    wrapped = torch.nn.ModuleDict({"encoder": encoder})
    result = wrapped.load_state_dict(state, strict=False)
    # The checkpoint also holds the pretraining decoder and per-dataset heads, which an encoder-only
    # model rightly ignores. Anything else unexpected, or any missing key, is a real
    # mismatch.
    stray = [k for k in result.unexpected_keys if not k.startswith(("decoders.", "dataset_heads."))]
    if result.missing_keys or stray:
        raise RuntimeError(
            f"Monroe checkpoint did not load cleanly: missing={result.missing_keys[:5]} "
            f"unexpected={stray[:5]}"
        )
    encoder.eval()

    # os.sched_getaffinity is Linux-only and monroe's default calls it.
    built = featurize_smiles(smiles, n_workers=os.cpu_count())
    by_smiles = {smi: graph for smi, graph in built}
    missing = [s for s in smiles if s not in by_smiles]
    if missing:
        raise RuntimeError(
            f"Monroe could not featurize {len(missing)} molecules, e.g. {missing[:3]}"
        )

    # A planar (all z=0) layout is monroe's silent fallback when conformer generation
    # fails; the encoder reads 3D, so those embeddings are unreliable. Count, never hide.
    flat = [
        s for s in smiles
        if np.all(np.asarray(by_smiles[s]["pos_rdkit"])[:, 2] == 0)
    ]
    print(f"flat_2d_fallback={len(flat)}", file=sys.stderr, flush=True)

    out = []
    with torch.no_grad():
        for start in range(0, len(smiles), batch_size):
            graphs = [to_pyg(by_smiles[s]) for s in smiles[start:start + batch_size]]
            emb, _ = encoder(Batch.from_data_list(graphs))
            out.append(emb.cpu().numpy())
            done = min(start + batch_size, len(smiles))
            print(f"{done}/{len(smiles)}", file=sys.stderr, flush=True)
    np.save(out_file, np.concatenate(out).astype(np.float32))
    json.dump({"flat_2d": flat}, open(out_file + ".flat.json", "w"))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
"""


# Same checkpoint loading as _MONROE_SCRIPT, but the graph is built from a supplied
# conformer instead of Monroe's own InChI round-trip and random ETKDG embedding. The
# graph builder is the one Monroe uses for its SDF route (`build_mol_from_sdf`), so
# only the coordinates differ.
_MONROE_COORDS_SCRIPT = """
import sys
import numpy as np

def main(sdf_file, out_file, ckpt_dir, batch_size):
    import torch
    from rdkit import Chem
    from torch_geometric.data import Batch
    from monroe.eval.embed import to_pyg
    from monroe.model.featurizer import build_from_mol_and_coords
    from monroe.model.ckpt import _build_encoder, _load_config_and_weights, _resolve_ckpt_dir

    hp, state = _load_config_and_weights(
        _resolve_ckpt_dir(ckpt_dir), device=torch.device("cpu"), use_ema=False
    )
    encoder = _build_encoder(hp)
    result = torch.nn.ModuleDict({"encoder": encoder}).load_state_dict(state, strict=False)
    heads = ("decoders.", "dataset_heads.")
    stray = [k for k in result.unexpected_keys if not k.startswith(heads)]
    if result.missing_keys or stray:
        raise RuntimeError(f"Monroe checkpoint mismatch: {result.missing_keys[:5]} {stray[:5]}")
    encoder.eval()

    graphs = []
    for i, mol in enumerate(Chem.SDMolSupplier(sdf_file, removeHs=False)):
        if mol is None:
            raise RuntimeError(f"SDF record {i} could not be read")
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        pos = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
        graphs.append(build_from_mol_and_coords(
            mol, pos_rdkit=pos, stereo_augmentation=True, symmetrize=True
        ))

    out = []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            chunk = graphs[start:start + batch_size]
            emb, _ = encoder(Batch.from_data_list([to_pyg(g) for g in chunk]))
            out.append(emb.cpu().numpy())
            done = min(start + batch_size, len(graphs))
            print(f"{done}/{len(graphs)}", file=sys.stderr, flush=True)
    np.save(out_file, np.concatenate(out).astype(np.float32))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
"""


def monroe_etkdg_conformer(smiles: str, seed: int):
    """One conformer by Monroe's own `predict_structure` recipe, with an explicit seed.

    ETKDGv3 with random coordinates, then MMFF94s, UFF where MMFF has no parameters.
    Monroe itself never seeds, so its embeddings are not reproducible; this is. Raises
    `ValueError` if embedding fails, which the caller handles (Monroe's own answer is a
    flat 2D layout, `monroe_conformer_or_flat`).
    """
    from rdkit import Chem
    from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = rdDistGeom.ETKDGv3()
    params.enforceChirality = True
    params.useRandomCoords = True
    params.maxIterations = 500
    params.randomSeed = seed
    if rdDistGeom.EmbedMolecule(mol, params) < 0:
        raise ValueError(f"ETKDG failed for {smiles}")
    props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, "MMFF94s")
    if props is not None:
        rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, props).Minimize(maxIts=200)
    else:
        rdForceFieldHelpers.UFFOptimizeMolecule(mol, maxIters=200)
    return mol


def monroe_conformer_or_flat(smiles: str, seed: int):
    """`monroe_etkdg_conformer`, retried on three seeds, then Monroe's flat 2D fallback.

    Returns `(mol, is_flat)` so the caller can count fallbacks rather than hide them.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    for attempt in range(3):
        try:
            return monroe_etkdg_conformer(smiles, seed + 1000 * attempt), False
        except ValueError:
            continue
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.Compute2DCoords(mol)
    return mol, True


def embed_monroe_conformers(mols: list, batch_size: int = 16) -> np.ndarray:
    """Monroe embeddings of molecules that already carry a 3D conformer.

    Each molecule must have explicit hydrogens and one conformer; row i of the result is
    `mols[i]`. Used to feed externally optimised geometries (`cyp.shorku_qm`) to Monroe
    in place of its own random ETKDG conformer. Runs in a child process like `embed`.
    """
    from rdkit import Chem

    work = Path(tempfile.mkdtemp(prefix="cyp_monroe_coords_"))
    sdf_file, out_file, script = work / "mols.sdf", work / "emb.npy", work / "run.py"
    with Chem.SDWriter(str(sdf_file)) as writer:
        for mol in mols:
            writer.write(mol)
    script.write_text(_MONROE_COORDS_SCRIPT)
    cmd = [
        sys.executable, str(script), str(sdf_file), str(out_file),
        str(fetch_monroe_weights()), str(batch_size),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"monroe coords subprocess failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
        )
    emb = np.load(out_file)
    if emb.shape[0] != len(mols) or not np.isfinite(emb).all():
        raise RuntimeError(f"monroe coords: bad output shape {emb.shape} for {len(mols)} mols")
    return emb


def fetch_monroe_weights(dest: Path = MONROE_DIR) -> Path:
    """Download Monroe's checkpoint (config + 300 MB weights) and verify its hash.

    Returns the checkpoint directory. Idempotent: an existing, hash-correct file is
    left alone. A hash mismatch raises rather than proceeding, because an LFS pointer
    (134 bytes) fetched from the wrong URL would otherwise surface as an opaque
    `torch.load` failure inside the child process.
    """
    import urllib.request

    dest.mkdir(parents=True, exist_ok=True)
    weights, config = dest / "weights.pt", dest / "config.json"
    if not config.exists():
        urllib.request.urlretrieve(_MONROE_CONFIG_URL, config)
    if not weights.exists():
        urllib.request.urlretrieve(_MONROE_WEIGHTS_URL, weights)
    digest = hashlib.sha256()
    with weights.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != _MONROE_WEIGHTS_SHA256:
        raise RuntimeError(
            f"{weights} has sha256 {digest.hexdigest()[:12]}..., expected "
            f"{_MONROE_WEIGHTS_SHA256[:12]}...; delete it and re-fetch."
        )
    return dest


def embed(
    kind: str,
    smiles: list[str],
    batch_size: int = 16,
    log_path: Path | None = None,
) -> np.ndarray:
    """Embed SMILES with a frozen encoder, in a child process.

    Args:
        kind: "atom_jepa" or "monroe".
        smiles: Molecules to embed. Order is preserved; a molecule the encoder cannot
            featurize raises rather than being dropped, so row i is always `smiles[i]`.
        batch_size: Molecules per forward pass.
        log_path: If given, the child's stderr (progress, fallback counts) goes here,
            so a long run can be watched with `tail -f`.

    Returns:
        Atom-JEPA: `(n, 8, 256)`, one slice per encoder block (index -1 is the last
        block after the final norm, the package default). Monroe: `(n, 720)`.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")

    work = Path(tempfile.mkdtemp(prefix=f"cyp_{kind}_"))
    smi_file, out_file, script = work / "smiles.json", work / "emb.npy", work / "run.py"
    smi_file.write_text(json.dumps(list(smiles)))

    if kind == "atom_jepa":
        script.write_text(_ATOM_JEPA_SCRIPT)
        cmd = [sys.executable, str(script), str(smi_file), str(out_file), str(batch_size)]
    else:
        script.write_text(_MONROE_SCRIPT)
        ckpt = fetch_monroe_weights()
        cmd = [
            sys.executable,
            str(script),
            str(smi_file),
            str(out_file),
            str(ckpt),
            str(batch_size),
        ]

    stderr = log_path.open("w") if log_path is not None else subprocess.PIPE
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=stderr, text=True)
    finally:
        if log_path is not None:
            stderr.close()
    if result.returncode != 0:
        tail = log_path.read_text()[-2000:] if log_path is not None else result.stderr[-2000:]
        raise RuntimeError(
            f"{kind} embedding subprocess failed (exit {result.returncode}):\n{tail}"
        )

    emb = np.load(out_file)
    if emb.shape[0] != len(smiles):
        raise RuntimeError(f"{kind}: expected {len(smiles)} rows, got {emb.shape[0]}")
    if not np.isfinite(emb).all():
        raise RuntimeError(f"{kind}: embedding contains non-finite values")
    return emb


def embed_cached(
    kind: str,
    smiles: list[str],
    cache_dir: Path,
    batch_size: int = 16,
) -> np.ndarray:
    """`embed` over the *unique* SMILES, cached once, then indexed back per row.

    The four regression endpoints share 1,309 compounds, so embedding the union once
    instead of per endpoint saves ~25% of a run that takes about an hour. The cache is
    keyed by a hash of the sorted unique SMILES, so a data-snapshot change invalidates
    it instead of silently serving embeddings for a different compound set.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    unique = sorted(set(smiles))
    key = hashlib.sha256("\n".join(unique).encode()).hexdigest()[:12]
    path = cache_dir / f"{kind}_{key}.npy"
    if path.exists():
        table = np.load(path)
    else:
        table = embed(kind, unique, batch_size=batch_size, log_path=cache_dir / f"{kind}_{key}.log")
        np.save(path, table)
    row_of = {s: i for i, s in enumerate(unique)}
    return table[[row_of[s] for s in smiles]]
