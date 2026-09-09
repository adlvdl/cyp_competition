"""Fingerprint dispatch via scikit-fingerprints.

Ported from the PXR repo's `generate_fingerprint`. One entry point so that swapping
representations in an experiment is a string change, not a rewrite.

PXR evidence on which to use: ECFP4 (radius 2, 2048 bits) was the workhorse for tree
models, Mordred descriptors suited XGBoost, and learned graph representations
(Chemprop/CheMeleon) beat every fingerprint model consistently. Start with ECFP here
for speed, and treat fingerprints as the floor rather than the ceiling.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_FP_REGISTRY: dict[str, tuple[str, str]] = {
    "ecfp": ("skfp.fingerprints", "ECFPFingerprint"),
    "morgan": ("skfp.fingerprints", "ECFPFingerprint"),
    "maccs": ("skfp.fingerprints", "MACCSFingerprint"),
    "rdkit": ("skfp.fingerprints", "RDKitFingerprint"),
    "atompair": ("skfp.fingerprints", "AtomPairFingerprint"),
    "torsion": ("skfp.fingerprints", "TopologicalTorsionFingerprint"),
    "avalon": ("skfp.fingerprints", "AvalonFingerprint"),
    "mordred": ("skfp.fingerprints", "MordredFingerprint"),
    "mqn": ("skfp.fingerprints", "MQNsFingerprint"),
    "pubchem": ("skfp.fingerprints", "PubChemFingerprint"),
}

AVAILABLE = tuple(_FP_REGISTRY)


def _resolve(name: str):
    if name not in _FP_REGISTRY:
        raise ValueError(
            f"Unknown fingerprint {name!r}. Available: {sorted(_FP_REGISTRY)}"
        )
    module_path, class_name = _FP_REGISTRY[name]
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def compute(smiles: Sequence[str], kind: str = "ecfp", **kwargs) -> np.ndarray:
    """Compute a fingerprint matrix for `smiles`.

    Conformer-requiring fingerprints get ETKDG conformers generated automatically.

    Args:
        smiles: SMILES strings.
        kind: One of `AVAILABLE`.
        **kwargs: Forwarded to the skfp class (e.g. `radius=3, n_bits=1024`).

    Returns:
        Array of shape (len(smiles), n_features).
    """
    fp_class = _resolve(kind)
    featurizer = fp_class(**kwargs) if kwargs else fp_class()

    inputs: Sequence = smiles
    if getattr(featurizer, "requires_conformers", False):
        from skfp.preprocessing import ConformerGenerator, MolFromSmilesTransformer

        mols = MolFromSmilesTransformer().transform(list(smiles))
        inputs = ConformerGenerator().transform(mols)

    return np.asarray(featurizer.transform(list(inputs)))


def ecfp4(smiles: Sequence[str], n_bits: int = 2048) -> np.ndarray:
    """ECFP4 (radius 2) at `n_bits` -- the PXR default for fingerprint models."""
    return compute(smiles, "ecfp", radius=2, fp_size=n_bits)
