"""Adapted from MolGpKa's `src/predict_pka.py` (Xundrug/MolGpKa, MIT).

Top-level inference: given an RDKit molecule, return per-atom pKa for every
candidate acidic and basic site. Two behavioural fixes over upstream, both found
while vendoring rather than assumed:

- The acid/base models are loaded once at import time rather than once per call.
  Upstream's `predict_acid`/`predict_base` reconstruct `GCNNet` and reload the 6MB
  checkpoint from disk on every single molecule, which is a correctness-neutral but
  real cost across a corpus of thousands.
- `predict()` no longer standardises with `rdMolStandardize.Uncharger` by default.
  That step assumes the input's formal charges represent its actual protonation
  state, which is true for a curated benchmark but not for arbitrary pipeline SMILES
  -- and it round-trips through `Chem.MolToSmiles`/`MolFromSmiles`, which silently
  drops a molecule that fails to re-parse rather than raising. Callers that want the
  original uncharging behaviour can pass `uncharge=True` explicitly.
"""

from __future__ import annotations

from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from .descriptor import mol_to_graph
from .ionization_sites import ionization_sites
from .net import GCNNet

_WEIGHTS_DIR = Path(__file__).resolve().parent


def _load_model(weight_file: str) -> GCNNet:
    model = GCNNet()
    state_dict = torch.load(_WEIGHTS_DIR / weight_file, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# Loaded once at import time -- see module docstring.
_ACID_MODEL = _load_model("weight_acid.pth")
_BASE_MODEL = _load_model("weight_base.pth")


def _predict_sites(mol: Chem.Mol, acid_or_base: str) -> dict[int, float]:
    model = _ACID_MODEL if acid_or_base == "acid" else _BASE_MODEL
    results: dict[int, float] = {}
    for atom_idx in ionization_sites(mol, acid_or_base):
        x, edge_index, batch = mol_to_graph(mol, atom_idx)
        with torch.no_grad():
            pka = model(x, edge_index, batch)
        results[atom_idx] = float(pka.item())
    return results


def predict(mol: Chem.Mol, uncharge: bool = False) -> tuple[dict[int, float], dict[int, float]]:
    """Per-atom pKa for every candidate ionization site in `mol`.

    Args:
        mol: An RDKit molecule. Explicit hydrogens are added internally
            (`Chem.AddHs`) -- pass a molecule as parsed from SMILES, without doing
            this yourself first.
        uncharge: Reproduce upstream's default of neutralising formal charges via
            `rdMolStandardize.Uncharger` before prediction. Off by default here --
            see the module docstring for why forcing this on arbitrary input is not
            always the right call.

    Returns:
        `(base_pka, acid_pka)` -- each a `{atom_index: predicted_pKa}` dict, keyed
        on the **heavy-atom** index in `mol` (after `AddHs`, so indices match what
        `Chem.AddHs(mol)` produces, not the input `mol`'s own numbering if it had
        implicit hydrogens). Sites the SMARTS table finds none of yield an empty dict
        for that acid/base half, not a missing key or an exception.
    """
    if uncharge:
        from rdkit.Chem.MolStandardize import rdMolStandardize

        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        reparsed = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        if reparsed is None:
            raise ValueError("molecule failed to re-parse after uncharging")
        mol = reparsed

    mol = AllChem.AddHs(mol)
    return _predict_sites(mol, "base"), _predict_sites(mol, "acid")


def strongest_base_pka(mol: Chem.Mol, uncharge: bool = False) -> float | None:
    """The most basic site's pKa, or None if the molecule has no basic site.

    The single number most CYP2D6 work wants: the fraction protonated at
    physiological pH is governed by the *most* basic site, not by an arithmetic mean
    across every ionizable atom the SMARTS table happens to match.
    """
    base_pka, _ = predict(mol, uncharge=uncharge)
    return max(base_pka.values()) if base_pka else None
