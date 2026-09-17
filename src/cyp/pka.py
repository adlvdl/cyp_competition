"""CYP2D6-motivated pKa features, from the vendored MolGpKa model.

Prediction itself is `vendor/molgpka/` (see `vendor/README.md` for what was adapted
and why) -- this module only wires it into a per-compound feature.

## Why this exists

`pharmacophore.py`'s `n_basic_nitrogen` is a binary sp3-nitrogen heuristic: every
qualifying atom counts equally, whether it is a strongly basic aliphatic amine
(pKa ~9-10, fully protonated at physiological pH) or a weakly basic one next to an
electron-withdrawing group (pKa ~4-5, mostly neutral). CYP2D6's pharmacophore is
about the *protonated* nitrogen specifically -- the positive charge is what engages
Glu216/Asp301 -- so a predicted pKa lets the feature reflect fraction-protonated
rather than mere atom count.

## Why every prediction runs in a subprocess

`import cyp` loads LightGBM's OpenMP runtime first, deliberately (see `cyp/__init__.py`
and the "OpenMP load order" entry in CLAUDE.md's environment-traps section) -- any
project code that later imports LightGBM or fits it is safe because of that ordering.
MolGpKa needs torch, a *second* OpenMP runtime, and torch has never been imported in
this process at that point. Confirmed directly: `import cyp` then
`from molgpka import predict` **segfaults** (exit code 139), the same failure mode
CLAUDE.md documents for TabICL/TabPFN after LightGBM, just with a new library hitting
it. There is no import order that avoids this reliably in a shared, long-running
process -- an already-imported `cyp.data`/`cyp.fingerprints` call elsewhere in the
same session has already loaded LightGBM by the time any notebook cell reaches this
module. Every prediction here therefore runs in a short-lived child process, which
gets its own clean runtime -- the same fix `tabular_models.predict_subprocess` uses
for TabICL/TabPFN, adapted for a scalar-in-scalar-out call instead of an array one.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from . import constants as C

#: Physiological pH, for the Henderson-Hasselbalch fraction-protonated conversion.
PHYSIOLOGICAL_PH = 7.4

_CHILD_SCRIPT = f"""
import sys
sys.path.insert(0, {str(C.PROJECT_ROOT / "vendor")!r})
from molgpka import predict as molgpka_predict
from rdkit import Chem

smiles_list = sys.argv[1].split("\\t")
results = []
for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        results.append(None)
        continue
    results.append(molgpka_predict.strongest_base_pka(mol))
print("RESULT_JSON_START", flush=True)
import json as _json
print(_json.dumps(results), flush=True)
"""


def strongest_base_pkas(smiles_list: list[str]) -> list[float | None]:
    """The most basic ionizable site's predicted pKa per compound, one subprocess call.

    Batched over the whole list rather than one process per compound -- torch model
    load is the fixed cost here (a few hundred ms for two 6MB checkpoints), so
    batching amortises it across every compound instead of paying it per row.

    Returns `None` for a compound MolGpKa finds no basic site on, or that fails to
    parse as SMILES -- same convention as `vendor.molgpka.predict.strongest_base_pka`.

    A SMILES string containing an embedded null byte is a different failure: it
    fails before the child process even starts, since `subprocess.run` rejects such
    an argv with `ValueError: embedded null byte` for the whole batch, not just that
    row. Real SMILES never legitimately contain one, so this is corrupted input, not
    an ordinary parse failure -- treat a `ValueError` here as a signal to check the
    upstream data rather than something to swallow into a `None`.
    """
    if not smiles_list:
        return []
    # Tab-separated argv is safe here: SMILES never contains a literal tab, and this
    # avoids a temp-file round trip for what is normally a few thousand short strings.
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_SCRIPT, "\t".join(smiles_list)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"MolGpKa subprocess failed:\n{result.stderr}")
    marker = "RESULT_JSON_START\n"
    idx = result.stdout.find(marker)
    if idx == -1:
        raise RuntimeError(
            f"MolGpKa subprocess produced no result:\n{result.stdout}\n{result.stderr}"
        )
    return json.loads(result.stdout[idx + len(marker) :])


def strongest_base_pka(smiles: str) -> float | None:
    """Single-compound convenience wrapper around `strongest_base_pkas`.

    Prefer the batched form in a loop -- each call here pays a full subprocess
    start and model load, which is fine for one-off use but wasteful over a corpus.
    """
    return strongest_base_pkas([smiles])[0]


def fraction_protonated(pka: float, ph: float = PHYSIOLOGICAL_PH) -> float:
    """Henderson-Hasselbalch fraction of a base protonated at `ph`.

    For a base, `pKa = pH + log10([B]/[BH+])`, so `[BH+]/([B]+[BH+]) = 1/(1+10^(pH-pKa))`.
    At `pka == ph` this is exactly 0.5 by construction; well above `ph` it saturates
    to 1 (fully protonated, cationic); well below, to 0 (fully neutral).
    """
    return float(1.0 / (1.0 + 10.0 ** (ph - pka)))


def basic_nitrogen_weight(smiles: str, ph: float = PHYSIOLOGICAL_PH) -> float:
    """Fraction of `smiles`'s strongest base protonated at `ph`, or 0.0 if none.

    The direct replacement for `pharmacophore.n_basic_nitrogen > 0`: instead of a
    binary "has a basic nitrogen", this is a continuous 0-1 weight reflecting how
    much of that nitrogen is actually charged at physiological pH -- which is the
    quantity the CYP2D6 pharmacophore mechanism (a cationic amine engaging an acidic
    residue) is actually about. 0.0 for both "no basic site found" and "found one
    but it's essentially never protonated", which is the correct reading for either.
    """
    pka = strongest_base_pka(smiles)
    if pka is None:
        return 0.0
    return fraction_protonated(pka, ph)


def basic_nitrogen_weights(smiles: list[str], ph: float = PHYSIOLOGICAL_PH) -> np.ndarray:
    """`basic_nitrogen_weight` over a list, as a float array in input order.

    One subprocess call for the whole list -- prefer this over a Python loop calling
    `basic_nitrogen_weight` per compound, which would pay a full process start and
    model load per row instead of once for the batch.
    """
    pkas = strongest_base_pkas(list(smiles))
    return np.array(
        [0.0 if pka is None else fraction_protonated(pka, ph) for pka in pkas], dtype=float
    )
