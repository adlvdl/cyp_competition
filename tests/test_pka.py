"""Tests for the vendored MolGpKa pKa predictor (`src/cyp/pka.py`).

Every real prediction runs a subprocess (see `pka.py`'s module docstring for why:
`import cyp` loads LightGBM's OpenMP runtime, and importing torch into that same
process for MolGpKa segfaults, mirroring the TabICL/TabPFN-after-LightGBM failure
CLAUDE.md documents). That makes these tests slower than a typical unit test --
each one that calls `strongest_base_pkas` pays a real subprocess start and a real
6MB-checkpoint load -- but there is no in-process alternative to test against.
"""

from __future__ import annotations

import numpy as np
import pytest

from cyp import pka

# A known strong aliphatic amine (pKa ~9.3 experimentally) and a known aromatic
# system with no basic site, used across several tests as a cheap, stable pair.
QUINIDINE_LIKE = "CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12"
NICOTINE = "CN1CCC[C@H]1c1cccnc1"  # experimental pyrrolidine-N pKa ~8.0
BENZENE = "c1ccccc1"


def test_fraction_protonated_is_half_at_the_pka():
    assert pka.fraction_protonated(7.4, ph=7.4) == pytest.approx(0.5)


def test_fraction_protonated_saturates_for_strong_and_weak_bases():
    # A base far more basic than the local pH is essentially fully protonated.
    assert pka.fraction_protonated(11.0, ph=7.4) == pytest.approx(1.0, abs=1e-3)
    # A base far weaker than the local pH is essentially fully neutral.
    assert pka.fraction_protonated(3.0, ph=7.4) == pytest.approx(0.0, abs=1e-3)


def test_strongest_base_pkas_does_not_segfault_after_import_cyp():
    """The regression this module exists to prevent: `import cyp` loads LightGBM's
    OpenMP runtime, and a subsequent in-process torch import (MolGpKa's dependency)
    segfaults -- confirmed directly while building this module (`exit code 139`).
    Merely completing this call without the test process dying is the assertion;
    the numeric checks below are secondary."""
    result = pka.strongest_base_pkas([NICOTINE])
    assert result[0] is not None


def test_aromatic_only_molecule_has_no_basic_site():
    assert pka.strongest_base_pka(BENZENE) is None
    assert pka.basic_nitrogen_weight(BENZENE) == 0.0


def test_known_amines_predict_plausible_pka():
    """Nicotine's pyrrolidine nitrogen has an experimental pKa of ~8.0; this checks
    the vendored rewrite lands in the right chemical neighbourhood, not an exact
    value -- MolGpKa is itself an approximate model, and this is a plausibility
    check on the vendoring (parameter loading, feature order), not a benchmark of
    MolGpKa's own accuracy."""
    result = pka.strongest_base_pka(NICOTINE)
    assert result is not None
    assert 5.0 < result < 12.0


def test_batched_and_single_calls_agree():
    """The batched subprocess call and the single-compound convenience wrapper must
    return the same answer for the same molecule -- a divergence would mean the
    batching (tab-joining, JSON round-trip) silently corrupts or reorders results."""
    single = pka.strongest_base_pka(NICOTINE)
    batched = pka.strongest_base_pkas([BENZENE, NICOTINE, QUINIDINE_LIKE])
    assert batched[1] == pytest.approx(single)


def test_batch_preserves_input_order():
    """A reordering here would silently pair the wrong compound with the wrong
    pKa in any downstream feature matrix -- the failure mode that matters most,
    since it produces a plausible-looking but wrong result rather than a crash."""
    results = pka.strongest_base_pkas([NICOTINE, BENZENE, QUINIDINE_LIKE])
    assert results[1] is None  # benzene, unambiguously no basic site
    assert results[0] is not None
    assert results[2] is not None


def test_unparseable_smiles_returns_none_not_an_exception():
    results = pka.strongest_base_pkas(["not_a_molecule", NICOTINE])
    assert results[0] is None
    assert results[1] is not None


def test_empty_batch_returns_empty_list():
    assert pka.strongest_base_pkas([]) == []


def test_basic_nitrogen_weights_returns_array_in_input_order():
    weights = pka.basic_nitrogen_weights([BENZENE, NICOTINE, QUINIDINE_LIKE])
    assert isinstance(weights, np.ndarray)
    assert weights.shape == (3,)
    assert weights[0] == 0.0
    # A stronger base should be at least as protonated at physiological pH.
    assert weights[2] >= weights[1] - 1e-6


def test_null_byte_in_smiles_raises_rather_than_silently_dropping():
    """A stray embedded null byte (corrupted upstream data, not a real SMILES
    character) makes `subprocess.run` itself reject the argv with `ValueError`,
    which crashes the *whole batch* rather than just that one row -- worse than the
    graceful per-row None handling for an ordinarily-unparseable SMILES. Pinned as
    a known, documented sharp edge rather than silently accepted: found while
    writing this test, not anticipated in the design."""
    with pytest.raises(ValueError, match="null byte"):
        pka.strongest_base_pkas(["\x00invalid", NICOTINE])
