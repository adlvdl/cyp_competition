"""Tests for `ensemble.caruana_selection`.

Selection is the part of an ensemble that can fail quietly. A selector that spreads
weight evenly regardless of candidate quality still produces a working blend -- it is
just a plain mean wearing a fitted object's clothes, and notebook 14's whole reason
for reporting a uniform control is that the two must be distinguishable. These pin
the two properties that make the difference: a worthless candidate gets no weight,
and the optimisation direction is honoured.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import matthews_corrcoef

from cyp.ensemble import caruana_selection


def test_pure_noise_earns_no_weight():
    """A candidate carrying no signal must be excluded, not averaged in."""
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.random(n) < 0.21
    signal = np.clip(0.5 + 0.8 * (y - 0.5) + rng.normal(0, 0.25, n), 0, 1)
    predictions = {
        "good_a": signal,
        "good_b": np.clip(signal + rng.normal(0, 0.05, n), 0, 1),
        "noise": rng.random(n),
    }

    weights = caruana_selection(
        predictions,
        y,
        lambda truth, prob: matthews_corrcoef(truth, prob >= 0.5),
        maximize=True,
        seed=0,
    )

    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert weights["noise"] < 0.05, weights
    assert weights["good_a"] + weights["good_b"] > 0.9, weights


def test_minimize_selects_for_a_low_score():
    """`maximize=False` must not silently maximize.

    An error metric passed with the wrong flag would select the *worst* candidate
    while still returning a plausible-looking weight vector.
    """
    rng = np.random.default_rng(1)
    y = rng.normal(size=500)
    predictions = {"good": y + rng.normal(0, 0.1, 500), "bad": rng.normal(size=500)}

    def mae(truth: np.ndarray, pred: np.ndarray) -> float:
        return float(np.mean(np.abs(truth - pred)))

    weights = caruana_selection(predictions, y, mae, maximize=False, seed=0)

    assert weights["good"] > weights["bad"], weights
