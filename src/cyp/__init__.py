"""Modelling code for the OpenADMET CYP Inhibition Blind Challenge.

Several modules are ported from the PXR blind challenge repo
(https://github.com/adlvdl/pxr_challenge); see each module's docstring for what was
carried over and which of that challenge's lessons shaped it.
"""

from . import (
    calibration,
    constants,
    cv,
    data,
    embedding,
    ensemble,
    evaluation,
    fingerprints,
    mcs,
    metrics,
    mmp,
    scaffolds,
    similarity,
    submission,
)

# `interactive` is deliberately not imported here: it pulls in marimo and Altair,
# which are notebook-only extras. Import it explicitly from a notebook.

__all__ = [
    "calibration",
    "constants",
    "cv",
    "data",
    "embedding",
    "ensemble",
    "evaluation",
    "fingerprints",
    "mcs",
    "metrics",
    "mmp",
    "scaffolds",
    "similarity",
    "submission",
]
