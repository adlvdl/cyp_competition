"""Modelling code for the OpenADMET CYP Inhibition Blind Challenge.

Several modules are ported from the PXR blind challenge repo
(https://github.com/adlvdl/pxr_challenge); see each module's docstring for what was
carried over and which of that challenge's lessons shaped it.
"""

# ── OpenMP load order (macOS) ───────────────────────────────────────────────────
#
# Three packages in this environment each vendor their own OpenMP runtime:
# lightgbm (lib_lightgbm.dylib), scikit-learn (sklearn/.dylibs/libomp.dylib) and
# torch (torch/lib/libomp.dylib). Whichever loads first wins, and the others bind
# to it. If LightGBM is not first, its `fit` **segfaults** -- a hard crash that
# takes the process down, with no catchable exception:
#
#     from skfp.fingerprints import ECFPFingerprint   # pulls in sklearn's libomp
#     ECFPFingerprint().transform([...])
#     lgb.LGBMRegressor().fit(X, y)                   # <- segmentation fault
#
# Importing lightgbm here, before anything else in the package, makes its runtime
# the one everything binds to. Verified to survive a later `import torch` as well,
# which is what lets a notebook mix tree models with the graph/tabular foundation
# models at all. `KMP_DUPLICATE_LIB_OK=TRUE` does *not* fix this -- it was tried.
#
# Keep this first. Moving it below the package imports reintroduces the crash.
try:
    import lightgbm as _lightgbm  # noqa: F401
except ImportError:  # pragma: no cover -- lightgbm is a hard dependency
    pass

from . import (
    calibration,
    constants,
    cv,
    data,
    embedding,
    ensemble,
    evaluation,
    fingerprints,
    graph_models,
    matrix_factorization,
    mcs,
    metrics,
    mmp,
    multitask,
    scaffolds,
    similarity,
    submission,
    tabular_models,
    timings,
)

# `interactive` is deliberately not imported here: it pulls in marimo and Altair,
# which are notebook-only extras. Import it explicitly from a notebook.
#
# `graph_models`, `tabular_models` and `matrix_factorization` *are* imported, even
# though they need the heavyweight `deep` extra: each defers its torch / chemprop /
# smurff / tabpfn / tabicl import into the function or method that needs it, so the
# modules themselves cost nothing and `import cyp` still works on a core install.
# The ImportError, when it comes, points at the one model you actually tried to use.

__all__ = [
    "calibration",
    "constants",
    "cv",
    "data",
    "embedding",
    "ensemble",
    "evaluation",
    "fingerprints",
    "graph_models",
    "matrix_factorization",
    "mcs",
    "metrics",
    "mmp",
    "multitask",
    "scaffolds",
    "similarity",
    "submission",
    "tabular_models",
    "timings",
]
