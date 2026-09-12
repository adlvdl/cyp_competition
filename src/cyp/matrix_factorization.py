"""Macau: Bayesian matrix factorization with fingerprints as side information.

Ported from the PXR repo's `marimo_notebooks/4_ml_optimization_2.py`. Macau extends
BPMF with side information: the fingerprint matrix enters as row-side features, and
Gibbs sampling jointly infers latent compound factors `U`, latent target factors `V`,
and a link matrix `Beta` connecting fingerprints to `U`. Prediction for an unseen
compound imputes its `U` through `Beta`, then computes `U . V`.

## Why it earns a place in the CYP comparison specifically

CLAUDE.md's "Four targets, four different training sets": the four regression
endpoints do not share rows, because a compound only has a pIC50 where the primary
screen flagged it. Per-endpoint counts run 1,285-2,335 out of 4,905 rows, so a dense
4-column target matrix does not exist and `dropna()` leaves almost nothing.

A sparse matrix is the native input format for matrix factorization -- missing
entries are simply absent, not imputed. Macau therefore borrows strength across
endpoints without needing the dense overlap that a conventional multitask model
assumes, which is a structurally better fit for this data than it was for PXR's
single endpoint. `multitask_target_matrix` builds that sparse matrix.

That said, PXR's retrospective records multitask/auxiliary training *hurting* there,
so treat this as an experiment with a control (single-task Macau, and the tree
baselines), not an assumed win.

## Known failure mode

PXR saw occasional NaN predictions from smurff under numerical instability -- its
comparison code drops those rows explicitly. Check for NaN before scoring rather
than letting them propagate into a mean.
"""

from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
import polars as pl
import scipy.sparse as sp


class MacauModel:
    """Bayesian matrix factorization (Macau via smurff), fingerprints as side info.

    Fully Bayesian: predictions are the average over posterior Gibbs samples taken
    after burn-in, which means the posterior spread is available as a free
    uncertainty estimate -- see `predict_std`, which matters here because ST-RAE
    forgives errors inside a compound's credible interval.

    Operates on a numeric feature matrix, so it must be paired with a precomputed
    fingerprint or embedding rather than raw SMILES.
    """

    def __init__(
        self,
        num_latent: int = 16,
        burnin: int = 100,
        nsamples: int = 200,
        univariate: bool = False,
        direct: bool = True,
        num_threads: int | None = None,
        seed: int = 42,
    ) -> None:
        """
        Args:
            num_latent: Latent dimensions. Higher captures more structure at the cost
                of compute and overfitting risk.
            burnin: Gibbs iterations discarded before collecting samples.
            nsamples: Posterior samples collected; predictions average over these.
            univariate: Use the faster univariate sampler instead of the joint
                multivariate one. Faster on large data, may converge more slowly.
            direct: Cholesky (direct) solver for the link matrix; otherwise conjugate
                gradient. Direct is faster for small side-info matrices, CG scales
                better to very wide fingerprints.
            num_threads: OpenMP threads; None lets smurff decide.
            seed: Gibbs sampler seed.
        """
        self.num_latent = num_latent
        self.burnin = burnin
        self.nsamples = nsamples
        self.univariate = univariate
        self.direct = direct
        self.num_threads = num_threads
        self.seed = seed
        self._predict_session = None
        self._n_targets = 1

    def fit(self, X: np.ndarray, y: np.ndarray | sp.spmatrix) -> MacauModel:
        """Fit by Gibbs sampling, with `X` as row side information.

        Accepts either a 1-D target array (single-task) or a sparse `(n, k)` matrix
        (multitask, from `multitask_target_matrix`). In the single-task case a sparse
        `(n, 1)` COO matrix is built automatically with NaN entries omitted -- which
        is what lets a partially-labelled endpoint column pass through untouched.
        """
        import smurff

        X = np.asarray(X, dtype=np.float64)

        if sp.issparse(y):
            Y_train = y.astype(np.float64)
        else:
            values = np.asarray(y, dtype=np.float64).flatten()
            # Omit rather than impute: an absent entry is information-free to the
            # sampler, whereas an imputed one would be a fabricated observation.
            mask = ~np.isnan(values)
            Y_train = sp.coo_matrix(
                (values[mask], (np.where(mask)[0], np.zeros(int(mask.sum()), dtype=int))),
                shape=(len(values), 1),
            )
        self._n_targets = Y_train.shape[1]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_name = os.path.join(tmpdir, "smurff_model.hdf5")
            # smurff is noisy on two levels: verbose=0 quiets the C++ layer, and
            # raising the Python loggers suppresses its INFO lines. session.run()
            # would also spawn a tqdm bar per fold, so we drive init()/step()
            # directly instead -- equivalent, minus the progress bar.
            smurff_logger = logging.getLogger("smurff")
            root_logger = logging.getLogger()
            prev_smurff, prev_root = smurff_logger.level, root_logger.level
            smurff_logger.setLevel(logging.ERROR)
            root_logger.setLevel(logging.ERROR)
            try:
                session = smurff.MacauSession(
                    Ytrain=Y_train,
                    # Side info per matrix dimension: fingerprints on rows
                    # (compounds), nothing on columns (endpoints).
                    side_info=[X, None],
                    num_latent=self.num_latent,
                    burnin=self.burnin,
                    nsamples=self.nsamples,
                    univariate=self.univariate,
                    direct=self.direct,
                    num_threads=self.num_threads,
                    seed=self.seed,
                    verbose=0,
                    save_name=save_name,
                    save_freq=1,
                )
                session.init()
                while session.step():
                    pass
            finally:
                smurff_logger.setLevel(prev_smurff)
                root_logger.setLevel(prev_root)
            self._predict_session = session.makePredictSession()
        return self

    def _posterior_samples(self, X: np.ndarray) -> np.ndarray:
        """Raw posterior samples, shape `(nsamples, n_test, n_targets)`."""
        if self._predict_session is None:
            raise RuntimeError("Call fit() before predict().")
        samples = self._predict_session.predict((np.asarray(X, dtype=np.float64), slice(None)))
        return np.asarray(samples, dtype=float)

    def predict(self, X: np.ndarray, target: int = 0) -> np.ndarray:
        """Posterior-mean prediction for one target column.

        Args:
            X: Test fingerprint matrix.
            target: Which target column to return. Always 0 for single-task; for a
                multitask fit this indexes the endpoint order used to build the
                target matrix.
        """
        samples = self._posterior_samples(X)
        mean = samples.mean(axis=0)
        return np.asarray(mean, dtype=float).reshape(len(np.atleast_2d(X)), -1)[:, target]

    def predict_std(self, X: np.ndarray, target: int = 0) -> np.ndarray:
        """Posterior standard deviation -- Macau's free uncertainty estimate.

        The spread across Gibbs samples is a genuine posterior, not a heuristic, so
        it is worth checking against the challenge's credible intervals: a model that
        knows which compounds it is unsure about can be ensembled or shrunk toward
        the mean selectively, which is what ST-RAE rewards.
        """
        samples = self._posterior_samples(X)
        std = samples.std(axis=0)
        return np.asarray(std, dtype=float).reshape(len(np.atleast_2d(X)), -1)[:, target]


def multitask_target_matrix(
    frames: dict[str, pl.DataFrame],
    key: str = "Molecule_Name",
    target_col: str = "y_true",
) -> tuple[sp.coo_matrix, list[str], list[str]]:
    """Build the sparse `(n_compounds, n_endpoints)` target matrix Macau needs.

    This exists because of the data fact in CLAUDE.md that shapes every multitask
    decision here: the four regression endpoints do not share rows. A compound
    measured for CYP3A4 usually has no CYP2D6 value, so the dense matrix a
    conventional multitask model wants does not exist. A sparse matrix represents
    that honestly -- an unmeasured (compound, endpoint) pair is simply an absent
    entry, and the sampler never sees a fabricated zero.

    Args:
        frames: Endpoint name -> its labelled subset, each with `key` and `target_col`.
            Use `data.labelled_subset` per endpoint to build these.
        key: Compound identifier column, used to align rows across endpoints.
        target_col: Column holding the measured value.

    Returns:
        `(Y, compounds, endpoints)` -- the sparse matrix, the row order as compound
        identifiers, and the column order as endpoint names. Keep `endpoints` around:
        it is the index `MacauModel.predict(..., target=i)` expects.
    """
    endpoints = list(frames)
    # Union of every compound seen in any endpoint, sorted for a deterministic row
    # order across runs.
    compounds = sorted({name for frame in frames.values() for name in frame[key].to_list()})
    row_of = {name: i for i, name in enumerate(compounds)}

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for col, endpoint in enumerate(endpoints):
        frame = frames[endpoint]
        for name, value in zip(frame[key].to_list(), frame[target_col].to_list(), strict=True):
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            rows.append(row_of[name])
            cols.append(col)
            values.append(float(value))

    Y = sp.coo_matrix(
        (values, (rows, cols)), shape=(len(compounds), len(endpoints)), dtype=np.float64
    )
    return Y, compounds, endpoints
