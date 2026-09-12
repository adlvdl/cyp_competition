"""Tabular foundation models: TabPFN and TabICL.

These are transformers pretrained on millions of synthetic tabular datasets that
predict by in-context learning -- the training rows are fed in as context at
inference time, so there is no gradient descent and no per-dataset fitting. That
makes them a natural fit for this challenge's regime: 1,285-2,335 rows per endpoint
is exactly the low-data range where they are reported to beat gradient boosting.

## Why both

They fail in different directions, so neither alone answers the question:

| | TabPFN 8.5 | TabICL 2.2 |
|:--|:--|:--|
| feature cap | **500 hard** -- ECFP4 and CheMeleon both need reduction | ~2,000, native |
| regression head | binned classification of the target | 999 quantiles, pinball loss |
| licence | non-commercial, **token-gated** | Apache 2.0, weights download freely |
| molecular evidence | 86.2% win rate w/ CheMeleon | stronger on general tabular |

(Molecular evidence: arXiv 2604.16123, which benchmarks both across seven molecular
representations on 58 Polaris/MoleculeACE tasks.)

The feature cap is the practical divide: `TabPFNRegressor` refuses >500 columns, so
`_reduce_features` PCA-projects first (the literature's standard workaround). TabICL
takes a 2048-bit fingerprint unmodified.

## Representation matters more than the model

arXiv 2604.16123 benchmarks both across seven molecular representations and finds
Morgan/ECFP performs "substantially worse" than CheMeleon embeddings and 2D
descriptors. A TFM on raw ECFP4 is the configuration that paper says not to run.
Pair these with `graph_models.chemeleon_embed` or Mordred descriptors -- ECFP is
included in the sweep only as the control that reproduces that finding.

## TabPFN needs a token

TabPFN 8.x gates its weight download behind a one-time licence acceptance, and it
cannot prompt for that from a notebook. Register at https://ux.priorlabs.ai, accept
the licence, copy the API key, and export `TABPFN_TOKEN` before running. `TabPFNModel`
raises with those instructions rather than letting the underlying error surface
mid-CV. TabICL has no such gate, which is one practical reason it is the default
choice of the two here.

The licence also constrains use: TabPFN weights are non-commercial and explicitly
exclude "internal commercial decision-making". For a public research challenge that
is fine, but it is a reason not to build the *final* pipeline solely on TabPFN.

## The OpenMP collision, and why `predict_subprocess` exists

PyTorch ships its own OpenMP runtime. Once torch is imported into a process, a
subsequent LightGBM fit **segfaults** on macOS -- not an exception, a hard crash that
takes the kernel with it. TabPFN and TabICL are torch models, so calling `fit` on
either poisons the process for any later tree fit. PXR hit this and solved it the
same way: run the torch model in a child process.

`predict_subprocess` does that, and is what a comparison notebook should call, since
mixing these with LightGBM in one session is the normal case. The in-process classes
stay available for a torch-only script, where the collision cannot arise.

## Uncertainty

Both predict a full distribution, not a point estimate, and `predict_interval`
exposes it. That is unusually well matched to this challenge: ST-RAE scores zero
error for any prediction landing inside a compound's `_conf_low`/`_conf_high` band,
so a model that natively emits an interval is doing something the metric rewards
directly -- and CLAUDE.md notes the Innovation award explicitly credits
uncertainty-aware models. TFM intervals are reported to be well calibrated
out-of-the-box, but verify that on our own data with
`evaluation.bias_by_potency_bin` before relying on it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: TabPFN's own hard limit: it refuses inputs wider than 500 columns outright.
TABPFN_MODEL_FEATURE_LIMIT = 500

#: What we actually feed it, which is lower and set by memory. Measured on a 16 GB
#: M-series machine at n=2335 (CYP3A4), PCA-reduced from a 2048-dim embedding:
#:
#:     d=128, est=2    10s    4.50 GB   corr 0.158
#:     d=128, est=8    47s    5.23 GB   corr 0.161
#:     d=192, est=2     9s    6.58 GB   corr 0.232
#:     d=256, est=2    12s    8.44 GB   corr 0.232
#:     d=256, est=4    22s    8.44 GB   corr 0.240
#:     d=500, est=2    51s    7.50 GB   corr 0.269
#:     d=500, est=4   140s   10.83 GB   corr 0.236
#:     d=500, est=8   164s    8.19 GB   corr 0.240
#:
#: TabPFN differs from TabICL in two ways that matter. Ensemble size is nearly free
#: in memory here (+0.7 GB from 2 to 8) where for TabICL it roughly tripled, so the
#: binding constraint is runtime rather than RSS. And the extra features are *used*:
#: corr jumps from ~0.16 at 128 features to ~0.23 by 192 and plateaus there, so
#: cutting TabPFN to TabICL's 128 would discard real signal.
#:
#: 192 is where that curve flattens: it matches d=500's accuracy (0.232 vs 0.236) at
#: 6.6 GB instead of 10.8 GB and 9s instead of 140s. The non-monotonic RSS in the
#: table above (500x8 measuring lower than 500x4) is allocator reuse across runs,
#: not a real inversion -- treat every figure as +/- 1-2 GB and leave headroom.
TABPFN_MAX_FEATURES = 192

#: Ensemble size for TabPFN. Memory barely moves with this, but runtime does
#: (d=256: 12s at 2 estimators, 22s at 4) and accuracy did not improve, so 2 keeps
#: 25 folds x 4 endpoints tractable.
TABPFN_N_ESTIMATORS = 2

#: TabICL has no *hard* feature cap, but it has a brutal soft one. It attends over
#: the feature axis and materializes the whole context at once, so peak memory grows
#: with both feature count and ensemble size -- and it does so fast enough to swap a
#: laptop to a standstill rather than raise MemoryError. Measured on a 16 GB M-series
#: machine at n=1600, n_estimators=2:
#:
#:     d=64      2s    2.0 GB
#:     d=256     7s    6.0 GB
#:     d=512    18s   10.4 GB
#:     d=2048   did not finish in 5 min (swapping)
#:
#: and at n=2335 (CYP3A4, our largest endpoint) on PCA-reduced features:
#:
#:     d=128, n_estimators=8   35s   10.96 GB
#:     d=128, n_estimators=4    9s    7.17 GB
#:     d=128, n_estimators=2    5s    3.89 GB
#:     d=32,  n_estimators=8    8s    5.05 GB
#:
#: Both levers matter and `n_estimators` is the stronger one. On synthetic data with
#: a recoverable signal, accuracy was identical (corr 0.999) across every one of
#: those settings, so the extra memory bought nothing -- which is why the defaults
#: below are conservative rather than maximal.
#:
#: 128 features at 2 estimators leaves ~12 GB free on a 16 GB machine, enough for an
#: editor and a browser. Raise either only after measuring on the target machine, and
#: watch actual RSS while you do -- the failure mode is a frozen desktop, not an
#: exception you can catch.
TABICL_MAX_FEATURES = 128

#: Ensemble size for TabICL. The library default is 8; this is deliberately lower
#: because 8 costs ~11 GB at our largest endpoint (see above) for no measured
#: accuracy gain. Raise it if a machine has the headroom and CV shows it helps.
TABICL_N_ESTIMATORS = 2


#: Above this many (features x estimators), a TabICL fit is expected to exceed the
#: memory of a 16 GB machine. Derived from the measurements on TABICL_MAX_FEATURES:
#: 128x8 = 1024 cost ~11 GB, 128x2 = 256 cost ~3.9 GB.
_TABICL_BUDGET = 512


def check_tabicl_budget(n_features: int, n_estimators: int, strict: bool = True) -> None:
    """Refuse a TabICL configuration likely to exhaust memory.

    This exists because the failure mode is not an exception. TabICL allocates until
    the machine starts swapping, at which point the desktop freezes and the only way
    out is a hard restart -- there is nothing to catch and nothing to report. A
    cheap up-front check is the only useful guard.

    Args:
        n_features: Feature count *after* any PCA reduction.
        n_estimators: Ensemble size.
        strict: Raise when over budget. Set False to warn instead, when you have
            measured headroom on the target machine and accept the risk.

    Raises:
        ValueError: When over budget and `strict`.
    """
    cost = n_features * n_estimators
    if cost <= _TABICL_BUDGET:
        return
    message = (
        f"TabICL configuration likely to exhaust memory: {n_features} features x "
        f"{n_estimators} estimators = {cost}, over the {_TABICL_BUDGET} budget "
        f"(~11 GB was measured at 1024 on a 16 GB machine). This does not fail "
        f"cleanly -- it swaps until the machine locks up. Reduce max_features or "
        f"n_estimators, or pass strict=False if you have measured the headroom."
    )
    if strict:
        raise ValueError(message)
    import warnings

    warnings.warn(message, ResourceWarning, stacklevel=2)


#: Feature count above which a TabPFN fit is expected to need more than a few GB.
#: From the table on TABPFN_MAX_FEATURES: 256 features cost ~8.4 GB, 192 cost ~6.6.
#: Unlike TabICL this does not scale much with ensemble size, so the check is on
#: features alone.
_TABPFN_FEATURE_BUDGET = 256


def check_tabpfn_budget(n_features: int, strict: bool = True) -> None:
    """Warn or refuse when a TabPFN configuration is likely to strain memory.

    Softer than `check_tabicl_budget` on purpose: TabPFN's growth is gradual and it
    hard-fails above `TABPFN_MODEL_FEATURE_LIMIT` anyway, so this is about staying
    off the swap cliff rather than preventing a lock-up. It still matters on a 16 GB
    machine, where 500 features measured 10.8 GB with an editor open.

    Args:
        n_features: Feature count after any PCA reduction.
        strict: Raise when over budget; False warns instead.

    Raises:
        ValueError: When over budget and `strict`.
    """
    if n_features > TABPFN_MODEL_FEATURE_LIMIT:
        raise ValueError(
            f"TabPFN accepts at most {TABPFN_MODEL_FEATURE_LIMIT} features, got "
            f"{n_features}. Reduce with PCA first -- `TabPFNModel` does this "
            f"automatically via its `max_features` argument."
        )
    if n_features <= _TABPFN_FEATURE_BUDGET:
        return
    message = (
        f"TabPFN with {n_features} features is memory-hungry: ~10.8 GB was measured "
        f"at 500 features on a 16 GB machine, against ~6.6 GB at "
        f"{TABPFN_MAX_FEATURES}. Accuracy plateaued by {TABPFN_MAX_FEATURES} in that "
        f"test, so the extra width may buy nothing. Pass strict=False to proceed."
    )
    if strict:
        raise ValueError(message)
    import warnings

    warnings.warn(message, ResourceWarning, stacklevel=2)


def load_env(path: str | None = None) -> bool:
    """Load `.env` from the project root into the environment, if present.

    Exists so `TABPFN_TOKEN` reaches the process without every notebook and shell
    having to `source .env` first. Values already set in the environment win, so an
    explicit `export` still overrides the file.

    Returns:
        True if a `.env` file was found and read.
    """
    import os

    from . import constants as C

    env_path = Path(path) if path else C.PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return False

    try:
        from dotenv import load_dotenv
    except ImportError:
        # Minimal fallback: python-dotenv arrives via the `deep` extra, so a core
        # install parses the handful of KEY=value lines itself rather than failing.
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
        return True

    load_dotenv(env_path, override=False)
    return True


def _require_tabpfn_token() -> None:
    """Fail fast, with instructions, when TabPFN's licence token is missing.

    TabPFN 8.x will not download weights without a one-time licence acceptance and
    cannot prompt for it non-interactively. Without this check the failure surfaces
    as a `TabPFNLicenseError` from inside the first CV fold, several minutes into a
    run; better to refuse before any folds are spent.

    Reads `.env` first, so a token stored there works without sourcing it by hand.

    Note the token must be the **API key** from the account page (it looks like
    `tabpfn_sk_...`), not a session JWT copied from the browser -- the server
    returns 401 "Invalid authentication credentials" for the latter, which is
    indistinguishable from having no token at all until you check the HTTP status.

    Skipped when a local checkpoint is already configured, since that path needs no
    download.
    """
    import os

    if os.environ.get("TABPFN_TOKEN") or os.environ.get("TABPFN_MODEL_CACHE_DIR"):
        return

    load_env()
    if os.environ.get("TABPFN_TOKEN") or os.environ.get("TABPFN_MODEL_CACHE_DIR"):
        return
    raise RuntimeError(
        "TabPFN needs a one-time licence acceptance before it can download weights.\n"
        "  1. Register and log in at https://ux.priorlabs.ai\n"
        "  2. Accept the licence on the Licenses tab\n"
        "  3. Copy the API key from https://ux.priorlabs.ai/account\n"
        "  4. export TABPFN_TOKEN=\"<your-api-key>\" (or put it in .env)\n"
        "     It must be the API key (tabpfn_sk_...), not a browser session token.\n"
        "TabICL (`tabicl` method) needs no token and is the Apache-2.0 alternative."
    )


def _reduce_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    max_features: int = TABPFN_MAX_FEATURES,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """PCA-project to `max_features` columns when `X_train` is wider than that.

    Fitted on the training fold only and applied to the test fold, so no test-set
    variance structure leaks into the projection. Returns the inputs untouched when
    they already fit, so the common narrow-descriptor case costs nothing.

    PCA rather than variance-based feature selection: on a sparse bit vector the
    highest-variance bits are just the most common substructures, which throws away
    the rare-substructure signal that distinguishes analogs -- precisely what matters
    on a test set of 75 hits x ~10 analogs each.
    """
    n_components = min(max_features, X_train.shape[0], X_train.shape[1])
    if X_train.shape[1] <= max_features:
        return X_train, X_test

    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, random_state=seed)
    return pca.fit_transform(X_train), pca.transform(X_test)


class TabPFNModel:
    """TabPFN regressor, with automatic PCA reduction to its 500-feature ceiling.

    Runs on CPU by default. PXR pinned TabPFN to CPU deliberately: MPS hits its
    allocator ceiling partway through a CV run, and the in-context forward pass is
    fast enough on CPU at this dataset size that the trade is not worth debugging
    mid-competition.

    There is no `train()` in the usual sense -- `fit` just stores the context, and
    all the work happens in `predict`.
    """

    def __init__(
        self,
        n_estimators: int = TABPFN_N_ESTIMATORS,
        device: str = "cpu",
        max_features: int = TABPFN_MAX_FEATURES,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        """
        Args:
            n_estimators: Ensemble members, each a different feature/sample
                permutation of the context. PXR used 8.
            device: "cpu", "mps" or "cuda". See the class note on why CPU is default.
            max_features: PCA target when the input is wider. TabPFN hard-fails above
                500, so raising this will not work.
            random_state: Seed for the ensemble permutations and the PCA.
            **kwargs: Forwarded to `TabPFNRegressor`.
        """
        self.n_estimators = n_estimators
        self.device = device
        self.max_features = max_features
        self.random_state = random_state
        self.kwargs = kwargs
        self._model = None
        self._pca = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> TabPFNModel:
        _require_tabpfn_token()
        from tabpfn import TabPFNRegressor

        X = np.asarray(X, dtype=np.float32)
        # The projection is fitted here and reused in predict(), so train and test
        # land in the same subspace.
        if X.shape[1] > self.max_features:
            from sklearn.decomposition import PCA

            n_components = min(self.max_features, X.shape[0], X.shape[1])
            self._pca = PCA(n_components=n_components, random_state=self.random_state)
            X = self._pca.fit_transform(X)

        check_tabpfn_budget(X.shape[1])

        self._model = TabPFNRegressor(
            n_estimators=self.n_estimators,
            device=self.device,
            random_state=self.random_state,
            # Our row counts sit inside TabPFN's pretraining range, but fingerprint
            # width can still trip its heuristics; the PCA above keeps us legal.
            ignore_pretraining_limits=True,
            **self.kwargs,
        )
        self._model.fit(X, np.asarray(y, dtype=np.float32))
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return self._pca.transform(X) if self._pca is not None else X

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Posterior mean of the predicted distribution."""
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        return np.asarray(self._model.predict(self._transform(X)), dtype=float).flatten()

    def predict_interval(self, X: np.ndarray, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Central `1 - 2*alpha` predictive interval, as `(lower, upper)`.

        The point of exposing this separately from `predict`: ST-RAE forgives any
        error falling inside a compound's credible interval, so a calibrated
        predictive interval is directly actionable here rather than decoration.
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict_interval().")
        quantiles = self._model.predict(
            self._transform(X), output_type="quantiles", quantiles=[alpha, 1 - alpha]
        )
        return (
            np.asarray(quantiles[0], dtype=float).flatten(),
            np.asarray(quantiles[1], dtype=float).flatten(),
        )


class TabICLModel:
    """TabICL regressor -- no feature cap, quantile-trained regression head.

    Preferred over `TabPFNModel` when the representation is wide (a 2048-bit
    fingerprint or a CheMeleon embedding) because it consumes those natively instead
    of through a PCA bottleneck, and its head is trained on the pinball loss across
    999 quantiles rather than by binning the target -- which should make
    `predict_interval` better behaved in the sparse potent tail, where CLAUDE.md
    records LightGBM underpredicting by ~1 log unit.

    Apache-2.0 weights, so no licence question attaches to a submission built on it.
    """

    def __init__(
        self,
        n_estimators: int = TABICL_N_ESTIMATORS,
        device: str = "cpu",
        max_features: int = TABICL_MAX_FEATURES,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        """
        Args:
            n_estimators: Ensemble members (normalization x feature-permutation views).
            device: "cpu", "mps" or "cuda". CPU default for the same reason as TabPFN.
            max_features: PCA target when the input is wider. This is a *memory*
                limit, not a model limit -- see `TABICL_MAX_FEATURES` for the
                measurements. Passing a 2048-dim embedding unreduced will swap a
                16 GB machine to a standstill.
            random_state: Seed for the ensemble views and the PCA.
            **kwargs: Forwarded to `TabICLRegressor`.
        """
        self.n_estimators = n_estimators
        self.device = device
        self.max_features = max_features
        self.random_state = random_state
        self.kwargs = kwargs
        self._model = None
        self._pca = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> TabICLModel:
        from tabicl import TabICLRegressor

        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] > self.max_features:
            from sklearn.decomposition import PCA

            n_components = min(self.max_features, X.shape[0], X.shape[1])
            self._pca = PCA(n_components=n_components, random_state=self.random_state)
            X = self._pca.fit_transform(X).astype(np.float32)

        check_tabicl_budget(X.shape[1], self.n_estimators)

        self._model = TabICLRegressor(
            n_estimators=self.n_estimators,
            device=self.device,
            random_state=self.random_state,
            **self.kwargs,
        )
        self._model.fit(X, np.asarray(y, dtype=np.float32))
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return self._pca.transform(X).astype(np.float32) if self._pca is not None else X

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        preds = self._model.predict(self._transform(X))
        return np.asarray(preds, dtype=float).flatten()

    def predict_interval(self, X: np.ndarray, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        """Central `1 - 2*alpha` predictive interval, as `(lower, upper)`."""
        if self._model is None:
            raise RuntimeError("Call fit() before predict_interval().")
        # Note the axis order difference from TabPFN: TabICL returns a single
        # (n_test, n_quantiles) array, while TabPFN returns a list of (n_test,)
        # arrays, one per quantile. Hence the column indexing here and the element
        # indexing there.
        quantiles = np.asarray(
            self._model.predict(
                self._transform(X),
                output_type="quantiles",
                alphas=[alpha, 1 - alpha],
            ),
            dtype=float,
        )
        return quantiles[:, 0].flatten(), quantiles[:, 1].flatten()


class TabPFNClassifierModel:
    """TabPFN classifier for the TDI track.

    `predict_proba` matters more than `predict` here: TDI is ~21% positive and MCC
    punishes a majority-class predictor, so the decision threshold needs tuning on
    probabilities rather than being left at 0.5. TabPFN handles class imbalance
    without explicit reweighting -- the synthetic pretraining covers imbalanced
    priors -- so there is no `class_weight` analogue to set.
    """

    def __init__(
        self,
        n_estimators: int = TABPFN_N_ESTIMATORS,
        device: str = "cpu",
        max_features: int = TABPFN_MAX_FEATURES,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        self.n_estimators = n_estimators
        self.device = device
        self.max_features = max_features
        self.random_state = random_state
        self.kwargs = kwargs
        self._model = None
        self._pca = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> TabPFNClassifierModel:
        _require_tabpfn_token()
        from tabpfn import TabPFNClassifier

        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] > self.max_features:
            from sklearn.decomposition import PCA

            n_components = min(self.max_features, X.shape[0], X.shape[1])
            self._pca = PCA(n_components=n_components, random_state=self.random_state)
            X = self._pca.fit_transform(X)

        self._model = TabPFNClassifier(
            n_estimators=self.n_estimators,
            device=self.device,
            random_state=self.random_state,
            ignore_pretraining_limits=True,
            **self.kwargs,
        )
        self._model.fit(X, np.asarray(y).astype(int))
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return self._pca.transform(X) if self._pca is not None else X

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        return np.asarray(self._model.predict(self._transform(X))).astype(bool)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Positive-class probability, for MCC-optimal threshold tuning."""
        if self._model is None:
            raise RuntimeError("Call fit() before predict_proba().")
        return np.asarray(self._model.predict_proba(self._transform(X)), dtype=float)[:, 1]


class TabICLClassifierModel:
    """TabICL classifier for the TDI track. See `TabPFNClassifierModel` on why
    `predict_proba` is the method that matters for MCC."""

    def __init__(
        self,
        n_estimators: int = TABICL_N_ESTIMATORS,
        device: str = "cpu",
        max_features: int = TABICL_MAX_FEATURES,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        self.n_estimators = n_estimators
        self.device = device
        self.max_features = max_features
        self.random_state = random_state
        self.kwargs = kwargs
        self._model = None
        self._pca = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> TabICLClassifierModel:
        from tabicl import TabICLClassifier

        X = np.asarray(X, dtype=np.float32)
        # Same memory cap as the regressor -- see TABICL_MAX_FEATURES.
        if X.shape[1] > self.max_features:
            from sklearn.decomposition import PCA

            n_components = min(self.max_features, X.shape[0], X.shape[1])
            self._pca = PCA(n_components=n_components, random_state=self.random_state)
            X = self._pca.fit_transform(X).astype(np.float32)

        self._model = TabICLClassifier(
            n_estimators=self.n_estimators,
            device=self.device,
            random_state=self.random_state,
            **self.kwargs,
        )
        self._model.fit(X, np.asarray(y).astype(int))
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return self._pca.transform(X).astype(np.float32) if self._pca is not None else X

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        return np.asarray(self._model.predict(self._transform(X))).astype(bool)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Positive-class probability, for MCC-optimal threshold tuning."""
        if self._model is None:
            raise RuntimeError("Call fit() before predict_proba().")
        proba = self._model.predict_proba(self._transform(X))
        return np.asarray(proba, dtype=float)[:, 1]


# ── subprocess isolation ────────────────────────────────────────────────────────

#: Child script for `predict_subprocess`. Kept as a string rather than a module file
#: so there is exactly one copy of this logic and no import-path assumptions in the
#: child. It writes predictions (and optionally interval bounds) to .npy files.
_TFM_SCRIPT = r"""
import json, sys
import numpy as np

payload = json.loads(sys.argv[1])
X_train = np.load(payload["x_train"])
y_train = np.load(payload["y_train"])
X_test = np.load(payload["x_test"])
kind = payload["kind"]
n_estimators = payload["n_estimators"]
device = payload["device"]
alpha = payload["alpha"]
lower = upper = None

if kind == "tabicl":
    from sklearn.decomposition import PCA
    from tabicl import TabICLRegressor

    # Memory cap, not a model cap: TabICL attends over the feature axis and needs
    # roughly 20 MB per feature, so an unreduced 2048-dim embedding would swap the
    # machine rather than raise. See TABICL_MAX_FEATURES in the parent module.
    max_features = payload["tabicl_max_features"]
    if X_train.shape[1] > max_features:
        pca = PCA(
            n_components=min(max_features, *X_train.shape), random_state=payload["seed"]
        )
        X_train = pca.fit_transform(X_train).astype("float32")
        X_test = pca.transform(X_test).astype("float32")

    model = TabICLRegressor(
        n_estimators=n_estimators, device=device, random_state=payload["seed"]
    )
    model.fit(X_train, y_train)
    preds = np.asarray(model.predict(X_test), dtype=float).flatten()
    if payload["want_interval"]:
        q = np.asarray(
            model.predict(X_test, output_type="quantiles", alphas=[alpha, 1 - alpha]),
            dtype=float,
        )
        lower, upper = q[:, 0].flatten(), q[:, 1].flatten()
elif kind == "tabpfn":
    from sklearn.decomposition import PCA
    from tabpfn import TabPFNRegressor

    max_features = payload["max_features"]
    if X_train.shape[1] > max_features:
        pca = PCA(
            n_components=min(max_features, *X_train.shape), random_state=payload["seed"]
        )
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    model = TabPFNRegressor(
        n_estimators=n_estimators,
        device=device,
        random_state=payload["seed"],
        ignore_pretraining_limits=True,
    )
    model.fit(X_train, y_train)
    preds = np.asarray(model.predict(X_test), dtype=float).flatten()
    if payload["want_interval"]:
        q = model.predict(X_test, output_type="quantiles", quantiles=[alpha, 1 - alpha])
        lower = np.asarray(q[0], dtype=float).flatten()
        upper = np.asarray(q[1], dtype=float).flatten()
else:
    raise ValueError(f"Unknown kind: {kind}")

np.save(payload["out_pred"], preds)
if payload["want_interval"]:
    np.save(payload["out_lower"], lower)
    np.save(payload["out_upper"], upper)
"""


def predict_subprocess(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    kind: str = "tabicl",
    n_estimators: int | None = None,
    device: str = "cpu",
    seed: int = 42,
    want_interval: bool = False,
    alpha: float = 0.1,
    max_features: int = TABPFN_MAX_FEATURES,
    tabicl_max_features: int = TABICL_MAX_FEATURES,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Fit a tabular foundation model and predict, in an isolated child process.

    Use this instead of `TabICLModel`/`TabPFNModel` whenever the same process also
    fits LightGBM or XGBoost -- which is the normal case in a comparison notebook.
    Importing torch into a process makes a later LightGBM fit segfault on macOS
    (see the module docstring), and a child process is the only reliable fix: the
    torch runtime is loaded, used and discarded without ever touching the parent.

    Args:
        X_train: Training features.
        y_train: Training targets.
        X_test: Test features.
        kind: "tabicl" or "tabpfn".
        n_estimators: Ensemble members. None picks the kind's default -- deliberately
            low for TabICL, where 8 costs ~11 GB at our largest endpoint.
        device: "cpu", "mps" or "cuda".
        seed: Random seed.
        want_interval: Also return the central `1 - 2*alpha` interval bounds.
        alpha: Tail probability for the interval.
        max_features: PCA cap for TabPFN (its hard 500-feature model limit).
        tabicl_max_features: PCA cap for TabICL. This one is a *memory* limit --
            TabICL accepts wide input but needs ~20 MB per feature, so leaving a
            2048-dim embedding unreduced will freeze a 16 GB machine.

    Returns:
        `(predictions, lower, upper)`; `lower`/`upper` are None unless
        `want_interval` is set.
    """
    if kind not in ("tabicl", "tabpfn"):
        raise ValueError(f"kind must be 'tabicl' or 'tabpfn', not {kind!r}")
    if kind == "tabpfn":
        _require_tabpfn_token()
    if n_estimators is None:
        n_estimators = (
            TABICL_N_ESTIMATORS if kind == "tabicl" else TABPFN_N_ESTIMATORS
        )
    # Check before spawning. For TabICL especially, the child would swap the machine
    # and a frozen desktop reports nothing back to this process.
    effective_features = (
        min(tabicl_max_features, np.shape(X_train)[1])
        if kind == "tabicl"
        else min(max_features, np.shape(X_train)[1])
    )
    if kind == "tabicl":
        check_tabicl_budget(effective_features, n_estimators)
    else:
        check_tabpfn_budget(effective_features)

    import json
    import os
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        script = tmp / "tfm_predict.py"
        script.write_text(_TFM_SCRIPT)

        paths = {name: tmp / f"{name}.npy" for name in ("x_train", "y_train", "x_test")}
        np.save(paths["x_train"], np.asarray(X_train, dtype=np.float32))
        np.save(paths["y_train"], np.asarray(y_train, dtype=np.float32))
        np.save(paths["x_test"], np.asarray(X_test, dtype=np.float32))

        payload = {
            "x_train": str(paths["x_train"]),
            "y_train": str(paths["y_train"]),
            "x_test": str(paths["x_test"]),
            "out_pred": str(tmp / "pred.npy"),
            "out_lower": str(tmp / "lower.npy"),
            "out_upper": str(tmp / "upper.npy"),
            "kind": kind,
            "n_estimators": n_estimators,
            "device": device,
            "seed": seed,
            "want_interval": want_interval,
            "alpha": alpha,
            "max_features": max_features,
            "tabicl_max_features": tabicl_max_features,
        }

        # The child inherits this process's environment, which is what carries
        # TABPFN_TOKEN through -- `_require_tabpfn_token` above has already loaded
        # .env into it if needed, so the child never has to find the file itself.
        result = subprocess.run(
            [sys.executable, str(script), json.dumps(payload)],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        if result.returncode != 0:
            raise RuntimeError(f"{kind} subprocess failed:\n{result.stderr[-2000:]}")

        preds = np.load(payload["out_pred"])
        if not want_interval:
            return preds, None, None
        return preds, np.load(payload["out_lower"]), np.load(payload["out_upper"])


def run_cv_subprocess(
    frame,
    endpoint: str,
    features: np.ndarray,
    kind: str = "tabicl",
    n_outer: int = 5,
    n_inner: int = 5,
    seed: int = 42,
    n_estimators: int | None = None,
    device: str = "cpu",
    want_interval: bool = False,
    tabicl_max_features: int = TABICL_MAX_FEATURES,
    on_fold=None,
) -> "object":
    """Nested scaffold CV for a tabular foundation model, one fold per subprocess.

    The OpenMP-safe counterpart to `models.run_cv` for TFMs. `run_cv` would import
    torch into the calling kernel, after which any LightGBM fit in the same notebook
    segfaults -- so every fold is fitted in a child process instead.

    Returns the same long OOF frame schema as `models.run_cv`, so `evaluation`,
    `calibration` and `ensemble` consume it identically. With `want_interval`, two
    extra columns `pred_lower`/`pred_upper` carry the model's own 80% predictive
    interval -- distinct from `y_lower`/`y_upper`, which are the *assay's* credible
    interval that ST-RAE scores against.

    Args:
        frame: Model-ready frame from `data.training_frame`.
        endpoint: Endpoint name, recorded in the OOF frame.
        features: Row-aligned feature matrix (CheMeleon embeddings recommended).
        kind: "tabicl" or "tabpfn".
        n_outer: Outer CV repeats.
        n_inner: Folds per repeat.
        seed: Split seed.
        n_estimators: Ensemble members per fit. None picks the kind's default.
        device: Device for the child process.
        want_interval: Also record predictive interval bounds.
        tabicl_max_features: PCA cap applied per fold for TabICL. Fitted inside the
            fold, so the projection never sees the held-out rows.
        on_fold: Optional `(fold, n_folds) -> None` progress callback; see
            `cv.FoldCallback`. Each fold spawns a subprocess, so without this a
            multi-fold run shows nothing until it finishes.

    Returns:
        Long OOF frame, one row per (fold, compound).
    """
    import polars as pl

    from .cv import report_fold, scaffold_splits

    if len(features) != frame.height:
        raise ValueError(
            f"features has {len(features)} rows but frame has {frame.height}; "
            "they must be row-aligned."
        )

    features = np.asarray(features)
    indexed = frame.with_row_index("_row")
    n_folds = n_outer * n_inner
    records: list[dict] = []

    for fold, outer, inner, train, _val, test in scaffold_splits(
        indexed, n_outer=n_outer, n_inner=n_inner, seed=seed
    ):
        preds, lower, upper = predict_subprocess(
            features[train["_row"].to_numpy()],
            train["y_true"].to_numpy(),
            features[test["_row"].to_numpy()],
            kind=kind,
            n_estimators=n_estimators,
            device=device,
            seed=seed,
            want_interval=want_interval,
            tabicl_max_features=tabicl_max_features,
        )

        for i in range(test.height):
            record = {
                "method": kind,
                "endpoint": endpoint,
                "fold": fold,
                "outer_fold": outer,
                "inner_fold": inner,
                "Molecule_Name": test["Molecule_Name"][i],
                "y_true": float(test["y_true"][i]),
                "y_pred": float(preds[i]),
                "y_lower": float(test["y_lower"][i])
                if test["y_lower"][i] is not None
                else None,
                "y_upper": float(test["y_upper"][i])
                if test["y_upper"][i] is not None
                else None,
            }
            if want_interval:
                record["pred_lower"] = float(lower[i])
                record["pred_upper"] = float(upper[i])
            records.append(record)

        report_fold(on_fold, fold, n_folds)

    return pl.DataFrame(records)
