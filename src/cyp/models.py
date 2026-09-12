"""Baseline models and the CV harnesses that run them.

Deliberately light: LightGBM/XGBoost/RF on fingerprints train in minutes, which is
what gets a valid submission banked early. The PXR result says graph models
(Chemprop, CheMeleon) will beat all of these -- CheMeleon won every PXR CV comparison
-- so treat this as the floor to beat, not the destination.

Two baselines are included on purpose. PXR anchored every comparison against
predict-the-mean and 1-NN, which is how you find out whether a model has learned
anything at all. Under ST-RAE the mean predictor scores exactly 1.0, so any model
scoring above 1.0 is actively worse than a constant.

Regression (direct inhibition, `run_cv`/`fit_predict_test`) and classification (TDI,
`run_cv_classification`/`fit_predict_test_classification`) are separate code paths
rather than one branching function: their OOF schemas differ (TDI has no credible
intervals to carry, and predictions are booleans, not floats), and keeping them
separate matches the fact that they are scored by entirely different metrics
(ST-RAE vs MCC).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge

from . import fingerprints
from .cv import report_fold, scaffold_splits


class MeanBaseline:
    """Predict the training mean for every compound. Scores ST-RAE = 1.0 by
    construction -- the line every real model must clear."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> MeanBaseline:
        self.value_ = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.value_)


class NearestNeighbourBaseline:
    """1-NN by Tanimoto similarity on binary fingerprints.

    A strong baseline on analog-rich sets like this one: if a model cannot beat
    "copy the most similar training compound", it is not adding chemistry.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> NearestNeighbourBaseline:
        self.X_ = np.asarray(X, dtype=bool)
        self.y_ = np.asarray(y, dtype=float)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        query = np.asarray(X, dtype=bool)
        # Tanimoto = |A & B| / |A| + |B| - |A & B|, vectorized over the training set.
        intersection = query.astype(np.float32) @ self.X_.T.astype(np.float32)
        query_bits = query.sum(axis=1, keepdims=True)
        train_bits = self.X_.sum(axis=1, keepdims=True).T
        union = query_bits + train_bits - intersection
        similarity = np.divide(
            intersection, union, out=np.zeros_like(intersection), where=union > 0
        )
        return self.y_[np.argmax(similarity, axis=1)]


def _lightgbm(**kwargs):
    from lightgbm import LGBMRegressor

    params = {"n_estimators": 500, "learning_rate": 0.05, "verbose": -1, "n_jobs": -1}
    return LGBMRegressor(**(params | kwargs))


def _xgboost(**kwargs):
    from xgboost import XGBRegressor

    params = {"n_estimators": 500, "learning_rate": 0.05, "n_jobs": -1, "tree_method": "hist"}
    return XGBRegressor(**(params | kwargs))


def _chemprop():
    from .graph_models import ChempropModel

    return ChempropModel()


def _chemeleon():
    from .graph_models import ChempropChemeleonModel

    return ChempropChemeleonModel()


def _tabpfn():
    from .tabular_models import TabPFNModel

    return TabPFNModel()


def _tabicl():
    from .tabular_models import TabICLModel

    return TabICLModel()


def _macau():
    from .matrix_factorization import MacauModel

    return MacauModel()


#: Model factories. PXR found HPO paid off substantially for tree models
#: (XGBoost 0.64 -> 0.52 MAE) and barely at all for neural nets, so tune here first.
#:
#: Everything from "chemprop" down needs the `deep` extra (`uv sync --all-extras`).
#: Each is imported inside its factory rather than at module scope so that
#: `import cyp` keeps working without torch/chemprop/smurff installed -- the same
#: reason `interactive` is kept out of `cyp/__init__`.
MODEL_FACTORIES: dict[str, Callable[[], object]] = {
    "mean": MeanBaseline,
    "knn": NearestNeighbourBaseline,
    "ridge": lambda: Ridge(alpha=1.0),
    "rf": lambda: RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42),
    "lgbm": _lightgbm,
    "xgb": _xgboost,
    "chemprop": _chemprop,
    "chemeleon": _chemeleon,
    "tabpfn": _tabpfn,
    "tabicl": _tabicl,
    "macau": _macau,
}

#: Methods that need the `deep` extra installed. Used to fail with a clear
#: instruction rather than a bare ImportError three folds into a CV run.
DEEP_METHODS = frozenset({"chemprop", "chemeleon", "tabpfn", "tabicl", "macau"})


def _needs_smiles(model: object) -> bool:
    """Whether this model consumes SMILES strings rather than a feature matrix.

    Graph models featurize internally from the molecular graph, so `run_cv` hands
    them the SMILES column; everything else gets the precomputed fingerprint rows.
    """
    return bool(getattr(model, "requires_smiles", False))


def _featurize(
    frame: pl.DataFrame,
    fingerprint: str,
    n_bits: int,
    features: np.ndarray | None,
) -> np.ndarray:
    """Feature matrix for `frame`, either precomputed or freshly fingerprinted.

    `features` exists for representations that are not cheap to recompute and not
    produced by `fingerprints.compute` -- chiefly CheMeleon embeddings, which come
    from a subprocess (`graph_models.chemeleon_embed`). Computing those once in the
    notebook and passing them in beats regenerating them per method.
    """
    if features is not None:
        if len(features) != frame.height:
            raise ValueError(
                f"features has {len(features)} rows but frame has {frame.height}; "
                "they must be row-aligned."
            )
        return np.asarray(features)
    return fingerprints.compute(frame["SMILES"].to_list(), fingerprint, fp_size=n_bits)


def run_cv(
    frame: pl.DataFrame,
    endpoint: str,
    methods: tuple[str, ...] = ("mean", "knn", "lgbm"),
    fingerprint: str = "ecfp",
    n_bits: int = 2048,
    n_outer: int = 5,
    n_inner: int = 5,
    seed: int = 42,
    features: np.ndarray | None = None,
    p_val: float = 0.0,
    on_fold=None,
) -> pl.DataFrame:
    """Run nested scaffold CV for `methods` on one endpoint.

    Featurization happens once up front (a representation does not depend on the
    split), then each fold trains on its own rows. Returns the long OOF frame that
    `evaluation`, `calibration` and `ensemble` all consume.

    Args:
        frame: Model-ready frame from `data.training_frame`.
        endpoint: Endpoint name, recorded in the OOF frame.
        methods: Keys of `MODEL_FACTORIES`.
        fingerprint: Representation name, ignored when `features` is given.
        n_bits: Fingerprint width, ignored when `features` is given.
        n_outer: Outer CV repeats.
        n_inner: Folds per repeat.
        seed: Split seed.
        features: Precomputed, row-aligned feature matrix (e.g. CheMeleon
            embeddings) to use instead of fingerprinting `frame`.
        p_val: Fraction of each training fold held out for validation. Graph models
            need this for early stopping -- pass ~0.1 when `methods` includes
            chemprop or chemeleon, so their stopping set respects the scaffold
            grouping instead of being carved out at random inside the model.
        on_fold: Optional `(fold, n_folds) -> None` progress callback, invoked after
            each fold. A 5x5 run on a graph model is ~37 minutes per endpoint, which
            looks identical to a hang without this. See `cv.FoldCallback`.
    """
    unknown = set(methods) - set(MODEL_FACTORIES)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")

    X_all = _featurize(frame, fingerprint, n_bits, features)
    indexed = frame.with_row_index("_row")
    n_folds = n_outer * n_inner

    records: list[dict] = []
    for fold, outer, inner, train, val, test in scaffold_splits(
        indexed, n_outer=n_outer, n_inner=n_inner, seed=seed, p_val=p_val
    ):
        X_train = X_all[train["_row"].to_numpy()]
        X_test = X_all[test["_row"].to_numpy()]
        y_train = train["y_true"].to_numpy()

        for method in methods:
            model = MODEL_FACTORIES[method]()

            if _needs_smiles(model):
                # Graph models featurize from the molecular graph themselves, so
                # they take SMILES and their own validation split for early stopping.
                fit_kwargs = {}
                if val is not None:
                    fit_kwargs = {
                        "smiles_val": val["SMILES"].to_list(),
                        "y_val": val["y_true"].to_numpy(),
                    }
                model.fit(train["SMILES"].to_list(), y_train, **fit_kwargs)
                y_pred = np.asarray(model.predict(test["SMILES"].to_list()), dtype=float)
            else:
                model.fit(X_train, y_train)
                y_pred = np.asarray(model.predict(X_test), dtype=float)

            for i in range(test.height):
                records.append(
                    {
                        "method": method,
                        "endpoint": endpoint,
                        "fold": fold,
                        "outer_fold": outer,
                        "inner_fold": inner,
                        "Molecule_Name": test["Molecule_Name"][i],
                        "y_true": float(test["y_true"][i]),
                        "y_pred": float(y_pred[i]),
                        "y_lower": float(test["y_lower"][i])
                        if test["y_lower"][i] is not None
                        else None,
                        "y_upper": float(test["y_upper"][i])
                        if test["y_upper"][i] is not None
                        else None,
                    }
                )

        # After every method has been fitted on this fold, so a tick means one
        # complete fold rather than a partial one.
        report_fold(on_fold, fold, n_folds)
    return pl.DataFrame(records)


def fit_predict_test(
    frame: pl.DataFrame,
    test_smiles: list[str],
    method: str = "lgbm",
    fingerprint: str = "ecfp",
    n_bits: int = 2048,
    train_features: np.ndarray | None = None,
    test_features: np.ndarray | None = None,
) -> np.ndarray:
    """Train on the full labelled set and predict the blind test compounds.

    Args:
        frame: Full labelled training frame for the endpoint.
        test_smiles: Blind test SMILES.
        method: Key of `MODEL_FACTORIES`.
        fingerprint: Representation, ignored when features are passed.
        n_bits: Fingerprint width, ignored when features are passed.
        train_features: Precomputed train features, row-aligned with `frame`.
        test_features: Precomputed test features, row-aligned with `test_smiles`.
            Must be passed together with `train_features` and produced by the same
            featurizer -- `graph_models.chemeleon_embed` returns both at once
            precisely so they cannot drift apart.
    """
    if (train_features is None) != (test_features is None):
        raise ValueError("Pass train_features and test_features together, or neither.")

    model = MODEL_FACTORIES[method]()
    if _needs_smiles(model):
        model.fit(frame["SMILES"].to_list(), frame["y_true"].to_numpy())
        return np.asarray(model.predict(list(test_smiles)), dtype=float)

    if train_features is not None:
        X_train, X_test = np.asarray(train_features), np.asarray(test_features)
    else:
        X_train = fingerprints.compute(frame["SMILES"].to_list(), fingerprint, fp_size=n_bits)
        X_test = fingerprints.compute(test_smiles, fingerprint, fp_size=n_bits)

    model.fit(X_train, frame["y_true"].to_numpy())
    return np.asarray(model.predict(X_test), dtype=float)


# ── TDI classification ──────────────────────────────────────────────────────────
#
# TDI labels are imbalanced (~21-22% positive, see CLAUDE.md), which is exactly why
# MCC -- not accuracy -- is the challenge metric: a classifier that always predicts
# "not TDI" scores ~78% accuracy while being useless. `MajorityBaseline` makes that
# failure mode visible rather than letting a real model's accuracy look deceptively
# good in isolation; a model must beat it on MCC, not accuracy, to mean anything.


class MajorityBaseline:
    """Predict the majority class for every compound. MCC = 0.0 by construction --
    the line every real classifier must clear, and the trap plain accuracy hides."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> MajorityBaseline:
        self.value_ = bool(np.mean(y) >= 0.5)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.value_)


class NearestNeighbourClassifier:
    """1-NN by Tanimoto similarity -- the classification analogue of
    `NearestNeighbourBaseline`, same rationale."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> NearestNeighbourClassifier:
        self.X_ = np.asarray(X, dtype=bool)
        self.y_ = np.asarray(y, dtype=bool)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        query = np.asarray(X, dtype=bool)
        intersection = query.astype(np.float32) @ self.X_.T.astype(np.float32)
        query_bits = query.sum(axis=1, keepdims=True)
        train_bits = self.X_.sum(axis=1, keepdims=True).T
        union = query_bits + train_bits - intersection
        similarity = np.divide(
            intersection, union, out=np.zeros_like(intersection), where=union > 0
        )
        return self.y_[np.argmax(similarity, axis=1)]


def _lightgbm_classifier(**kwargs):
    from lightgbm import LGBMClassifier

    params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "verbose": -1,
        "n_jobs": -1,
        # TDI positives are the minority class; without this the trees optimize
        # for the majority and MCC collapses toward 0, the same failure the
        # MajorityBaseline demonstrates.
        "class_weight": "balanced",
    }
    return LGBMClassifier(**(params | kwargs))


def _xgboost_classifier(**kwargs):
    from xgboost import XGBClassifier

    params = {"n_estimators": 500, "learning_rate": 0.05, "n_jobs": -1, "tree_method": "hist"}
    return XGBClassifier(**(params | kwargs))


def _chemprop_classifier():
    from .graph_models import ChempropModel

    return ChempropModel(pred_type="classification")


def _chemeleon_classifier():
    from .graph_models import ChempropChemeleonModel

    return ChempropChemeleonModel(pred_type="classification")


def _tabpfn_classifier():
    from .tabular_models import TabPFNClassifierModel

    return TabPFNClassifierModel()


def _tabicl_classifier():
    from .tabular_models import TabICLClassifierModel

    return TabICLClassifierModel()


#: Classification analogues of `MODEL_FACTORIES`. No `macau` entry: Macau factorizes
#: a real-valued matrix, and a binary TDI label is not what it is built to model.
#: Graph models here return probabilities, so `run_cv_classification` thresholds
#: them -- at 0.5 by default, which is not MCC-optimal on a ~21%-positive label and
#: should be tuned (PLAN.md item 5).
CLASSIFIER_FACTORIES: dict[str, Callable[[], object]] = {
    "majority": MajorityBaseline,
    "knn": NearestNeighbourClassifier,
    "rf": lambda: RandomForestClassifier(
        n_estimators=500, n_jobs=-1, random_state=42, class_weight="balanced"
    ),
    "lgbm": _lightgbm_classifier,
    "xgb": _xgboost_classifier,
    "chemprop": _chemprop_classifier,
    "chemeleon": _chemeleon_classifier,
    "tabpfn": _tabpfn_classifier,
    "tabicl": _tabicl_classifier,
}


def _to_labels(raw: np.ndarray, threshold: float) -> np.ndarray:
    """Coerce a classifier's output to boolean labels.

    Graph models return positive-class probabilities while the tree and baseline
    classifiers return labels directly, so this normalizes both. A float array in
    [0, 1] that is not already all-0/1 is treated as probabilities and thresholded.
    """
    raw = np.asarray(raw)
    if raw.dtype == bool:
        return raw
    values = raw.astype(float)
    is_probability = (
        values.min() >= 0.0 and values.max() <= 1.0 and not np.all(np.isin(values, (0.0, 1.0)))
    )
    return (values >= threshold) if is_probability else values.astype(bool)


def run_cv_classification(
    frame: pl.DataFrame,
    endpoint: str,
    methods: tuple[str, ...] = ("majority", "knn", "lgbm"),
    fingerprint: str = "ecfp",
    n_bits: int = 2048,
    n_outer: int = 5,
    n_inner: int = 5,
    seed: int = 42,
    features: np.ndarray | None = None,
    p_val: float = 0.0,
    threshold: float = 0.5,
    on_fold=None,
) -> pl.DataFrame:
    """Run nested scaffold CV for TDI classifiers on one isoform.

    Same protocol and OOF schema shape as `run_cv`, but `y_true`/`y_pred` are
    booleans and there are no credible-interval columns (TDI labels carry no
    interval; MCC does not use one).

    Args:
        frame: Model-ready frame from `data.tdi_training_frame`.
        endpoint: Isoform name, recorded in the OOF frame.
        methods: Keys of `CLASSIFIER_FACTORIES`.
        fingerprint: Representation, ignored when `features` is given.
        n_bits: Fingerprint width, ignored when `features` is given.
        n_outer: Outer CV repeats.
        n_inner: Folds per repeat.
        seed: Split seed.
        features: Precomputed, row-aligned feature matrix.
        p_val: Validation fraction per training fold; graph models need ~0.1.
        threshold: Probability cutoff for models that emit probabilities. 0.5 is the
            default but *not* MCC-optimal on a ~21%-positive label -- tune it on OOF
            predictions rather than shipping this value.
    """
    unknown = set(methods) - set(CLASSIFIER_FACTORIES)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")

    X_all = _featurize(frame, fingerprint, n_bits, features)
    indexed = frame.with_row_index("_row")
    n_folds = n_outer * n_inner

    records: list[dict] = []
    for fold, outer, inner, train, val, test in scaffold_splits(
        indexed, n_outer=n_outer, n_inner=n_inner, seed=seed, p_val=p_val
    ):
        X_train = X_all[train["_row"].to_numpy()]
        X_test = X_all[test["_row"].to_numpy()]
        y_train = train["y_true"].to_numpy()

        for method in methods:
            model = CLASSIFIER_FACTORIES[method]()

            if _needs_smiles(model):
                fit_kwargs = {}
                if val is not None:
                    fit_kwargs = {
                        "smiles_val": val["SMILES"].to_list(),
                        "y_val": val["y_true"].to_numpy(),
                    }
                model.fit(train["SMILES"].to_list(), y_train, **fit_kwargs)
                raw = np.asarray(model.predict(test["SMILES"].to_list()), dtype=float)
            else:
                model.fit(X_train, y_train)
                raw = model.predict(X_test)

            y_pred = _to_labels(raw, threshold)

            for i in range(test.height):
                records.append(
                    {
                        "method": method,
                        "endpoint": endpoint,
                        "fold": fold,
                        "outer_fold": outer,
                        "inner_fold": inner,
                        "Molecule_Name": test["Molecule_Name"][i],
                        "y_true": bool(test["y_true"][i]),
                        "y_pred": bool(y_pred[i]),
                    }
                )

        report_fold(on_fold, fold, n_folds)
    return pl.DataFrame(records)


def fit_predict_test_classification(
    frame: pl.DataFrame,
    test_smiles: list[str],
    method: str = "lgbm",
    fingerprint: str = "ecfp",
    n_bits: int = 2048,
    train_features: np.ndarray | None = None,
    test_features: np.ndarray | None = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """Train a TDI classifier on the full labelled set and predict the blind test
    compounds. Returns boolean labels -- the TDI submission wants hard labels, not
    probabilities, since MCC is computed on the labels as submitted.

    `threshold` applies to models that emit probabilities. Tune it against OOF MCC
    before a submission: 0.5 is a convention, not an optimum, on a ~21%-positive label.
    """
    if (train_features is None) != (test_features is None):
        raise ValueError("Pass train_features and test_features together, or neither.")

    model = CLASSIFIER_FACTORIES[method]()
    if _needs_smiles(model):
        model.fit(frame["SMILES"].to_list(), frame["y_true"].to_numpy())
        return _to_labels(np.asarray(model.predict(list(test_smiles)), dtype=float), threshold)

    if train_features is not None:
        X_train, X_test = np.asarray(train_features), np.asarray(test_features)
    else:
        X_train = fingerprints.compute(frame["SMILES"].to_list(), fingerprint, fp_size=n_bits)
        X_test = fingerprints.compute(test_smiles, fingerprint, fp_size=n_bits)

    model.fit(X_train, frame["y_true"].to_numpy())
    return _to_labels(model.predict(X_test), threshold)
