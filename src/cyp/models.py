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
from .cv import scaffold_splits


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


#: Model factories. PXR found HPO paid off substantially for tree models
#: (XGBoost 0.64 -> 0.52 MAE) and barely at all for neural nets, so tune here first.
MODEL_FACTORIES: dict[str, Callable[[], object]] = {
    "mean": MeanBaseline,
    "knn": NearestNeighbourBaseline,
    "ridge": lambda: Ridge(alpha=1.0),
    "rf": lambda: RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42),
    "lgbm": _lightgbm,
    "xgb": _xgboost,
}


def run_cv(
    frame: pl.DataFrame,
    endpoint: str,
    methods: tuple[str, ...] = ("mean", "knn", "lgbm"),
    fingerprint: str = "ecfp",
    n_bits: int = 2048,
    n_outer: int = 5,
    n_inner: int = 5,
    seed: int = 42,
) -> pl.DataFrame:
    """Run nested scaffold CV for `methods` on one endpoint.

    Featurization happens once up front (fingerprints do not depend on the split),
    then each fold trains on its own rows. Returns the long OOF frame that
    `evaluation`, `calibration` and `ensemble` all consume.
    """
    unknown = set(methods) - set(MODEL_FACTORIES)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")

    X_all = fingerprints.compute(frame["SMILES"].to_list(), fingerprint, fp_size=n_bits)
    indexed = frame.with_row_index("_row")

    records: list[dict] = []
    for fold, outer, inner, train, _val, test in scaffold_splits(
        indexed, n_outer=n_outer, n_inner=n_inner, seed=seed
    ):
        X_train = X_all[train["_row"].to_numpy()]
        X_test = X_all[test["_row"].to_numpy()]
        y_train = train["y_true"].to_numpy()

        for method in methods:
            model = MODEL_FACTORIES[method]()
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
    return pl.DataFrame(records)


def fit_predict_test(
    frame: pl.DataFrame,
    test_smiles: list[str],
    method: str = "lgbm",
    fingerprint: str = "ecfp",
    n_bits: int = 2048,
) -> np.ndarray:
    """Train on the full labelled set and predict the blind test compounds."""
    X_train = fingerprints.compute(frame["SMILES"].to_list(), fingerprint, fp_size=n_bits)
    X_test = fingerprints.compute(test_smiles, fingerprint, fp_size=n_bits)
    model = MODEL_FACTORIES[method]()
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


CLASSIFIER_FACTORIES: dict[str, Callable[[], object]] = {
    "majority": MajorityBaseline,
    "knn": NearestNeighbourClassifier,
    "rf": lambda: RandomForestClassifier(
        n_estimators=500, n_jobs=-1, random_state=42, class_weight="balanced"
    ),
    "lgbm": _lightgbm_classifier,
    "xgb": _xgboost_classifier,
}


def run_cv_classification(
    frame: pl.DataFrame,
    endpoint: str,
    methods: tuple[str, ...] = ("majority", "knn", "lgbm"),
    fingerprint: str = "ecfp",
    n_bits: int = 2048,
    n_outer: int = 5,
    n_inner: int = 5,
    seed: int = 42,
) -> pl.DataFrame:
    """Run nested scaffold CV for TDI classifiers on one isoform.

    Same protocol and OOF schema shape as `run_cv`, but `y_true`/`y_pred` are
    booleans and there are no credible-interval columns (TDI labels carry no
    interval; MCC does not use one).
    """
    unknown = set(methods) - set(CLASSIFIER_FACTORIES)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")

    X_all = fingerprints.compute(frame["SMILES"].to_list(), fingerprint, fp_size=n_bits)
    indexed = frame.with_row_index("_row")

    records: list[dict] = []
    for fold, outer, inner, train, _val, test in scaffold_splits(
        indexed, n_outer=n_outer, n_inner=n_inner, seed=seed
    ):
        X_train = X_all[train["_row"].to_numpy()]
        X_test = X_all[test["_row"].to_numpy()]
        y_train = train["y_true"].to_numpy()

        for method in methods:
            model = CLASSIFIER_FACTORIES[method]()
            model.fit(X_train, y_train)
            y_pred = np.asarray(model.predict(X_test), dtype=bool)

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
    return pl.DataFrame(records)


def fit_predict_test_classification(
    frame: pl.DataFrame,
    test_smiles: list[str],
    method: str = "lgbm",
    fingerprint: str = "ecfp",
    n_bits: int = 2048,
) -> np.ndarray:
    """Train a TDI classifier on the full labelled set and predict the blind test
    compounds. Returns boolean labels -- the TDI submission wants hard labels, not
    probabilities, since MCC is computed on the labels as submitted."""
    X_train = fingerprints.compute(frame["SMILES"].to_list(), fingerprint, fp_size=n_bits)
    X_test = fingerprints.compute(test_smiles, fingerprint, fp_size=n_bits)
    model = CLASSIFIER_FACTORIES[method]()
    model.fit(X_train, frame["y_true"].to_numpy())
    return np.asarray(model.predict(X_test), dtype=bool)
