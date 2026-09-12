"""Multitask training: one model over all four endpoints at once.

Every other harness in this repo trains one model per endpoint, which is the right
default given that the endpoints do not share rows (CLAUDE.md: "Four targets, four
different training sets"). This module is the controlled alternative to that, so the
question "does sharing across endpoints help?" gets an answer rather than an
assumption.

## Why it might help here

1,309 of the 4,905 labelled compounds carry more than one endpoint. Three endpoints
sit at or above the mean-predictor line on fingerprints, while CYP3A4 has 2,335 rows
and genuinely learns -- so there is a plausible story where the weak endpoints borrow
strength from the strong one. `PLAN.md` ranks representation and data as the levers
that matter, and this is a data lever that costs no new data.

## Why it might not

PXR's retrospective records multitask and auxiliary-assay training *hurting* there,
despite top teams extracting value from the same sources. That is why every function
here is designed to be compared against its single-task twin on identical folds, not
to replace it.

## The three strategies, and why they differ

The model families cannot share one multitask implementation, because "multitask"
means something different to each:

- **`stacked`** (trees, TFMs): one model trained on the vertical concatenation of
  every endpoint's rows, with the endpoint identity appended as one-hot features.
  The model sees 6,525 rows instead of 1,285-2,335 and can learn endpoint-specific
  offsets from the indicator columns. This is the only option for a model with a
  single scalar output.
- **`native`** (Macau): a genuine sparse `(n_compounds, n_endpoints)` factorization.
  Missing entries are absent rather than imputed, which is what matrix factorization
  is built for, and latent compound factors are shared across endpoints by
  construction. The closest thing here to real multitask learning.
- **`multitarget`** (Chemprop): one D-MPNN with a four-output head, trained on a wide
  frame where unmeasured cells are empty. Chemprop masks missing targets in its loss
  natively, so the shared encoder sees every compound while each output head only
  learns from its own labels.

All three return the same long OOF schema as `models.run_cv`, so `evaluation`,
`calibration` and `ensemble` consume them unchanged and the comparison against
single-task is a paired one.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from . import fingerprints
from .cv import fold_assignment_splits, shared_scaffold_folds


def endpoint_indicator_frame(
    frames: dict[str, pl.DataFrame],
    key: str = "Molecule_Name",
) -> tuple[pl.DataFrame, list[str]]:
    """Stack every endpoint's rows into one long frame tagged with its endpoint.

    The result has one row per (compound, endpoint) measurement -- 6,525 rows against
    the 1,285-2,335 any single endpoint provides. `endpoint` is carried as a column
    so a featurizer can one-hot it, and `y_true`/`y_lower`/`y_upper` keep their
    single-task meaning so the scoring path is unchanged.

    Returns:
        `(stacked, endpoints)` -- the long frame and the endpoint order, which fixes
        the one-hot column order for `stack_features`.
    """
    endpoints = list(frames)
    parts = [
        frame.with_columns(pl.lit(endpoint).alias("endpoint")) for endpoint, frame in frames.items()
    ]
    stacked = pl.concat(parts, how="vertical_relaxed").sort([key, "endpoint"])
    return stacked, endpoints


def stack_features(X: np.ndarray, endpoint_labels: list[str], endpoints: list[str]) -> np.ndarray:
    """Append one-hot endpoint indicators to a feature matrix.

    Without these the stacked model sees the same compound twice with two different
    targets and no way to tell which is which -- it would learn the average across
    endpoints, which is worse than any single-task model rather than better. The
    indicators let it learn a per-endpoint offset at minimum, and interactions
    between chemistry and isoform at best.
    """
    indicators = np.zeros((len(endpoint_labels), len(endpoints)), dtype=np.float32)
    index = {e: i for i, e in enumerate(endpoints)}
    for row, label in enumerate(endpoint_labels):
        indicators[row, index[label]] = 1.0
    return np.hstack([np.asarray(X, dtype=np.float32), indicators])


def _oof_records(
    test: pl.DataFrame,
    y_pred: np.ndarray,
    method: str,
    fold: int,
    outer: int,
    inner: int,
    endpoint_col: str = "endpoint",
) -> list[dict]:
    """Build OOF rows in `cv.oof_frame`'s schema, one per test compound."""
    records = []
    for i in range(test.height):
        records.append(
            {
                "method": method,
                "endpoint": test[endpoint_col][i],
                "fold": fold,
                "outer_fold": outer,
                "inner_fold": inner,
                "Molecule_Name": test["Molecule_Name"][i],
                "y_true": float(test["y_true"][i]),
                "y_pred": float(y_pred[i]),
                "y_lower": float(test["y_lower"][i]) if test["y_lower"][i] is not None else None,
                "y_upper": float(test["y_upper"][i]) if test["y_upper"][i] is not None else None,
            }
        )
    return records


def run_cv_stacked(
    frames: dict[str, pl.DataFrame],
    method: str = "lgbm",
    fingerprint: str = "ecfp",
    n_bits: int = 2048,
    n_outer: int = 5,
    n_inner: int = 5,
    seed: int = 42,
    features: dict[str, np.ndarray] | None = None,
    assignments: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Multitask by row-stacking, for any model with a single scalar output.

    Trains one model on every endpoint's rows at once, with one-hot endpoint
    indicators appended to the features. Folds come from `shared_scaffold_folds` so a
    compound is on the same side of the split for every endpoint -- without that, a
    compound's CYP3A4 label could be in training while its CYP2D6 label is in test,
    and the score would be inflated by leakage.

    Args:
        frames: Endpoint name -> that endpoint's frame from `data.training_frame`.
        method: Key of `models.MODEL_FACTORIES`. Graph models are not supported here
            -- use `run_cv_multitarget` for Chemprop.
        fingerprint: Representation, ignored when `features` is given.
        n_bits: Fingerprint width, ignored when `features` is given.
        n_outer: Outer repeats.
        n_inner: Folds per repeat.
        seed: Split seed.
        features: Precomputed per-endpoint feature matrices, each row-aligned with
            that endpoint's frame (e.g. CheMeleon embeddings).
        assignments: Fold table from `shared_scaffold_folds`. Pass the *same* table
            used for the single-task arm so the comparison is paired.

    Returns:
        Long OOF frame, directly comparable to `models.run_cv` output.
    """
    from .models import MODEL_FACTORIES, _needs_smiles

    if assignments is None:
        _, assignments = shared_scaffold_folds(frames, n_outer=n_outer, n_inner=n_inner, seed=seed)

    stacked, endpoints = endpoint_indicator_frame(frames)

    # Featurize per endpoint, then stack in the same order as the frame, so every
    # row's features line up with its own endpoint's representation.
    if features is not None:
        blocks = []
        for endpoint in endpoints:
            frame, matrix = frames[endpoint], np.asarray(features[endpoint])
            if len(matrix) != frame.height:
                raise ValueError(
                    f"features[{endpoint!r}] has {len(matrix)} rows but the frame has "
                    f"{frame.height}; they must be row-aligned."
                )
            blocks.append(
                pl.DataFrame(
                    {
                        "Molecule_Name": frame["Molecule_Name"],
                        "endpoint": pl.lit(endpoint, dtype=pl.Utf8),
                        "_feat": matrix.tolist(),
                    }
                )
            )
        feature_frame = pl.concat(blocks, how="vertical_relaxed")
        joined = stacked.join(feature_frame, on=["Molecule_Name", "endpoint"], how="left")
        X_all = np.array(joined["_feat"].to_list(), dtype=np.float32)
    else:
        X_all = fingerprints.compute(stacked["SMILES"].to_list(), fingerprint, fp_size=n_bits)

    X_all = stack_features(X_all, stacked["endpoint"].to_list(), endpoints)
    indexed = stacked.with_row_index("_row")

    probe = MODEL_FACTORIES[method]()
    if _needs_smiles(probe):
        raise ValueError(
            f"{method!r} consumes SMILES, not a feature matrix; use "
            "run_cv_multitarget for graph models."
        )

    records: list[dict] = []
    for fold, outer, inner, train, _val, test in fold_assignment_splits(indexed, assignments):
        model = MODEL_FACTORIES[method]()
        model.fit(X_all[train["_row"].to_numpy()], train["y_true"].to_numpy())
        y_pred = np.asarray(model.predict(X_all[test["_row"].to_numpy()]), dtype=float)
        records.extend(_oof_records(test, y_pred, f"{method}_multitask", fold, outer, inner))

    return pl.DataFrame(records)


def run_cv_native(
    frames: dict[str, pl.DataFrame],
    features: dict[str, np.ndarray] | None = None,
    fingerprint: str = "ecfp",
    n_bits: int = 2048,
    n_outer: int = 5,
    n_inner: int = 5,
    seed: int = 42,
    assignments: pl.DataFrame | None = None,
    **macau_kwargs,
) -> pl.DataFrame:
    """Multitask by sparse matrix factorization -- Macau's native formulation.

    The closest thing here to genuine multitask learning. Macau factorizes a sparse
    `(n_compounds, n_endpoints)` matrix in which an unmeasured pair is simply absent,
    never imputed, and latent compound factors are shared across all four endpoints
    by construction. Nothing has to be stacked or one-hot encoded: the missingness
    the other strategies work around is the format this model expects.

    Features are per compound (not per compound-endpoint), which is the other reason
    this differs from `run_cv_stacked`: a compound has one fingerprint, and the
    factorization maps it to one latent vector used for every endpoint.

    Args:
        frames: Endpoint name -> that endpoint's frame.
        features: Precomputed per-endpoint matrices, row-aligned with each frame.
            Deduplicated onto compounds internally, taking the first occurrence --
            the representation does not depend on which endpoint the row came from.
        fingerprint: Representation, ignored when `features` is given.
        n_bits: Fingerprint width, ignored when `features` is given.
        n_outer: Outer repeats.
        n_inner: Folds per repeat.
        seed: Split seed.
        assignments: Fold table from `shared_scaffold_folds`; pass the same one used
            for the single-task arm.
        **macau_kwargs: Forwarded to `MacauModel`.

    Returns:
        Long OOF frame with method "macau_multitask".
    """
    import scipy.sparse as sp

    from .matrix_factorization import MacauModel

    if assignments is None:
        _, assignments = shared_scaffold_folds(frames, n_outer=n_outer, n_inner=n_inner, seed=seed)

    endpoints = list(frames)

    # One row per compound, with its SMILES and (optionally) its feature vector.
    compound_rows = []
    for endpoint in endpoints:
        frame = frames[endpoint]
        block = {
            "Molecule_Name": frame["Molecule_Name"].to_list(),
            "SMILES": frame["SMILES"].to_list(),
        }
        if features is not None:
            matrix = np.asarray(features[endpoint])
            if len(matrix) != frame.height:
                raise ValueError(
                    f"features[{endpoint!r}] has {len(matrix)} rows but the frame has "
                    f"{frame.height}; they must be row-aligned."
                )
            block["_feat"] = matrix.tolist()
        compound_rows.append(pl.DataFrame(block))

    compounds = (
        pl.concat(compound_rows, how="vertical_relaxed")
        .unique(subset=["Molecule_Name"], keep="first")
        .sort("Molecule_Name")
    )
    row_of = {name: i for i, name in enumerate(compounds["Molecule_Name"].to_list())}

    if features is not None:
        X_all = np.array(compounds["_feat"].to_list(), dtype=np.float32)
    else:
        X_all = fingerprints.compute(compounds["SMILES"].to_list(), fingerprint, fp_size=n_bits)

    # Long table of every measurement, so a fold's train/test split is a filter.
    stacked, _ = endpoint_indicator_frame(frames)
    stacked = stacked.with_columns(
        pl.col("Molecule_Name").replace_strict(row_of, return_dtype=pl.Int64).alias("_row")
    )
    column_of = {e: i for i, e in enumerate(endpoints)}

    records: list[dict] = []
    for fold, outer, inner, train, _val, test in fold_assignment_splits(stacked, assignments):
        # Build the sparse target matrix from the training measurements only. Test
        # compounds keep their rows (their features are side information) but hold
        # no observed values, which is exactly how Macau predicts for them.
        rows = train["_row"].to_numpy()
        cols = np.array([column_of[e] for e in train["endpoint"].to_list()])
        values = train["y_true"].to_numpy().astype(float)
        Y = sp.coo_matrix(
            (values, (rows, cols)),
            shape=(len(row_of), len(endpoints)),
            dtype=np.float64,
        )

        model = MacauModel(seed=seed, **macau_kwargs)
        model.fit(X_all, Y)

        # Predict each test measurement from its own endpoint's column.
        test_rows = test["_row"].to_numpy()
        X_test = X_all[test_rows]
        per_column = {column_of[e]: model.predict(X_test, target=column_of[e]) for e in endpoints}
        y_pred = np.array(
            [per_column[column_of[e]][i] for i, e in enumerate(test["endpoint"].to_list())]
        )
        records.extend(_oof_records(test, y_pred, "macau_multitask", fold, outer, inner))

    return pl.DataFrame(records)


def run_cv_multitarget(
    frames: dict[str, pl.DataFrame],
    from_foundation: str | None = None,
    n_outer: int = 5,
    n_inner: int = 5,
    seed: int = 42,
    p_val: float = 0.1,
    assignments: pl.DataFrame | None = None,
    **chemprop_kwargs,
) -> pl.DataFrame:
    """Multitask by multi-target regression -- one Chemprop D-MPNN, four outputs.

    Chemprop masks missing targets in its loss, so a wide frame with empty cells is
    trained correctly without imputation: the shared message-passing encoder sees
    every compound in the union, while each output head learns only from the
    compounds that actually have that endpoint. That shared encoder is the mechanism
    by which a weak endpoint could benefit from CYP3A4's 2,335 rows.

    Args:
        frames: Endpoint name -> that endpoint's frame.
        from_foundation: Backbone to warm-start from, e.g. "CHEMELEON"; None trains
            from scratch.
        n_outer: Outer repeats.
        n_inner: Folds per repeat.
        seed: Split seed.
        p_val: Validation fraction per training fold, for early stopping.
        assignments: Fold table from `shared_scaffold_folds`.
        **chemprop_kwargs: Forwarded to `ChempropMultitargetModel`.

    Returns:
        Long OOF frame with method "chemprop_multitask" (or "chemeleon_multitask").
    """
    from .graph_models import ChempropMultitargetModel

    if assignments is None:
        _, assignments = shared_scaffold_folds(frames, n_outer=n_outer, n_inner=n_inner, seed=seed)

    endpoints = list(frames)
    name = "chemeleon_multitask" if from_foundation else "chemprop_multitask"

    # Wide frame: one row per compound, one column per endpoint, nulls where a
    # compound was never measured for that isoform. This is the shape Chemprop's
    # multi-target mode expects, and the nulls are what its loss masks.
    wide = None
    for endpoint in endpoints:
        block = frames[endpoint].select("Molecule_Name", "SMILES", pl.col("y_true").alias(endpoint))
        wide = (
            block
            if wide is None
            else wide.join(block, on=["Molecule_Name", "SMILES"], how="full", coalesce=True)
        )
    wide = wide.sort("Molecule_Name")

    # The long table drives scoring: a prediction is only scored where a label
    # exists, so the wide frame's nulls never reach the metric.
    stacked, _ = endpoint_indicator_frame(frames)

    records: list[dict] = []
    for fold, outer, inner, train_long, _val, test_long in fold_assignment_splits(
        stacked, assignments
    ):
        train_names = set(train_long["Molecule_Name"].to_list())
        train_wide = wide.filter(pl.col("Molecule_Name").is_in(train_names))

        # Carve the early-stopping set out of the training compounds, keeping it
        # scaffold-clean by construction since the fold split already was.
        rng = np.random.default_rng(seed + fold)
        order = rng.permutation(train_wide.height)
        n_val = max(1, int(p_val * train_wide.height))
        val_wide = train_wide[order[:n_val]]
        fit_wide = train_wide[order[n_val:]]

        model = ChempropMultitargetModel(
            targets=endpoints, from_foundation=from_foundation, **chemprop_kwargs
        )
        model.fit(
            fit_wide["SMILES"].to_list(),
            fit_wide.select(endpoints).to_numpy(),
            smiles_val=val_wide["SMILES"].to_list(),
            y_val=val_wide.select(endpoints).to_numpy(),
        )

        test_names = list(dict.fromkeys(test_long["Molecule_Name"].to_list()))
        test_wide = wide.filter(pl.col("Molecule_Name").is_in(set(test_names))).sort(
            "Molecule_Name"
        )
        predictions = model.predict(test_wide["SMILES"].to_list())

        # predictions is (n_test_compounds, n_endpoints); map each scored
        # measurement back to its compound's row and its endpoint's column.
        pred_index = {n: i for i, n in enumerate(test_wide["Molecule_Name"].to_list())}
        column_of = {e: i for i, e in enumerate(endpoints)}
        y_pred = np.array(
            [
                predictions[pred_index[n], column_of[e]]
                for n, e in zip(
                    test_long["Molecule_Name"].to_list(),
                    test_long["endpoint"].to_list(),
                    strict=True,
                )
            ]
        )
        records.extend(_oof_records(test_long, y_pred, name, fold, outer, inner))

    return pl.DataFrame(records)


#: Which multitask strategy each method uses. The mapping is not arbitrary: it
#: reflects what "multitask" can even mean for that model class -- a single-output
#: regressor can only be row-stacked, Macau factorizes natively, and Chemprop has a
#: real multi-target head. See the module docstring.
STRATEGY: dict[str, str] = {
    "ridge": "stacked",
    "rf": "stacked",
    "lgbm": "stacked",
    "xgb": "stacked",
    "tabicl": "stacked",
    "tabpfn": "stacked",
    "macau": "native",
    "chemprop": "multitarget",
    "chemeleon": "multitarget",
}


def run_cv_multitask(
    frames: dict[str, pl.DataFrame],
    method: str,
    features: dict[str, np.ndarray] | None = None,
    fingerprint: str = "ecfp",
    n_bits: int = 2048,
    n_outer: int = 5,
    n_inner: int = 5,
    seed: int = 42,
    assignments: pl.DataFrame | None = None,
    **kwargs,
) -> pl.DataFrame:
    """Run the multitask arm for `method`, dispatching on its strategy.

    One entry point so a notebook can loop over methods without knowing which of the
    three formulations each one needs. Always pass the same `assignments` used for
    the single-task arm: identical folds are what make the two comparable with a
    paired test rather than an eyeball.

    Args:
        frames: Endpoint name -> that endpoint's frame.
        method: A key of `STRATEGY`.
        features: Precomputed per-endpoint feature matrices (e.g. CheMeleon).
            Ignored by the multitarget strategy, which featurizes from SMILES.
        fingerprint: Representation, ignored when `features` is given.
        n_bits: Fingerprint width.
        n_outer: Outer repeats.
        n_inner: Folds per repeat.
        seed: Split seed.
        assignments: Fold table from `shared_scaffold_folds`.
        **kwargs: Forwarded to the underlying runner.

    Returns:
        Long OOF frame whose `method` column is `f"{method}_multitask"`.
    """
    if method not in STRATEGY:
        raise ValueError(f"No multitask strategy for {method!r}. Known: {sorted(STRATEGY)}")

    strategy = STRATEGY[method]
    shared = {
        "n_outer": n_outer,
        "n_inner": n_inner,
        "seed": seed,
        "assignments": assignments,
    }

    if strategy == "stacked":
        # TFMs must not be fitted in this process -- torch would poison any later
        # LightGBM fit (see tabular_models on the OpenMP collision). The stacked
        # runner goes through MODEL_FACTORIES, which constructs them in-process, so
        # those are routed to the subprocess-backed path instead.
        if method in ("tabicl", "tabpfn"):
            return _run_cv_stacked_tfm(
                frames,
                kind=method,
                features=features,
                fingerprint=fingerprint,
                n_bits=n_bits,
                **shared,
                **kwargs,
            )
        return run_cv_stacked(
            frames,
            method=method,
            features=features,
            fingerprint=fingerprint,
            n_bits=n_bits,
            **shared,
            **kwargs,
        )

    if strategy == "native":
        return run_cv_native(
            frames,
            features=features,
            fingerprint=fingerprint,
            n_bits=n_bits,
            **shared,
            **kwargs,
        )

    return run_cv_multitarget(
        frames,
        from_foundation="CHEMELEON" if method == "chemeleon" else None,
        **shared,
        **kwargs,
    )


def _run_cv_stacked_tfm(
    frames: dict[str, pl.DataFrame],
    kind: str,
    features: dict[str, np.ndarray] | None = None,
    fingerprint: str = "ecfp",
    n_bits: int = 2048,
    n_outer: int = 5,
    n_inner: int = 5,
    seed: int = 42,
    assignments: pl.DataFrame | None = None,
    **kwargs,
) -> pl.DataFrame:
    """Row-stacked multitask for TabICL/TabPFN, one fold per subprocess.

    Same construction as `run_cv_stacked`, but each fold's fit happens in a child
    process. Importing torch into this kernel would make every later LightGBM fit
    segfault, and a comparison notebook fits both.

    Note the stacked frame is ~6,525 rows against 2,335 for the largest single
    endpoint, which pushes both TFMs further up their memory curves -- the
    per-model budgets in `tabular_models` still apply and are enforced there.
    """
    from .tabular_models import predict_subprocess

    if assignments is None:
        _, assignments = shared_scaffold_folds(frames, n_outer=n_outer, n_inner=n_inner, seed=seed)

    stacked, endpoints = endpoint_indicator_frame(frames)

    if features is not None:
        blocks = []
        for endpoint in endpoints:
            frame, matrix = frames[endpoint], np.asarray(features[endpoint])
            if len(matrix) != frame.height:
                raise ValueError(
                    f"features[{endpoint!r}] has {len(matrix)} rows but the frame has "
                    f"{frame.height}; they must be row-aligned."
                )
            blocks.append(
                pl.DataFrame(
                    {
                        "Molecule_Name": frame["Molecule_Name"],
                        "endpoint": pl.lit(endpoint, dtype=pl.Utf8),
                        "_feat": matrix.tolist(),
                    }
                )
            )
        joined = stacked.join(
            pl.concat(blocks, how="vertical_relaxed"),
            on=["Molecule_Name", "endpoint"],
            how="left",
        )
        X_all = np.array(joined["_feat"].to_list(), dtype=np.float32)
    else:
        X_all = fingerprints.compute(stacked["SMILES"].to_list(), fingerprint, fp_size=n_bits)

    X_all = stack_features(X_all, stacked["endpoint"].to_list(), endpoints)
    indexed = stacked.with_row_index("_row")

    records: list[dict] = []
    for fold, outer, inner, train, _val, test in fold_assignment_splits(indexed, assignments):
        preds, _, _ = predict_subprocess(
            X_all[train["_row"].to_numpy()],
            train["y_true"].to_numpy(),
            X_all[test["_row"].to_numpy()],
            kind=kind,
            seed=seed,
            **kwargs,
        )
        records.extend(_oof_records(test, preds, f"{kind}_multitask", fold, outer, inner))

    return pl.DataFrame(records)
