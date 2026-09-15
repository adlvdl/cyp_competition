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
from .cv import fold_assignment_splits, report_fold, shared_scaffold_folds


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


def _stacked_feature_matrix(
    stacked: pl.DataFrame,
    frames: dict[str, pl.DataFrame],
    endpoints: list[str],
    features: dict[str, np.ndarray] | None,
    fingerprint: str,
    n_bits: int,
) -> np.ndarray:
    """Feature matrix for a row-stacked frame, with one-hot endpoint indicators.

    Featurizes per endpoint and joins back onto `stacked` so every row's features
    line up with the endpoint that row came from, then appends the indicators. Three
    stacked runners need exactly this, and the join is the part that is easy to get
    subtly wrong -- a row aligned to the wrong endpoint's matrix produces no error,
    just a quietly worse model.
    """
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
                        # A plain repeated value, not pl.lit(): the DataFrame
                        # constructor takes data, and an Expr is not data.
                        "endpoint": [endpoint] * frame.height,
                        "_feat": matrix.tolist(),
                    }
                )
            )
        joined = stacked.join(
            pl.concat(blocks, how="vertical_relaxed"),
            on=["Molecule_Name", "endpoint"],
            how="left",
        )
        X = np.array(joined["_feat"].to_list(), dtype=np.float32)
    else:
        X = fingerprints.compute(stacked["SMILES"].to_list(), fingerprint, fp_size=n_bits)

    return stack_features(X, stacked["endpoint"].to_list(), endpoints)


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
    on_fold=None,
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
        on_fold: Optional `(fold, n_folds) -> None` progress callback; see
            `cv.FoldCallback`.

    Returns:
        Long OOF frame, directly comparable to `models.run_cv` output.
    """
    from .models import MODEL_FACTORIES, _needs_smiles

    if assignments is None:
        _, assignments = shared_scaffold_folds(frames, n_outer=n_outer, n_inner=n_inner, seed=seed)

    stacked, endpoints = endpoint_indicator_frame(frames)
    # Featurize per endpoint and align back onto the stacked rows, so every row's
    # features come from its own endpoint's representation.
    X_all = _stacked_feature_matrix(stacked, frames, endpoints, features, fingerprint, n_bits)
    indexed = stacked.with_row_index("_row")
    n_folds = assignments["fold"].n_unique()

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
        report_fold(on_fold, fold, n_folds)

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
    on_fold=None,
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
        on_fold: Optional `(fold, n_folds) -> None` progress callback.
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
    n_folds = assignments["fold"].n_unique()

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
        report_fold(on_fold, fold, n_folds)

    return pl.DataFrame(records)


def run_cv_multitarget(
    frames: dict[str, pl.DataFrame],
    from_foundation: str | None = None,
    n_outer: int = 5,
    n_inner: int = 5,
    seed: int = 42,
    p_val: float = 0.1,
    assignments: pl.DataFrame | None = None,
    on_fold=None,
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
        on_fold: Optional `(fold, n_folds) -> None` progress callback. Chemprop is
            the slowest method here (~90s per fold at 50 epochs), so this is the
            harness that most needs one.
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
    n_folds = assignments["fold"].n_unique()

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
        report_fold(on_fold, fold, n_folds)

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
    on_fold=None,
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
        on_fold: Optional `(fold, n_folds) -> None` progress callback, forwarded to
            whichever runner handles this method.
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
        "on_fold": on_fold,
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
    on_fold=None,
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
    X_all = _stacked_feature_matrix(stacked, frames, endpoints, features, fingerprint, n_bits)
    indexed = stacked.with_row_index("_row")
    n_folds = assignments["fold"].n_unique()

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
        report_fold(on_fold, fold, n_folds)

    return pl.DataFrame(records)


def fit_predict_test_multitarget(
    frames: dict[str, pl.DataFrame],
    test_smiles: list[str],
    from_foundation: str | None = None,
    p_val: float = 0.1,
    seed: int = 42,
    **chemprop_kwargs,
) -> dict[str, np.ndarray]:
    """Train one multi-target D-MPNN on every endpoint, predict the blind test set.

    The production counterpart to `run_cv_multitarget`: same wide-frame construction
    and the same masked-loss handling of unmeasured cells, but fitted once on all
    labelled data rather than per fold.

    Deliberately *not* cached to disk. CLAUDE.md's convention is to cache CV (slow,
    repeatable) and never the final fit, because a stale final-fit cache can silently
    disagree with a retrained CV cache -- a real bug caught while building this repo,
    where a submission kept using calibration parameters from before a CV retrain.
    One fit is cheap next to 25 folds.

    Args:
        frames: Endpoint name -> that endpoint's labelled frame.
        test_smiles: Blind test SMILES, in submission row order.
        from_foundation: Backbone to warm-start from, e.g. "CHEMELEON"; None trains
            from scratch.
        p_val: Fraction held out for early stopping. Carved at random from the
            training compounds -- there is no held-out fold here, so scaffold
            grouping cannot be preserved; this only picks a stopping point.
        seed: Seed for the validation carve-out.
        **chemprop_kwargs: Forwarded to `ChempropMultitargetModel`.

    Returns:
        Endpoint name -> predictions aligned with `test_smiles`.
    """
    from .graph_models import ChempropMultitargetModel

    endpoints = list(frames)

    # Wide frame: one row per compound, one column per endpoint, null where that
    # compound was never measured for that isoform. Chemprop masks the nulls.
    wide = None
    for endpoint in endpoints:
        block = frames[endpoint].select("Molecule_Name", "SMILES", pl.col("y_true").alias(endpoint))
        wide = (
            block
            if wide is None
            else wide.join(block, on=["Molecule_Name", "SMILES"], how="full", coalesce=True)
        )
    wide = wide.sort("Molecule_Name")

    rng = np.random.default_rng(seed)
    order = rng.permutation(wide.height)
    n_val = max(1, int(p_val * wide.height))
    val_wide, fit_wide = wide[order[:n_val]], wide[order[n_val:]]

    model = ChempropMultitargetModel(
        targets=endpoints, from_foundation=from_foundation, **chemprop_kwargs
    )
    model.fit(
        fit_wide["SMILES"].to_list(),
        fit_wide.select(endpoints).to_numpy(),
        smiles_val=val_wide["SMILES"].to_list(),
        y_val=val_wide.select(endpoints).to_numpy(),
    )

    predictions = model.predict(list(test_smiles))
    return {endpoint: predictions[:, i].astype(float) for i, endpoint in enumerate(endpoints)}


# ── TDI classification ──────────────────────────────────────────────────────────
#
# The regression runners above cannot be reused for TDI: `_oof_records` casts
# `y_true` to float and carries `y_lower`/`y_upper`, which a boolean label does not
# have, and the metric is MCC rather than ST-RAE. The split mirrors
# `models.run_cv_classification` versus `models.run_cv` -- separate harnesses rather
# than a branch, for the same reason CLAUDE.md gives there.
#
# ## Why the premise is weaker here, and why that is the point
#
# The regression multitask win rests on 1,309 of 4,905 compounds (26.7%) carrying
# more than one endpoint, giving a shared encoder real cross-endpoint signal. For
# TDI only **259 of 4,822** compounds (5.4%) appear in both isoforms. So the
# mechanism that won the regression track is mostly absent, and the honest
# expectation is that multitask does less here -- possibly nothing. That is a
# measurement worth taking rather than an assumption worth making in either
# direction: the whole reason this repo runs paired arms is that PXR found multitask
# *hurt* where it was expected to help.
#
# There is no `native` strategy for TDI. Macau factorizes a real-valued matrix, and a
# boolean label is not what it models -- `models.CLASSIFIER_FACTORIES` omits it for
# the same reason.


def _oof_records_classification(
    test: pl.DataFrame,
    y_prob: np.ndarray,
    method: str,
    fold: int,
    outer: int,
    inner: int,
    threshold: float,
    endpoint_col: str = "endpoint",
) -> list[dict]:
    """Build classification OOF rows, matching `models.run_cv_classification`.

    Both the hard label and the probability behind it are recorded. The label is what
    `evaluation.fold_metrics_classification` scores; the probability is what a
    threshold sweep needs, and re-running 25 folds of Chemprop just to recover it
    would be the expensive kind of mistake. Models that emit labels rather than
    probabilities (the majority baseline, 1-NN) round-trip through this unchanged.
    """
    records = []
    for i in range(test.height):
        probability = float(y_prob[i])
        records.append(
            {
                "method": method,
                "endpoint": test[endpoint_col][i],
                "fold": fold,
                "outer_fold": outer,
                "inner_fold": inner,
                "Molecule_Name": test["Molecule_Name"][i],
                "y_true": bool(test["y_true"][i]),
                "y_pred": bool(probability >= threshold),
                "y_prob": probability,
            }
        )
    return records


def _probabilities(model: object, X: np.ndarray) -> np.ndarray:
    """Positive-class probabilities from any classifier in `CLASSIFIER_FACTORIES`.

    `predict_proba` where it exists, falling back to `predict` for the baselines that
    only emit labels. Returning 0.0/1.0 for those is not a fudge: a model with no
    notion of confidence genuinely has none, and thresholding it at anything in
    (0, 1] reproduces its own labels exactly.
    """
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X), dtype=float)
        # sklearn returns (n, n_classes); our wrappers already return (n,).
        if proba.ndim == 2:
            return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        return proba
    return np.asarray(model.predict(X), dtype=float)


def run_cv_stacked_classification(
    frames: dict[str, pl.DataFrame],
    method: str = "lgbm",
    fingerprint: str = "ecfp",
    n_bits: int = 2048,
    n_outer: int = 5,
    n_inner: int = 5,
    seed: int = 42,
    features: dict[str, np.ndarray] | None = None,
    assignments: pl.DataFrame | None = None,
    threshold: float = 0.5,
    folds: list[int] | None = None,
    on_fold=None,
) -> pl.DataFrame:
    """Row-stacked multitask TDI classification, for single-output classifiers.

    The classification twin of `run_cv_stacked`: one model trained on both isoforms'
    rows at once with one-hot isoform indicators appended, so the model can learn an
    isoform-specific offset instead of averaging the two.

    TFMs (`tabicl`, `tabpfn`) are routed to a subprocess rather than constructed
    in-process, for the OpenMP reason in `tabular_models` -- importing torch into a
    kernel that also fits LightGBM makes the LightGBM fit segfault.

    Args:
        frames: Isoform name -> that isoform's frame from `data.tdi_training_frame`.
        method: Key of `models.CLASSIFIER_FACTORIES`. Graph models are not supported
            here -- use `run_cv_multitarget_classification` for Chemprop.
        fingerprint: Representation, ignored when `features` is given.
        n_bits: Fingerprint width, ignored when `features` is given.
        n_outer: Outer repeats.
        n_inner: Folds per repeat.
        seed: Split seed.
        features: Precomputed per-isoform feature matrices, row-aligned with each
            frame (e.g. CheMeleon embeddings).
        assignments: Fold table from `shared_scaffold_folds`. Pass the *same* table
            used for the single-task arm so the comparison is paired.
        threshold: Probability cutoff. 0.5 is the convention, not the MCC optimum on
            a ~21%-positive label; `y_prob` is recorded so it can be tuned after.
        folds: Restrict to these fold indices. This is what lets a notebook run
            fold-major across methods (see `notebooks/04_methods_tdi.py`) instead of
            finishing every fold of one method before starting the next.
        on_fold: Optional `(fold, n_folds) -> None` progress callback.

    Returns:
        Long OOF frame with method `f"{method}_multitask"`.
    """
    from .models import CLASSIFIER_FACTORIES, _needs_smiles

    if assignments is None:
        _, assignments = shared_scaffold_folds(frames, n_outer=n_outer, n_inner=n_inner, seed=seed)

    stacked, endpoints = endpoint_indicator_frame(frames)
    X_all = _stacked_feature_matrix(stacked, frames, endpoints, features, fingerprint, n_bits)
    indexed = stacked.with_row_index("_row")
    n_folds = assignments["fold"].n_unique()

    probe = CLASSIFIER_FACTORIES[method]()
    if _needs_smiles(probe):
        raise ValueError(
            f"{method!r} consumes SMILES, not a feature matrix; use "
            "run_cv_multitarget_classification for graph models."
        )

    wanted = None if folds is None else set(folds)
    records: list[dict] = []
    for fold, outer, inner, train, _val, test in fold_assignment_splits(indexed, assignments):
        if wanted is not None and fold not in wanted:
            continue
        X_train = X_all[train["_row"].to_numpy()]
        X_test = X_all[test["_row"].to_numpy()]
        y_train = train["y_true"].to_numpy()

        if method in ("tabicl", "tabpfn"):
            from .tabular_models import predict_subprocess

            y_prob, _, _ = predict_subprocess(
                X_train, y_train, X_test, kind=method, task="classification", seed=seed
            )
        else:
            model = CLASSIFIER_FACTORIES[method]()
            model.fit(X_train, y_train)
            y_prob = _probabilities(model, X_test)

        records.extend(
            _oof_records_classification(
                test, y_prob, f"{method}_multitask", fold, outer, inner, threshold
            )
        )
        report_fold(on_fold, fold, n_folds)

    return pl.DataFrame(records)


def run_cv_multitarget_classification(
    frames: dict[str, pl.DataFrame],
    from_foundation: str | None = None,
    n_outer: int = 5,
    n_inner: int = 5,
    seed: int = 42,
    p_val: float = 0.1,
    assignments: pl.DataFrame | None = None,
    threshold: float = 0.5,
    folds: list[int] | None = None,
    on_fold=None,
    **chemprop_kwargs,
) -> pl.DataFrame:
    """Multitask TDI by multi-target classification -- one D-MPNN, one head per isoform.

    The classification twin of `run_cv_multitarget`. Chemprop masks missing targets
    in its loss, so the wide frame carries an empty cell wherever a compound was
    never assayed for an isoform -- which is the overwhelming majority here, since
    only 259 of 4,822 compounds have both.

    That sparsity is exactly why this arm is worth running rather than assuming: the
    shared encoder still sees every compound in the union (4,822 against CYP2D6's
    1,497), so CYP2D6 could borrow representation from CYP3A4's much larger set even
    though almost no compound carries both labels. Whether it does is the question.

    Args:
        frames: Isoform name -> that isoform's frame.
        from_foundation: Backbone to warm-start from, e.g. "CHEMELEON"; None trains
            from scratch.
        n_outer: Outer repeats.
        n_inner: Folds per repeat.
        seed: Split seed.
        p_val: Validation fraction per training fold, for early stopping.
        assignments: Fold table from `shared_scaffold_folds`.
        threshold: Probability cutoff; see `run_cv_stacked_classification`.
        folds: Restrict to these fold indices, for fold-major execution.
        on_fold: Optional `(fold, n_folds) -> None` progress callback.
        **chemprop_kwargs: Forwarded to `ChempropMultitargetModel`.

    Returns:
        Long OOF frame with method "chemprop_multitask" (or "chemeleon_multitask").
    """
    from .graph_models import ChempropMultitargetModel

    if assignments is None:
        _, assignments = shared_scaffold_folds(frames, n_outer=n_outer, n_inner=n_inner, seed=seed)

    endpoints = list(frames)
    name = "chemeleon_multitask" if from_foundation else "chemprop_multitask"

    # Wide frame: one row per compound, one column per isoform, null where that
    # compound was never assayed for it. Booleans go in as 0.0/1.0 because the CSV
    # writer emits a numeric column and chemprop's classification task type reads
    # it as the class label; None stays None and is masked out of the loss.
    wide = None
    for endpoint in endpoints:
        block = frames[endpoint].select(
            "Molecule_Name", "SMILES", pl.col("y_true").cast(pl.Float64).alias(endpoint)
        )
        wide = (
            block
            if wide is None
            else wide.join(block, on=["Molecule_Name", "SMILES"], how="full", coalesce=True)
        )
    wide = wide.sort("Molecule_Name")

    stacked, _ = endpoint_indicator_frame(frames)
    n_folds = assignments["fold"].n_unique()

    wanted = None if folds is None else set(folds)
    records: list[dict] = []
    for fold, outer, inner, train_long, _val, test_long in fold_assignment_splits(
        stacked, assignments
    ):
        if wanted is not None and fold not in wanted:
            continue
        train_names = set(train_long["Molecule_Name"].to_list())
        train_wide = wide.filter(pl.col("Molecule_Name").is_in(train_names))

        rng = np.random.default_rng(seed + fold)
        order = rng.permutation(train_wide.height)
        n_val = max(1, int(p_val * train_wide.height))
        val_wide = train_wide[order[:n_val]]
        fit_wide = train_wide[order[n_val:]]

        model = ChempropMultitargetModel(
            targets=endpoints,
            from_foundation=from_foundation,
            pred_type="classification",
            **chemprop_kwargs,
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
        # (n_test_compounds, n_isoforms) of positive-class probabilities.
        predictions = model.predict(test_wide["SMILES"].to_list())

        pred_index = {n: i for i, n in enumerate(test_wide["Molecule_Name"].to_list())}
        column_of = {e: i for i, e in enumerate(endpoints)}
        y_prob = np.array(
            [
                predictions[pred_index[n], column_of[e]]
                for n, e in zip(
                    test_long["Molecule_Name"].to_list(),
                    test_long["endpoint"].to_list(),
                    strict=True,
                )
            ]
        )
        records.extend(
            _oof_records_classification(test_long, y_prob, name, fold, outer, inner, threshold)
        )
        report_fold(on_fold, fold, n_folds)

    return pl.DataFrame(records)


#: Multitask strategy per TDI method. No `native` entry: Macau factorizes a
#: real-valued matrix and has no classification formulation, which is also why
#: `models.CLASSIFIER_FACTORIES` omits it.
STRATEGY_CLASSIFICATION: dict[str, str] = {
    "rf": "stacked",
    "lgbm": "stacked",
    "xgb": "stacked",
    "knn": "stacked",
    "tabicl": "stacked",
    "tabpfn": "stacked",
    "chemprop": "multitarget",
    "chemeleon": "multitarget",
}


def run_cv_multitask_classification(
    frames: dict[str, pl.DataFrame],
    method: str,
    features: dict[str, np.ndarray] | None = None,
    fingerprint: str = "ecfp",
    n_bits: int = 2048,
    n_outer: int = 5,
    n_inner: int = 5,
    seed: int = 42,
    assignments: pl.DataFrame | None = None,
    threshold: float = 0.5,
    folds: list[int] | None = None,
    on_fold=None,
    **kwargs,
) -> pl.DataFrame:
    """Run the multitask TDI arm for `method`, dispatching on its strategy.

    The classification twin of `run_cv_multitask`. Always pass the same
    `assignments` used for the single-task arm: identical folds are what make the
    two comparable with a paired test rather than an eyeball.

    Args:
        frames: Isoform name -> that isoform's frame.
        method: A key of `STRATEGY_CLASSIFICATION`.
        features: Precomputed per-isoform feature matrices. Ignored by the
            multitarget strategy, which featurizes from SMILES.
        fingerprint: Representation, ignored when `features` is given.
        n_bits: Fingerprint width.
        n_outer: Outer repeats.
        n_inner: Folds per repeat.
        seed: Split seed.
        assignments: Fold table from `shared_scaffold_folds`.
        threshold: Probability cutoff for the recorded hard labels.
        folds: Restrict to these fold indices, for fold-major execution.
        on_fold: Optional `(fold, n_folds) -> None` progress callback.
        **kwargs: Forwarded to the underlying runner.

    Returns:
        Long OOF frame whose `method` column is `f"{method}_multitask"`.
    """
    if method not in STRATEGY_CLASSIFICATION:
        raise ValueError(
            f"No multitask classification strategy for {method!r}. "
            f"Known: {sorted(STRATEGY_CLASSIFICATION)}"
        )

    shared = {
        "n_outer": n_outer,
        "n_inner": n_inner,
        "seed": seed,
        "assignments": assignments,
        "threshold": threshold,
        "folds": folds,
        "on_fold": on_fold,
    }

    if STRATEGY_CLASSIFICATION[method] == "stacked":
        return run_cv_stacked_classification(
            frames,
            method=method,
            features=features,
            fingerprint=fingerprint,
            n_bits=n_bits,
            **shared,
            **kwargs,
        )

    return run_cv_multitarget_classification(
        frames,
        from_foundation="CHEMELEON" if method == "chemeleon" else None,
        **shared,
        **kwargs,
    )


def run_cv_singletask_classification(
    frames: dict[str, pl.DataFrame],
    method: str,
    features: dict[str, np.ndarray] | None = None,
    fingerprint: str = "ecfp",
    n_bits: int = 2048,
    seed: int = 42,
    assignments: pl.DataFrame | None = None,
    threshold: float = 0.5,
    p_val: float = 0.1,
    folds: list[int] | None = None,
    on_fold=None,
    **model_kwargs,
) -> pl.DataFrame:
    """Single-task TDI CV driven by the *shared* fold assignment -- the control arm.

    `models.run_cv_classification` draws its own per-isoform scaffold splits, which
    is right on its own but wrong as a multitask control: the two arms must be
    scored on identical test rows for `evaluation.paired_bootstrap` between them to
    be a paired test rather than an eyeball comparison.

    This lives in `src/cyp` rather than in a notebook deliberately. Notebook 03 kept
    the equivalent helper inline, and that is precisely how its TFM fits ended up
    calling `MODEL_FACTORIES[...]()` in-process right after Macau had run -- an
    OpenMP deadlock that cost 14.5 silent hours (CLAUDE.md, "Environment traps").
    Routing TFMs through `predict_subprocess` is a property of the harness, not
    something each notebook should be trusted to remember.

    Args:
        frames: Isoform name -> that isoform's frame.
        method: Key of `models.CLASSIFIER_FACTORIES`.
        features: Precomputed per-isoform feature matrices, row-aligned per frame.
        fingerprint: Representation, ignored when `features` is given.
        n_bits: Fingerprint width.
        seed: Seed for the validation carve-out.
        assignments: Fold table from `shared_scaffold_folds`; required in practice,
            since the point of this function is the shared folds.
        threshold: Probability cutoff for the recorded hard labels.
        p_val: Validation fraction, used only by the graph models for early stopping.
        folds: Restrict to these fold indices, for fold-major execution.
        on_fold: Optional `(fold, n_folds) -> None` progress callback. Fires once per
            (isoform, fold), since that is one model fit.
        **model_kwargs: Forwarded to the model constructor (e.g. `epochs` for
            Chemprop). Whatever is passed to the multitask arm must be passed here
            too: an arm trained for more epochs than its twin is a confounded
            comparison, not a multitask effect.

    Returns:
        Long OOF frame whose `method` column is `f"{method}_singletask"`.
    """
    from . import fingerprints as fp_module
    from .models import CLASSIFIER_FACTORIES, _needs_smiles

    if assignments is None:
        _, assignments = shared_scaffold_folds(frames, seed=seed)

    needs_val = method in ("chemprop", "chemeleon")
    wanted = None if folds is None else set(folds)
    n_folds = assignments["fold"].n_unique()

    records: list[dict] = []
    for endpoint, frame in frames.items():
        indexed = frame.with_row_index("_row")
        if features is not None:
            X_all = np.asarray(features[endpoint])
            if len(X_all) != frame.height:
                raise ValueError(
                    f"features[{endpoint!r}] has {len(X_all)} rows but the frame has "
                    f"{frame.height}; they must be row-aligned."
                )
        else:
            X_all = fp_module.compute(frame["SMILES"].to_list(), fingerprint, fp_size=n_bits)

        for fold, outer, inner, train, val, test in fold_assignment_splits(
            indexed, assignments, p_val=p_val if needs_val else 0.0, seed=seed
        ):
            if wanted is not None and fold not in wanted:
                continue
            y_train = train["y_true"].to_numpy()

            if method in ("tabicl", "tabpfn"):
                # Never in-process: see this function's docstring and CLAUDE.md.
                from .tabular_models import predict_subprocess

                if model_kwargs:
                    # The subprocess runner takes its settings as explicit arguments,
                    # so silently dropping these would leave the two arms configured
                    # differently with nothing to show for it.
                    raise TypeError(
                        f"model_kwargs {sorted(model_kwargs)} are not forwarded to the "
                        f"{method!r} subprocess; pass them to predict_subprocess instead."
                    )
                y_prob, _, _ = predict_subprocess(
                    X_all[train["_row"].to_numpy()],
                    y_train,
                    X_all[test["_row"].to_numpy()],
                    kind=method,
                    task="classification",
                    seed=seed,
                )
            else:
                model = CLASSIFIER_FACTORIES[method]()
                # Factories take no arguments, so per-run settings are applied to the
                # constructed instance. Anything unknown is a typo, not a no-op.
                for key, value in model_kwargs.items():
                    if not hasattr(model, key):
                        raise TypeError(f"{method!r} has no attribute {key!r}")
                    setattr(model, key, value)
                if _needs_smiles(model):
                    fit_kwargs = (
                        {
                            "smiles_val": val["SMILES"].to_list(),
                            "y_val": val["y_true"].to_numpy(),
                        }
                        if val is not None
                        else {}
                    )
                    model.fit(train["SMILES"].to_list(), y_train, **fit_kwargs)
                    # Graph models emit positive-class probabilities directly.
                    y_prob = np.asarray(model.predict(test["SMILES"].to_list()), dtype=float)
                else:
                    model.fit(X_all[train["_row"].to_numpy()], y_train)
                    y_prob = _probabilities(model, X_all[test["_row"].to_numpy()])

            records.extend(
                _oof_records_classification(
                    test.with_columns(pl.lit(endpoint).alias("endpoint")),
                    y_prob,
                    f"{method}_singletask",
                    fold,
                    outer,
                    inner,
                    threshold,
                )
            )
            report_fold(on_fold, fold, n_folds)

    return pl.DataFrame(records)
