"""Tests for the model families added in notebooks/03_methods.py.

These deliberately avoid running a real fit for the heavyweight models: a Chemprop
epoch or a TabPFN forward pass is far too slow for a test suite, and a model that
trains is not the thing most likely to break. What breaks silently is the *plumbing*
around them -- a quantile array read along the wrong axis, a multitask matrix that
imputes where it should omit, a SMILES model handed a fingerprint matrix. Those are
what is covered here.

The one exception is Macau, which fits fast enough at toy size to test end to end,
and is worth it because smurff's sampler is the component with a known failure mode
(NaN predictions under numerical instability, seen in PXR).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from cyp import matrix_factorization, models, tabular_models

# ── multitask target matrix ────────────────────────────────────────────────────


def _frame(names: list[str], values: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"Molecule_Name": names, "y_true": values})


def test_multitask_matrix_is_sparse_not_imputed() -> None:
    """The whole point of the sparse encoding: an unmeasured (compound, endpoint)
    pair must be *absent*, not a zero. A zero would be a fabricated pIC50 of 0, which
    the sampler would treat as a real, extremely inactive observation."""
    frames = {
        "A": _frame(["m1", "m2"], [5.0, 6.0]),
        "B": _frame(["m2", "m3"], [7.0, 8.0]),
    }
    Y, compounds, endpoints = matrix_factorization.multitask_target_matrix(frames)

    assert compounds == ["m1", "m2", "m3"]
    assert endpoints == ["A", "B"]
    # 3 compounds x 2 endpoints = 6 cells, but only 4 measurements exist.
    assert Y.shape == (3, 2)
    assert Y.nnz == 4

    dense = Y.toarray()
    assert dense[0, 0] == 5.0  # m1 in A
    assert dense[1, 0] == 6.0  # m2 in A
    assert dense[1, 1] == 7.0  # m2 in B
    assert dense[2, 1] == 8.0  # m3 in B
    # m1/B and m3/A were never measured and must not appear as entries.
    coords = set(zip(Y.row.tolist(), Y.col.tolist(), strict=True))
    assert (0, 1) not in coords
    assert (2, 0) not in coords


def test_multitask_matrix_skips_nulls() -> None:
    """Null and NaN targets are dropped rather than written as entries."""
    frames = {"A": _frame(["m1", "m2", "m3"], [5.0, None, float("nan")])}
    Y, compounds, _ = matrix_factorization.multitask_target_matrix(frames)
    assert len(compounds) == 3
    assert Y.nnz == 1


def test_multitask_matrix_aligns_rows_across_endpoints() -> None:
    """A compound measured in two endpoints occupies one row, not two -- this is what
    lets the factorization share latent structure between endpoints at all."""
    frames = {
        "A": _frame(["shared", "only_a"], [1.0, 2.0]),
        "B": _frame(["shared", "only_b"], [3.0, 4.0]),
    }
    Y, compounds, _ = matrix_factorization.multitask_target_matrix(frames)
    row = compounds.index("shared")
    dense = Y.toarray()
    assert dense[row, 0] == 1.0
    assert dense[row, 1] == 3.0


# ── Macau ──────────────────────────────────────────────────────────────────────


def test_macau_fits_and_predicts_without_nan() -> None:
    """End-to-end on toy data. NaN predictions are smurff's known instability mode
    (PXR had to filter them), so asserting their absence is the point here."""
    smurff = pytest.importorskip("smurff")  # noqa: F841

    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(120, 32)).astype(float)
    y = X[:, :4].sum(axis=1) + rng.normal(scale=0.1, size=120)

    model = matrix_factorization.MacauModel(num_latent=8, burnin=10, nsamples=20)
    model.fit(X[:100], y[:100])

    preds = model.predict(X[100:])
    assert preds.shape == (20,)
    assert not np.isnan(preds).any()

    # The posterior spread is the free uncertainty estimate; it must be real.
    std = model.predict_std(X[100:])
    assert std.shape == (20,)
    assert (std >= 0).all()


def test_macau_requires_fit_before_predict() -> None:
    pytest.importorskip("smurff")
    with pytest.raises(RuntimeError, match="fit"):
        matrix_factorization.MacauModel().predict(np.zeros((3, 4)))


# ── tabular foundation models ──────────────────────────────────────────────────


def test_pca_reduction_only_triggers_above_the_cap() -> None:
    """A narrow descriptor set must pass through untouched -- reducing it would
    destroy interpretable columns for no reason."""
    X_train = np.random.default_rng(0).normal(size=(50, 20))
    X_test = np.random.default_rng(1).normal(size=(10, 20))
    out_train, out_test = tabular_models._reduce_features(X_train, X_test, max_features=500)
    assert out_train is X_train
    assert out_test is X_test


def test_pca_reduction_respects_tabpfn_cap() -> None:
    """A 2048-bit fingerprint exceeds TabPFN's hard 500-feature limit and must come
    back within it, with train and test in the same subspace."""
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(80, 2048))
    X_test = rng.normal(size=(20, 2048))
    out_train, out_test = tabular_models._reduce_features(X_train, X_test, max_features=500)

    # n_components cannot exceed n_samples, so the cap here is 80, not 500.
    assert out_train.shape[1] <= 500
    assert out_train.shape[1] == out_test.shape[1]
    assert out_train.shape[0] == 80
    assert out_test.shape[0] == 20


def test_tabpfn_token_guard_names_the_alternative(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a token TabPFN cannot download weights. The error must say so before
    any folds are spent, and must point at TabICL as the ungated option.

    `load_env` is stubbed out because the guard falls back to reading the project's
    real `.env`, which on a configured machine does have a token -- this test is
    about the no-token path, not about whether this developer happens to have one.
    """
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    monkeypatch.delenv("TABPFN_MODEL_CACHE_DIR", raising=False)
    monkeypatch.setattr(tabular_models, "load_env", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="TabICL"):
        tabular_models._require_tabpfn_token()


def test_tabpfn_guard_accepts_a_token_from_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A token in `.env` must satisfy the guard without anyone sourcing the file --
    that is the whole point of the fallback."""
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    monkeypatch.delenv("TABPFN_MODEL_CACHE_DIR", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("TABPFN_TOKEN=tabpfn_sk_dummy\n")
    # Bind the real function first: referring to `tabular_models.load_env` inside
    # the replacement would resolve to the patched name and recurse.
    real_load_env = tabular_models.load_env
    monkeypatch.setattr(tabular_models, "load_env", lambda *a, **k: real_load_env(env_file))
    tabular_models._require_tabpfn_token()
    assert os.environ.get("TABPFN_TOKEN") == "tabpfn_sk_dummy"


def test_load_env_does_not_override_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """An explicit export must win over the file, so a one-off override works."""
    monkeypatch.setenv("TABPFN_TOKEN", "from_environment")
    env_file = tmp_path / ".env"
    env_file.write_text("TABPFN_TOKEN=from_file\n")
    tabular_models.load_env(env_file)
    assert os.environ["TABPFN_TOKEN"] == "from_environment"


def test_load_env_returns_false_when_absent(tmp_path) -> None:
    assert tabular_models.load_env(tmp_path / "nope.env") is False


def test_tabpfn_token_guard_passes_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TABPFN_TOKEN", "dummy")
    tabular_models._require_tabpfn_token()


def test_tabular_models_require_fit_before_predict() -> None:
    for cls in (
        tabular_models.TabPFNModel,
        tabular_models.TabICLModel,
        tabular_models.TabPFNClassifierModel,
        tabular_models.TabICLClassifierModel,
    ):
        with pytest.raises(RuntimeError, match="fit"):
            cls().predict(np.zeros((3, 4)))


# ── model registry and dispatch ────────────────────────────────────────────────


def test_new_methods_are_registered() -> None:
    """A method missing from the registry fails only once CV reaches it, which on a
    25-fold run is a long way in."""
    for method in ("chemprop", "chemeleon", "tabpfn", "tabicl", "macau"):
        assert method in models.MODEL_FACTORIES
    for method in ("chemprop", "chemeleon", "tabpfn", "tabicl"):
        assert method in models.CLASSIFIER_FACTORIES
    # Macau factorizes a real-valued matrix; a binary TDI label is not its problem.
    assert "macau" not in models.CLASSIFIER_FACTORIES


def test_deep_methods_set_matches_the_registry() -> None:
    assert models.DEEP_METHODS <= set(models.MODEL_FACTORIES)


def test_smiles_dispatch_identifies_graph_models() -> None:
    """`run_cv` hands SMILES to graph models and a feature matrix to everything else;
    getting this backwards would train Chemprop on stringified bit vectors."""
    pytest.importorskip("chemprop")
    from cyp import graph_models

    assert models._needs_smiles(graph_models.ChempropModel())
    assert models._needs_smiles(graph_models.ChempropChemeleonModel())
    assert not models._needs_smiles(models.MeanBaseline())
    assert not models._needs_smiles(models.MODEL_FACTORIES["lgbm"]())


def test_run_cv_rejects_misaligned_features() -> None:
    """Precomputed features that do not line up with the frame would silently train
    on the wrong labels -- the worst kind of bug, since it still produces numbers."""
    frame = pl.DataFrame(
        {
            "Molecule_Name": ["a", "b", "c"],
            "SMILES": ["CCO", "CCC", "CCN"],
            "y_true": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(ValueError, match="row-aligned"):
        models._featurize(frame, "ecfp", 2048, np.zeros((2, 10)))


def test_featurize_passes_through_precomputed_features() -> None:
    frame = pl.DataFrame(
        {"Molecule_Name": ["a", "b"], "SMILES": ["CCO", "CCC"], "y_true": [1.0, 2.0]}
    )
    features = np.arange(8).reshape(2, 4)
    out = models._featurize(frame, "ecfp", 2048, features)
    assert np.array_equal(out, features)


def test_fit_predict_test_rejects_half_supplied_features() -> None:
    """Train features without test features means the two sides were produced by
    different featurizers -- guaranteed nonsense, so refuse rather than broadcast."""
    frame = pl.DataFrame({"Molecule_Name": ["a"], "SMILES": ["CCO"], "y_true": [1.0]})
    with pytest.raises(ValueError, match="together"):
        models.fit_predict_test(frame, ["CCC"], train_features=np.zeros((1, 4)))


# ── uncertainty plumbing (notebook 06) ──────────────────────────────────────────
#
# No real fit here, same rationale as the module docstring: a Chemprop epoch is too
# slow for the suite, and the CLI argument construction (which task-type an
# uncertainty method needs, whether ensemble-size is threaded through) is what
# actually breaks silently. An end-to-end smoke fit of all four methods was run
# manually while building this -- see notebooks/06_uncertainty.py's own quick mode
# for the equivalent check inside the notebook.


def test_ensemble_uncertainty_requires_ensemble_size() -> None:
    pytest.importorskip("chemprop")
    from cyp import graph_models

    with pytest.raises(ValueError, match="ensemble_size"):
        graph_models.ChempropModel(uncertainty_method="ensemble", ensemble_size=1)


def test_unknown_uncertainty_method_rejected() -> None:
    pytest.importorskip("chemprop")
    from cyp import graph_models

    with pytest.raises(ValueError, match="uncertainty_method"):
        graph_models.ChempropModel(uncertainty_method="bogus")


def test_mve_task_type_overrides_plain_regression() -> None:
    pytest.importorskip("chemprop")
    from cyp import graph_models

    model = graph_models.ChempropModel(uncertainty_method="mve")
    args = model._base_train_args("target")
    idx = args.index("--task-type")
    assert args[idx + 1] == "regression-mve"


def test_evidential_task_type_overrides_plain_regression() -> None:
    pytest.importorskip("chemprop")
    from cyp import graph_models

    model = graph_models.ChempropModel(uncertainty_method="evidential")
    args = model._base_train_args("target")
    idx = args.index("--task-type")
    assert args[idx + 1] == "regression-evidential"


def test_ensemble_and_dropout_keep_plain_regression_task_type() -> None:
    """Ensemble and dropout read out of an ordinary regression head -- disagreement
    across checkpoints, or across dropout-active resamples -- so unlike mve/evidential
    they must not change --task-type."""
    pytest.importorskip("chemprop")
    from cyp import graph_models

    for method in ("ensemble", "dropout"):
        kwargs = {"uncertainty_method": method}
        if method == "ensemble":
            kwargs["ensemble_size"] = 4
        model = graph_models.ChempropModel(**kwargs)
        args = model._base_train_args("target")
        idx = args.index("--task-type")
        assert args[idx + 1] == "regression"


def test_ensemble_size_only_passed_at_train_time_for_ensemble_method() -> None:
    pytest.importorskip("chemprop")
    from cyp import graph_models

    plain = graph_models.ChempropModel()
    assert "--ensemble-size" not in plain._base_train_args("target")

    ensembled = graph_models.ChempropModel(uncertainty_method="ensemble", ensemble_size=4)
    args = ensembled._base_train_args("target")
    idx = args.index("--ensemble-size")
    assert args[idx + 1] == "4"


def test_predict_with_uncertainty_requires_uncertainty_method_set() -> None:
    pytest.importorskip("chemprop")
    from cyp import graph_models

    model = graph_models.ChempropModel()
    with pytest.raises(ValueError, match="uncertainty_method"):
        model.predict(["CCO"], return_uncertainty=True)


def test_multitarget_predict_with_uncertainty_requires_uncertainty_method_set() -> None:
    pytest.importorskip("chemprop")
    from cyp import graph_models

    model = graph_models.ChempropMultitargetModel(targets=["a", "b"])
    with pytest.raises(ValueError, match="uncertainty_method"):
        model.predict(["CCO"], return_uncertainty=True)


def test_multitarget_mve_task_type_overrides_plain_regression() -> None:
    pytest.importorskip("chemprop")
    from cyp import graph_models

    model = graph_models.ChempropMultitargetModel(targets=["a", "b"], uncertainty_method="mve")
    args = model._base_train_args("ignored")
    idx = args.index("--task-type")
    assert args[idx + 1] == "regression-mve"


def test_model_path_args_point_at_the_checkpoint_directory() -> None:
    """`--model-paths` must be the directory, not a specific `model_0/best.pt` file --
    an ensemble writes `model_0`, `model_1`, ... and only the directory form lets
    chemprop discover every checkpoint for the ensemble uncertainty estimator."""
    pytest.importorskip("chemprop")
    from cyp import graph_models

    model = graph_models.ChempropModel(model_dir=graph_models.Path("/tmp/some_model_dir"))
    args = model._model_path_args()
    assert args == ["--model-paths", "/tmp/some_model_dir"]


# ── probability thresholding ───────────────────────────────────────────────────


def test_to_labels_thresholds_probabilities() -> None:
    """Graph classifiers emit probabilities while tree classifiers emit labels;
    `_to_labels` must normalize both without mangling either."""
    probs = np.array([0.1, 0.6, 0.49, 0.51])
    assert models._to_labels(probs, 0.5).tolist() == [False, True, False, True]
    # A non-default threshold is the whole reason this is a parameter: 0.5 is not
    # MCC-optimal on a ~21%-positive label.
    assert models._to_labels(probs, 0.3).tolist() == [False, True, True, True]


def test_to_labels_passes_booleans_through() -> None:
    labels = np.array([True, False, True])
    assert models._to_labels(labels, 0.5).tolist() == [True, False, True]


def test_to_labels_treats_hard_zero_one_as_labels() -> None:
    """An all-0/1 float array is a label array that lost its dtype, not a set of
    probabilities that happen to be saturated -- thresholding it must be a no-op."""
    hard = np.array([0.0, 1.0, 1.0, 0.0])
    assert models._to_labels(hard, 0.9).tolist() == [False, True, True, False]


# ── OpenMP load order ──────────────────────────────────────────────────────────


def test_lightgbm_fits_after_fingerprint_computation() -> None:
    """The regression test for the segfault that `cyp/__init__` guards against.

    lightgbm, scikit-learn and torch each vendor their own OpenMP runtime. If
    LightGBM's is not loaded first, computing a fingerprint (which pulls in
    sklearn's) and then fitting LightGBM **segfaults** -- the process dies, so this
    test does not fail, it crashes the whole pytest run. That is exactly why it is
    worth having: a silent reordering of imports in `__init__` would otherwise only
    show up as an unexplained crash mid-notebook.
    """
    import lightgbm as lgb

    from cyp import fingerprints

    X = fingerprints.compute(["CCO", "CCC", "CCN", "c1ccccc1"], "ecfp", fp_size=256)
    assert X.shape[0] == 4

    rng = np.random.default_rng(0)
    lgb.LGBMRegressor(n_estimators=5, verbose=-1).fit(rng.random((50, 10)), rng.random(50))


def test_cyp_import_loads_lightgbm_first() -> None:
    """`cyp/__init__` must import lightgbm before anything else. Asserting the
    module is present after `import cyp` is the cheap proxy for that ordering."""
    import sys

    import cyp  # noqa: F401

    assert "lightgbm" in sys.modules


def test_device_detection_does_not_import_torch() -> None:
    """`device()` runs on every Chemprop fit. If it imported torch into this
    process, every later LightGBM fit in the same notebook would segfault -- so it
    shells out instead, and that property is what this asserts."""
    pytest.importorskip("torch")
    import subprocess
    import sys

    # A clean interpreter, since torch may already be loaded by an earlier test.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from cyp.graph_models import device; d = device(); "
            "print(d, 'torch' in sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    reported_device, torch_loaded = result.stdout.split()
    assert reported_device in ("cpu", "mps", "cuda")
    assert torch_loaded == "False"


# ── memory budget ──────────────────────────────────────────────────────────────


def test_tabicl_budget_allows_the_defaults() -> None:
    """The shipped defaults must pass their own guard, or nothing runs."""
    tabular_models.check_tabicl_budget(
        tabular_models.TABICL_MAX_FEATURES, tabular_models.TABICL_N_ESTIMATORS
    )


def test_tabicl_budget_blocks_a_machine_freezing_config() -> None:
    """2048 features at 8 estimators is the configuration that locked up a 16 GB
    machine. It must be refused up front: TabICL does not raise MemoryError, it
    swaps until the desktop dies, so there is nothing to catch after the fact."""
    with pytest.raises(ValueError, match="exhaust memory"):
        tabular_models.check_tabicl_budget(2048, 8)


def test_tabicl_budget_can_warn_instead_of_raising() -> None:
    """An operator with measured headroom can opt out, but not silently."""
    with pytest.warns(ResourceWarning, match="exhaust memory"):
        tabular_models.check_tabicl_budget(2048, 8, strict=False)


def test_tabicl_defaults_are_conservative() -> None:
    """Guards the measured numbers against a careless bump. 8 estimators at 128
    features cost ~11 GB for no accuracy gain (corr 0.999 either way), which is why
    the default is 2 -- see the TABICL_MAX_FEATURES comment for the full table."""
    assert tabular_models.TABICL_N_ESTIMATORS <= 4
    assert tabular_models.TABICL_MAX_FEATURES <= 256
    product = tabular_models.TABICL_MAX_FEATURES * tabular_models.TABICL_N_ESTIMATORS
    assert product <= tabular_models._TABICL_BUDGET


def test_tabicl_reduces_wide_input() -> None:
    """A 2048-dim CheMeleon embedding must be projected down before it reaches the
    model -- unreduced is the case that froze the machine."""
    model = tabular_models.TabICLModel()
    assert model.max_features == tabular_models.TABICL_MAX_FEATURES
    # The reduction helper is what fit() uses; check it caps as expected.
    rng = np.random.default_rng(0)
    reduced, _ = tabular_models._reduce_features(
        rng.normal(size=(300, 2048)),
        rng.normal(size=(50, 2048)),
        max_features=tabular_models.TABICL_MAX_FEATURES,
    )
    assert reduced.shape[1] <= tabular_models.TABICL_MAX_FEATURES


def test_tabpfn_budget_allows_the_shipped_default() -> None:
    tabular_models.check_tabpfn_budget(tabular_models.TABPFN_MAX_FEATURES)


def test_tabpfn_budget_rejects_above_the_model_limit() -> None:
    """Above 500 features TabPFN refuses outright; the message must say to reduce
    rather than leaving the caller to decode a library error."""
    with pytest.raises(ValueError, match="at most 500 features"):
        tabular_models.check_tabpfn_budget(2048)


def test_tabpfn_budget_warns_between_budget_and_model_limit() -> None:
    """500 features is legal for the model but measured 10.8 GB on a 16 GB machine,
    with no accuracy gain over 192 -- allowed under protest, not silently."""
    with pytest.warns(ResourceWarning, match="memory-hungry"):
        tabular_models.check_tabpfn_budget(500, strict=False)
    with pytest.raises(ValueError, match="memory-hungry"):
        tabular_models.check_tabpfn_budget(500)


def test_tabpfn_default_stays_under_its_own_budget() -> None:
    """Guards the measured numbers against a careless bump."""
    assert tabular_models.TABPFN_MAX_FEATURES <= tabular_models._TABPFN_FEATURE_BUDGET
    assert tabular_models.TABPFN_MAX_FEATURES <= tabular_models.TABPFN_MODEL_FEATURE_LIMIT
    assert tabular_models.TABPFN_N_ESTIMATORS <= 4


def test_the_two_tfms_have_separate_budgets() -> None:
    """TabPFN and TabICL fail differently -- TabICL's memory scales hard with
    ensemble size, TabPFN's barely does, and TabPFN actually uses the extra features
    where TabICL did not. Sharing one budget would mis-serve both."""
    assert tabular_models.TABICL_MAX_FEATURES != tabular_models.TABPFN_MAX_FEATURES


def test_run_cv_reports_each_fold() -> None:
    """`models.run_cv` must tick per fold, not per call: a 5x5 run is 25 fits and a
    caller that only learns about completion at the end cannot show progress."""
    from cyp import constants as C
    from cyp import data, models

    frame = data.training_frame(C.REGRESSION_ENDPOINTS[1]).head(200)
    seen: list[tuple[int, int]] = []
    models.run_cv(
        frame,
        "test",
        methods=("mean",),
        n_bits=128,
        n_outer=1,
        n_inner=3,
        on_fold=lambda fold, total: seen.append((fold, total)),
    )
    assert [f for f, _ in seen] == [0, 1, 2]
    assert all(total == 3 for _, total in seen)


def test_report_fold_swallows_callback_errors() -> None:
    from cyp import cv

    def explode(fold: int, total: int) -> None:
        raise RuntimeError("bar died")

    cv.report_fold(explode, 0, 5)  # must not raise
    cv.report_fold(None, 0, 5)


def test_openmp_guard_warns_when_another_runtime_is_resident() -> None:
    """Regression guard for a silent 14.5-hour hang.

    smurff and lightgbm each vendor an OpenMP runtime; importing torch alongside one
    can deadlock on a barrier with no exception and no CPU use. The 2026-09-13 run
    lost most of a day to this because the single-task arm fitted TabICL in-process
    straight after Macau. The in-process classes must at least say so.
    """
    import warnings

    import cyp  # noqa: F401 - imports lightgbm, which is one of the conflicting runtimes

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tabular_models._warn_if_openmp_conflict("TabICL")
    assert len(caught) == 1
    assert "predict_subprocess" in str(caught[0].message)


def test_openmp_guard_is_quiet_when_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """No warning when no conflicting runtime is loaded -- the guard must not cry
    wolf in a torch-only script, where the in-process path is correct."""
    import sys
    import warnings

    clean = {k: v for k, v in sys.modules.items() if k not in ("smurff", "lightgbm")}
    monkeypatch.setattr(sys, "modules", clean)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tabular_models._warn_if_openmp_conflict("TabICL")
    assert not caught


def test_checkpoint_args_repeats_path_for_ensemble_size():
    """Chemprop's `--checkpoint` takes `nargs="+"` and, when given at all, silently
    sets `ensemble_size = len(checkpoint paths)` (`chemprop/cli/train.py`,
    `train_model`) -- so warm-starting a 4-member ensemble from one pretrained
    checkpoint needs that path listed 4 times, not once. Passing it once collapsed
    ensemble_size to 1 with only a log warning, which is exactly what broke
    notebook 06's first full run: the ensemble arm trained a single checkpoint and
    the later `predict --uncertainty-method ensemble` call raised, since chemprop
    refuses ensemble uncertainty from fewer than two models."""
    pytest.importorskip("chemprop")
    from cyp import graph_models

    ckpt = graph_models.Path("/tmp/some_checkpoint/model_0/best.pt")

    ensembled = graph_models.ChempropModel(uncertainty_method="ensemble", ensemble_size=4)
    args = ensembled._checkpoint_args(ckpt)
    assert args == ["--checkpoint"] + [str(ckpt)] * 4

    plain = graph_models.ChempropModel()
    args = plain._checkpoint_args(ckpt)
    assert args == ["--checkpoint", str(ckpt)]

    mve = graph_models.ChempropModel(uncertainty_method="mve")
    args = mve._checkpoint_args(ckpt)
    assert args == ["--checkpoint", str(ckpt)]


def test_descriptor_columns_appear_in_train_args() -> None:
    pytest.importorskip("chemprop")
    from cyp import graph_models

    model = graph_models.ChempropMultitargetModel(
        targets=["a", "b"], descriptor_columns=["d1", "d2"]
    )
    args = model._base_train_args("ignored")
    idx = args.index("--descriptors-columns")
    assert args[idx + 1 : idx + 3] == ["d1", "d2"]

    plain = graph_models.ChempropMultitargetModel(targets=["a", "b"])
    assert "--descriptors-columns" not in plain._base_train_args("ignored")


def test_fit_without_descriptors_rejects_a_model_that_needs_them() -> None:
    """A model built with descriptor_columns has an FFN sized for them -- fitting
    without any would silently train the wrong architecture rather than fail, so this
    must raise before chemprop is ever invoked."""
    pytest.importorskip("chemprop")
    from cyp import graph_models

    model = graph_models.ChempropMultitargetModel(
        targets=["a", "b"], descriptor_columns=["d1", "d2"]
    )
    with pytest.raises(ValueError, match="descriptor_columns"):
        model.fit(["CCO", "CCN"], np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_fit_with_descriptors_rejects_a_model_that_has_none() -> None:
    """The reverse mismatch: an array passed to a plain model would be silently
    ignored by chemprop rather than raising, since no --descriptors-columns names it."""
    pytest.importorskip("chemprop")
    from cyp import graph_models

    model = graph_models.ChempropMultitargetModel(targets=["a", "b"])
    with pytest.raises(ValueError, match="descriptor_columns"):
        model.fit(
            ["CCO", "CCN"],
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            descriptors=np.zeros((2, 2)),
        )


def test_descriptors_shape_is_checked() -> None:
    pytest.importorskip("chemprop")
    from cyp import graph_models

    model = graph_models.ChempropMultitargetModel(
        targets=["a", "b"], descriptor_columns=["d1", "d2"]
    )
    with pytest.raises(ValueError, match="shape"):
        model.fit(
            ["CCO", "CCN"],
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            descriptors=np.zeros((2, 3)),  # wrong number of descriptor columns
        )


def test_descriptor_columns_end_to_end_fit_and_predict(tmp_path) -> None:
    """A real (tiny) fit and predict, to prove --descriptors-columns actually reaches
    chemprop and the FFN's width agrees between train and predict.

    Pinned after a manual smoke test found predict fails outright (a matrix-shape
    RuntimeError from the FFN, not a quality problem) when descriptors are supplied
    at fit time but omitted at predict time -- the two must always travel together.
    """
    pytest.importorskip("chemprop")
    from cyp import graph_models

    smiles = ["CCO", "CCN", "CCC", "c1ccccc1", "CN1CCC[C@H]1c1cccnc1", "CCOCC", "CC(C)O", "CCCCO"]
    y = np.array([[4.5], [5.1], [3.2], [4.8], [5.5], [3.9], [4.1], [4.4]])
    descriptors = np.array(
        [
            [1.2, 0.3],
            [0.8, 0.1],
            [2.1, 0.5],
            [0.0, 0.9],
            [1.0, 0.2],
            [1.5, 0.4],
            [1.1, 0.35],
            [1.3, 0.28],
        ]
    )

    model = graph_models.ChempropMultitargetModel(
        targets=["target"],
        descriptor_columns=["d1", "d2"],
        model_dir=tmp_path / "model",
        epochs=3,
    )
    model.fit(smiles, y, descriptors=descriptors)
    preds = model.predict(smiles, descriptors=descriptors)
    assert preds.shape == (len(smiles), 1)
    assert np.isfinite(preds).all()

    with pytest.raises(ValueError, match="descriptor_columns"):
        model.predict(smiles)


# ── atom-level features, --atom-features-path (notebook 11's docking-derived arm) ──
#
# `--atom-features-path` takes a different code path through chemprop's CLI than
# `--descriptors-columns`: confirmed directly against the installed chemprop
# (2.2.1) that it is read once per training invocation and applied to whichever
# CSV chemprop is currently parsing, so it only works when `--data-path` is a
# single combined file split via `--splits-file` -- the two-separate-files pattern
# every other test above uses cannot pair one array with two differently-sized
# CSVs. `_fit_with_atom_features` is the code path that follows from that.


def test_atom_feature_width_appears_nowhere_in_train_args() -> None:
    """`atom_feature_width` only sizes the FFN's expected input; the CLI flag
    itself is added by `_fit_with_atom_features`, not `_base_train_args` -- unlike
    `descriptor_columns`, which chemprop reads from the CSV `--data-path` points
    to, atom features are a arg naming a wholly separate `.npz` file that depends
    on how `fit` splits its rows, not on anything `_base_train_args` can know."""
    pytest.importorskip("chemprop")
    from cyp import graph_models

    model = graph_models.ChempropMultitargetModel(targets=["a"], atom_feature_width=3)
    assert "--atom-features-path" not in model._base_train_args("ignored")


def test_fit_without_atom_features_rejects_a_model_that_needs_them() -> None:
    pytest.importorskip("chemprop")
    from cyp import graph_models

    model = graph_models.ChempropMultitargetModel(targets=["a"], atom_feature_width=3)
    with pytest.raises(ValueError, match="atom_feature_width"):
        model.fit(
            ["CCO", "CCN"],
            np.array([[1.0], [2.0]]),
            smiles_val=["CCF"],
            y_val=np.array([[1.0]]),
        )


def test_fit_with_atom_features_rejects_a_model_that_has_none(tmp_path) -> None:
    pytest.importorskip("chemprop")
    from cyp import graph_models

    model = graph_models.ChempropMultitargetModel(targets=["a"])
    with pytest.raises(ValueError, match="atom_feature_width"):
        model.fit(
            ["CCO", "CCN"],
            np.array([[1.0], [2.0]]),
            smiles_val=["CCF"],
            y_val=np.array([[1.0]]),
            atom_features_path=tmp_path / "x.npz",
        )


def test_atom_features_require_an_explicit_validation_split() -> None:
    """An internal random split reorders `smiles_train`, but an already-written
    `.npz`'s row order cannot be reordered along with it -- this must raise before
    chemprop is ever invoked, not silently misalign atoms to the wrong compound."""
    pytest.importorskip("chemprop")
    from cyp import graph_models

    model = graph_models.ChempropMultitargetModel(targets=["a"], atom_feature_width=3)
    with pytest.raises(ValueError, match="explicit smiles_val"):
        model.fit(
            ["CCO", "CCN", "CCC"],
            np.array([[1.0], [2.0], [3.0]]),
            atom_features_path=Path("/tmp/does_not_need_to_exist.npz"),
        )


def test_atom_features_path_shape_is_checked(tmp_path) -> None:
    pytest.importorskip("chemprop")
    from cyp import graph_models

    npz_path = tmp_path / "atom_feats.npz"
    # 2 compounds' worth of arrays, but fit below claims 3 total (2 train + 1 val).
    np.savez(npz_path, np.zeros((4, 3)), np.zeros((5, 3)))
    model = graph_models.ChempropMultitargetModel(targets=["a"], atom_feature_width=3)
    with pytest.raises(ValueError, match="arrays"):
        model.fit(
            ["CCO", "CCN"],
            np.array([[1.0], [2.0]]),
            smiles_val=["CCF"],
            y_val=np.array([[1.0]]),
            atom_features_path=npz_path,
        )


def test_atom_features_path_width_is_checked(tmp_path) -> None:
    pytest.importorskip("chemprop")
    from cyp import graph_models

    npz_path = tmp_path / "atom_feats.npz"
    # Right compound count, wrong column width (2 instead of the model's 3).
    np.savez(npz_path, np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)))
    model = graph_models.ChempropMultitargetModel(targets=["a"], atom_feature_width=3)
    with pytest.raises(ValueError, match="shaped"):
        model.fit(
            ["CCO", "CCN"],
            np.array([[1.0], [2.0]]),
            smiles_val=["CCF"],
            y_val=np.array([[1.0]]),
            atom_features_path=npz_path,
        )


def test_atom_features_end_to_end_fit_and_predict(tmp_path) -> None:
    """A real (tiny) fit and predict through the single-combined-file /
    `--splits-file` path, to prove `--atom-features-path` actually reaches
    chemprop and the encoder's message-passing width agrees between train and
    predict -- pinned after this repo's own discovery (notebook 11, 2026-09-19)
    that the two-separate-files pattern silently cannot carry a molecule-specific
    atom array at all, only after paying for an hour of upstream computation
    first. This is the regression test for that."""
    pytest.importorskip("chemprop")
    from rdkit import Chem

    from cyp import graph_models

    smiles_train = [
        "CCO",
        "CCN",
        "CCC",
        "c1ccccc1",
        "CN1CCC[C@H]1c1cccnc1",
        "CCOCC",
        "CC(C)O",
        "CCCCO",
        "CCCl",
        "CCBr",
        "CCF",
        "CCI",
        "COC",
        "CCCN",
        "c1ccncc1",
    ]
    smiles_val = ["CC(N)C(=O)O", "CCN(CC)CC", "c1ccc2ccccc2c1"]
    y_train = np.linspace(3.0, 6.0, len(smiles_train)).reshape(-1, 1)
    y_val = np.array([[4.2], [4.8], [5.1]])

    rng = np.random.default_rng(0)

    def make_npz(smis, path):
        arrs = [
            rng.random((Chem.MolFromSmiles(s).GetNumAtoms(), 3)).astype(np.float32) for s in smis
        ]
        np.savez(path, *arrs)

    train_npz = tmp_path / "combined.npz"
    make_npz(smiles_train + smiles_val, train_npz)

    model = graph_models.ChempropMultitargetModel(
        targets=["target"],
        atom_feature_width=3,
        model_dir=tmp_path / "model",
        epochs=5,
    )
    model.fit(
        smiles_train,
        y_train,
        smiles_val=smiles_val,
        y_val=y_val,
        atom_features_path=train_npz,
    )

    test_npz = tmp_path / "test.npz"
    make_npz(smiles_val, test_npz)
    preds = model.predict(smiles_val, atom_features_path=test_npz)
    assert preds.shape == (len(smiles_val), 1)
    assert np.isfinite(preds).all()

    with pytest.raises(ValueError, match="atom_feature_width"):
        model.predict(smiles_val)
