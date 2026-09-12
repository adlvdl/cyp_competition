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
    monkeypatch.setattr(
        tabular_models, "load_env", lambda *a, **k: real_load_env(env_file)
    )
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
    lgb.LGBMRegressor(n_estimators=5, verbose=-1).fit(
        rng.random((50, 10)), rng.random(50)
    )


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
    assert (
        tabular_models.TABPFN_MAX_FEATURES <= tabular_models._TABPFN_FEATURE_BUDGET
    )
    assert (
        tabular_models.TABPFN_MAX_FEATURES
        <= tabular_models.TABPFN_MODEL_FEATURE_LIMIT
    )
    assert tabular_models.TABPFN_N_ESTIMATORS <= 4


def test_the_two_tfms_have_separate_budgets() -> None:
    """TabPFN and TabICL fail differently -- TabICL's memory scales hard with
    ensemble size, TabPFN's barely does, and TabPFN actually uses the extra features
    where TabICL did not. Sharing one budget would mis-serve both."""
    assert tabular_models.TABICL_MAX_FEATURES != tabular_models.TABPFN_MAX_FEATURES
