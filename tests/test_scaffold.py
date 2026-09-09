"""Smoke tests: data is present, splits are sound, and submissions validate.

These guard the things that silently break a run -- a stale dataset, a leaky split,
a calibration that peeks at its own fold, and a submission the Space would reject.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from cyp import (
    calibration,
    cv,
    data,
    ensemble,
    evaluation,
    fingerprints,
    mcs,
    metrics,
    models,
    submission,
)
from cyp import constants as C

# ── data ──────────────────────────────────────────────────────────────────────


def test_snapshot_resolution():
    snapshots = C.available_snapshots()
    assert snapshots, "no data snapshots found -- run `make data`"
    assert all(s.isdigit() and len(s) == 8 for s in snapshots)
    assert C.latest_snapshot() == snapshots[-1]
    assert C.snapshot_dir(snapshots[-1]).is_dir()
    assert data.load_test(snapshot=snapshots[-1]).height == C.TEST_SET_SIZE


def test_test_set_shape():
    test = data.load_test()
    assert test.height == C.TEST_SET_SIZE
    assert set(test.columns) == {"Molecule_Name", "SMILES"}
    assert test["Molecule_Name"].n_unique() == C.TEST_SET_SIZE


def test_train_has_targets_and_intervals():
    train = data.load_train_inhibition()
    for endpoint in C.REGRESSION_ENDPOINTS:
        assert endpoint in train.columns
        assert endpoint + C.CONF_LOW_SUFFIX in train.columns
        assert endpoint + C.CONF_HIGH_SUFFIX in train.columns
        assert train[endpoint].is_not_null().sum() > 500


def test_training_frame_schema():
    frame = data.training_frame("CYP3A4_pIC50_direct_inhibition")
    assert set(frame.columns) == {"Molecule_Name", "SMILES", "y_true", "y_lower", "y_upper"}
    assert frame["y_true"].null_count() == 0
    # Credible intervals must bracket the point estimate.
    bracketed = frame.filter(
        pl.col("y_lower").is_not_null() & pl.col("y_upper").is_not_null()
    )
    assert (bracketed["y_lower"] <= bracketed["y_true"]).all()
    assert (bracketed["y_upper"] >= bracketed["y_true"]).all()


def test_tdi_labels_present():
    for isoform in C.TDI_ISOFORMS:
        frame = data.tdi_training_frame(isoform)
        assert frame.height > 1000
        assert frame["y_true"].any()  # both classes present
        assert not frame["y_true"].all()


# ── featurization & splits ────────────────────────────────────────────────────


def test_fingerprints_run():
    smiles = data.load_test()["SMILES"].head(32).to_list()
    fp = fingerprints.ecfp4(smiles, n_bits=512)
    assert fp.shape == (32, 512)
    assert fp.sum() > 0


def test_scaffold_splits_are_disjoint_and_grouped():
    frame = data.load_test().head(200)
    seen_test_ids: list[str] = []
    for _fold, _outer, _inner, train, val, test in cv.scaffold_splits(frame, n_outer=1):
        assert val is None
        ids_train = set(train["Molecule_Name"])
        ids_test = set(test["Molecule_Name"])
        assert not (ids_train & ids_test), "compound leaked across the split"
        # No scaffold may span train and test.
        scaf_train = {cv.murcko_scaffold(s) for s in train["SMILES"]}
        scaf_test = {cv.murcko_scaffold(s) for s in test["SMILES"]}
        assert not (scaf_train & scaf_test), "scaffold leaked across the split"
        seen_test_ids.extend(ids_test)
    # One outer repeat must partition the data exactly.
    assert len(seen_test_ids) == frame.height
    assert len(set(seen_test_ids)) == frame.height


def test_validation_split_is_carved_from_train():
    frame = data.load_test().head(100)
    for _, _, _, train, val, test in cv.random_splits(frame, n_outer=1, p_val=0.1):
        assert val is not None and val.height > 0
        assert not set(train["Molecule_Name"]) & set(val["Molecule_Name"])
        assert not set(val["Molecule_Name"]) & set(test["Molecule_Name"])
        break


# ── metrics ───────────────────────────────────────────────────────────────────


def test_st_rae_zero_inside_interval():
    y = np.array([5.0, 6.0, 4.0])
    lo, hi = y - 0.5, y + 0.5
    assert metrics.st_rae(y, y, lo, hi) == 0.0
    # Inside the band still costs nothing; outside costs distance to the edge.
    assert metrics.st_rae(y, y + 0.25, lo, hi) == 0.0
    assert metrics.st_rae(y, y + 2.0, lo, hi) > 0.0


def test_st_rae_mean_predictor_is_one():
    """The metric is normalized so a constant mean predictor scores exactly 1.0."""
    frame = data.training_frame("CYP3A4_pIC50_direct_inhibition")
    y = frame["y_true"].to_numpy()
    score = metrics.st_rae(
        y,
        np.full_like(y, y.mean()),
        frame["y_lower"].to_numpy(),
        frame["y_upper"].to_numpy(),
    )
    assert abs(score - 1.0) < 1e-9


# ── calibration ───────────────────────────────────────────────────────────────


def _fake_oof(n_per_fold: int = 20, n_folds: int = 25) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for fold in range(n_folds):
        for i in range(n_per_fold):
            y = float(rng.uniform(3, 8))
            rows.append(
                {
                    "method": "m",
                    "endpoint": "e",
                    "fold": fold,
                    "outer_fold": fold // 5,
                    "inner_fold": fold % 5,
                    "Molecule_Name": f"c{fold}_{i}",
                    "y_true": y,
                    # Deliberately shrunk toward the mean, the PXR failure mode.
                    "y_pred": 5.5 + 0.5 * (y - 5.5),
                }
            )
    return pl.DataFrame(rows)


def test_crossfit_calibration_corrects_shrinkage():
    oof = _fake_oof()
    raw_mae = float((oof["y_pred"] - oof["y_true"]).abs().mean())
    calibrated = calibration.crossfit_calibrate(oof, "linear")
    cal_mae = float((calibrated["y_cal"] - calibrated["y_true"]).abs().mean())
    assert cal_mae < raw_mae, "linear calibration should undo systematic shrinkage"
    assert calibrated.height == oof.height


def test_crossfit_calibration_does_not_leak():
    """A calibrator must never be fit on the fold it transforms."""
    oof = _fake_oof()
    for fold in (0, 7, 24):
        outer = fold // 5
        fit = oof.filter((pl.col("outer_fold") == outer) & (pl.col("fold") != fold))
        assert fold not in fit["fold"].to_list()
        assert fit["outer_fold"].unique().to_list() == [outer]


def test_raw_calibration_is_identity():
    oof = _fake_oof()
    out = calibration.crossfit_calibrate(oof, "raw").sort("Molecule_Name")
    ref = oof.sort("Molecule_Name")
    assert np.allclose(out["y_cal"].to_numpy(), ref["y_pred"].to_numpy())


# ── evaluation & ensembling ───────────────────────────────────────────────────


def test_paired_bootstrap_detects_no_difference():
    rng = np.random.default_rng(1)
    y = rng.uniform(3, 8, 200)
    a = y + rng.normal(0, 0.5, 200)
    b = y + rng.normal(0, 0.5, 200)
    result = evaluation.paired_bootstrap(y, a, b, metric="mae", n_resamples=500)
    assert result["p_value"] > 0.05, "equivalent models should not look different"

    # A genuinely worse model should be resolvable.
    worse = y + rng.normal(0, 2.0, 200)
    result = evaluation.paired_bootstrap(y, a, worse, metric="mae", n_resamples=500)
    assert result["diff"] < 0 and result["p_value"] < 0.05


def test_holm_bonferroni_is_monotone():
    table = evaluation.holm_bonferroni({"a": 0.001, "b": 0.30, "c": 0.04})
    assert table["p_value"].to_list() == sorted(table["p_value"].to_list())
    # Once a comparison fails, everything after it fails too.
    flags = table["significant"].to_list()
    assert flags == sorted(flags, reverse=True)


def test_weighted_average_normalizes_and_drops_zeros():
    preds = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0]), "c": np.array([9.0, 9.0])}
    blended = ensemble.weighted_average(preds, {"a": 1, "b": 1, "c": 0})
    assert np.allclose(blended, [2.0, 3.0])
    # Unnormalized weights give the same answer as normalized ones.
    assert np.allclose(blended, ensemble.weighted_average(preds, {"a": 5, "b": 5, "c": 0}))


def test_sweep_weights_finds_the_better_model():
    y = np.linspace(3, 8, 50)
    good, bad = y + 0.05, y - 3.0
    table = ensemble.sweep_weights({"good": good, "bad": bad}, y, metric="mae")
    best = table.row(0, named=True)
    assert best["w_good"] > best["w_bad"]


# ── multiple-comparison heatmaps ─────────────────────────────────────────────────


def _fold_scores_two_methods(gap: float, n_folds: int = 25) -> pl.DataFrame:
    """Synthetic per-fold metric frame: 'good' beats 'bad' by `gap` on average, with
    fold-to-fold interaction noise -- a constant gap with zero noise degenerates
    the repeated-measures error term (residual SS -> 0), which real CV data never
    does, so the noise here is what makes this a realistic test case."""
    rng = np.random.default_rng(0)
    rows = []
    for fold in range(n_folds):
        base = rng.uniform(0.5, 1.5)
        noise = rng.normal(0, 0.05)
        rows.append({"method": "good", "fold": fold, "st_rae": base})
        rows.append({"method": "bad", "fold": fold, "st_rae": base + gap + noise})
    return pl.DataFrame(rows)


def test_rm_tukey_hsd_detects_real_difference():
    frame = _fold_scores_two_methods(gap=0.5)
    _, means, diff, pc = mcs.rm_tukey_hsd(frame, "st_rae", "method")
    assert means.loc["bad", "st_rae"] > means.loc["good", "st_rae"]
    assert diff.loc["good", "bad"] < 0  # good scores lower (better) than bad
    assert pc.loc["good", "bad"] < 0.05


def test_rm_tukey_hsd_null_case_not_significant():
    """Two methods with no real gap (only sampling noise) should not be flagged as
    different -- the whole point of the correction. PXR's own retrospective is the
    cautionary tale: sorting by a difference this small produced a random ordering."""
    frame = _fold_scores_two_methods(gap=0.0)
    _, _, diff, pc = mcs.rm_tukey_hsd(frame, "st_rae", "method")
    assert pc.loc["good", "bad"] > 0.05
    assert abs(diff.loc["good", "bad"]) < 0.05


def _fold_scores_three_methods(n_folds: int = 25) -> pl.DataFrame:
    """best < middle < worst on a minimize-is-better metric, with fold noise."""
    rng = np.random.default_rng(1)
    rows = []
    for fold in range(n_folds):
        base = rng.uniform(0.5, 1.5)
        for name, offset in [("best", 0.0), ("middle", 0.5), ("worst", 1.0)]:
            rows.append(
                {
                    "method": name,
                    "fold": fold,
                    "st_rae": base + offset + rng.normal(0, 0.05),
                }
            )
    return pl.DataFrame(rows)


def test_rm_tukey_hsd_orders_best_to_worst_minimize():
    """For a minimize-is-better metric (default), the returned index should read
    best-to-worst, not alphabetically -- alphabetical here would be best, middle,
    worst by coincidence, so the frame is built to make that collision impossible."""
    frame = _fold_scores_three_methods()
    _, means, _, _ = mcs.rm_tukey_hsd(frame, "st_rae", "method")
    assert list(means.index) == ["best", "middle", "worst"]
    assert means["st_rae"].is_monotonic_increasing


def test_rm_tukey_hsd_orders_best_to_worst_maximize():
    """Same data, but framed as a maximize-is-better metric (e.g. MCC) by negating
    the values -- the order must reverse to best-to-worst under that direction too."""
    frame = _fold_scores_three_methods().with_columns((-pl.col("st_rae")).alias("mcc"))
    _, means, _, _ = mcs.rm_tukey_hsd(frame, "mcc", "method", higher_is_better=True)
    assert list(means.index) == ["best", "middle", "worst"]
    assert means["mcc"].is_monotonic_decreasing


def test_make_mcs_grid_orders_axes_best_to_worst():
    """End-to-end: the figure-producing entry point must apply the same ordering,
    driven by its `higher_is_better` argument."""
    frame = _fold_scores_three_methods()
    fig = mcs.make_mcs_grid(
        {"panel": frame}, metric_col="st_rae", higher_is_better={"panel": False}
    )
    ax = fig.axes[0]
    labels = [t.get_text().split("\n")[0] for t in ax.get_xticklabels()]
    assert labels == ["best", "middle", "worst"]
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_mcs_grid_saves_png(tmp_path):
    frame = _fold_scores_two_methods(gap=0.5)
    out_path = tmp_path / "mcs.png"
    fig = mcs.make_mcs_grid(
        {"CYP_TEST": frame},
        metric_col="st_rae",
        higher_is_better={"CYP_TEST": False},
        save_path=out_path,
    )
    assert out_path.exists() and out_path.stat().st_size > 0
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_macro_averaged_fold_metrics_matches_plain_mean():
    """The leaderboard's 'MA' row is a plain arithmetic mean across endpoints for
    each resample -- confirm the fold-index analogue does the same thing, on a
    case simple enough to check by hand."""
    ep_a = pl.DataFrame({"method": ["m", "m"], "fold": [0, 1], "st_rae": [1.0, 2.0]})
    ep_b = pl.DataFrame({"method": ["m", "m"], "fold": [0, 1], "st_rae": [3.0, 4.0]})
    macro = evaluation.macro_averaged_fold_metrics(
        {"A": ep_a, "B": ep_b}, metric_col="st_rae"
    )
    got = dict(zip(macro["fold"].to_list(), macro["st_rae"].to_list(), strict=True))
    assert got == {0: 2.0, 1: 3.0}  # mean(1,3)=2, mean(2,4)=3


def test_macro_averaged_fold_metrics_requires_full_endpoint_coverage():
    """A (method, fold) missing from one endpoint's frame must raise, not silently
    average over fewer endpoints -- that would quietly change what the number means
    for just that fold, which is worse than failing loudly."""
    ep_a = pl.DataFrame({"method": ["m", "m"], "fold": [0, 1], "st_rae": [1.0, 2.0]})
    ep_b = pl.DataFrame({"method": ["m"], "fold": [0], "st_rae": [3.0]})  # missing fold 1
    try:
        evaluation.macro_averaged_fold_metrics({"A": ep_a, "B": ep_b}, metric_col="st_rae")
    except ValueError as exc:
        assert "missing an endpoint" in str(exc)
    else:
        raise AssertionError("expected a ValueError for incomplete endpoint coverage")


def test_macro_average_feeds_tukey_hsd():
    """End-to-end: macro-average across two synthetic endpoints, then confirm Tukey
    HSD on the result still correctly separates a real difference from noise."""
    frame_a = _fold_scores_two_methods(gap=0.5)
    frame_b = _fold_scores_two_methods(gap=0.5)
    macro = evaluation.macro_averaged_fold_metrics(
        {"epA": frame_a, "epB": frame_b}, metric_col="st_rae"
    )
    _, means, diff, pc = mcs.rm_tukey_hsd(macro, "st_rae", "method")
    assert means.loc["bad", "st_rae"] > means.loc["good", "st_rae"]
    assert diff.loc["good", "bad"] < 0
    assert pc.loc["good", "bad"] < 0.05


# ── submissions ───────────────────────────────────────────────────────────────


def test_majority_baseline_scores_zero_mcc():
    """The line every real TDI classifier must clear -- MCC = 0.0 by construction,
    the classification analogue of the mean predictor's ST-RAE = 1.0."""
    from sklearn.metrics import matthews_corrcoef

    frame = data.tdi_training_frame("CYP3A4")
    y = frame["y_true"].to_numpy()
    model = models.MajorityBaseline().fit(np.zeros((len(y), 4)), y)
    pred = model.predict(np.zeros((len(y), 4)))
    assert matthews_corrcoef(y, pred) == 0.0


def test_tdi_cv_beats_majority_baseline():
    """A real classifier should clear the MCC=0.0 majority-baseline line -- the
    same anchor-against-a-baseline discipline as the regression track."""
    frame = data.tdi_training_frame("CYP3A4").head(400)
    oof = models.run_cv_classification(
        frame, "CYP3A4_is_TDI", methods=("majority", "lgbm"), n_outer=1, n_bits=256
    )
    summary = evaluation.compare_methods_classification(oof)
    majority_mcc = summary.filter(pl.col("method") == "majority")["mcc_mean"].item()
    lgbm_mcc = summary.filter(pl.col("method") == "lgbm")["mcc_mean"].item()
    assert majority_mcc == 0.0
    assert lgbm_mcc >= majority_mcc


def test_tdi_predictions_are_boolean():
    frame = data.tdi_training_frame("CYP2D6").head(300)
    test_smiles = data.load_test()["SMILES"].head(20).to_list()
    preds = models.fit_predict_test_classification(
        frame, test_smiles, method="lgbm", n_bits=256
    )
    assert preds.dtype == np.bool_
    assert len(preds) == 20


def test_built_submissions_validate(tmp_path):
    n = C.TEST_SET_SIZE
    activity = submission.build_activity_submission(
        {ep: np.full(n, 4.5) for ep in C.REGRESSION_ENDPOINTS}, tmp_path / "activity.csv"
    )
    ok, errors = submission.validate_activity(activity)
    assert ok, errors

    tdi = submission.build_tdi_submission(
        {f"{cyp}_is_TDI": np.zeros(n, dtype=bool) for cyp in C.TDI_ISOFORMS},
        tmp_path / "tdi.csv",
    )
    ok, errors = submission.validate_tdi(tdi)
    assert ok, errors

    written = pl.read_csv(activity)
    assert written.columns == C.SUBMISSION_ACTIVITY_COLUMNS
    assert written.height == n


def test_submission_rejects_missing_predictions(tmp_path):
    """A DataFrame that covers only some test compounds must fail loudly."""
    partial = pl.DataFrame(
        {"Molecule_Name": data.load_test()["Molecule_Name"].head(10)}
    ).with_columns([pl.lit(5.0).alias(ep) for ep in C.REGRESSION_ENDPOINTS])
    try:
        submission.build_activity_submission(partial, tmp_path / "bad.csv")
    except ValueError as exc:
        assert "missing predictions" in str(exc).lower()
    else:
        raise AssertionError("expected a ValueError for incomplete predictions")
