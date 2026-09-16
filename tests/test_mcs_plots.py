"""Tests for the comparison plots.

These are statistical figures, so the tests check the parts that could silently say
something false -- the sign of an effect, which verdict a colour encodes, whether
`top_n` trims before or after the Tukey correction -- rather than pixel output.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import pytest  # noqa: E402

from cyp import mcs  # noqa: E402


def _fold_scores(means: dict[str, float], n_folds: int = 12, sd: float = 0.01) -> pl.DataFrame:
    """Synthetic per-fold metrics with a known mean per method and shared fold noise.

    A common per-fold offset is what makes this a repeated-measures design, which is
    the structure both plots assume.
    """
    rng = np.random.default_rng(0)
    offsets = rng.normal(0, sd, n_folds)
    rows = []
    for method, mean in means.items():
        for f in range(n_folds):
            rows.append(
                {"method": method, "fold": f, "score": mean + offsets[f] + rng.normal(0, sd / 4)}
            )
    return pl.DataFrame(rows)


# ── verdict logic ──────────────────────────────────────────────────────────────


def test_verdict_reads_sign_through_higher_is_better() -> None:
    """The same raw difference means opposite things for MCC and ST-RAE, and getting
    this backwards would colour every plot the wrong way round."""
    # Lower ST-RAE is better, so a negative difference is an improvement.
    assert mcs._verdict(-0.05, 0.01, higher_is_better=False) == "better"
    assert mcs._verdict(+0.05, 0.01, higher_is_better=False) == "worse"
    # Higher MCC is better, so the reverse.
    assert mcs._verdict(+0.05, 0.01, higher_is_better=True) == "better"
    assert mcs._verdict(-0.05, 0.01, higher_is_better=True) == "worse"


def test_non_significant_is_similar_whatever_the_sign() -> None:
    """A large point estimate with a high p-value is still 'we cannot tell' -- the
    PXR lesson this repo exists to encode."""
    assert mcs._verdict(-0.5, 0.9, higher_is_better=False) == "similar"
    assert mcs._verdict(+0.5, 0.9, higher_is_better=True) == "similar"


# ── reference forest plot ──────────────────────────────────────────────────────


def test_forest_defaults_to_the_best_method_as_reference() -> None:
    frame = _fold_scores({"good": 0.5, "mid": 0.7, "bad": 0.9})
    fig = mcs.reference_forest_plot(frame, "score", higher_is_better=False)
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    # Best method first, and its row carries the reference marker at exactly zero.
    assert labels[0].startswith("good")


def test_forest_orders_best_to_worst_for_either_direction() -> None:
    frame = _fold_scores({"lo": 0.2, "mid": 0.5, "hi": 0.8})
    top_min = mcs.reference_forest_plot(frame, "score", higher_is_better=False).axes[0]
    top_max = mcs.reference_forest_plot(frame, "score", higher_is_better=True).axes[0]
    assert [t.get_text() for t in top_min.get_yticklabels()][0].startswith("lo")
    assert [t.get_text() for t in top_max.get_yticklabels()][0].startswith("hi")


def test_forest_rejects_an_unknown_reference() -> None:
    frame = _fold_scores({"a": 0.5, "b": 0.6})
    with pytest.raises(ValueError, match="not among"):
        mcs.reference_forest_plot(frame, "score", higher_is_better=False, reference="nope")


def test_forest_top_n_keeps_the_reference_even_if_it_ranks_low() -> None:
    """Asking 'is anything better than the incumbent?' must not drop the incumbent
    when the incumbent is losing -- that is precisely the interesting case."""
    frame = _fold_scores({"a": 0.1, "b": 0.2, "c": 0.3, "worst": 0.9})
    fig = mcs.reference_forest_plot(
        frame, "score", higher_is_better=False, reference="worst", top_n=2
    )
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert any(t.startswith("worst") for t in labels)
    assert len(labels) == 3  # top 2 plus the reference


def test_forest_reference_row_sits_at_zero() -> None:
    frame = _fold_scores({"a": 0.3, "b": 0.6})
    fig = mcs.reference_forest_plot(frame, "score", higher_is_better=False, reference="a")
    # The reference marker is the only point at exactly x=0.
    xs = [ln.get_xdata()[0] for ln in fig.axes[0].lines if ln.get_marker() == "o"]
    assert min(abs(x) for x in xs) == 0.0


def test_forest_writes_a_file(tmp_path) -> None:
    frame = _fold_scores({"a": 0.3, "b": 0.6})
    out = tmp_path / "forest.png"
    mcs.reference_forest_plot(frame, "score", higher_is_better=False, save_path=out)
    assert out.exists() and out.stat().st_size > 0


# ── paired arm plot ────────────────────────────────────────────────────────────


def test_paired_plot_has_one_row_per_model_not_per_arm() -> None:
    """The entire point: halve the category count by pairing."""
    frame = _fold_scores(
        {
            "lgbm_singletask": 0.5,
            "lgbm_multitask": 0.45,
            "xgb_singletask": 0.6,
            "xgb_multitask": 0.62,
        }
    )
    fig = mcs.paired_arm_plot(frame, "score", higher_is_better=False)
    assert len(fig.axes[0].get_yticklabels()) == 2


def test_paired_plot_difference_is_treatment_minus_control() -> None:
    """Sign errors here would invert the headline conclusion of a whole notebook."""
    # multitask is 0.1 *lower* than singletask, and lower is better here.
    frame = _fold_scores({"m_singletask": 0.6, "m_multitask": 0.5}, sd=0.001)
    fig = mcs.paired_arm_plot(frame, "score", higher_is_better=False)
    xs = [ln.get_xdata()[0] for ln in fig.axes[0].lines if ln.get_marker() == "o"]
    assert xs[0] < 0  # multitask - singletask is negative


def test_paired_plot_ignores_models_missing_an_arm() -> None:
    frame = _fold_scores({"a_singletask": 0.5, "a_multitask": 0.4, "lonely_singletask": 0.9})
    fig = mcs.paired_arm_plot(frame, "score", higher_is_better=False)
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert len(labels) == 1
    assert labels[0].startswith("a")


def test_paired_plot_raises_when_nothing_is_paired() -> None:
    frame = _fold_scores({"a_singletask": 0.5, "b_singletask": 0.6})
    with pytest.raises(ValueError, match="both a"):
        mcs.paired_arm_plot(frame, "score", higher_is_better=False)


def test_paired_plot_flags_a_real_difference_and_not_a_null_one() -> None:
    """A clearly separated pair must come out significant and a coincident pair must
    not -- otherwise the colours are decoration."""
    sep = _fold_scores({"m_singletask": 0.9, "m_multitask": 0.3}, sd=0.001)
    fig = mcs.paired_arm_plot(sep, "score", higher_is_better=False)
    colours = {ln.get_color() for ln in fig.axes[0].lines if ln.get_marker() == "o"}
    assert mcs.VERDICT_COLOURS["better"] in colours

    same = _fold_scores({"m_singletask": 0.5, "m_multitask": 0.5}, sd=0.02)
    fig2 = mcs.paired_arm_plot(same, "score", higher_is_better=False)
    colours2 = {ln.get_color() for ln in fig2.axes[0].lines if ln.get_marker() == "o"}
    assert mcs.VERDICT_COLOURS["similar"] in colours2


# ── grid trimming ──────────────────────────────────────────────────────────────


def test_top_n_trims_the_grid_to_the_best_methods() -> None:
    frame = _fold_scores({f"m{i}": 0.1 * i for i in range(10)})
    fig = mcs.make_mcs_grid(
        {"panel": frame}, metric_col="score", higher_is_better={"panel": False}, top_n=4
    )
    # 4 rows, not 10.
    assert len(fig.axes[0].get_yticklabels()) == 4


def test_top_n_respects_the_metric_direction() -> None:
    """Which end counts as 'top' flips with the metric, and keeping the wrong two
    would show the worst methods while claiming to show the best."""
    frame = _fold_scores({"lo": 0.1, "mid": 0.5, "worst": 0.9})
    lower = mcs.make_mcs_grid(
        {"p": frame}, metric_col="score", higher_is_better={"p": False}, top_n=2
    )
    higher = mcs.make_mcs_grid(
        {"p": frame}, metric_col="score", higher_is_better={"p": True}, top_n=2
    )
    # Labels read "name  (mean)"; the name is the leading token.
    kept_low = {t.get_text().split()[0] for t in lower.axes[0].get_yticklabels()}
    kept_high = {t.get_text().split()[0] for t in higher.axes[0].get_yticklabels()}
    assert kept_low == {"lo", "mid"}
    assert kept_high == {"worst", "mid"}


def test_top_n_below_two_is_rejected() -> None:
    frame = _fold_scores({"a": 0.1, "b": 0.5})
    with pytest.raises(ValueError, match="at least 2"):
        mcs.make_mcs_grid({"p": frame}, metric_col="score", higher_is_better={"p": False}, top_n=1)
