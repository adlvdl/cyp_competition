"""Tests for the placement correction.

The module's claims are mathematical rather than empirical, so they can be pinned
exactly on planted populations where the answer is known -- which is the whole reason
for preferring this to the reference implementation's leaderboard-probing approach.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from cyp import placement as P


def _board(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """What a leaderboard would report for these predictions."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "r2": 1 - ss_res / ss_tot,
        "mae": float(np.abs(y_true - y_pred).mean()),
        "spearman": float(stats.spearmanr(y_true, y_pred).statistic),
        "pearson": float(np.corrcoef(y_true, y_pred)[0, 1]),
    }


def _submission(
    y_true: np.ndarray, centre: float, scale: float, corr: float, seed: int
) -> np.ndarray:
    """A prediction vector with a chosen centre, spread and correlation to truth."""
    noise = np.random.default_rng(seed).normal(0, 1, len(y_true))
    signal = (y_true - y_true.mean()) / y_true.std()
    return centre + scale * (corr * signal + np.sqrt(1 - corr**2) * noise)


def test_decomposition_reconstructs_r2_exactly():
    """`2*rho*k - k^2 - b^2` is an identity, not an approximation."""
    rng = np.random.default_rng(0)
    y_true = rng.normal(4.5, 1.2, 500)
    y_pred = _submission(y_true, centre=3.8, scale=0.7, corr=0.6, seed=1)

    assert P.decompose(y_true, y_pred)["r2"] == pytest.approx(_board(y_true, y_pred)["r2"])


def test_optimal_spread_is_rho_not_one():
    """The central claim: matching the truth's spread is worse than shrinking to rho.

    A model correlating 0.7 with reality should be 70% as wide as reality. Getting
    this wrong is regression-to-the-mean applied in the wrong direction.
    """
    rng = np.random.default_rng(3)
    y_true = rng.normal(4.0, 1.5, 800)
    y_pred = _submission(y_true, centre=4.7, scale=0.5, corr=0.7, seed=2)
    rho = _board(y_true, y_pred)["pearson"]

    def r2_at(k: float) -> float:
        placed = P.affine(y_pred, y_true.mean(), k * y_true.std(), floor=None)
        return _board(y_true, placed)["r2"]

    assert r2_at(rho) > r2_at(1.0)
    assert r2_at(rho) == pytest.approx(rho**2, abs=1e-9)
    # And it is a maximum, not merely better than one alternative.
    assert r2_at(rho) > r2_at(rho - 0.1)
    assert r2_at(rho) > r2_at(rho + 0.1)


def test_affine_placement_preserves_ranking():
    """A positive-scale affine map cannot reorder; only the floor can create ties."""
    rng = np.random.default_rng(5)
    y_pred = rng.normal(4.5, 0.6, 400)
    placed = P.affine(y_pred, target_mean=3.1, target_sd=1.6, floor=None)

    assert P.check_rank_preserved(y_pred, placed) == 0
    assert stats.spearmanr(y_pred, placed).statistic == pytest.approx(1.0)


def test_floor_ties_are_counted_not_hidden():
    """Predictions pushed below the floor are clamped, and the count is reported."""
    y_pred = np.linspace(4.0, 5.0, 100)
    placed = P.affine(y_pred, target_mean=P.FLOOR - 1.0, target_sd=0.5)

    assert P.check_rank_preserved(y_pred, placed) > 0
    assert placed.min() == pytest.approx(P.FLOOR)


@pytest.mark.parametrize(
    ("true_mean", "true_sd"),
    [(3.107, 1.599), (4.412, 1.553), (4.880, 1.272), (4.830, 1.101)],
)
def test_intersect_recovers_a_planted_population(true_mean: float, true_sd: float):
    """Several submissions' R2 curves meet at the population that produced them.

    This is the honest replacement for the reference entry's leaderboard probing: it
    needs no extra submission, only scores already earned by submissions that differ
    in placement.
    """
    y_true = np.random.default_rng(11).normal(true_mean, true_sd, 750)
    submissions = []
    for centre, scale, corr, seed in [
        (4.60, 0.50, 0.85, 1),
        (5.30, 0.90, 0.78, 2),
        (4.20, 1.40, 0.72, 3),
    ]:
        y_pred = _submission(y_true, centre, scale, corr, seed)
        board = _board(y_true, y_pred)
        submissions.append((y_pred, board["r2"], board["spearman"]))

    solved, disagreement = P.intersect(submissions)

    assert solved.mean == pytest.approx(true_mean, abs=0.25)
    assert solved.sd == pytest.approx(true_sd, abs=0.2)
    assert disagreement < 0.3


def test_intersect_needs_two_submissions():
    y_pred = np.random.default_rng(0).normal(4.5, 0.5, 100)
    with pytest.raises(ValueError, match="at least two"):
        P.intersect([(y_pred, 0.3, 0.6)])


def test_single_submission_solve_recovers_the_spread():
    """`solve_moments` gets the spread right even where it picks the wrong sign.

    Documented behaviour, not an aspiration: MAE is symmetric in the offset's
    direction, so one submission cannot identify which side of its own centre the
    population sits on. The spread is unaffected by that ambiguity, which is why a
    single-submission result is usable for shrinkage and not for shifting.
    """
    true_mean, true_sd = 4.880, 1.272
    y_true = np.random.default_rng(13).normal(true_mean, true_sd, 750)
    y_pred = _submission(y_true, centre=4.60, scale=0.50, corr=0.85, seed=1)
    board = _board(y_true, y_pred)

    solved = P.solve_moments(y_pred, r2=board["r2"], mae=board["mae"], spearman=board["spearman"])

    assert solved.sd == pytest.approx(true_sd, abs=0.15)
    # The centre is one of the two symmetric candidates; both are the same distance
    # from the model's own centre.
    offset = abs(float(np.mean(y_pred)) - solved.mean)
    assert offset == pytest.approx(abs(float(np.mean(y_pred)) - true_mean), abs=0.25)


def test_pearson_from_spearman_is_monotone_and_bounded():
    assert P.pearson_from_spearman(0.0) == pytest.approx(0.0)
    assert P.pearson_from_spearman(1.0) == pytest.approx(1.0)
    assert P.pearson_from_spearman(0.8) > P.pearson_from_spearman(0.5)
    # Slightly above the rank correlation for a bivariate normal, never below.
    for rho_s in (0.2, 0.4, 0.6, 0.8):
        assert P.pearson_from_spearman(rho_s) >= rho_s


def test_strae_optimum_sits_above_the_r2_optimum():
    """ST-RAE forgives high predictions and punishes low ones, so its centre is higher.

    The asymmetry comes from the credible intervals: low-activity compounds carry wide
    ones, so predicting a compound too inactive is nearly free, while the actives'
    narrow intervals punish under-prediction. This is why the reference entry
    deliberately mis-places CYP2D6 relative to its solved true centre.
    """
    rng = np.random.default_rng(17)
    n = 600
    y_true = rng.normal(3.5, 1.4, n)
    # Interval width falls with potency, as in the real assay.
    half_width = np.clip(2.2 - 0.35 * (y_true - y_true.min()), 0.15, 2.2)
    y_pred = _submission(y_true, centre=4.6, scale=0.5, corr=0.65, seed=4)

    strae_mean, _, _ = P.strae_optimal_placement(
        y_true, y_pred, y_true - half_width, y_true + half_width
    )

    assert strae_mean > y_true.mean()


def test_shrink_to_correlation_leaves_the_centre_alone():
    rng = np.random.default_rng(19)
    y_pred = rng.normal(4.5, 0.8, 500)
    shrunk = P.shrink_to_correlation(y_pred, pearson=0.6, floor=None)

    assert float(np.mean(shrunk)) == pytest.approx(float(np.mean(y_pred)))
    assert float(np.std(shrunk)) == pytest.approx(0.6 * float(np.std(y_pred)))


def test_moments_from_r2_returns_a_symmetric_pair():
    """Every spread admits two centres, above and below the model's own."""
    y_pred = np.random.default_rng(23).normal(4.5, 0.6, 300)
    candidates = P.moments_from_r2(y_pred, r2=0.35, pearson=0.75)

    assert candidates
    by_sd: dict[float, list[float]] = {}
    for candidate in candidates:
        by_sd.setdefault(candidate.sd, []).append(candidate.mean)
    for _sd, means in by_sd.items():
        if len(means) == 2:
            assert float(np.mean(means)) == pytest.approx(float(np.mean(y_pred)), abs=1e-9)


def test_r2_ceiling_requires_a_correlation():
    with pytest.raises(ValueError, match="pearson is required"):
        P.r2_optimal_placement(P.Moments(mean=4.0, sd=1.2))


def test_strae_resolves_the_sign_that_mae_cannot():
    """ST-RAE's interval asymmetry distinguishes the mirrored candidates.

    The gap this closes is real and was found on our own submission: at every
    endpoint the two mirrored populations implied MAEs identical to four decimals,
    so `solve_moments` was choosing the direction arbitrarily. ST-RAE is not
    symmetric in the offset -- low-activity compounds carry wide credible intervals
    and potent ones carry narrow ones -- so a population below the predictions scores
    differently from one equally far above.
    """
    rng = np.random.default_rng(31)
    true_mean, true_sd, rho = 3.1, 1.5, 0.45

    # A reference set describing how interval width falls with potency.
    reference_truth = rng.normal(true_mean, true_sd, 1200)
    reference_width = np.clip(1.6 - 0.30 * (reference_truth - reference_truth.min()), 0.18, 1.6)

    # Predictions sitting well above the population, as ours do on CYP2D6.
    z = rng.normal(0, 1, 750)
    truth = true_mean + true_sd * z
    y_pred = 4.5 + 0.5 * (rho * z + np.sqrt(1 - rho**2) * rng.normal(0, 1, 750))
    width = np.interp(truth, np.sort(reference_truth), reference_width[np.argsort(reference_truth)])
    from cyp.metrics import st_rae as _st_rae

    board = _st_rae(truth, y_pred, truth - width / 2, truth + width / 2)

    offset = float(np.mean(y_pred)) - true_mean
    candidates = [
        P.Moments(true_mean, true_sd, rho),
        P.Moments(float(np.mean(y_pred)) + offset, true_sd, rho),
    ]
    winner, simulated = P.resolve_sign_with_strae(
        y_pred, candidates, board, reference_truth, reference_width
    )

    assert winner.mean == pytest.approx(true_mean)
    assert len(simulated) == 2


def test_sign_resolver_needs_a_pearson():
    y_pred = np.random.default_rng(0).normal(4.5, 0.5, 100)
    truth = np.random.default_rng(1).normal(3.0, 1.0, 200)
    with pytest.raises(ValueError, match="pearson"):
        P.resolve_sign_with_strae(
            y_pred, [P.Moments(3.0, 1.0)], 1.0, truth, np.full(200, 0.5)
        )
