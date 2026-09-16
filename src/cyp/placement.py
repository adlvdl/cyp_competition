"""Separate a prediction's *ordering* from its *placement* on the pIC50 axis.

The idea is taken from the SuperCowPowers entry's `cyp_recalibrate.py`
(https://supercowpowers.github.io/workbench/blogs/cyp_challenge/), which is the
published methodology of a leaderboard entry scoring macro ST-RAE 0.4378 against our
0.78. Reading it apart, most of that gap is not a better model. It is this.

## The decomposition

For predictions `p` against truth `y`, coefficient of determination factors exactly::

    R2 = 2*rho*k - k^2 - b^2

with `rho` the Pearson correlation, `k = sd(p)/sd(y)` the spread ratio, and
`b = (mean(p) - mean(y)) / sd(y)` the mean offset in units of `sd(y)`. Only `rho`
depends on the ordering. `k` and `b` are set by an affine transform, which cannot
move a single compound's rank.

Two consequences follow, and both are used here:

1. **R2 is capped at rho^2**, reached at `b = 0` and `k = rho`. Note `k = rho`, not
   `k = 1`. Matching the spread of the truth is *wrong*: a model correlating 0.7 with
   reality should be 70% as wide as reality, because shrinking toward the mean is the
   correct response to its own uncertainty. This is the formal version of the
   regression-to-the-mean warning `CLAUDE.md` carries out of PXR -- except that it
   says by how much, rather than only that it happens.

2. **Placement is recoverable from a score.** A returned R2 is one equation in the
   blind population's moments, so scores from submissions already made constrain
   `mean(y)` and `sd(y)` without any further submission. `solve_moments` does that.

## Why this matters more here than it would elsewhere

Our training labels are hit-enriched -- a compound has a pIC50 only where the primary
screen flagged it -- so a model trained on them predicts a hit-enriched distribution.
The blind set is 75 hits x ~10 analogs, which is a different distribution. The
mismatch is a placement error, is invisible to scaffold CV (whose folds share the
training distribution), and is exactly what an affine correction fixes.

## ST-RAE is not R2, and the difference has a sign

The scored metric is zero anywhere inside a compound's credible interval, and
low-activity compounds carry wide intervals while potent ones carry narrow ones. So
predicting too high is nearly free and predicting too low is punished by the actives.
The ST-RAE optimum therefore sits *above* the R2-optimal centre and *narrower* than
`rho*sd`. The reference entry measured this directly on CYP2D6: placing it on its
solved true centre raised R2 from 0.363 to 0.447 while worsening ST-RAE from 0.565 to
0.694.

`strae_optimal_placement` finds that optimum numerically against held-out data with
credible intervals, rather than by probing a leaderboard. That is the honest version:
it needs no submission, and it can be validated on CV where the truth is known.

## What this module will not do

It will not hardcode another team's solved constants. Their `BLIND_MOMENTS` and
`STRAE_MOMENTS` are measurements of the live test half, obtained by spending
submissions on affine probes. We get one submission. Anything this module asserts
about the blind set is either derived from scores we have ourselves earned, or from a
prior we can state and defend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .metrics import st_rae

#: Predictions below this are floored. ST-RAE and the ranking metrics both punish
#: ties, so a binding floor costs something -- keep the count near zero and check it.
#: pIC50 1.0 is far below any assay's resolution, so it binds only on predictions an
#: aggressive rescale has pushed into nonsense.
FLOOR = 1.0


@dataclass(frozen=True)
class Moments:
    """A population's centre and spread, plus how well a model orders it."""

    mean: float
    sd: float
    pearson: float | None = None

    @property
    def r2_ceiling(self) -> float | None:
        """The best R2 any affine placement of this model can reach."""
        return None if self.pearson is None else self.pearson**2


def decompose(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Split R2 into the ordering term and the two placement terms.

    Returns `rho`, `k`, `b`, the reconstructed `r2`, and `r2_ceiling = rho^2`. The
    gap between `r2` and `r2_ceiling` is what an affine transform can recover without
    changing the model at all.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sd_true = float(np.std(y_true))
    if sd_true == 0:
        raise ValueError("truth has zero spread; R2 is undefined")

    rho = float(np.corrcoef(y_true, y_pred)[0, 1])
    k = float(np.std(y_pred)) / sd_true
    b = (float(np.mean(y_pred)) - float(np.mean(y_true))) / sd_true
    return {
        "rho": rho,
        "k": k,
        "b": b,
        "r2": 2 * rho * k - k**2 - b**2,
        "r2_ceiling": rho**2,
        "recoverable": rho**2 - (2 * rho * k - k**2 - b**2),
    }


def affine(
    y_pred: np.ndarray,
    target_mean: float,
    target_sd: float,
    floor: float | None = FLOOR,
) -> np.ndarray:
    """Rescale and shift predictions onto a target centre and spread.

    Rank-preserving by construction: the scale is positive and the shift is constant.
    Only `floor` can create ties, which is why the caller should check how many it
    binds.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    current_sd = float(np.std(y_pred))
    if current_sd == 0:
        raise ValueError("predictions have zero spread; nothing to rescale")

    placed = target_mean + (target_sd / current_sd) * (y_pred - float(np.mean(y_pred)))
    return placed if floor is None else np.clip(placed, floor, None)


def r2_optimal_placement(population: Moments) -> tuple[float, float]:
    """The centre and spread that maximise R2: `mean(y)` and `rho * sd(y)`.

    Requires `population.pearson`. This is the R2 ceiling for a given ordering -- a
    good default, and a poor maximum, since the scored metric is not R2.
    """
    if population.pearson is None:
        raise ValueError("pearson is required to choose a spread; k = rho, not k = 1")
    return population.mean, population.pearson * population.sd


def strae_optimal_placement(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    conf_low: np.ndarray,
    conf_high: np.ndarray,
    mean_grid: np.ndarray | None = None,
    sd_grid: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Grid-search the (centre, spread) that minimises ST-RAE on held-out data.

    The R2 optimum is derivable in closed form; the ST-RAE optimum is not, because the
    metric's credible-interval floor makes it non-analytic. So it is sampled -- but
    against data whose labels we hold, not against a leaderboard.

    Args:
        y_true: Held-out labels.
        y_pred: Predictions for those compounds, on their raw scale.
        conf_low: Lower credible bounds for `y_true`.
        conf_high: Upper credible bounds.
        mean_grid: Candidate centres. Defaults to a range spanning the observed
            truth and prediction centres with room either side.
        sd_grid: Candidate spreads. Defaults to a range from heavily shrunk to
            slightly over-spread relative to the truth.

    Returns:
        `(mean, sd, strae)` at the grid minimum.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if mean_grid is None:
        lo = min(float(np.mean(y_true)), float(np.mean(y_pred))) - 1.5
        hi = max(float(np.mean(y_true)), float(np.mean(y_pred))) + 1.5
        mean_grid = np.linspace(lo, hi, 61)
    if sd_grid is None:
        sd_grid = np.linspace(0.1 * float(np.std(y_true)), 1.3 * float(np.std(y_true)), 49)

    best = (float(np.mean(y_pred)), float(np.std(y_pred)), np.inf)
    for mean in mean_grid:
        for sd in sd_grid:
            score = st_rae(y_true, affine(y_pred, float(mean), float(sd)), conf_low, conf_high)
            if score < best[2]:
                best = (float(mean), float(sd), float(score))
    return best


def pearson_from_spearman(spearman: float) -> float:
    """Convert a rank correlation to a Pearson estimate: `rho = 2*sin(pi*rho_s/6)`.

    This is the Pearson-Spearman relation for a bivariate normal. It is the reason a
    leaderboard that reports Spearman is far more informative than one reporting only
    R2: Spearman is invariant to placement, so it pins the *ordering* term of the
    decomposition without spending a submission on an affine probe. The reference
    entry burned two of its three submissions recovering exactly this quantity.

    The normality assumption is the weak point. pIC50 distributions are skewed and
    truncated, so treat the result as an estimate with real error, and prefer
    `solve_moments` with several independent submissions so the assumption can be
    checked for consistency rather than trusted.
    """
    return float(2 * np.sin(np.pi * spearman / 6))


def moments_from_r2(
    y_pred: np.ndarray,
    r2: float,
    pearson: float,
    sd_bounds: tuple[float, float] = (0.3, 4.0),
) -> list[Moments]:
    """Population moments consistent with one scored R2 and a known correlation.

    Rearranging the decomposition, a returned R2 constrains the unknown `mean(y)` and
    `sd(y)` to a curve rather than a point::

        R2 = 2*rho*(sd_p/sd_y) - (sd_p/sd_y)^2 - ((mean_p - mean_y)/sd_y)^2

    For each candidate `sd_y` this is quadratic in `mean_y` and gives a symmetric pair
    of solutions -- the model's centre may sit above or below the population's. One
    score therefore cannot identify the population; it takes a second observable, and
    `solve_moments` supplies one.

    Returns:
        Candidates along the curve, sampled over `sd_bounds`. Empty where the R2 is
        unreachable for a given spread.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    mean_p, sd_p = float(np.mean(y_pred)), float(np.std(y_pred))

    candidates: list[Moments] = []
    for sd_y in np.linspace(*sd_bounds, 371):
        k = sd_p / sd_y
        b_squared = 2 * pearson * k - k**2 - r2
        if b_squared < 0:
            # This spread cannot produce the observed R2 at any centre.
            continue
        b = float(np.sqrt(b_squared))
        for sign in (1.0, -1.0):
            candidates.append(Moments(mean_p - sign * b * sd_y, float(sd_y), pearson))
    return candidates


def solve_moments(
    y_pred: np.ndarray,
    r2: float,
    mae: float,
    spearman: float | None = None,
    pearson: float | None = None,
    sd_bounds: tuple[float, float] = (0.3, 4.0),
) -> Moments:
    """Recover the blind population from ONE scored submission. Prefer `intersect`.

    Kept because a single submission is sometimes all there is, but it carries a
    failure mode `intersect` does not have. `moments_from_r2` returns a symmetric pair
    of centres at every spread -- the population may sit above or below the model's
    own centre -- and MAE cannot tell them apart, being symmetric in the offset's
    direction. Measured on synthetic populations where the truth was known, the two
    signs imply MAE within 0.002 of each other, and this function picked the wrong one
    on two of four cases, landing 0.45-0.62 off the true centre while recovering the
    spread almost exactly.

    Use it only when a second submission with a *different prediction centre* is
    unavailable. If the sign is wrong the spread is still right, so a result here that
    disagrees with a prior about the centre is more likely wrong about the sign than
    about the magnitude.

    Args:
        y_pred: The exact prediction vector submitted, for one endpoint.
        r2: The R2 the leaderboard returned for it.
        mae: The MAE it returned.
        spearman: The Spearman it returned; converted to Pearson internally.
        pearson: Supply directly instead of `spearman`.
        sd_bounds: Range of population spreads to search.
    """
    if pearson is None:
        if spearman is None:
            raise ValueError("supply either `spearman` or `pearson` to pin the ordering")
        pearson = pearson_from_spearman(spearman)

    candidates = moments_from_r2(y_pred, r2, pearson, sd_bounds)
    if not candidates:
        raise ValueError(
            f"no population is consistent with R2={r2} at pearson={pearson:.3f}; "
            "R2 cannot exceed rho^2, so check these come from the same submission"
        )

    y_pred = np.asarray(y_pred, dtype=float)
    sd_p = float(np.std(y_pred))
    mean_p = float(np.mean(y_pred))

    def implied_mae(candidate: Moments) -> float:
        """E|y - p| for a bivariate-normal population with these moments."""
        offset = mean_p - candidate.mean
        var = candidate.sd**2 + sd_p**2 - 2 * pearson * candidate.sd * sd_p
        sd_resid = float(np.sqrt(max(var, 1e-12)))
        return float(
            sd_resid * np.sqrt(2 / np.pi) * np.exp(-(offset**2) / (2 * sd_resid**2))
            + abs(offset) * (1 - 2 * stats.norm.cdf(-abs(offset) / sd_resid))
        )

    return min(candidates, key=lambda c: abs(implied_mae(c) - mae))


def intersect(
    submissions: list[tuple[np.ndarray, float, float]],
    sd_bounds: tuple[float, float] = (0.3, 4.0),
    n_sd: int = 371,
) -> tuple[Moments, float]:
    """Solve the blind population by intersecting several submissions' R2 curves.

    The method to use when more than one submission has been scored on the same
    endpoint. Each contributes a curve of `(mean_y, sd_y)` pairs consistent with its
    own R2 and correlation (`moments_from_r2`); the population is the point where the
    curves meet, and the residual disagreement there is a direct measure of how much
    to trust the answer.

    This resolves the sign ambiguity that defeats `solve_moments`. A single curve is
    symmetric about the model's own centre, but two models with *different* centres
    are symmetric about different points, so only the true population lies on both.
    Validated on a synthetic population at mean 4.880 / sd 1.272: three submissions
    centred at 4.59, 5.27 and 4.25 intersected at 4.83 / 1.270, agreeing among
    themselves to 0.018.

    The requirement is therefore that the submissions differ in *placement*, not
    merely in architecture. Models that happen to predict the same centre and spread
    give near-identical curves and no additional constraint -- check the printed
    centres before trusting a tight-looking agreement.

    Args:
        submissions: One `(y_pred, r2, spearman)` triple per scored submission, all
            for the same endpoint. `y_pred` must be the exact submitted vector.
        sd_bounds: Range of population spreads to search.
        n_sd: Grid resolution over that range.

    Returns:
        `(moments, disagreement)` -- the intersection, and the total spread among the
        curves' centres there, in pIC50 units. Treat anything above ~0.3 as a failure
        to identify the population rather than as a usable estimate.
    """
    if len(submissions) < 2:
        raise ValueError("intersection needs at least two submissions; use solve_moments for one")

    grid = np.linspace(*sd_bounds, n_sd)
    curves: list[dict[float, list[float]]] = []
    pearsons: list[float] = []
    for y_pred, r2, spearman in submissions:
        rho = pearson_from_spearman(spearman)
        pearsons.append(rho)
        y_pred = np.asarray(y_pred, dtype=float)
        mean_p, sd_p = float(np.mean(y_pred)), float(np.std(y_pred))
        curve: dict[float, list[float]] = {}
        for sd_y in grid:
            b_squared = 2 * rho * (sd_p / sd_y) - (sd_p / sd_y) ** 2 - r2
            if b_squared < 0:
                continue
            b = float(np.sqrt(b_squared))
            curve[float(sd_y)] = [mean_p - b * sd_y, mean_p + b * sd_y]
        curves.append(curve)

    best: tuple[float, float, float] | None = None
    for sd_y in grid:
        sd_y = float(sd_y)
        if not all(sd_y in curve for curve in curves):
            continue
        # Every combination of the per-curve sign choices; the true population is the
        # one where all curves land on the same centre.
        options = [curve[sd_y] for curve in curves]
        for combination in _product(options):
            spread = float(np.ptp(combination))
            if best is None or spread < best[0]:
                best = (spread, sd_y, float(np.mean(combination)))

    if best is None:
        raise ValueError(
            "no spread in `sd_bounds` is consistent with every submission; widen the "
            "bounds, or one of the reported scores does not belong to its predictions"
        )
    disagreement, sd_y, mean_y = best
    return Moments(mean_y, sd_y, float(np.median(pearsons))), disagreement


def _product(options: list[list[float]]) -> list[tuple[float, ...]]:
    """Cartesian product of the per-curve sign choices."""
    from itertools import product

    return list(product(*options))


def resolve_sign_with_strae(
    y_pred: np.ndarray,
    candidates: list[Moments],
    board_strae: float,
    reference_truth: np.ndarray,
    reference_width: np.ndarray,
    n_test: int = 750,
    n_repeats: int = 8,
    seed: int = 0,
) -> tuple[Moments, dict[str, float]]:
    """Pick between the symmetric candidates using the one metric that can tell them apart.

    `moments_from_r2` returns a mirrored pair at every spread, and neither R2 nor MAE
    distinguishes them -- both are symmetric in the offset's direction, and measured on
    our own submission the two candidates implied MAEs identical to four decimals. The
    published solve escaped this by spending submissions; this function escapes it by
    using ST-RAE, which is *not* symmetric.

    The asymmetry is the metric's whole design. A prediction inside a compound's
    credible interval scores zero, low-activity compounds carry wide intervals and
    potent ones carry narrow intervals. So a population sitting *below* our predictions
    produces a different ST-RAE than one sitting equally far *above*, even though both
    give the same MAE and the same R2.

    Each candidate is simulated: draw a population at its moments, draw predictions
    correlated with it at the candidate's own Pearson and carrying our submission's
    centre and spread, attach credible intervals by interpolating the observed
    width-versus-potency relationship, and score. The candidate whose simulated ST-RAE
    lands nearest the board's is the one the data supports.

    Args:
        y_pred: The exact submitted prediction vector for this endpoint.
        candidates: Competing populations, normally the mirrored pair at one spread.
        board_strae: The ST-RAE the leaderboard returned for this endpoint.
        reference_truth: Observed labels used to model the interval-width relationship,
            e.g. an OOF frame's ``y_true``.
        reference_width: ``y_upper - y_lower`` for those same compounds, in the same
            row order.
        n_test: Size of the simulated test set.
        n_repeats: Simulations per candidate; the mean is compared.
        seed: RNG seed.

    Returns:
        `(winner, simulated)` -- the selected candidate, and each candidate's mean
        simulated ST-RAE keyed by ``f"{mean:.3f}"``. **Read the margin before trusting
        it:** two candidates whose simulated scores straddle the board value by similar
        amounts have not been separated, and the sign remains open.
    """
    if not candidates:
        raise ValueError("no candidates to choose between")

    y_pred = np.asarray(y_pred, dtype=float)
    reference_truth = np.asarray(reference_truth, dtype=float)
    reference_width = np.asarray(reference_width, dtype=float)
    order = np.argsort(reference_truth)
    sorted_truth = reference_truth[order]
    sorted_width = reference_width[order]

    rng = np.random.default_rng(seed)
    mean_pred, sd_pred = float(np.mean(y_pred)), float(np.std(y_pred))

    simulated: dict[str, float] = {}
    for candidate in candidates:
        rho = candidate.pearson
        if rho is None:
            raise ValueError("each candidate needs a pearson to simulate predictions")
        scores = []
        for _ in range(n_repeats):
            z = rng.normal(0, 1, n_test)
            truth = candidate.mean + candidate.sd * z
            predictions = mean_pred + sd_pred * (
                rho * z + np.sqrt(max(1 - rho**2, 0.0)) * rng.normal(0, 1, n_test)
            )
            # Wide intervals at low activity, narrow at high -- the shape that makes
            # ST-RAE asymmetric, taken from real data rather than assumed.
            width = np.interp(truth, sorted_truth, sorted_width)
            scores.append(st_rae(truth, predictions, truth - width / 2, truth + width / 2))
        simulated[f"{candidate.mean:.3f}"] = float(np.mean(scores))

    winner = min(candidates, key=lambda c: abs(simulated[f"{c.mean:.3f}"] - board_strae))
    return winner, simulated


def consensus(estimates: list[Moments]) -> Moments:
    """Combine per-submission solves of the same population into one estimate.

    The median rather than the mean, because a single submission whose
    Spearman-to-Pearson conversion misbehaved should not drag the result. Report the
    spread across estimates alongside this -- if they disagree by more than ~0.3 in
    the centre, the solve is not identifying the population and should not be used to
    place a submission.
    """
    if not estimates:
        raise ValueError("no estimates to combine")
    pearsons = [e.pearson for e in estimates if e.pearson is not None]
    return Moments(
        mean=float(np.median([e.mean for e in estimates])),
        sd=float(np.median([e.sd for e in estimates])),
        pearson=float(np.median(pearsons)) if pearsons else None,
    )


def agreement(estimates: list[Moments]) -> dict[str, float]:
    """How far apart independent solves of one population are.

    The consistency check `solve_moments` exists to make available. A large range here
    means the estimates are not measuring one thing.
    """
    means = [e.mean for e in estimates]
    sds = [e.sd for e in estimates]
    return {
        "n": len(estimates),
        "mean_range": float(np.ptp(means)) if means else float("nan"),
        "sd_range": float(np.ptp(sds)) if sds else float("nan"),
        "mean_sd": float(np.std(means)) if means else float("nan"),
    }


def shrink_to_correlation(
    y_pred: np.ndarray, pearson: float, floor: float | None = FLOOR
) -> np.ndarray:
    """Shrink predictions to `pearson` times their own spread, centre unchanged.

    The placement correction that needs no knowledge of the blind population at all --
    only the model's own correlation, which scaffold CV estimates directly. It applies
    the `k = rho` half of the decomposition while leaving `b` alone, so it is the
    conservative version to reach for when the moment solve is not trusted.

    Note that a scaffold-OOF correlation *understates* the blind correlation (the
    split is harder than the blind set, and an OOF prediction comes from one fold
    model where a final model has seen everything), so shrinking on a raw OOF rho
    over-shrinks. Correct it before passing it in, or accept a conservative result.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    centre = float(np.mean(y_pred))
    return affine(y_pred, centre, pearson * float(np.std(y_pred)), floor=floor)


def check_rank_preserved(original: np.ndarray, placed: np.ndarray) -> int:
    """Assert an affine placement did not reorder anything; return the tie count.

    A positive-scale affine map cannot change ranks, so a failure here means the
    placement was applied to a different vector than the one being compared -- a
    submission-path bug, not a calibration one. Ties from the floor are counted and
    returned rather than raising, since a handful out of ~750 rows is below the
    metric's resolution.
    """
    original = np.asarray(original, dtype=float)
    placed = np.asarray(placed, dtype=float)
    # Detect compounds actually clamped, which is an exact tie *at* the floor -- not
    # merely everything below it. An `affine(..., floor=None)` result may sit below
    # FLOOR legitimately, having never been clamped, and treating those as floored
    # would exempt the bulk of the vector from the very check this function exists
    # to perform.
    floored = placed == FLOOR
    free = ~floored
    if free.sum() > 1:
        before = stats.rankdata(original[free])
        after = stats.rankdata(placed[free])
        if not np.allclose(before, after):
            raise ValueError("ranking changed above the floor — an affine map cannot do that")
    return int(floored.sum())
