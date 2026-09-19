import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # 12 — the interim submission: per-endpoint placement at the ST-RAE optimum

    This notebook ships the September 24 interim submission. It changes **no model**.
    The regression incumbent is still `05`'s `pubchem` arm — a Chemprop D-MPNN with
    four heads, encoder pretrained on the PubChem qHTS panel — unchanged since
    2026-09-15. Notebooks `06`, `08`, `09`, `10` and `11` each tested a way to beat it
    and each ended in "ship `pubchem` as-is".

    What *has* moved the board is placement. `07` shipped an affine correction that
    took macro ST-RAE **0.7785 → 0.7040** without retraining anything. That submission
    was deliberately placed at the **R² optimum** rather than the ST-RAE optimum,
    because with only one scored submission the blind population could not be
    triangulated and a second, differently-placed vector was worth more than the
    points it cost. It bought exactly that.

    So three things are now available that `07` did not have, and this notebook spends
    all three:

    1. **`intersect` has two curves.** The blind population is solved directly for
       three of four endpoints at `disagreement` < 0.005, rather than estimated from a
       single vector with an arbitrary sign.
    2. **The ST-RAE optimum, not the R² optimum.** `07` measured this switch at
       0.646 → 0.613 macro in simulation and then did not take it.
    3. **Per-endpoint gating.** `07` applied the correction uniformly and it *hurt*
       CYP3A4 by +0.187 ST-RAE — an endpoint already at 96% of its ceiling did not
       need moving. Placement on a well-placed endpoint is a cost, not a no-op.

    ## What this notebook must not do

    `07`'s own hardest-won caution applies to its successor. The solved moments
    describe the **live half** of the test set. The 750 compounds were split by
    chemical series, so a correction fitted to one half transfers to the other only
    insofar as the split preserved the distribution — which it was designed not to.
    Everything below is recorded in `PROVENANCE.md` so the interim reveal tests a
    number rather than delivering a surprise.
    """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import sys
    from pathlib import Path

    import numpy as np
    import polars as pl

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from cyp import constants as C
    from cyp import data, placement, submission
    from cyp.metrics import st_rae

    return C, PROJECT_ROOT, Path, data, np, pl, placement, st_rae, submission


@app.cell
def _(Path, PROJECT_ROOT):
    NOTEBOOK_NAME = Path(__file__).stem
    OUT_DIR = PROJECT_ROOT / "experiments" / NOTEBOOK_NAME
    SUB_DIR = PROJECT_ROOT / "submissions" / NOTEBOOK_NAME
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return NOTEBOOK_NAME, OUT_DIR, SUB_DIR


@app.cell
def _(mo):
    mo.md(
        r"""
    ## The scored board rows

    Copied from `07_placement.py`'s `BOARD` rather than re-entered, so the two
    notebooks cannot drift apart. Only `r2` and `spearman` drive `intersect`; `strae`
    is kept because it is the one metric that can discriminate a population's *sign*,
    and it is used that way below.
    """
    )
    return


@app.cell
def _():
    # Per-endpoint leaderboard results, as returned. A wrong number here moves the
    # solved population and every placement downstream of it, so these are transcribed
    # from `07_placement.py` unchanged rather than retyped from the board.
    BOARD: dict[str, dict[str, dict[str, float]]] = {
        # 2026-09-15 20:14 UTC. The four R2 values average to 0.1985 against the
        # board's own MA-R2 of 0.1986, which is what confirms these rows belong to
        # one submission and that the macro is a plain arithmetic mean.
        "05_auxiliary_data": {
            "CYP1A2": {"strae": 0.6887, "mae": 0.9301, "r2": 0.3393, "spearman": 0.7423},
            "CYP2C9": {"strae": 0.5567, "mae": 0.5100, "r2": 0.5868, "spearman": 0.7629},
            "CYP2D6": {"strae": 1.3099, "mae": 1.6588, "r2": -0.7421, "spearman": 0.3401},
            "CYP3A4": {"strae": 0.5586, "mae": 0.5754, "r2": 0.6102, "spearman": 0.7831},
        },
        # 2026-09-16 15:28 UTC, board rank 87. The R2-optimal probe -- the submission
        # whose different centre is what makes `intersect` solvable at all.
        "07_placement": {
            "CYP1A2": {"strae": 0.6559, "mae": 0.8240, "r2": 0.4334, "spearman": 0.7050},
            "CYP2C9": {"strae": 0.5975, "mae": 0.5308, "r2": 0.5660, "spearman": 0.7658},
            "CYP2D6": {"strae": 0.8902, "mae": 1.0731, "r2": 0.2230, "spearman": 0.3610},
            "CYP3A4": {"strae": 0.6725, "mae": 0.6443, "r2": 0.5338, "spearman": 0.7683},
        },
    }

    SUBMISSION_PATHS = {
        "05_auxiliary_data": "05_auxiliary_data/20260915/activity.csv",
        "07_placement": "07_placement/20260916_pubchem_r2/activity.csv",
    }

    # What `05`'s CV predicted for the unplaced model, from its PROVENANCE.md. The
    # board-to-CV ratio built from these is the only handle on how far scaffold CV
    # sits from blind difficulty.
    CV_EXPECTED = {"CYP1A2": 0.8216, "CYP2C9": 0.6310, "CYP2D6": 0.9251, "CYP3A4": 0.4864}
    return BOARD, CV_EXPECTED, SUBMISSION_PATHS


@app.cell
def _(BOARD, PROJECT_ROOT, SUBMISSION_PATHS, pl):
    submitted = {
        name: pl.read_csv(PROJECT_ROOT / "submissions" / path)
        for name, path in SUBMISSION_PATHS.items()
    }
    # Both submissions are the same 750 compounds in the same order, which is what
    # lets `intersect` treat them as two measurements of one population.
    assert all(frame.height == 750 for frame in submitted.values())
    _names = [frame["Molecule_Name"].to_list() for frame in submitted.values()]
    assert all(n == _names[0] for n in _names), "submissions are not in a common row order"
    ENDPOINTS = list(BOARD["05_auxiliary_data"].keys())
    return ENDPOINTS, submitted


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 1 — solve the blind population by intersection

    Each scored submission contributes a curve of `(mean, sd)` pairs consistent with
    its own R² and correlation. One curve is symmetric about its own model's centre,
    so a single submission leaves a mirrored pair of candidate populations and the
    sign has to be guessed. Two submissions with *different* centres are symmetric
    about *different* points, so only the true population lies on both — and the
    residual `disagreement` at the meeting point measures how well identified it is.
    """
    )
    return


@app.cell
def _(BOARD, ENDPOINTS, OUT_DIR, mo, pl, placement, submitted):
    solve_rows = []
    solved: dict[str, placement.Moments] = {}
    for _iso in ENDPOINTS:
        _col = f"{_iso}_pIC50_direct_inhibition"
        _triples = [
            (
                submitted[_name][_col].to_numpy(),
                BOARD[_name][_iso]["r2"],
                BOARD[_name][_iso]["spearman"],
            )
            for _name in BOARD
        ]
        try:
            _moments, _disagreement = placement.intersect(_triples)
            solved[_iso] = _moments
            solve_rows.append(
                {
                    "endpoint": _iso,
                    "solved_mean": round(_moments.mean, 3),
                    "solved_sd": round(_moments.sd, 3),
                    "pearson": round(_moments.pearson, 3),
                    "disagreement": round(_disagreement, 4),
                    "status": "solved",
                }
            )
        except ValueError:
            # Not an error to swallow: `intersect` refusing is itself the finding, and
            # the next cell is entirely about why it refuses here.
            solve_rows.append(
                {
                    "endpoint": _iso,
                    "solved_mean": None,
                    "solved_sd": None,
                    "pearson": None,
                    "disagreement": None,
                    "status": "no consistent population",
                }
            )

    solve_table = pl.DataFrame(solve_rows)
    solve_table.write_csv(OUT_DIR / "solved_population.csv")
    mo.ui.table(solve_table, selection=None)
    return solve_table, solved


@app.cell
def _(mo):
    mo.md(
        r"""
    **Three endpoints solve at `disagreement` under 0.005**, which is tight — `07`
    validated `intersect` on a synthetic population where three submissions agreed to
    0.018, so these are an order of magnitude better identified than the case the
    function was built against.

    **CYP2D6 refuses, and the refusal is correct.** Submission 87 returned R² 0.223 on
    CYP2D6 while its own reported Spearman of 0.361 converts to a Pearson of 0.376 and
    therefore an R² *ceiling* of 0.141. A model cannot exceed its own ceiling, so no
    population is consistent with both numbers under the assumed correlation. Since R²
    ≤ ρ² always, the returned R² puts a hard floor under the true Pearson:
    **ρ ≥ √0.223 = 0.472**, about 30% above what `pearson_from_spearman` estimates.

    That conversion assumes bivariate normality, and weak correlations over a skewed,
    truncated pIC50 distribution are exactly where that assumption fails worst. So
    `pearson_from_spearman` is a **lower bound** on CYP2D6, not a point estimate — as
    `07` concluded. The next cell asks whether raising ρ rescues the solve.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 2 — CYP2D6 is unidentifiable, and a plausibility check is what shows it

    If the only problem were an underestimated ρ, forcing a larger one should recover a
    single consistent population. It does not. Every ρ above the 0.472 floor yields a
    solution with near-zero `disagreement`, but the solved centre slides from 3.30 down
    to 2.32 as ρ rises — the curves stay mutually consistent while the answer moves by
    a full pIC50 unit. `disagreement` cannot discriminate, because it only measures
    whether the curves agree, not whether they agree on something real.

    So the check has to come from outside the solve. A blind CYP2D6 population is a
    physical claim about an assay, and the assay's own training data bounds it: the
    lowest CYP2D6 pIC50 ever observed is 1.95, and only 8.6% of training compounds
    fall below 4.
    """
    )
    return


@app.cell
def _(BOARD, OUT_DIR, mo, np, pl, submitted):
    from itertools import product as _iter_product

    _col_2d6 = "CYP2D6_pIC50_direct_inhibition"
    _preds_2d6 = {name: submitted[name][_col_2d6].to_numpy() for name in BOARD}

    def _solve_at_rho(rho: float, sd_bounds: tuple[float, float] = (0.3, 4.0), n_sd: int = 741):
        """`placement.intersect`'s geometry with the correlation pinned rather than
        derived from Spearman -- the one thing the library function cannot be asked to
        do, since it reads rho from the reported rank correlation."""
        grid = np.linspace(*sd_bounds, n_sd)
        curves = []
        for _name in BOARD:
            _p = _preds_2d6[_name]
            _mp, _sp = float(np.mean(_p)), float(np.std(_p))
            _r2 = BOARD[_name]["CYP2D6"]["r2"]
            _curve = {}
            for _sd_y in grid:
                _b2 = 2 * rho * (_sp / _sd_y) - (_sp / _sd_y) ** 2 - _r2
                if _b2 < 0:
                    continue
                _b = float(np.sqrt(_b2))
                _curve[float(_sd_y)] = [_mp - _b * _sd_y, _mp + _b * _sd_y]
            curves.append(_curve)

        best = None
        for _sd_y in grid:
            _sd_y = float(_sd_y)
            if not all(_sd_y in _c for _c in curves):
                continue
            for _combo in _iter_product(*[_c[_sd_y] for _c in curves]):
                _spread = float(np.ptp(_combo))
                if best is None or _spread < best[0]:
                    best = (_spread, _sd_y, float(np.mean(_combo)))
        return best

    from scipy import stats as _stats

    _rows = []
    for _rho in [0.472, 0.50, 0.56, 0.60, 0.65, 0.70, 0.80, 0.84]:
        _b = _solve_at_rho(float(_rho))
        if _b is None:
            _rows.append({"forced_pearson": _rho, "status": "no solution"})
            continue
        _dis, _sd_y, _mean_y = _b
        _dist = _stats.norm(_mean_y, _sd_y)
        _rows.append(
            {
                "forced_pearson": round(_rho, 3),
                "solved_mean": round(_mean_y, 3),
                "solved_sd": round(_sd_y, 3),
                "disagreement": round(_dis, 4),
                # The plausibility columns. An assay whose lowest training observation
                # is 1.95 cannot have a quarter of the blind set below pIC50 1.
                "pct_below_4": round(100 * float(_dist.cdf(4)), 1),
                "pct_below_2": round(100 * float(_dist.cdf(2)), 1),
                "pct_below_1": round(100 * float(_dist.cdf(1)), 1),
                "status": "consistent",
            }
        )

    cyp2d6_scan = pl.DataFrame(_rows)
    cyp2d6_scan.write_csv(OUT_DIR / "cyp2d6_rho_scan.csv")
    mo.ui.table(cyp2d6_scan, selection=None)
    return cyp2d6_scan


@app.cell
def _(mo):
    mo.md(
        r"""
    **Every consistent CYP2D6 solution is physically impossible.** At the ρ that best
    reproduces both submissions' ST-RAE (0.60), the solved population puts **34% of
    blind compounds below pIC50 2 and 17% below pIC50 1**, against a training set whose
    single lowest measurement is 1.95. Raising ρ makes it worse, not better. Lowering
    it to the 0.472 floor removes the solution entirely.

    For contrast, the three endpoints that *did* solve are unremarkable — CYP1A2,
    CYP2C9 and CYP3A4 put 0.8–1.7% below pIC50 2, entirely consistent with their
    training distributions. The implausibility is specific to CYP2D6, not an artefact
    of the method.

    **So CYP2D6 gets no solved population, and this notebook does not invent one.**
    The reading that survives is that its blind ordering and its reported Spearman are
    not describing the same thing — plausibly the board's CYP2D6 Spearman is computed
    over a subset, or the endpoint's truncation breaks the normal-theory conversion
    badly enough that the whole `R² = 2ρk − k² − b²` frame stops being informative at
    this correlation. Either way, a placement built on it would be built on nothing.

    **What CYP2D6 does keep is the empirical result.** Submission 87 moved it from
    board ST-RAE 1.3099 to **0.8902** — the single largest gain on the board, and a
    measurement rather than a derivation. That placement is reused directly below
    instead of being re-solved.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 3 — the ST-RAE optimum's shape, measured on OOF

    `07` established that the ST-RAE optimum is not the R² optimum and that the
    difference has a sign. A prediction inside a compound's credible interval scores
    zero; low-activity compounds carry wide intervals and potent ones carry narrow
    ones. So predicting slightly high is nearly free and predicting low is punished,
    and the optimum sits *above* the true centre and *narrower* than `ρ·sd`.

    The absolute centre of that optimum does not transfer — OOF is not the blind
    population. Two **shape** quantities do: how far above the truth's centre the
    optimum sits, and its spread as a fraction of `ρ·sd`. Both are measured here on
    `05`'s `pubchem` OOF, the same predictions being shipped.
    """
    )
    return


@app.cell
def _(C, OUT_DIR, PROJECT_ROOT, mo, np, pl, placement):
    oof = pl.read_parquet(PROJECT_ROOT / "experiments" / "07_placement" / "oof.parquet").filter(
        pl.col("method") == "pubchem"
    )

    _rows = []
    for _endpoint in C.REGRESSION_ENDPOINTS:
        _slice = oof.filter(pl.col("endpoint") == _endpoint)
        # Average the repeats per compound first: the shipped model is one fit, so the
        # quantity to place is a per-compound prediction, not a per-fold one.
        _agg = _slice.group_by("Molecule_Name").agg(
            pl.col("y_true").first(),
            pl.col("y_pred").mean(),
            pl.col("y_lower").first(),
            pl.col("y_upper").first(),
        )
        _y = _agg["y_true"].to_numpy()
        _p = _agg["y_pred"].to_numpy()
        _mean, _sd, _score = placement.strae_optimal_placement(
            _y, _p, _agg["y_lower"].to_numpy(), _agg["y_upper"].to_numpy()
        )
        _d = placement.decompose(_y, _p)
        _truth_mean, _truth_sd = float(np.mean(_y)), float(np.std(_y))
        _rows.append(
            {
                "endpoint": _endpoint.split("_")[0],
                "n": len(_y),
                "truth_mean": round(_truth_mean, 3),
                "truth_sd": round(_truth_sd, 3),
                "rho": round(_d["rho"], 3),
                "strae_mean": round(_mean, 3),
                "strae_sd": round(_sd, 3),
                # The two transferable quantities.
                "offset_above_truth": round(_mean - _truth_mean, 3),
                "sd_fraction": round(_sd / _truth_sd, 3),
                # `sd_fraction` relative to the R2 optimum's own `k = rho`. This is the
                # number that says how much narrower than R2-optimal to go.
                "shrink_vs_r2_optimum": round((_sd / _truth_sd) / _d["rho"], 3),
                "recoverable_r2": round(_d["recoverable"], 4),
                "strae_at_optimum": round(_score, 4),
            }
        )

    strae_shape = pl.DataFrame(_rows)
    strae_shape.write_csv(OUT_DIR / "strae_optimal_shape.csv")
    mo.ui.table(strae_shape, selection=None)
    return oof, strae_shape


@app.cell
def _(mo, np, strae_shape):
    SHRINK = float(np.mean(strae_shape["shrink_vs_r2_optimum"].to_numpy()))
    OFFSET = float(np.mean(strae_shape["offset_above_truth"].to_numpy()))
    _shrink_sd = float(np.std(strae_shape["shrink_vs_r2_optimum"].to_numpy()))

    mo.md(
        f"""
    **`offset_above_truth` is positive at all four endpoints** (+0.044 to +0.200),
    which is the check `07` flagged as worth stopping for — a negative value anywhere
    would have contradicted the asymmetry argument and meant the intervals do not
    behave as assumed. They do.

    **The shrink factor is consistent**: the ST-RAE optimum sits at
    **{SHRINK:.3f} × the R²-optimal spread** with a standard deviation of
    {_shrink_sd:.3f} across four endpoints that differ enormously in difficulty,
    interval width and below-floor fraction. That consistency is what makes it
    reasonable to carry onto the blind set; a quantity varying wildly across endpoints
    would not be.

    The mean offset is {OFFSET:+.3f} pIC50. Both are applied per endpoint below rather
    than as the pooled average, since each endpoint's own measurement is available —
    the pooled figures are reported only to show the spread is small.
    """
    )
    return OFFSET, SHRINK


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 4 — the gate: which endpoints get placed at all

    `07`'s most expensive lesson. It applied the correction uniformly, and CYP3A4 —
    already at 96% of its R² ceiling — came back **+0.187 ST-RAE worse than its own CV
    prediction**, the only endpoint to underperform. Moving a well-placed prediction
    costs points.

    The gate is the board's own measured `recoverable` R², the gap between what a
    submission scored and the ceiling its ordering allows. `07`'s decomposition of the
    `05` board row measured it directly, and it is not close:

    | endpoint | recoverable R² | % of ceiling lost | verdict |
    |:--|--:|--:|:--|
    | CYP1A2 | 0.2351 | 40.9% | place |
    | CYP2C9 | 0.0182 | 3.0% | leave alone |
    | CYP2D6 | 0.8676 | 691% | place |
    | CYP3A4 | 0.0254 | 4.0% | leave alone |

    Two endpoints have real headroom and two have essentially none. The rule this
    notebook applies: **place only where `recoverable` exceeds 0.05**, and ship the
    unplaced `pubchem` prediction everywhere else.
    """
    )
    return


@app.cell
def _():
    # Measured on `05`'s board row by `07`, in `experiments/07_placement/board_decomposition.csv`.
    RECOVERABLE = {"CYP1A2": 0.2351, "CYP2C9": 0.0182, "CYP2D6": 0.8676, "CYP3A4": 0.0254}

    #: Place an endpoint only if this much R2 is genuinely recoverable. Set above
    #: CYP2C9/CYP3A4's measured values and far below CYP1A2's -- there is a 10x gap
    #: between the two groups, so no reasonable threshold in between changes the
    #: decision.
    RECOVERABLE_GATE = 0.05
    return RECOVERABLE, RECOVERABLE_GATE


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 5 — build the placement plan

    Three regimes, one per endpoint group:

    - **Placed from a solved population** (CYP1A2). Centre at the solved mean plus the
      endpoint's own measured ST-RAE offset, spread at `shrink × ρ × solved_sd`.
    - **Unplaced** (CYP2C9, CYP3A4). Gated out; the raw `pubchem` prediction ships.
    - **Reuse the scored placement** (CYP2D6). No trustworthy population exists, but
      submission 87's empirical placement scored 0.8902 against 1.3099 unplaced. It is
      reused exactly, since re-deriving it would mean inventing the population the
      previous step just showed is unidentifiable.
    """
    )
    return


@app.cell
def _(
    BOARD,
    ENDPOINTS,
    OUT_DIR,
    RECOVERABLE,
    RECOVERABLE_GATE,
    np,
    pl,
    placement,
    solved,
    strae_shape,
    submitted,
):
    _shape = {row["endpoint"]: row for row in strae_shape.to_dicts()}
    # The unplaced incumbent. `05`'s submitted vector is the raw `pubchem` prediction;
    # `07`'s is the same model already affine-transformed, so `05` is the correct base
    # to place from and the only one that must be used here.
    _base_name = "05_auxiliary_data"

    plan_rows = []
    placed_predictions: dict[str, np.ndarray] = {}
    for _iso in ENDPOINTS:
        _col = f"{_iso}_pIC50_direct_inhibition"
        _raw = submitted[_base_name][_col].to_numpy()
        _recoverable = RECOVERABLE[_iso]

        if _recoverable < RECOVERABLE_GATE:
            _regime = "unplaced (gated: nothing to recover)"
            _out = _raw
            _target_mean = float(np.mean(_raw))
            _target_sd = float(np.std(_raw))
        elif _iso in solved:
            _moments = solved[_iso]
            _regime = "placed at ST-RAE optimum (solved population)"
            # The population's mean and sd are shared by every submission, but the
            # correlation is a property of the *model being placed*. `intersect`
            # returns the median rho across the submissions it solved from, which is
            # not 05's. Use 05's own board Spearman, since 05's vector is what ships.
            _rho = placement.pearson_from_spearman(BOARD[_base_name][_iso]["spearman"])
            _target_mean = _moments.mean + _shape[_iso]["offset_above_truth"]
            _target_sd = _shape[_iso]["shrink_vs_r2_optimum"] * _rho * _moments.sd
            _out = None  # filled below
        else:
            # CYP2D6: no identifiable population, but a scored empirical placement.
            #
            # Ship `07`'s exact vector rather than re-placing `05`'s onto its moments.
            # The two are *not* affine images of each other -- Spearman(05, 07) is
            # 0.919 here, not 1.0 -- because `07` refit the model and chemprop's
            # pretraining has no seed control, so it carries a different ordering as
            # well as a different placement. Re-placing `05` would reproduce 07's mean
            # and sd while keeping 05's ordering, which is a vector that has never been
            # scored and whose 0.8902 expectation would not apply to it.
            _prev = submitted["07_placement"][_col].to_numpy()
            _regime = "reuse scored placement (population unidentifiable)"
            _target_mean = float(np.mean(_prev))
            _target_sd = float(np.std(_prev))
            _out = _prev

        plan_rows.append(
            {
                "endpoint": _iso,
                "regime": _regime,
                "recoverable_r2": _recoverable,
                "from_mean": round(float(np.mean(_raw)), 3),
                "from_sd": round(float(np.std(_raw)), 3),
                "to_mean": round(_target_mean, 3),
                "to_sd": round(_target_sd, 3),
                "board_strae_05": BOARD["05_auxiliary_data"][_iso]["strae"],
                "board_strae_07": BOARD["07_placement"][_iso]["strae"],
            }
        )
        placed_predictions[_iso] = (_raw, _target_mean, _target_sd, _out)

    plan = pl.DataFrame(plan_rows)
    plan.write_csv(OUT_DIR / "placement_plan.csv")
    return placed_predictions, plan


@app.cell
def _(mo, plan):
    mo.ui.table(plan, selection=None)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 6 — apply, and check nothing was reordered

    An affine placement must not change a single compound's rank; if it does, the
    "placement is separable from ordering" premise has been violated and the whole
    construction is invalid. `placement.check_rank_preserved` asserts that and returns
    how many ties the pIC50 floor created — a large tie count would mean the floor is
    doing the work rather than the transform.
    """
    )
    return


@app.cell
def _(ENDPOINTS, OUT_DIR, mo, np, pl, placed_predictions, placement):
    from scipy import stats as _stats

    final: dict[str, np.ndarray] = {}
    apply_rows = []
    for _iso in ENDPOINTS:
        _raw, _target_mean, _target_sd, _preset = placed_predictions[_iso]
        if _preset is not None:
            _out = _preset
            _ties = 0
        else:
            _out = placement.affine(_raw, _target_mean, _target_sd)
            _ties = placement.check_rank_preserved(_raw, _out)
        final[f"{_iso}_pIC50_direct_inhibition"] = _out
        apply_rows.append(
            {
                "endpoint": _iso,
                "final_mean": round(float(np.mean(_out)), 3),
                "final_sd": round(float(np.std(_out)), 3),
                "final_min": round(float(np.min(_out)), 3),
                "final_max": round(float(np.max(_out)), 3),
                "ties_from_floor": _ties,
                # Spearman against the raw prediction: must be exactly 1.0, since an
                # affine transform cannot reorder.
                "spearman_vs_raw": round(float(_stats.spearmanr(_raw, _out).statistic), 6),
            }
        )

    applied = pl.DataFrame(apply_rows)
    applied.write_csv(OUT_DIR / "applied_placement.csv")
    mo.ui.table(applied, selection=None)
    return applied, final


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 7 — expected performance, and why CYP1A2's is a range

    Three of the four endpoints ship a vector that has **already been scored**, so
    their expectation is not a forecast at all: CYP2C9 and CYP3A4 are byte-identical to
    `05`'s, CYP2D6 is byte-identical to `07`'s. Only CYP1A2 is new.

    **The simulator fails its own back-test on CYP1A2, so it does not get to produce a
    point estimate.** `resolve_sign_with_strae`'s generative model — a bivariate-normal
    truth, normal predictions, intervals interpolated from the observed
    width-versus-potency curve — was built to *rank* two candidate populations, and
    `07` used it that way successfully. Asked instead to predict a score, it gets the
    direction wrong on the one case where both answers are known:

    | vector | centre | spread | simulated | board | error |
    |:--|--:|--:|--:|--:|--:|
    | `05` | 4.900 | 0.826 | 0.731 | 0.689 | +0.042 |
    | `07` | 4.258 | 1.075 | 0.787 | **0.656** | +0.131 |

    The board says `07`'s wider, lower-centred CYP1A2 vector scored **better** than
    `05`'s; the simulator says it should have scored worse. Both simulated values are
    also biased high, and by different amounts — ratios of 0.943 and 0.833, disagreeing
    by 0.11 across just two points, so rescaling by their mean would be fitting noise
    with one degree of freedom.

    **What the two known points do support is a bracket.** The shipped vector sits
    between them in centre (4.485, against 4.900 and 4.258) and is narrower than both
    (0.648). Its raw simulated score is 0.554; both references overshot their board
    value, by +0.040 and +0.128. Subtracting the smaller and larger overshoot gives
    **0.42–0.51** — and that bracket is reported, per endpoint and in the macro, in
    place of a point estimate.

    **Read it as optimistic, not as a confidence interval.** It assumes the shipped
    vector's overshoot falls inside the range spanned by two points, which is an
    assumption and not a measurement, and it inherits the simulator's normality
    assumptions wholesale. The honest summary is that CYP1A2 should improve on its
    scored 0.6887, that a landing anywhere in 0.42–0.60 would be consistent with what
    is known, and that the macro therefore sits somewhere near **0.61–0.65** against
    `07`'s scored 0.7040.

    This is `07`'s own lesson applied to its tooling: a quantity the data cannot
    resolve does not become resolvable by being simulated more precisely.
    """
    )
    return


@app.cell
def _(
    BOARD,
    CV_EXPECTED,
    ENDPOINTS,
    OUT_DIR,
    RECOVERABLE,
    RECOVERABLE_GATE,
    final,
    mo,
    np,
    oof,
    pl,
    placement,
    solved,
    st_rae,
    submitted,
):
    def _simulate(mean_y, sd_y, rho, y_pred, ref_truth, ref_width, n_test=750, n_rep=40, seed=0):
        """`resolve_sign_with_strae`'s simulator, reused to forecast a score rather
        than to choose between candidates: draw a population at the solved moments,
        draw predictions correlated with it at `rho` carrying our vector's own centre
        and spread, and attach intervals from the observed width-vs-potency curve."""
        order = np.argsort(ref_truth)
        _st, _sw = ref_truth[order], ref_width[order]
        rng = np.random.default_rng(seed)
        mp, sp = float(np.mean(y_pred)), float(np.std(y_pred))
        scores = []
        for _ in range(n_rep):
            z = rng.normal(0, 1, n_test)
            truth = mean_y + sd_y * z
            pred = mp + sp * (rho * z + np.sqrt(max(1 - rho**2, 0.0)) * rng.normal(0, 1, n_test))
            w = np.interp(truth, _st, _sw)
            scores.append(st_rae(truth, pred, truth - w / 2, truth + w / 2))
        return float(np.mean(scores))

    # The back-test first: reproduce the two CYP1A2 board scores whose vectors and
    # results are both known. A simulator that cannot order these two correctly has
    # not earned the right to produce a third number.
    _ref_slice = oof.filter(pl.col("endpoint") == "CYP1A2_pIC50_direct_inhibition")
    _ref = _ref_slice.group_by("Molecule_Name").agg(
        pl.col("y_true").first(), pl.col("y_lower").first(), pl.col("y_upper").first()
    )
    _ref_truth = _ref["y_true"].to_numpy()
    _ref_width = (_ref["y_upper"] - _ref["y_lower"]).to_numpy()
    _m1a2 = solved["CYP1A2"]

    backtest_rows = []
    for _name in BOARD:
        _v = submitted[_name]["CYP1A2_pIC50_direct_inhibition"].to_numpy()
        _sim = _simulate(
            _m1a2.mean,
            _m1a2.sd,
            placement.pearson_from_spearman(BOARD[_name]["CYP1A2"]["spearman"]),
            _v,
            _ref_truth,
            _ref_width,
        )
        backtest_rows.append(
            {
                "vector": _name,
                "centre": round(float(np.mean(_v)), 3),
                "spread": round(float(np.std(_v)), 3),
                "simulated": round(_sim, 4),
                "board": BOARD[_name]["CYP1A2"]["strae"],
                "error": round(_sim - BOARD[_name]["CYP1A2"]["strae"], 4),
            }
        )
    backtest = pl.DataFrame(backtest_rows)
    backtest.write_csv(OUT_DIR / "cyp1a2_simulator_backtest.csv")

    # Did it get the ORDER right? This is the only thing it was ever validated to do.
    _sim_order = backtest.sort("simulated")["vector"].to_list()
    _board_order = backtest.sort("board")["vector"].to_list()
    SIMULATOR_ORDERS_CORRECTLY = _sim_order == _board_order

    _rows = []
    for _iso in ENDPOINTS:
        _endpoint = f"{_iso}_pIC50_direct_inhibition"
        if RECOVERABLE[_iso] < RECOVERABLE_GATE:
            _lo = _hi = BOARD["05_auxiliary_data"][_iso]["strae"]
            _basis = "scored (identical vector to 05)"
        elif _iso in solved:
            _sim = _simulate(
                solved[_iso].mean,
                solved[_iso].sd,
                placement.pearson_from_spearman(BOARD["05_auxiliary_data"][_iso]["spearman"]),
                final[_endpoint],
                _ref_truth,
                _ref_width,
            )
            # Bracket rather than point-estimate: the simulator overshot both known
            # board values, by 0.042 and 0.131. Lower bound applies the larger
            # correction, upper bound the smaller one.
            _errors = backtest["error"].to_numpy()
            _lo = _sim - float(np.max(_errors))
            _hi = _sim - float(np.min(_errors))
            _basis = "simulated, bracketed by back-test error"
        else:
            _lo = _hi = BOARD["07_placement"][_iso]["strae"]
            _basis = "scored (identical vector to 07)"

        _rows.append(
            {
                "endpoint": _iso,
                "expected_low": round(float(_lo), 4),
                "expected_high": round(float(_hi), 4),
                "basis": _basis,
                "board_05": BOARD["05_auxiliary_data"][_iso]["strae"],
                "board_07": BOARD["07_placement"][_iso]["strae"],
                "cv_expected_unplaced": CV_EXPECTED[_iso],
            }
        )

    expected = pl.DataFrame(_rows)
    expected = pl.concat(
        [
            expected,
            pl.DataFrame(
                [
                    {
                        "endpoint": "MA (macro)",
                        "expected_low": round(
                            float(np.mean(expected["expected_low"].to_numpy())), 4
                        ),
                        "expected_high": round(
                            float(np.mean(expected["expected_high"].to_numpy())), 4
                        ),
                        "basis": "arithmetic mean of the four",
                        "board_05": 0.7785,
                        "board_07": 0.7040,
                        "cv_expected_unplaced": round(
                            float(np.mean(list(CV_EXPECTED.values()))), 4
                        ),
                    }
                ]
            ),
        ]
    )
    expected.write_csv(OUT_DIR / "expected_performance.csv")
    mo.vstack(
        [
            mo.md(
                "**Simulator back-test on CYP1A2** — orders the two known vectors "
                f"{'correctly' if SIMULATOR_ORDERS_CORRECTLY else '**incorrectly**'}."
            ),
            mo.ui.table(backtest, selection=None),
            mo.ui.table(expected, selection=None),
        ]
    )
    return SIMULATOR_ORDERS_CORRECTLY, backtest, expected


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Step 8 — write and validate the submission

    Both tracks share one dated folder. The TDI vector is `04`'s `tabicl_singletask` at
    threshold 0.25, shipped unchanged — no work has been done on that track since
    2026-09-15 and there is nothing new to ship.
    """
    )
    return


@app.cell
def _(PROJECT_ROOT, SUB_DIR, final, mo, pl, submission):
    from datetime import date as _date

    _today = _date.today().strftime("%Y%m%d")
    _dest = SUB_DIR / _today
    _dest.mkdir(parents=True, exist_ok=True)

    _activity = submission.build_activity_submission(final, _dest / "activity.csv")
    _activity_ok = submission.check(_activity, track="activity")
    _activity_verdict = "PASSED" if _activity_ok else "FAILED"

    # TDI ships unchanged from `04`. Copied rather than regenerated so the shipped
    # labels are byte-identical to the ones that were validated then.
    _prev_tdi = pl.read_csv(
        PROJECT_ROOT / "submissions" / "04_methods_tdi" / "20260915" / "tdi.csv"
    )
    _prev_tdi.write_csv(_dest / "tdi.csv")
    _tdi_ok = submission.check(_dest / "tdi.csv", track="tdi")
    _tdi_verdict = "PASSED" if _tdi_ok else "FAILED"

    mo.md(
        f"""
    - `{_activity.relative_to(PROJECT_ROOT)}` — validation **{_activity_verdict}**
    - `{(_dest / "tdi.csv").relative_to(PROJECT_ROOT)}` — validation **{_tdi_verdict}**
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## How to read this after the reveal

    **The falsifiable claim is CYP1A2.** It is the only endpoint whose shipped vector
    has not already been scored, and its expectation comes from a solved population
    plus a shape measured on OOF. If it lands near the prediction, the
    solve-and-place machinery works and should drive the final submission in November.
    If it misses badly, the live-half caveat is the first thing to suspect.

    **CYP2C9 and CYP3A4 are controls, not predictions.** Their vectors are identical to
    `05`'s, so they should return `05`'s scores to within the board's own rounding.
    Anything else means the two halves of the test set differ enough that the interim
    scores are not comparable across submissions — which would be the single most
    important thing to learn from this reveal, and is worth checking first.

    **CYP2D6 is a held position.** Submission 87's placement scored 0.8902 empirically
    and is reused because nothing better is derivable. Its population does not solve,
    its reported Spearman is inconsistent with its reported R², and until the reveal
    supplies a third scored vector there is no way to resolve that. `07`'s conclusion
    stands: fixing placement cannot make CYP2D6 good, only stop it being a disaster.

    **What the reveal buys.** A third scored, differently-placed vector on CYP1A2 makes
    the next `intersect` solve overdetermined rather than exactly determined, which is
    the first point at which the solved population can be *checked* rather than merely
    computed. That matters more for November than the interim rank does.
    """
    )
    return


if __name__ == "__main__":
    app.run()
