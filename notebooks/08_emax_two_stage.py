import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # 08 — Emax as a two-stage feature: a negative result

    `07_placement.py` decomposed the board and found the remaining macro gap is almost
    entirely **ordering on CYP2D6**. Submission 87 scored Spearman 0.36 there against
    0.70–0.77 on the other three endpoints, which caps its R² at 0.14 whatever the
    placement. Lifting that Spearman to the level of its neighbours is worth ~0.10 on
    the macro R² — several times everything placement has left on all four endpoints
    combined.

    The most promising-looking lever for it was **Emax**. `CLAUDE.md` records that the
    direct-inhibition Emax readout correlates **+0.770** (Spearman) with CYP2D6 potency
    against −0.39 to −0.46 on the other isoforms, and calls it "the strongest auxiliary
    signal in the release" for exactly this endpoint. That correlation is higher than
    our *model's* ordering of the same endpoint.

    ## Why it cannot simply be used as a feature

    The blinded test set is `Molecule_Name,SMILES` and nothing else — `05` states this
    already: whatever a model reads at training time has to exist at prediction time
    too. So Emax cannot be an input column.

    The obvious repair is **two-stage**: train a model to predict Emax from structure,
    then feed the *predicted* Emax as a feature into the CYP2D6 model. That is a real
    design, not a leak, because stage 1 sees only SMILES. This notebook tests it.

    ## What has to be measured, and why an oracle arm is not enough

    A two-stage design has a ceiling and a realistic value, and they can be very far
    apart. Three arms on identical scaffold folds:

    | arm | Emax column fed to stage 2 | what it measures |
    |:--|:--|:--|
    | **baseline** | none | the ordering we already have |
    | **oracle** | the *measured* Emax | the ceiling, unattainable by construction |
    | **predicted** | stage-1 model output | what a submission could actually do |

    Reporting only the oracle arm would be the error this notebook exists to avoid: it
    answers "would Emax help if we knew it?", which is not the question. The gap
    between oracle and predicted is the whole result.

    Scored in **Spearman**, not ST-RAE. Ordering is the target here, and ST-RAE
    confounds ordering with placement — which is precisely what hid this gap until
    `07` decomposed the board.
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
    from scipy import stats

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from cyp import constants as C
    from cyp import cv, data, fingerprints

    return C, PROJECT_ROOT, Path, cv, data, fingerprints, np, pl, stats


@app.cell
def _(Path, PROJECT_ROOT):
    NOTEBOOK_NAME = Path(__file__).stem
    OUT_DIR = PROJECT_ROOT / "experiments" / NOTEBOOK_NAME
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return NOTEBOOK_NAME, OUT_DIR


@app.cell
def _(mo):
    mo.md(
        r"""
    ## The co-measurement check, which decides the shape of everything after it

    Before modelling anything: **where does Emax exist?** A feature is only worth a
    two-stage pipeline if stage 1 has training data the scored model does not already
    have. If Emax is recorded on exactly the compounds that already carry a pIC50,
    then stage 1 learns structure→Emax from the identical rows stage 2 learns
    structure→pIC50 from, and it cannot contribute information the direct model lacks.
    """
    )
    return


@app.cell
def _(C, OUT_DIR, data, mo, pl):
    _inh = data.load_train_inhibition()
    _em = data.load_train_emax()

    _rows = []
    for _iso in C.ISOFORMS:
        _pic50 = f"{_iso}_pIC50_direct_inhibition"
        _emax = f"{_iso}_EmaxVsPosCtrl_direct_inhibition"
        _j = _em.select(["Molecule_Name", _emax]).join(
            _inh.select(["Molecule_Name", _pic50]), on="Molecule_Name", how="full"
        )
        _has_e = _j[_emax].is_not_null().to_numpy()
        _has_p = _j[_pic50].is_not_null().to_numpy()
        _rows.append(
            {
                "isoform": _iso,
                "emax_only": int((_has_e & ~_has_p).sum()),
                "pic50_only": int((~_has_e & _has_p).sum()),
                "both": int((_has_e & _has_p).sum()),
            }
        )

    coverage = pl.DataFrame(_rows)
    coverage.write_csv(OUT_DIR / "emax_coverage.csv")

    mo.vstack(
        [
            mo.ui.table(coverage, selection=None),
            mo.md(
                "**`emax_only` is the column that matters.** It counts compounds where "
                "a stage-1 model would gain training signal the scored model does not "
                "already hold. Zero there means Emax and pIC50 are **co-measured** — "
                "both fall out of the same fitted dose-response curve, so Emax is "
                "another readout of that fit rather than independent evidence about "
                "the compound. That also inflates the headline +0.770 correlation "
                "relative to what an independently-measured feature would show."
            ),
        ]
    )
    return (coverage,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## The three arms

    LightGBM on ECFP4-2048 throughout. A tree model rather than Chemprop is the right
    choice here: the question is whether an extra *column* carries usable signal, and
    that is answered fastest and most legibly by a model that takes columns. If the
    predicted-Emax column were to help, re-testing it inside the graph model would be
    the follow-up — this arm ordering is deliberately cheap-first.

    Stage 1 is fitted **inside each training fold** and never sees the held-out
    compounds, so the predicted arm carries no leakage. Its own quality is recorded
    per fold, because a two-stage design can only be as good as its first stage and
    that number explains the result better than the final score does.
    """
    )
    return


@app.cell
def _(cv, data, fingerprints, pl):
    ENDPOINT = "CYP2D6_pIC50_direct_inhibition"
    EMAX_COL = "CYP2D6_EmaxVsPosCtrl_direct_inhibition"
    N_OUTER, N_INNER = 3, 5

    _inh = data.load_train_inhibition()
    _em = data.load_train_emax()
    frame = (
        _inh.join(_em.select(["Molecule_Name", EMAX_COL]), on="Molecule_Name", how="left")
        .filter(pl.col(ENDPOINT).is_not_null() & pl.col(EMAX_COL).is_not_null())
        .with_row_index("ridx")
    )

    X = fingerprints.compute(frame["SMILES"].to_list(), "ecfp", n_bits=2048)
    y = frame[ENDPOINT].to_numpy()
    emax = frame[EMAX_COL].to_numpy()

    folds = [
        (_tr["ridx"].to_numpy(), _te["ridx"].to_numpy())
        for _f, _o, _i, _tr, _va, _te in cv.scaffold_splits(
            frame, n_outer=N_OUTER, n_inner=N_INNER, seed=0
        )
    ]
    return EMAX_COL, ENDPOINT, N_INNER, N_OUTER, X, emax, folds, frame, y


@app.cell
def _(X, emax, folds, frame, mo, np, stats, y):
    import lightgbm as lgb

    def _fit(features, target, seed=0):
        return lgb.LGBMRegressor(n_estimators=300, random_state=seed, verbose=-1).fit(
            features, target
        )

    oof = {arm: np.full(len(y), np.nan) for arm in ("baseline", "oracle", "predicted")}
    stage1_quality = []

    for _fold, (_tr, _te) in enumerate(folds):
        # Arm 1 -- fingerprints alone, the ordering we already have.
        oof["baseline"][_te] = _fit(X[_tr], y[_tr]).predict(X[_te])

        # Arm 2 -- the measured Emax handed to the model. Unattainable on the blind
        # set, which has no Emax column; this is the ceiling, not a candidate.
        _f_tr = np.hstack([X[_tr], emax[_tr, None]])
        _f_te = np.hstack([X[_te], emax[_te, None]])
        oof["oracle"][_te] = _fit(_f_tr, y[_tr]).predict(_f_te)

        # Arm 3 -- stage 1 fitted on the training fold only, then its predictions used
        # as the feature. This is the arm a submission could actually ship.
        _stage1 = _fit(X[_tr], emax[_tr])
        _pred_tr = _stage1.predict(X[_tr])
        _pred_te = _stage1.predict(X[_te])
        _f_tr = np.hstack([X[_tr], _pred_tr[:, None]])
        _f_te = np.hstack([X[_te], _pred_te[:, None]])
        oof["predicted"][_te] = _fit(_f_tr, y[_tr]).predict(_f_te)

        stage1_quality.append(
            {
                "fold": _fold,
                "emax_spearman": float(stats.spearmanr(emax[_te], _pred_te).statistic),
            }
        )

    mo.md(f"Ran {len(folds)} scaffold folds on {frame.height:,} CYP2D6 compounds.")
    return oof, stage1_quality


@app.cell
def _(OUT_DIR, mo, np, oof, pl, stage1_quality, stats, y):
    _rows = []
    for _arm in ("baseline", "oracle", "predicted"):
        _p = oof[_arm]
        _ok = ~np.isnan(_p)
        _rows.append(
            {
                "arm": _arm,
                "spearman": round(float(stats.spearmanr(y[_ok], _p[_ok]).statistic), 4),
                "pearson": round(float(stats.pearsonr(y[_ok], _p[_ok]).statistic), 4),
            }
        )
    results = pl.DataFrame(_rows)
    results = results.with_columns(
        (pl.col("spearman") - pl.col("spearman").filter(pl.col("arm") == "baseline").first())
        .round(4)
        .alias("delta_vs_baseline")
    )
    results.write_csv(OUT_DIR / "two_stage_results.csv")

    stage1 = pl.DataFrame(stage1_quality)
    stage1.write_csv(OUT_DIR / "stage1_quality.csv")

    mo.vstack(
        [
            mo.ui.table(results, selection=None),
            mo.md(
                f"**Stage 1 predicts Emax at Spearman "
                f"{stage1['emax_spearman'].mean():.3f}** (mean over folds, "
                f"sd {stage1['emax_spearman'].std():.3f}) — which is the number that "
                "explains the whole table."
            ),
        ]
    )
    return results, stage1


@app.cell
def _(mo):
    mo.md(
        r"""
    ## What this measures

    **The oracle arm is spectacular and the predicted arm is worse than doing nothing.**
    Emax carries genuine information about CYP2D6 potency — the oracle arm proves that
    beyond doubt. But Emax is only predictable from structure at roughly the same
    quality as pIC50 itself, which is unsurprising once the coverage table is read:
    the two quantities come from the same curve fit on the same compounds, so nothing
    about Emax is easier to learn than the thing we already fail to learn.

    Feeding a noisy predicted column into stage 2 does not merely fail to help. It
    **hurts**, because the tree spends splits on a feature that is mostly noise, at the
    expense of fingerprint bits that are not.

    ### The general rule this is an instance of

    An auxiliary variable is only usable as a two-stage feature when it is **more
    predictable from structure than the target is**. Correlation with the target is
    necessary and nowhere near sufficient — a perfectly correlated variable that is
    exactly as hard to predict adds nothing, and a noisy estimate of it subtracts.

    Check the coverage table before the correlation table. An auxiliary column with
    zero rows outside the labelled set cannot supply stage 1 with anything the scored
    model does not already have, and no amount of correlation repairs that.

    ### What this does not rule out

    Emax remains legitimate as an auxiliary **target** — an extra head in a multitask
    model, which is how `05` used the challenge's unused arms. That shape is unaffected
    by this result: a head costs nothing at prediction time, needs no stage-1 model,
    and cannot inject noise into the scored endpoint's features. What is ruled out is
    Emax as an input *column*, by either route.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Where CYP2D6 ordering goes next

    The cheap shortcut is gone and the gap is unchanged: CYP2D6 orders at Spearman
    ~0.33–0.36 against 0.70–0.77 elsewhere, and that still caps the macro.

    `PLAN.md` names the standing hypothesis, written before any of this and still
    untested: CYP2D6 binding is dominated by a basic-nitrogen/aromatic pharmacophore
    that circular fingerprints represent poorly. Note that `PLAN.md` also rules out the
    easy alternative explanation — CYP2D6 is the *most precisely measured* endpoint
    (median credible interval 0.27), so the weak ordering is the model's failure and
    not the assay's.

    That points at representation and at 3D/structural information, which `PLAN.md`
    deferred to the window after the interim deadline. It is the one lever left that
    targets the binding mode directly rather than re-encoding the same fingerprint
    information.
    """
    )
    return


if __name__ == "__main__":
    app.run()
