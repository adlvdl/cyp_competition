import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # 10 — pKa-weighted basic nitrogen for CYP2D6

    `09_cyp2d6_representation.py` confirmed CYP2D6's pharmacophore is real (a basic
    nitrogen near an aromatic system, +0.284 Spearman with potency) but found that
    encoding it as 3D geometry did not selectively help — `ecfp+pharmacophore_3d`
    lifted every endpoint, largest on CYP2C9, not CYP2D6. Independent published work
    (Rydberg & Olsen, *ACS Med Chem Lett* 2012, CYP2D6 site-of-metabolism) reached a
    similar shape of answer from a different angle: **topological** distance to a
    protonated nitrogen was sufficient there, with no conformers needed at all.

    That paper's descriptor is conditioned on a *protonated* nitrogen specifically.
    `pharmacophore.n_basic_nitrogen` is not — it is a binary sp3-nitrogen structural
    heuristic that treats a strongly basic aliphatic amine (pKa ~9-10, essentially
    always charged at physiological pH) identically to a weakly basic one next to an
    electron-withdrawing group (pKa ~4-5, essentially always neutral). The CYP2D6
    pharmacophore is specifically about the *cationic* nitrogen engaging
    Glu216/Asp301, so a feature that cannot distinguish those two cases is encoding
    the wrong quantity, independent of geometry.

    ## The feature

    `pka.basic_nitrogen_weight` (`src/cyp/pka.py`, wrapping the vendored MolGpKa GCN,
    `vendor/molgpka/`) replaces the binary indicator with the Henderson-Hasselbalch
    fraction protonated at pH 7.4, from a predicted per-atom pKa rather than an
    sp3/aromatic/amide structural rule. Same underlying chemistry, continuous instead
    of binary, and weighted by the actual physical quantity the mechanism depends on.

    ## Arms, on the same folds and harness as 09

    | arm | basic-N feature | what it isolates |
    |:--|:--|:--|
    | `ecfp` | none | the incumbent, repeated for reference |
    | `ecfp+pharmacophore_2d` | binary (`n_basic_nitrogen`) | 09's best non-3D arm |
    | `ecfp+pka_weight` | continuous (predicted pKa → fraction protonated) | does sharper help? |

    All four endpoints, scored in Spearman, for the same reason 09 did: a lift on
    CYP2D6 specifically, not on the lipophilicity-driven endpoints, is what would
    support the mechanism rather than just being a better feature in general.
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
    from cyp import cv, data, fingerprints, pharmacophore, pka

    return C, PROJECT_ROOT, Path, cv, data, fingerprints, np, pharmacophore, pka, pl, stats


@app.cell
def _(Path, PROJECT_ROOT):
    NOTEBOOK_NAME = Path(__file__).stem
    OUT_DIR = PROJECT_ROOT / "experiments" / NOTEBOOK_NAME
    CACHE_DIR = OUT_DIR / "cache"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR, NOTEBOOK_NAME, OUT_DIR


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Does predicted pKa track potency better than the binary indicator, directly?

    Before any model: the same style of check 09 ran for the 3D distance. If the
    continuous pKa weight does not correlate with CYP2D6 potency more strongly than
    the binary basic-N indicator, the modelling arms below are unlikely to show a
    different story.
    """
    )
    return


@app.cell
def _(C, CACHE_DIR, data, mo, pharmacophore, pka, pl, stats):
    ENDPOINTS = C.REGRESSION_ENDPOINTS

    _inh = data.load_train_inhibition()

    def endpoint_frame(endpoint: str) -> pl.DataFrame:
        return _inh.filter(pl.col(endpoint).is_not_null()).with_row_index("ridx")

    _rows = []
    for _endpoint in ENDPOINTS:
        _frame = endpoint_frame(_endpoint)
        _smiles = _frame["SMILES"].to_list()
        _y = _frame[_endpoint].to_numpy()

        _cache = CACHE_DIR / f"pka_weight_{_endpoint.split('_')[0]}.parquet"
        if _cache.exists():
            _weights = pl.read_parquet(_cache)["weight"].to_numpy()
        else:
            _weights = pka.basic_nitrogen_weights(_smiles)
            pl.DataFrame({"weight": _weights}).write_parquet(_cache)

        _binary = (
            pharmacophore.matrix(
                pharmacophore.descriptors(_smiles, include_3d=False), ["n_basic_nitrogen"]
            ).ravel()
            > 0
        ).astype(float)

        _rows.append(
            {
                "endpoint": _endpoint.split("_")[0],
                "binary_basicN_spearman": round(float(stats.spearmanr(_y, _binary).statistic), 4),
                "pka_weight_spearman": round(float(stats.spearmanr(_y, _weights).statistic), 4),
                "pct_nonzero_weight": round(100 * float((_weights > 0.01).mean()), 1),
            }
        )

    direct_check = pl.DataFrame(_rows)
    mo.ui.table(direct_check, selection=None)
    return ENDPOINTS, direct_check, endpoint_frame


@app.cell
def _(mo):
    mo.md(
        r"""
    ## CV across arms and endpoints

    LightGBM on ECFP4-2048 plus one extra column, same harness as `09` (3 outer
    repeats x 5 inner folds, seed 0) so the two notebooks' numbers are directly
    comparable without re-deriving a baseline.
    """
    )
    return


@app.cell
def _(CACHE_DIR, ENDPOINTS, cv, endpoint_frame, fingerprints, np, pl, pka):
    N_OUTER, N_INNER = 5, 5

    def ecfp_matrix(endpoint: str, smiles: list[str]) -> np.ndarray:
        path = CACHE_DIR / f"ecfp_{endpoint.split('_')[0]}.npy"
        if path.exists():
            return np.load(path)
        matrix = fingerprints.compute(smiles, "ecfp", n_bits=2048)
        np.save(path, matrix)
        return matrix

    def pka_weight_column(endpoint: str, smiles: list[str]) -> np.ndarray:
        cache = CACHE_DIR / f"pka_weight_{endpoint.split('_')[0]}.parquet"
        if cache.exists():
            return pl.read_parquet(cache)["weight"].to_numpy()
        weights = pka.basic_nitrogen_weights(smiles)
        pl.DataFrame({"weight": weights}).write_parquet(cache)
        return weights

    folds_by_endpoint = {}
    for _endpoint in ENDPOINTS:
        _frame = endpoint_frame(_endpoint)
        folds_by_endpoint[_endpoint] = [
            (_tr["ridx"].to_numpy(), _te["ridx"].to_numpy())
            for _f, _o, _i, _tr, _va, _te in cv.scaffold_splits(
                _frame, n_outer=N_OUTER, n_inner=N_INNER, seed=0
            )
        ]
    return ecfp_matrix, folds_by_endpoint, pka_weight_column


@app.cell
def _(
    ENDPOINTS,
    ecfp_matrix,
    endpoint_frame,
    folds_by_endpoint,
    np,
    pharmacophore,
    pka_weight_column,
    pl,
    stats,
):
    import lightgbm as lgb

    ARMS = ("ecfp", "ecfp+pharmacophore_2d", "ecfp+pka_weight")

    records = []
    for _endpoint in ENDPOINTS:
        _frame = endpoint_frame(_endpoint)
        _smiles = _frame["SMILES"].to_list()
        _y = _frame[_endpoint].to_numpy()
        _X_ecfp = ecfp_matrix(_endpoint, _smiles)
        _binary_col = (
            pharmacophore.matrix(
                pharmacophore.descriptors(_smiles, include_3d=False), ["n_basic_nitrogen"]
            )
            > 0
        ).astype(float)
        _pka_col = pka_weight_column(_endpoint, _smiles).reshape(-1, 1)

        for _arm in ARMS:
            if _arm == "ecfp":
                _X = _X_ecfp
            elif _arm == "ecfp+pharmacophore_2d":
                _X = np.hstack([_X_ecfp, _binary_col])
            else:
                _X = np.hstack([_X_ecfp, _pka_col])

            _oof = np.full(len(_y), np.nan)
            for _tr, _te in folds_by_endpoint[_endpoint]:
                _model = lgb.LGBMRegressor(n_estimators=300, random_state=0, verbose=-1)
                _model.fit(_X[_tr], _y[_tr])
                _oof[_te] = _model.predict(_X[_te])

            records.append(
                {
                    "endpoint": _endpoint.split("_")[0],
                    "arm": _arm,
                    "spearman": round(float(stats.spearmanr(_y, _oof).statistic), 4),
                }
            )

    results = pl.DataFrame(records)
    return ARMS, results


@app.cell
def _(OUT_DIR, mo, pl, results):
    results.write_csv(OUT_DIR / "arm_results.csv")
    spearman_table = results.pivot(values="spearman", index="arm", on="endpoint")

    _base = results.filter(pl.col("arm") == "ecfp").select(
        "endpoint", pl.col("spearman").alias("ecfp_spearman")
    )
    lift = (
        results.join(_base, on="endpoint")
        .with_columns((pl.col("spearman") - pl.col("ecfp_spearman")).round(4).alias("lift_vs_ecfp"))
        .filter(pl.col("arm") != "ecfp")
        .sort("arm", "endpoint")
    )
    lift.write_csv(OUT_DIR / "lift_vs_ecfp.csv")

    mo.vstack(
        [
            mo.ui.table(spearman_table, selection=None),
            mo.md(
                "**Read `ecfp+pka_weight` against `ecfp+pharmacophore_2d` on CYP2D6 "
                "specifically.** A lift there that the binary indicator did not "
                "already capture is the result this notebook is built to find; "
                "matching or losing to the binary arm says the sharper feature did "
                "not translate into a sharper model."
            ),
            mo.ui.table(lift, selection=None),
        ]
    )
    return lift, spearman_table


@app.cell
def _(mo):
    mo.md(
        r"""
    ## How to read this

    **This is a swap of one column, not a new representation.** Unlike `09`'s 3D
    arms, `ecfp+pka_weight` changes only whether the basic-nitrogen signal is binary
    or continuous — everything else about the feature set is identical to
    `ecfp+pharmacophore_2d`. That makes this the cleanest possible test of whether
    the *sharpness* of a feature, not its shape, was the limiting factor.

    **Apply the same standing rule as every other notebook.** These are point
    estimates on 15 folds. Before treating any lift here as real, run
    `evaluation.paired_bootstrap` between `ecfp+pka_weight` and
    `ecfp+pharmacophore_2d` on CYP2D6 specifically, and `holm_bonferroni` if more
    than one comparison is being read at once.

    **A null result here is still informative.** If a mechanistically sharper
    feature (real predicted pKa vs. a structural heuristic) does no better than the
    binary version, that argues the bottleneck is not feature precision on this one
    atom-level property — pushing the open question further toward either genuine
    chemical-space novelty (`02`'s scorecard: CYP2D6 has the lowest test-set
    nearest-neighbour similarity and the highest extrapolation rate of the four
    endpoints) or a representation that captures the *whole* pharmacophore
    (nitrogen position *and* the aromatic system's geometry together, which no arm
    in `09` or here tests jointly) rather than one atom property at a time.
    """
    )
    return


if __name__ == "__main__":
    app.run()
