import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # 18 — Does the ensemble ranking survive the blind set's shift?

    `17` found the Monroe + `pubchem` blend beats `pubchem` by ~0.05 macro ST-RAE, and that
    pools A (2 models), B (4) and C (20) differ by 0.005–0.014 with the larger pool ahead.
    A resampling at blind-set size said the blend gap is safe and the pool gaps hold in
    72–99% of draws. That resampling stayed inside the CV distribution. PXR failed on a
    shift: hit rate 1.6% → 8.1% and cliff share 35% → 60% between train and test, which
    reordered finalists that CV had separated by 0.001.

    This notebook reweights the OOF toward the blind set on the two shift axes that can be
    measured without blind labels, then repeats the ranking test.

    | scheme | matched on | blind side |
    |:--|:--|:--|
    | `novelty` | nearest-neighbour ECFP4 similarity | blind vs an 80% training subset |
    | `potency` | predicted pIC50 | `05`'s submitted `pubchem` blind predictions |
    | `both` | product of the two weights | — |

    **Pre-registered failure criterion.** C is not worth its cost if, under any scheme, its
    advantage over A or B on CYP2D6 changes sign, or `p_first_better` for C vs B falls below
    0.5 at 300 compounds per endpoint. The effective sample size of each weighting is
    reported, and a scheme whose ESS collapses is not trusted.

    Limit: two axes. A shift in the label distribution itself (PXR's hit rate) cannot be
    seen without blind labels, and `potency` is only a proxy for it.
    """
    )
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import polars as pl

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from rdkit import DataStructs

    from cyp import constants as C
    from cyp import cv, data, evaluation, similarity
    from cyp.metrics import st_rae

    OUT_DIR = PROJECT_ROOT / "experiments" / Path(__file__).stem
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return (
        C,
        DataStructs,
        OUT_DIR,
        PROJECT_ROOT,
        cv,
        data,
        evaluation,
        mo,
        np,
        pl,
        similarity,
        st_rae,
    )


@app.cell
def _(C, PROJECT_ROOT, pl):
    KEYS = ["endpoint", "fold", "Molecule_Name"]
    ENDPOINTS = list(C.REGRESSION_ENDPOINTS)
    _e17 = PROJECT_ROOT / "experiments" / "17_regression_ensemble"
    _pubchem = (
        pl.read_parquet(PROJECT_ROOT / "experiments" / "05_auxiliary_data" / "oof.parquet")
        .filter(pl.col("method") == "pubchem")
        .select(
            *KEYS, "outer_fold", "y_true", "y_lower", "y_upper", pl.col("y_pred").alias("pubchem")
        )
    )
    _ens = pl.read_parquet(_e17 / "ensemble_oof.parquet").pivot(
        on="method", index=KEYS, values="y_pred"
    )
    OOF = _pubchem.join(
        _ens.select(*KEYS, "caruana_A", "caruana_B", "caruana_C", "uniform_A"), on=KEYS, how="inner"
    )
    assert OOF.height == _pubchem.height, "17's ensemble OOF is missing rows"
    OOF.group_by("endpoint").len().sort("endpoint")
    return ENDPOINTS, KEYS, OOF


@app.cell
def _(DataStructs, ENDPOINTS, KEYS, OOF, PROJECT_ROOT, cv, data, np, pl, similarity):
    # Nearest-neighbour similarity, ECFP4 ("ecfp4_1k", the fingerprint 02's scorecard used),
    # of every OOF row to ITS OWN fold's training split. Against the full set a held-out
    # row would find itself.
    FP = "ecfp4_1k"
    _frames = {_e: data.training_frame(_e) for _e in ENDPOINTS}
    _, _assign = cv.shared_scaffold_folds(_frames)

    _nn_rows = []
    TRAIN_BV = {}
    for _e, _f in _frames.items():
        _bv = similarity.bitvects_for(_f, FP)
        TRAIN_BV[_e] = _bv
        _idx = _f.with_row_index("_row")
        for _fd, _o, _i, _tr, _v, _te in cv.fold_assignment_splits(_idx, _assign):
            _train_bv = [_bv[j] for j in _tr["_row"].to_list()]
            _sims = np.array(
                [
                    max(DataStructs.BulkTanimotoSimilarity(_bv[j], _train_bv))
                    for j in _te["_row"].to_list()
                ]
            )
            _nn_rows.append(
                pl.DataFrame(
                    {
                        "endpoint": _e,
                        "fold": _fd,
                        "Molecule_Name": _te["Molecule_Name"].to_list(),
                        "nn_sim": _sims,
                    }
                )
            )
    NN = pl.concat(_nn_rows)
    OOF_NN = OOF.join(NN, on=KEYS, how="inner")
    assert OOF_NN.height == OOF.height

    # Blind side. Against an 80% random subset of each endpoint's training compounds (five
    # draws pooled), because the CV models each saw 80%; against 100% the blind compounds
    # would look artificially closer. The full-set figure is kept as a sensitivity.
    _blind = pl.read_csv(
        next((PROJECT_ROOT / "data" / "raw").glob("*/cyp-challenge-TEST-BLINDED.csv"))
    )
    BLIND = _blind
    _bbv = similarity.bitvects_for(_blind, FP)
    _rng = np.random.default_rng(0)
    BLIND_NN80, BLIND_NN100 = {}, {}
    for _e in ENDPOINTS:
        _n = len(TRAIN_BV[_e])
        BLIND_NN100[_e] = np.array(
            [max(DataStructs.BulkTanimotoSimilarity(b, TRAIN_BV[_e])) for b in _bbv]
        )
        _pooled = []
        for _ in range(5):
            _sub = [TRAIN_BV[_e][j] for j in _rng.choice(_n, int(0.8 * _n), replace=False)]
            _pooled.append(
                np.array([max(DataStructs.BulkTanimotoSimilarity(b, _sub)) for b in _bbv])
            )
        BLIND_NN80[_e] = np.concatenate(_pooled)

    _q = [0.1, 0.25, 0.5, 0.75, 0.9]
    _rows = []
    for _e in ENDPOINTS:
        _o = OOF_NN.filter(pl.col("endpoint") == _e)["nn_sim"].to_numpy()
        _rows.append(
            {
                "endpoint": _e,
                "set": "OOF (80% train)",
                **{f"p{int(q * 100)}": round(float(np.quantile(_o, q)), 3) for q in _q},
            }
        )
        _rows.append(
            {
                "endpoint": _e,
                "set": "blind vs 80%",
                **{f"p{int(q * 100)}": round(float(np.quantile(BLIND_NN80[_e], q)), 3) for q in _q},
            }
        )
        _rows.append(
            {
                "endpoint": _e,
                "set": "blind vs 100%",
                **{
                    f"p{int(q * 100)}": round(float(np.quantile(BLIND_NN100[_e], q)), 3) for q in _q
                },
            }
        )
    nn_profile = pl.DataFrame(_rows)
    return BLIND, BLIND_NN100, BLIND_NN80, OOF_NN, nn_profile


@app.cell
def _(OUT_DIR, mo, nn_profile):
    nn_profile.write_csv(OUT_DIR / "nn_similarity_profile.csv")
    mo.vstack([mo.md("**Nearest-neighbour similarity: OOF vs blind set**"), nn_profile])
    return


@app.cell
def _(ENDPOINTS, OOF_NN, PROJECT_ROOT, BLIND, np, pl):
    # Predicted-potency proxy for label shift. The blind side is 05's submitted `pubchem`
    # predictions (unplaced, so comparable to the OOF `pubchem`); only compounds with a
    # prediction matter, which is all of them because the model predicts every endpoint.
    _sub = pl.read_csv(
        next((PROJECT_ROOT / "submissions" / "05_auxiliary_data").glob("*/activity.csv"))
    )
    BLIND_PRED = {_e: _sub[_e].drop_nulls().to_numpy() for _e in ENDPOINTS}
    _rows = []
    for _e in ENDPOINTS:
        _o = OOF_NN.filter(pl.col("endpoint") == _e)["pubchem"].to_numpy()
        _b = BLIND_PRED[_e]
        _rows.append(
            {
                "endpoint": _e,
                "oof_mean": round(float(_o.mean()), 3),
                "blind_mean": round(float(_b.mean()), 3),
                "oof_sd": round(float(_o.std()), 3),
                "blind_sd": round(float(_b.std()), 3),
                "oof_share_ge5": round(float((_o >= 5).mean()), 3),
                "blind_share_ge5": round(float((_b >= 5).mean()), 3),
            }
        )
    pred_profile = pl.DataFrame(_rows)
    pred_profile
    return BLIND_PRED, pred_profile


@app.cell
def _(BLIND_NN80, BLIND_PRED, ENDPOINTS, OOF_NN, OUT_DIR, mo, np, pl):
    # Importance weights. Ten bins from the pooled quantiles; w = blind share / OOF share.
    # Row weights within an endpoint are normalised to sum to the row count so ESS reads as
    # a fraction of the rows.
    def _weights(oof_vals, blind_vals, n_bins=10):
        _edges = np.quantile(np.concatenate([oof_vals, blind_vals]), np.linspace(0, 1, n_bins + 1))
        _edges[0], _edges[-1] = -np.inf, np.inf
        _oi = np.clip(np.digitize(oof_vals, _edges[1:-1]), 0, n_bins - 1)
        _bi = np.clip(np.digitize(blind_vals, _edges[1:-1]), 0, n_bins - 1)
        _po = np.bincount(_oi, minlength=n_bins) / len(oof_vals)
        _pb = np.bincount(_bi, minlength=n_bins) / len(blind_vals)
        _ratio = np.where(_po > 0, _pb / np.maximum(_po, 1e-12), 0.0)
        return _ratio[_oi]

    _parts = []
    for _e in ENDPOINTS:
        _s = OOF_NN.filter(pl.col("endpoint") == _e)
        _wn = _weights(_s["nn_sim"].to_numpy(), BLIND_NN80[_e])
        _wp = _weights(_s["pubchem"].to_numpy(), BLIND_PRED[_e])
        _parts.append(
            _s.with_columns(
                pl.Series("w_novelty", _wn),
                pl.Series("w_potency", _wp),
                pl.Series("w_both", _wn * _wp),
            )
        )
    W = pl.concat(_parts)

    def _ess(w):
        return float(w.sum() ** 2 / (w**2).sum() / len(w))

    ess = pl.DataFrame(
        [
            {
                "endpoint": _e,
                "scheme": _k,
                "ess_fraction": round(
                    _ess(W.filter(pl.col("endpoint") == _e)[f"w_{_k}"].to_numpy()), 3
                ),
            }
            for _e in ENDPOINTS
            for _k in ("novelty", "potency", "both")
        ]
    ).pivot(on="scheme", index="endpoint", values="ess_fraction")
    ess.write_csv(OUT_DIR / "effective_sample_size.csv")
    mo.vstack(
        [
            mo.md(
                "**Effective sample size as a fraction of OOF rows** (1.0 = no reweighting cost)"
            ),
            ess,
        ]
    )
    return (W,)


@app.cell
def _(ENDPOINTS, OUT_DIR, W, mo, np, pl, st_rae):
    _PAIRS = [
        ("uniform_A", "pubchem", "Monroe+pubchem blend vs pubchem"),
        ("caruana_B", "caruana_A", "B vs A"),
        ("caruana_C", "caruana_A", "C vs A"),
        ("caruana_C", "caruana_B", "C vs B"),
    ]
    _METHODS = ["pubchem", "uniform_A", "caruana_A", "caruana_B", "caruana_C"]
    _repeats = sorted(W["outer_fold"].unique().to_list())

    _pool = {}
    for _e in ENDPOINTS:
        for _r in _repeats:
            _s = W.filter((pl.col("endpoint") == _e) & (pl.col("outer_fold") == _r))
            _pool[(_e, _r)] = {
                "y": _s["y_true"].to_numpy(),
                "lo": _s["y_lower"].to_numpy(),
                "hi": _s["y_upper"].to_numpy(),
                "p": {m: _s[m].to_numpy() for m in _METHODS},
                "w": {k: _s[f"w_{k}"].to_numpy() for k in ("novelty", "potency", "both")},
            }

    _rows = []
    for _scheme in ("none", "novelty", "potency", "both"):
        for _n in (150, 300, 600):
            _rng = np.random.default_rng(0)
            _macro_d = {_lab: [] for _, _, _lab in _PAIRS}
            _d2006 = {_lab: [] for _, _, _lab in _PAIRS}
            for _ in range(2000):
                _r = _repeats[_rng.integers(len(_repeats))]
                _m = {_k: 0.0 for _k in _METHODS}
                _d26 = {_k: 0.0 for _k in _METHODS}
                for _e in ENDPOINTS:
                    _p = _pool[(_e, _r)]
                    _w = np.ones(len(_p["y"])) if _scheme == "none" else _p["w"][_scheme]
                    _idx = _rng.choice(len(_w), size=_n, replace=True, p=_w / _w.sum())
                    for _k in _METHODS:
                        _s = st_rae(
                            _p["y"][_idx], _p["p"][_k][_idx], _p["lo"][_idx], _p["hi"][_idx]
                        )
                        _m[_k] += _s / len(ENDPOINTS)
                        if "CYP2D6" in _e:
                            _d26[_k] = _s
                for _a, _b, _lab in _PAIRS:
                    _macro_d[_lab].append(_m[_a] - _m[_b])
                    _d2006[_lab].append(_d26[_a] - _d26[_b])
            for _a, _b, _lab in _PAIRS:
                _x, _y = np.array(_macro_d[_lab]), np.array(_d2006[_lab])
                _rows.append(
                    {
                        "scheme": _scheme,
                        "n_per_endpoint": _n,
                        "comparison": _lab,
                        "macro_mean_diff": round(float(_x.mean()), 4),
                        "macro_p_first_better": round(float((_x < 0).mean()), 3),
                        "macro_p5": round(float(np.percentile(_x, 5)), 4),
                        "macro_p95": round(float(np.percentile(_x, 95)), 4),
                        "cyp2d6_mean_diff": round(float(_y.mean()), 4),
                        "cyp2d6_p_first_better": round(float((_y < 0).mean()), 3),
                    }
                )
    shift_ranking = pl.DataFrame(_rows)
    shift_ranking.write_csv(OUT_DIR / "ranking_under_shift.csv")
    mo.vstack(
        [mo.md("**Ranking under reweighting.** `none` reproduces 17's resampling."), shift_ranking]
    )
    return (shift_ranking,)


@app.cell
def _(ENDPOINTS, OUT_DIR, W, BLIND_NN80, evaluation, mo, np, pl):
    # Unweighted cross-check: score only the rows less novel-than-blind-median... i.e. the
    # rows at or below the blind set's 33rd-percentile similarity, per endpoint, so the
    # slice is defined by the blind set and not by the OOF. Paired bootstrap per endpoint.
    _pairs = [
        ("uniform_A", "pubchem"),
        ("caruana_B", "caruana_A"),
        ("caruana_C", "caruana_A"),
        ("caruana_C", "caruana_B"),
    ]
    _rows = []
    for _e in ENDPOINTS:
        _thr = float(np.quantile(BLIND_NN80[_e], 0.33))
        _s = W.filter((pl.col("endpoint") == _e) & (pl.col("nn_sim") <= _thr))
        for _a, _b in _pairs:
            _r = evaluation.paired_bootstrap(
                _s["y_true"].to_numpy(),
                _s[_a].to_numpy(),
                _s[_b].to_numpy(),
                metric="st_rae",
                y_lower=_s["y_lower"].to_numpy(),
                y_upper=_s["y_upper"].to_numpy(),
            )
            _rows.append(
                {
                    "endpoint": _e,
                    "slice": f"nn_sim <= {_thr:.3f}",
                    "n": _s.height,
                    "comparison": f"{_a} - {_b}",
                    **{k: round(v, 4) for k, v in _r.items()},
                }
            )
    low_slice = pl.DataFrame(_rows)
    _h = evaluation.holm_bonferroni(
        {f"{r['comparison']}|{r['endpoint']}": r["p_value"] for r in _rows}
    )
    low_slice = (
        low_slice.with_columns((pl.col("comparison") + "|" + pl.col("endpoint")).alias("c"))
        .join(_h.select(pl.col("comparison").alias("c"), "significant"), on="c")
        .drop("c")
    )
    low_slice.write_csv(OUT_DIR / "low_similarity_slice.csv")
    mo.vstack(
        [mo.md("**Lowest-similarity third (defined by the blind set), unweighted**"), low_slice]
    )
    return


if __name__ == "__main__":
    app.run()
