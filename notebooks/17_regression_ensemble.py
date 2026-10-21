import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # 17 — Regression ensemble: Monroe, `pubchem`, and what else belongs in the blend

    `16` found TabICL on Monroe at macro ST-RAE 0.7020 and a plain 50/50 blend with `05`'s
    `pubchem` (0.7160) at **0.6660**, better on 25 of 25 folds. Two questions follow.

    1. **Is the Monroe gain real?** Monroe pretrains on PCBA, which contains the Veith
       CYP panel the challenge derives from. `scripts/audit_monroe_pcba_overlap.py` found
       225 of 4,905 training compounds (4.6%) and 1 of 750 blind compounds carrying a
       label there. The blind set is clean. For CV, the sensitivity test below drops the
       225 and checks the gain survives.
    2. **Should weaker or more correlated models join the blend?** `14` gave a rule of
       thumb that ensembles pay at r 0.30–0.62 and not at ~0.94. `16` already broke it:
       Monroe and `pubchem` correlate at 0.77–0.90 and the blend still won every fold.
       Correlation of *predictions* is the wrong quantity. The right one is correlation of
       *errors*, and this notebook measures what each of 20 candidates adds on top of the
       Monroe + `pubchem` blend, strong or weak.

    ## Design

    Every candidate is an existing OOF on the same shared folds and rows (alignment is
    asserted). Nothing is retrained. Ensemble weights are fitted per endpoint on four outer
    repeats and scored on the fifth, rotated, as in `14`, so every ensemble number is
    out-of-sample with respect to the weights. Pools compared:

    | pool | members |
    |:--|:--|
    | A | `pubchem`, `tabicl_monroe` |
    | B | A + `tabicl_atom_jepa`, `tabicl_chemeleon` (the frozen-encoder family) |
    | C | all 20 candidates |

    Each pool runs Caruana selection and a uniform average. The uniform average is the
    control that shows whether the *weights* or just the *averaging* is doing the work.
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

    from cyp import constants as C
    from cyp import ensemble, evaluation, mcs
    from cyp.metrics import st_rae

    OUT_DIR = PROJECT_ROOT / "experiments" / Path(__file__).stem
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return C, OUT_DIR, PROJECT_ROOT, ensemble, evaluation, mcs, mo, np, pl, st_rae


@app.cell
def _(C, PROJECT_ROOT, pl):
    KEYS = ["endpoint", "fold", "Molecule_Name"]
    _exp = PROJECT_ROOT / "experiments"

    # Candidate name -> (file, method inside it). `tabicl_singletask` in 03 is the same
    # fit as 16's `tabicl_chemeleon` control, so it is left out rather than counted twice.
    # 05's `chemprop_multitask` control is a separate retrain of 03's; both stay, named by
    # source, because their disagreement is real run-to-run variance.
    _c03 = _exp / "03_methods" / "cache"
    _src = {
        "chemprop_multitask_03": (
            _c03 / "mt_chemprop_multitask_full.parquet",
            "chemprop_multitask",
        ),
        "chemprop_singletask": (
            _c03 / "mt_chemprop_singletask_full.parquet",
            "chemprop_singletask",
        ),
        "lgbm_multitask": (_c03 / "mt_lgbm_multitask_full.parquet", "lgbm_multitask"),
        "lgbm_singletask": (_c03 / "mt_lgbm_singletask_full.parquet", "lgbm_singletask"),
        "xgb_multitask": (_c03 / "mt_xgb_multitask_full.parquet", "xgb_multitask"),
        "xgb_singletask": (_c03 / "mt_xgb_singletask_full.parquet", "xgb_singletask"),
        "macau_multitask": (_c03 / "mt_macau_multitask_full.parquet", "macau_multitask"),
        "macau_singletask": (_c03 / "mt_macau_singletask_full.parquet", "macau_singletask"),
        "tabpfn_multitask": (_c03 / "mt_tabpfn_multitask_full.parquet", "tabpfn_multitask"),
        "tabpfn_singletask": (_c03 / "mt_tabpfn_singletask_full.parquet", "tabpfn_singletask"),
        "tabicl_multitask": (_c03 / "mt_tabicl_multitask_full.parquet", "tabicl_multitask"),
        "pubchem": (_exp / "05_auxiliary_data" / "oof.parquet", "pubchem"),
        "pubchem_frozen": (_exp / "05_auxiliary_data" / "oof.parquet", "pubchem_frozen"),
        "aux_screen": (_exp / "05_auxiliary_data" / "oof.parquet", "aux_screen"),
        "augmented": (_exp / "05_auxiliary_data" / "oof.parquet", "augmented"),
        "selection": (_exp / "05_auxiliary_data" / "oof.parquet", "selection"),
        "chemprop_multitask_05": (_exp / "05_auxiliary_data" / "oof.parquet", "chemprop_multitask"),
        "tabicl_chemeleon": (_exp / "16_encoders" / "oof.parquet", "tabicl_chemeleon"),
        "tabicl_monroe": (_exp / "16_encoders" / "oof.parquet", "tabicl_monroe"),
        "tabicl_atom_jepa": (_exp / "16_encoders" / "oof.parquet", "tabicl_atom_jepa"),
    }
    CANDIDATES = list(_src)

    # outer_fold is what the weight-fitting rotation holds out. Only 03/05 carry it.
    _truth = (
        pl.read_parquet(_exp / "05_auxiliary_data" / "oof.parquet")
        .filter(pl.col("method") == "pubchem")
        .select(*KEYS, "outer_fold", "y_true", "y_lower", "y_upper")
        .sort(KEYS)
    )
    _wide = _truth
    _cache = {}
    for _name, (_file, _method) in _src.items():
        if _file not in _cache:
            _cache[_file] = pl.read_parquet(_file)
        _s = _cache[_file].filter(pl.col("method") == _method).sort(KEYS)
        assert _s.height == _truth.height, f"{_name}: {_s.height} rows, expected {_truth.height}"
        assert (_s.select(KEYS) == _truth.select(KEYS)).to_numpy().all(), (
            f"{_name}: rows misaligned"
        )
        assert float((_s["y_true"] - _truth["y_true"]).abs().max()) < 1e-9, (
            f"{_name}: y_true differs"
        )
        _wide = _wide.with_columns(_s["y_pred"].alias(_name))

    WIDE = _wide
    ENDPOINTS = list(C.REGRESSION_ENDPOINTS)
    WIDE.select("endpoint").group_by("endpoint").len().sort("endpoint")
    return CANDIDATES, ENDPOINTS, KEYS, WIDE


@app.cell
def _(CANDIDATES, ENDPOINTS, KEYS, OUT_DIR, WIDE, evaluation, pl):
    def long_frame(names, columns=None):
        """Long OOF frame for `evaluation.fold_metrics`, one method per column."""
        cols = columns or names
        return pl.concat(
            [
                WIDE.select(
                    *KEYS, "y_true", "y_lower", "y_upper", pl.col(c).alias("y_pred")
                ).with_columns(pl.lit(n).alias("method"))
                for n, c in zip(names, cols, strict=True)
            ]
        )

    def macro_by_fold(long):
        """Per-fold macro ST-RAE, one row per (method, fold)."""
        folds = evaluation.fold_metrics(long)
        return evaluation.macro_averaged_fold_metrics(
            {e: folds.filter(pl.col("endpoint") == e) for e in ENDPOINTS}, metric_col="st_rae"
        )

    single = macro_by_fold(long_frame(CANDIDATES))
    single_table = (
        single.group_by("method")
        .agg(
            pl.col("st_rae").mean().round(4).alias("macro_st_rae"),
            pl.col("st_rae").std().round(4).alias("std"),
        )
        .sort("macro_st_rae")
    )
    single_table.write_csv(OUT_DIR / "single_model_macro.csv")
    single_table
    return long_frame, macro_by_fold, single_table


@app.cell
def _(CANDIDATES, ENDPOINTS, OUT_DIR, WIDE, mo, np, pl):
    # Correlation of ERRORS, not predictions. A blend helps when members are wrong in
    # different directions on the same compound; two models can predict almost the same
    # values (high prediction r) and still disagree in the residual that matters.
    def _mean_r(frame_cols, residual):
        _m = np.zeros((len(frame_cols), len(frame_cols)))
        for _e in ENDPOINTS:
            _s = WIDE.filter(pl.col("endpoint") == _e)
            _y = _s["y_true"].to_numpy()
            _x = np.vstack([(_s[c].to_numpy() - (_y if residual else 0.0)) for c in frame_cols])
            _m += np.corrcoef(_x) / len(ENDPOINTS)
        return _m

    _err = _mean_r(CANDIDATES, residual=True)
    _prd = _mean_r(CANDIDATES, residual=False)
    error_corr = pl.DataFrame(
        {"model": CANDIDATES, **{c: _err[:, i].round(3) for i, c in enumerate(CANDIDATES)}}
    )
    error_corr.write_csv(OUT_DIR / "error_correlation.csv")
    pred_corr = pl.DataFrame(
        {"model": CANDIDATES, **{c: _prd[:, i].round(3) for i, c in enumerate(CANDIDATES)}}
    )
    pred_corr.write_csv(OUT_DIR / "prediction_correlation.csv")

    _i, _j = CANDIDATES.index("pubchem"), CANDIDATES.index("tabicl_monroe")
    mo.md(
        f"Mean over endpoints. `pubchem` vs `tabicl_monroe`: prediction r **{_prd[_i, _j]:.3f}**, "
        f"error r **{_err[_i, _j]:.3f}**. Full matrices in `error_correlation.csv` and "
        "`prediction_correlation.csv`."
    )
    return


@app.cell
def _(CANDIDATES, OUT_DIR, WIDE, long_frame, macro_by_fold, mo, np, pl):
    # Marginal value of each extra candidate on top of the Monroe + pubchem blend.
    # Uniform 3-way average so no weight is fitted: this is a screen, not a result. The
    # nested Caruana run below is what gets believed.
    BASE = ["pubchem", "tabicl_monroe"]
    _base_pred = WIDE.select(pl.mean_horizontal(BASE).alias("p"))["p"]
    _base_err = (_base_pred - WIDE["y_true"]).to_numpy()

    _w = WIDE.with_columns(_base_pred.alias("blend_base"))

    def _macro(cols):
        _frame = _w.with_columns(pl.mean_horizontal(cols).alias("_b"))
        _long = pl.concat(
            [
                _frame.select(
                    "endpoint",
                    "fold",
                    "Molecule_Name",
                    "y_true",
                    "y_lower",
                    "y_upper",
                    pl.col("_b").alias("y_pred"),
                ).with_columns(pl.lit("x").alias("method"))
            ]
        )
        return float(macro_by_fold(_long)["st_rae"].mean())

    _base_macro = _macro(BASE)
    _rows = []
    for _c in CANDIDATES:
        if _c in BASE:
            continue
        _own = float(macro_by_fold(long_frame([_c]))["st_rae"].mean())
        _rows.append(
            {
                "candidate": _c,
                "standalone_macro": round(_own, 4),
                "error_r_with_base_blend": round(
                    float(np.corrcoef((_w[_c] - _w["y_true"]).to_numpy(), _base_err)[0, 1]), 3
                ),
                "three_way_macro": round(_macro([*BASE, _c]), 4),
                "delta_vs_base_blend": round(_macro([*BASE, _c]) - _base_macro, 4),
            }
        )
    marginal = pl.DataFrame(_rows).sort("delta_vs_base_blend")
    marginal.write_csv(OUT_DIR / "marginal_value.csv")
    mo.vstack(
        [
            mo.md(
                f"**Adding one candidate to the Monroe + `pubchem` uniform blend** (base macro "
                f"{_base_macro:.4f}). Negative delta helps. Sorted best first."
            ),
            marginal,
        ]
    )
    return BASE, marginal


@app.cell
def _(CANDIDATES, ENDPOINTS, OUT_DIR, WIDE, ensemble, mo, np, pl, st_rae):
    POOLS = {
        "A": ["pubchem", "tabicl_monroe"],
        "B": ["pubchem", "tabicl_monroe", "tabicl_atom_jepa", "tabicl_chemeleon"],
        "C": list(CANDIDATES),
    }
    # Bag settings: with a library of 2-4 members a half-size bag would be one model, which
    # makes "selection" degenerate, so small pools see every member. The 20-member pool uses
    # the standard half-library bags.
    _BAG = {"A": (1, 1.0), "B": (1, 1.0), "C": (20, 0.5)}

    _outer = sorted(WIDE["outer_fold"].unique().to_list())
    _parts = {f"caruana_{p}": [] for p in POOLS} | {f"uniform_{p}": [] for p in POOLS}
    _weights = []

    with mo.status.progress_bar(
        total=len(POOLS) * len(ENDPOINTS) * len(_outer), title="Caruana"
    ) as _bar:
        for _p, _members in POOLS.items():
            _n_bags, _frac = _BAG[_p]
            for _e in ENDPOINTS:
                _ep = WIDE.filter(pl.col("endpoint") == _e)
                for _held in _outer:
                    _fit = _ep.filter(pl.col("outer_fold") != _held)
                    _test = _ep.filter(pl.col("outer_fold") == _held)
                    _lo, _hi = _fit["y_lower"].to_numpy(), _fit["y_upper"].to_numpy()

                    def _score(_y, _pred, _lo=_lo, _hi=_hi):
                        return st_rae(_y, _pred, _lo, _hi)

                    _w = ensemble.caruana_selection(
                        {m: _fit[m].to_numpy() for m in _members},
                        _fit["y_true"].to_numpy(),
                        _score,
                        maximize=False,
                        n_bags=_n_bags,
                        bag_frac=_frac,
                        max_iter=50,
                    )
                    _weights.extend(
                        {"pool": _p, "endpoint": _e, "held_out": _held, "model": m, "weight": v}
                        for m, v in _w.items()
                    )
                    _keep = _test.select(
                        "endpoint", "fold", "Molecule_Name", "y_true", "y_lower", "y_upper"
                    )
                    _parts[f"caruana_{_p}"].append(
                        _keep.with_columns(
                            pl.Series("y_pred", sum(_w[m] * _test[m].to_numpy() for m in _members))
                        )
                    )
                    _parts[f"uniform_{_p}"].append(
                        _keep.with_columns(
                            pl.Series(
                                "y_pred", np.mean([_test[m].to_numpy() for m in _members], axis=0)
                            )
                        )
                    )
                    _bar.update()

    ENS_LONG = pl.concat(
        [pl.concat(v).with_columns(pl.lit(k).alias("method")) for k, v in _parts.items()]
    )
    weights = pl.DataFrame(_weights)
    weights.write_csv(OUT_DIR / "caruana_weights_by_rotation.csv")
    ENS_LONG.write_parquet(OUT_DIR / "ensemble_oof.parquet")
    return ENS_LONG, POOLS, weights


@app.cell
def _(ENDPOINTS, ENS_LONG, OUT_DIR, WIDE, evaluation, long_frame, macro_by_fold, mo, np, pl):
    # Ensembles vs the three references that matter: the shipped model (`pubchem`), the
    # new single model (`tabicl_monroe`), and the plain 50/50 blend from 16.
    _refs = long_frame(["pubchem", "tabicl_monroe"])
    ALL_MACRO = macro_by_fold(pl.concat([ENS_LONG, _refs], how="vertical_relaxed"))
    macro_summary = (
        ALL_MACRO.group_by("method")
        .agg(
            pl.col("st_rae").mean().round(4).alias("macro_st_rae"),
            pl.col("st_rae").std().round(4).alias("std"),
        )
        .sort("macro_st_rae")
    )
    macro_summary.write_csv(OUT_DIR / "ensemble_macro.csv")

    _wide = ALL_MACRO.pivot(on="method", index="fold", values="st_rae")

    def _paired(a, b):
        _d = (_wide[a] - _wide[b]).to_numpy()
        _rng = np.random.default_rng(42)
        _bs = np.array([_rng.choice(_d, len(_d)).mean() for _ in range(10_000)])
        return {
            "a": a,
            "b": b,
            "diff": round(float(_d.mean()), 4),
            "ci_low": round(float(np.percentile(_bs, 2.5)), 4),
            "ci_high": round(float(np.percentile(_bs, 97.5)), 4),
            "folds_a_better": int((_d < 0).sum()),
        }

    _pairs = []
    for _a in ("caruana_A", "caruana_B", "caruana_C", "uniform_A", "uniform_B", "uniform_C"):
        _pairs.append(_paired(_a, "pubchem"))
    for _a in ("caruana_A", "caruana_B", "caruana_C", "uniform_B", "uniform_C"):
        _pairs.append(_paired(_a, "uniform_A"))
    for _a in ("caruana_B", "caruana_C"):
        _pairs.append(_paired(_a, "caruana_A"))
    paired_macro = pl.DataFrame(_pairs)
    paired_macro.write_csv(OUT_DIR / "ensemble_paired_macro.csv")

    # Per endpoint vs pubchem, bootstrap over compounds, Holm over the family.
    _rows = []
    _ref = _refs.filter(pl.col("method") == "pubchem").select(
        "endpoint", "fold", "Molecule_Name", pl.col("y_pred").alias("pp")
    )
    for _m in ("caruana_A", "caruana_B", "caruana_C"):
        for _e in ENDPOINTS:
            _a = ENS_LONG.filter((pl.col("method") == _m) & (pl.col("endpoint") == _e)).join(
                _ref.filter(pl.col("endpoint") == _e), on=["endpoint", "fold", "Molecule_Name"]
            )
            _r = evaluation.paired_bootstrap(
                _a["y_true"].to_numpy(),
                _a["y_pred"].to_numpy(),
                _a["pp"].to_numpy(),
                metric="st_rae",
                y_lower=_a["y_lower"].to_numpy(),
                y_upper=_a["y_upper"].to_numpy(),
            )
            _rows.append({"method": _m, "endpoint": _e, **{k: round(v, 4) for k, v in _r.items()}})
    per_endpoint = pl.DataFrame(_rows)
    _h = evaluation.holm_bonferroni({f"{r['method']}|{r['endpoint']}": r["p_value"] for r in _rows})
    per_endpoint = (
        per_endpoint.with_columns((pl.col("method") + "|" + pl.col("endpoint")).alias("c"))
        .join(_h.select(pl.col("comparison").alias("c"), "significant"), on="c")
        .drop("c")
    )
    per_endpoint.write_csv(OUT_DIR / "ensemble_vs_pubchem_per_endpoint.csv")

    mo.vstack(
        [
            macro_summary,
            mo.md("**Paired over folds (macro). Negative `diff` favours `a`.**"),
            paired_macro,
            per_endpoint,
        ]
    )
    return ALL_MACRO, macro_summary, paired_macro, per_endpoint


@app.cell
def _(OUT_DIR, mo, pl, weights):
    # Weight stability across the five rotations. A weight that swings between rotations
    # is fitting fold noise (14's test of whether an ensemble is shippable).
    weight_stability = (
        weights.group_by("pool", "endpoint", "model")
        .agg(
            pl.col("weight").mean().round(3).alias("mean"),
            pl.col("weight").min().round(3).alias("min"),
            pl.col("weight").max().round(3).alias("max"),
        )
        .filter(pl.col("mean") > 0)
        .sort("pool", "endpoint", "mean", descending=[False, False, True])
    )
    weight_stability.write_csv(OUT_DIR / "caruana_weight_stability.csv")
    mo.vstack(
        [
            mo.md("**Caruana weights, mean and range over the 5 held-out rotations**"),
            weight_stability,
        ]
    )
    return


@app.cell
def _(ENS_LONG, OUT_DIR, PROJECT_ROOT, long_frame, macro_by_fold, mo, np, pl):
    # R1 sensitivity. Drop every training compound that carries a PCBA CYP label in Monroe's
    # pretraining corpus and recompute. If Monroe's lead is label exposure, it shrinks here.
    _overlap = set(
        pl.read_csv(PROJECT_ROOT / "experiments" / "16_encoders" / "monroe_pcba_overlap.csv")[
            "Molecule_Name"
        ]
    )
    _long = pl.concat(
        [
            long_frame(["pubchem", "tabicl_monroe", "tabicl_chemeleon"]),
            ENS_LONG.filter(
                pl.col("method").is_in(["uniform_A", "caruana_A", "caruana_B", "caruana_C"])
            ),
        ],
        how="vertical_relaxed",
    )
    _clean = _long.filter(~pl.col("Molecule_Name").is_in(_overlap))
    _rows = []
    for _tag, _frame in (("all compounds", _long), ("overlap removed", _clean)):
        _m = macro_by_fold(_frame)
        _w = _m.pivot(on="method", index="fold", values="st_rae")
        for _name in _w.columns[1:]:
            _rows.append(
                {"set": _tag, "method": _name, "macro_st_rae": round(float(_w[_name].mean()), 4)}
            )
    sens = pl.DataFrame(_rows).pivot(on="set", index="method", values="macro_st_rae")
    sens = sens.with_columns(
        (pl.col("overlap removed") - pl.col("all compounds")).round(4).alias("shift")
    )
    sens.write_csv(OUT_DIR / "r1_overlap_sensitivity.csv")

    # Does Monroe's lead over CheMeleon and pubchem survive?
    _m = macro_by_fold(_clean).pivot(on="method", index="fold", values="st_rae")
    _lines = []
    for _other in ("tabicl_chemeleon", "pubchem"):
        _d = (_m["tabicl_monroe"] - _m[_other]).to_numpy()
        _rng = np.random.default_rng(42)
        _bs = np.array([_rng.choice(_d, len(_d)).mean() for _ in range(10_000)])
        _lines.append(
            f"- Monroe − {_other} on {_clean['Molecule_Name'].n_unique():,}-compound clean set: "
            f"**{_d.mean():+.4f}**, 95% CI "
            f"[{np.percentile(_bs, 2.5):+.4f}, {np.percentile(_bs, 97.5):+.4f}]"
        )
    mo.vstack([mo.md("\n".join(_lines)), sens])
    return


@app.cell
def _(ALL_MACRO, ENDPOINTS, ENS_LONG, OUT_DIR, WIDE, evaluation, mo, np, pl):
    # Pool-vs-pool: is the bigger pool distinguishable from the smaller one? One family of
    # three comparisons (A-B, A-C, B-C), Holm-corrected, tested two ways:
    #   - per endpoint, bootstrap over compounds (the repo's usual paired_bootstrap)
    #   - macro, over the 25 folds, and again over the 5 outer-repeat means. The 25 folds
    #     share training data across repeats, so fold-level intervals are optimistic; the
    #     5-repeat view is the conservative one (n=5, so power is low by construction).
    from scipy import stats as _st

    _fold_to_outer = WIDE.select("fold", "outer_fold").unique().sort("fold")
    _w = ALL_MACRO.pivot(on="method", index="fold", values="st_rae").join(_fold_to_outer, on="fold")
    _pairs = [("caruana_B", "caruana_A"), ("caruana_C", "caruana_A"), ("caruana_C", "caruana_B")]

    _rows = []
    for _a, _b in _pairs:
        _d = (_w[_a] - _w[_b]).to_numpy()
        _rng = np.random.default_rng(42)
        _bs = np.array([_rng.choice(_d, len(_d)).mean() for _ in range(10_000)])
        _rep = (
            _w.group_by("outer_fold")
            .agg((pl.col(_a) - pl.col(_b)).mean().alias("d"))
            .sort("outer_fold")["d"]
            .to_numpy()
        )
        _t = _rep.mean() / (_rep.std(ddof=1) / np.sqrt(len(_rep)))
        _rows.append(
            {
                "comparison": f"{_a} - {_b}",
                "macro_diff": round(float(_d.mean()), 4),
                "fold_ci_low": round(float(np.percentile(_bs, 2.5)), 4),
                "fold_ci_high": round(float(np.percentile(_bs, 97.5)), 4),
                "folds_better": int((_d < 0).sum()),
                "repeat_diffs": " ".join(f"{x:+.4f}" for x in _rep),
                "repeats_better": int((_rep < 0).sum()),
                "repeat_t_p": round(float(2 * _st.t.sf(abs(_t), df=len(_rep) - 1)), 4),
            }
        )
    pool_macro = pl.DataFrame(_rows)
    _holm = evaluation.holm_bonferroni({r["comparison"]: r["repeat_t_p"] for r in _rows})
    pool_macro = pool_macro.join(
        _holm.select("comparison", pl.col("significant").alias("repeat_holm_significant")),
        on="comparison",
    )
    pool_macro.write_csv(OUT_DIR / "pool_vs_pool_macro.csv")

    _rows = []
    for _a, _b in _pairs:
        for _e in ENDPOINTS:
            _x = ENS_LONG.filter((pl.col("method") == _a) & (pl.col("endpoint") == _e)).join(
                ENS_LONG.filter((pl.col("method") == _b) & (pl.col("endpoint") == _e)).select(
                    "endpoint", "fold", "Molecule_Name", pl.col("y_pred").alias("pb")
                ),
                on=["endpoint", "fold", "Molecule_Name"],
            )
            _r = evaluation.paired_bootstrap(
                _x["y_true"].to_numpy(),
                _x["y_pred"].to_numpy(),
                _x["pb"].to_numpy(),
                metric="st_rae",
                y_lower=_x["y_lower"].to_numpy(),
                y_upper=_x["y_upper"].to_numpy(),
            )
            _rows.append(
                {
                    "comparison": f"{_a} - {_b}",
                    "endpoint": _e,
                    **{k: round(v, 4) for k, v in _r.items()},
                }
            )
    pool_ep = pl.DataFrame(_rows)
    _h = evaluation.holm_bonferroni(
        {f"{r['comparison']}|{r['endpoint']}": r["p_value"] for r in _rows}
    )
    pool_ep = (
        pool_ep.with_columns((pl.col("comparison") + "|" + pl.col("endpoint")).alias("c"))
        .join(_h.select(pl.col("comparison").alias("c"), "significant"), on="c")
        .drop("c")
    )
    pool_ep.write_csv(OUT_DIR / "pool_vs_pool_per_endpoint.csv")
    mo.vstack(
        [
            mo.md("**Caruana pools against each other (negative favours the first)**"),
            pool_macro,
            pool_ep,
        ]
    )
    return


@app.cell
def _(C, ENDPOINTS, ENS_LONG, KEYS, OUT_DIR, WIDE, mo, np, pl, st_rae):
    # PXR's lesson (delavega.ai, 2026-07-30): three finalists 0.0039 MAE apart in CV were
    # reordered on the unblinded set, because the CV difference was smaller than the noise
    # of a 260-compound test. CV intervals above ask "is the difference real in this data".
    # They do not ask the decision question: "on a test set the size of ours, how often does
    # the ranking hold?" This cell answers that.
    #
    # The blind set is ~75 hit series of ~10 analogs, split into two halves by series, so the
    # unit of independence is the series, not the compound. A compound-level bootstrap
    # therefore overstates the effective sample size. This resamples whole Murcko-scaffold
    # groups instead, K per endpoint, from one outer repeat at a time (each compound once).
    from cyp import cv, data

    _frames = {_e: data.training_frame(_e) for _e in C.REGRESSION_ENDPOINTS}
    _, _assign = cv.shared_scaffold_folds(_frames)
    _scaf = dict(
        zip(_assign["Molecule_Name"].to_list(), _assign["scaffold"].to_list(), strict=False)
    )

    _METHODS = ["pubchem", "tabicl_monroe", "caruana_A", "caruana_B", "caruana_C", "uniform_A"]
    _ens = ENS_LONG.filter(pl.col("method").is_in(_METHODS)).pivot(
        on="method", index=KEYS, values="y_pred"
    )
    _base = WIDE.select(
        *KEYS, "outer_fold", "y_true", "y_lower", "y_upper", "pubchem", "tabicl_monroe"
    )
    _tab = (
        _base.drop("pubchem", "tabicl_monroe")
        .join(_ens, on=KEYS, how="inner")
        .join(_base.select(*KEYS, "pubchem", "tabicl_monroe"), on=KEYS)
        if False
        else _base.join(
            _ens.select(*KEYS, "caruana_A", "caruana_B", "caruana_C", "uniform_A"), on=KEYS
        )
    )
    _tab = _tab.with_columns(
        pl.col("Molecule_Name").replace_strict(_scaf, default=None).alias("scaffold")
    )

    # Precompute, per (endpoint, outer repeat): arrays and scaffold -> row-index groups.
    _pools = {}
    for _e in ENDPOINTS:
        for _r in sorted(_tab["outer_fold"].unique().to_list()):
            _s = _tab.filter((pl.col("endpoint") == _e) & (pl.col("outer_fold") == _r))
            _codes = _s["scaffold"].fill_null("none").to_numpy()
            _order = np.argsort(_codes, kind="stable")
            _u, _start = np.unique(_codes[_order], return_index=True)
            _groups = np.split(_order, _start[1:])
            _pools[(_e, _r)] = {
                "y": _s["y_true"].to_numpy(),
                "lo": _s["y_lower"].to_numpy(),
                "hi": _s["y_upper"].to_numpy(),
                "p": {m: _s[m].to_numpy() for m in _METHODS},
                "groups": _groups,
            }
    _repeats = sorted(_tab["outer_fold"].unique().to_list())
    _sizes = np.array([len(g) for g in _pools[(ENDPOINTS[0], _repeats[0])]["groups"]])

    _PAIRS = [
        ("uniform_A", "pubchem", "Monroe+pubchem blend vs pubchem"),
        ("caruana_B", "caruana_A", "B vs A"),
        ("caruana_C", "caruana_A", "C vs A"),
        ("caruana_C", "caruana_B", "C vs B"),
    ]
    _rows = []
    for _K in (75, 150, 300, 600):
        _rng = np.random.default_rng(0)
        _diff = {lab: [] for _, _, lab in _PAIRS}
        _ncomp = []
        for _ in range(2000):
            _r = _repeats[_rng.integers(len(_repeats))]
            _macro = {m: 0.0 for m in _METHODS}
            _n = 0
            for _e in ENDPOINTS:
                _pl = _pools[(_e, _r)]
                _pick = _rng.integers(len(_pl["groups"]), size=_K)
                _idx = np.concatenate([_pl["groups"][g] for g in _pick])
                _n += len(_idx)
                for m in _METHODS:
                    _macro[m] += st_rae(
                        _pl["y"][_idx], _pl["p"][m][_idx], _pl["lo"][_idx], _pl["hi"][_idx]
                    ) / len(ENDPOINTS)
            _ncomp.append(_n / len(ENDPOINTS))
            for a, b, lab in _PAIRS:
                _diff[lab].append(_macro[a] - _macro[b])
        for _, _, lab in _PAIRS:
            _d = np.array(_diff[lab])
            _rows.append(
                {
                    "groups_per_endpoint": _K,
                    "compounds_per_endpoint": round(float(np.mean(_ncomp))),
                    "comparison": lab,
                    "mean_diff": round(float(_d.mean()), 4),
                    "sd_of_diff": round(float(_d.std()), 4),
                    "p_first_better": round(float((_d < 0).mean()), 3),
                    "p5": round(float(np.percentile(_d, 5)), 4),
                    "p95": round(float(np.percentile(_d, 95)), 4),
                }
            )
    ranking_stability = pl.DataFrame(_rows)
    ranking_stability.write_csv(OUT_DIR / "ranking_stability_at_test_size.csv")
    mo.vstack(
        [
            mo.md(
                "**How often does the ranking hold on a blind-set-sized sample of whole "
                "scaffold series?** `p_first_better` is the share of 2,000 draws where the "
                "first-named arm scores lower macro ST-RAE."
            ),
            ranking_stability,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
