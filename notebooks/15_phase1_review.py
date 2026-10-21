import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # 15 — Phase 1 review: the figures for the blog post

    Trains nothing. Every figure here is drawn from a cache or a CSV that an earlier
    notebook already wrote, plus one dated snapshot of the public leaderboards
    (`scripts/snapshot_leaderboard.py`). The point of putting them in a notebook is
    that each number in the post traces back to a file rather than to a pasted table.

    | figure | question | source |
    |:--|:--|:--|
    | `endpoint_scorecard` | why are the four endpoints not comparable? | `02` |
    | `representation_ladder` | did the representation or the model move the score? | `03`, `05` |
    | `multitask_paired` | does sharing across endpoints help? | `03` OOF |
    | `pretraining_arms` | does public data help, and in which form? | `05` OOF |
    | `cyp2d6_routes` | what did four routes to the CYP2D6 pharmacophore buy? | `09`, `10`, `11` |
    | `tdi_threshold_sweep` | where is MCC maximized on a ~21%-positive label? | `04` OOF |
    | `tdi_ensemble` | why did the ensemble work for TDI? | `13`, `14`, `07` |
    | `placement` | what does placement do to the same ordering? | `07`, `12`, board |
    | `interim_board` | where do we sit once the full test set is scored? | leaderboard snapshot |

    ## One decision worth stating: the unit of the paired comparison is the fold

    `03` and `05` ran `evaluation.paired_bootstrap` on the pooled 32,625 out-of-fold
    rows. That resamples *rows*, but every compound appears in five outer repeats, so
    the rows are not independent and the intervals come out narrower than the data
    supports. It also pools the four endpoints into one ST-RAE, whose denominator is
    then the error of a single constant fitted to all four, not the macro-average the
    leaderboard ranks on.

    The comparisons below instead take the difference in **macro-averaged ST-RAE on
    each of the 25 shared folds** and bootstrap those 25 differences. That is the same
    repeated-measures unit `mcs.paired_arm_plot` and the MCS heatmaps use, it matches
    the quantity the board ranks on, and it is still anti-conservative (folds from
    different repeats overlap), so read the intervals as a lower bound on the
    uncertainty. Holm-Bonferroni is applied within each family of comparisons.
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

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    # `cyp` first: it claims the OpenMP runtime slot for LightGBM before sklearn or
    # torch can (see CLAUDE.md, "OpenMP load order"). Nothing below needs a model,
    # but `evaluation` imports sklearn, so the order still matters.
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    from cyp import constants as C
    from cyp import evaluation

    return C, PROJECT_ROOT, Path, evaluation, np, pl, plt


@app.cell
def _(Path, PROJECT_ROOT):
    NOTEBOOK_NAME = Path(__file__).stem
    OUT_DIR = PROJECT_ROOT / "experiments" / NOTEBOOK_NAME
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    #: Pinned like a data snapshot: the board moves every day, and a figure that
    #: silently re-reads "latest" would stop matching the numbers quoted in the post.
    LEADERBOARD_SNAPSHOT = "20261005"
    #: Our username on the board, for picking our own row out of the snapshot.
    OURS = "adlvdl"
    return LEADERBOARD_SNAPSHOT, NOTEBOOK_NAME, OURS, OUT_DIR


@app.cell
def _(plt):
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 10.5,
            "axes.titleweight": "regular",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )

    #: The verdict colours match `mcs.reference_forest_plot`, so a green, grey or red
    #: dot means the same thing here as in every forest plot the repo has produced.
    COLOR = {
        "better": "#2ca02c",
        "worse": "#d62728",
        "similar": "#8c8c8c",
        "reference": "#1f77b4",
        "highlight": "#e07b00",
        "neutral": "#7f9bbd",
        "light": "#c7d3e3",
        "dark": "#1a4a78",
    }
    return (COLOR,)


@app.cell
def _(C, OUT_DIR, evaluation, np, pl):
    def save_fig(fig, name: str) -> None:
        """Write a figure the way every notebook here does: PNG, 300 dpi, tight."""
        fig.savefig(OUT_DIR / name, dpi=300, bbox_inches="tight")

    def short(endpoint: str) -> str:
        """'CYP2D6_pIC50_direct_inhibition' -> 'CYP2D6' (also fine on 'CYP2D6_is_TDI')."""
        return endpoint.split("_")[0]

    def macro_by_fold(oof: pl.DataFrame) -> pl.DataFrame:
        """Per-(method, fold) macro-averaged ST-RAE: the CV analogue of the board's MA row."""
        folds = evaluation.fold_metrics(oof)
        return evaluation.macro_averaged_fold_metrics(
            {e: folds.filter(pl.col("endpoint") == e) for e in C.REGRESSION_ENDPOINTS},
            metric_col="st_rae",
        )

    def fold_vector(macro: pl.DataFrame, method: str) -> np.ndarray:
        """One method's per-fold scores ordered by fold, so two methods pair row for row."""
        return macro.filter(pl.col("method") == method).sort("fold")["st_rae"].to_numpy()

    def macro_summary(macro: pl.DataFrame) -> pl.DataFrame:
        """Mean and spread of the per-fold macro score, one row per method."""
        return macro.group_by("method").agg(
            pl.col("st_rae").mean().alias("mean"),
            pl.col("st_rae").std().alias("sd"),
            pl.len().alias("n_folds"),
        )

    def paired_fold_bootstrap(
        a: np.ndarray, b: np.ndarray, n_resamples: int = 10_000, seed: int = 42
    ) -> dict[str, float]:
        """Bootstrap the mean of the per-fold difference a - b.

        The two arms were scored on identical folds, so the per-fold difference is the
        repeated measure. The p-value follows `evaluation.paired_bootstrap`: twice the
        smaller tail of the resampled differences. With 10,000 resamples the smallest
        value it can report is 0, which the post writes as p < 0.0001.
        """
        diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        rng = np.random.default_rng(seed)
        resampled = diff[rng.integers(0, len(diff), size=(n_resamples, len(diff)))].mean(axis=1)
        p = 2.0 * min((resampled <= 0).mean(), (resampled >= 0).mean())
        return {
            "diff": float(diff.mean()),
            "ci_low": float(np.percentile(resampled, 2.5)),
            "ci_high": float(np.percentile(resampled, 97.5)),
            "p_value": float(min(p, 1.0)),
            "n_folds": len(diff),
        }

    def with_verdicts(rows: pl.DataFrame, key: str) -> pl.DataFrame:
        """Add a Holm-corrected `holm_significant` flag over a family of comparisons.

        Every comparison in a figure is one family: the point of the correction is that
        six models tested for a multitask effect will throw up a p of 0.03 by chance.
        """
        holm = evaluation.holm_bonferroni(dict(zip(rows[key], rows["p_value"], strict=True)))
        return rows.join(
            holm.select(pl.col("comparison").alias(key), "threshold", "significant"),
            on=key,
            how="left",
        ).rename({"significant": "holm_significant", "threshold": "holm_threshold"})

    return (
        fold_vector,
        macro_by_fold,
        macro_summary,
        paired_fold_bootstrap,
        save_fig,
        short,
        with_verdicts,
    )


@app.cell
def _(LEADERBOARD_SNAPSHOT, PROJECT_ROOT, pl):
    LB = pl.read_csv(
        PROJECT_ROOT / "data" / "leaderboard" / LEADERBOARD_SNAPSHOT / "leaderboards.csv"
    )

    def board_wide(board: str, track: str, scope: str) -> pl.DataFrame:
        """One row per entry for one leaderboard table, one column per metric."""
        return (
            LB.filter(
                (pl.col("board") == board) & (pl.col("track") == track) & (pl.col("scope") == scope)
            )
            .pivot(on="metric", index=["entry", "rank", "is_ours"], values="value")
            .sort("entry")
        )

    def our_row(board: str, track: str, scope: str) -> dict:
        """Our own entry in one table, as a plain dict of metric -> value plus `rank`."""
        row = board_wide(board, track, scope).filter(pl.col("is_ours"))
        if row.height != 1:
            raise ValueError(f"expected exactly one row of ours in {board}/{track}/{scope}")
        return row.row(0, named=True)

    return LB, board_wide, our_row


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Figure 1 — The four endpoints are not the same problem

    `02`'s scorecard, drawn once. ST-RAE scores zero inside a compound's credible
    interval, so the same model quality scores differently depending on how wide the
    intervals are, how many compounds sit in the forgiven region below pIC50 4, and
    how much of the test set has a training neighbor to learn from.
    """
    )
    return


@app.cell
def _(COLOR, PROJECT_ROOT, pl, plt, save_fig):
    scorecard = pl.read_csv(
        PROJECT_ROOT / "experiments" / "02_chemical_space" / "endpoint_scorecard.csv"
    )
    _panels = [
        ("median_CI_width", "Median credible interval\nwidth (pIC50 units)", "{:.2f}"),
        ("below_floor_pct", "Labels below pIC50 4\n(% of labeled compounds)", "{:.1f}"),
        (
            "median_test_nn_sim",
            "Median similarity of a test\ncompound to its nearest\ntraining compound (ECFP4)",
            "{:.3f}",
        ),
        (
            "pct_extrapolation",
            "Test compounds with no\ntraining neighbor above\n0.3 similarity (%)",
            "{:.1f}",
        ),
    ]
    _fig, _axes = plt.subplots(1, 4, figsize=(11, 3.3))
    _labels = scorecard["endpoint"].to_list()
    for _ax, (_col, _title, _fmt) in zip(_axes, _panels, strict=True):
        _values = scorecard[_col].to_list()
        _colors = [COLOR["highlight"] if _e == "CYP2D6" else COLOR["neutral"] for _e in _labels]
        _bars = _ax.bar(range(len(_labels)), _values, color=_colors, width=0.72)
        for _bar, _v in zip(_bars, _values, strict=True):
            _ax.text(
                _bar.get_x() + _bar.get_width() / 2,
                _bar.get_height(),
                _fmt.format(_v),
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
        _ax.set_xticks(range(len(_labels)), [_l.replace("CYP", "") for _l in _labels])
        _ax.set_xlabel("CYP isoform")
        _ax.set_title(_title, fontsize=9.5)
        _ax.set_ylim(0, max(_values) * 1.18)
        _ax.tick_params(axis="y", labelsize=8.5)
    _fig.tight_layout()
    save_fig(_fig, "endpoint_scorecard.png")
    _fig
    return (scorecard,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Figure 2 — Representation moved the score; the model on top of it barely did

    Macro ST-RAE (mean over the 25 CV folds) for every representation and model
    combination `03` and `05` produced, plus the multitask and pretrained Chemprop
    that came out of them. Rows are representations; markers within a row are models.
    The question the layout answers is whether the spread *within* a row (model
    choice) is larger than the step *between* rows (representation).
    """
    )
    return


@app.cell
def _(C, OUT_DIR, PROJECT_ROOT, macro_by_fold, macro_summary, pl):
    _cache = PROJECT_ROOT / "experiments" / "03_methods" / "cache"

    FP_LABEL = {
        "ecfp": "ECFP4",
        "atompair": "Atom pair",
        "avalon": "Avalon",
        "maccs": "MACCS",
        "mqn": "MQN",
        "rdkit": "RDKit fingerprint",
        "torsion": "Topological torsion",
        "mordred": "Mordred descriptors",
    }
    MODEL_LABEL = {
        "lgbm": "LightGBM",
        "xgb": "XGBoost",
        "tabicl": "TabICL",
        "tabpfn": "TabPFN",
        "macau": "Macau",
        "chemprop": "Chemprop, single-task",
        "chemprop_multitask": "Chemprop, multitask",
        "pubchem": "Chemprop, multitask, PubChem-pretrained",
    }

    _rows = []

    # Fingerprint and descriptor sweep: one cache file per (fingerprint, endpoint),
    # per-endpoint scaffold folds. The `mean` baseline is left out; it is drawn as the
    # 1.0 reference line instead.
    for _fp, _label in FP_LABEL.items():
        _oof = pl.concat(
            [
                pl.read_parquet(_cache / f"fp_{_fp}_{_e}_full.parquet")
                for _e in C.REGRESSION_ENDPOINTS
            ],
            how="vertical_relaxed",
        ).filter(pl.col("method").is_in(["lgbm", "xgb"]))
        for _r in macro_summary(macro_by_fold(_oof)).iter_rows(named=True):
            _rows.append(
                {"representation": _label, "group": "fingerprint", "model": _r["method"], **_r}
            )

    # Frozen CheMeleon embedding: three models, same per-endpoint folds. Chemprop
    # trained from scratch on the graph is the learned-representation comparison.
    for _m in ["tabicl", "tabpfn", "macau", "chemprop"]:
        _oof = pl.concat(
            [
                pl.read_parquet(_cache / f"new_{_m}_{_e}_full.parquet")
                for _e in C.REGRESSION_ENDPOINTS
            ],
            how="vertical_relaxed",
        )
        _r = macro_summary(macro_by_fold(_oof)).row(0, named=True)
        _rep = "Learned graph (Chemprop)" if _m == "chemprop" else "CheMeleon embedding, frozen"
        _rows.append(
            {
                "representation": _rep,
                "group": "graph" if _m == "chemprop" else "frozen",
                "model": _m,
                **_r,
            }
        )

    # Shared-fold runs: multitask Chemprop (03) and the PubChem-pretrained one (05).
    _mt = pl.read_parquet(_cache / "mt_chemprop_multitask_full.parquet")
    _r = macro_summary(macro_by_fold(_mt)).row(0, named=True)
    _rows.append(
        {
            "representation": "Learned graph (Chemprop)",
            "group": "graph",
            "model": "chemprop_multitask",
            **_r,
        }
    )

    _o5 = pl.read_parquet(
        PROJECT_ROOT / "experiments" / "05_auxiliary_data" / "oof.parquet"
    ).filter(pl.col("method") == "pubchem")
    _r = macro_summary(macro_by_fold(_o5)).row(0, named=True)
    _rows.append(
        {
            "representation": "Learned graph, pretrained on PubChem",
            "group": "pretrained",
            "model": "pubchem",
            **_r,
        }
    )

    ladder = pl.DataFrame(_rows).select("group", "representation", "model", "mean", "sd", "n_folds")
    ladder.write_csv(OUT_DIR / "representation_ladder.csv")
    return FP_LABEL, MODEL_LABEL, ladder


@app.cell
def _(COLOR, FP_LABEL, MODEL_LABEL, ladder, pl, plt, save_fig):
    # Row order: fingerprints worst to best at the top, then each step up in
    # representation below them, so the figure reads as a ladder going down.
    _fp_order = (
        ladder.filter(pl.col("group") == "fingerprint")
        .group_by("representation")
        .agg(pl.col("mean").min().alias("best"))
        .sort("best", descending=True)["representation"]
        .to_list()
    )
    _order = [
        *_fp_order,
        "CheMeleon embedding, frozen",
        "Learned graph (Chemprop)",
        "Learned graph, pretrained on PubChem",
    ]
    _ypos = {_rep: _i for _i, _rep in enumerate(_order)}

    _style = {
        "lgbm": ("o", "#1f77b4"),
        "xgb": ("s", "#8c564b"),
        "tabicl": ("D", "#d62728"),
        "tabpfn": ("^", "#9467bd"),
        "macau": ("v", "#e377c2"),
        "chemprop": ("P", "#2ca02c"),
        "chemprop_multitask": ("P", "#17803d"),
        "pubchem": ("*", "#0b4f2a"),
    }

    _fig, _ax = plt.subplots(figsize=(9, 5.4))
    _seen: set[str] = set()
    for _r in ladder.iter_rows(named=True):
        _marker, _c = _style[_r["model"]]
        _label = MODEL_LABEL[_r["model"]] if _r["model"] not in _seen else None
        _seen.add(_r["model"])
        _y = _ypos[_r["representation"]]
        # Two models in one row would overprint at the same y, so nudge by model.
        _nudge = {"lgbm": -0.12, "xgb": 0.12, "tabicl": -0.2, "tabpfn": 0.0, "macau": 0.2}.get(
            _r["model"], 0.0
        )
        if _r["model"] in ("chemprop", "chemprop_multitask"):
            _nudge = -0.12 if _r["model"] == "chemprop" else 0.12
        _ax.errorbar(
            _r["mean"],
            _y + _nudge,
            xerr=_r["sd"],
            fmt=_marker,
            color=_c,
            ecolor=_c,
            elinewidth=0.9,
            capsize=0,
            markersize=9 if _marker == "*" else 7,
            alpha=0.95,
            label=_label,
        )

    _ax.axvline(1.0, color="#555555", linestyle=":", linewidth=1)
    # Bottom of the line, where nothing else is drawn: at the top it collides with the
    # MACCS and MQN rows, which sit almost on the 1.0 line.
    _ax.text(
        1.004, len(_order) - 1.0, "predicting\nthe mean", fontsize=8, color="#555555", va="center"
    )
    _ax.set_yticks(range(len(_order)), _order)
    _ax.invert_yaxis()
    _ax.set_xlim(0.64, 1.06)
    _ax.set_xlabel(
        "Macro ST-RAE, mean over 25 folds (lower is better; bars are ±1 SD across folds)"
    )
    _ax.tick_params(axis="y", length=0)
    _ax.grid(axis="x", color="#e6e6e6", linewidth=0.7)
    _ax.set_axisbelow(True)
    _handles, _labels = _ax.get_legend_handles_labels()
    # Upper left is empty: the weak fingerprints all sit at 0.85 and above.
    _ax.legend(_handles, _labels, loc="upper left", fontsize=8.5, ncol=1)
    _fig.tight_layout()
    save_fig(_fig, "representation_ladder.png")
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Figure 3 — Multitask learning helps when the encoder is learned

    Six models, each trained single-task and multitask on identical shared folds
    (`cv.shared_scaffold_folds`). The plotted quantity is the per-fold difference in
    macro ST-RAE, multitask minus single-task, so left of zero favors multitask.
    """
    )
    return


@app.cell
def _(
    OUT_DIR,
    PROJECT_ROOT,
    fold_vector,
    macro_by_fold,
    macro_summary,
    paired_fold_bootstrap,
    pl,
    with_verdicts,
):
    _cache = PROJECT_ROOT / "experiments" / "03_methods" / "cache"
    _mt_oof = pl.concat(
        [pl.read_parquet(_p) for _p in sorted(_cache.glob("mt_*_full.parquet"))],
        how="vertical_relaxed",
    )
    _macro = macro_by_fold(_mt_oof)
    _means = {r["method"]: r["mean"] for r in macro_summary(_macro).iter_rows(named=True)}

    MT_MODELS = ["xgb", "lgbm", "chemprop", "tabpfn", "macau", "tabicl"]
    _rows = []
    for _m in MT_MODELS:
        _res = paired_fold_bootstrap(
            fold_vector(_macro, f"{_m}_multitask"), fold_vector(_macro, f"{_m}_singletask")
        )
        _rows.append(
            {
                "model": _m,
                "single": _means[f"{_m}_singletask"],
                "multi": _means[f"{_m}_multitask"],
                **_res,
            }
        )
    mt_paired = with_verdicts(pl.DataFrame(_rows), "model").sort("diff")
    mt_paired.write_csv(OUT_DIR / "multitask_fold_paired.csv")
    mt_paired
    return MT_MODELS, mt_paired


@app.cell
def _(COLOR, mt_paired, plt, save_fig):
    _rep = {
        "xgb": "XGBoost · ECFP4",
        "lgbm": "LightGBM · ECFP4",
        "chemprop": "Chemprop · learned graph",
        "tabpfn": "TabPFN · frozen CheMeleon",
        "macau": "Macau · frozen CheMeleon",
        "tabicl": "TabICL · frozen CheMeleon",
    }
    _rows = list(mt_paired.iter_rows(named=True))
    _fig, _ax = plt.subplots(figsize=(8.6, 4.2))
    for _i, _r in enumerate(_rows):
        if _r["holm_significant"] and _r["ci_high"] < 0:
            _c = COLOR["better"]
        elif _r["holm_significant"] and _r["ci_low"] > 0:
            _c = COLOR["worse"]
        else:
            _c = COLOR["similar"]
        _ax.errorbar(
            _r["diff"],
            _i,
            xerr=[[_r["diff"] - _r["ci_low"]], [_r["ci_high"] - _r["diff"]]],
            fmt="o",
            color=_c,
            ecolor=_c,
            capsize=4,
            elinewidth=1.8,
            markersize=7,
        )
        _ax.text(
            0.0155,
            _i,
            f"{_r['single']:.3f} → {_r['multi']:.3f}",
            va="center",
            fontsize=9,
            color="#333333",
        )
    _ax.axvline(0, color="black", linestyle="--", linewidth=1)
    _ax.set_yticks(range(len(_rows)), [_rep[_r["model"]] for _r in _rows])
    _ax.invert_yaxis()
    _ax.set_xlim(-0.07, 0.036)
    _ax.set_xlabel(
        "Change in macro ST-RAE, multitask minus single-task (left of zero favors multitask)"
    )
    _ax.grid(axis="x", color="#ebebeb", linewidth=0.7)
    _ax.set_axisbelow(True)
    _ax.tick_params(axis="y", length=0)
    _ax.text(0.0155, -0.75, "single → multi", fontsize=8.5, color="#555555")
    _fig.tight_layout()
    save_fig(_fig, "multitask_paired.png")
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Figure 4 — Pretraining on public data helps only with the encoder unfrozen

    `05`'s five arms against `chemprop_multitask`, the `03` winner, on shared folds.
    All six are one Chemprop D-MPNN with four masked heads, so a difference is
    attributable to the training data, not the architecture.
    """
    )
    return


@app.cell
def _(
    OUT_DIR,
    PROJECT_ROOT,
    fold_vector,
    macro_by_fold,
    macro_summary,
    paired_fold_bootstrap,
    pl,
    with_verdicts,
):
    _o5 = pl.read_parquet(PROJECT_ROOT / "experiments" / "05_auxiliary_data" / "oof.parquet")
    _macro = macro_by_fold(_o5)
    _means = {r["method"]: r["mean"] for r in macro_summary(_macro).iter_rows(named=True)}
    _control = "chemprop_multitask"
    _rows = []
    for _arm in ["pubchem", "pubchem_frozen", "aux_screen", "augmented", "selection"]:
        _res = paired_fold_bootstrap(fold_vector(_macro, _arm), fold_vector(_macro, _control))
        _rows.append({"arm": _arm, "macro": _means[_arm], "control": _means[_control], **_res})
    pre_paired = with_verdicts(pl.DataFrame(_rows), "arm").sort("diff")
    pre_paired.write_csv(OUT_DIR / "pretraining_fold_paired.csv")
    pre_paired
    return (pre_paired,)


@app.cell
def _(COLOR, pre_paired, plt, save_fig):
    _label = {
        "pubchem": "Encoder pretrained on PubChem qHTS, then fine-tuned",
        "pubchem_frozen": "Same pretrained encoder, kept frozen",
        "aux_screen": "Screen log2 fold change as four extra heads",
        "augmented": "Screen negatives added as censored labels",
        "selection": "Inverse-propensity reweighting",
    }
    _rows = list(pre_paired.iter_rows(named=True))
    _fig, _ax = plt.subplots(figsize=(9, 3.9))
    for _i, _r in enumerate(_rows):
        if _r["holm_significant"] and _r["ci_high"] < 0:
            _c = COLOR["better"]
        elif _r["holm_significant"] and _r["ci_low"] > 0:
            _c = COLOR["worse"]
        else:
            _c = COLOR["similar"]
        _ax.errorbar(
            _r["diff"],
            _i,
            xerr=[[_r["diff"] - _r["ci_low"]], [_r["ci_high"] - _r["diff"]]],
            fmt="o",
            color=_c,
            ecolor=_c,
            capsize=4,
            elinewidth=1.8,
            markersize=7,
        )
        _ax.text(0.046, _i, f"{_r['macro']:.3f}", va="center", fontsize=9, color="#333333")
    _ax.axvline(0, color=COLOR["reference"], linestyle="--", linewidth=1.2)
    _ax.set_yticks(range(len(_rows)), [_label[_r["arm"]] for _r in _rows])
    _ax.invert_yaxis()
    _ax.set_xlim(-0.04, 0.056)
    _ax.set_xlabel(
        "Change in macro ST-RAE against multitask Chemprop with no extra data "
        "(left of zero is better)"
    )
    _ax.text(0.046, -0.75, "macro\nST-RAE", fontsize=8.5, color="#555555", va="bottom")
    _ax.grid(axis="x", color="#ebebeb", linewidth=0.7)
    _ax.set_axisbelow(True)
    _ax.tick_params(axis="y", length=0)
    _fig.tight_layout()
    save_fig(_fig, "pretraining_arms.png")
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Figure 5 — Four routes to the CYP2D6 pharmacophore

    The hypothesis (`PLAN.md`): CYP2D6 binds through a basic nitrogen and an aromatic
    ring that circular fingerprints express poorly, so a better description of that
    pharmacophore should help CYP2D6 *selectively*. Three panels, in the order the
    evidence arrived.

    - **A** is the direct check. Does the feature track potency on CYP2D6 and not
      elsewhere?
    - **B** is the model check. Does adding it to ECFP4 lift LightGBM, and is the
      lift larger than what the shipped model already scores?
    - **C** is the structure-based route on CYP2D6 alone, in the metric the board uses.
    """
    )
    return


@app.cell
def _(OUT_DIR, PROJECT_ROOT, pl):
    _e = PROJECT_ROOT / "experiments"
    direct = pl.read_csv(_e / "10_pka_features" / "direct_check.csv")

    _a09 = pl.read_csv(_e / "09_cyp2d6_representation" / "arm_results.csv")
    _a10 = pl.read_csv(_e / "10_pka_features" / "arm_results.csv")
    _incumbent = pl.read_csv(_e / "09_cyp2d6_representation" / "incumbent_reference.csv")

    # `ecfp` is the same run in both notebooks (same folds, seed 0); the check below
    # makes that an assertion rather than an assumption, since panel B draws one bar
    # from each notebook.
    _ecfp09 = _a09.filter(pl.col("arm") == "ecfp").select("endpoint", "spearman")
    _ecfp10 = _a10.filter(pl.col("arm") == "ecfp").select("endpoint", "spearman")
    assert (
        _ecfp09.join(_ecfp10, on="endpoint")
        .select((pl.col("spearman") - pl.col("spearman_right")).abs().max())
        .item()
        < 1e-9
    ), "09 and 10 disagree on the ECFP4 baseline"

    routes = pl.concat(
        [
            _a09.filter(pl.col("arm").is_in(["ecfp", "ecfp+pharmacophore_3d"])).select(
                "endpoint", "arm", "spearman"
            ),
            _a10.filter(pl.col("arm").is_in(["ecfp+pka_weight"])).select(
                "endpoint", "arm", "spearman"
            ),
            _incumbent.rename({"chemprop_pubchem_spearman": "spearman"})
            .with_columns(pl.lit("chemprop_pubchem (shipped)").alias("arm"))
            .select("endpoint", "arm", "spearman"),
        ]
    )
    routes.write_csv(OUT_DIR / "cyp2d6_routes_spearman.csv")

    docking = pl.read_csv(_e / "11_docking_3d" / "chemprop_summary.csv").select("arm", "st_rae")
    return direct, docking, routes


@app.cell
def _(COLOR, direct, docking, np, plt, routes, save_fig):
    _eps = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
    _x = np.arange(len(_eps))
    _fig = plt.figure(figsize=(10.4, 7.6))
    _gs = _fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], hspace=0.42, wspace=0.28)

    # --- A: the feature is selective for CYP2D6 ---------------------------------
    _ax = _fig.add_subplot(_gs[0, 0])
    _w = 0.38
    _b = [direct.filter(direct["endpoint"] == _e)["binary_basicN_spearman"].item() for _e in _eps]
    _p = [direct.filter(direct["endpoint"] == _e)["pka_weight_spearman"].item() for _e in _eps]
    _ax.bar(_x - _w / 2, _b, _w, color=COLOR["light"], label="Basic nitrogen present (0/1)")
    _ax.bar(
        _x + _w / 2, _p, _w, color=COLOR["dark"], label="Basic nitrogen, weighted by predicted pKa"
    )
    for _xi, _v in zip(_x - _w / 2, _b, strict=True):
        _ax.text(
            _xi,
            _v + (0.012 if _v >= 0 else -0.012),
            f"{_v:.2f}",
            ha="center",
            va="bottom" if _v >= 0 else "top",
            fontsize=8,
        )
    for _xi, _v in zip(_x + _w / 2, _p, strict=True):
        _ax.text(
            _xi,
            _v + (0.012 if _v >= 0 else -0.012),
            f"{_v:.2f}",
            ha="center",
            va="bottom" if _v >= 0 else "top",
            fontsize=8,
        )
    _ax.axhline(0, color="black", linewidth=0.8)
    _ax.set_xticks(_x, [_e.replace("CYP", "") for _e in _eps])
    _ax.set_xlabel("CYP isoform")
    _ax.set_ylabel("Spearman with pIC50")
    _ax.set_ylim(-0.24, 0.56)
    _ax.set_title("A. The feature tracks potency on CYP2D6 only")
    _ax.legend(fontsize=8, loc="upper left")

    # --- B: what it buys inside a model ----------------------------------------
    _ax = _fig.add_subplot(_gs[0, 1])
    _arms = [
        ("ecfp", "LightGBM · ECFP4", COLOR["light"]),
        ("ecfp+pharmacophore_3d", "+ 7 pharmacophore descriptors (3D)", COLOR["neutral"]),
        ("ecfp+pka_weight", "+ pKa-weighted basic nitrogen", COLOR["dark"]),
        (
            "chemprop_pubchem (shipped)",
            "Shipped model (PubChem-pretrained Chemprop)",
            COLOR["highlight"],
        ),
    ]
    _w = 0.2
    for _k, (_arm, _lab, _c) in enumerate(_arms):
        _vals = [
            routes.filter((routes["arm"] == _arm) & (routes["endpoint"] == _e))["spearman"].item()
            for _e in _eps
        ]
        _ax.bar(_x + (_k - 1.5) * _w, _vals, _w, color=_c, label=_lab)
    _ax.set_xticks(_x, [_e.replace("CYP", "") for _e in _eps])
    _ax.set_xlabel("CYP isoform")
    _ax.set_ylabel("Spearman, out-of-fold")
    _ax.set_ylim(0, 1.0)
    _ax.set_title("B. +0.03 Spearman on CYP2D6, still below the shipped model")
    _ax.legend(fontsize=7.6, loc="upper left", ncol=1)

    # --- C: docking, in the board's metric, CYP2D6 only ---------------------------
    _ax = _fig.add_subplot(_gs[1, :])
    _names = {
        "lgbm_ecfp": ("LightGBM · ECFP4", "lgbm"),
        "lgbm_unimol_etkdg": ("LightGBM · Uni-Mol embedding, free conformer", "lgbm"),
        "lgbm_unimol_docked": ("LightGBM · Uni-Mol embedding, docked pose", "lgbm"),
        "lgbm_docking_score": ("LightGBM · smina docking score", "lgbm"),
        "lgbm_plif": ("LightGBM · protein-ligand interaction fingerprint", "lgbm"),
        "chemprop_pubchem_docking_score": ("Pretrained Chemprop + docking score", "cp"),
        "chemprop_atom_plif": ("Chemprop from scratch + atom-level interactions", "cp"),
        "chemprop_pubchem_plif": ("Pretrained Chemprop + interaction fingerprint", "cp"),
        "pubchem (05 chemprop, reused OOF)": (
            "Shipped model: pretrained Chemprop, no structure",
            "inc",
        ),
    }
    _rows = [(_names[r["arm"]], r["st_rae"]) for r in docking.iter_rows(named=True)]
    _rows.sort(key=lambda t: -t[1])
    _color = {"lgbm": COLOR["light"], "cp": COLOR["neutral"], "inc": COLOR["highlight"]}
    for _i, ((_lab, _kind), _v) in enumerate(_rows):
        _ax.hlines(_i, 0.9, _v, color="#d9d9d9", linewidth=1)
        _ax.plot(
            _v,
            _i,
            "o",
            color=_color[_kind],
            markersize=9,
            markeredgecolor="#444444",
            markeredgewidth=0.6,
        )
        _ax.text(_v + 0.002, _i, f"{_v:.3f}", va="center", fontsize=8.5)
    _ax.axvline(1.0, color="#555555", linestyle=":", linewidth=1)
    _ax.text(1.0015, -0.62, "predicting the mean", fontsize=8, color="#555555", va="center")
    _ax.set_yticks(range(len(_rows)), [t[0][0] for t in _rows])
    _ax.set_ylim(len(_rows) - 0.4, -1.0)
    _ax.set_xlim(0.9, 1.085)
    _ax.set_xlabel("CYP2D6 ST-RAE, 5×5 scaffold CV (lower is better)")
    _ax.set_title(
        "C. Docking CYP2D6: the interaction fingerprint helps LightGBM, "
        "nothing beats the shipped model"
    )
    _ax.tick_params(axis="y", length=0)
    _ax.grid(axis="x", color="#ebebeb", linewidth=0.7)
    _ax.set_axisbelow(True)
    save_fig(_fig, "cyp2d6_routes.png")
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Figure 6 — On a ~21%-positive label, MCC peaks well below 0.5

    Macro MCC across both TDI isoforms against the decision threshold, single-task
    arms from `04` on the 25 shared folds. The grid and the macro-averaging are
    `04`'s own.
    """
    )
    return


@app.cell
def _(C, OUT_DIR, PROJECT_ROOT, evaluation, np, pl):
    _cache = PROJECT_ROOT / "experiments" / "04_methods_tdi" / "cache"
    tdi_oof = pl.concat(
        [pl.read_parquet(_p) for _p in sorted(_cache.glob("tdi_*_fold*_full.parquet"))],
        how="vertical_relaxed",
    )

    def macro_mcc_by_fold(frame: pl.DataFrame, threshold: float) -> pl.DataFrame:
        """Per-fold macro MCC over both isoforms after relabeling at `threshold`.

        `y_pred` is rebuilt from `y_prob` rather than read, because the cached labels
        were cut at whatever threshold the run used and the point of the sweep is to
        move it.
        """
        scored = frame.with_columns((pl.col("y_prob") >= float(threshold)).alias("y_pred"))
        folds = evaluation.fold_metrics_classification(scored)
        return evaluation.macro_averaged_fold_metrics(
            {i: folds.filter(pl.col("endpoint") == i) for i in C.TDI_ISOFORMS}, metric_col="mcc"
        )

    THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)
    _rows = []
    for _m in sorted(tdi_oof["method"].unique().to_list()):
        _sub = tdi_oof.filter(pl.col("method") == _m)
        for _t in THRESHOLDS:
            _macro = macro_mcc_by_fold(_sub, float(_t))
            _rows.append(
                {
                    "method": _m,
                    "threshold": float(_t),
                    "macro_mcc": float(_macro["mcc"].mean()),
                    # Share of compounds called positive: the mechanism behind the curve.
                    "share_predicted_positive": float((_sub["y_prob"] >= float(_t)).mean()),
                }
            )
    tdi_sweep = pl.DataFrame(_rows)
    tdi_sweep.write_csv(OUT_DIR / "tdi_threshold_sweep.csv")

    # One threshold per method, as in `04`: the macro-MCC maximum, ties broken toward
    # the larger threshold (fewer positives called on an imbalanced label).
    best_threshold = {
        r["method"]: r["threshold"]
        for r in tdi_sweep.sort(["macro_mcc", "threshold"], descending=[True, True])
        .group_by("method")
        .first()
        .iter_rows(named=True)
    }

    # Anchor the sweep to the positive rate it is being judged against.
    TDI_POSITIVE_RATE = float(
        tdi_oof.filter(pl.col("method") == "majority_singletask")["y_true"].mean()
    )
    return TDI_POSITIVE_RATE, best_threshold, macro_mcc_by_fold, tdi_oof, tdi_sweep


@app.cell
def _(TDI_POSITIVE_RATE, plt, save_fig, tdi_sweep):
    _style = {
        "tabicl_singletask": ("TabICL · frozen CheMeleon", "#d62728"),
        "tabpfn_singletask": ("TabPFN · frozen CheMeleon", "#9467bd"),
        "lgbm_singletask": ("LightGBM · ECFP4 (class-balanced)", "#1f77b4"),
        "xgb_singletask": ("XGBoost · ECFP4 (class-balanced)", "#8c564b"),
        "chemprop_singletask": ("Chemprop · learned graph", "#2ca02c"),
        "majority_singletask": ("Always predict negative", "#8c8c8c"),
    }
    _fig, _ax = plt.subplots(figsize=(8.8, 4.8))
    for _m, (_lab, _c) in _style.items():
        _sub = tdi_sweep.filter(tdi_sweep["method"] == _m).sort("threshold")
        _ax.plot(
            _sub["threshold"],
            _sub["macro_mcc"],
            marker="o",
            color=_c,
            markersize=3.2,
            linewidth=1.6,
            label=_lab,
            linestyle="--" if _m == "majority_singletask" else "-",
        )
        if _m != "majority_singletask":
            _best = _sub.sort("macro_mcc", descending=True).row(0, named=True)
            _ax.plot(
                _best["threshold"],
                _best["macro_mcc"],
                "o",
                color=_c,
                markersize=9,
                markerfacecolor="white",
                markeredgewidth=1.8,
            )
    _ax.axvline(0.5, color="#555555", linestyle=":", linewidth=1)
    _ax.text(0.505, 0.255, "default 0.5", fontsize=8.5, color="#555555", va="top")
    _ax.set_xlabel("Decision threshold on the predicted probability")
    _ax.set_ylabel("Macro MCC (CYP3A4 and CYP2D6), mean over 25 folds")
    _ax.set_ylim(-0.01, 0.265)
    _ax.set_xlim(0.03, 0.97)
    # The positive rate rides in the legend title: any free corner of the axes is
    # crossed by a curve or by the always-negative baseline at zero.
    _ax.legend(
        fontsize=8.5,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        title=f"{TDI_POSITIVE_RATE:.1%} of training labels are positive",
        title_fontsize=8.5,
        alignment="left",
    )
    _ax.grid(axis="y", color="#ebebeb", linewidth=0.7)
    _ax.set_axisbelow(True)
    _fig.tight_layout()
    save_fig(_fig, "tdi_threshold_sweep.png")
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Figure 7 — The TDI ensemble works because its members disagree

    **A** is `14`'s result, every number out-of-sample with respect to the weights:
    weights fitted on four outer repeats, scored on the fifth, rotated. **B** is the
    reason it worked here and would not have on regression: how much the members'
    out-of-fold predictions agree with each other.
    """
    )
    return


@app.cell
def _(OUT_DIR, PROJECT_ROOT, np, pl):
    _e = PROJECT_ROOT / "experiments"
    ens_summary = pl.read_csv(_e / "14_tdi_ensemble" / "summary.csv")
    ens_folds = pl.read_csv(_e / "14_tdi_ensemble" / "per_outer_fold_mcc.csv")

    def _pairwise_r(frame: pl.DataFrame, value: str, group: str) -> pl.DataFrame:
        """Pearson r between every pair of arms, per endpoint, on the same rows."""
        out = []
        for (endpoint,), part in frame.group_by(["endpoint"]):
            wide = part.pivot(on=group, index=["fold", "Molecule_Name"], values=value).drop_nulls()
            arms = [c for c in wide.columns if c not in ("fold", "Molecule_Name")]
            for i, a in enumerate(arms):
                for b in arms[i + 1 :]:
                    out.append(
                        {
                            "endpoint": endpoint,
                            "pair": f"{a} vs {b}",
                            "r": float(np.corrcoef(wide[a].to_numpy(), wide[b].to_numpy())[0, 1]),
                        }
                    )
        return pl.DataFrame(out)

    _tdi = pl.read_parquet(_e / "13_tdi_pretrained" / "oof.parquet")
    _reg = pl.read_parquet(_e / "07_placement" / "oof.parquet")
    agreement = pl.concat(
        [
            _pairwise_r(_tdi, "y_prob", "method").with_columns(
                pl.lit("TDI: four decorrelated arms").alias("track")
            ),
            _pairwise_r(_reg, "y_pred", "method").with_columns(
                pl.lit("Regression: four PubChem-family arms").alias("track")
            ),
        ]
    )
    # `group_by` order is not stable across runs, and the dot jitter in Figure 7 follows
    # row order, so sort before anything is drawn or written.
    agreement = agreement.sort("track", "endpoint", "pair")
    agreement.write_csv(OUT_DIR / "member_agreement.csv")
    return agreement, ens_folds, ens_summary


@app.cell
def _(COLOR, agreement, ens_folds, ens_summary, np, pl, plt, save_fig):
    _fig, (_ax, _bx) = plt.subplots(
        1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1.25, 1]}
    )

    # --- A: macro MCC per arm, with the five held-out repeats as dots ----------------
    _label = {
        "caruana": "Caruana-weighted blend",
        "uniform": "Plain mean of the four",
        "tabicl_singletask": "TabICL (the incumbent)",
        "chemprop_pubchem": "Chemprop, PubChem-pretrained",
        "chemprop_pubchem_frozen": "Chemprop, pretrained, frozen",
        "chemprop_scratch": "Chemprop, from scratch",
    }
    _macro_by_repeat = (
        ens_folds.group_by("arm", "held_out")
        .agg(pl.col("mcc").mean().alias("macro"))
        .sort("arm", "held_out")
    )
    _order = ens_summary.sort("macro", descending=True)["arm"].to_list()
    _color = {
        "caruana": COLOR["better"],
        "uniform": COLOR["similar"],
        "tabicl_singletask": COLOR["reference"],
    }
    for _i, _arm in enumerate(_order):
        _mean = ens_summary.filter(ens_summary["arm"] == _arm)["macro"].item()
        _c = _color.get(_arm, COLOR["light"])
        _ax.barh(_i, _mean, color=_c, height=0.62)
        _dots = _macro_by_repeat.filter(_macro_by_repeat["arm"] == _arm)["macro"].to_numpy()
        _ax.plot(_dots, np.full(len(_dots), _i), "o", color="#333333", markersize=3.4, alpha=0.75)
        _ax.text(max(_mean, _dots.max()) + 0.006, _i, f"{_mean:.3f}", va="center", fontsize=9)
    _ax.set_yticks(range(len(_order)), [_label[_a] for _a in _order])
    _ax.invert_yaxis()
    _ax.set_xlim(0, 0.34)
    _ax.set_xlabel("Macro MCC, weights fitted on four repeats and scored on the fifth")
    _ax.set_title("A. Selection beats averaging; dots are the five held-out repeats")
    _ax.tick_params(axis="y", length=0)
    # --- B: how much the members agree -------------------------------------------------
    _tracks = ["TDI: four decorrelated arms", "Regression: four PubChem-family arms"]
    _rng = np.random.default_rng(0)
    for _k, _t in enumerate(_tracks):
        _r = agreement.filter(agreement["track"] == _t)["r"].to_numpy()
        _bx.plot(
            _k + _rng.uniform(-0.12, 0.12, len(_r)),
            _r,
            "o",
            color=COLOR["dark"] if _k == 0 else COLOR["highlight"],
            markersize=6.5,
            alpha=0.8,
        )
        _bx.hlines(np.median(_r), _k - 0.3, _k + 0.3, color="#333333", linewidth=1.5)
        _bx.text(
            _k + 0.33,
            np.median(_r),
            f"median {np.median(_r):.2f}\nrange {_r.min():.2f} to {_r.max():.2f}",
            va="center",
            fontsize=8.5,
        )
    _bx.set_xticks([0, 1], ["TDI\n(12 pairs)", "Regression\n(24 pairs)"])
    _bx.set_xlim(-0.5, 1.95)
    _bx.set_ylim(0, 1.02)
    _bx.set_ylabel("Pearson r between two members' predictions")
    _bx.set_title("B. Members that agree leave nothing to average")
    _fig.tight_layout()
    save_fig(_fig, "tdi_ensemble.png")
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## TDI statistics the post quotes

    Two comparisons on macro MCC, each with the unit of resampling chosen to match
    what the data supports.

    **Multitask against single-task (`04`'s five models).** Fold-level, as for the
    regression figures: the per-fold macro MCC difference at each arm's own tuned
    threshold, 25 shared folds.

    **`13`'s arms and `14`'s ensemble against the TabICL incumbent.** These exist only
    as out-of-fold probabilities, and the ensemble was scored on five held-out
    repeats, too few to bootstrap. So the unit is the **compound**: a compound's rows
    from all five repeats are resampled together, because they are five looks at one
    compound and not five compounds. `13` and `14` resampled rows instead, which treats
    the five looks as independent and narrows the interval, and they pooled both
    isoforms into a single MCC while the board ranks the macro-average. Their point
    estimates therefore differ slightly from the ones here (the ensemble's gain is
    +0.0197 there, +0.0185 as a macro difference).
    """
    )
    return


@app.cell
def _(
    OUT_DIR, best_threshold, macro_mcc_by_fold, paired_fold_bootstrap, pl, tdi_oof, with_verdicts
):
    _rows = []
    for _model in ["xgb", "lgbm", "tabicl", "tabpfn", "chemprop"]:
        _fold_scores = {}
        for _arm in ("singletask", "multitask"):
            _name = f"{_model}_{_arm}"
            _fold_scores[_arm] = (
                macro_mcc_by_fold(tdi_oof.filter(pl.col("method") == _name), best_threshold[_name])
                .sort("fold")["mcc"]
                .to_numpy()
            )
        _res = paired_fold_bootstrap(_fold_scores["multitask"], _fold_scores["singletask"])
        _rows.append(
            {
                "model": _model,
                "single": float(_fold_scores["singletask"].mean()),
                "multi": float(_fold_scores["multitask"].mean()),
                "single_threshold": best_threshold[f"{_model}_singletask"],
                "multi_threshold": best_threshold[f"{_model}_multitask"],
                **_res,
            }
        )
    tdi_mt_paired = with_verdicts(pl.DataFrame(_rows), "model").sort("diff")
    tdi_mt_paired.write_csv(OUT_DIR / "tdi_multitask_fold_paired.csv")
    tdi_mt_paired
    return (tdi_mt_paired,)


@app.cell
def _(OUT_DIR, PROJECT_ROOT, np, pl, with_verdicts):
    from sklearn.metrics import matthews_corrcoef

    _e = PROJECT_ROOT / "experiments"
    _arms = ["chemprop_pubchem", "chemprop_pubchem_frozen", "chemprop_scratch", "tabicl_singletask"]
    #: Every optimum in `04` and `13` sat here (`13_tdi_pretrained/best_thresholds.csv`),
    #: and `14` fixed it so a blend cannot gain by quietly moving the cutoff.
    _threshold = 0.25
    _candidates = [*_arms, "caruana", "uniform"]
    _endpoints = ["CYP3A4_is_TDI", "CYP2D6_is_TDI"]

    _oof = pl.read_parquet(_e / "13_tdi_pretrained" / "oof.parquet")
    _weights = pl.read_csv(_e / "14_tdi_ensemble" / "caruana_weights.csv")
    _wide = (
        _oof.pivot(
            on="method",
            index=["endpoint", "fold", "outer_fold", "Molecule_Name", "y_true"],
            values="y_prob",
        )
        .join(
            _weights.rename({a: f"w_{a}" for a in _arms}),
            left_on=["endpoint", "outer_fold"],
            right_on=["endpoint", "held_out"],
            how="left",
        )
        .with_columns(
            sum(pl.col(a) * pl.col(f"w_{a}") for a in _arms).alias("caruana"),
            (sum(pl.col(a) for a in _arms) / len(_arms)).alias("uniform"),
        )
    )

    def _mcc(tp, fp, tn, fn):
        """MCC from confusion counts, vectorized over bootstrap resamples."""
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return np.where(den > 0, (tp * tn - fp * fn) / np.where(den == 0, 1.0, den), 0.0)

    _n_resamples, _chunk = 10_000, 2_000
    _rng = np.random.default_rng(0)
    _dist: dict[tuple[str, str], np.ndarray] = {}
    _point: dict[tuple[str, str], float] = {}
    for _ep in _endpoints:
        _d = _wide.filter(pl.col("endpoint") == _ep)
        _mols = _d["Molecule_Name"].unique().to_list()
        _slot = {m: i for i, m in enumerate(_mols)}
        _mi = np.array([_slot[m] for m in _d["Molecule_Name"].to_list()])
        _y = _d["y_true"].to_numpy().astype(bool)
        _n = len(_mols)

        # Per-compound confusion counts summed over its repeats, so a resample is just
        # a weighted sum and every candidate is scored on the same resample (paired).
        _counts = {}
        for _c in _candidates:
            _pred = _d[_c].to_numpy() >= _threshold
            _cm = np.zeros((_n, 4))
            for _k, _hit in enumerate([_pred & _y, _pred & ~_y, ~_pred & ~_y, ~_pred & _y]):
                np.add.at(_cm[:, _k], _mi, _hit)
            _counts[_c] = _cm
            _total = _cm.sum(axis=0)
            _point[(_ep, _c)] = float(_mcc(*_total))
            # The vectorized MCC must agree with sklearn's on the real predictions.
            assert abs(_point[(_ep, _c)] - matthews_corrcoef(_y, _pred)) < 1e-9, (_ep, _c)

        _parts = {c: [] for c in _candidates}
        for _start in range(0, _n_resamples, _chunk):
            _w = _rng.multinomial(
                _n, np.full(_n, 1.0 / _n), size=min(_chunk, _n_resamples - _start)
            ).astype(float)
            for _c in _candidates:
                _t = _w @ _counts[_c]
                _parts[_c].append(_mcc(_t[:, 0], _t[:, 1], _t[:, 2], _t[:, 3]))
        for _c in _candidates:
            _dist[(_ep, _c)] = np.concatenate(_parts[_c])

    def _compare(a: str, b: str, scope: str) -> dict:
        eps = _endpoints if scope == "macro" else [scope]
        diff = np.mean([_dist[(e, a)] - _dist[(e, b)] for e in eps], axis=0)
        point = float(np.mean([_point[(e, a)] - _point[(e, b)] for e in eps]))
        p = 2.0 * min((diff <= 0).mean(), (diff >= 0).mean())
        return {
            "comparison": f"{a} - {b}",
            "scope": scope,
            "diff": point,
            "ci_low": float(np.percentile(diff, 2.5)),
            "ci_high": float(np.percentile(diff, 97.5)),
            "p_value": float(min(p, 1.0)),
        }

    _families = {
        "ensemble vs incumbent": [
            ("caruana", "tabicl_singletask"),
            ("uniform", "tabicl_singletask"),
        ],
        "arms vs incumbent": [
            ("chemprop_pubchem", "tabicl_singletask"),
            ("chemprop_pubchem_frozen", "tabicl_singletask"),
            ("chemprop_scratch", "tabicl_singletask"),
        ],
        "what pretraining does": [
            ("chemprop_pubchem", "chemprop_scratch"),
            ("chemprop_pubchem", "chemprop_pubchem_frozen"),
        ],
    }
    _out = []
    for _family, _pairs in _families.items():
        for _scope in ["macro", *_endpoints]:
            _rows = pl.DataFrame([_compare(a, b, _scope) for a, b in _pairs])
            _out.append(
                with_verdicts(_rows, "comparison").with_columns(pl.lit(_family).alias("family"))
            )
    tdi_cluster = pl.concat(_out).select(
        "family", "scope", "comparison", "diff", "ci_low", "ci_high", "p_value", "holm_significant"
    )
    tdi_cluster.write_csv(OUT_DIR / "tdi_cluster_bootstrap.csv")
    tdi_cluster
    return (tdi_cluster,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Figure 8 — One model configuration, placed three ways

    `R² = 2ρk − k² − b²`, with `ρ` the Pearson correlation, `k = sd(pred)/sd(true)`
    and `b` the offset of the mean in units of `sd(true)`. Only `ρ` depends on how the
    model ranks compounds. `k` and `b` are set by an affine transform, which moves no
    compound's rank.

    **A** is the formula. **B** is the board: the same regression configuration scored
    in three versions that differ in placement. `07`'s vector is a refit (chemprop's
    pretraining has no seed control), so the 05-to-07 step is placement plus a rerun,
    while `12` reuses `05`'s vectors on CYP1A2, CYP2C9 and CYP3A4 and `07`'s on CYP2D6.
    """
    )
    return


@app.cell
def _(OUT_DIR, PROJECT_ROOT, our_row, pl, short):
    _e = PROJECT_ROOT / "experiments"
    _exp = pl.read_csv(_e / "12_final_placement" / "expected_performance.csv")
    _decomp = pl.read_csv(_e / "07_placement" / "board_decomposition.csv")

    _rows = []
    for _ep in ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]:
        _x = _exp.filter(pl.col("endpoint") == _ep).row(0, named=True)
        _live = our_row("live", "regression", _ep)
        _interim = our_row("interim", "regression", _ep)
        _rows.append(
            {
                "endpoint": _ep,
                "unplaced_05_live": _x["board_05"],
                "r2_placed_07_live": _x["board_07"],
                "gated_12_live": _live["ST-RAE"],
                "gated_12_interim": _interim["ST-RAE"],
            }
        )
    _macro_live = our_row("live", "regression", "overall")
    _macro_interim = our_row("interim", "regression", "overall")
    _m = _exp.filter(pl.col("endpoint") == "MA (macro)").row(0, named=True)
    _rows.append(
        {
            "endpoint": "Macro",
            "unplaced_05_live": _m["board_05"],
            "r2_placed_07_live": _m["board_07"],
            "gated_12_live": _macro_live["MA-ST-RAE"],
            "gated_12_interim": _macro_interim["MA-ST-RAE"],
        }
    )
    placement_scores = pl.DataFrame(_rows)
    placement_scores.write_csv(OUT_DIR / "placement_board_scores.csv")

    # Implied Pearson per endpoint from the board's Spearman on submission 05, the
    # value `07` used for the R² ceiling.
    implied_rho = dict(zip(_decomp["endpoint"], _decomp["implied_pearson"], strict=True))
    return implied_rho, placement_scores


@app.cell
def _(COLOR, implied_rho, np, placement_scores, plt, save_fig):
    _fig, (_ax, _bx) = plt.subplots(
        1, 2, figsize=(11.2, 4.3), gridspec_kw={"width_ratios": [1, 1.55]}
    )

    # --- A: R2 against the spread ratio, offset b = 0 ------------------------------------
    _k = np.linspace(0, 1.45, 300)
    for _ep, _c in [("CYP3A4", COLOR["dark"]), ("CYP2D6", COLOR["highlight"])]:
        _rho = implied_rho[_ep]
        _r2 = 2 * _rho * _k - _k**2
        _ax.plot(_k, _r2, color=_c, linewidth=2, label=f"{_ep}  (ρ = {_rho:.2f})")
        _ax.plot(_rho, _rho**2, "o", color=_c, markersize=8)
        _ax.plot(
            1.0,
            2 * _rho - 1,
            "s",
            color=_c,
            markersize=7,
            markerfacecolor="white",
            markeredgewidth=1.6,
        )
    _ax.axhline(0, color="black", linewidth=0.8)
    _ax.axvline(1.0, color="#999999", linestyle=":", linewidth=1)
    _ax.text(
        1.03, -0.07, "spread matches\nthe truth (k = 1)", fontsize=8, color="#666666", va="top"
    )
    _ax.annotate(
        "best possible:\nk = ρ and R² = ρ²",
        xy=(implied_rho["CYP3A4"], implied_rho["CYP3A4"] ** 2),
        xytext=(0.04, 0.58),
        fontsize=8.5,
        arrowprops={"arrowstyle": "-", "color": "#777777"},
    )
    _ax.set_xlabel("Spread ratio k = sd(prediction) / sd(truth)")
    _ax.set_ylabel("R², with the mean placed correctly (b = 0)")
    _ax.set_ylim(-0.75, 0.75)
    _ax.set_xlim(0, 1.45)
    _ax.set_title("A. Matching the truth's spread is not the optimum")
    _ax.legend(fontsize=8.5, loc="lower left")

    # --- B: board ST-RAE for three versions of one model ----------------------------------
    _eps = placement_scores["endpoint"].to_list()
    _x = np.arange(len(_eps))
    _w = 0.25
    _series = [
        ("unplaced_05_live", "Unplaced (05)", COLOR["light"]),
        ("r2_placed_07_live", "All four endpoints placed (07)", COLOR["neutral"]),
        ("gated_12_live", "Placed only where it pays (12)", COLOR["dark"]),
    ]
    for _j, (_col, _lab, _c) in enumerate(_series):
        _vals = placement_scores[_col].to_numpy()
        _bars = _bx.bar(_x + (_j - 1) * _w, _vals, _w, color=_c, label=_lab)
        for _b, _v in zip(_bars, _vals, strict=True):
            _bx.text(
                _b.get_x() + _b.get_width() / 2,
                _v + 0.012,
                f"{_v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.6,
            )
    _bx.plot(
        _x + _w,
        placement_scores["gated_12_interim"].to_numpy(),
        "D",
        color=COLOR["highlight"],
        markersize=7,
        markeredgecolor="#444444",
        markeredgewidth=0.6,
        label="Same as 12, full test set (interim)",
    )
    _bx.set_xticks(_x, [_e.replace("CYP", "") if _e != "Macro" else "Macro" for _e in _eps])
    _bx.set_xlabel("CYP isoform")
    _bx.set_ylabel("Board ST-RAE (lower is better)")
    _bx.set_ylim(0, 1.5)
    _bx.set_title("B. One model configuration, three placements: live half")
    _bx.legend(fontsize=8, loc="upper right", ncol=1)
    _fig.tight_layout()
    save_fig(_fig, "placement.png")
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Figure 9 — Where we sit on the interim leaderboard

    **A.** Spearman's ρ does not depend on placement, so across the field the macro
    ST-RAE is mostly a function of it. Where a point sits relative to that trend is
    what placement (and calibration) contribute beyond ordering. **B.** The TDI track.

    Five of the 219 entries are off the axes (a macro ST-RAE of 1.3 or more, up to
    3,115, or a macro Spearman below 0.2). Four of them, with a macro ST-RAE above 2
    or a Spearman below 0.2, are also left out of the trend fit; all five stay in the
    rank counts.
    """
    )
    return


@app.cell
def _(OUT_DIR, board_wide, np, pl):
    _reg = board_wide("interim", "regression", "overall").rename(
        {"MA-ST-RAE": "strae", "MA-Spearman's ρ": "rho"}
    )
    _tdi = board_wide("interim", "tdi", "overall").rename({"MA-MCC": "mcc"})

    # Robust to the handful of degenerate entries: fit only where both axes are sane.
    _fit_mask = (_reg["strae"] < 2.0) & (_reg["rho"] > 0.2)
    _slope, _intercept = np.polyfit(
        _reg.filter(_fit_mask)["rho"].to_numpy(), _reg.filter(_fit_mask)["strae"].to_numpy(), 1
    )
    _fit = _reg.filter(_fit_mask)
    _r = float(np.corrcoef(_fit["rho"].to_numpy(), _fit["strae"].to_numpy())[0, 1])
    _ours = _reg.filter(pl.col("is_ours")).row(0, named=True)
    _predicted = _slope * _ours["rho"] + _intercept
    _resid = (_fit["strae"] - (_slope * _fit["rho"] + _intercept)).to_numpy()

    board_position = {
        "n_regression": _reg.height,
        "n_fit": _fit.height,
        "slope": float(_slope),
        "intercept": float(_intercept),
        "pearson_r": _r,
        "our_strae": _ours["strae"],
        "our_rho": _ours["rho"],
        "our_strae_rank": _ours["rank"],
        "our_rho_rank": int((_reg["rho"] > _ours["rho"]).sum() + 1),
        "trend_strae_at_our_rho": float(_predicted),
        "our_residual": float(_ours["strae"] - _predicted),
        "share_of_field_with_lower_residual": float(
            (_resid < (_ours["strae"] - _predicted)).mean()
        ),
        "top_strae": float(_reg["strae"].min()),
        "top_rho": float(_reg.sort("strae").row(0, named=True)["rho"]),
        "median_rho": float(_reg["rho"].median()),
        "n_tdi": _tdi.height,
        "our_mcc": float(_tdi.filter(pl.col("is_ours"))["mcc"].item()),
        "our_mcc_rank": int(_tdi.filter(pl.col("is_ours"))["rank"].item()),
        "top_mcc": float(_tdi["mcc"].max()),
        "median_mcc": float(_tdi["mcc"].median()),
    }
    pl.DataFrame(
        {"quantity": list(board_position), "value": [float(v) for v in board_position.values()]}
    ).write_csv(OUT_DIR / "board_position.csv")
    board_position
    return (board_position,)


@app.cell
def _(COLOR, board_position, board_wide, np, plt, save_fig):
    from matplotlib.ticker import MaxNLocator

    _reg = board_wide("interim", "regression", "overall").rename(
        {"MA-ST-RAE": "strae", "MA-Spearman's ρ": "rho"}
    )
    _tdi = board_wide("interim", "tdi", "overall").rename({"MA-MCC": "mcc"})
    _fig, (_ax, _bx) = plt.subplots(
        1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.35, 1]}
    )

    _shown = _reg.filter((_reg["strae"] < 1.3) & (_reg["rho"] > 0.2))
    _off = _reg.height - _shown.height
    _others = _shown.filter(~_shown["is_ours"])
    _ax.scatter(
        _others["rho"], _others["strae"], s=16, color=COLOR["neutral"], alpha=0.55, linewidths=0
    )
    _xs = np.linspace(0.45, 0.85, 50)
    _ax.plot(
        _xs,
        board_position["slope"] * _xs + board_position["intercept"],
        color="#555555",
        linewidth=1.2,
        linestyle="--",
        label=(
            f"field trend (r = {board_position['pearson_r']:.2f}, n = {board_position['n_fit']})"
        ).replace("-", "−"),
    )
    _ax.scatter(
        [board_position["our_rho"]],
        [board_position["our_strae"]],
        s=95,
        color=COLOR["highlight"],
        edgecolor="#333333",
        linewidth=0.8,
        zorder=5,
        label=f"ours: rank {board_position['our_strae_rank']} on ST-RAE, "
        f"{board_position['our_rho_rank']} on Spearman",
    )
    _ax.plot(
        [board_position["our_rho"]] * 2,
        [board_position["our_strae"], board_position["trend_strae_at_our_rho"]],
        color=COLOR["highlight"],
        linewidth=1.4,
        zorder=4,
    )
    _ax.set_xlabel("Macro Spearman ρ (unaffected by placement)")
    _ax.set_ylabel("Macro ST-RAE (lower is better)")
    _ax.set_xlim(0.45, 0.85)
    _ax.set_ylim(0.3, 1.3)
    _ax.invert_yaxis()
    _ax.set_title(f"A. Regression, full test set ({_off} entries off the axes)")
    _ax.legend(fontsize=8.5, loc="lower right")

    _bx.hist(
        _tdi["mcc"], bins=np.arange(-0.05, 0.45, 0.025), color=COLOR["light"], edgecolor="white"
    )
    _bx.axvline(
        board_position["our_mcc"],
        color=COLOR["highlight"],
        linewidth=2.2,
        label=(
            f"ours: {board_position['our_mcc']:.3f}, "
            f"rank {board_position['our_mcc_rank']} of {board_position['n_tdi']}"
        ),
    )
    _bx.axvline(
        board_position["top_mcc"],
        color=COLOR["dark"],
        linewidth=1.4,
        linestyle="--",
        label=f"top: {board_position['top_mcc']:.3f}",
    )
    _bx.set_xlabel("Macro MCC, CYP3A4 and CYP2D6")
    _bx.set_ylabel("Entries")
    _bx.yaxis.set_major_locator(MaxNLocator(integer=True))
    _bx.set_title("B. TDI, full test set")
    _bx.legend(fontsize=8.5, loc="upper left")
    _fig.tight_layout()
    save_fig(_fig, "interim_board.png")
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Tables the post quotes

    No figure, just the numbers that would otherwise be pasted from a notebook output:
    whether widening the public data beyond PubChem helped (`07`'s arms, same
    fold-level comparison as Figure 4), how the CV expectation compares with the
    board, and where each endpoint sits in the field on the interim leaderboard.
    """
    )
    return


@app.cell
def _(
    OUT_DIR,
    PROJECT_ROOT,
    fold_vector,
    macro_by_fold,
    macro_summary,
    paired_fold_bootstrap,
    pl,
    with_verdicts,
):
    # `07` refit the `pubchem` arm (chemprop's pretraining has no seed control, so it
    # is the same configuration, not the same weights) and ran three wider-data arms
    # beside it on shared folds. The refit is also the only measure available of how
    # much two runs of one pretraining configuration differ.
    _o7 = pl.read_parquet(PROJECT_ROOT / "experiments" / "07_placement" / "oof.parquet")
    _macro7 = macro_by_fold(_o7)
    _means7 = {r["method"]: r["mean"] for r in macro_summary(_macro7).iter_rows(named=True)}
    _rows = []
    for _arm in ["pubchem_union", "full_union", "full_union_aux"]:
        _res = paired_fold_bootstrap(fold_vector(_macro7, _arm), fold_vector(_macro7, "pubchem"))
        _rows.append({"arm": _arm, "macro": _means7[_arm], "control": _means7["pubchem"], **_res})
    union_paired = with_verdicts(pl.DataFrame(_rows), "arm").sort("diff")
    union_paired.write_csv(OUT_DIR / "public_union_fold_paired.csv")

    # Same configuration, two pretraining runs: 05's checkpoint and 07's.
    _o5 = pl.read_parquet(PROJECT_ROOT / "experiments" / "05_auxiliary_data" / "oof.parquet")
    _m5 = {r["method"]: r["mean"] for r in macro_summary(macro_by_fold(_o5)).iter_rows(named=True)}
    pretrain_rerun = {
        "pubchem_05": _m5["pubchem"],
        "pubchem_07_refit": _means7["pubchem"],
        "difference": _m5["pubchem"] - _means7["pubchem"],
        "control_05": _m5["chemprop_multitask"],
    }
    union_paired, pretrain_rerun
    return pretrain_rerun, union_paired


@app.cell
def _(OUT_DIR, PROJECT_ROOT, our_row, pl):
    _e = PROJECT_ROOT / "experiments"
    _cv = pl.read_csv(_e / "05_auxiliary_data" / "per_endpoint.csv").filter(
        pl.col("method") == "pubchem"
    )
    _cv_macro = pl.read_csv(_e / "05_auxiliary_data" / "macro_table.csv").filter(
        pl.col("method") == "pubchem"
    )
    _exp = pl.read_csv(_e / "12_final_placement" / "expected_performance.csv")

    _rows = []
    for _ep in ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]:
        _rows.append(
            {
                "endpoint": _ep,
                "cv_unplaced": float(_cv[f"{_ep}_pIC50_direct_inhibition"].item()),
                "board_05_live_unplaced": float(
                    _exp.filter(pl.col("endpoint") == _ep)["board_05"].item()
                ),
                "board_12_live": our_row("live", "regression", _ep)["ST-RAE"],
                "board_12_interim": our_row("interim", "regression", _ep)["ST-RAE"],
            }
        )
    _rows.append(
        {
            "endpoint": "Macro",
            "cv_unplaced": float(_cv_macro["macro_st_rae"].item()),
            "board_05_live_unplaced": float(
                _exp.filter(pl.col("endpoint") == "MA (macro)")["board_05"].item()
            ),
            "board_12_live": our_row("live", "regression", "overall")["MA-ST-RAE"],
            "board_12_interim": our_row("interim", "regression", "overall")["MA-ST-RAE"],
        }
    )
    cv_vs_board = pl.DataFrame(_rows)
    cv_vs_board.write_csv(OUT_DIR / "cv_vs_board.csv")
    cv_vs_board
    return (cv_vs_board,)


@app.cell
def _(OUT_DIR, board_wide, pl):
    # Where each endpoint sits in the interim field, in the board's metric and in
    # Spearman's rho, which placement cannot change.
    _rows = []
    for _ep in ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]:
        _t = board_wide("interim", "regression", _ep)
        for _metric, _lower_is_better in [("ST-RAE", True), ("Spearman's ρ", False), ("R²", False)]:
            _ours = _t.filter(pl.col("is_ours"))[_metric].item()
            _better = (_t[_metric] < _ours) if _lower_is_better else (_t[_metric] > _ours)
            _rows.append(
                {
                    "endpoint": _ep,
                    "metric": _metric,
                    "ours": _ours,
                    "best": float(_t[_metric].min() if _lower_is_better else _t[_metric].max()),
                    "median": float(_t[_metric].median()),
                    "our_rank": int(_better.sum() + 1),
                    "n": _t.height,
                }
            )
    interim_by_endpoint = pl.DataFrame(_rows)
    interim_by_endpoint.write_csv(OUT_DIR / "interim_by_endpoint.csv")
    interim_by_endpoint
    return (interim_by_endpoint,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## How to read this

    **Figures 3 and 4 use fold-level paired intervals, not `03`'s and `05`'s pooled
    bootstrap.** The two agree on every verdict that matters here, and the fold-level
    intervals are wider, which is the honest direction. Where they disagree, the post
    says so. The disagreement is TabICL's multitask arm, which the pooled bootstrap
    in `03` calls significantly worse and the fold-level interval does not.

    **Figure 8's live-half scores for `05` and `07` are the board's** (read from
    `12_final_placement/expected_performance.csv`, which recorded them when they
    returned). Submission `12`'s live and interim scores come from the dated
    leaderboard snapshot. The interim board scores the full 750 compounds, the live
    board half of them, so the diamonds are not on the same compounds as the bars.
    """
    )
    return


if __name__ == "__main__":
    app.run()
