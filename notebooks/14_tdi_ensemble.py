import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # 14 — TDI: does an ensemble close the gap?

    `13` tested the last single-model hypothesis for the TDI gap and rejected it. Three
    candidates have now been eliminated:

    | hypothesis | result |
    |:--|:--|
    | threshold calibration | retuning buys +0.003 / +0.005; the shipped 0.25 is near-optimal |
    | two-stage via the pIC50 shift | oracle MCC 0.69/0.90 but loses to a direct classifier |
    | a pretrained encoder | transfers (+0.075 over scratch) yet still 0.039 behind TabICL |

    What remains is the **ensemble**. The contender at board macro MCC 0.4025 (against
    our 0.2111) Caruana-combines two decorrelated candidates, and their own write-up
    calls that the larger part of their TDI margin over a single model.

    ## Why this is worth running, and cheap

    `13`'s four arms are already computed, share folds, and are **genuinely
    decorrelated** — pairwise correlation of their OOF probabilities runs 0.30–0.62:

    | pair (CYP2D6 / CYP3A4) | r |
    |:--|--:|
    | `chemprop_pubchem` vs `chemprop_pubchem_frozen` | 0.54 / 0.62 |
    | `chemprop_pubchem` vs `tabicl_singletask` | 0.43 / 0.61 |
    | `chemprop_pubchem_frozen` vs `tabicl_singletask` | 0.30 / 0.57 |

    That is the regime where ensembling pays. For contrast, `07`'s four *regression*
    arms correlate at r ≈ 0.94 — four variants of one pretrained chemprop, where a
    blend has almost nothing to average away. Here three of the four arms are a
    different model family from the incumbent entirely.

    This notebook trains nothing. It reads `experiments/13_tdi_pretrained/oof.parquet`.

    ## The honest-evaluation problem, and how the nested folds solve it

    Selecting ensemble weights and then scoring them on the same predictions measures
    the selection, not the ensemble — and with four candidates and a metric as
    flexible as MCC, that optimism is not small. The 5×5 design gives a clean way out:
    weights are fitted on four outer repeats and scored on the fifth, rotated. Every
    reported number below is out-of-sample **with respect to the weights**, which is
    the only claim an ensemble can honestly make from OOF alone.
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

    from cyp import ensemble, evaluation

    return PROJECT_ROOT, Path, ensemble, evaluation, np, pl


@app.cell
def _(Path, PROJECT_ROOT):
    NOTEBOOK_NAME = Path(__file__).stem
    OUT_DIR = PROJECT_ROOT / "experiments" / NOTEBOOK_NAME
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return NOTEBOOK_NAME, OUT_DIR


@app.cell
def _(PROJECT_ROOT, mo, pl):
    oof = pl.read_parquet(PROJECT_ROOT / "experiments" / "13_tdi_pretrained" / "oof.parquet")
    ARMS = sorted(oof["method"].unique().to_list())
    ENDPOINTS = sorted(oof["endpoint"].unique().to_list())
    OUTER_FOLDS = sorted(oof["outer_fold"].unique().to_list())

    # Every arm must cover exactly the same (endpoint, fold, compound) rows, or a
    # blend is averaging different compounds per arm and the comparison is void.
    _counts = oof.group_by("method").agg(pl.len())["len"].to_list()
    assert len(set(_counts)) == 1, f"arms cover different rows: {_counts}"
    assert oof["y_prob"].is_nan().sum() == 0, "NaN probabilities in the source OOF"

    #: `04` and `13` both found every optimum here; it is fixed rather than re-tuned
    #: so that the ensemble is compared against its members at the same operating
    #: point, and a gain cannot come from quietly moving the threshold instead.
    THRESHOLD = 0.25

    mo.md(
        f"**{oof.height:,} OOF rows**, {len(ARMS)} arms × {len(ENDPOINTS)} endpoints × "
        f"{oof['fold'].n_unique()} folds ({len(OUTER_FOLDS)} outer repeats). "
        f"Threshold fixed at {THRESHOLD}."
    )
    return ARMS, ENDPOINTS, OUTER_FOLDS, THRESHOLD, oof


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Two blends, and why both

    - **Caruana** (`ensemble.caruana_selection`) — forward selection with replacement,
      bagged. Robust to correlated candidates and to a metric that cannot be
      decomposed per row, which MCC cannot: it comes from a confusion matrix over the
      whole set.
    - **Uniform** — the plain mean of all four arms, no fitting at all.

    The uniform blend is the control that matters. If Caruana and a plain mean score
    the same, the selection is doing nothing and the gain is just averaging — worth
    knowing before shipping a fitted object that can drift.
    """
    )
    return


@app.cell
def _(ARMS, ENDPOINTS, OUTER_FOLDS, OUT_DIR, THRESHOLD, ensemble, mo, np, oof, pl):
    from sklearn.metrics import matthews_corrcoef

    def _mcc_at(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        return float(matthews_corrcoef(y_true, y_prob >= THRESHOLD))

    def _probs(frame: pl.DataFrame) -> dict[str, np.ndarray]:
        """Arm -> probability vector, all aligned on the same row order."""
        return {
            arm: frame.filter(pl.col("method") == arm)
            .sort(["endpoint", "fold", "Molecule_Name"])["y_prob"]
            .to_numpy()
            for arm in ARMS
        }

    def _truth(frame: pl.DataFrame) -> np.ndarray:
        return (
            frame.filter(pl.col("method") == ARMS[0])
            .sort(["endpoint", "fold", "Molecule_Name"])["y_true"]
            .to_numpy()
        )

    rows, weight_rows = [], []
    for _endpoint in ENDPOINTS:
        _ep = oof.filter(pl.col("endpoint") == _endpoint)

        for _held_out in OUTER_FOLDS:
            _fit = _ep.filter(pl.col("outer_fold") != _held_out)
            _test = _ep.filter(pl.col("outer_fold") == _held_out)

            # Weights come only from the four repeats this held-out one is not in.
            _weights = ensemble.caruana_selection(
                _probs(_fit), _truth(_fit), _mcc_at, maximize=True, seed=0
            )
            weight_rows.append({"endpoint": _endpoint, "held_out": _held_out, **_weights})

            _test_probs = _probs(_test)
            _y = _truth(_test)
            _caruana = sum(_weights[a] * _test_probs[a] for a in ARMS)
            _uniform = np.mean([_test_probs[a] for a in ARMS], axis=0)

            _scores = {a: _mcc_at(_y, _test_probs[a]) for a in ARMS}
            _scores["caruana"] = _mcc_at(_y, _caruana)
            _scores["uniform"] = _mcc_at(_y, _uniform)
            for _name, _value in _scores.items():
                rows.append(
                    {
                        "endpoint": _endpoint,
                        "held_out": _held_out,
                        "arm": _name,
                        "mcc": _value,
                    }
                )

    per_fold = pl.DataFrame(rows)
    per_fold.write_csv(OUT_DIR / "per_outer_fold_mcc.csv")
    weights_table = pl.DataFrame(weight_rows)
    weights_table.write_csv(OUT_DIR / "caruana_weights.csv")

    summary = (
        per_fold.group_by(["arm", "endpoint"])
        .agg(pl.col("mcc").mean().round(4).alias("mcc"))
        .pivot(values="mcc", index="arm", on="endpoint")
    )
    summary = summary.with_columns(
        pl.mean_horizontal([pl.col(e) for e in ENDPOINTS]).round(4).alias("macro")
    ).sort("macro", descending=True)
    summary.write_csv(OUT_DIR / "summary.csv")
    mo.ui.table(summary, selection=None)
    return _mcc_at, per_fold, summary, weights_table


@app.cell
def _(mo, weights_table):
    mo.vstack(
        [
            mo.md(
                "**Caruana weights per held-out repeat.** Read these for stability: "
                "weights that swing between repeats mean the selection is fitting "
                "fold noise, and the uniform blend is the safer object to ship."
            ),
            mo.ui.table(weights_table, selection=None),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Is the difference one the CV can resolve?

    The standing rule, and PXR's most expensive mistake. Paired bootstrap over the
    held-out outer repeats, each blend against the incumbent, Holm-corrected.
    """
    )
    return


@app.cell
def _(ARMS, ENDPOINTS, OUT_DIR, THRESHOLD, evaluation, mo, np, oof, pl, weights_table):

    INCUMBENT = "tabicl_singletask"

    # Rebuild the held-out predictions as one long vector per candidate, so the
    # bootstrap resamples compounds across every outer repeat at once.
    _pieces = {name: [] for name in [*ARMS, "caruana", "uniform"]}
    _truth_pieces = []
    for _endpoint in ENDPOINTS:
        _ep = oof.filter(pl.col("endpoint") == _endpoint)
        for _held_out in sorted(_ep["outer_fold"].unique().to_list()):
            _test = _ep.filter(pl.col("outer_fold") == _held_out)
            _p = {
                a: _test.filter(pl.col("method") == a)
                .sort(["endpoint", "fold", "Molecule_Name"])["y_prob"]
                .to_numpy()
                for a in ARMS
            }
            _y = (
                _test.filter(pl.col("method") == ARMS[0])
                .sort(["endpoint", "fold", "Molecule_Name"])["y_true"]
                .to_numpy()
            )
            _w = (
                weights_table.filter(
                    (pl.col("endpoint") == _endpoint) & (pl.col("held_out") == _held_out)
                )
                .select(ARMS)
                .row(0)
            )
            _weights = dict(zip(ARMS, _w, strict=True))
            for a in ARMS:
                _pieces[a].append(_p[a])
            _pieces["caruana"].append(sum(_weights[a] * _p[a] for a in ARMS))
            _pieces["uniform"].append(np.mean([_p[a] for a in ARMS], axis=0))
            _truth_pieces.append(_y)

    stacked = {k: np.concatenate(v) for k, v in _pieces.items()}
    y_all = np.concatenate(_truth_pieces)

    _rows, _pvalues = [], {}
    for _cand in ["caruana", "uniform"]:
        _res = evaluation.paired_bootstrap(
            y_all,
            stacked[_cand] >= THRESHOLD,
            stacked[INCUMBENT] >= THRESHOLD,
            metric="mcc",
            seed=0,
        )
        _pvalues[_cand] = _res["p_value"]
        _rows.append(
            {
                "candidate": _cand,
                "vs": INCUMBENT,
                "diff": round(_res["diff"], 4),
                "ci_low": round(_res["ci_low"], 4),
                "ci_high": round(_res["ci_high"], 4),
                "p_raw": round(_res["p_value"], 5),
            }
        )

    comparison = pl.DataFrame(_rows).join(
        evaluation.holm_bonferroni(_pvalues),
        left_on="candidate",
        right_on="comparison",
        how="left",
    )
    comparison.write_csv(OUT_DIR / "paired_comparison.csv")
    mo.ui.table(comparison, selection=None)
    return comparison, stacked, y_all


@app.cell
def _(mo):
    mo.md(
        r"""
    ## How to read this

    **The comparison that decides anything is `caruana` vs `tabicl_singletask`.** A
    blend that does not beat its own best member at a resolvable margin is not worth
    the extra machinery — four models to fit and a weight vector to keep in sync with
    them, against one model that is already validated and on the board.

    **Check `uniform` against `caruana` before preferring the fitted one.** If they
    are indistinguishable, ship the mean: it has no fitted parameters, cannot drift
    from the OOF it was chosen on, and needs no weight table in `PROVENANCE.md`.

    **These numbers are out-of-sample with respect to the weights but still
    threshold-optimistic**, in the same way `04`'s were: 0.25 was chosen on OOF. The
    board discount `04` measured was concentrated on CYP3A4 (OOF 0.374 → board
    0.305) while CYP2D6 landed almost exactly. Expect the same shape here.

    **If the ensemble wins, it is still only half the contender's gap.** Their margin
    is 0.19 macro MCC over us; nothing in this repo has yet found more than a few
    hundredths. An ensemble win would be worth shipping and would not be a
    resolution of why they are ahead.
    """
    )
    return


if __name__ == "__main__":
    app.run()
