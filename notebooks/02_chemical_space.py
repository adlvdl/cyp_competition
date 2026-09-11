import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # 02 — Chemical space and SAR exploration

    No modelling here. This notebook asks what the four CYP endpoints look like as
    *chemistry*, to explain the finding that `01_baseline.py` left open: LightGBM on
    ECFP4 learns CYP3A4 (ST-RAE 0.72) but sits at or above the mean-predictor line
    on CYP2C9, CYP1A2 and CYP2D6. CLAUDE.md's reading is that this is "not a tuning
    problem — it needs better representations or more data". That is a hypothesis,
    and this notebook tests the representation half of it directly.

    The structure follows the PXR challenge's `1a`–`1e` notebooks
    ([repo](https://github.com/adlvdl/pxr_challenge)), generalised from PXR's single
    pEC50 to four sparse endpoints plus two TDI isoforms:

    | Section | PXR origin | Question |
    |:--|:--|:--|
    | Chemical space embeddings | 1b | Is potency spatially organised, or scattered? |
    | Train/test coverage | 1d | Can a model reach the test compounds at all? |
    | Similarity distributions | 1c | What does "similar" mean for *this* dataset? |
    | Activity cliffs | 1c | Where does the similarity–activity assumption break? |
    | Matched molecular pairs | 1a, 1b | Which specific transformations move potency? |
    | Scaffold analysis | 1e | Is coverage a scaffold problem or a substituent problem? |

    Everything reusable lives in `src/cyp/` — `embedding.py`, `similarity.py`,
    `mmp.py`, `scaffolds.py`, `interactive.py` — so the plots here are a few lines
    each and the next notebook can reuse them. Static PNGs are written to
    `experiments/02_chemical_space/`; the interactive Altair charts are
    notebook-only, since their point is hovering over compounds.

    **Caching.** The expensive steps are the UMAP/t-SNE embeddings, the ~12M-pair
    similarity matrices, and `mmpdb fragment`. Each is cached under
    `experiments/02_chemical_space/cache/`, gated on file existence in the same
    pattern `01_baseline.py` uses for CV. Set `QUICK = True` while iterating.

    Unlike `01_baseline.py`, not all of this cache is committed. The raw pairwise
    similarity tables run to ~170 MB and are a pure function of the data snapshot
    with no training involved, so they are gitignored and regenerate in minutes; the
    small derived caches, the scorecard CSVs and the figures are committed. Pin a
    `snapshot=` in the loader cell to reproduce this against an older download.
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

    # This notebook lives in notebooks/; the package lives in src/.
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from cyp import constants as C
    from cyp import data, embedding, interactive, mmp, scaffolds, similarity

    return (
        C,
        PROJECT_ROOT,
        Path,
        data,
        embedding,
        interactive,
        mmp,
        mo,
        np,
        pl,
        scaffolds,
        similarity,
    )


@app.cell
def _(PROJECT_ROOT, Path):
    # Derive the output directory from the filename so a rename cannot silently
    # orphan the outputs (CLAUDE.md convention, same as 01_baseline.py).
    NOTEBOOK_NAME = Path(__file__).stem
    OUT_DIR = PROJECT_ROOT / "experiments" / NOTEBOOK_NAME
    CACHE_DIR = OUT_DIR / "cache"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR, NOTEBOOK_NAME, OUT_DIR


@app.cell
def _():
    # QUICK subsamples the compound set so the whole notebook runs in ~1 minute.
    # Every cached artefact is keyed on this, so quick and full runs never collide.
    QUICK = False

    CACHE_SUFFIX = "quick" if QUICK else "full"
    QUICK_N = 600
    return CACHE_SUFFIX, QUICK, QUICK_N


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Load and assemble

    Three tables carry structures: the inhibition training set (4,905 rows, four
    sparse pIC50 endpoints), the TDI training set (6,145 rows, two boolean labels),
    and the blinded test set (750 rows, identifiers only).

    They overlap but are not nested, so the combined frame below is a full outer
    join keyed on `Molecule_Name` with membership flags — PXR's `all_compounds`
    pattern from notebook 1a. The flags are what let a single embedding be coloured
    by "which dataset is this compound in", which is the first question to ask of
    any blind challenge.
    """
    )
    return


@app.cell
def _(C, data, pl):
    train_inhib = data.load_train_inhibition()
    train_tdi = data.load_train_tdi()
    test = data.load_test()

    # Endpoint values keyed by compound. The TDI table repeats the direct-inhibition
    # columns, so take those from the inhibition table alone and pull only the
    # boolean TDI labels from the TDI table -- otherwise the join produces two
    # copies of every pIC50 column.
    _inhib = train_inhib.select(
        ["Molecule_Name", "SMILES", *C.REGRESSION_ENDPOINTS]
    ).with_columns(pl.lit(True).alias("in_inhibition"))

    _tdi = train_tdi.select(
        ["Molecule_Name", "SMILES", *C.CLASSIFICATION_ENDPOINTS]
    ).with_columns(pl.lit(True).alias("in_tdi"))

    _test = test.select(["Molecule_Name", "SMILES"]).with_columns(
        pl.lit(True).alias("in_test")
    )

    all_compounds = (
        _inhib.join(_tdi, on=["Molecule_Name", "SMILES"], how="full", coalesce=True)
        .join(_test, on=["Molecule_Name", "SMILES"], how="full", coalesce=True)
        .with_columns(
            pl.col("in_inhibition").fill_null(False),
            pl.col("in_tdi").fill_null(False),
            pl.col("in_test").fill_null(False),
        )
        # One row per structure. Molecule_Name is the submission key, so dedupe on
        # it rather than on SMILES.
        .unique(subset=["Molecule_Name"], keep="first")
    )

    print(f"Combined: {all_compounds.height} unique compounds")
    print(
        all_compounds.select(
            pl.col("in_inhibition").sum().alias("inhibition"),
            pl.col("in_tdi").sum().alias("tdi"),
            pl.col("in_test").sum().alias("test"),
        )
    )
    all_compounds
    return all_compounds, test, train_inhib, train_tdi


@app.cell
def _(all_compounds, pl):
    # Overlap between the two training tables: compounds measured in both assay
    # conditions are the ones where a TDI shift can be reasoned about directly.
    _overlap = all_compounds.select(
        (pl.col("in_inhibition") & pl.col("in_tdi")).sum().alias("inhibition_and_tdi"),
        (pl.col("in_inhibition") & ~pl.col("in_tdi")).sum().alias("inhibition_only"),
        (~pl.col("in_inhibition") & pl.col("in_tdi")).sum().alias("tdi_only"),
        (pl.col("in_test") & (pl.col("in_inhibition") | pl.col("in_tdi")))
        .sum()
        .alias("test_also_in_train"),
    )
    _overlap
    return


@app.cell
def _(all_compounds, pl, QUICK, QUICK_N):
    # The working frame for every plot below. In QUICK mode keep all test compounds
    # (they are the scarce, interesting half) and subsample the training set.
    if QUICK:
        _test_rows = all_compounds.filter(pl.col("in_test"))
        _train_rows = all_compounds.filter(~pl.col("in_test")).head(QUICK_N)
        compounds = pl.concat([_train_rows, _test_rows])
    else:
        compounds = all_compounds

    print(f"Working set: {compounds.height} compounds")
    return (compounds,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Endpoint-by-endpoint label structure

    Before any embedding: how sparse is each endpoint, and how much of its range
    sits below the assay's resolution floor?

    CLAUDE.md already records the headline (`pIC50 < 4` is below resolution and the
    metric downweights it; CYP3A4 has 40% of its compounds there, CYP2D6 only 8.6%).
    Recomputing it here keeps this notebook self-contained against a dataset
    amendment — the data has been amended mid-challenge before.
    """
    )
    return


@app.cell
def _(C, np, pl, train_inhib):
    _rows = []
    for _endpoint in C.REGRESSION_ENDPOINTS:
        _values = train_inhib[_endpoint].drop_nulls().to_numpy()
        _low = train_inhib[f"{_endpoint}{C.CONF_LOW_SUFFIX}"]
        _high = train_inhib[f"{_endpoint}{C.CONF_HIGH_SUFFIX}"]
        _width = (_high - _low).drop_nulls().to_numpy()

        _rows.append(
            {
                "endpoint": _endpoint.replace("_pIC50_direct_inhibition", ""),
                "n": int(_values.size),
                "coverage_pct": round(100 * _values.size / train_inhib.height, 1),
                "mean": round(float(_values.mean()), 2),
                "sd": round(float(_values.std()), 2),
                "min": round(float(_values.min()), 2),
                "max": round(float(_values.max()), 2),
                "median_CI_width": round(float(np.median(_width)), 2),
                "below_floor_pct": round(
                    100 * float((_values < C.INACTIVE_PIC50_FLOOR).mean()), 1
                ),
            }
        )

    endpoint_summary = pl.DataFrame(_rows)
    endpoint_summary
    return (endpoint_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The two columns that matter together are `median_CI_width` and
    `below_floor_pct`, and they explain a lot of the baseline table.

    ST-RAE forgives any prediction landing inside the credible interval. CYP2D6 has
    the *tightest* intervals and the *fewest* sub-floor compounds — so it forgives
    least and offers no pool of cheap, near-free predictions. CYP3A4 is the mirror
    image. A meaningful part of CYP3A4's apparent lead is therefore structural
    rather than a claim that its SAR is easier to learn, which is exactly why the
    rest of this notebook looks at the chemistry rather than at more metrics.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Chemical space embeddings

    ECFP6 (radius 3, 4096-bit, count-based) reduced two ways: UMAP on Jaccard
    distance (= 1 − Tanimoto on bit vectors, so neighbourhoods mean what a chemist
    expects) and t-SNE on 50 PCA components. Both are unsupervised and see no
    labels, so fitting them on train and test together is both safe and the point.
    """
    )
    return


@app.cell
def _(CACHE_DIR, CACHE_SUFFIX, compounds, embedding, np, pl):
    # Fingerprints and both embeddings are the slow part of this notebook, so cache
    # the coordinates rather than recomputing them on every notebook reload.
    _cache_path = CACHE_DIR / f"embeddings_{CACHE_SUFFIX}.parquet"

    if _cache_path.exists():
        _cached = pl.read_parquet(_cache_path)
        embedded = compounds.join(_cached, on="Molecule_Name", how="left")
        print(f"Loaded cached embeddings from {_cache_path.name}")
    else:
        _fps = embedding.fingerprint_matrix(compounds)
        print(f"ECFP6 matrix: {_fps.shape}")
        embedded = embedding.add_umap(compounds, _fps)
        embedded = embedding.add_tsne(embedded, _fps)
        embedded.select(
            ["Molecule_Name", "UMAP_x", "UMAP_y", "TSNE_x", "TSNE_y"]
        ).write_parquet(_cache_path)
        print(f"Computed and cached embeddings to {_cache_path.name}")

    embedded
    return (embedded,)


@app.cell
def _(embedded, pl):
    # Literal hex colours assigned in Python, highest priority first. Test compounds
    # must win: they are the smallest group and the one the eye needs to find.
    DATASET_LEGEND = {
        "#c0c0c080": "Inhibition only",
        "#6baed6": "TDI only",
        "#1a3a6b": "Inhibition + TDI",
        "#e15759": "Test set",
    }

    embedded_colored = embedded.with_columns(
        pl.when(pl.col("in_test"))
        .then(pl.lit("#e15759"))
        .when(pl.col("in_inhibition") & pl.col("in_tdi"))
        .then(pl.lit("#1a3a6b"))
        .when(pl.col("in_tdi"))
        .then(pl.lit("#6baed6"))
        .otherwise(pl.lit("#c0c0c080"))
        .alias("dataset_color")
    )
    return DATASET_LEGEND, embedded_colored


@app.cell
def _(DATASET_LEGEND, OUT_DIR, embedded_colored, embedding, mo):
    _fig = embedding.embedding_scatter(
        embedded_colored,
        x_col="UMAP_x",
        y_col="UMAP_y",
        color_col="dataset_color",
        color_legend=DATASET_LEGEND,
        title="UMAP (ECFP6, Jaccard) — dataset membership",
        x_title="UMAP 1",
        y_title="UMAP 2",
        point_size=7,
        save_path=OUT_DIR / "umap_dataset_membership.png",
    )
    mo.center(mo.as_html(_fig))
    return


@app.cell
def _(DATASET_LEGEND, OUT_DIR, embedded_colored, embedding, mo):
    _fig = embedding.embedding_scatter(
        embedded_colored,
        x_col="TSNE_x",
        y_col="TSNE_y",
        color_col="dataset_color",
        color_legend=DATASET_LEGEND,
        title="t-SNE (ECFP6, PCA-50) — dataset membership",
        x_title="t-SNE 1",
        y_title="t-SNE 2",
        point_size=7,
        save_path=OUT_DIR / "tsne_dataset_membership.png",
    )
    mo.center(mo.as_html(_fig))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Potency across chemical space, one panel per endpoint

    The four endpoints do not share rows, so no single scatter can show them all.
    Each panel below colours only the compounds labelled for that endpoint and
    greys out the rest, on identical coordinates — so the panels are directly
    comparable.

    This is the plot that bears on the central question. If an endpoint's potent
    compounds form coherent regions, a fingerprint model has local structure to
    exploit and the weak ST-RAE is a tuning or data-volume problem. If they are
    scattered through a space defined by ECFP similarity, then ECFP similarity is
    not the right notion of neighbourhood for that endpoint, and no amount of
    tuning a tree model on those features will fix it.
    """
    )
    return


@app.cell
def _(C, OUT_DIR, embedded, embedding, mo):
    _fig = embedding.endpoint_grid(
        embedded,
        endpoints=C.REGRESSION_ENDPOINTS,
        titles=[e.replace("_pIC50_direct_inhibition", "") for e in C.REGRESSION_ENDPOINTS],
        x_col="UMAP_x",
        y_col="UMAP_y",
        cmap="viridis",
        point_size=11,
        save_path=OUT_DIR / "umap_by_endpoint.png",
    )
    mo.center(mo.as_html(_fig))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Local potency coherence, quantified

    Eyeballing a UMAP is unreliable — the projection distorts distance, and a
    scatter of 4,900 points hides structure under overplotting. So measure the same
    thing directly in fingerprint space, with no projection involved.

    For each labelled compound, take its *k* nearest neighbours by ECFP4 Tanimoto
    among compounds labelled for the same endpoint, and compare the spread of
    potency within that neighbourhood against the spread across the whole endpoint.
    The ratio is a nearest-neighbour coherence score:

    - **ratio ≪ 1** — neighbours agree; local structure exists for a model to use.
    - **ratio ≈ 1** — knowing a compound's structural neighbours tells you nothing
      about its potency that the global mean did not already tell you.

    This is the numerical form of the question the panels above ask visually, and
    it is where the fingerprint-representation hypothesis either holds or fails.
    """
    )
    return


@app.cell
def _(C, CACHE_DIR, CACHE_SUFFIX, compounds, np, pl, similarity):
    _cache_path = CACHE_DIR / f"neighbour_coherence_{CACHE_SUFFIX}.parquet"

    if _cache_path.exists():
        neighbour_coherence = pl.read_parquet(_cache_path)
        print(f"Loaded cached coherence from {_cache_path.name}")
    else:
        _k = 5
        _rows = []

        for _endpoint in C.REGRESSION_ENDPOINTS:
            _labelled = compounds.filter(pl.col(_endpoint).is_not_null())
            _values = _labelled[_endpoint].to_numpy().astype(float)

            # Full similarity matrix within this endpoint's compounds. Self-similarity
            # is masked so a compound is never its own neighbour.
            _sims = similarity.cross_similarities(_labelled, _labelled, "ecfp4_1k")
            np.fill_diagonal(_sims, -1.0)

            _neighbour_idx = np.argsort(-_sims, axis=1)[:, :_k]
            _neighbour_values = _values[_neighbour_idx]
            _neighbour_sims = np.take_along_axis(_sims, _neighbour_idx, axis=1)

            # Mean absolute deviation of each compound from its neighbours, against
            # the same quantity computed globally (mean |x - y| over random pairs,
            # which for a normal distribution is ~1.13 sd).
            _local_dev = np.abs(_neighbour_values - _values[:, None]).mean()
            _global_dev = np.abs(_values[:, None] - _values[None, :]).mean()

            _rows.append(
                {
                    "endpoint": _endpoint.replace("_pIC50_direct_inhibition", ""),
                    "n": int(_values.size),
                    "mean_nn_similarity": round(float(_neighbour_sims.mean()), 3),
                    "local_dev": round(float(_local_dev), 3),
                    "global_dev": round(float(_global_dev), 3),
                    "coherence_ratio": round(float(_local_dev / _global_dev), 3),
                }
            )

        neighbour_coherence = pl.DataFrame(_rows)
        neighbour_coherence.write_parquet(_cache_path)
        print(f"Computed and cached coherence to {_cache_path.name}")

    neighbour_coherence
    return (neighbour_coherence,)


@app.cell
def _(OUT_DIR, mo, neighbour_coherence):
    def _plot_coherence(df):
        """Bar chart of the coherence ratio, with the 'no local signal' line at 1.0."""
        import matplotlib.pyplot as plt

        _labels = df["endpoint"].to_list()
        _ratios = df["coherence_ratio"].to_list()
        # Deliberately one colour for all four bars. An earlier version split them
        # at 0.9, which drew a green/red distinction between 0.894 and 0.920 that
        # the data does not support -- the finding here is that every endpoint is
        # close to 1.0, not that one of them is meaningfully different.
        _colors = ["#d73027"] * len(_ratios)

        with plt.style.context("seaborn-v0_8-whitegrid"):
            fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)
            _bars = ax.bar(_labels, _ratios, color=_colors, edgecolor="white", linewidth=0.8)
            ax.axhline(1.0, color="#333", linestyle="--", linewidth=1.2)
            ax.text(
                len(_labels) - 0.45, 1.005,
                "no local signal", fontsize=8, color="#333", ha="right", va="bottom",
            )
            for _bar, _ratio in zip(_bars, _ratios, strict=True):
                ax.text(
                    _bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 0.008,
                    f"{_ratio:.3f}", ha="center", va="bottom", fontsize=9,
                )
            ax.set_ylabel("local / global potency deviation", fontsize=10)
            ax.set_ylim(0, max(1.08, max(_ratios) * 1.08))
            ax.set_title(
                "Nearest-neighbour potency coherence (ECFP4, k=5)\n"
                "lower = structural neighbours share potency",
                fontsize=11,
            )
            ax.tick_params(axis="both", labelsize=9)
            fig.tight_layout()
        return fig

    _fig = _plot_coherence(neighbour_coherence)
    _fig.savefig(OUT_DIR / "neighbour_coherence.png", dpi=300, bbox_inches="tight")
    mo.center(mo.as_html(_fig))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Train/test coverage

    The test set is 75 hits × ~10 analogs. Two things follow, and both are
    measurable here.

    First, the analog structure means the test set is *internally* far more similar
    than the training set is — the three-way density plot below should show
    `within_test` shifted right of `within_train`. Second, and more consequentially:
    for every test compound, how close is its nearest training neighbour? A test
    compound at Tanimoto 0.2 from anything in training is an extrapolation, and
    ST-RAE will charge full price for getting it wrong.
    """
    )
    return


@app.cell
def _(CACHE_DIR, CACHE_SUFFIX, compounds, pl, similarity):
    _cache_path = CACHE_DIR / f"similarity_train_test_{CACHE_SUFFIX}.parquet"

    if _cache_path.exists():
        train_test_sims = pl.read_parquet(_cache_path)
        print(f"Loaded cached similarities from {_cache_path.name}")
    else:
        _train = compounds.filter(~pl.col("in_test"))
        _test = compounds.filter(pl.col("in_test"))

        _within_train = similarity.pairwise_similarities(
            _train, "Molecule_Name", "ecfp4_1k"
        ).select("similarity").with_columns(pl.lit("within_train").alias("comparison"))

        _within_test = similarity.pairwise_similarities(
            _test, "Molecule_Name", "ecfp4_1k"
        ).select("similarity").with_columns(pl.lit("within_test").alias("comparison"))

        _cross = similarity.cross_similarities(_train, _test, "ecfp4_1k")
        _train_vs_test = pl.DataFrame(
            {
                "similarity": pl.Series(_cross.ravel(), dtype=pl.Float32),
                "comparison": pl.Series(
                    ["train_vs_test"] * _cross.size, dtype=pl.Utf8
                ),
            }
        )

        train_test_sims = pl.concat([_within_train, _within_test, _train_vs_test])
        train_test_sims.write_parquet(_cache_path)
        print(f"Computed and cached similarities to {_cache_path.name}")

    print(train_test_sims.group_by("comparison").agg(pl.len().alias("n_pairs")))
    return (train_test_sims,)


@app.cell
def _(OUT_DIR, mo, similarity, train_test_sims):
    _fig = similarity.plot_similarity_distributions(
        train_test_sims,
        group_col="comparison",
        group_order=["within_train", "within_test", "train_vs_test"],
        colors=["#4e79a7", "#59a14f", "#e15759"],
        title="ECFP4 Tanimoto — train / test structure",
        save_path=OUT_DIR / "similarity_train_test.png",
    )
    mo.center(mo.as_html(_fig))
    return


@app.cell
def _(pl, train_test_sims):
    # The densities overlap heavily at this scale, so read the upper tail numerically.
    train_test_sims.group_by("comparison").agg(
        pl.len().alias("n_pairs"),
        pl.col("similarity").median().round(3).alias("median"),
        pl.col("similarity").quantile(0.95).round(3).alias("p95"),
        pl.col("similarity").quantile(0.99).round(3).alias("p99"),
        pl.col("similarity").quantile(0.999).round(3).alias("p99.9"),
        pl.col("similarity").max().round(3).alias("max"),
    ).sort("comparison")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Nearest training neighbour, per test compound and per endpoint

    Coverage is endpoint-specific, and this is the part most easily missed. A test
    compound might have a close training neighbour *overall* while that neighbour
    carries no CYP2C9 label — in which case, for CYP2C9 specifically, the compound
    is uncovered. Since the four endpoints have between 1,285 and 2,335 labelled
    compounds, the effective coverage differs a lot between them.
    """
    )
    return


@app.cell
def _(C, CACHE_DIR, CACHE_SUFFIX, compounds, pl, similarity):
    _cache_path = CACHE_DIR / f"test_coverage_{CACHE_SUFFIX}.parquet"

    if _cache_path.exists():
        test_coverage = pl.read_parquet(_cache_path)
        print(f"Loaded cached coverage from {_cache_path.name}")
    else:
        _test = compounds.filter(pl.col("in_test")).select(["Molecule_Name", "SMILES"])
        _frames = []

        for _endpoint in C.REGRESSION_ENDPOINTS:
            _reference = compounds.filter(
                pl.col(_endpoint).is_not_null() & ~pl.col("in_test")
            )
            _nn = similarity.nearest_neighbours(
                _test, _reference, fp_name="ecfp4_1k", carry=[_endpoint]
            )
            _frames.append(
                _nn.rename({f"nn_{_endpoint}": "nn_value"}).with_columns(
                    pl.lit(_endpoint.replace("_pIC50_direct_inhibition", "")).alias(
                        "endpoint"
                    )
                )
            )

        test_coverage = pl.concat(_frames)
        test_coverage.write_parquet(_cache_path)
        print(f"Computed and cached coverage to {_cache_path.name}")

    test_coverage
    return (test_coverage,)


@app.cell
def _(pl, test_coverage):
    # Bands chosen against the ECFP4 rule of thumb: 0.4 is the conventional
    # "structurally similar" line, so below it a model is interpolating between
    # things a chemist would not call analogs.
    coverage_summary = (
        test_coverage.with_columns(
            pl.when(pl.col("nn_similarity") >= 0.6)
            .then(pl.lit("close (>=0.6)"))
            .when(pl.col("nn_similarity") >= 0.4)
            .then(pl.lit("similar (0.4-0.6)"))
            .when(pl.col("nn_similarity") >= 0.3)
            .then(pl.lit("weak (0.3-0.4)"))
            .otherwise(pl.lit("extrapolation (<0.3)"))
            .alias("coverage_band")
        )
        .group_by(["endpoint", "coverage_band"])
        .agg(pl.len().alias("n_test"))
        .with_columns(
            (100 * pl.col("n_test") / pl.col("n_test").sum().over("endpoint"))
            .round(1)
            .alias("pct")
        )
        .sort(["endpoint", "coverage_band"])
    )
    coverage_summary
    return (coverage_summary,)


@app.cell
def _(coverage_summary, pl, test_coverage):
    # Median nearest-neighbour similarity is the single comparable number per
    # endpoint: how reachable is the test set, given only that endpoint's labels?
    _ = coverage_summary
    test_coverage.group_by("endpoint").agg(
        pl.col("nn_similarity").median().round(3).alias("median_nn_sim"),
        pl.col("nn_similarity").quantile(0.25).round(3).alias("q25_nn_sim"),
        (pl.col("nn_similarity") < 0.3).mean().mul(100).round(1).alias("pct_extrapolation"),
        (pl.col("nn_similarity") >= 0.6).mean().mul(100).round(1).alias("pct_close"),
    ).sort("endpoint")
    return


@app.cell
def _(OUT_DIR, coverage_summary, mo):
    def _plot_coverage(df):
        """Stacked coverage bands per endpoint."""
        import matplotlib.pyplot as plt
        import numpy as np

        _order = [
            "close (>=0.6)",
            "similar (0.4-0.6)",
            "weak (0.3-0.4)",
            "extrapolation (<0.3)",
        ]
        _colors = ["#1a9850", "#a6d96a", "#fdae61", "#d73027"]
        _endpoints = sorted(df["endpoint"].unique().to_list())

        _lookup = {
            (r["endpoint"], r["coverage_band"]): r["pct"] for r in df.iter_rows(named=True)
        }

        with plt.style.context("seaborn-v0_8-whitegrid"):
            fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
            _bottom = np.zeros(len(_endpoints))
            for _band, _color in zip(_order, _colors, strict=True):
                _values = np.array([_lookup.get((e, _band), 0.0) for e in _endpoints])
                ax.bar(
                    _endpoints, _values, bottom=_bottom, label=_band,
                    color=_color, edgecolor="white", linewidth=0.6,
                )
                for _x, (_v, _b) in enumerate(zip(_values, _bottom, strict=True)):
                    if _v >= 4:
                        ax.text(
                            _x, _b + _v / 2, f"{_v:.0f}%",
                            ha="center", va="center", fontsize=8,
                        )
                _bottom += _values

            ax.set_ylabel("% of the 750 test compounds", fontsize=10)
            ax.set_ylim(0, 100)
            ax.set_title(
                "Nearest training neighbour of each test compound, by endpoint\n"
                "(ECFP4 Tanimoto, reference restricted to that endpoint's labelled compounds)",
                fontsize=11,
            )
            ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncols=4)
            ax.tick_params(axis="both", labelsize=9)
            fig.tight_layout()
        return fig

    _fig = _plot_coverage(coverage_summary)
    _fig.savefig(OUT_DIR / "test_coverage_bands.png", dpi=300, bbox_inches="tight")
    mo.center(mo.as_html(_fig))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### What is the nearest training neighbour actually measured on?

    The bands above answer "how close is the closest analog". This section answers
    the follow-up, which is the PXR 1d question ported to CYP's six tracks: having
    found that analog, **what do we know about it?**

    The distinction matters because the two are computed differently. Above, each
    endpoint gets its own neighbour drawn only from compounds labelled for that
    endpoint — four endpoints, four different analogs per test compound. Here there
    is **one global neighbour** per test compound, taken across the whole training
    set, and the question is how many of the six tracks (four pIC50 + two TDI) that
    single analog carries.

    That number is the real information content of a neighbour. Two analogs at
    identical Tanimoto 0.55 are not equally useful if one is measured on all six
    tracks and the other only on CYP3A4 — and since the four endpoints do not share
    rows (CLAUDE.md), the second case is common rather than exotic.
    """
    )
    return


@app.cell
def _(C, CACHE_DIR, CACHE_SUFFIX, compounds, pl, similarity):
    _cache_path = CACHE_DIR / f"nn_label_profile_{CACHE_SUFFIX}.parquet"

    if _cache_path.exists():
        nn_profile = pl.read_parquet(_cache_path)
        print(f"Loaded cached NN profile from {_cache_path.name}")
    else:
        _test = compounds.filter(pl.col("in_test")).select(["Molecule_Name", "SMILES"])
        # The reference is every non-test compound that carries at least one label,
        # in either track -- the neighbour is global, not per endpoint.
        _reference = compounds.filter(
            ~pl.col("in_test") & (pl.col("in_inhibition") | pl.col("in_tdi"))
        )

        nn_profile = similarity.neighbour_label_profile(
            _test,
            _reference,
            regression_endpoints=C.REGRESSION_ENDPOINTS,
            classification_endpoints=C.CLASSIFICATION_ENDPOINTS,
        )
        nn_profile.write_parquet(_cache_path)
        print(f"Computed and cached NN profile to {_cache_path.name}")

    print(f"{nn_profile.height} test compounds profiled against their global NN")
    nn_profile
    return (nn_profile,)


@app.cell
def _(nn_profile, pl):
    # How many of the six tracks does the single nearest neighbour carry?
    tracks_measured = (
        nn_profile.group_by("n_tracks_measured")
        .agg(
            pl.len().alias("n_test_compounds"),
            pl.col("nn_similarity").median().round(3).alias("median_nn_sim"),
        )
        .with_columns(
            (100 * pl.col("n_test_compounds") / pl.col("n_test_compounds").sum())
            .round(1)
            .alias("pct")
        )
        .sort("n_tracks_measured", descending=True)
    )
    tracks_measured
    return (tracks_measured,)


@app.cell
def _(C, nn_profile, pl):
    # Per track: is the nearest neighbour measured on it at all, and if so, how
    # potent? "not tested" is the share of test compounds whose closest analog
    # carries no label on that track -- a blind spot the coverage-band plot cannot
    # show, because there the reference was already filtered to labelled compounds.
    _rows = []
    for _endpoint in [*C.REGRESSION_ENDPOINTS, *C.CLASSIFICATION_ENDPOINTS]:
        _is_tdi = _endpoint in C.CLASSIFICATION_ENDPOINTS
        _short = _endpoint.replace("_pIC50_direct_inhibition", "").replace(
            "_is_TDI", " (TDI)"
        )
        _measured = nn_profile[f"nn_{_endpoint}"].is_not_null().sum()

        _row = {
            "track": _short,
            "nn_measured": int(_measured),
            "nn_measured_pct": round(100 * _measured / nn_profile.height, 1),
        }
        if _is_tdi:
            # For a boolean track the meaningful split is positive vs negative.
            _positive = nn_profile.filter(pl.col(f"nn_{_endpoint}")).height
            _row["detail"] = (
                f"{_positive} TDI+ / {int(_measured) - _positive} TDI-"
                if _measured
                else "-"
            )
        else:
            _potent = nn_profile.filter(pl.col(f"nn_{_endpoint}") >= 6.0).height
            _row["detail"] = f"{_potent} potent (>=6)"
        _rows.append(_row)

    track_coverage = pl.DataFrame(_rows)
    track_coverage
    return (track_coverage,)


@app.cell
def _(C, nn_profile, pl):
    # The full potency-band breakdown, per regression track.
    _frames = []
    for _endpoint in C.REGRESSION_ENDPOINTS:
        _short = _endpoint.replace("_pIC50_direct_inhibition", "")
        _frames.append(
            nn_profile.group_by(f"band_{_endpoint}")
            .agg(pl.len().alias("n_test_compounds"))
            .rename({f"band_{_endpoint}": "band"})
            .with_columns(pl.lit(_short).alias("track"))
        )

    nn_potency_bands = (
        pl.concat(_frames)
        .with_columns(
            (100 * pl.col("n_test_compounds") / nn_profile.height).round(1).alias("pct")
        )
        .pivot(on="band", index="track", values="pct")
        .sort("track")
    )

    print("% of the 750 test compounds, by their global NN's potency on each track")
    nn_potency_bands
    return (nn_potency_bands,)


@app.cell
def _(C, OUT_DIR, mo, nn_profile, tracks_measured):
    def _plot_nn_activity(profile, tracks_df):
        """Two panels: how many tracks the NN carries, and its potency per track."""
        import matplotlib.pyplot as plt
        import numpy as np
        import polars as pl

        _BANDS = [
            ("potent (>=6)", "#1a9850"),
            ("moderate (5-6)", "#a6d96a"),
            ("weak (4-5)", "#fdae61"),
            ("inactive (<4)", "#d73027"),
            ("not tested", "#cccccc"),
        ]

        with plt.style.context("seaborn-v0_8-whitegrid"):
            fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), dpi=300)

            # Left: distribution of how many of the six tracks the NN is measured on.
            _counts = tracks_df.sort("n_tracks_measured")
            _labels = [str(v) for v in _counts["n_tracks_measured"].to_list()]
            _values = _counts["pct"].to_list()
            _bars = axes[0].bar(
                _labels, _values, color="#4e79a7", edgecolor="white", linewidth=0.8
            )
            for _bar, _pct, _n in zip(
                _bars, _values, _counts["n_test_compounds"].to_list(), strict=True
            ):
                axes[0].text(
                    _bar.get_x() + _bar.get_width() / 2,
                    _bar.get_height() + 0.6,
                    f"{_pct}%\n({_n})",
                    ha="center", va="bottom", fontsize=8,
                )
            axes[0].set_xlabel(
                "tracks the nearest neighbour is measured on (of 6)", fontsize=10
            )
            axes[0].set_ylabel("% of the 750 test compounds", fontsize=10)
            axes[0].set_title(
                "How much does the closest analog tell us?", fontsize=11, pad=8
            )
            axes[0].set_ylim(0, max(_values) * 1.25)

            # Right: stacked potency bands per regression track.
            _tracks = [e.replace("_pIC50_direct_inhibition", "") for e in C.REGRESSION_ENDPOINTS]
            _bottom = np.zeros(len(_tracks))
            for _band, _color in _BANDS:
                _shares = []
                for _endpoint in C.REGRESSION_ENDPOINTS:
                    _n = profile.filter(pl.col(f"band_{_endpoint}") == _band).height
                    _shares.append(100 * _n / profile.height)
                _shares = np.array(_shares)
                axes[1].bar(
                    _tracks, _shares, bottom=_bottom, label=_band,
                    color=_color, edgecolor="white", linewidth=0.6,
                )
                for _x, (_v, _b) in enumerate(zip(_shares, _bottom, strict=True)):
                    if _v >= 5:
                        axes[1].text(
                            _x, _b + _v / 2, f"{_v:.0f}%",
                            ha="center", va="center", fontsize=8,
                        )
                _bottom += _shares

            axes[1].set_ylabel("% of the 750 test compounds", fontsize=10)
            axes[1].set_ylim(0, 100)
            axes[1].set_title(
                "Potency of that same analog, per track", fontsize=11, pad=8
            )
            axes[1].legend(
                fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncols=5
            )

            for _ax in axes:
                _ax.tick_params(axis="both", labelsize=9)
            fig.tight_layout()
        return fig

    _fig = _plot_nn_activity(nn_profile, tracks_measured)
    _fig.savefig(OUT_DIR / "nn_activity_profile.png", dpi=300, bbox_inches="tight")
    mo.center(mo.as_html(_fig))
    return


@app.cell
def _(nn_profile, pl):
    # Does a better-measured neighbour also tend to be a closer one? If the two are
    # independent, then track coverage is a genuinely separate axis of difficulty
    # from structural distance, and the coverage-band plot above was only half the
    # picture.
    nn_profile.group_by("n_regression_tracks").agg(
        pl.len().alias("n_test_compounds"),
        pl.col("nn_similarity").median().round(3).alias("median_nn_sim"),
        pl.col("nn_similarity").quantile(0.25).round(3).alias("q25_nn_sim"),
    ).sort("n_regression_tracks", descending=True)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    #### How many distinct training compounds are the test set's neighbours?

    The potency numbers above look anomalous at first read — the neighbours are
    enriched for potent compounds several times over the training pool's own base
    rate. That is not an artifact. It is the challenge's stated test-set design
    (75 hits × ~10 analogs) becoming visible: if the test compounds are analogs of
    hits, then their nearest training neighbours *are* those hits, and hits are
    potent by selection.

    The check is to count distinct neighbours. If 750 test compounds map onto far
    fewer than 750 training compounds, with each attracting a cluster of roughly ten,
    the design is confirmed directly.
    """
    )
    return


@app.cell
def _(nn_profile, pl):
    nn_cluster_sizes = (
        nn_profile.group_by("nn_Molecule_Name")
        .agg(pl.len().alias("n_test_compounds"))
        .sort("n_test_compounds", descending=True)
    )

    print(
        f"{nn_cluster_sizes.height} distinct training compounds are the nearest "
        f"neighbour of {nn_profile.height} test compounds"
    )
    print(
        f"median cluster size {nn_cluster_sizes['n_test_compounds'].median():.0f}, "
        f"max {nn_cluster_sizes['n_test_compounds'].max()}, "
        f"{nn_cluster_sizes.filter(pl.col('n_test_compounds') >= 5).height} neighbours "
        "attract >=5 test compounds each"
    )
    nn_cluster_sizes.head(15)
    return (nn_cluster_sizes,)


@app.cell
def _(C, nn_profile, pl, train_inhib):
    # Potency of the neighbour set against the training pool's own base rate. A
    # large gap is the numerical signature of "the neighbours are the hits".
    _rows = []
    for _endpoint in C.REGRESSION_ENDPOINTS:
        _pool = train_inhib[_endpoint].drop_nulls()
        _nn = nn_profile[f"nn_{_endpoint}"].drop_nulls()
        _rows.append(
            {
                "track": _endpoint.replace("_pIC50_direct_inhibition", ""),
                "train_pool_pct_potent": round(100 * float((_pool >= 6.0).mean()), 1),
                "nn_pct_potent": round(100 * float((_nn >= 6.0).mean()), 1),
                "enrichment_x": round(
                    float((_nn >= 6.0).mean() / (_pool >= 6.0).mean()), 1
                )
                if float((_pool >= 6.0).mean()) > 0
                else None,
            }
        )

    nn_potency_enrichment = pl.DataFrame(_rows)
    nn_potency_enrichment
    return (nn_potency_enrichment,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    This changes how the coverage numbers should be read, in two directions at once.

    **Favourable:** the test set is not scattered novel chemistry. It clusters
    tightly around a small, well-characterised, potent subset of the training data —
    the compounds most likely to have been measured on every track.

    **Unfavourable, and the sharper point:** because those neighbours are potent by
    selection, *nearest-neighbour information is systematically biased upward* for
    the test set. A model that leans on local structure will inherit that bias and
    over-predict potency on test analogs that are in fact weaker than the hit they
    resemble. That is the same regression-to-the-mean failure PXR flagged, running in
    the opposite direction — and it is precisely what ST-RAE charges for, since the
    test set's tight-interval potent compounds are where errors are least forgiven.

    Worth checking on the submission model with `evaluation.bias_by_potency_bin`,
    which CLAUDE.md already prescribes per run. The prediction from this section is a
    positive bias concentrated on test compounds whose NN is potent — a sharper,
    more testable claim than "check for shrinkage".
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Three things to take from this section.

    **The share of test compounds whose closest analog is measured on all six
    tracks** is the best single summary of how much context the training set gives.
    Where that share is low, most test compounds are being predicted from an analog
    that was itself only partially characterised.

    **The "not tested" slice in the right panel is a blind spot the earlier
    coverage-band plot could not show.** There, the reference set was pre-filtered to
    compounds labelled for the endpoint in question, so a neighbour always had a
    value by construction. Here the neighbour is chosen on structure alone, which is
    what a model actually does — and the gap between the two is the share of test
    compounds whose nearest analog carries no information on the track being
    predicted.

    **If median NN similarity is flat across the `n_regression_tracks` groups**, then
    how well-measured a neighbour is, is independent of how close it is. That makes
    label coverage a second axis of difficulty rather than a restatement of the
    structural one — and it is the axis that public data (PLAN.md priority 3) would
    move, since ChEMBL and PubChem carry all four isoforms on overlapping compound
    sets.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Interactive — test compounds coloured by nearest-neighbour similarity

    The test set projected on its own UMAP, coloured by how close its nearest
    CYP3A4-labelled training neighbour is. Hover a point to see the test compound;
    the panel names its nearest training analog and that analog's measured potency.

    Use this to sanity-check the extrapolation tail: the dark points are the
    compounds where any model is guessing.
    """
    )
    return


@app.cell
def _(embedding, pl, test_coverage):
    # A UMAP of the test set alone -- a separate projection from the global one, so
    # that 750 compounds get the full resolution of the plot rather than being
    # squeezed into one corner of a 5,600-compound embedding.
    _test_3a4 = test_coverage.filter(pl.col("endpoint") == "CYP3A4")
    test_embedded = embedding.add_umap(_test_3a4, smiles_col="SMILES")
    return (test_embedded,)


@app.cell
def _(interactive, mo, test_embedded):
    _chart = interactive.scatter(
        test_embedded,
        x="UMAP_x",
        y="UMAP_y",
        key_fields=["Molecule_Name"],
        color="nn_similarity",
        color_title="NN similarity",
        tooltip=[
            ("Molecule_Name", "Test compound", None),
            ("nn_similarity", "NN Tanimoto", ".3f"),
            ("nn_Molecule_Name", "Nearest train", None),
            ("nn_value", "NN CYP3A4 pIC50", ".2f"),
        ],
        title="Test set UMAP — nearest CYP3A4 training neighbour",
        x_title="UMAP 1",
        y_title="UMAP 2",
    )
    coverage_chart = mo.ui.altair_chart(_chart)
    return (coverage_chart,)


@app.cell
def _(coverage_chart, interactive, mo, test_embedded):
    _row = interactive.selected_row(coverage_chart, test_embedded, ["Molecule_Name"])
    _panel = interactive.structure_panel(
        _row,
        fields=[
            ("nn_similarity", "NN Tanimoto", ".3f"),
            ("nn_Molecule_Name", "Nearest train", ""),
            ("nn_value", "NN CYP3A4 pIC50", ".2f"),
        ],
        placeholder="Hover a test compound to see its structure",
    )
    mo.hstack([coverage_chart, _panel], align="start")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## What does "similar" mean here?

    A Tanimoto value is meaningless without naming the fingerprint that produced
    it. PXR's 1c made this point with exactly this plot, and the conclusion
    transfers: MACCS (166 structural keys) reports systematically higher
    similarities than ECFP6 (4096 hashed circular bits) on the same compound pairs,
    so a single threshold cannot serve both.

    The working thresholds — MACCS ≥ 0.8, ECFP4 ≥ 0.4 — are rules of thumb carried
    from PXR. The percentile table below checks them against *this* dataset before
    the cliff analysis relies on them: if a threshold sits above a fingerprint's
    p99.9, it selects almost nothing and any cliff count from it is noise.
    """
    )
    return


@app.cell
def _(CACHE_DIR, CACHE_SUFFIX, compounds, pl, similarity):
    _cache_path = CACHE_DIR / f"fp_similarity_{CACHE_SUFFIX}.parquet"

    if _cache_path.exists():
        fp_similarities = pl.read_parquet(_cache_path)
        print(f"Loaded cached fingerprint similarities from {_cache_path.name}")
    else:
        # Restricted to compounds with at least one regression label: the cliff
        # analysis downstream needs activity values, so unlabelled pairs are dead
        # weight in a quadratic computation.
        _labelled = compounds.filter(pl.col("in_inhibition"))
        fp_similarities = pl.concat(
            [
                similarity.pairwise_similarities(_labelled, "Molecule_Name", _fp)
                for _fp in ("maccs", "ecfp4_1k", "ecfp4_4k", "ecfp6")
            ]
        )
        fp_similarities.write_parquet(_cache_path)
        print(f"Computed and cached fingerprint similarities to {_cache_path.name}")

    print(f"{fp_similarities.height:,} rows across 4 fingerprints")
    return (fp_similarities,)


@app.cell
def _(OUT_DIR, fp_similarities, mo, similarity):
    _fig = similarity.plot_similarity_distributions(
        fp_similarities,
        group_col="fingerprint",
        group_order=["maccs", "ecfp4_1k", "ecfp4_4k", "ecfp6"],
        colors=["#4e79a7", "#f28e2b", "#e15759", "#59a14f"],
        title="Pairwise Tanimoto by fingerprint — inhibition training compounds",
        save_path=OUT_DIR / "similarity_by_fingerprint.png",
    )
    mo.center(mo.as_html(_fig))
    return


@app.cell
def _(fp_similarities, similarity):
    similarity.similarity_percentiles(fp_similarities)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Read the `p99`/`p99.9` columns against `similarity.CLIFF_SIM_THRESHOLDS`. Where
    the threshold sits between p99 and the max, it is selecting a genuine tail of
    structurally related pairs — which is what it is for. Where it exceeds p99.9,
    the cliff counts below rest on a handful of pairs and should not be compared
    across endpoints.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Activity cliffs

    An activity cliff is a structurally similar pair whose potencies differ by ≥2
    log units. Cliffs are the direct counterexample to the assumption every
    fingerprint model makes, so a high cliff rate on an endpoint is evidence that
    the representation — not the model — is the binding constraint.

    Rates are reported per endpoint as a fraction of *labelled similar pairs*,
    which is the only comparable form: the endpoints differ in how many compounds
    they label, so raw cliff counts mostly measure label density.
    """
    )
    return


@app.cell
def _(C, compounds, fp_similarities, pl, similarity):
    _ecfp4 = fp_similarities.filter(pl.col("fingerprint") == "ecfp4_1k")
    cliff_rates_ecfp4 = similarity.cliff_summary(
        _ecfp4, compounds, C.REGRESSION_ENDPOINTS
    ).with_columns(
        pl.col("endpoint").str.replace("_pIC50_direct_inhibition", "")
    )
    cliff_rates_ecfp4
    return (cliff_rates_ecfp4,)


@app.cell
def _(C, compounds, fp_similarities, pl, similarity):
    _maccs = fp_similarities.filter(pl.col("fingerprint") == "maccs")
    cliff_rates_maccs = similarity.cliff_summary(
        _maccs, compounds, C.REGRESSION_ENDPOINTS
    ).with_columns(
        pl.col("endpoint").str.replace("_pIC50_direct_inhibition", "")
    )
    cliff_rates_maccs
    return (cliff_rates_maccs,)


@app.cell
def _(OUT_DIR, cliff_rates_ecfp4, cliff_rates_maccs, mo):
    def _plot_cliff_rates(ecfp4_df, maccs_df):
        """Cliff rate per endpoint, both fingerprints side by side."""
        import matplotlib.pyplot as plt
        import numpy as np

        _endpoints = ecfp4_df["endpoint"].to_list()
        _x = np.arange(len(_endpoints))
        _width = 0.38

        _maccs_lookup = {
            r["endpoint"]: r["cliff_rate_pct"] for r in maccs_df.iter_rows(named=True)
        }

        with plt.style.context("seaborn-v0_8-whitegrid"):
            fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
            _b1 = ax.bar(
                _x - _width / 2, ecfp4_df["cliff_rate_pct"].to_list(), _width,
                label="ECFP4 >= 0.4", color="#f28e2b", edgecolor="white",
            )
            _b2 = ax.bar(
                _x + _width / 2, [_maccs_lookup.get(e, 0.0) for e in _endpoints], _width,
                label="MACCS >= 0.8", color="#4e79a7", edgecolor="white",
            )
            for _bars in (_b1, _b2):
                for _bar in _bars:
                    ax.text(
                        _bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 0.2,
                        f"{_bar.get_height():.1f}", ha="center", va="bottom", fontsize=8,
                    )
            ax.set_xticks(_x, _endpoints)
            ax.set_ylabel("% of similar labelled pairs that are cliffs", fontsize=10)
            ax.set_title(
                "Activity cliff rate per endpoint (|delta pIC50| >= 2)", fontsize=11
            )
            ax.legend(fontsize=9)
            ax.tick_params(axis="both", labelsize=9)
            fig.tight_layout()
        return fig

    _fig = _plot_cliff_rates(cliff_rates_ecfp4, cliff_rates_maccs)
    _fig.savefig(OUT_DIR / "cliff_rates_by_endpoint.png", dpi=300, bbox_inches="tight")
    mo.center(mo.as_html(_fig))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Do the two fingerprints agree on which pairs are cliffs?

    PXR found they largely do not — a pair flagged by MACCS is often not flagged by
    ECFP4, because the two disagree about what "similar" means long before activity
    enters the picture. That makes "activity cliff" a fingerprint-relative label
    rather than a property of the chemistry, and it is worth confirming here before
    drawing conclusions from cliff counts.
    """
    )
    return


@app.cell
def _(OUT_DIR, compounds, fp_similarities, mo, pl, similarity):
    _CLIFF_ENDPOINT = "CYP3A4_pIC50_direct_inhibition"

    _ecfp4_cliffs = similarity.activity_cliffs(
        fp_similarities.filter(pl.col("fingerprint") == "ecfp4_1k"),
        compounds,
        _CLIFF_ENDPOINT,
    )
    _maccs_cliffs = similarity.activity_cliffs(
        fp_similarities.filter(pl.col("fingerprint") == "maccs"),
        compounds,
        _CLIFF_ENDPOINT,
    )

    _fig = similarity.plot_cliff_venn(
        {
            "ECFP4 >= 0.4": similarity.cliff_pair_set(_ecfp4_cliffs),
            "MACCS >= 0.8": similarity.cliff_pair_set(_maccs_cliffs),
        },
        title="CYP3A4 activity cliffs — fingerprint agreement\n(|delta pIC50| >= 2)",
        save_path=OUT_DIR / "cliff_venn_cyp3a4.png",
    )
    mo.center(mo.as_html(_fig))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Interactive — browse the cliffs

    Every ECFP4 cliff for the selected endpoint, plotted as similarity against
    |Δ pIC50|. The top-right corner is the worst case for a fingerprint model:
    near-identical structures with two-plus orders of magnitude between them.

    Hover a point to see both molecules drawn together. The switcher changes
    endpoint, so the same view can be read for the endpoint that actually learns
    (CYP3A4) and the one that does not (CYP2D6).
    """
    )
    return


@app.cell
def _(C, mo):
    endpoint_selector = mo.ui.dropdown(
        options={e.replace("_pIC50_direct_inhibition", ""): e for e in C.REGRESSION_ENDPOINTS},
        value="CYP2D6",
        label="Endpoint",
    )
    endpoint_selector
    return (endpoint_selector,)


@app.cell
def _(compounds, endpoint_selector, fp_similarities, pl, similarity):
    _endpoint = endpoint_selector.value

    _structures = compounds.select(["Molecule_Name", "SMILES"])

    cliff_browse_df = (
        similarity.activity_cliffs(
            fp_similarities.filter(pl.col("fingerprint") == "ecfp4_1k"),
            compounds,
            _endpoint,
        )
        .join(
            _structures.rename({"Molecule_Name": "ID1", "SMILES": "smiles1"}),
            on="ID1",
            how="left",
        )
        .join(
            _structures.rename({"Molecule_Name": "ID2", "SMILES": "smiles2"}),
            on="ID2",
            how="left",
        )
        .with_columns(
            pl.max_horizontal("value_1", "value_2").alias("max_value"),
        )
    )

    print(f"{cliff_browse_df.height} ECFP4 cliffs for {_endpoint}")
    cliff_browse_df
    return (cliff_browse_df,)


@app.cell
def _(cliff_browse_df, endpoint_selector, interactive, mo):
    _label = endpoint_selector.value.replace("_pIC50_direct_inhibition", "")

    if cliff_browse_df.height == 0:
        cliff_chart = None
        _out = mo.md(f"**No ECFP4 cliffs found for {_label}.**")
    else:
        cliff_chart = mo.ui.altair_chart(
            interactive.scatter(
                cliff_browse_df,
                x="similarity",
                y="delta",
                key_fields=["ID1", "ID2"],
                color="max_value",
                color_title="max pIC50",
                tooltip=[
                    ("ID1", "Molecule 1", None),
                    ("ID2", "Molecule 2", None),
                    ("similarity", "Tanimoto", ".3f"),
                    ("value_1", "pIC50 1", ".2f"),
                    ("value_2", "pIC50 2", ".2f"),
                    ("delta", "|delta|", ".2f"),
                ],
                title=f"{_label} activity cliffs — ECFP4 Tanimoto >= 0.4, |delta| >= 2",
                x_title="Tanimoto similarity (ECFP4)",
                y_title="|delta pIC50|",
            )
        )
        _out = cliff_chart
    _out
    return (cliff_chart,)


@app.cell
def _(cliff_browse_df, cliff_chart, interactive, mo):
    if cliff_chart is None:
        _view = mo.md("")
    else:
        _row = interactive.selected_row(cliff_chart, cliff_browse_df, ["ID1", "ID2"])
        _view = mo.hstack(
            [
                cliff_chart,
                interactive.pair_panel(
                    _row,
                    extra=[
                        ("similarity", "Tanimoto", ".3f"),
                        ("delta", "|delta pIC50|", ".2f"),
                    ],
                ),
            ],
            align="start",
        )
    _view
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Matched molecular pairs

    Similarity thresholds are a blunt way to say "these two compounds differ by one
    change". MMPs say it precisely: two molecules sharing a core, differing by one
    transformation that `mmpdb` names explicitly (`[*:1]C1CCCO1>>[*:1]c1ccco1`).

    That precision is what makes MMP cliffs *actionable* where threshold cliffs are
    only diagnostic — the output reads as SAR ("saturating this ring costs 2 logs
    on 2D6") rather than as a list of pair IDs. It is also the honest version of
    the analysis, since it does not inherit a fingerprint's opinion about
    similarity.

    `mmpdb fragment` is the slow step (minutes over ~5.6k compounds) and is cached
    under `experiments/02_chemical_space/cache/`.
    """
    )
    return


@app.cell
def _(CACHE_DIR, CACHE_SUFFIX, compounds, mmp, pl):
    _cache_path = CACHE_DIR / f"mmp_annotated_{CACHE_SUFFIX}.parquet"

    if _cache_path.exists():
        mmp_pairs = pl.read_parquet(_cache_path)
        print(f"Loaded cached MMP pairs from {_cache_path.name}")
    else:
        _raw = mmp.build_mmp_table(
            compounds, CACHE_DIR, stem=f"compounds_{CACHE_SUFFIX}"
        )
        print(f"mmpdb produced {_raw.height:,} raw pairs")
        # core_transform_ratio < 1.0 keeps pairs that share more than they change --
        # the loosest defensible MMP definition, which PXR used on a comparably
        # diverse set. See mmp.annotate_mmp_sizes for when to tighten it.
        mmp_pairs = mmp.annotate_mmp_sizes(_raw, max_core_ratio=1.0)
        mmp_pairs.write_parquet(_cache_path)
        print(f"Computed and cached MMP pairs to {_cache_path.name}")

    print(f"{mmp_pairs.height:,} pairs after core/transform filtering")
    mmp_pairs
    return (mmp_pairs,)


@app.cell
def _(C, compounds, mmp, mmp_pairs, pl):
    # MMP cliff counts per endpoint, the transformation-based analogue of the
    # threshold-based table above.
    _rows = []
    for _endpoint in C.REGRESSION_ENDPOINTS:
        _labelled = compounds.filter(pl.col(_endpoint).is_not_null())
        _names = set(_labelled["Molecule_Name"].to_list())
        _both_labelled = mmp_pairs.filter(
            pl.col("ID1").is_in(_names) & pl.col("ID2").is_in(_names)
        )
        _cliffs = mmp.mmp_cliffs(mmp_pairs, compounds, _endpoint)
        _rows.append(
            {
                "endpoint": _endpoint.replace("_pIC50_direct_inhibition", ""),
                "labelled_mmp_pairs": _both_labelled.height,
                "mmp_cliffs": _cliffs.height,
                "cliff_rate_pct": round(
                    100 * _cliffs.height / _both_labelled.height, 2
                )
                if _both_labelled.height
                else 0.0,
                "max_delta": round(float(_cliffs["delta"].max()), 2)
                if _cliffs.height
                else 0.0,
            }
        )

    mmp_cliff_summary = pl.DataFrame(_rows)
    mmp_cliff_summary
    return (mmp_cliff_summary,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Which transformations move potency most?

    Grouping MMP pairs by their transformation and ranking by mean |Δ pIC50| turns
    the pair list into readable SAR. A transformation appearing several times with a
    consistently large delta is a real structural determinant of potency for that
    isoform; one appearing once is an anecdote.
    """
    )
    return


@app.cell
def _(compounds, endpoint_selector, mmp, mmp_pairs, pl):
    _endpoint = endpoint_selector.value

    top_transforms = (
        mmp.mmp_cliffs(mmp_pairs, compounds, _endpoint, delta_threshold=0.0)
        .group_by("transform")
        .agg(
            pl.len().alias("n_pairs"),
            pl.col("delta").mean().round(2).alias("mean_delta"),
            pl.col("delta").max().round(2).alias("max_delta"),
        )
        # Two occurrences is a low bar, but on this dataset most transformations
        # occur once -- requiring more leaves an empty table.
        .filter(pl.col("n_pairs") >= 2)
        .sort("mean_delta", descending=True)
        .head(20)
    )

    print(f"Top transformations by mean |delta| for {_endpoint}")
    top_transforms
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### MMP network

    Nodes are compounds with at least one MMP partner; edges are the pairs. Node
    colour is CYP3A4 potency, so contiguous coloured regions are analog series
    where the SAR is smooth, and adjacent red/green nodes are cliffs.

    Disconnected components are laid out separately and packed into a grid — an MMP
    graph is mostly small components, and laying it out as one graph wastes the
    canvas.
    """
    )
    return


@app.cell
def _(OUT_DIR, compounds, mmp, mmp_pairs, mo, pl):
    _ENDPOINT = "CYP3A4_pIC50_direct_inhibition"

    # Restrict to compounds that both have a CYP3A4 label and an MMP partner --
    # colouring the network by an endpoint means nodes without that label would be
    # drawn as holes.
    _connected = set(mmp_pairs["ID1"].to_list()) | set(mmp_pairs["ID2"].to_list())
    _nodes = compounds.filter(
        pl.col("Molecule_Name").is_in(_connected) & pl.col(_ENDPOINT).is_not_null()
    )

    print(f"Network: {_nodes.height} nodes")

    _fig = mmp.plot_network(
        _nodes,
        mmp_pairs,
        property_col=_ENDPOINT,
        property_title="CYP3A4 pIC50",
        node_size=14,
        title="MMP network — CYP3A4-labelled compounds, coloured by potency",
        save_path=OUT_DIR / "mmp_network_cyp3a4.png",
    )
    mo.center(mo.as_html(_fig))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Scaffold analysis

    `cv.murcko_scaffold` gives one scaffold per molecule and is the right tool for
    grouping CV folds. It is the wrong tool for asking about coverage: against a
    test set of 75 hits × ~10 analogs, most test compounds will be Murcko-novel, and
    that answer is both predictable and unhelpful.

    So decompose into a scaffold *network* instead — every ring system, every
    linked pair, and the full multi-ring core — and ask which of those the training
    set covers. A test compound can be Murcko-novel while every ring system in it
    is well represented in training, and that is a completely different modelling
    situation from genuine novelty.
    """
    )
    return


@app.cell
def _(CACHE_DIR, CACHE_SUFFIX, compounds, pl, scaffolds):
    _cache_path = CACHE_DIR / f"scaffolds_{CACHE_SUFFIX}.parquet"

    if _cache_path.exists():
        scaffold_long = pl.read_parquet(_cache_path)
        print(f"Loaded cached scaffolds from {_cache_path.name}")
    else:
        scaffold_long = scaffolds.decompose(compounds["SMILES"].to_list())
        scaffold_long.write_parquet(_cache_path)
        print(f"Computed and cached scaffolds to {_cache_path.name}")

    print(scaffold_long["scaffold_type"].value_counts())
    print(f"{scaffold_long['scaffold_smiles'].n_unique():,} unique scaffolds")
    scaffold_long
    return (scaffold_long,)


@app.cell
def _(compounds, scaffold_long, scaffolds):
    scaffold_table = scaffolds.scaffold_counts(
        scaffold_long,
        membership=compounds.select(
            ["SMILES", "in_inhibition", "in_tdi", "in_test"]
        ),
        flag_columns=["in_inhibition", "in_tdi", "in_test"],
    )
    scaffold_table
    return (scaffold_table,)


@app.cell
def _(pl, scaffold_table):
    # The coverage question, per scaffold level: of the scaffolds present in the
    # test set, how many also appear in the training set?
    scaffold_table.filter(pl.col("n_in_test") > 0).group_by("scaffold_type").agg(
        pl.len().alias("test_scaffolds"),
        (pl.col("n_in_inhibition") > 0).sum().alias("also_in_inhibition"),
    ).with_columns(
        (100 * pl.col("also_in_inhibition") / pl.col("test_scaffolds"))
        .round(1)
        .alias("covered_pct")
    ).sort("scaffold_type")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Read this table from `ring_system` down to `full_scaffold`. High coverage at
    the ring-system level with low coverage at the full-scaffold level is the
    signature of the challenge's stated design: the test compounds are built from
    rings the training set has seen, combined in ways it has not. That is the
    favourable case — the model has seen the pieces — and it is a materially
    different claim from "the test set is novel chemistry".
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Scaffolds where potency swings most

    Per-scaffold activity spread, for scaffolds with at least three labelled
    molecules. A large spread means potency on that core is driven by substituents
    rather than the core itself — the hardest case for a fingerprint model, and a
    threshold-free complement to the cliff analysis.
    """
    )
    return


@app.cell
def _(compounds, endpoint_selector, scaffold_long, scaffolds):
    scaffold_spread = scaffolds.scaffold_activity(
        scaffold_long,
        compounds,
        endpoint=endpoint_selector.value,
        min_molecules=3,
    )

    print(
        f"{scaffold_spread.height} scaffolds with >=3 labelled molecules "
        f"for {endpoint_selector.value}"
    )
    scaffold_spread.head(25)
    return (scaffold_spread,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Interactive — scaffold coverage, test vs training

    Each point is a scaffold, positioned by how many training molecules contain it
    (x) against how many test molecules do (y). Hover to draw the scaffold.

    - **Top-left** — well represented in the test set, barely in training. These
      are where the test set will be hardest, and the best argument for public data
      augmentation.
    - **Bottom-right** — heavily trained, rarely tested. Effort spent here does not
      move the leaderboard.
    """
    )
    return


@app.cell
def _(interactive, mo, pl, scaffold_table):
    _plot_df = (
        scaffold_table.filter(
            (pl.col("n_in_test") >= 1) & (pl.col("n_in_inhibition") >= 1)
        )
        .rename({"scaffold_smiles": "SMILES"})
        .with_columns(pl.col("SMILES").alias("Molecule_Name"))
    )

    _chart = interactive.scatter(
        _plot_df,
        x="n_in_inhibition",
        y="n_in_test",
        key_fields=["SMILES"],
        color="scaffold_heavy_atoms",
        color_title="heavy atoms",
        tooltip=[
            ("scaffold_type", "Scaffold level", None),
            ("n_in_inhibition", "Training molecules", None),
            ("n_in_test", "Test molecules", None),
            ("scaffold_heavy_atoms", "Heavy atoms", None),
        ],
        title="Scaffold coverage — training vs test occurrence",
        x_title="molecules in the inhibition training set",
        y_title="molecules in the test set",
    )

    scaffold_chart = mo.ui.altair_chart(_chart)
    scaffold_plot_df = _plot_df
    return scaffold_chart, scaffold_plot_df


@app.cell
def _(interactive, mo, scaffold_chart, scaffold_plot_df):
    _row = interactive.selected_row(scaffold_chart, scaffold_plot_df, ["SMILES"])
    _panel = interactive.structure_panel(
        _row,
        name_col="scaffold_type",
        fields=[
            ("n_in_inhibition", "Training molecules", ""),
            ("n_in_test", "Test molecules", ""),
            ("scaffold_heavy_atoms", "Heavy atoms", ""),
        ],
        placeholder="Hover a scaffold to see its structure",
    )
    mo.hstack([scaffold_chart, _panel], align="start")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## TDI: is time-dependent inhibition a different chemical space?

    The TDI track has its own pattern in the baseline — LightGBM reaches MCC 0.259
    on CYP3A4 but only 0.116 on CYP2D6, mirroring the regression track's ordering.
    CLAUDE.md flags that as worth investigating jointly rather than as two separate
    problems, so: do the TDI positives occupy a distinct region of chemical space?

    A yes would point at a structural alert — TDI is mechanistically about reactive
    metabolite formation, which is the kind of thing substructure features capture
    well. A no means the boolean label is doing something the structure alone does
    not explain.
    """
    )
    return


@app.cell
def _(C, embedded, mo, pl):
    _tdi = embedded.filter(pl.col("in_tdi"))

    _counts = _tdi.select(
        [
            pl.col(_e).sum().alias(f"{_e}_positive")
            for _e in C.CLASSIFICATION_ENDPOINTS
        ]
        + [
            pl.col(_e).is_not_null().sum().alias(f"{_e}_labelled")
            for _e in C.CLASSIFICATION_ENDPOINTS
        ]
    )
    mo.vstack([mo.md("**TDI label balance**"), _counts])
    return


@app.cell
def _(C, OUT_DIR, embedded, mo, pl):
    def _plot_tdi_space(df):
        """TDI positives against negatives on the shared UMAP, one panel per isoform."""
        import matplotlib.pyplot as plt

        with plt.style.context("seaborn-v0_8-whitegrid"):
            fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=300)

            for ax, endpoint in zip(axes, C.CLASSIFICATION_ENDPOINTS, strict=True):
                _labelled = df.filter(pl.col(endpoint).is_not_null())
                _pos = _labelled.filter(pl.col(endpoint))
                _neg = _labelled.filter(~pl.col(endpoint))

                ax.scatter(
                    df["UMAP_x"].to_numpy(), df["UMAP_y"].to_numpy(),
                    c="#ededed", s=5, linewidths=0, alpha=0.5,
                )
                ax.scatter(
                    _neg["UMAP_x"].to_numpy(), _neg["UMAP_y"].to_numpy(),
                    c="#4e79a7", s=9, linewidths=0, alpha=0.55,
                    label=f"not TDI (n={_neg.height})",
                )
                ax.scatter(
                    _pos["UMAP_x"].to_numpy(), _pos["UMAP_y"].to_numpy(),
                    c="#d73027", s=14, linewidths=0, alpha=0.85,
                    label=f"TDI (n={_pos.height})",
                )
                ax.set_title(endpoint.replace("_is_TDI", " TDI"), fontsize=11, pad=8)
                ax.set_xlabel("UMAP 1", fontsize=9)
                ax.set_ylabel("UMAP 2", fontsize=9)
                ax.tick_params(axis="both", labelsize=8)
                ax.legend(fontsize=8, loc="best")

            fig.tight_layout()
        return fig

    _fig = _plot_tdi_space(embedded)
    _fig.savefig(OUT_DIR / "umap_tdi_labels.png", dpi=300, bbox_inches="tight")
    mo.center(mo.as_html(_fig))
    return


@app.cell
def _(C, compounds, np, pl, similarity):
    # The classification analogue of the potency-coherence score: among a compound's
    # k structural neighbours, how often does the TDI label agree with its own?
    # Compared against the rate expected from label balance alone, which is what a
    # model that ignores structure would achieve.
    _k = 5
    _rows = []

    for _endpoint in C.CLASSIFICATION_ENDPOINTS:
        _labelled = compounds.filter(pl.col(_endpoint).is_not_null())
        _labels = _labelled[_endpoint].to_numpy().astype(bool)

        _sims = similarity.cross_similarities(_labelled, _labelled, "ecfp4_1k")
        np.fill_diagonal(_sims, -1.0)
        _neighbour_idx = np.argsort(-_sims, axis=1)[:, :_k]
        _neighbour_labels = _labels[_neighbour_idx]

        _agreement = (_neighbour_labels == _labels[:, None]).mean()
        _positive_rate = float(_labels.mean())
        # Agreement a label-blind predictor would reach by chance.
        _chance = _positive_rate**2 + (1 - _positive_rate) ** 2

        _rows.append(
            {
                "isoform": _endpoint.replace("_is_TDI", ""),
                "n_labelled": int(_labels.size),
                "positive_rate_pct": round(100 * _positive_rate, 1),
                "nn_label_agreement": round(float(_agreement), 3),
                "chance_agreement": round(float(_chance), 3),
                "lift": round(float(_agreement - _chance), 3),
            }
        )

    tdi_coherence = pl.DataFrame(_rows)
    tdi_coherence
    return (tdi_coherence,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    `lift` is the number to read: how much more often structural neighbours share a
    TDI label than label balance alone would produce. A lift near zero means ECFP
    neighbourhoods carry no TDI information, which would explain a weak MCC far
    better than any hyperparameter would.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Summary

    The tables below collect what this notebook measured, per endpoint, so the
    next notebook can act on it rather than re-deriving it.
    """
    )
    return


@app.cell
def _(
    cliff_rates_ecfp4,
    endpoint_summary,
    mmp_cliff_summary,
    neighbour_coherence,
    pl,
    test_coverage,
):
    _coverage = test_coverage.group_by("endpoint").agg(
        pl.col("nn_similarity").median().round(3).alias("median_test_nn_sim"),
        (pl.col("nn_similarity") < 0.3).mean().mul(100).round(1).alias("pct_extrapolation"),
    )

    endpoint_scorecard = (
        endpoint_summary.select(
            ["endpoint", "n", "median_CI_width", "below_floor_pct"]
        )
        .join(
            neighbour_coherence.select(["endpoint", "mean_nn_similarity", "coherence_ratio"]),
            on="endpoint",
            how="left",
        )
        .join(_coverage, on="endpoint", how="left")
        .join(
            cliff_rates_ecfp4.select(
                ["endpoint", pl.col("cliff_rate_pct").alias("ecfp4_cliff_rate_pct")]
            ),
            on="endpoint",
            how="left",
        )
        .join(
            mmp_cliff_summary.select(
                ["endpoint", pl.col("cliff_rate_pct").alias("mmp_cliff_rate_pct")]
            ),
            on="endpoint",
            how="left",
        )
        .sort("endpoint")
    )

    endpoint_scorecard
    return (endpoint_scorecard,)


@app.cell
def _(
    OUT_DIR,
    endpoint_scorecard,
    mo,
    nn_cluster_sizes,
    nn_potency_bands,
    nn_potency_enrichment,
    scaffold_spread,
    tdi_coherence,
    track_coverage,
    tracks_measured,
):
    endpoint_scorecard.write_csv(OUT_DIR / "endpoint_scorecard.csv")
    tdi_coherence.write_csv(OUT_DIR / "tdi_coherence.csv")
    scaffold_spread.write_csv(OUT_DIR / "scaffold_activity_spread.csv")
    tracks_measured.write_csv(OUT_DIR / "nn_tracks_measured.csv")
    track_coverage.write_csv(OUT_DIR / "nn_track_coverage.csv")
    nn_potency_bands.write_csv(OUT_DIR / "nn_potency_bands.csv")
    nn_potency_enrichment.write_csv(OUT_DIR / "nn_potency_enrichment.csv")
    nn_cluster_sizes.write_csv(OUT_DIR / "nn_cluster_sizes.csv")

    mo.md(
        f"""
    Wrote to `{OUT_DIR.relative_to(OUT_DIR.parents[1])}/`:

    - `endpoint_scorecard.csv` — the per-endpoint summary above
    - `tdi_coherence.csv` — TDI neighbourhood label agreement
    - `scaffold_activity_spread.csv` — per-scaffold potency spread for the selected endpoint
    - `nn_tracks_measured.csv` — how many tracks each test compound's global NN carries
    - `nn_track_coverage.csv` — per-track measurement rate of that NN
    - `nn_potency_bands.csv` — that NN's potency band, per track
    - `nn_potency_enrichment.csv` — NN potency vs the training pool's base rate
    - `nn_cluster_sizes.csv` — how many test compounds share each NN
    - 10 PNG figures
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### How to read the scorecard

    Against the `01_baseline.py` ST-RAE ordering (CYP3A4 0.720 → CYP2C9 0.961 →
    CYP1A2 0.986 → CYP2D6 1.063), the columns split into two kinds of explanation.

    **Representational** — `coherence_ratio`, `ecfp4_cliff_rate_pct`,
    `mmp_cliff_rate_pct`, `median_test_nn_sim`. These say whether ECFP
    neighbourhoods carry potency information at all.

    **Metric-structural** — `median_CI_width` and `below_floor_pct`. These change
    what ST-RAE *charges* for a given prediction quality without saying anything
    about whether the chemistry is learnable.

    What this run actually found:

    **1. The coherence ratio is 0.89–0.94 on every endpoint, CYP3A4 included.**
    Knowing a compound's five nearest ECFP4 neighbours reduces the uncertainty in
    its potency by roughly 6–11% versus knowing nothing but the global mean. That is
    a weak signal *everywhere*, and it reframes the baseline table: CYP3A4 is not a
    representation that works while three others fail — it is the same weak
    representation, scored on the most forgiving endpoint. The spread across the
    four (0.894 to 0.941) is far too narrow to explain an ST-RAE spread of 0.720 to
    1.063 on its own.

    **2. Most of CYP3A4's apparent lead is metric-structural.** It combines the
    widest forgiveness (40.4% of compounds below the pIC50 4 floor, where ST-RAE
    downweights) with by far the best test coverage (1.9% extrapolation against
    7–11% elsewhere, and 2,335 labelled compounds against 1,285–1,493). CYP2D6 is
    the mirror image on every one of those: tightest intervals (0.27), fewest
    sub-floor compounds (8.6%), worst coverage (10.8% extrapolation, median NN
    0.469). The endpoint ordering in the baseline tracks these columns more closely
    than it tracks anything about SAR difficulty.

    **3. Cliff rates do not explain the ordering, and cut the other way.** CYP3A4
    is second-highest on both cliff measures (ECFP4 14.2%, MMP 8.6%) while being the
    endpoint that learns best, and CYP2D6 is lowest or near-lowest on both (6.8% /
    2.1%) while learning worst. Both rankings are led by CYP1A2 (16.5% / 8.8%), an
    endpoint that sits at the mean-predictor line. So the weak endpoints are not
    weak because their SAR is unusually discontinuous — a hypothesis this notebook
    was built to test, and which it rules out.

    **4. TDI labels are nearly structure-blind at this resolution.** Neighbour label
    agreement beats chance by 0.059 on CYP3A4 and 0.020 on CYP2D6 — the same
    ordering as the MCC baseline (0.259 vs 0.116), and the same conclusion: both
    lifts are small, and CYP2D6's is close to nothing.

    **What follows.** The representational ceiling is the binding constraint, and it
    binds on all four endpoints rather than three — so PLAN.md's priority 2 (graph
    models) is the right next notebook, and its expected gain is larger than the
    baseline table suggests, since even CYP3A4 has headroom. Two corollaries worth
    carrying forward: coverage is the single most endpoint-discriminating column in
    the scorecard, which strengthens the case for priority 3 (public data) on CYP2D6
    and CYP2C9 specifically; and nothing here is a reason to tune the LightGBM
    baseline harder — PXR's HPO lesson applies to tree models on a *working*
    representation, which this is not.

    One caveat on the coherence ratio: it is computed at k=5 on ECFP4, so it
    measures what *this* representation resolves, not what is inherently learnable.
    That is exactly the quantity that should improve if a learned representation
    helps — which makes it worth recomputing on graph-model embeddings as a direct
    check, rather than inferring the benefit from CV scores alone.
    """
    )
    return


if __name__ == "__main__":
    app.run()
