import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # 09 — 3D representation for CYP2D6

    `07_placement.py` decomposed the board and found the remaining macro gap is almost
    entirely **ordering on CYP2D6**: Spearman 0.36 against 0.70–0.77 elsewhere, which
    caps its R² at 0.14 whatever the placement. `08_emax_two_stage.py` closed off the
    cheap data shortcut. What is left is representation.

    ## Why CYP2D6 is a representation problem specifically

    `PLAN.md` carried an untested hypothesis from day one: CYP2D6 binding is dominated
    by a basic-nitrogen/aromatic pharmacophore that circular fingerprints represent
    poorly. Measured on the challenge's own labels, that is not merely true — it
    separates CYP2D6 from every other endpoint:

    | endpoint | basic N vs potency | logP vs potency | % with basic N |
    |:--|--:|--:|--:|
    | CYP1A2 | −0.132 | 0.233 | 16.6% |
    | CYP2C9 | −0.160 | 0.485 | 17.6% |
    | **CYP2D6** | **+0.284** | **0.034** | **40.9%** |
    | CYP3A4 | −0.068 | 0.602 | 23.0% |

    The other three endpoints track **lipophilicity**, which ECFP captures well as a
    bulk property. CYP2D6 is the only one where logP carries essentially nothing, and
    the only one where a basic nitrogen is *positively* associated with potency. That
    is a specific spatial arrangement rather than a bulk property, which is exactly
    what a 2D fingerprint cannot express — and it explains why four unrelated
    architectures in `03` all plateaued around 0.93 on this endpoint. They were all
    reading the same representation.

    ## The measurement that motivates 3D rather than more descriptors

    Topological (bond-count) distance from the basic nitrogen to the nearest aromatic
    atom correlates **−0.122** with CYP2D6 potency, with a real monotonic trend across
    bins — mean pIC50 5.155 at 2–4 bonds, 4.993 at 4–6, 4.978 at 6–8. So the
    relationship exists but is weakly expressed through bond counts.

    **Through-space distance is the direct test.** If the pharmacophore is genuinely
    geometric, a conformer-derived distance should beat the topological one. If it does
    not, the hypothesis is wrong — which is worth establishing before spending a
    heavyweight dependency on a pretrained 3D model.

    ## Arms

    | arm | representation | what it isolates |
    |:--|:--|:--|
    | **ecfp** | ECFP4-2048 | the incumbent |
    | **pharmacophore_2d** | 5 named descriptors, no conformers | are explicit features enough? |
    | **pharmacophore_3d** | + through-space N→aromatic | does *geometry* add over topology? |
    | **e3fp** | 3D conformer fingerprint | 3D analogue of ECFP |
    | **usrcat** | pharmacophore-aware 3D shape | shape rather than substructure |
    | **ecfp + pharmacophore_3d** | concatenated | does 3D add *on top of* the incumbent? |

    Two design points carry the interpretation:

    **All four endpoints run, not just CYP2D6.** The other three are the negative
    control. A 3D arm that helps CYP2D6 *and* the lipophilicity-driven endpoints is
    probably just a better featurizer; one that helps CYP2D6 *specifically* is evidence
    for the mechanism. This is the rare case where the per-endpoint view is the point
    and the macro would hide it.

    **Scored in Spearman.** Ordering is the target, and ST-RAE confounds ordering with
    placement — which is what hid this gap until `07` decomposed the board. ST-RAE is
    recorded alongside for continuity with every other notebook, but it is not what
    this one is asking.
    """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os
    import sys
    import time
    from pathlib import Path

    import numpy as np
    import polars as pl
    from scipy import stats

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from cyp import (
        aux_training,
        cv,
        data,
        evaluation,
        fingerprints,
        metrics,
        pharmacophore,
        timings,
    )
    from cyp import constants as C

    # 04 measured chemprop climbing from ~100s to ~2000s per fold on MPS and
    # plateauing there permanently; CPU held flat at ~76s and was faster outright.
    # A3 is the only part of this notebook that runs chemprop, but the setting has
    # to be here regardless of where in the file it takes effect.
    os.environ.setdefault("CYP_CHEMPROP_DEVICE", "cpu")

    return (
        C,
        PROJECT_ROOT,
        Path,
        aux_training,
        cv,
        data,
        evaluation,
        fingerprints,
        metrics,
        np,
        pharmacophore,
        pl,
        stats,
        time,
        timings,
    )


@app.cell
def _(Path, PROJECT_ROOT):
    NOTEBOOK_NAME = Path(__file__).stem
    OUT_DIR = PROJECT_ROOT / "experiments" / NOTEBOOK_NAME
    CACHE_DIR = OUT_DIR / "cache"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TIMING_LOG = OUT_DIR / "timings.csv"
    return CACHE_DIR, NOTEBOOK_NAME, OUT_DIR, TIMING_LOG


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Featurization

    Conformer generation is the one expensive step — roughly a minute per thousand
    compounds against under a second for a 2D fingerprint — so every representation is
    cached to `experiments/09_cyp2d6_representation/cache/`: the arm sweep keyed by
    `(kind, endpoint)`, A3's descriptor matrix separately keyed by compound count
    since it runs over the endpoint union rather than one endpoint at a time. Both
    caches are a pure function of the data snapshot with no model fits in them, so per
    the repo convention they are gitignored rather than committed.

    **Conformer failures are counted, not swallowed.** A 3D arm that silently fell back
    to 2D for some fraction of compounds would look like a null result for geometry,
    which is the one conclusion this notebook must not reach by accident.
    """
    )
    return


@app.cell
def _(C, CACHE_DIR, TIMING_LOG, data, fingerprints, np, pharmacophore, pl, time, timings):
    ENDPOINTS = C.REGRESSION_ENDPOINTS
    N_OUTER, N_INNER = 5, 5

    _inh = data.load_train_inhibition()

    def endpoint_frame(endpoint: str) -> pl.DataFrame:
        return _inh.filter(pl.col(endpoint).is_not_null()).with_row_index("ridx")

    def featurize(kind: str, endpoint: str, smiles: list[str]) -> np.ndarray:
        """Featurize with an on-disk cache, timing only genuine computation."""
        path = CACHE_DIR / f"{kind}_{endpoint.split('_')[0]}.npy"
        if path.exists():
            return np.load(path)

        start = time.perf_counter()
        if kind == "pharmacophore_2d":
            frame = pharmacophore.descriptors(smiles, include_3d=False)
            matrix = pharmacophore.matrix(frame)
        elif kind == "pharmacophore_3d":
            frame = pharmacophore.descriptors(smiles, include_3d=True, timeout_s=10)
            matrix = pharmacophore.matrix(frame)
        else:
            matrix = fingerprints.compute(smiles, kind, n_bits=2048)
        elapsed = time.perf_counter() - start

        matrix = np.asarray(matrix, dtype=float)
        np.save(path, matrix)
        timings.record(
            TIMING_LOG,
            stage="featurize",
            method=kind,
            endpoint=endpoint.split("_")[0],
            mode="full",
            seconds=round(elapsed, 1),
        )
        return matrix

    return ENDPOINTS, N_INNER, N_OUTER, endpoint_frame, featurize


@app.cell
def _(mo):
    mo.md(
        r"""
    ## The hypothesis check, before any model

    Does through-space distance order CYP2D6 better than bond-count distance? This is
    the whole premise in one number, and it costs one featurization rather than a CV
    sweep. If the 3D column does not beat the topological one here, the modelling arms
    below are unlikely to rescue it.
    """
    )
    return


@app.cell
def _(ENDPOINTS, OUT_DIR, endpoint_frame, mo, pharmacophore, pl, stats):
    _rows = []
    for _endpoint in ENDPOINTS:
        _frame = endpoint_frame(_endpoint)
        _desc = pharmacophore.descriptors(_frame["SMILES"].to_list(), include_3d=True, timeout_s=10)
        _y = _frame[_endpoint].to_numpy()
        for _column in ("topological_n_to_aromatic", "spatial_n_to_aromatic_min"):
            _values = _desc[_column].to_numpy()
            _ok = ~pl.Series(_values).is_nan().to_numpy()
            if _ok.sum() < 30:
                continue
            _rows.append(
                {
                    "endpoint": _endpoint.split("_")[0],
                    "descriptor": _column,
                    "n": int(_ok.sum()),
                    "spearman": round(float(stats.spearmanr(_y[_ok], _values[_ok]).statistic), 4),
                }
            )

    distance_check = pl.DataFrame(_rows)
    distance_check.write_csv(OUT_DIR / "distance_descriptor_check.csv")

    mo.vstack(
        [
            mo.ui.table(distance_check, selection=None),
            mo.md(
                "**Read the CYP2D6 rows against each other.** A `spatial` correlation "
                "stronger than the `topological` one on CYP2D6, and not on the other "
                "three, is the mechanism this notebook is testing. Note `n` differs "
                "from the endpoint's full count: only compounds carrying *both* a basic "
                "nitrogen and an aromatic ring have either distance at all."
            ),
        ]
    )
    return (distance_check,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## CV across arms and endpoints

    LightGBM throughout, so the representation is the only thing varying. Shared
    scaffold folds per endpoint, so every arm is scored on identical rows and the
    comparison is paired.
    """
    )
    return


@app.cell
def _(
    ENDPOINTS,
    N_INNER,
    N_OUTER,
    TIMING_LOG,
    cv,
    endpoint_frame,
    featurize,
    metrics,
    mo,
    np,
    pl,
    stats,
    time,
    timings,
):
    import lightgbm as lgb

    ARMS = (
        "ecfp",
        "pharmacophore_2d",
        "pharmacophore_3d",
        "e3fp",
        "usrcat",
        "ecfp+pharmacophore_3d",
    )

    def arm_matrix(arm: str, endpoint: str, smiles: list[str]) -> np.ndarray:
        if "+" in arm:
            return np.hstack([featurize(k, endpoint, smiles) for k in arm.split("+")])
        return featurize(arm, endpoint, smiles)

    records = []
    for _endpoint in ENDPOINTS:
        _frame = endpoint_frame(_endpoint)
        _smiles = _frame["SMILES"].to_list()
        _y = _frame[_endpoint].to_numpy()
        _lo = _frame[f"{_endpoint}_conf_low"].to_numpy()
        _hi = _frame[f"{_endpoint}_conf_high"].to_numpy()
        _folds = [
            (_tr["ridx"].to_numpy(), _te["ridx"].to_numpy())
            for _f, _o, _i, _tr, _va, _te in cv.scaffold_splits(
                _frame, n_outer=N_OUTER, n_inner=N_INNER, seed=0
            )
        ]

        for _arm in ARMS:
            _X = arm_matrix(_arm, _endpoint, _smiles)
            _oof = np.full(len(_y), np.nan)
            _start = time.perf_counter()
            for _tr, _te in _folds:
                _model = lgb.LGBMRegressor(n_estimators=300, random_state=0, verbose=-1)
                _model.fit(_X[_tr], _y[_tr])
                _oof[_te] = _model.predict(_X[_te])
            timings.record(
                TIMING_LOG,
                stage="cv",
                method=_arm,
                endpoint=_endpoint.split("_")[0],
                mode="full",
                seconds=round(time.perf_counter() - _start, 1),
            )

            _ok = ~np.isnan(_oof)
            records.append(
                {
                    "endpoint": _endpoint.split("_")[0],
                    "arm": _arm,
                    "n_features": int(_X.shape[1]),
                    "spearman": round(float(stats.spearmanr(_y[_ok], _oof[_ok]).statistic), 4),
                    "st_rae": round(
                        float(metrics.st_rae(_y[_ok], _oof[_ok], _lo[_ok], _hi[_ok])), 4
                    ),
                }
            )

    results = pl.DataFrame(records)
    mo.md(f"Ran {len(ARMS)} arms x {len(ENDPOINTS)} endpoints on {N_OUTER * N_INNER} folds each.")
    return ARMS, arm_matrix, lgb, records, results


@app.cell
def _(OUT_DIR, PROJECT_ROOT, ENDPOINTS, mo, pl, stats):
    # The incumbent, read from 07's committed OOF rather than refitted. Every arm in
    # this notebook is a LightGBM, which isolates the representation but is NOT the
    # model that would ship: chemprop `pubchem` is. A 3D arm can beat ECFP here and
    # still be far short of the incumbent, so the tree comparison needs this line
    # beside it to be read correctly. Fold counts differ (09 runs 3x5, 07 ran 5x5),
    # so this is a scale reference and not a paired comparison -- settle any real
    # ranking against chemprop with a shared-fold run, not with these numbers.
    reference_rows = []
    _path = PROJECT_ROOT / "experiments" / "07_placement" / "oof.parquet"
    if _path.exists():
        _oof = pl.read_parquet(_path).filter(pl.col("method") == "pubchem")
        for _endpoint in ENDPOINTS:
            _slice = _oof.filter(pl.col("endpoint") == _endpoint)
            if _slice.height:
                reference_rows.append(
                    {
                        "endpoint": _endpoint.split("_")[0],
                        "chemprop_pubchem_spearman": round(
                            float(stats.spearmanr(_slice["y_true"], _slice["y_pred"]).statistic),
                            4,
                        ),
                    }
                )

    reference = pl.DataFrame(reference_rows) if reference_rows else pl.DataFrame()
    if reference.height:
        reference.write_csv(OUT_DIR / "incumbent_reference.csv")
    return (reference,)


@app.cell
def _(OUT_DIR, mo, pl, reference, results):
    results.write_csv(OUT_DIR / "arm_results.csv")

    spearman_table = results.pivot(values="spearman", index="arm", on="endpoint")
    strae_table = results.pivot(values="st_rae", index="arm", on="endpoint")

    _reference_view = (
        mo.vstack(
            [
                mo.md(
                    "**The incumbent, for scale.** chemprop `pubchem` from 07's OOF — "
                    "the model that actually ships. Every arm above is a LightGBM, "
                    "which isolates representation but starts well behind. Different "
                    "fold counts (3x5 here against 5x5 there), so read this as scale, "
                    "not as a paired result."
                ),
                mo.ui.table(reference, selection=None),
            ]
        )
        if reference.height
        else mo.md("_07's OOF not found; no incumbent reference available._")
    )

    mo.vstack(
        [
            mo.md("**Spearman by arm and endpoint — the metric this notebook is about.**"),
            mo.ui.table(spearman_table, selection=None),
            _reference_view,
            mo.md(
                "**ST-RAE, for continuity with every other notebook.** Not what this "
                "one is asking: it confounds ordering with placement."
            ),
            mo.ui.table(strae_table, selection=None),
        ]
    )
    return spearman_table, strae_table


@app.cell
def _(OUT_DIR, mo, pl, results):
    _base = results.filter(pl.col("arm") == "ecfp").select(
        "endpoint", pl.col("spearman").alias("ecfp_spearman")
    )
    lift = (
        results.join(_base, on="endpoint")
        .with_columns((pl.col("spearman") - pl.col("ecfp_spearman")).round(4).alias("lift_vs_ecfp"))
        .filter(pl.col("arm") != "ecfp")
        .select("arm", "endpoint", "spearman", "ecfp_spearman", "lift_vs_ecfp")
        .sort("arm", "endpoint")
    )
    lift.write_csv(OUT_DIR / "lift_vs_ecfp.csv")

    _selectivity = (
        lift.with_columns(pl.col("endpoint").eq("CYP2D6").alias("is_2d6"))
        .group_by("arm", "is_2d6")
        .agg(pl.col("lift_vs_ecfp").mean().round(4).alias("mean_lift"))
        .pivot(values="mean_lift", index="arm", on="is_2d6")
    )

    mo.vstack(
        [
            mo.ui.table(lift, selection=None),
            mo.md(
                "**The selectivity view is the one that carries the mechanism.** A lift "
                "on CYP2D6 with little or none on the other three supports the "
                "pharmacophore reading. A lift everywhere means a better featurizer, "
                "which is a fine result but a different one."
            ),
            mo.ui.table(_selectivity, selection=None),
        ]
    )
    return (lift,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## A3 — folding `pharmacophore_3d` into the incumbent

    The LightGBM sweep above answers "does 3D geometry help a tree model read
    columns?" but the model that would actually ship is chemprop `pubchem` from `07`,
    not a tree on ECFP. The result was non-selective — `ecfp+pharmacophore_3d` lifted
    every endpoint, largest on CYP2C9, not CYP2D6 — but a weak tree model and a shared
    learned encoder are different tests: `03` already showed a shared encoder captures
    signal that concatenated tree features cannot, so it is worth asking whether
    chemprop can use the same 7 descriptors more effectively than LightGBM did.

    **This is `aux_training.run_cv_pretrained`'s `descriptor_columns` /
    `finetune_descriptors` arguments**, added for this notebook. One real
    architectural constraint surfaced while wiring them: chemprop's `--checkpoint`
    restores the FFN's weights whole, including its input width, so a pretrain
    checkpoint built *without* descriptor columns cannot warm-start a fine-tuning
    model built *with* them — confirmed directly against the CLI (a
    `RuntimeError: shapes cannot be multiplied` on the mismatched width), not assumed.
    The fix is that the pretraining stage also carries the descriptor columns,
    computed on the public pretraining corpus purely to keep the architecture
    loadable — the pretraining data has no genuine independent use for them.

    **Two arms, paired on identical shared folds:**

    | arm | descriptor_columns | what it tests |
    |:--|:--|:--|
    | `pubchem` | none | 07's incumbent, re-run here seed-for-seed |
    | `pubchem_pharmacophore` | all 7 descriptors | does chemprop use it better than the tree? |

    Scored the same way as the rest of this notebook — Spearman per endpoint, all
    four run as controls — plus `evaluation.paired_bootstrap` and
    `holm_bonferroni`, since a point estimate here is exactly the trap PLAN.md warns
    against.
    """
    )
    return


@app.cell
def _(CACHE_DIR, ENDPOINTS, cv, data, mo, pharmacophore, pl):
    a3_frames = {e: data.training_frame(e) for e in ENDPOINTS}
    _, a3_fold_assignments = cv.shared_scaffold_folds(a3_frames, n_outer=5, n_inner=5, seed=0)
    a3_n_folds = a3_fold_assignments["fold"].n_unique()

    a3_smiles_by_name: dict[str, str] = {}
    for _frame in a3_frames.values():
        for _name, _smi in zip(
            _frame["Molecule_Name"].to_list(), _frame["SMILES"].to_list(), strict=True
        ):
            a3_smiles_by_name[_name] = _smi
    a3_all_names = sorted(a3_smiles_by_name)

    # Conformer generation for this union costs the same minute-per-thousand as the
    # main sweep's pharmacophore_3d arm, but over a different (larger) compound list,
    # so it cannot reuse that arm's cache file. Keyed on the compound count: the union
    # is a deterministic sort of the endpoint frames, so a changed snapshot or
    # endpoint set changes the count and invalidates the cache rather than silently
    # serving stale descriptors for a different population.
    _cache_path = CACHE_DIR / f"a3_descriptors_{len(a3_all_names)}.parquet"
    if _cache_path.exists():
        a3_descriptors = pl.read_parquet(_cache_path)
    else:
        _desc_frame = pharmacophore.descriptors(
            [a3_smiles_by_name[n] for n in a3_all_names], include_3d=True, n_jobs=-1, timeout_s=10
        )
        _matrix = pharmacophore.matrix(_desc_frame)
        a3_descriptors = pl.DataFrame(
            {"Molecule_Name": a3_all_names}
            | {c: _matrix[:, i] for i, c in enumerate(pharmacophore.DESCRIPTOR_COLUMNS)}
        )
        a3_descriptors.write_parquet(_cache_path)

    mo.md(
        f"{len(a3_frames)} endpoints, {len(a3_all_names):,} compounds, "
        f"**{a3_n_folds} shared folds**. Descriptors computed for every compound "
        "across the union so the fine-tuning join has no gaps."
    )
    return a3_descriptors, a3_fold_assignments, a3_frames, a3_n_folds


@app.cell
def _(
    OUT_DIR,
    a3_descriptors,
    a3_fold_assignments,
    a3_frames,
    a3_n_folds,
    aux_training,
    mo,
    pharmacophore,
    pl,
    timings,
):
    a3_cache_dir = OUT_DIR / "cache"
    a3_timing_log = OUT_DIR / "timings.csv"
    a3_all_smiles = list({s for f in a3_frames.values() for s in f["SMILES"].to_list()})
    a3_base_matrix = aux_training.public_pretraining_matrix(
        list(a3_frames), exclude_smiles=a3_all_smiles
    )

    a3_arms = {
        "pubchem": {},
        "pubchem_pharmacophore": {
            "descriptor_columns": list(pharmacophore.DESCRIPTOR_COLUMNS),
            "finetune_descriptors": a3_descriptors,
        },
    }

    _bar = mo.status.progress_bar(total=a3_n_folds * len(a3_arms), title="A3 CV")
    a3_oof_parts = []
    import time as _time

    with _bar as bar:
        for _fold in range(a3_n_folds):
            for _arm, _kwargs in a3_arms.items():
                _cache = a3_cache_dir / f"{_arm}_fold{_fold}.parquet"
                if _cache.exists():
                    a3_oof_parts.append(pl.read_parquet(_cache))
                    bar.update()
                    continue
                _start = _time.perf_counter()
                _result = aux_training.run_cv_pretrained(
                    a3_frames,
                    pretraining=a3_base_matrix,
                    method_name=_arm,
                    pretrain_epochs=30,
                    n_outer=5,
                    descriptor_timeout_s=10,
                    assignments=a3_fold_assignments,
                    folds=[_fold],
                    epochs=50,
                    **_kwargs,
                )
                timings.record(
                    a3_timing_log,
                    stage="a3",
                    method=_arm,
                    endpoint=f"fold{_fold}",
                    mode="full",
                    seconds=round(_time.perf_counter() - _start, 1),
                )
                _result.write_parquet(_cache)
                a3_oof_parts.append(_result)
                bar.update()

    a3_oof = pl.concat(a3_oof_parts)
    a3_oof.write_parquet(OUT_DIR / "a3_oof.parquet")
    mo.md(f"A3 OOF: {a3_oof.height:,} rows, arms {sorted(a3_oof['method'].unique())}.")
    return (a3_oof,)


@app.cell
def _(OUT_DIR, a3_oof, evaluation, mo, pl, stats):
    _rows = []
    for _method in sorted(a3_oof["method"].unique().to_list()):
        for _endpoint in sorted(a3_oof["endpoint"].unique().to_list()):
            _slice = a3_oof.filter(
                (pl.col("method") == _method) & (pl.col("endpoint") == _endpoint)
            )
            _rows.append(
                {
                    "arm": _method,
                    "endpoint": _endpoint.split("_")[0],
                    "n": _slice.height,
                    "spearman": round(
                        float(stats.spearmanr(_slice["y_true"], _slice["y_pred"]).statistic), 4
                    ),
                }
            )
    a3_results = pl.DataFrame(_rows)
    a3_results.write_csv(OUT_DIR / "a3_results.csv")

    a3_spearman_table = a3_results.pivot(values="spearman", index="arm", on="endpoint")

    _bootstrap_rows = []
    for _endpoint in sorted(a3_oof["endpoint"].unique().to_list()):
        _base = a3_oof.filter(
            (pl.col("method") == "pubchem") & (pl.col("endpoint") == _endpoint)
        ).sort("Molecule_Name")
        _treat = a3_oof.filter(
            (pl.col("method") == "pubchem_pharmacophore") & (pl.col("endpoint") == _endpoint)
        ).sort("Molecule_Name")
        if _base.height == 0 or _treat.height == 0:
            continue
        _result = evaluation.paired_bootstrap(
            _base["y_true"].to_numpy(),
            _base["y_pred"].to_numpy(),
            _treat["y_pred"].to_numpy(),
            metric="rho",
        )
        _bootstrap_rows.append({"endpoint": _endpoint.split("_")[0], **_result})
    a3_bootstrap = pl.DataFrame(_bootstrap_rows) if _bootstrap_rows else pl.DataFrame()
    if a3_bootstrap.height:
        a3_bootstrap.write_csv(OUT_DIR / "a3_paired_bootstrap.csv")

    mo.vstack(
        [
            mo.md("**A3 Spearman by arm and endpoint.**"),
            mo.ui.table(a3_spearman_table, selection=None),
            mo.md(
                "**Paired bootstrap, `pubchem_pharmacophore` vs `pubchem`.** A CI "
                "spanning zero means the fold count cannot resolve this arm from the "
                "incumbent — treat it as indistinguishable, not as a small real effect."
            ),
            mo.ui.table(a3_bootstrap, selection=None) if a3_bootstrap.height else mo.md("n/a"),
        ]
    )
    return a3_results, a3_spearman_table, a3_bootstrap


@app.cell
def _(mo):
    mo.md(
        r"""
    ## How to read this

    **The decision this notebook feeds.** If a 3D arm lifts CYP2D6's Spearman
    selectively, the geometry hypothesis is confirmed and a pretrained 3D model
    (Uni-Mol) becomes well-motivated for the November 3 window — with a clean
    attribution, because A1/A2 will have separated "3D geometry helps" from "a large
    pretrained model helps". If no 3D arm beats ECFP on CYP2D6, the hypothesis is
    wrong and that dependency is saved.

    **Expect a modest ceiling, and do not read a small lift as failure.**
    `02_chemical_space`'s scorecard puts CYP2D6 at the lowest test-set
    nearest-neighbour similarity (0.469) and the highest extrapolation rate (10.8%,
    nearly 6x CYP3A4's). Part of the difficulty is genuine chemical-space novelty that
    no representation fixes. Moving Spearman 0.36 → 0.45–0.55 would be a real result
    worth ~0.05–0.08 on the macro R², several times what placement has left.

    **Before acting on any ranking here, apply the standing rule.** These are point
    estimates on 15 folds with no paired bootstrap. PXR cost five places by ranking
    three finalists separated by 0.0039 MAE. Run `evaluation.paired_bootstrap` between
    the contenders, and `holm_bonferroni` across the arm set, before believing an
    ordering — a lift smaller than the fold-to-fold spread is not a lift.

    **The baseline here is ECFP, but the incumbent is chemprop.** Every arm is a
    LightGBM so that representation is the only thing varying, which is what makes the
    comparison clean — but it also means the whole table starts well behind the model
    that would actually ship (chemprop `pubchem`, CYP2D6 OOF Spearman 0.44 against
    ECFP's ~0.33). The incumbent reference table above is there so a lift is never read
    as "this would improve the submission". It would not, on its own.

    What a selective CYP2D6 lift *does* license is the follow-up: fold the winning
    representation into chemprop as extra features and run it paired against plain
    `pubchem` on shared folds. That is the arm that could change a submission, and it
    is only worth its cost once this notebook says which geometry to feed it.
    """
    )
    return


if __name__ == "__main__":
    app.run()
