# CLAUDE.md

Modelling code for the OpenADMET CYP Inhibition Blind Challenge. See [README.md](README.md) for challenge background, data dictionary, and dates.

## Environment

`uv` only — there is no conda/mamba on this machine. Run everything through `uv run ...` or the Makefile; do not `pip install` into the system Python.

```bash
make setup     # uv sync --all-extras
make test      # pytest
make lint      # ruff check + format
make baseline  # notebooks/01_baseline.py, run as a script
```

## Code conventions

Carried over from the PXR challenge repo so the two stay consistent:

- **Polars, not Pandas.** Project code uses Polars throughout. Pandas appears only at the boundary: the vendored official validators call `pd.read_csv` internally, and sklearn takes numpy. Do not introduce Pandas into `src/cyp`.
- **Type hints on every parameter and return value.**
- Double quotes; f-strings with single-quoted keys are the exception.
- Comment generously and explain *why*, not what.

## Non-negotiables

**Never reimplement the scoring metrics.** `src/cyp/metrics.py` imports Soft-Threshold RAE from `vendor/cyp_challenge_tutorial/`, a pinned copy of the challenge backend's own code (upstream `858ae63`, fetched 2026-09-08). A hand-rolled ST-RAE will silently disagree with the leaderboard. If upstream changes, re-vendor the files rather than editing them — treat `vendor/` as read-only.

**Never rename the endpoint columns.** `CYP3A4_pIC50_direct_inhibition`, `CYP2D6_is_TDI`, and friends are matched by exact string in the scoring backend. They live in `src/cyp/constants.py`; check `vendor/cyp_challenge_tutorial/evaluation/config.py` before touching them.

**Always validate a submission before it leaves the repo.** `submission.check(path, track=...)` runs the official validator. Teams get one submission, so an invalid file is a real cost.

**Do not use a random split as the headline validation.** The test set is 75 hits × ~10 analogs each; random K-fold leaks near-duplicates and reports scores that will not hold up. Default to `cv.scaffold_splits`. If you report a `cv.random_splits` number, label it as optimistic.

**The competition ranks on the macro-average across endpoints, not on any single endpoint.** The vendored backend's `compute_macro_bootstrap_results` (`vendor/cyp_challenge_tutorial/evaluation/evaluate_predictions.py`) takes a plain arithmetic mean of the metric across all four regression endpoints (or both TDI isoforms) for each bootstrap resample — that macro value, not any per-CYP score, is what gets sorted on the leaderboard. A model can win on individual endpoints and still lose (or tie) on the metric that decides rank. Use `evaluation.macro_averaged_fold_metrics` to build the CV analogue of this before comparing models or picking a submission — see its docstring for the one real caveat (it matches fold *index* across endpoints rather than resampling the same compounds for every endpoint at once, since our endpoints don't share a fixed row count the way the leaderboard's bootstrap does).

## Data facts that bite

- **Targets are sparse, not missing-at-random.** A compound has a pIC50 only where the primary screen flagged it, so `dropna()` on the full frame leaves almost nothing. Filter one endpoint at a time via `data.labelled_subset(df, endpoint)`. Per-endpoint counts range from 1,285 (CYP2C9) to 2,335 (CYP3A4) out of 4,905 rows.
- **Four targets, four different training sets.** Because of the above, the four regression endpoints do not share rows. Multi-task setups need to handle that explicitly rather than assuming a dense 4-column matrix.
- **TDI labels are imbalanced** (~21% positive for CYP3A4, ~22% for CYP2D6) and only exist for CYP3A4/CYP2D6. MCC punishes majority-class predictors, so accuracy is not a useful guide here.
- **pIC50 < 4 is below assay resolution.** The metric downweights that region. Do not chase precision there.
- **Credible intervals are the point of ST-RAE.** `_conf_low`/`_conf_high` mean a prediction inside the band scores zero error — wide-interval (low-activity) compounds are forgiving, tight-interval ones are not. Uncertainty-aware models have real headroom here, and the Innovation award explicitly rewards that.
- **The dataset has been amended mid-challenge before.** Downloads go to dated snapshots, `data/raw/YYYYMMDD/`, never overwriting an existing one. `make data` writes today's snapshot and diffs it against the previous; `make data-diff` diffs without downloading. Loaders default to the newest snapshot and take `snapshot="YYYYMMDD"` to pin an older one — use that when reproducing a past result, and record the date next to any result worth reproducing. Never flatten this into a single directory or delete an old snapshot: the point is that a dataset amendment stays visible.

## Measured baseline (2026-09-09, `notebooks/01_baseline.py` → `experiments/01_baseline/`)

LightGBM on ECFP4, full nested scaffold CV (5x5, 25 folds), mean ST-RAE (1.0 = no better than a constant):

| Endpoint | raw | +linear calibration |
|:--|--:|--:|
| CYP3A4 | 0.720 | 0.707 |
| CYP2C9 | 0.961 | 0.883 |
| CYP1A2 | 0.986 | 0.903 |
| CYP2D6 | 1.063 | 0.947 |

**Only CYP3A4 genuinely learns from fingerprints.** The other three sit at or above the mean-predictor line before calibration. That is the central problem to solve, and it is not a tuning problem — it needs better representations (graph models) or more data (the single-concentration screen).

Linear calibration helps on *every* endpoint here, unlike in PXR where it looked like noise. Never ship an uncalibrated submission without checking this table first.

**TDI classification** (same notebook, same run): LightGBM with class-balanced weighting clears the MCC=0.0 majority-class baseline on both scored isoforms —

| Isoform | majority (baseline) | LightGBM |
|:--|--:|--:|
| CYP3A4 | 0.000 | **0.259** |
| CYP2D6 | 0.000 | 0.116 |

CYP2D6 TDI is markedly harder than CYP3A4 TDI, mirroring the regression track's pattern where CYP2D6 is the hardest endpoint. Worth investigating together rather than as separate problems.

**MCS heatmaps, per endpoint** (`experiments/01_baseline/mcs_heatmap_{regression,tdi}.png`, via `src/cyp/mcs.py`) confirm this visually: on CYP1A2/CYP2C9/CYP3A4, `lgbm` beats `mean` at p<0.01; on CYP2D6 it only reaches p<0.05, the weakest significance level on the whole grid. Same pattern in TDI — CYP2D6's non-`lgbm` methods are less cleanly separated from each other than CYP3A4's are. Treat CYP2D6 as needing a different approach (representation or data), not more of the same tuning.

**MCS heatmaps, macro-averaged (the one that matters for rank)** — `mcs_heatmap_{regression,tdi}_macro.png`, via `evaluation.macro_averaged_fold_metrics`. The leaderboard ranks on the macro-average across all four regression endpoints (and both TDI isoforms), not on any single endpoint — see the Non-negotiables entry below. On the macro-averaged ST-RAE, `lgbm` and `xgb` are **not** significantly different from each other (p=0.239) despite both clearly beating `mean`/`ridge`/`knn` — a materially different conclusion than any single per-endpoint panel gives, and the one that should actually drive a submission choice. On macro-averaged TDI MCC, `lgbm` (0.187) significantly beats every other method including `xgb` (0.142).

## Lessons carried from the PXR challenge

Hard-won on a completed blind challenge ([repo](https://github.com/adlvdl/pxr_challenge), [blog](https://www.delavega.ai/blog.html)). They cost real leaderboard places; do not relearn them.

- **Do not rank models on differences the data cannot resolve.** In PXR, three finalists spanned 0.0039 MAE and the ordering reversed completely between phases, costing five places. Sixteen of seventeen submissions were statistically indistinguishable. Use `evaluation.paired_bootstrap` before believing any ranking pairwise, `holm_bonferroni` when comparing many variants, and `mcs.make_mcs_grid` for a repeated-measures-ANOVA view across every method at once (same idea as PXR's own MCS heatmaps) — a grid with few or no stars is telling you the CV cannot support a ranking, same as a high paired-bootstrap p-value.
- **Regression to the mean is the dominant failure mode**, and no amount of tuning fixed it in PXR. It is already visible here (LightGBM underpredicts potent CYP3A4 compounds by ~1 log unit). Check `evaluation.bias_by_potency_bin` on every run.
- **Calibration is cheap insurance.** Dismissed in PXR CV at 0.0001 MAE, it won the blind set. Always cross-fit (`calibration.crossfit_calibrate`) for evaluation; fit on all OOF data only for the final test predictions.
- **Ensembling suppresses catastrophic errors** rather than improving typical accuracy — which is what ST-RAE rewards, being a ratio of summed errors. Track `ensemble.catastrophic_rate`.
- **HPO pays off for tree models, not neural nets.** XGBoost went 0.64 → 0.52 MAE in PXR; graph models barely moved. Spend tuning budget accordingly.
- **Multitask/pretraining on auxiliary assay data hurt in PXR** despite helping top teams. Worth trying with the single-concentration screen here, but validate against a model that ignores it.
- **What PXR missed and cost the most:** public data beyond the challenge release, and 3D/structural information — most of the top ten used docking or cofolding. CYP has abundant public inhibition data (ChEMBL, PubChem BioAssay) and well-resolved crystal structures.

## Conventions

- New featurizers go in `fingerprints.py`, registered in `_FP_REGISTRY` so they stay swappable by name. New models go in `models.py`'s `MODEL_FACTORIES`.
- **Exploration and plotting live in four modules, ported from the PXR repo's `1a`–`1e` notebooks** (see `notebooks/02_chemical_space.py` for all of them in use): `embedding.py` (UMAP/t-SNE plus the per-endpoint panel grid), `similarity.py` (pairwise/cross Tanimoto, nearest-neighbour coverage, activity cliffs), `mmp.py` (the `mmpdb` shell-out and the network plot), `scaffolds.py` (ring-system/linked/full scaffold decomposition). `interactive.py` holds the Altair-plus-RDKit hover pattern and is deliberately **not** imported by `cyp/__init__.py` — it pulls in marimo, and `import cyp` must stay usable without the notebook extra. Add to these rather than re-deriving a plot in a notebook.
- **UMAP on fingerprints uses Jaccard distance, not Euclidean.** Jaccard on a bit vector is 1 − Tanimoto, i.e. the similarity a chemist actually reasons about; Euclidean on a 4096-bit vector is dominated by heavy-atom count. `embedding.add_umap` defaults to it and binarises counts first.
- **A Tanimoto threshold is meaningless without naming its fingerprint.** `similarity.CLIFF_SIM_THRESHOLDS` carries the working values (MACCS ≥ 0.8, ECFP4 ≥ 0.4); check them against `similarity.similarity_percentiles` on the current snapshot before trusting a cliff count, since a threshold above a fingerprint's p99.9 selects noise.
- **`cv.murcko_scaffold` and `scaffolds.decompose` answer different questions.** Murcko gives one scaffold per molecule and is for grouping CV folds. The scaffold *network* gives every ring system, linked pair and full core, and is for coverage: a test compound can be Murcko-novel while every ring in it is well represented in training, which is a very different modelling situation from genuine novelty.
- **Timings are recorded to `experiments/<notebook>/timings.csv` as each measurement is taken, via `cyp.timings.record`.** This is a consequence of caching CV: on every run after the first the training branch is skipped, so a timer living inside that branch measures nothing, and the committed run — the one anyone reads later — reports an empty table. Writing per measurement also means a run that dies partway through a multi-hour sweep still leaves behind the timings for everything that finished. `timings.summary` gives per-method totals and `timings.extrapolate` projects a quick-mode run onto the full 5×5, which is how you decide whether a method is affordable *before* starting it rather than after. The CSV is committed like other derived summaries; re-timing a method replaces its row rather than appending, so totals cannot silently double.
- A cache that is large and purely derived from the data snapshot (no training involved) is gitignored rather than committed — see the `experiments/**/cache/` rules in `.gitignore`. Committed caches are the ones that took model fits to produce. Derived summaries and figures are always committed.
- Any plot a notebook generates is saved as a PNG under that notebook's `experiments/<notebook_name>/` directory (`fig.savefig(..., dpi=300, bbox_inches="tight")`), not left only as inline notebook output — see `mcs.make_mcs_grid`'s `save_path` for the pattern.
- Modelling work is a marimo notebook per run: `notebooks/NN_name.py` (pure `.py`, diffs in git). Anything reused across notebooks belongs in `src/cyp/`, not copy-pasted between notebooks.
- A notebook writes its diagnostics to `experiments/<notebook_name>/` and its submission(s) to `submissions/<notebook_name>/<date>/`, deriving the name from its own filename (see `NOTEBOOK_NAME = Path(__file__).stem` in `01_baseline.py`) rather than hardcoding it — this is what ties a submission back to the exact code that produced it. Both tracks (`activity.csv`, `tdi.csv`) from the same run share one dated subfolder; a notebook producing submissions from different runs gets one dated subfolder per run. Every submission folder gets a `PROVENANCE.md` recording model, calibration, CV protocol, data snapshot, **and an expected-performance table (per endpoint plus the macro-average) for both tracks** — computed fresh from CV for the exact submitted method/calibration, not reused from a diagnostic table keyed on a different (possibly different) method. This is what lets you check the interim leaderboard reveal against a number instead of a surprise; a large gap between expected and actual is worth investigating, not shrugging off.
- Classification (TDI) gets its own model factories (`models.CLASSIFIER_FACTORIES`), CV harness (`models.run_cv_classification`) and comparison table (`evaluation.compare_methods_classification`) rather than branching the regression versions — the OOF schemas differ (no credible intervals, boolean predictions) and the metrics are unrelated (MCC vs ST-RAE). `models.MajorityBaseline` is the classification analogue of `MeanBaseline`: MCC = 0.0 by construction, the line a real classifier must clear.
- Save OOF predictions as `oof.parquet` in the experiment directory. Every comparison, calibration and ensemble step reads that schema — see `cv.oof_frame`.
- **Cache CV, not the final fit.** CV (5x5 folds x methods x endpoints) is the slow, repeatable part of a notebook, so cache its OOF output to `experiments/<notebook_name>/cache/`, gated on `if cache_path.exists(): read else: compute+write`, keyed by whatever varies the result (mode, method, endpoint/isoform) — see `01_baseline.py`'s CV cells. Do **not** cache the final fit-on-everything-and-predict-the-test-set step in its own file: it is cheap by comparison (one fit per endpoint, not 25 folds) and a separate cache for it can silently go stale against a retrained CV cache — a real bug caught while building this, where the submission kept using calibration parameters from before a CV retrain. Delete a cache file (or bump a mode string) to force a retrain; commit the cache directory like other experiment outputs.

## Multitask vs single-task

Every harness in `models.py` trains **one model per endpoint**; `multitask.py` is the controlled alternative, so "does sharing across endpoints help?" gets measured rather than assumed. PXR found multitask and auxiliary-assay training *hurt* there, so the multitask arm always runs against its single-task twin, never as a replacement.

**Folds must be shared, and this is a correctness issue, not tidiness.** 1,309 of the 4,905 labelled compounds carry more than one endpoint. With per-endpoint `scaffold_splits`, a compound can be in CYP3A4's training set and CYP2D6's test set at once — so a multitask model would be scored on a compound whose label it had partly seen. `cv.shared_scaffold_folds` assigns folds over the *union* of compounds and `cv.fold_assignment_splits` applies them per endpoint; `tests/test_multitask.py` asserts zero cross-endpoint overlap on the real data. Sharing folds is also what makes the arms *paired*, so `evaluation.paired_bootstrap` between them is legitimate.

**"Multitask" means three different things**, because the model families differ in what they can express — `multitask.STRATEGY` maps each method to its own:

- `stacked` (trees, TFMs): one model on all 6,525 rows with the endpoint one-hot encoded. The only option for a single-output regressor. Without the indicators the model sees one compound with four different targets and learns their average, which is worse than any single-task model.
- `native` (Macau): a genuine sparse `(n_compounds, 4)` factorization. Missing entries are absent rather than imputed — the format matrix factorization is built for — and latent compound factors are shared by construction.
- `multitarget` (Chemprop/CheMeleon): one D-MPNN with four output heads, trained on a wide frame where unmeasured cells are **empty, not zero**. Chemprop masks blanks in its loss; a 0.0 would be read as a real measurement of an extremely inactive compound. Only 41 of 4,905 compounds have all four endpoints, so dropping incomplete rows is not an option.

**Early signal (1 outer repeat, ECFP4-1024, 2026-09-12):** stacked LightGBM beat single-task on all three weak endpoints (CYP1A2 0.893 vs 0.987, CYP2C9 0.936 vs 0.977, CYP2D6 1.031 vs 1.095) and was flat on CYP3A4 (0.737 vs 0.735) — consistent with weak endpoints borrowing strength from the one that has enough rows of its own. This is 5 folds and one paired bootstrap short of a conclusion; do not act on it before the full 5×5.

## Environment traps on this machine (16 GB Apple Silicon)

Both of these were found the hard way while building `notebooks/03_methods.py`. Neither fails cleanly, which is why they are written down rather than left to be rediscovered.

**OpenMP load order — `import cyp` must come first.** Three packages here vendor their own OpenMP runtime: `lightgbm` (`lib_lightgbm.dylib`), `scikit-learn` (`sklearn/.dylibs/libomp.dylib`) and `torch` (`torch/lib/libomp.dylib`). Whichever loads first wins and the others bind to it. If LightGBM is not first, computing a fingerprint (which pulls in sklearn's) and then fitting LightGBM **segfaults** — a hard process death, no catchable exception. `src/cyp/__init__.py` imports lightgbm before anything else to claim the slot; keep that import at the top. `KMP_DUPLICATE_LIB_OK=TRUE` does *not* fix this — it was tried. The conflict is also bidirectional: once LightGBM's runtime is loaded, an in-process TabICL or TabPFN fit segfaults instead, which is why every torch model runs in a subprocess (`tabular_models.predict_subprocess`, `graph_models.chemeleon_embed`, the Chemprop CLI, and `graph_models.device`). Do not "simplify" any of those into an in-process call.

**The two tabular foundation models need separate memory budgets.** They fail differently, so one shared setting mis-serves both. Measured at n=2335 (CYP3A4), PCA-reduced from a 2048-dim CheMeleon embedding:

| | features | n_est | peak RSS | time | corr |
|:--|--:|--:|--:|--:|--:|
| TabICL | 128 | 2 | 3.89 GB | 5s | — |
| TabICL | 128 | 8 | 10.96 GB | 35s | — |
| TabPFN | 128 | 2 | 4.50 GB | 10s | 0.158 |
| TabPFN | **192** | **2** | **6.58 GB** | **9s** | **0.232** |
| TabPFN | 256 | 4 | 8.44 GB | 22s | 0.240 |
| TabPFN | 500 | 4 | 10.83 GB | 140s | 0.236 |

TabICL's memory scales hard with *ensemble size* (2→8 estimators roughly tripled it); TabPFN's barely moves with it (+0.7 GB) but scales with *features*, and unlike TabICL it genuinely uses them — accuracy climbs from 0.158 at 128 features to 0.232 by 192, then plateaus. Hence `TABICL_MAX_FEATURES = 128` / `TABICL_N_ESTIMATORS = 2` versus `TABPFN_MAX_FEATURES = 192` / `TABPFN_N_ESTIMATORS = 2`, with `check_tabicl_budget` and `check_tabpfn_budget` enforcing them. Treat every figure as ±1–2 GB (allocator reuse makes them non-monotonic across runs) and leave headroom.

**TabPFN's token must be the API key, not a session JWT.** Put `TABPFN_TOKEN` in `.env` at the project root (gitignored); `tabular_models.load_env` loads it automatically, and the subprocess runner passes the environment to its child. The value has to be the API key from the account page at ux.priorlabs.ai — it looks like `tabpfn_sk_...`. A JWT copied out of a browser session decodes as a valid unexpired token but gets a 401 "Invalid authentication credentials" from `api.priorlabs.ai/protected/`, and TabPFN reports that identically to having no token at all. If TabPFN claims no token while one is clearly set, check the HTTP status against that endpoint before anything else.

**TabICL memory will freeze the machine.** It attends over the feature axis and materializes the full context, so peak RSS grows with both feature count and ensemble size. Measured at n=2335 (CYP3A4): 2048 features x 8 estimators never finished and swapped the machine to a standstill; 128x8 took 10.96 GB; 128x4 took 7.17 GB; **128x2 took 3.89 GB**. Accuracy was identical (corr 0.999) across all of them on recoverable synthetic signal, so the extra memory buys nothing. Hence `tabular_models.TABICL_MAX_FEATURES = 128` (PCA-reduced, fitted per fold) and `TABICL_N_ESTIMATORS = 2`, with `check_tabicl_budget` refusing an over-budget config before it allocates. There is no `MemoryError` to catch here — the desktop locks up and the run reports nothing — so raise those limits only while watching actual RSS. For scale: Chemprop is ~2 GB per fit and the tree models are under 1 GB.

## Scope

Ask before: adding a heavyweight dependency (deep-learning frameworks, pretrained model downloads), changing anything in `vendor/`, or overwriting an existing file in `submissions/`.
