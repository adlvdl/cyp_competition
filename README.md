# cyp_competition

Code for the [OpenADMET CYP Inhibition Blind Challenge](https://huggingface.co/spaces/openadmet/cyp-challenge) — predicting cytochrome P450 inhibition across four human isoforms.

CYP enzymes metabolize most marketed small-molecule drugs, so unexpected inhibition causes drug–drug interactions. Time-dependent (mechanism-based) inhibition is worse still: the interaction persists after the parent compound clears.

## Quick start

```bash
make setup                     # uv sync — creates .venv with all dependencies
make test                      # smoke tests: data loads, splits hold, submissions validate
make baseline                  # run notebooks/01_baseline.py end to end → a validated submission
make lab                       # marimo edit — open notebooks interactively (toggle "Quick mode" there)
```

The datasets are committed under [data/raw/20260908/](data/raw/20260908/), so no download is needed to start. See [Data snapshots](#data-snapshots) for how re-downloads are tracked.

## The two tracks

| | Direct Inhibition | Time-Dependent Inhibition |
|:--|:--|:--|
| Task | Regression | Binary classification |
| Isoforms | CYP1A2, CYP2C9, CYP2D6, CYP3A4 | CYP3A4, CYP2D6 only |
| Target | pIC50 | `is_TDI` — IC50 shift > 2-fold after NADPH preincubation |
| Primary metric | Macro-averaged Soft-Threshold RAE (**lower is better**) | Matthews Correlation Coefficient (**higher is better**) |
| Predictions | 750 × 4 floats | 750 × 2 booleans |

**Soft-Threshold RAE** scores a prediction as its distance to the nearest bound of the true value's credible interval — anything landing *inside* the interval scores zero error. The denominator is the same error function applied to a constant mean predictor, so **ST-RAE = 1.0 means "no better than predicting the mean."** Compounds with pIC50 < 4 are downweighted, since that is below assay resolution.

The tracks are scored independently and submitted as two separate CSVs.

## Data

Five files from [`openadmet/cyp-challenge-train-test`](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test):

| File | Rows | What it is |
|:--|--:|:--|
| `cyp-challenge-TRAIN_inhibition.csv` | 4,905 | Direct-inhibition pIC50 + credible intervals + std, per isoform |
| `cyp-challenge-TRAIN_TDI.csv` | 6,145 | `is_TDI` labels, plus pIC50 in both the TDI and direct conditions |
| `cyp-challenge-TRAIN_Emax.csv` | 6,145 | Max effect vs positive control, both conditions |
| `cyp-challenge-single-concentration-TRAIN.csv` | 17,504 | Long-format primary screen (compound × enzyme × plate) |
| `cyp-challenge-TEST-BLINDED.csv` | 750 | Test set — identifiers and SMILES only |

**Targets are sparse.** A compound only has a dose-response curve where the primary screen flagged it, so every target column is mostly NaN:

| Endpoint | Labelled rows |
|:--|--:|
| CYP1A2 pIC50 | 1,412 |
| CYP2C9 pIC50 | 1,285 |
| CYP2D6 pIC50 | 1,493 |
| CYP3A4 pIC50 | 2,335 |
| CYP3A4 `is_TDI` | 3,584 (764 positive) |
| CYP2D6 `is_TDI` | 1,497 (324 positive) |

Filter per-endpoint with `data.labelled_subset(df, endpoint)`. Note the TDI classes are imbalanced (~20%), which is exactly why MCC is the metric.

The single-concentration screen covers far more compounds than any dose-response table and is the main source of auxiliary signal for compounds that never got a curve.

## Data snapshots

OpenADMET has amended the dataset mid-challenge, so downloads are kept as dated snapshots rather than overwritten:

```
data/raw/20260908/    cyp-challenge-*.csv + the dataset card
```

Each `make data` run writes a new `data/raw/YYYYMMDD/` directory and then reports, file by file, what changed against the previous snapshot:

```bash
make data          # download into today's snapshot, then diff vs the previous one
make data-diff     # diff the two most recent snapshots, downloading nothing
```

Loaders default to the most recent snapshot. Pin a date to reproduce an older result:

```python
data.load_train_inhibition()                    # latest
data.load_train_inhibition(snapshot="20260908") # pinned
```

`constants.available_snapshots()` lists what's on disk. Because snapshots are committed, a re-download that changes the data shows up in both the diff report and `git status` — record the snapshot date alongside any result you intend to reproduce.

## Repository layout

```
data/raw/YYYYMMDD/   dated dataset snapshots, committed
src/cyp/
  constants.py       endpoint names, paths — must match the scoring backend
  data.py            loaders, per-endpoint filtering, model-ready frames
  fingerprints.py    fingerprint dispatch (scikit-fingerprints)
  cv.py              nested scaffold/random CV splits
  models.py          baselines + tree models + CV harnesses (regression and TDI classification)
  calibration.py     leakage-free cross-fit calibration
  ensemble.py        weighted blending and weight search
  evaluation.py      paired bootstrap, Holm-Bonferroni, bias diagnostics, macro-average across endpoints
  mcs.py             repeated-measures Tukey HSD + multiple-comparison heatmaps
  metrics.py         local scoring via the official ST-RAE implementation
  submission.py      build + validate the two submission CSVs
  download.py        fetch data into a dated snapshot + diff vs the previous one
notebooks/           marimo notebooks, one per modelling run (NN_name.py)
experiments/<nb>/    diagnostics written by notebook <nb> (OOF predictions, CV summary, ...)
submissions/<nb>/    submissions written by notebook <nb>, one dated subfolder each, with PROVENANCE.md (incl. expected CV performance, per-endpoint and macro-averaged)
vendor/              pinned copy of the official tutorial's scoring code
tests/               smoke tests
```

A notebook's outputs are named after the notebook itself: `notebooks/01_baseline.py` writes to `experiments/01_baseline/` and `submissions/01_baseline/<date>/`, so a submission always traces back to the exact code that produced it.

`cv.py`, `calibration.py`, `ensemble.py`, `evaluation.py`, `mcs.py` and `fingerprints.py` are ported from the [PXR blind challenge repo](https://github.com/adlvdl/pxr_challenge), adapted to Polars and to CYP's four sparse endpoints. See [PLAN.md](PLAN.md) for what that challenge's retrospectives imply here.

## Current status

Validated baseline submissions for **both tracks** exist at [submissions/01_baseline/20260909/](submissions/01_baseline/20260909/), produced by [notebooks/01_baseline.py](notebooks/01_baseline.py).

**Direct inhibition (regression).** LightGBM on ECFP4 under nested scaffold CV (5x5, 25 folds), mean ST-RAE (**1.0 = no better than predicting a constant**):

| Endpoint | raw | +linear calibration |
|:--|--:|--:|
| CYP3A4 | 0.720 | **0.707** |
| CYP2C9 | 0.961 | 0.883 |
| CYP1A2 | 0.986 | 0.903 |
| CYP2D6 | 1.063 | 0.947 |

Only CYP3A4 clears the mean predictor by a wide margin before calibration; the other three are close to it or above. Closing that gap is the work — see [PLAN.md](PLAN.md).

**TDI (classification).** LightGBM with class-balanced weighting, same CV protocol, mean MCC (**0.0 = no better than always predicting the majority class**):

| Isoform | majority (baseline) | LightGBM |
|:--|--:|--:|
| CYP3A4 | 0.000 | **0.259** |
| CYP2D6 | 0.000 | 0.116 |

**Which differences are real, per endpoint?** [experiments/01_baseline/mcs_heatmap_regression.png](experiments/01_baseline/mcs_heatmap_regression.png) and [mcs_heatmap_tdi.png](experiments/01_baseline/mcs_heatmap_tdi.png) show a repeated-measures Tukey HSD across all 25 CV folds for every pair of methods, per endpoint — mean difference plus significance stars, corrected for testing every pair at once. LightGBM beats the mean baseline at p<0.01 on CYP1A2/CYP2C9/CYP3A4 but only reaches p<0.05 on CYP2D6 — the weakest significance level on the whole grid, and one more sign CYP2D6 needs a different approach rather than more tuning of the same models.

**Which differences are real, on the metric that decides rank?** The competition ranks on the **macro-average across all four endpoints** (the leaderboard backend's "MA" pseudo-endpoint — see CLAUDE.md), not on any single CYP. [mcs_heatmap_regression_macro.png](experiments/01_baseline/mcs_heatmap_regression_macro.png) and [mcs_heatmap_tdi_macro.png](experiments/01_baseline/mcs_heatmap_tdi_macro.png) run the same Tukey HSD on that macro-averaged metric, via `evaluation.macro_averaged_fold_metrics`. On this run: `lgbm` and `xgb` are **not significantly different** on macro-averaged ST-RAE (p=0.239) despite both clearly beating `mean`/`ridge`/`knn` — this is the number that should actually decide a submission, not the per-endpoint panels above.

## Validating a split

The test set is an **analog expansion**: 75 primary hits, each with ~10 close analogs. A random split leaks near-duplicates across folds and produces validation scores that will not track the leaderboard. Use `cv.scaffold_splits` by default; `cv.random_splits` is there as a deliberately optimistic baseline for comparison.

## Producing a submission

```python
from cyp import submission

path = submission.build_activity_submission(
    {"CYP1A2_pIC50_direct_inhibition": preds_1a2, ...},  # 750 values each, test-set order
    "submissions/01_baseline/20260908/activity.csv",
)
submission.check(path, track="activity")

tdi_path = submission.build_tdi_submission(
    {"CYP3A4_is_TDI": preds_3a4, "CYP2D6_is_TDI": preds_2d6},  # boolean labels, not probabilities
    "submissions/01_baseline/20260908/tdi.csv",
)
submission.check(tdi_path, track="tdi")
```

`check` runs the challenge's own validator (vendored from the tutorial), so a file that passes here is one the Space will accept. Also available from the shell:

```bash
make check-activity FILE=submissions/.../activity.csv
make check-tdi      FILE=submissions/.../tdi.csv
```

**One submission per team**, so validate before uploading.

## Scoring locally

```python
from cyp import data, metrics
metrics.score_activity(truth_df, preds_df)   # per-isoform ST-RAE/MAE/R2 + macro-average row
metrics.score_tdi(truth_df, preds_df)        # per-isoform MCC + macro-average
```

These call the vendored official implementation. The real leaderboard bootstraps and reports confidence intervals, so treat small local differences as noise.

## Key dates

All deadlines 23:59 UTC.

| Date | Milestone |
|:--|:--|
| August 17, 2026 | Data released, submissions opened |
| **September 24, 2026** | **Intermediate leaderboard deadline** |
| September 25, 2026 | Intermediate leaderboard — one-time full test set reveal |
| **November 3, 2026** | **Final submission deadline** |
| From November 4, 2026 | Final leaderboard, webinars, wrap-up |

The live leaderboard scores only half the test set (split by chemical series); the full set is scored at the intermediate reveal and at the end.

## Rules worth remembering

- One submission per lab or cooperative group.
- Teams using proprietary CYP data must disclose it.
- An **Innovation in ML Award** is judged separately from leaderboard rank — novel architectures, creative data usage, and uncertainty quantification all count.
- Open-sourcing your code earns community recognition.

## Links

- [Challenge Space](https://huggingface.co/spaces/openadmet/cyp-challenge) · [Dataset](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test)
- [Official tutorial repo](https://github.com/OpenADMET/CYP-Challenge-Tutorial)
- [Announcement post](https://openadmet.ghost.io/announcing-openadmets-cyp-inhibition-blind-challenge/) · [Underway post](https://openadmet.ghost.io/openadmets-cyp-challenge-is-underway/)
- Discord: `#cyp-challenge`
