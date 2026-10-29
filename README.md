# cyp_competition

Code for the [OpenADMET CYP Inhibition Blind Challenge](https://huggingface.co/spaces/openadmet/cyp-challenge) — predicting cytochrome P450 inhibition across four human isoforms.

CYP enzymes metabolize most marketed small-molecule drugs, so unexpected inhibition causes drug–drug interactions. Time-dependent (mechanism-based) inhibition is worse still: the interaction persists after the parent compound clears.

## Final submission

The repo is final until the test labels are released. The entry on the live board is two files, one per track, from two submissions that are both scored:

| track | file | live score | rank |
|:--|:--|--:|--:|
| Direct inhibition | [31_interval_loss/20261026_gated/activity.csv](submissions/31_interval_loss/20261026_gated/) | macro ST-RAE **0.5573** | 74/261 |
| TDI | [24_tdi_encoders/20261023/tdi.csv](submissions/24_tdi_encoders/20261023/) (copied unchanged into the 31 folder) | macro MCC **0.2768** | 79/137 |

Live scores cover half the test set. Ranks are from the 2026-10-26 snapshot ([data/leaderboard/20261026/](data/leaderboard/20261026/)) and are only comparable within one snapshot. The entry uses no proprietary data and the code is open.

| endpoint | ST-RAE | Spearman | | isoform | MCC |
|:--|--:|--:|:--|:--|--:|
| CYP1A2 | 0.5711 | 0.808 | | CYP3A4 | 0.3906 |
| CYP2C9 | 0.4093 | 0.834 | | CYP2D6 | 0.1629 |
| CYP2D6 | 0.7997 | 0.440 | | | |
| CYP3A4 | 0.4489 | 0.842 | | | |

### Regression model (submission 31, gated)

A per-endpoint Caruana blend of four members. Every blind-set member is a seed mean, and every member is fitted on 100% of the labelled rows:

| member | what it is | seeds | public data |
|:--|:--|--:|:--|
| `pubchem` | Chemprop D-MPNN, encoder pretrained on the PubChem NCGC qHTS CYP panel, then fine-tuned multitask on the four endpoints | 5 | PubChem qHTS (pretraining) |
| `tabicl_monroe` | TabICL on frozen [Monroe](https://github.com/blazejba/monroe) 3D embeddings (PCA 128) | 5 | Monroe's own pretraining corpus |
| `tabicl_atom_jepa` | TabICL on frozen Atom-JEPA embeddings | 5 | none by us (encoder weights are third-party) |
| `pooled_workbench` | One from-scratch multitask Chemprop on challenge rows plus public rows, 17 heads | 3 | PubChem qHTS `max_response`, ChEMBL 37 |


Weights are equal splits among the selected members, fitted on all 25 scaffold folds ([weights.csv](submissions/31_interval_loss/20261026_gated/weights.csv)):

| endpoint | `pubchem` | `tabicl_monroe` | `tabicl_atom_jepa` | `pooled_workbench` |
|:--|--:|--:|--:|--:|
| CYP1A2 | 0.20 | 0.40 | 0.20 | 0.20 |
| CYP2C9 | 0.33 | 0.33 | 0 | 0.33 |
| CYP2D6 | 0.25 | 0.25 | 0.25 | 0.25 |
| CYP3A4 | 0.33 | 0.33 | 0 | 0.33 |

Two choices separate 31 from the earlier 22:

1. **Interval loss.** `pubchem` and `pooled_workbench` are retrained with `interval-mae`, which is zero inside a compound's credible interval and linear outside it, the same shape as ST-RAE ([notebooks 28 to 31b](#ideas-from-other-entrants)). CYP2D6 keeps the L1-trained pooled seeds because the interval version was worse there in CV.
2. **Gated placement.** CYP1A2 and CYP2D6 are affine-mapped onto the moments solved in notebooks 07 and 12. CYP2C9 and CYP3A4 are left at the blend's own mean and spread, because a fixed-moment placement erased about 85% of the interval gain in CV (31b) and had cost those two endpoints ST-RAE in 07.

Progress on the live board (macro ST-RAE): 12 `0.6578` → 17 `0.6209` → 22 `0.5711` → 31 `0.5573`. The gain from 22 to 31 landed on the two unplaced endpoints (CYP3A4 −0.035, CYP2C9 −0.016), as CV predicted. The macro Spearman moved only 0.724 → 0.731: the gains since 22 are position relative to the credible intervals, not ordering.

### TDI model (submission 24)

A per-isoform Caruana blend of six arms (MCC at a fixed threshold of 0.25), each refit on 100% of the labelled rows and seed-averaged:

| arm | encoder | seeds |
|:--|:--|--:|
| `tabicl_singletask` | frozen CheMeleon | 5 |
| `tabicl_monroe` | frozen Monroe | 5 |
| `tabicl_atom_jepa` | frozen Atom-JEPA | 5 |
| `chemprop_pubchem` | PubChem-pretrained, fine-tuned | 3 |
| `chemprop_pubchem_frozen` | PubChem-pretrained, encoder frozen | 3 |
| `chemprop_scratch` | from scratch | 3 |

CV macro MCC was 0.2945 against 0.2650 for the previous four-arm blend (notebook 23, Holm-significant). The live half scored **0.2768**, up from 0.2028 for the earlier blend, so unlike that one the CV gain held on the board. The A1 rule from notebook 27 (train TDI only on rows that have a direct-inhibition arm) improved CV by about +0.03 but **was not built into any submission**.

### Expected against live

| submission | CV expectation | live |
|:--|--:|--:|
| 31 regression (gated) | about 0.627 (22's 0.639 minus roughly 0.012) | 0.5573 |
| 24 TDI | 0.2945 | 0.2768 |

CV cannot see placement and the live half is small, so neither number forecasts the final test score.

## Quick start

```bash
make setup                     # uv sync — creates .venv with all dependencies
make test                      # smoke tests: data loads, splits hold, submissions validate
make baseline                  # run notebooks/01_baseline.py end to end → a validated submission
make lab                       # marimo edit — open notebooks interactively (toggle "Quick mode" there)
```

The challenge datasets are committed under [data/raw/](data/raw/), so no download is needed to start. Public CYP data is under [data/external/](data/external/). See [Data snapshots](#data-snapshots) for how re-downloads are tracked.

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

### Public data (regression only)

The challenge allows public data if it is disclosed. Three public CYP sources are downloaded into dated snapshots under [data/external/](data/external/), each by its own module. **No public label is ever written to a scored endpoint column.** Each source gets its own head on its own scale, and public structures matching any challenge compound (train or blind) are removed by InChIKey connectivity block, not SMILES (`external.inchikey_skeleton`).

| source | module | snapshot | what it supplies | used in the final regression |
|:--|:--|:--|:--|:--|
| NCGC qHTS CYP panel, PubChem AIDs 410 (CYP1A2), 883 (CYP2C9), 884 (CYP3A4), 891 (CYP2D6). Veith et al. | `cyp.external` | `20260915`, `20260916` | ~12,900 compounds with pIC50 on one protocol, plus `max_response` (efficacy) at near-full coverage, which carries the inactive half of the library | `pubchem` pretraining; `max_response` heads in `pooled_workbench` |
| ChEMBL 37, CYP1A2 / 2C9 / 2C19 / 2D6 / 3A4 | `cyp.chembl` | `20260916` | 41,403 compounds, ~14,900 new skeletons, no low-end range (only fitted curves carry a `pchembl_value`) | pIC50 heads in `pooled_workbench` |
| Tox21 P450-Glo, CYP2C9 / 2D6 / 3A4 (no CYP1A2), filtered by the luciferase counter-screen | `cyp.tox21` | `20260916` | ~2,400 explicit inactives per isoform | in the union matrix only, **not in the final model** |

The challenge's own unused arms also act as auxiliary targets (`aux_training.challenge_auxiliary_matrix`, `auxiliary.screen_targets`): the single-concentration screen log2fc, TDI-condition pIC50, and Emax. These need no download.

`aux_training.full_union_matrix` assembles all sources into 24,718 compounds × 16 heads × 125,193 labels. Notebook 05 found that pretraining on the PubChem panel helps only with the encoder unfrozen. Notebook 20 found a from-scratch model trained on the pooled rows is worse alone and improves the blend (macro ST-RAE −0.021), because its errors are decorrelated from the other members.

The TDI track uses **no public CYP data**. The PubChem-pretrained TDI arms reuse the same qHTS pretraining, binarised (`aux_training.binarise_pretraining_matrix`).

### Pretrained encoders (third-party weights)

Frozen embeddings from [Monroe](https://github.com/blazejba/monroe) and Atom-JEPA (the `encoders` extra, `cyp.encoders`), and CheMeleon (`graph_models.chemeleon_embed`). Monroe pretrains on PCBA, which contains the Veith panel the challenge derives from. 225 of 4,905 training compounds and 1 of 750 blind compounds carry a label there (`scripts/audit_monroe_pcba_overlap.py`). Dropping the 225 leaves the CV gains unchanged.

### Third-party data

**xTB conformer ensembles and semi-empirical descriptors** for every challenge compound, published by Oleg Gromov (challenge user `Shorku`) in [Shorku/OpenADMET_Blind_Challenge_Predicting_CYP_Inhibition](https://github.com/Shorku/OpenADMET_Blind_Challenge_Predicting_CYP_Inhibition/tree/a98faaf41e0630046d0c6f11bd11b6cfe55687fc/data) under the MIT licence (Copyright (c) 2026 Oleg Gromov), pinned at commit `a98faaf`. Method: O. Gromov, *J. Chem. Theory Comput.* 2025, [doi:10.1021/acs.jctc.5c00425](https://doi.org/10.1021/acs.jctc.5c00425). 12,037 structures, 102,624 conformers. `python -m cyp.shorku_qm` fetches it to `data/external/shorku_qm/YYYYMMDD/` (`geoms.zip`, 134 MB, is gitignored and hash-checked on download). Used by `scripts/probe_xtb_monroe.py` and `notebooks/26_monroe_conformers.py`.

## Data snapshots

OpenADMET has amended the dataset mid-challenge, so downloads are kept as dated snapshots rather than overwritten:

```
data/raw/20260908/    cyp-challenge-*.csv + the dataset card
data/raw/20261020/    re-download after the mid-challenge amendment; same row counts
```

The final submissions were built on snapshot `20261020`. The baseline and early notebooks used `20260908`.

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
  graph_models.py    Chemprop (single-task, multitarget, pretrained) and CheMeleon embeddings, all via subprocess
  tabular_models.py  TabICL / TabPFN in isolated child processes
  encoders.py        frozen Monroe / Atom-JEPA embeddings
  aux_training.py    auxiliary/public-data label matrices, pretraining CV, pooled multitask CV
  external.py, chembl.py, tox21.py   public-data downloaders (dated snapshots under data/external/)
  shorku_qm.py       third-party xTB conformer ensembles (data/external/shorku_qm/), see Data
  interval_loss.py, interval_launcher.py   ST-RAE-shaped training loss injected into the Chemprop CLI
  auxiliary.py       challenge-side auxiliary targets (screen log2fc, TDI-condition pIC50, Emax)
  pharmacophore.py, pka.py, docking.py, plif.py, unimol.py   CYP2D6 structure experiments (notebooks 09 to 11), none shipped
  placement.py       affine placement of predictions and the blind-set moment solve
  multitask.py       stacked / native / multitarget multitask harnesses with shared folds
notebooks/           marimo notebooks, one per modelling run (NN_name.py); build_*.py scripts assemble submissions
scripts/             leaderboard snapshots, Monroe/PCBA overlap audit, contention watcher
data/leaderboard/    dated snapshots of the public leaderboards
experiments/<nb>/    diagnostics written by notebook <nb> (OOF predictions, CV summary, ...)
submissions/<nb>/    submissions written by notebook <nb>, one dated subfolder each, with PROVENANCE.md (incl. expected CV performance, per-endpoint and macro-averaged)
vendor/              pinned copy of the official tutorial's scoring code
tests/               smoke tests
```

A notebook's outputs are named after the notebook itself: `notebooks/01_baseline.py` writes to `experiments/01_baseline/` and `submissions/01_baseline/<date>/`, so a submission always traces back to the exact code that produced it.

`cv.py`, `calibration.py`, `ensemble.py`, `evaluation.py`, `mcs.py` and `fingerprints.py` are ported from the [PXR blind challenge repo](https://github.com/adlvdl/pxr_challenge), adapted to Polars and to CYP's four sparse endpoints. See [PLAN.md](PLAN.md) for what that challenge's retrospectives imply here.

## Current status

**Latest (2026-10-29).** The repo is final pending the test-label release. The submitted entry is described in [Final submission](#final-submission). Full findings per notebook are in [CLAUDE.md](CLAUDE.md).

| submission | regression macro ST-RAE (live) | TDI macro MCC (live) | status |
|:--|--:|--:|:--|
| [12_final_placement](submissions/12_final_placement/20260919/) | 0.6578 (0.6384 on the full set at the interim reveal) | 0.2111 | scored |
| [17_regression_ensemble](submissions/17_regression_ensemble/20261020/) | 0.6209 | 0.2028 | scored, regression rank 85/257, TDI rank 121 on 2026-10-21 |
| [22_pooled_seed_ensemble](submissions/22_pooled_seed_ensemble/20261023/) | 0.5711 | 0.2028 (17's) | scored, rank 66/252 on 2026-10-23 |
| [24_tdi_encoders](submissions/24_tdi_encoders/20261023/) | 0.5711 (22's) | **0.2768** | scored, TDI rank 77/134 on 2026-10-24 |
| [31_interval_loss](submissions/31_interval_loss/20261026_gated/) (**final**) | **0.5573** | 0.2768 (24's) | scored, rank 74/261 on 2026-10-26 |

An ungated build of 31 sits in `submissions/31_interval_loss/20261026/` (placement on all four endpoints, not recommended). It was not submitted.

Scores are from the live leaderboard (half the test set), snapshots in [data/leaderboard/](data/leaderboard/). Ranks are only comparable within one snapshot because the field keeps growing. Cross-validation cannot see placement, so a CV number is not a forecast of the board.

Not shipped, with CV evidence: the TDI A1 rule and gate feature (notebooks 27, 29), and the xTB conformer embeddings (notebook 26, no measurable gain).

### Baseline (notebook 01)

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

## Running notebook 11 (docking / Uni-Mol)

[notebooks/11_docking_3d.py](notebooks/11_docking_3d.py) is the only code needing `unimol_tools`, and **it is not installed by `make setup`**. It pins `numpy<2.3.0`, and numpy 2.2.x makes TabICL return all-NaN probabilities on a ~2,400-row classification fit while raising nothing — which silently scores the whole TDI track at MCC 0.000. `pyproject.toml` therefore pins `numpy>=2.4` and leaves `unimol_tools` out of every extra; `uv sync` fails outright if both are requested.

Uni-Mol already runs in a child process, so it does not have to be the same interpreter. Point `CYP_UNIMOL_PYTHON` at a throwaway environment:

```bash
uv venv /tmp/unimol                                    # separate env, older numpy
VIRTUAL_ENV=/tmp/unimol uv pip install unimol_tools
CYP_UNIMOL_PYTHON=/tmp/unimol/bin/python uv run python notebooks/11_docking_3d.py
```

Unset, the notebook still reruns as committed: its CYP2D6 Uni-Mol embeddings are cached under [experiments/11_docking_3d/cache/](experiments/11_docking_3d/cache/), and the cached branch never starts a child. Only a *new* isoform, or a cleared cache, actually calls Uni-Mol — and that raises rather than skipping, so set the variable first if you are extending the run. Everything else in notebook 11 (smina docking, ProLIF interaction fingerprints) needs no extra environment.

`unimol.available()` reports whether the child interpreter can import the package, and is the check to use before adding a new Uni-Mol arm.

The docking itself also needs `smina` (`brew install brewsci/bio/smina`); gnina is CUDA-only and cannot run on an arm64 Mac. Budget generously — docking CYP2D6 alone took 15.6h here, with per-chunk times varying 30x under ordinary desktop contention.

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

- Teams using proprietary CYP data must disclose it.
- An **Innovation in ML Award** is judged separately from leaderboard rank — novel architectures, creative data usage, and uncertainty quantification all count.
- Open-sourcing your code earns community recognition.

## Ideas from other entrants

Two ideas came from the SuperCowPowers workbench notes and were adopted well before the others. On 2026-10-25 we also read the public model reports linked from the live leaderboard (40 distinct links, 39 reachable) and took three more. Ideas only: none of their code, data or checkpoints is used here. We did not use the pretrained checkpoints in jeremy's repo because it carries no licence.

| idea | taken from | tested in |
|:--|:--|:--|
| **Placement.** Separate a prediction vector's ordering from its position on the pIC50 axis. An affine transform sets mean and spread without changing any rank, and the calibration that maximises R² is not the one that minimises ST-RAE. Adopted 2026-09-16, the earliest idea here. The R² decomposition (`R2 = 2*rho*k - k^2 - b^2`), the `k = rho` optimum, the blind-moment solve and the per-endpoint gating are our own work. We did not reuse their solved constants. | SuperCowPowers (Workbench), [working notes](https://supercowpowers.github.io/workbench/blogs/cyp_challenge/), [code](https://github.com/SuperCowPowers/workbench/tree/main/ml_pipelines/OpenADMET/cyp). Described in [Phase 1 write-up](https://www.delavega.ai/posts/2026_10_20_cyp_phase1.html). | [07](notebooks/07_placement.py), [12](notebooks/12_final_placement.py), `src/cyp/placement.py`. Applied to CYP1A2 and CYP2D6 in the final regression; gated off for CYP2C9 and CYP3A4 |
| **Pooled multitask model.** One from-scratch Chemprop trained on challenge rows plus public rows, with public sources as extra heads on extra rows and down-weighted (`union-p30`). | SuperCowPowers, same notes and code. We replicated the recipe, with public rows disjoint from every challenge compound. | [20](notebooks/20_pooled_workbench.py), [21](notebooks/21_pooled_ensemble.py): worse alone, macro ST-RAE −0.021 inside the blend. Ships as `pooled_workbench` |
| Train TDI only on compounds that have a direct-inhibition arm. 1,249 of the 3,584 labelled CYP3A4 rows have none and are all negative, which dilutes the positive rate from 32.7% to 21.3% against a blind set where every compound has both arms. | JacksonBurns, [NOTES.md](https://github.com/JacksonBurns/cyp-challenge/blob/86213d34157eeeea7635b6943c3f8c3798548514/NOTES.md) (sec. 23). We verified the counts on our own snapshot. | [27](notebooks/27_tdi_prior_and_gate.py), [29](notebooks/29_tdi_blend_a1.py): +0.033 macro MCC on one readout, +0.030 inside the Caruana blend (CV only) |
| Feed TDI models a leak-free predicted direct pIC50 as a gate feature. | Nomad-link-id (Igor Eduardo, Cortexa Social), [methods report](https://huggingface.co/Nomad-link-id/openadmet-cyp-methods) | [27](notebooks/27_tdi_prior_and_gate.py): +0.015, adds nothing on top of the rule above |
| Train against the credible interval (zero loss inside `[conf_low, conf_high]`) instead of the point estimate, matching ST-RAE. | mokhalaf22, [REPORT.md](https://github.com/mokhalaf22/openadmet-cyp-public/blob/main/REPORT.md); doctawho42 (Lizard Wizard), [method report](https://github.com/doctawho42/CYPChallenge/blob/main/docs/method_report.md) | [28](notebooks/28_interval_loss.py), loss in `src/cyp/interval_launcher.py`: macro ST-RAE −0.023 on from-scratch multitask chemprop, Spearman unchanged. [30](notebooks/30_interval_loss_shipped.py) on the shipped models: `pubchem` −0.046 vs its MSE config and −0.027 vs L1 (25 folds, Holm-significant), pooled model −0.037 vs L1 (5 folds, CYP2D6 worse), Spearman unchanged in both |

Also read and not adopted: [jeremy](https://github.com/jeremycheminf/openadmet_scripts/tree/main/CYP_Challenge) (closest to our stack, but its edge on TDI is an encoder pretrained on a corpus we cannot reuse), [stir_bar](https://github.com/lachrymator/openadmet-cyp-challenge-public) (pulls near-neighbours of the blind structures into training; JacksonBurns measured +0.0002 from the same move), [KNIMEST](https://github.com/KNIMEST/openadmet-cyp-model-report) (its 0.81 to 0.60 jump came from in-sample predicted log2FC, chosen after seeing the blind score, and moved R² far more than Spearman).

## Links

- [Challenge Space](https://huggingface.co/spaces/openadmet/cyp-challenge) · [Dataset](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test)
- [Official tutorial repo](https://github.com/OpenADMET/CYP-Challenge-Tutorial)
- [Announcement post](https://openadmet.ghost.io/announcing-openadmets-cyp-inhibition-blind-challenge/) · [Underway post](https://openadmet.ghost.io/openadmets-cyp-challenge-is-underway/)
- Discord: `#cyp-challenge`
