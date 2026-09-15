# Method-comparison submission — TDI track

- Generated: 2026-09-15
- Notebook: `notebooks/04_methods_tdi.py`
- Data snapshot: `20260908`
- CV: nested scaffold, 5x5 folds, shared across isoforms (`cv.shared_scaffold_folds`)

## Time-dependent inhibition (classification)

- Model: **tabicl_singletask** — TabICL in-context classifier, one model per isoform
- Representation: frozen CheMeleon embeddings (2048-dim), the configuration
  arXiv 2604.16123 found best for tabular foundation models on molecular tasks
- Decision threshold: **0.25**, tuned on OOF predictions against macro MCC
- Validation: PASSED (`submission.check(..., track="tdi")`)

**Expected performance (5x5 CV, threshold 0.25, higher MCC is better;
0.0 = no better than predicting the majority class):**

| Endpoint | Expected MCC (mean) | std across folds |
|:--|--:|--:|
| CYP3A4 | 0.3750 | 0.0408 |
| CYP2D6 | 0.1216 | 0.0467 |
| MA (macro-average) | **0.2483** | 0.0265 |

Compare the `MA (macro-average)` row against the interim leaderboard's macro MCC —
that row, not either isoform, determines rank.

For reference, `01_baseline`'s LightGBM/ECFP4 submission scored CYP3A4 0.259 /
CYP2D6 0.116 at the default 0.5 threshold.

## Why this model

It won the comparison in `experiments/04_methods_tdi/` on the macro-average, and the
margin is one the CV can resolve. On the MCS grid (repeated-measures Tukey HSD,
`mcs_heatmap_tdi_multitask_macro.png`) `tabicl_singletask` is significantly better
than `tabpfn_singletask` (p<0.05), `lgbm_singletask` (p<0.01), `xgb_singletask`
(p<0.05), both chemprop arms (p<0.01) and the majority baseline (p<0.01).

It is **not** separable from its own multitask twin, which is the next section.

## Single-task, because multitask does not help here

This notebook's whole purpose was to test whether the multitask win from
`03_methods` (regression) transfers to TDI. **It does not.** Paired bootstrap on
identical folds, 25,405 paired measurements per model, Holm-corrected across the
family of five:

| model | diff (multitask − single-task) | 95% CI | raw p | Holm |
|:--|--:|:--|--:|:--|
| xgb | −0.0330 | [−0.045, −0.021] | <0.0001 | **significant — multitask hurts** |
| tabicl | −0.0124 | [−0.024, −0.000] | 0.046 | not significant |
| lgbm | — | — | 0.171 | not significant |
| chemprop | −0.0061 | [−0.019, +0.008] | 0.380 | not significant |
| tabpfn | −0.0031 | [−0.015, +0.009] | 0.606 | not significant |

Every point estimate is negative and the only one surviving correction shows
multitask *hurting*. The structural reason is in CLAUDE.md: the regression win rests
on 1,309 of 4,905 compounds (26.7%) carrying several endpoints, but only **259 of
4,822 TDI compounds (5.4%)** appear in both isoforms — the lever is five times
scarcer, across half as many tasks. Single-task is both better and simpler, so it
ships.

Note also that chemprop, which won the regression track outright, is **last** among
real methods here (0.152 vs TabICL's 0.248).

## Read this before comparing against the leaderboard

**The test set is predicted far more positive than the training set.**

| | training rate | OOF at t=0.25 | **test set** |
|:--|--:|--:|--:|
| CYP3A4 | 21.3% | 33.1% | **48.8%** |
| CYP2D6 | 21.6% | 24.4% | **28.8%** |

Checked, not assumed. Two things rule out a bug: predicting the *training* compounds
with the final model returns a mean probability of 0.212 against a 21.3% true rate —
calibrated exactly — while the test compounds return 0.291. And the same
over-prediction is visible in CV (33% at t=0.25), which is the behaviour that earned
the best MCC in the first place: on an imbalanced label, MCC is maximized by trading
precision for recall.

Two effects compound on the test set. The blind set is 75 hits x ~10 analogs each
(CLAUDE.md), so it is deliberately enriched around active scaffolds and should
genuinely carry more positives than the training distribution. And TabICL is an
in-context learner, so a final fit on 100% of the data behaves differently from a
fold fit on 80%.

If the interim reveal shows MCC far below 0.2483, this is the first thing to examine:
the threshold may need raising for the test distribution. The CV cannot answer that —
0.30 costs 0.015 macro MCC and 0.35 costs 0.038 on training folds, so there is no
free safety margin to take.

## Other caveats

**The threshold is tuned on the folds it is scored on**, which is honest for
*choosing* a cutoff but mildly optimistic as a *reported* number — the same caveat
cross-fit calibration carries in `01_baseline`. A more defensible expectation is the
robustness figure: if the shipped threshold is wrong by ±0.10, macro MCC is **0.2102**,
which still exceeds every competing method's own peak (lgbm 0.2092, tabpfn 0.2199).

**MCC is not optimized at 0.5 on this data.** Every method's optimum fell between
0.10 and 0.25, and at 0.5 both chemprop arms score exactly 0.000 — they predict no
positives at all and collapse onto the majority baseline. A comparison at the default
cutoff would have ranked methods largely by how inflated their probabilities happen
to be.

**CV estimates, not leaderboard guarantees.** The real leaderboard bootstraps the
actual 750-compound test set; this resamples training-set CV folds, and the
macro-average matches fold *index* across isoforms rather than resampling the same
compounds for both (see `evaluation.macro_averaged_fold_metrics`).

**Chemprop's folds are not device-homogeneous** — MPS for folds 0-13, CPU for 14-24,
after MPS degraded ~20x mid-run. This does not touch the submitted model (TabICL
never uses the accelerator) and was checked not to matter for chemprop's own numbers
either: permutation test p=0.25 / p=0.52. See `experiments/04_methods_tdi/DEVICE_BOUNDARY.md`.
