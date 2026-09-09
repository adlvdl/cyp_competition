# Baseline submission

- Generated: 2026-09-09
- Notebook: `notebooks/01_baseline.py`
- Data snapshot: `20260908`
- CV: nested scaffold, 5x5 folds

## Direct inhibition (regression)

- Model: lgbm on ECFP4 (2048 bits), per-endpoint
- Calibration: linear (fit on all OOF predictions)
- Validation: PASSED

**Expected performance (5x5 CV, cross-fit calibrated, lower ST-RAE is better;
1.0 = no better than the mean):**

| Endpoint | Expected ST-RAE (mean) | std across folds |
|:--|--:|--:|
| CYP1A2 | 0.9033 | 0.0528 |
| CYP2C9 | 0.8826 | 0.0381 |
| CYP2D6 | 0.9472 | 0.0364 |
| CYP3A4 | 0.7069 | 0.0442 |
| MA (macro-average) | 0.8600 | 0.0233 |

Compare the `MA (macro-average)` row against the interim leaderboard's macro ST-RAE for
this submission -- that row, not any single endpoint, is what determines rank.
See `experiments/01_baseline/cv_summary.csv` for the full per-method comparison.

## TDI (classification)

- Model: lgbm on ECFP4 (2048 bits), per-isoform, class-balanced
- Validation: PASSED

**Expected performance (5x5 CV, higher MCC is better; 0.0 = no better than the majority class):**

| Endpoint | Expected MCC (mean) | std across folds |
|:--|--:|--:|
| CYP3A4 | 0.2588 | 0.0469 |
| CYP2D6 | 0.1155 | 0.0633 |
| MA (macro-average) | 0.1872 | 0.0350 |

Compare the `MA (macro-average)` row against the interim leaderboard's macro MCC.
See `experiments/01_baseline/tdi_cv_summary.csv` for the full per-method comparison.

## Caveat

These are CV estimates, not leaderboard guarantees -- the real leaderboard bootstraps
the actual 750-compound test set, while this resamples training-set CV folds (and,
for the macro-average, matches fold *index* across endpoints rather than the same
resampled compounds -- see `evaluation.macro_averaged_fold_metrics`). Expect the real
number to differ somewhat; a large gap is worth investigating (train/test distribution
shift, a scaffold split that was optimistic or pessimistic relative to the real split),
not just shrugged off.
