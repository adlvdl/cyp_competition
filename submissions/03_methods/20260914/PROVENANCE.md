# Method-comparison submission — activity track

- Generated: 2026-09-14
- Notebook: `notebooks/03_methods.py`
- Data snapshot: `20260908`
- CV: nested scaffold, 5x5 folds, shared folds across endpoints
  (`cv.shared_scaffold_folds`)

## Direct inhibition (regression)

- Model: **chemprop_multitask** — one Chemprop D-MPNN with four output heads, trained
  on all four endpoints at once with missing targets masked in the loss
- Representation: learned from the molecular graph (no fingerprint, no frozen
  embedding)
- Calibration: **none** — see the caveat below
- Validation: PASSED

**Expected performance (5x5 CV, uncalibrated, lower ST-RAE is better;
1.0 = no better than predicting the mean):**

| Endpoint | Expected ST-RAE (mean) | std across folds |
|:--|--:|--:|
| CYP1A2_pIC50_direct_inhibition | 0.8407 | 0.0712 |
| CYP2C9_pIC50_direct_inhibition | 0.6439 | 0.0646 |
| CYP2D6_pIC50_direct_inhibition | 0.9283 | 0.0490 |
| CYP3A4_pIC50_direct_inhibition | 0.4991 | 0.0324 |
| MA (macro-average) | 0.7280 | 0.0316 |

Compare the `MA (macro-average)` row against the interim leaderboard's macro ST-RAE
— that row, not any single endpoint, determines rank.

## Why this model

It won the comparison in `experiments/03_methods/` outright and, unlike PXR's
finalists, by a margin the CV can resolve: paired bootstrap against
`tabicl_singletask`, `chemprop_singletask` and `macau_singletask` all give p=0.0000
with the 95% CI entirely below zero, on 32,625 paired measurements each.

Two findings behind it, both in CLAUDE.md:

1. Representation dominated model choice up to a point — ECFP4 to CheMeleon bought
   ~0.17 on the macro, while swapping models *on* frozen CheMeleon bought ~0.012.
2. Multitask helps only when it is real multi-task learning. Row-stacking frozen
   features gained nothing on strong models and actively hurt TabICL; a shared
   *learned* encoder with per-endpoint heads gained the most of any model.

## TDI track: not included

This notebook never ran TDI. The multitask mechanism that won here depends on
compounds carrying several endpoints — 26.7% do for regression, but only **5.4%**
of TDI compounds appear in both isoforms, so the gain should not be assumed to
transfer. Notebook 04 runs the equivalent comparison for TDI and ships that track
once there is a measured number behind it. `submissions/01_baseline/` holds the last
validated TDI submission (LightGBM, CYP3A4 MCC 0.259 / CYP2D6 0.116) if one is
needed before then.

## Caveats

**Uncalibrated.** `01_baseline` found linear calibration improved every endpoint
(e.g. CYP2D6 1.063 to 0.947), so there is likely headroom left. It is deferred
rather than rushed: calibration must be cross-fit per endpoint and checked against
the multitask gain, which is its own piece of work.

**CV estimates, not leaderboard guarantees.** The real leaderboard bootstraps the
actual 750-compound test set; this resamples training-set CV folds, and the
macro-average matches fold *index* across endpoints rather than resampling the same
compounds for every endpoint (see `evaluation.macro_averaged_fold_metrics`). Expect
some difference; a large gap is worth investigating, not shrugging off.

**Cost.** Chemprop is by far the most expensive method here — 2h19m for the
single-task CV arm — and it degrades across long blocks of consecutive fits on MPS
(~30s/fold early, ~290s/fold late). See the MPS notes in CLAUDE.md before re-running.
