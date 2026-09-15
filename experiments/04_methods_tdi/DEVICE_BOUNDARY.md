# Chemprop accelerator boundary in this run

The full 5x5 run was interrupted and resumed with a different accelerator, so the
Chemprop OOF predictions are **not** device-homogeneous. Recorded here because the
cache gives no hint of it and a later reader would have no way to know.

| folds | accelerator | why |
|:--|:--|:--|
| 0-13 | MPS | the original run's auto-detected device |
| 14-24 | CPU | forced via `CYP_CHEMPROP_DEVICE=cpu` after MPS degraded |

Every other method is unaffected: trees and TFMs never touch the accelerator, and
the four fold-14 units already cached when the run was stopped
(`lgbm`/`xgb`, both arms) are device-irrelevant.

## Why the switch

`chemprop_singletask` per-fold wall time on MPS, in execution order:

```
fold   0    1    2    3     4    5    6     7     8     9    10    11    12
sec  100   91   97 1109  1791   98 1039  1766  1978  1996  2008  1987  2042
```

Three clean folds, then a rise to ~2000s that **plateaued and never recovered** --
a ~20x penalty. `chemprop_multitask` showed the same shape (peak 2057s at fold 9).
Every other method in the same interleaved run stayed flat throughout, so this is
specific to the sustained MPS workload, not machine-wide contention. Interleaving
methods within a fold did not prevent it.

## What to check before trusting the combined result

Whether the device changed the *predictions*, not just the speed. Compare
Chemprop's per-fold MCC on folds 0-13 against folds 14-24; a real difference means
the two blocks cannot be pooled and the MPS-fitted folds should be refit on CPU.
Note that fold index is confounded with device here, so a difference could also be
fold-to-fold variation -- compare against another method's fold-13/14 split as a
control before concluding anything.

Both arms are affected identically, so the *paired* multitask-vs-single-task
contrast within a fold stays internally valid either way.

## Verdict (checked 2026-09-15, after the run completed)

**The two blocks can be pooled. No refit needed.**

Macro MCC either side of the boundary, with every non-chemprop method as a control
for fold-to-fold drift (they ran the same folds and never touched the accelerator):

| method | MPS-era (0-13) | CPU-era (14-24) | delta |
|:--|--:|--:|--:|
| chemprop_singletask | 0.0677 | 0.0815 | +0.0138 |
| chemprop_multitask | 0.0759 | 0.0533 | −0.0226 |
| *controls (9 methods)* | — | — | mean −0.0008, sd 0.0103 |

`chemprop_multitask`'s −0.0226 sits 2.1 sd outside the *control deltas*, which looks
alarming until you notice that the control deltas' sd (0.0103) is far tighter than
chemprop's own fold-level noise (sd ≈ 0.05 in both eras, against ≈ 0.033 for lgbm).
The right test compares the gap to chemprop's own variability, not to the controls'.

A permutation test doing exactly that -- shuffling fold labels within each method,
20,000 draws, asking how often a 14-vs-11 split produces a gap this large by chance:

| arm | observed gap | p |
|:--|--:|--:|
| chemprop_multitask | −0.0226 | **0.254** |
| chemprop_singletask | +0.0138 | **0.516** |

Neither is distinguishable from chance.

**Why chemprop is so noisy here, and it is not the device.** Both arms return exactly
0.000 MCC on several folds *in both eras* (multitask: 4 of 14 MPS folds, 3 of 11 CPU
folds). A 0.000 means that fold predicted a single class, collapsing to the majority
baseline at the default 0.5 cutoff -- the calibration problem documented in CLAUDE.md,
not an accelerator artefact. It happens at about the same rate on CPU as on MPS.

Timing, by contrast, *was* strongly device-dependent -- see the table above and the
CPU figures: folds 14-20 ran 80.7, 75.7, 76.4, 75.9, 79.6, 75.8, 76.1s (single-task)
with no drift at all, against ~2000s on degraded MPS. CPU is both faster and stable.
