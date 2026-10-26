# 22's blend with interval-loss chemprop members + 24's TDI ensemble

- Generated: 2026-10-26
- Script: `notebooks/build_interval_loss_submission.py`
- Evidence: notebooks `28` (loss), `30` (on `pubchem` and pooled), `31_interval_final_fits`
  (refits), `31b` (blend)
- Idea source: mokhalaf22 and doctawho42 reports, see README "Ideas from other entrants".
  No code used.
- Challenge data snapshot: `20261020`
- **Not yet scored on the board.**

## Regression model

`22`'s five-member per-endpoint blend with the same weights (equal splits, `weights.csv`) and the
same placement. Two members are retrained with `--loss-function interval-mae`, which is zero
inside a compound's credible interval and linear outside it (ST-RAE's own shape):

- `pubchem`: 5 seeds, same pretrained checkpoint and recipe as `19`
- `pooled_workbench`: 3 seeds, CYP1A2 / CYP2C9 / CYP3A4. **CYP2D6 keeps the MAE seeds from `20`**,
  because the interval model was worse there in CV (+0.026 on 5 folds, untested for significance).
- TabICL members: `22`'s seed means, unchanged.

The loss rebuilds each cell's band from its own pIC50 (isotonic lookup, R2 0.84-0.98), because the
training file carries no bounds. Weights were not refit: they were fitted on the MAE members' OOF.

## Placement

Identical to `22`/`17`. Affine onto the moments of `12`'s vector, no rank changes.

| endpoint | raw_mean | raw_sd | placed_mean | placed_sd | floor_ties | spearman_vs_22 |
|---|---|---|---|---|---|---|
| CYP1A2_pIC50_direct_inhibition | 4.909 | 0.702 | 4.485 | 0.648 | 0 | 0.997 |
| CYP2C9_pIC50_direct_inhibition | 4.938 | 0.69 | 4.816 | 0.782 | 0 | 0.996 |
| CYP2D6_pIC50_direct_inhibition | 4.64 | 0.389 | 3.114 | 0.538 | 0 | 0.992 |
| CYP3A4_pIC50_direct_inhibition | 4.772 | 0.845 | 4.684 | 0.941 | 0 | 0.997 |

## Expected performance (CV, macro ST-RAE, 22's weights, members swapped)

| arm | n_folds | baseline | candidate | diff | ci_low | ci_high |
|---|---|---|---|---|---|---|
| A pubchem->interval | 25 | 0.6391 | 0.6231 | -0.0159 | -0.0183 | -0.0135 |
|   A CYP1A2 | 25 | 0.7379 | 0.7287 | -0.0092 | -0.0139 | -0.0047 |
|   A CYP2C9 | 25 | 0.5354 | 0.513 | -0.0224 | -0.029 | -0.0159 |
|   A CYP2D6 | 25 | 0.8645 | 0.8534 | -0.0111 | -0.015 | -0.0071 |
|   A CYP3A4 | 25 | 0.4184 | 0.3974 | -0.021 | -0.0246 | -0.0179 |
| B A + pooled->interval | 5 | 0.6345 | 0.6125 | -0.0221 | -0.0313 | -0.0109 |
|   B CYP1A2 | 5 | 0.7366 | 0.7244 | -0.0121 | -0.0269 | 0.0058 |
|   B CYP2C9 | 5 | 0.5248 | 0.4912 | -0.0336 | -0.0571 | -0.0129 |
|   B CYP2D6 | 5 | 0.8584 | 0.8507 | -0.0076 | -0.0183 | 0.003 |
|   B CYP3A4 | 5 | 0.4183 | 0.3835 | -0.0348 | -0.0587 | -0.0164 |

Rows A cover all 25 folds with only `pubchem` swapped. Rows B cover folds 0-4, the only folds where
a pooled interval OOF exists. Both are against the blend as shipped in `22`. The weights were fitted
on the baseline members, which favours the baseline. CV cannot see placement, and these submissions
already apply an affine placement, so part of the CV gain may not appear on the board. Spearman did
not move in `28`/`30`: the gain is position relative to the bands, not ordering.

## TDI

`24`'s six-arm encoder blend (`submissions/24_tdi_encoders/20261023/tdi.csv`), copied unchanged. It
has not been scored; `14`'s and `23`'s CV gains both shrank on the board.
