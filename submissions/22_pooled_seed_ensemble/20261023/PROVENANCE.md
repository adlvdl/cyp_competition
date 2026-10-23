# Seed-averaged pool B + pooled workbench regression ensemble + 14's TDI ensemble

- Generated: 2026-10-23
- Script: `notebooks/build_pooled_seed_submission.py`
- Evidence: notebooks `19` / `19b` (seed spread), `20` (pooled workbench-recipe model), `21` (pool B + pooled)
- Challenge data snapshot: `20261020`
- CV: nested scaffold 5x5, shared folds. Weights fitted on all 25 folds of single-seed OOF.
- **Not yet scored on the board.**

## Regression model

Per-endpoint Caruana blend of five models, each fitted on 100% of the labelled rows. Every
blind-set member is a seed mean:

- `pubchem`: 5 seeds of `05`'s recipe on the shared PubChem-pretrained checkpoint (pretraining seed fixed)
- `tabicl_monroe`, `tabicl_atom_jepa`: 5 TabICL seeds (42, 0-3), PCA to 128, 2 estimators
- `tabicl_chemeleon`: weight 0 on every endpoint, not fitted
- `pooled_workbench`: 3 seeds of a from-scratch multitask chemprop on challenge + public rows
  (17 heads, workbench `union-p30` task weights), public rows matching the blind set removed

Weights:

| endpoint | pubchem | tabicl_monroe | tabicl_atom_jepa | tabicl_chemeleon | pooled_workbench |
|---|---|---|---|---|---|
| CYP1A2 | 0.2 | 0.4 | 0.2 | 0.0 | 0.2 |
| CYP2C9 | 0.333 | 0.333 | 0.0 | 0.0 | 0.333 |
| CYP2D6 | 0.25 | 0.25 | 0.25 | 0.0 | 0.25 |
| CYP3A4 | 0.333 | 0.333 | 0.0 | 0.0 | 0.333 |

## Placement

Same as `17`: each endpoint's blend is affine-mapped onto the mean and sd of the vector `12`
shipped. This moves no compound's rank. `spearman_vs_17` is the rank agreement with 17's file.

| endpoint | raw_mean | raw_sd | placed_mean | placed_sd | floor_ties | spearman_vs_17 |
|---|---|---|---|---|---|---|
| CYP1A2_pIC50_direct_inhibition | 4.895 | 0.731 | 4.485 | 0.648 | 0 | 0.988 |
| CYP2C9_pIC50_direct_inhibition | 4.877 | 0.734 | 4.816 | 0.782 | 0 | 0.982 |
| CYP2D6_pIC50_direct_inhibition | 4.613 | 0.423 | 3.114 | 0.538 | 0 | 0.944 |
| CYP3A4_pIC50_direct_inhibition | 4.675 | 0.921 | 4.684 | 0.941 | 0 | 0.983 |

The live board scores half the test set. Placement moments were solved on that half and may not
transfer exactly to the other half (see CLAUDE.md, Placement).

## Expected performance (CV, placement-blind, single-seed members)

| endpoint | pool B + pooled (held-out weights) | pool B (17, held-out weights) | pubchem (05) |
|---|---|---|---|
| CYP2C9_pIC50_direct_inhibition | 0.5351 | 0.568 | 0.631 |
| CYP1A2_pIC50_direct_inhibition | 0.7379 | 0.7514 | 0.8216 |
| CYP2D6_pIC50_direct_inhibition | 0.8645 | 0.8815 | 0.9251 |
| CYP3A4_pIC50_direct_inhibition | 0.4183 | 0.4406 | 0.4864 |
| macro | 0.639 | 0.6604 | 0.716 |

CV has no seed-averaged OOF, so seed-averaging's gain (estimated 0.003-0.005 macro) is not in
this table, and the weights were fitted on single-seed members. CV cannot see placement. Do not
quote a board score from these.

## TDI

`14`'s shipped ensemble (`submissions/14_tdi_ensemble/20260920/tdi.csv`), copied unchanged.
