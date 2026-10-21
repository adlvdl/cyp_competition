# Pool B regression ensemble + 14's TDI ensemble

- Generated: 2026-10-20
- Script: `notebooks/build_pool_b_submission.py`
- Evidence: notebooks `16` (encoders), `17` (ensemble pools), `18` (shift check)
- Challenge data snapshot: `20261020` (byte-identical to `20260908` for every data file;
  only the dataset README changed, to add a separate structure track)
- CV: nested scaffold 5x5, shared folds, `experiments/16_encoders/oof.parquet` and
  `experiments/05_auxiliary_data/oof.parquet`
- **Not yet scored on the board.**

## Regression model

Per-endpoint Caruana blend of four models, each fitted on 100% of the labelled rows:

- `pubchem`: `05`'s PubChem-pretrained Chemprop (its shipped, unplaced blind predictions)
- `tabicl_monroe`, `tabicl_atom_jepa`, `tabicl_chemeleon`: TabICL (PCA to 128, 2 estimators,
  seed 42, the `03` readout) on frozen embeddings of Monroe, Atom-JEPA (last block) and CheMeleon

Weights are refit on all 25 folds of OOF:

| endpoint | pubchem | tabicl_monroe | tabicl_atom_jepa | tabicl_chemeleon |
|---|---|---|---|---|
| CYP1A2 | 0.25 | 0.5 | 0.25 | 0.0 |
| CYP2C9 | 0.5 | 0.5 | 0.0 | 0.0 |
| CYP2D6 | 0.4 | 0.2 | 0.2 | 0.2 |
| CYP3A4 | 0.5 | 0.5 | 0.0 | 0.0 |

Pool B was chosen over pool A (2 models) and pool C (20 models) in `17` and `18`: C beat B
in plain CV (-0.009 macro) but not on the lowest-similarity slice of the OOF or after
reweighting toward the blind set's novelty and predicted potency, where C-vs-B held in only
69-73% of resamples. Pools A and B were picked after seeing `16`, so their absolute gain over
`pubchem` is mildly flattered.

## Placement

Each endpoint's blend is affine-mapped onto the mean and sd of the vector `12` shipped (and
the board scored) for that endpoint. This moves no compound's rank.

| endpoint | raw_mean | raw_sd | placed_mean | placed_sd | floor_ties | spearman_vs_12 |
|---|---|---|---|---|---|---|
| CYP1A2_pIC50_direct_inhibition | 4.909 | 0.728 | 4.485 | 0.648 | 0 | 0.917 |
| CYP2C9_pIC50_direct_inhibition | 4.82 | 0.752 | 4.816 | 0.782 | 0 | 0.967 |
| CYP2D6_pIC50_direct_inhibition | 4.582 | 0.428 | 3.114 | 0.538 | 0 | 0.903 |
| CYP3A4_pIC50_direct_inhibition | 4.657 | 0.935 | 4.684 | 0.941 | 0 | 0.975 |

CYP1A2 and CYP2D6 carry `12`'s placement, CYP2C9 and CYP3A4 `05`'s unplaced moments (gated at
recoverable R2 < 0.05). The assumption: the blend's best spread is close to `pubchem`'s.
The blend correlates slightly better, so its optimal spread is slightly wider.

## Expected performance (CV, placement-blind)

| endpoint | pool B (held-out weights) | pubchem (05, shipped model) |
|---|---|---|
| CYP1A2_pIC50_direct_inhibition | 0.7514 | 0.8216 |
| CYP3A4_pIC50_direct_inhibition | 0.4406 | 0.4864 |
| CYP2C9_pIC50_direct_inhibition | 0.568 | 0.631 |
| CYP2D6_pIC50_direct_inhibition | 0.8815 | 0.9251 |
| macro | 0.6604 | 0.716 |

CV cannot see placement, and the board-to-CV ratio ran 0.8-1.4 per endpoint on earlier
submissions (`07`). Do not quote a board score from these. Pool B's column uses 17's
held-out weights; the shipped weights use every fold.

## TDI

`14`'s shipped ensemble (`submissions/14_tdi_ensemble/20260920/tdi.csv`), copied unchanged.
