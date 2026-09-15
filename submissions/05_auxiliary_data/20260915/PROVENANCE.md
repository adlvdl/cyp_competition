# Auxiliary and public data submission — activity track

- Generated: 2026-09-15
- Notebook: `notebooks/05_auxiliary_data.py`
- Challenge data snapshot: `20260908`
- External data snapshot: `20260915` (PubChem qHTS panel, AIDs 410/883/884/891)
- CV: nested scaffold, 5x5 folds, shared folds across endpoints and arms
  (`cv.shared_scaffold_folds` over the augmented-frame union)

## Direct inhibition (regression)

- Model: **pubchem** — one Chemprop D-MPNN with four output heads, encoder
  pretrained on the PubChem qHTS panel (public CYP inhibition data, challenge
  structures excluded by standardised-SMILES match), then fine-tuned on all four
  challenge endpoints with missing targets masked in the loss
- Representation: learned from the molecular graph, warm-started from a public-data
  checkpoint rather than trained from scratch (03's `chemprop_multitask`) or a
  frozen fingerprint/embedding
- Calibration: **none** — checked and found to make no difference (see below)
- Validation: PASSED

**Expected performance (5x5 CV, macro ST-RAE, lower is better;
1.0 = no better than predicting the mean):**

| Method | Macro ST-RAE | std | n folds |
|:--|--:|--:|--:|
| pubchem | 0.7160 | 0.0272 | 25 |
| aux_screen | 0.7214 | 0.0294 | 25 |
| augmented | 0.7236 | 0.0267 | 25 |
| chemprop_multitask | 0.7304 | 0.0274 | 25 |
| selection | 0.7334 | 0.0291 | 25 |
| pubchem_frozen | 0.7523 | 0.0327 | 25 |

**pubchem vs the 03 winner (chemprop_multitask), paired bootstrap on 25 shared
folds:** diff -0.0129, 95% CI [-0.0178,
-0.0079], p=0.0000. This is the
margin the CV can resolve, not just the point estimate.

## Why this model

Six arms were compared against `chemprop_multitask` on identical folds, all one
Chemprop D-MPNN with four masked heads so a difference is attributable to the
training data rather than the architecture. `pubchem` was the only arm that was
both the best point estimate and Holm-significant across the family of five
comparisons, and it won broadly rather than on one endpoint:

| Method | CYP3A4 | CYP1A2 | CYP2D6 | CYP2C9 |
|:--|--:|--:|--:|--:|
| augmented | 0.4906 | 0.8626 | 0.9171 | 0.6244 |
| aux_screen | 0.4845 | 0.8407 | 0.9256 | 0.6350 |
| chemprop_multitask | 0.5012 | 0.8377 | 0.9315 | 0.6510 |
| pubchem | 0.4864 | 0.8216 | 0.9251 | 0.6310 |
| pubchem_frozen | 0.5537 | 0.8519 | 0.9349 | 0.6687 |
| selection | 0.5020 | 0.8405 | 0.9373 | 0.6539 |

Two findings behind it, both new in this notebook:

1. **Public data helps here**, reversing PXR's finding that auxiliary-assay data
   hurt there. The mechanism is real transfer learning, not feature reuse --
   freezing the pretrained encoder during fine-tuning (`pubchem_frozen`) erased the
   gain and made the result significantly *worse* than no pretraining at all
   (+0.025, p<0.0001), so the model needs to adapt to the challenge's assay
   protocol rather than just inherit public-data features.
2. **The single-concentration screen also helps** (`aux_screen`, −0.009,
   Holm-significant) as an auxiliary Chemprop target, with no external dependency.
   It was not shipped here only because `pubchem`'s margin is larger, not because it
   failed -- worth a second look if the public-data dependency is undesirable for a
   future submission.

## What did not survive Holm correction

`augmented` (screen negatives as censored weak labels) showed a consistent but
unresolved effect (p=0.033 against a 0.025 threshold at its rank) -- plausible with
more weak-label yield, not concluded to be nothing. `selection` (inverse-propensity
correction for the screen-to-curve triage) showed no effect anywhere, including on
CYP2D6, the endpoint its hypothesis specifically targeted -- it is in fact the worst
arm on CYP2D6 of the six tested. The triage truncation described in this notebook's
early sections is real; this particular correction for it did not work.

## TDI track: not included

This notebook only ran the regression track. Notebook 04 established that
multitask does not transfer to TDI (5.4% co-measurement against 26.7% for
regression), and none of this notebook's four data sources naturally extend to a
classification target without their own dedicated measurement. The
`tdi_shift_summary` side section above found the TDI/direct pIC50 shift reaching
AUC 0.96 unaided on CYP2D6 -- a strong lead for a future notebook, not something
this run measured a model against.

## Caveats

**CV estimates, not leaderboard guarantees.** The real leaderboard bootstraps the
actual 750-compound test set; this resamples training-set CV folds, and the
macro-average matches fold index across endpoints rather than resampling the same
compounds for every endpoint. A large gap between this table and the interim reveal
is worth investigating, not shrugging off.

**The pretraining checkpoint is fit once, not per fold, and reused for the final
model.** This is legitimate because the public corpus contains no challenge label
(structural overlap removed), so no compound's test label ever reached the encoder
through it -- but it does mean the checkpoint was not re-validated against the
final all-data fit's exact training set. A future run wanting to be maximally
careful could re-pretrain once more against the exact final training compounds;
the cost is one more pretraining run, cheap next to the CV that already ran.

**Public data quality.** The PubChem qHTS panel is a single uniform protocol,
deliberately chosen over ChEMBL's much larger but heterogeneous pool -- but it is
still a different lab, different conditions, and a different concentration-response
range than the challenge's own assay. The gain measured here is real on this CV;
whether it holds at the same magnitude on the blind set is exactly what the interim
reveal will show.
