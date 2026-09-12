# Plan to the September 24 interim deadline

Written 2026-09-08. **16 days.** Deadline is 23:59 UTC on Sept 24; the interim leaderboard on Sept 25 is a one-time full-test-set reveal, which makes it the single most valuable feedback signal in the whole challenge.

## Where things stand

A valid submission already exists (`submissions/20260908_baseline/`), so the deadline is no longer a risk — everything from here is improvement. But the baseline says something uncomfortable:

| Endpoint | LightGBM ST-RAE | +linear calibration | Verdict |
|:--|--:|--:|:--|
| CYP3A4 | 0.758 | **0.742** | learns |
| CYP2C9 | 0.994 | 0.908 | barely |
| CYP1A2 | 1.007 | 0.927 | no better than the mean |
| CYP2D6 | 1.082 | 0.955 | worse than the mean |

ST-RAE of 1.0 means "no better than predicting a constant". **Fingerprint models fail on three of four endpoints.** Since the headline metric is the macro-average across all four, the three weak endpoints dominate the score — and that is where nearly all the available gain sits.

This is not a tuning problem. It is a representation and data problem.

## Why the weak endpoints are weak

Partly answered already:

| Endpoint | n | median CI width | spread (σ) | below pIC50 4 |
|:--|--:|--:|--:|--:|
| CYP3A4 | 2,335 | 0.38 | 1.09 | **40.4%** |
| CYP1A2 | 1,412 | 0.33 | 1.03 | 16.4% |
| CYP2C9 | 1,285 | 0.53 | 0.78 | 20.2% |
| CYP2D6 | 1,493 | **0.27** | 0.92 | 8.6% |

Two things fall out, and both cut against the obvious reading:

- **CYP2D6 is not noisier — it is the most precisely measured endpoint** (median CI 0.27). Mass spec beats fluorescence here. Tighter intervals make ST-RAE *stricter*: less of each error is forgiven, so the same model quality scores worse. The hardest endpoint is hard partly because it is measured best.
- **CYP3A4's lead is partly structural.** 40% of its compounds sit below pIC50 4, where intervals are wide and the metric explicitly downweights — a large pool of cheap, near-free predictions. CYP2D6 has 8.6%, so almost every compound must be predicted properly.

The remaining hypothesis is that CYP2D6 SAR is genuinely harder for ECFP: binding is dominated by a basic-nitrogen/aromatic pharmacophore that circular fingerprints represent poorly. That is a representation problem, which is exactly what item 2 below tests — so it moves to the front.

## Priorities, in order of expected value

**1. Finish the diagnosis (2 hours).** Remaining: per-endpoint error decomposition (where does ST-RAE actually accumulate — potent tail, mid-range, or the sub-4 region?) and whether the weak endpoints' errors concentrate on tight-interval compounds. Small, and it sharpens everything after it.

**2. Graph models — Chemprop and CheMeleon (2–3 days).** The single highest-expected-value item. CheMeleon won every PXR CV comparison outright, and learned representations are the standard answer when fingerprints plateau. The dependency install and MPS issues cost real time in PXR ("Stalling the M4 engine"), so start early rather than late.

**3. Public data augmentation (1–2 days).** PXR's own retrospective names this as one of the two biggest missed opportunities. CYP inhibition is one of the best-covered endpoints in public cheminformatics — ChEMBL and PubChem BioAssay have tens of thousands of measurements across all four isoforms. With three endpoints stuck at the mean-predictor line, more data is the most direct lever available. Disclose it on submission, as the rules require.

**4. The single-concentration screen (1 day).** 17,504 rows covering far more compounds than any dose-response table. PXR found auxiliary-assay multitask training *hurt*, so treat this as a genuine experiment with a control, not an assumed win.

**5. TDI track (1 day).** Currently unaddressed and it is half the competition. Labels are imbalanced (~21%), MCC punishes majority-class predictors, and the classifier needs a tuned decision threshold rather than the default 0.5. Cheap to get something reasonable.

**6. Ensemble and calibrate (1 day).** Only worth doing once there are genuinely different models to blend. In PXR ensembling suppressed catastrophic errors, which is exactly what a ratio-of-summed-errors metric like ST-RAE rewards.

## Suggested schedule

| Days | Work |
|:--|:--|
| Sep 8–9 | Diagnose weak endpoints; TDI baseline so both tracks have a submission |
| Sep 10–13 | Chemprop + CheMeleon, all four endpoints |
| Sep 14–16 | Public data (ChEMBL/PubChem) augmentation |
| Sep 17–18 | Single-concentration auxiliary experiment |
| Sep 19–21 | Ensemble, calibrate, run the paired-bootstrap comparison |
| Sep 22 | **Freeze.** Pick the submission, validate, upload |
| Sep 23–24 | Buffer |

Freezing on the 22nd is deliberate. PXR's most expensive mistake was picking between three finalists separated by 0.0039 MAE — pure noise — and getting it backwards. Leave time to choose deliberately rather than at the deadline.

## How to choose the final submission

The hardest-won PXR lesson, and the easiest to ignore under time pressure.

- Run `evaluation.paired_bootstrap` between the finalists. If p > 0.05, the data cannot tell them apart — **do not** pick by sorting on the mean.
- When variants are statistically tied, prefer on other grounds: calibrated over uncalibrated (it won PXR Phase 2 from behind), ensemble over single model (lower catastrophic-error rate), simpler over more tuned (less overfit to CV).
- Check `bias_by_potency_bin` on the finalist. Shrinkage in the top potency bin is the error ST-RAE actually punishes, since the low bins are forgiven by wide intervals and explicit downweighting.

## Deliberately not doing

- **3D docking / cofolding.** Most PXR top-ten finishers used it, and CYP structures are well resolved, so this is genuinely promising — but it is a multi-week project and there are 16 days. Revisit for the November 3 final deadline.
- **Heavy hyperparameter optimization.** PXR showed HPO pays for tree models and barely moves neural nets. Since the plan bets on graph models, tuning is a poor use of the remaining time.
- **A second submission per track.** One per team, so there is no hedging.

## Open questions

- Is Chemprop/CheMeleon setup on this machine going to repeat the PXR MPS problems? Worth a timeboxed spike on day one.
- Does the Innovation award make an uncertainty-quantification angle worth pursuing? ST-RAE rewards it structurally — predictions inside the credible interval score zero — and the award is judged separately from rank.
