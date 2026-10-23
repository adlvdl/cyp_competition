# CLAUDE.md

Modelling code for the OpenADMET CYP Inhibition Blind Challenge. See [README.md](README.md) for challenge background, data dictionary, and dates.

## Environment

`uv` only — no conda/mamba. Run everything through `uv run ...` or the Makefile; never `pip install` into system Python.

```bash
make setup     # uv sync --all-extras
make test      # pytest
make lint      # ruff check + format
make baseline  # notebooks/01_baseline.py, run as a script
```

## Code conventions

- **Polars, not Pandas.** Pandas only at the boundary (the vendored validators call `pd.read_csv`; sklearn takes numpy). Never in `src/cyp`.
- Type hints on every parameter and return value.
- Double quotes; f-strings with single-quoted keys are the exception.
- Comment generously and explain *why*, not what.

## Non-negotiables

1. **Never reimplement the scoring metrics.** `src/cyp/metrics.py` imports ST-RAE from `vendor/cyp_challenge_tutorial/` (upstream `858ae63`, 2026-09-08). Treat `vendor/` as read-only; re-vendor rather than edit.
2. **Never rename endpoint columns.** `CYP3A4_pIC50_direct_inhibition`, `CYP2D6_is_TDI` and friends are matched by exact string. They live in `src/cyp/constants.py`.
3. **Always validate before a submission leaves the repo.** `submission.check(path, track=...)`. Teams get one submission.
4. **No random split as headline validation.** Test set is 75 hits × ~10 analogs; K-fold leaks near-duplicates. Default `cv.scaffold_splits`; label any `cv.random_splits` number optimistic.
5. **Rank is the macro-average, not any endpoint.** `compute_macro_bootstrap_results` takes a plain arithmetic mean across the four regression endpoints (or both TDI isoforms). Use `evaluation.macro_averaged_fold_metrics` before comparing models or picking a submission.
6. **Ask before**: adding a heavyweight dependency, changing `vendor/`, or overwriting a file in `submissions/`.

## Data facts that bite

- **Targets are sparse, not missing-at-random.** A pIC50 exists only where the primary screen flagged it, so `dropna()` on the full frame leaves almost nothing. Filter per endpoint via `data.labelled_subset(df, endpoint)`. Counts: 1,285 (CYP2C9) to 2,335 (CYP3A4) of 4,905 rows. The four regression endpoints do not share rows.
- **TDI labels are imbalanced** (~21% CYP3A4, ~22% CYP2D6) and exist only for CYP3A4/CYP2D6. MCC punishes majority predictors; accuracy is useless.
- **pIC50 < 4 is below assay resolution.** The metric downweights it. Do not chase precision there.
- **Credible intervals are the point of ST-RAE.** A prediction inside `_conf_low`/`_conf_high` scores zero error — wide-interval (low-activity) compounds are forgiving, tight-interval ones are not.
- **Dataset has been amended mid-challenge.** Downloads go to dated snapshots `data/raw/YYYYMMDD/`, never overwriting. `make data` / `make data-diff`. Loaders take `snapshot="YYYYMMDD"`; record the date next to any result worth reproducing. Never flatten or delete a snapshot.

## Results by notebook

### 01_baseline (2026-09-09) — LightGBM/ECFP4, 5x5 nested scaffold CV

Mean ST-RAE (1.0 = no better than a constant):

| Endpoint | raw | +linear calibration |
|:--|--:|--:|
| CYP3A4 | 0.720 | 0.707 |
| CYP2C9 | 0.961 | 0.883 |
| CYP1A2 | 0.986 | 0.903 |
| CYP2D6 | 1.063 | 0.947 |

**Only CYP3A4 learns from fingerprints.** Linear calibration helps every endpoint (unlike PXR, where it looked like noise) — never ship uncalibrated without checking.

TDI: LightGBM with class-balanced weighting clears the MCC=0.0 baseline — CYP3A4 **0.259**, CYP2D6 0.116.

**Macro-averaged MCS heatmaps change the conclusion.** On macro ST-RAE, `lgbm` and `xgb` are **not** significantly different (p=0.239) though both beat `mean`/`ridge`/`knn` — a different answer than any per-endpoint panel gives. On macro TDI MCC, `lgbm` (0.187) beats every method including `xgb` (0.142).

### 02_chemical_space — `endpoint_scorecard.csv`

Read this before theorising about why an endpoint is hard.

| endpoint | n | median CI | below floor | test NN sim | % extrapolation | ECFP4 cliff % |
|:--|--:|--:|--:|--:|--:|--:|
| CYP1A2 | 1,412 | 0.33 | 16.4% | 0.525 | 7.2% | 16.53 |
| CYP2C9 | 1,285 | 0.53 | 20.2% | 0.519 | 9.5% | 6.00 |
| CYP2D6 | 1,493 | **0.27** | 8.6% | **0.469** | **10.8%** | 6.84 |
| CYP3A4 | 2,335 | 0.38 | **40.4%** | 0.537 | **1.9%** | 14.23 |

**CYP2D6 is hard on three independent axes** (tightest intervals, least training support, most extrapolation). **CYP3A4 is the mirror image** — 40.4% below the assay floor where the metric downweights, only 1.9% extrapolation. Much of CYP3A4's lead is structural, not earned. The two endpoints' scores are not directly comparable.

### 03_methods (2026-09-14) — regression architectures, 5x5, macro ST-RAE

| method | representation | macro ST-RAE | std |
|:--|:--|--:|--:|
| **chemprop_multitask** | learned graph, 4 heads | **0.7280** | 0.0316 |
| tabicl | frozen CheMeleon | 0.7599 | 0.0250 |
| chemprop | learned graph | 0.7628 | 0.0328 |
| macau | frozen CheMeleon | 0.7705 | 0.0238 |
| tabpfn | frozen CheMeleon | 0.7760 | 0.0264 |
| xgb_mordred | Mordred | 0.8027 | 0.0344 |
| lgbm_ecfp | ECFP4 | 0.9324 | 0.0298 |

Per endpoint, `chemprop_multitask` vs 01's lgbm/ECFP4: CYP1A2 0.841 (0.986), CYP2C9 0.644 (0.961), CYP2D6 **0.928** (1.063 — first time below a constant predictor), CYP3A4 0.499 (0.720). Winner separable from every rival at p=0.0000.

**Representation moved the numbers, model choice did not.** ECFP4 → Mordred → CheMeleon bought ~0.17; swapping models *on* frozen CheMeleon bought ~0.012. Four unrelated architectures converging that tightly says the frozen representation was the binding constraint.

**Multitask helps only when it is real multi-task learning** (shared folds, Holm-corrected):

| model | single | multi | diff | sig |
|:--|--:|--:|--:|:--|
| chemprop | 0.7635 | **0.7280** | −0.0355 | yes |
| xgb | 0.9516 | 0.9024 | −0.0348 | yes |
| lgbm | 0.9343 | 0.8977 | −0.0256 | yes |
| macau | 0.7705 | 0.7657 | −0.0045 | yes |
| tabpfn | 0.7760 | 0.7689 | −0.0019 | no (p=0.47) |
| tabicl | 0.7638 | 0.7686 | +0.0083 | yes — **hurts** |

The rule: **sharing a *learned* encoder helps; pooling rows of a *frozen* one does not.** Contradicts PXR; the structural difference is that 1,309 of 4,905 compounds carry more than one endpoint.

Chemprop cost 2h19m for its single-task arm (100 fits) vs seconds-to-minutes for trees.

### 04_methods_tdi (2026-09-15) — TDI track, 5x5, macro MCC at tuned threshold

| method | macro MCC | threshold |
|:--|--:|--:|
| **tabicl_singletask** | **0.2483** | 0.25 |
| tabicl_multitask | 0.2410 | 0.25 |
| tabpfn_singletask | 0.2199 | 0.20 |
| lgbm_singletask | 0.2092 | 0.10 |
| xgb_multitask | 0.1723 | 0.20 |
| chemprop_singletask | 0.1525 | 0.20 |
| majority | 0.0000 | — |

**Multitask does not transfer to TDI.** Every paired delta negative; Holm-corrected only `xgb` is significant (−0.0330, p<0.0001), and it shows multitask *hurting*. TabICL's −0.0124 has raw p=0.046 but fails its Holm threshold of 0.0125. Cause is the co-measurement gap: 26.7% of regression compounds carry several endpoints vs **5.4%** of TDI compounds, across half as many tasks. The two tracks ship different architectures for this reason.

**Chemprop is last among real methods here** having won regression outright. A learned shared encoder is not universally the answer.

**The threshold matters more than the model, and 0.5 is badly wrong.** Every optimum fell in 0.10–0.25. At 0.5 both chemprop arms score exactly 0.000 — easy to misread as "the graph model learned nothing", but its probabilities are well calibrated to the base rate (mean ≈ 0.205 vs 21.4% positive) and simply never cross 0.5; tree models clear it only because `class_weight="balanced"` inflates probabilities — so a comparison at 0.5 partly ranks by probability inflation. Always tune on OOF `y_prob` first.

**The test set predicts much more positive than training** (CYP3A4 48.8% vs 21.3% training rate). Verified not a bug: the final model predicts its own training compounds at mean 0.212 vs 21.3% true. Two real causes — the blind set is hit-enriched, and TabICL's in-context behaviour shifts between an 80% fold fit and a 100% final fit.

### 05_auxiliary_data (2026-09-15) — every arm paired against 03's winner

| method | shape | macro ST-RAE | std |
|:--|:--|--:|--:|
| **pubchem** | encoder pretrained on NCGC qHTS | **0.7160** | 0.0272 |
| aux_screen | 4 heads predicting screen log2fc | 0.7214 | 0.0294 |
| augmented | screen negatives as censored rows | 0.7236 | 0.0267 |
| chemprop_multitask (control) | — | 0.7304 | 0.0274 |
| selection | inverse-propensity reweighting | 0.7334 | 0.0291 |
| pubchem_frozen | pretrained, frozen | 0.7523 | 0.0327 |

Holm-corrected vs control: `pubchem`, `aux_screen`, `pubchem_frozen` significant (p=0.000); `augmented` (p=0.033) and `selection` (p=0.066) not. `pubchem_frozen` is significant in the **wrong direction**.

**Pretraining helps only with the encoder unfrozen** (0.7160 vs 0.7523 frozen). The gain is in the initialisation, not the features — same lesson as 03 from the other direction.

Shipped `pubchem` per endpoint: CYP3A4 0.4864, CYP2C9 0.6310, CYP1A2 0.8216, CYP2D6 0.9251. **No data shape moved CYP2D6 below 0.917** — the problem is representation, not data.

### 06_uncertainty (2026-09-16) — Chemprop's native UQ methods

All four are too weakly informative to build on. Spearman(abs error, claimed uncertainty): **MVE 0.172**, MC dropout 0.087, checkpoint ensemble 0.080, evidential regression **0.015** (essentially uninformative — worth knowing, since its theory most suggests it should work).

MVE promoted to full 5x5 (32,625 OOF rows): CYP1A2 0.253, CYP2D6 0.148, CYP3A4 0.143, CYP2C9 0.127. Real signal, far too weak for inverse-variance weighting or uncertainty-aware calibration, so the notebook ships nothing.

**Hard constraint:** the submission format has **no interval column** — `_conf_low`/`_conf_high` are the backend's ground-truth assay uncertainty, not something a submission sets. Uncertainty can only change how a prediction is *built*, never what is reported. The Innovation award is judged separately and rewards UQ, so this may be worth resuming for that reason — but not as a route to a better macro.

### 08_emax_two_stage — Emax cannot be a model feature, by either route

Do not re-attempt. 15 scaffold folds, LightGBM/ECFP4-2048, scored in Spearman:

| arm | CYP2D6 Spearman | vs baseline |
|:--|--:|--:|
| baseline (fingerprints) | 0.3274 | — |
| + **oracle** Emax | 0.8137 | +0.486 |
| + **predicted** Emax | 0.2296 | **−0.098** |

The blind set is `Molecule_Name,SMILES`, so Emax cannot be an input column; the two-stage repair is worse than doing nothing. Cause: Emax and pIC50 are **perfectly co-measured** on all four isoforms (both fall out of the same fitted curve), so stage 1 gains no coverage and predicts Emax at only Spearman ~0.207.

**General rule:** an auxiliary variable is usable as a two-stage feature only when it is **more predictable from structure than the target is**. Correlation with the target is necessary and nowhere near sufficient — read the coverage table before the correlation table. Does not affect Emax as an auxiliary **target** (an extra head, as 05 used it), which cannot inject noise into features.

### 09_cyp2d6_representation — the 3D pharmacophore hypothesis does not survive contact with data

`07`'s decomposition put CYP2D6's Spearman at 0.36 against 0.70–0.77 elsewhere, and `PLAN.md` carried an untested hypothesis since day one: CYP2D6 binding runs on a basic-nitrogen/aromatic pharmacophore that ECFP cannot express, so 3D conformer geometry should help *selectively* on CYP2D6. Measured on the challenge's own labels this looked promising going in — basic-N correlates +0.284 with CYP2D6 potency against −0.07 to −0.16 elsewhere, and logP carries essentially nothing on CYP2D6 (0.034) where it dominates the other three (0.23–0.60). It did not survive testing.

**The direct hypothesis check already weakens it.** Through-space N-to-aromatic distance (`pharmacophore.descriptors`, ETKDG conformers) beats topological (bond-count) distance on CYP2D6 (Spearman −0.165 vs −0.122), but it *also* beats it on CYP1A2 (−0.116 vs −0.063) — not the CYP2D6-selective signal the mechanism predicts.

**LightGBM sweep, 6 arms × 4 endpoints, 5x5 scaffold CV, scored in Spearman** (ECFP the incumbent; `pharmacophore_2d`/`_3d` are 5/7 named descriptors; `e3fp`/`usrcat` are hashed/shape 3D fingerprints):

| endpoint | ecfp | pharmacophore_3d | ecfp+pharmacophore_3d | lift (ecfp+3d vs ecfp) |
|:--|--:|--:|--:|--:|
| CYP1A2 | 0.4214 | 0.1742 | 0.4641 | +0.043 |
| CYP2C9 | 0.3713 | 0.4232 | 0.5598 | **+0.188** |
| CYP2D6 | 0.3365 | 0.2427 | 0.3766 | +0.040 |
| CYP3A4 | 0.6048 | 0.5541 | 0.7408 | +0.136 |

`ecfp+pharmacophore_3d` lifts every endpoint, but **smallest on CYP2D6**, largest on CYP2C9 and CYP3A4 — the reverse of the selectivity pattern that would support the mechanism. No 3D-only arm (`pharmacophore_3d`, `e3fp`, `usrcat`) beats plain ECFP on any endpoint. This is a better featurizer, not evidence for CYP2D6-specific geometry.

**A3 confirms it on the model that would actually ship.** Folding the 7 pharmacophore descriptors into chemprop as `--descriptors-columns` (warm-started from the same `pubchem`-pretrained encoder as `05`'s incumbent, shared 25-fold CV, paired bootstrap) made every endpoint *worse*, CYP2D6 included:

| endpoint | pubchem | pubchem_pharmacophore | diff | p |
|:--|--:|--:|--:|--:|
| CYP1A2 | 0.5368 | 0.5153 | −0.0215 | 0.0000 |
| CYP2C9 | 0.6697 | 0.6463 | −0.0234 | 0.0000 |
| CYP2D6 | 0.4482 | 0.4260 | −0.0222 | 0.0000 |
| CYP3A4 | 0.7960 | 0.7848 | −0.0112 | 0.0000 |

All four significant at 25 folds (an earlier 15-fold run left CYP2D6/CYP3A4 borderline, p=0.03–0.05 — more folds resolved it, did not reverse it).

**Conclusion: the hypothesis is wrong, or at least not expressible through this geometry.** Explicit 3D pharmacophore features hurt a learned encoder that already has ECFP-equivalent structural signal available; they do not help CYP2D6 selectively anywhere in this notebook. Do not spend the Uni-Mol/pretrained-3D dependency PLAN.md flagged as contingent on this result — the contingency did not fire. CYP2D6's gap remains a chemical-space problem (`02`'s scorecard: lowest test-NN similarity, highest extrapolation) rather than a representation-class problem.

**Infrastructure this notebook added, reusable elsewhere:** `pharmacophore.py` (named descriptors + ETKDG conformer generation, with a `signal.SIGALRM` `timeout_s` per molecule — a public-corpus conformer batch has no wall-clock bound otherwise and one slow molecule can stall a worker indefinitely); `graph_models.ChempropMultitargetModel`'s `descriptor_columns`/`--descriptors-columns` support, and `aux_training.run_cv_pretrained`'s matching `finetune_descriptors`/`descriptor_timeout_s`. One real constraint surfaced building this: chemprop's `--checkpoint` restores the FFN's input width whole, so a pretrain checkpoint built *without* descriptor columns cannot warm-start a fine-tune model built *with* them (`RuntimeError: shapes cannot be multiplied`, confirmed directly) — the pretraining stage has to carry the same descriptor columns purely to keep the architecture loadable, even though the public corpus has no genuine use for them. The resulting descriptor matrices are cached to `.chemprop_pretrain/<method>/` (gitignored, alongside the checkpoint) so a crash between finishing them and the checkpoint being written doesn't repeat a step that ran ~40 minutes on the 12,719-compound PubChem corpus at `n_jobs=-1`.

### 13_tdi_pretrained (2026-09-20) — pretraining transfers to TDI, and still loses to TabICL

TDI is the largest deficit in the repo (board macro MCC **0.2111**, rank 94/113, against a visible contender at **0.4025**, rank 12) and had one day of work against regression's ten. The contender's TDI stack is architecturally what `04` already ships — TabICL on frozen embeddings, threshold swept on OOF — so the testable difference was the encoder, and `05`'s PubChem pretraining had never been tried on this track. Four arms, shared folds, 5x5, threshold swept on OOF:

| arm | macro MCC | CYP3A4 | CYP2D6 | vs incumbent | Holm |
|:--|--:|--:|--:|--:|:--|
| **tabicl_singletask** (04's incumbent) | **0.2464** | 0.375 | 0.122 | — | — |
| chemprop_pubchem | 0.2182 | 0.323 | 0.119 | −0.0393 | sig |
| chemprop_pubchem_frozen | 0.1976 | 0.319 | 0.082 | −0.0529 | sig |
| chemprop_scratch | 0.1447 | 0.244 | 0.048 | −0.1144 | sig |

Every optimum at threshold 0.25, consistent with `04`. The incumbent reproduced `04`'s shipped 0.2483 to 0.002, which is what makes the rest of the table trustworthy.

**Pretraining transfers, and the control is what shows it.** `chemprop_pubchem` beats `chemprop_scratch` by **+0.0751** (95% CI [0.0616, 0.0890], p=0.0000, Holm-significant) — a large effect, and the first evidence that `05`'s biggest regression lever works on classification at all. Without the from-scratch control this table would have read as "chemprop is bad at TDI", attributing to the architecture what belongs to the initialisation.

**Freezing is NOT resolvable here, unlike on regression.** Unfrozen beats frozen by +0.0136, 95% CI **[−0.0001, 0.0271]**, p=0.0514 — fails Holm. `05` found the same direction on regression at p=0.000; on TDI the CV cannot resolve it. Do not quote the unfrozen/frozen gap as a TDI finding.

**Pretraining is real but too small to matter.** It moves chemprop +0.073 macro while chemprop starts 0.102 behind TabICL. **The encoder was never the binding constraint on this track** — frozen CheMeleon plus in-context learning suits 1,500–3,600 labelled rows better than a D-MPNN from any initialisation. By elimination the remaining candidate for the gap is the *ensemble*: the contender Caruana-combines two decorrelated candidates and its own README calls that the larger part of its TDI margin. The three chemprop arms here are decorrelated from TabICL and already computed, so that experiment needs no new training.

**The two-stage route is closed, and it is `08`'s rule with a different cause.** `is_TDI` is a pIC50 shift thresholded, and both pIC50 columns have near-complete coverage (1,497 of 1,497 on CYP2D6) exactly where the label is scarce — so the oracle is enormous: thresholding the *true* shift scores MCC **0.687** (CYP3A4) and **0.898** (CYP2D6). But the shift is barely predictable from structure (Spearman 0.216 / 0.124) and the two-stage arm **loses to a plain direct classifier at both isoforms** (−0.075, −0.044). `08` failed on *coverage* (Emax is perfectly co-measured, so stage 1 gained nothing); this fails on *predictability* — a difference of two noisy quantities amplifies noise. Read the predictability column before the oracle column.

**Threshold retuning is exhausted.** Measured on `04`'s OOF: the shipped 0.25 is within +0.003 (CYP2D6) and +0.005 (CYP3A4) of the best available. At matched recall (~0.46) our CYP2D6 precision is **0.120 against the contender's 0.364** — precision at fixed recall is an ordering property, so no threshold can reach it. Both tracks now have the same diagnosis: the gap is ordering, not placement or calibration.

**One number to discount in any expected-performance table.** `04`'s OOF-tuned threshold gave CYP3A4 0.374 and the board returned **0.305**; CYP2D6 landed almost exactly (0.119 vs 0.117). A threshold tuned on OOF is honest for choosing and optimistic as a reported score, and the optimism is concentrated on CYP3A4.

**Infrastructure added:** `aux_training.run_cv_pretrained_classification` (the classification sibling of `run_cv_pretrained`) and `binarise_pretraining_matrix`. Two silent failures found by probe and pinned by tests — see the numpy and chemprop-scaler entries in the environment-traps section.

### 14_tdi_ensemble (2026-09-20) — the ensemble is the first thing to beat 04's incumbent

`13` eliminated the last single-model hypothesis, leaving the ensemble. It wins, and it is the **first arm in either track to beat a shipped incumbent since `05`**. Trains nothing — reads `13`'s OOF.

**Weights are fitted on four outer repeats and scored on the fifth, rotated.** Selecting weights and scoring them on the same predictions measures the selection, not the ensemble; with four candidates and a metric as flexible as MCC that optimism is not small. Every number below is out-of-sample with respect to the weights.

| arm | CYP2D6 | CYP3A4 | macro | vs incumbent | Holm |
|:--|--:|--:|--:|--:|:--|
| **caruana** | 0.1356 | 0.3944 | **0.2650** | **+0.0197** | **sig** (CI [0.0113, 0.0283], p=0.0000) |
| uniform | 0.1315 | 0.3823 | 0.2569 | +0.0100 | no (CI [−0.0008, 0.0210], p=0.073) |
| tabicl_singletask (04) | 0.1192 | 0.3738 | 0.2465 | — | — |
| chemprop_pubchem | 0.1162 | 0.3215 | 0.2188 | — | — |
| chemprop_pubchem_frozen | 0.0780 | 0.3178 | 0.1979 | — | — |
| chemprop_scratch | 0.0467 | 0.2421 | 0.1444 | — | — |

**Selection is doing real work — the uniform control is what proves it.** A plain mean of the same four arms gains only +0.0100 and fails Holm, against Caruana's +0.0197 at p=0.0000. Without that control the win would have been unattributable between "averaging helps" and "the weights help". It is the weights.

**Decorrelation is why this works here and would not have worked on regression.** `13`'s arms correlate at **r = 0.30–0.62** — three of the four are a different model family from the incumbent. `07`'s four regression arms correlate at **r ≈ 0.94**, four variants of one pretrained chemprop, where a blend has nothing to average away. **Check pairwise correlation before spending time on an ensemble**; it predicts the outcome better than the members' individual scores do.

**The weights are stable, which is what makes this shippable.** Across all 10 held-out repeats: `tabicl` 0.49–0.61, `chemprop_pubchem` 0.20–0.27, `chemprop_pubchem_frozen` 0.12–0.23, `chemprop_scratch` 0.026–0.12. No swing that would suggest fitting fold noise. Note `chemprop_scratch` earns a consistently tiny but non-zero weight despite being the worst member by a wide margin (macro 0.1444) — Caruana keeps it as a decorrelated minority voice rather than dropping it.

**CYP3A4 carries the win; CYP2D6 does not resolve.** Per endpoint, Holm-corrected: CYP3A4 +0.0206 (CI [0.0106, 0.0308], p=0.0000, significant); CYP2D6 +0.0165 (CI [−0.0004, 0.0338], p=0.056, **not** significant). Same direction, half the data. Do not claim an ensemble gain on CYP2D6.

**It is worth shipping and it does not explain the gap.** Board macro MCC 0.2111 against the contender's 0.4025 is 0.19; this buys ~0.02 in CV. The ensemble was the last of four hypotheses and the only one that paid, so the remaining difference is something not yet identified — most likely their encoder corpus (a D-MPNN pretrained on ChEMBL/TDC/Polaris plus a Novartis-surrogate CYP panel, against our PubChem qHTS), which this repo cannot replicate.

**Infrastructure:** `ensemble.caruana_selection` — forward selection with replacement, bagged. Chosen over `sweep_weights` because **MCC cannot be decomposed per row** (it comes from a confusion matrix over the whole set), so a coarse weight grid has no structure to exploit; Caruana only ever asks "does adding this candidate improve the score", which any metric can answer. Also safer than a continuous optimizer on correlated candidates, which can reallocate weight destructively between near-duplicates. Pinned by `tests/test_ensemble_selection.py`.

### 15_phase1_review (2026-10-20) — trains nothing; re-tests 03/05/04 with the fold as the unit

`03` and `05` bootstrapped the pooled 32,625 OOF **rows**. Every compound appears in five outer repeats, so the intervals were too narrow, and pooling the four endpoints puts a single constant's error in the denominator rather than the macro-average the board ranks on. `15` bootstraps the 25 **per-fold macro differences** instead (still anti-conservative, since folds from different repeats overlap — read as a lower bound on uncertainty), Holm within each family. Where this changes an earlier claim, `15` wins:

| claim | earlier | fold-level, Holm |
|:--|:--|:--|
| 05 `pubchem` vs control | p=0.000 | −0.0143, p=0.0036, **significant** |
| 05 `aux_screen` vs control | significant | −0.0089, p=0.0256 vs threshold 0.0167, **not significant** |
| 05 `pubchem_frozen` vs control | significant, wrong direction | +0.0220, p=0.000, holds |
| 03 `xgb` / `lgbm` multitask gain | −0.0348 / −0.0256 | −0.049 / −0.037, both significant |
| 03 `tabpfn` multitask | −0.0019, not significant | −0.0071, p=0.05, not significant |
| 03 `tabicl` multitask | +0.0083, "hurts" | +0.0048, p=0.096, **not significant** |
| 04 TDI multitask | only `xgb` significant | same: `xgb` −0.034 only |

Interim board position (full test set, submission 12): macro ST-RAE 0.6384, rank 78/219. Spearman rank is lower (119/219) than ST-RAE rank, so part of our standing comes from placement, not ordering.

### 16_encoders (2026-10-20) — frozen 3D encoders: Monroe beats CheMeleon, Atom-JEPA does not

Same readout as `03`'s control (TabICL, PCA 128, 2 estimators, seed 42), only the embedding changes. Shared folds, macro ST-RAE:

| encoder | macro ST-RAE | vs CheMeleon |
|:--|--:|:--|
| **Monroe** | **0.7020** | better on all four (−0.073 / −0.078 / −0.077 / −0.019 on CYP1A2 / 2C9 / 3A4 / 2D6), Holm-significant |
| CheMeleon (control) | 0.7638 | — |
| Atom-JEPA (last block) | 0.7751 | worse on CYP3A4 (+0.033) and CYP2C9 (+0.023), tied elsewhere |

Monroe alone beats `pubchem` (0.7160) significantly only on CYP1A2 (−0.038); on the other three the difference is not significant. **A plain 50/50 blend of Monroe and `pubchem` scores 0.6660 and wins 25/25 folds**, which is what made `17` worth running.

**Leakage audit (`scripts/audit_monroe_pcba_overlap.py`).** Monroe pretrains on PCBA, which contains the Veith CYP panel the challenge derives from. 225 of 4,905 training compounds (4.6%) and **1 of 750 blind compounds** carry a label there. The blind set is clean; for CV, `17` re-ran the gain with the 225 dropped.

### 17_regression_ensemble (2026-10-20) — pool B ships; correlation of errors, not predictions, predicts blend value

Nothing retrained; weights fitted on four outer repeats and scored on the fifth, rotated. Pools: **A** `pubchem` + Monroe, **B** A + Atom-JEPA + CheMeleon, **C** all 20 candidates. Macro ST-RAE vs `pubchem`: A −0.0505, B −0.0556, C −0.0648 (all 25/25 folds). C−B −0.0091 (24/25 folds), B−A −0.0051 (18/25).

**`14`'s rule of thumb was wrong in one direction.** `14` said ensembles pay at r 0.30–0.62 and not at ~0.94. Monroe and `pubchem` correlate 0.77–0.90 and the blend still won every fold. Predictions correlate; *errors* are what matter, and `marginal_value.csv` measures each of the 20 candidates by what it adds to the Monroe+`pubchem` blend. Uniform averaging of pool C (−0.0376) is much worse than Caruana (−0.0648): with 20 members **selection, not averaging, is the gain**.

### 18_shift_check (2026-10-20) — does the pool ranking survive the blind set's shift

Reweighted OOF toward the blind set on two axes measurable without blind labels (nearest-neighbour similarity, predicted pIC50), pre-registered failure criteria. Effective sample sizes 0.19–0.86 (`effective_sample_size.csv`), so the strongest schemes are noisy.

- The Monroe+`pubchem` blend beats `pubchem` in **100%** of resamples under every scheme.
- C vs B at 300 compounds per endpoint: 88% unshifted, **70–79% shifted**. The pre-registered criterion (sign change on CYP2D6, or p<0.5) did **not** fire — CYP2D6 favours C in every scheme (92–95%). **B was shipped on judgement, not because the criterion tripped.**

### Board record, 17 (live, 2026-10-21) — regression improved everywhere, rank did not

Submission `17_regression_ensemble` (pool B, 12's placement). Snapshot `data/leaderboard/20261021/` (`scripts/snapshot_leaderboard.py`).

| endpoint | 12 (live) | 17 (live) |
|:--|--:|--:|
| CYP1A2 | 0.6256 | **0.5882** |
| CYP2C9 | 0.5567 | **0.5105** |
| CYP2D6 | 0.8902 | **0.8411** |
| CYP3A4 | 0.5586 | **0.5438** |
| macro ST-RAE | 0.6578 | **0.6209** |

**Rank went 84/205 to 85/257**: the field grew by 52 entries while we improved by 0.037, so a better score reads as a worse rank. Compare scores across snapshots, ranks only within one. CYP2D6 (−0.049) and CYP2C9 (−0.046) gained most; CYP3A4 moved least (−0.015).

**TDI did not follow its CV.** `14`'s ensemble (CV +0.0197 over `04`, Holm-significant) shipped inside 17 and scored macro MCC **0.2028** (rank 121; CYP3A4 0.3527, CYP2D6 0.0528) against the 0.2111 `04`'s model scored in 12. The live half is small and `13` already recorded OOF-threshold optimism of this size on CYP3A4, so this neither confirms nor refutes the ensemble, but it is not the +0.02 CV predicted.

### 19_seed_spread / 19b — seed noise in the final fits is real and was never controlled

Chemprop's `--pytorch-seed`/`--data-seed` were **never passed anywhere in the repo** before this; `fit_predict_test_multitarget(seed=)` only seeds the validation carve-out. `19` wraps `graph_models._run_chemprop_cli` to append both, and refits the shipped `pubchem` on 100% of the data with seeds 0–4 from the one pretrain checkpoint (`.chemprop_pretrain/pubchem`; the pretraining seed is still fixed). `19b` does the same for the two TabICL members (seeds 42, 0–3).

| member | seed-pair Spearman, mean | per-compound seed SD (pIC50) | worst case |
|:--|--:|--:|:--|
| `pubchem` | 0.934–0.976 | 0.10–0.15 | CYP2D6 0.934 (worst pair 0.873) |
| `tabicl_monroe` | 0.972–0.992 | 0.056–0.087 | CYP2D6 0.972 |
| `tabicl_atom_jepa` | 0.917–0.983 | 0.070–0.123 | CYP2D6 0.917 (worst pair 0.902) |

**TabICL is not deterministic across seeds** (PCA to 128 and 2 estimators leave real variance), and the noisiest member is the one with the weakest CV score. 05's shipped vector sits at Spearman 0.88–0.97 from the 5-seed mean. Expected gain from seed means on the blend is ~0.003–0.005 macro; CV has no seed-averaged OOF, so it is **unmeasured**, and the Caruana weights were fitted on single-seed members.

### 20_pooled_workbench / 21_pooled_ensemble (2026-10-22) — a worse model that improves the blend

Replicates the SuperCowPowers workbench regression recipe (`union-p30`, [blog](https://supercowpowers.github.io/workbench/blogs/cyp_challenge/)): **one from-scratch multitask chemprop** on challenge rows plus public rows, 17 heads (4 scored endpoints, challenge auxiliary heads, public sources as extra heads on extra rows `PUB-<i>`), inverse-count weights on the scored heads (mean-normalised to 1), 0.30 on the rest, template defaults (hidden 700, depth 5, MAE, patience 40). All public structures matching any challenge compound are excluded. Infrastructure: `aux_training.pooled_wide_frame`, `aux_training.run_cv_pooled`, and `task_weights`/`patience`/`warmup_epochs`/`loss_function` on `ChempropModel`.

| arm | macro ST-RAE |
|:--|--:|
| **Caruana, pool B + pooled** | **0.6390** |
| uniform, pool B + pooled | 0.6476 |
| Caruana, pool B (17) | 0.6604 |
| `pubchem` | 0.7160 |
| pooled alone | 0.7382 |

**Alone it loses; in the blend it wins.** Pooled vs `pubchem`: +0.0222 (worse), 5/25 folds, Holm-significant. Pool B + pooled vs pool B: **−0.0214** (95% CI [−0.0248, −0.0181]), 25/25 folds, Holm-significant, and significant on every endpoint (CYP2C9 −0.033, CYP3A4 −0.022, CYP2D6 −0.017, CYP1A2 −0.013). Uniform averaging gains almost the same (−0.0208), so the gain does not depend on the weight fit. Cause is decorrelation: pooled correlates with `pubchem` at r 0.72–0.88 and with the Monroe members at 0.68–0.86, against ~0.94 among the earlier chemprop variants. **Rank a candidate by its error correlation with the blend, never by its standalone score.**

Limits: Caruana weights come out as equal splits (1/3, 1/4, 1/5) identical across all five rotations, so selection is acting as member choice, not fine weighting. CV is blind to placement, so the 0.021 is untested on the board. 3 pooled seeds disagree more than `pubchem` seeds do (Pearson 0.88–0.95), so the shipped pooled vector is a 3-seed mean.

### 22_pooled_seed_ensemble (2026-10-23) — built, validated, not yet scored

`submissions/22_pooled_seed_ensemble/20261023/`, built by `notebooks/build_pooled_seed_submission.py`. Per-endpoint Caruana over five members, **every blind member a seed mean**: `pubchem` ×5, `tabicl_monroe` ×5, `tabicl_atom_jepa` ×5, pooled ×3, `tabicl_chemeleon` (weight 0 on all endpoints, not fitted; the script asserts this). Weights fitted on all 25 folds of `21`'s single-seed OOF. Placement identical to 17 (CYP1A2/CYP2D6 onto 12's moments, CYP2C9/CYP3A4 onto 05's unplaced). TDI is `14`'s, unchanged. Rank agreement with 17's file 0.94–0.99. Expected CV macro ST-RAE **0.639** (single-seed members, placement-blind) against 0.660 for 17. Passes `submission.check` on both tracks.

### Board record, 22 (live, 2026-10-23) — the pooled model and seed means paid more on the board than in CV

Snapshot `data/leaderboard/20261023/`. Regression macro ST-RAE **0.6209 → 0.5711**, rank 85/257 → **66/252** (CV had predicted 0.660 → 0.639). Per endpoint: CYP1A2 0.5882 → 0.5720, CYP2C9 0.5105 → 0.4249, CYP2D6 0.8411 → 0.8033, CYP3A4 0.5438 → 0.4840. Macro Spearman 0.691 → 0.724 and R² 0.488 → 0.538, so the gain is ordering, not placement (identical to 17). CYP2D6 stays the weak endpoint (Spearman 0.425, best entry 0.582). The extra board gain over CV is not separable between the pooled member, seed means and live-half noise.

### 23_tdi_encoders (2026-10-23) — the regression-side encoder and blend lessons transfer to TDI

TabICL (single-task, PCA 128, 2 estimators, seed 42) on frozen Monroe and Atom-JEPA embeddings, same folds and 0.25 threshold as `13`/`14`, `13`'s OOF reused for the four existing arms. Macro MCC, weights fitted off the held-out repeat:

| arm | CYP3A4 | CYP2D6 | macro |
|:--|--:|--:|--:|
| **caruana, 6 arms** | 0.421 | 0.168 | **0.2945** |
| `tabicl_monroe` | 0.382 | 0.163 | 0.2722 |
| caruana, `14`'s 4 arms | 0.394 | 0.136 | 0.2650 |
| `tabicl_singletask` (CheMeleon) | 0.374 | 0.119 | 0.2465 |
| `tabicl_atom_jepa` | 0.361 | 0.121 | 0.2413 |

Holm-corrected: 6-arm vs 4-arm blend **+0.0278** (CI [0.019, 0.037], p<0.0001); Monroe vs the CheMeleon incumbent +0.0166 (p=0.015); Atom-JEPA vs incumbent −0.0088, not significant, yet it still helps the blend as a decorrelated member (r 0.67–0.74 with the other TabICL arms). The uniform 6-arm average gains the same +0.0278 over the uniform 4-arm, so the gain does not depend on the weight fit. Monroe takes the largest weight (0.27 CYP3A4, 0.32 CYP2D6).

**Monroe's PCBA overlap does not drive it.** 225 challenge compounds appear in Monroe's pretraining corpus; dropping them leaves blend +0.0269 (CI [0.018, 0.036]) and Monroe alone +0.0146 (p=0.036). Two Atom-JEPA conformer failures among the 1,240 TDI compounds outside `16`'s cache were imputed with the mean embedding (`atom_jepa_bad.json`). The pruned 5-arm pool in the output (0.2970) was defined after seeing weights and is descriptive only.

**Expect shrinkage on the board.** `14`'s blend was 0.265 in CV and 0.2028 live.

### 24_tdi_encoders (2026-10-23) — built, validated, not yet scored

`submissions/24_tdi_encoders/20261023/`, built by `notebooks/build_tdi_encoders_submission.py`: `23`'s six-arm blend, every arm refit on 100% of the rows and seed-averaged (TabICL ×5, Chemprop ×3, per-seed blind probabilities in `experiments/24_tdi_seed_fits/`), weights from all 25 folds, threshold 0.25. Regression is `22`'s `activity.csv` unchanged. Label agreement with 17's TDI file 91.1% (CYP3A4) and 85.6% (CYP2D6).

## Placement (07) — separable from ordering, and where the leaderboard gap is

A contender scored macro ST-RAE **0.4378** against our 0.78 ([blog](https://supercowpowers.github.io/workbench/blogs/cyp_challenge/), [code](https://github.com/SuperCowPowers/workbench/tree/main/ml_pipelines/OpenADMET/cyp)). Most of the gap is not a better model. `src/cyp/placement.py` implements the part reproducible without spending submissions we do not have.

**`R2 = 2*rho*k - k^2 - b^2`**, with `rho` Pearson correlation, `k = sd(pred)/sd(true)`, `b` the mean offset in units of `sd(true)`. Only `rho` depends on ordering; `k` and `b` are set by an affine transform that moves no compound's rank. So:

- **R2 is capped at `rho^2`, reached at `k = rho`, not `k = 1`.** Matching the truth's spread is wrong: a model correlating 0.7 should be 70% as wide as reality. This is the quantitative form of PXR's regression-to-the-mean lesson.
- **A returned score is an equation in the blind population's moments** — scores already earned constrain `mean(y)` and `sd(y)` without a further submission.

**Do not hardcode the contender's solved constants.** Their `BLIND_MOMENTS`/`STRAE_MOMENTS` were bought with three affine probes plus four board probes of CYP2D6's centre; we get `rho` free because the board reports Spearman (`placement.pearson_from_spearman`), and our three submissions being *different models* is an advantage — each contributes a curve and the truth is where they meet (`placement.intersect`). Read its `disagreement`: above ~0.3 pIC50 units the solve has not identified the population and must not be used.

**The solve needs R2 and Spearman; ST-RAE alone cannot drive it** (not a squared-error metric, and not placement-invariant). What ST-RAE *does* support alone is calibrating CV against the board: every submission ships an expected per-endpoint ST-RAE in PROVENANCE.md, and the board-to-CV ratio measures how far scaffold CV sits from blind difficulty. Check its spread across submissions before extrapolating.

**ST-RAE's optimum is not R2's, and the difference has a sign.** ST-RAE scores zero inside a credible interval and low-activity compounds carry wide ones, so predicting too high is nearly free and too low is punished. The optimum sits *above* the true centre and *narrower* than `rho*sd`. `placement.strae_optimal_placement` finds it by grid search against held-out data, not by probing a board.

**CV is structurally blind to this.** Measured on 05's OOF, `recoverable` R2 is 0.000–0.002 everywhere — an affine correction only pays where the prediction population differs from the *scoring* population, and OOF is scored against the distribution it trained on. A flat CV result does not mean placement is worth nothing.

**05 interim reveal (2026-09-16)**, `experiments/07_placement/board_decomposition.csv`. The four per-task R2s average to 0.1985 against the board's MA-R2 of 0.1986, confirming the plain arithmetic mean:

| endpoint | ST-RAE | R2 | Spearman | R2 ceiling | recoverable | % ceiling lost |
|:--|--:|--:|--:|--:|--:|--:|
| CYP1A2 | 0.6887 | 0.3393 | 0.7423 | 0.5744 | 0.2351 | 40.9% |
| CYP2C9 | 0.5567 | 0.5868 | 0.7629 | 0.6050 | 0.0182 | 3.0% |
| CYP2D6 | **1.3099** | **−0.7421** | **0.3401** | 0.1255 | 0.8676 | 691% |
| CYP3A4 | 0.5586 | 0.6102 | 0.7831 | 0.6356 | 0.0254 | 4.0% |
| macro | 0.7785 | 0.1985 | 0.6571 | 0.4852 | 0.2866 | 59% |

**CYP2D6 is both a weak-ordering and a catastrophic-placement problem.** Spearman 0.34 caps R2 at 0.126 whatever we do, but we scored −0.742 — the 0.868 gap is pure placement. Fixing placement cannot make CYP2D6 good; it can stop it being a disaster. **The other three are already well placed**; only CYP1A2 has real headroom (40.9%). Do not spend placement effort on CYP2C9 or CYP3A4.

**MAE cannot resolve the sign; ST-RAE can.** `moments_from_r2` returns a mirrored pair of centres, and the two candidates implied MAEs identical to four decimals at every endpoint — so `solve_moments` was choosing direction arbitrarily. `placement.resolve_sign_with_strae` simulates each candidate against the board's ST-RAE; on the 05 reveal it picks **LOW at all four endpoints** and the simulated scores track the board closely.

Solved blind moments (05's board row, sign resolved by ST-RAE):

| endpoint | our mean | our sd | solved mean | solved sd | centre gap |
|:--|--:|--:|--:|--:|--:|
| CYP1A2 | 4.900 | 0.826 | 4.258 | 1.420 | +0.642 |
| CYP2C9 | 4.816 | 0.782 | 4.683 | 0.990 | +0.133 |
| CYP2D6 | 4.530 | 0.531 | 3.114 | 1.520 | **+1.416** |
| CYP3A4 | 4.684 | 0.941 | 4.502 | 1.150 | +0.182 |

Simulating at these moments reproduces the board (0.747 vs 0.7785), the end-to-end check. Applying the correction: macro **0.747 raw → 0.646 at `k = rho` → 0.613 at the ST-RAE optimum**, no model change. Our spreads are consistently narrower than the contender's (1.42/0.99/1.52/1.15 vs 4.412/1.101/1.599/1.272) and centres lower on CYP1A2/CYP3A4. Two independent solves of the same live half, disagreeing by 0.1–0.4; neither obviously authoritative. Treat the direction as settled and exact values as uncertain.

**The moments describe the live half only.** The 750-compound test set was split by chemical series; a correction fitted to the live half transfers to the final one only insofar as the split preserved the distribution, which it is designed not to. Record this in any PROVENANCE.md shipping a placed submission.

**Shipped and scored: macro 0.7785 → 0.7040** (`submissions/07_placement/20260916_pubchem_r2/`, rank 87, 2026-09-16). Placed at the **R2 optimum** deliberately, to buy a second differently-placed vector for `intersect`:

| endpoint | board ST-RAE | board R2 | board Spearman | CV expected | ratio |
|:--|--:|--:|--:|--:|--:|
| CYP1A2 | 0.6559 | 0.4334 | 0.7050 | 0.8051 | 0.815 |
| CYP2C9 | 0.5975 | 0.5660 | 0.7658 | 0.6144 | 0.972 |
| CYP2D6 | 0.8902 | 0.2230 | 0.3610 | 0.9158 | 0.972 |
| CYP3A4 | 0.6725 | 0.5338 | 0.7683 | 0.4854 | **1.385** |
| macro | **0.7040** | 0.4390 | 0.6500 | 0.7052 | 1.000 |

**CYP2D6 went R2 −0.742 → +0.223** and carries the whole macro gain. **But CYP3A4 got measurably worse** (+0.187 ST-RAE over CV) from being pushed to `k = rho` it did not need at 96% of its ceiling. **Placement on an already-well-placed endpoint is a cost, not a no-op** — apply per endpoint, gated on measured `recoverable`, never uniformly. The macro landing on the CV prediction was partly cancellation (CYP1A2 over-, CYP3A4 under-performed); a macro-only comparison would have hidden both.

**With two scored submissions `intersect` solves three endpoints** — CYP1A2 4.378/1.080, CYP2C9 4.871/1.200, CYP3A4 4.856/1.340, all `disagreement` under 0.005. **A two-vector `disagreement` is not evidence of correctness** — see `12`, where a third CYP1A2 vector pushed it from 0.0029 to 0.2804 and the three pairwise solves disagreed by 0.5 pIC50 on centre.

**CYP2D6 does not solve, and the failure is a finding.** Submission 87's board R2 (0.223) exceeds the ceiling implied by its own Spearman (0.361 → pearson 0.376 → ceiling 0.141), so `intersect` correctly refuses. The only consistent reading is that CYP2D6's true blind Pearson is **at least 0.472** — the bivariate-normal conversion underestimates by ~30%, because weak correlations are where a skewed, truncated pIC50 distribution departs furthest from normality. **Treat `pearson_from_spearman` as a lower bound on CYP2D6, not a point estimate.**

### 10_pka_features — a sharper basic-nitrogen feature is selective, and still does not reach the model

09 left one loose end: `pharmacophore.n_basic_nitrogen` is a binary sp3-nitrogen heuristic that scores a strongly basic amine (pKa ~9-10, always charged at pH 7.4) the same as a weakly basic one next to an electron-withdrawing group (pKa ~4-5, always neutral). CYP2D6's pharmacophore is about the *cationic* nitrogen engaging Glu216/Asp301, so the binary version encodes the wrong quantity. `pka.basic_nitrogen_weight` (vendored MolGpKa GCN, `vendor/molgpka/`) replaces it with the Henderson-Hasselbalch fraction protonated from a predicted per-atom pKa. One column swapped, everything else identical to 09's `ecfp+pharmacophore_2d` — the cleanest available test of whether feature *sharpness*, not shape, was the limit.

**The direct check is the strongest CYP2D6-selective signal in any notebook** (`direct_check.csv`, Spearman against potency):

| endpoint | binary basic-N | pKa weight | % nonzero |
|:--|--:|--:|--:|
| CYP1A2 | −0.134 | −0.017 | 27.3 |
| CYP2C9 | −0.157 | −0.069 | 25.0 |
| CYP2D6 | **0.276** | **0.342** | 53.9 |
| CYP3A4 | −0.065 | −0.002 | 24.9 |

Sharpening the feature raises CYP2D6 by +0.066 and moves the other three *toward zero*. This is the selectivity pattern 09 looked for and did not find — the mechanism is real and the continuous feature expresses it better.

**It does not survive into a model.** LightGBM/ECFP4-2048, same folds and harness as 09 (5x5 scaffold, seed 0), scored in Spearman:

| endpoint | ecfp | +pharmacophore_2d | +pka_weight | pka lift |
|:--|--:|--:|--:|--:|
| CYP1A2 | 0.4214 | 0.4245 | 0.4158 | −0.006 |
| CYP2C9 | 0.3713 | 0.3802 | 0.3739 | +0.003 |
| CYP2D6 | 0.3365 | 0.3684 | **0.3681** | +0.032 |
| CYP3A4 | 0.6048 | 0.6081 | 0.6070 | +0.002 |

`ecfp+pka_weight` and `ecfp+pharmacophore_2d` land within 0.0003 of each other on CYP2D6 (+0.0316 vs +0.0319 over ECFP) — a dead heat, on point estimates that 15 folds cannot separate anyway. **A 24% better univariate correlation bought nothing once ECFP was in the model**, because the binary indicator already tells a tree which compounds have the atom and ECFP's neighbouring bits supply enough context to grade it.

**The rule this adds:** improving an auxiliary feature's correlation with the target does not predict a model gain when a rich representation is already present — check the lift against the *existing feature set*, not against nothing. Same shape as 08's coverage lesson from a different direction: there the auxiliary was redundant in *coverage*, here it is redundant in *information*.

Combined with 09 and 11, three independent routes to the CYP2D6 basic-nitrogen pharmacophore (3D geometry, predicted pKa, docked protein contacts) all confirm the mechanism and none beat the incumbent. **The pharmacophore is not the bottleneck.** `02`'s scorecard is — lowest test-NN similarity (0.469), highest extrapolation (10.8%). Do not open a fourth pharmacophore route.

### 11_docking_3d (2026-09-19) — structure-based 3D: docking helps through protein contacts, not through Uni-Mol

09 closed the ligand-side 3D question (explicit pharmacophore geometry hurt a pretrained encoder). This asked what 09 couldn't: does the *protein* matter. **gnina could not be used** — CUDA-only (sole v1.3.3 asset is a 1.96 GB CUDA-12.8 Linux binary), and this machine is an arm64 Mac with no NVIDIA GPU. **smina** (the Vina fork gnina derives from) substitutes; `src/cyp/docking.py`, `src/cyp/plif.py` (ProLIF interaction fingerprints), `src/cyp/unimol.py`.

**A six-ligand probe before building anything found the trap first:** Uni-Mol's CLS embedding from a docked pose is nearly identical to the same molecule's free ETKDG conformer (cosine 0.80–1.00, against a 0.72 between-molecule floor) — it was pretrained on isolated ligands and cannot see the protein. So `unimol_docked` needs `unimol_etkdg` as a control, or a win would read as "Uni-Mol is a good 3D featurizer" rather than "docking helped."

**Scoped to CYP2D6 alone.** Full four-isoform docking was estimated at ~7h and actually cost **15.6h for CYP2D6 alone** — smina per-chunk time (100-molecule batches, `docking.dock_in_chunks`) varied 390s–11,662s (30x) on identical-complexity molecules, confirmed as ordinary desktop contention (`mediaanalysisd`, caught live twice) rather than thermal throttling (zero throttling events across 16h). Stopped CYP1A2 at 8/15 chunks (cached, resumable) and redirected the budget to CYP2D6 — weakest baseline, and the receptor most likely to show the literature pharmacophore.

**LGBM, 5x5 nested scaffold CV, five arms, CYP2D6 only:**

| comparison | ST-RAE diff | p | Holm-sig |
|:--|--:|--:|:--|
| `plif` vs `ecfp` | **−0.0196** | 0.0004 | yes |
| `docking_score` vs `ecfp` | −0.0150 | 0.0038 | yes |
| `unimol_docked` vs `unimol_etkdg` | **+0.0194 (worse)** | 0.0246 | yes |
| `unimol_etkdg` vs `ecfp` | −0.0160 | 0.1876 | no |

**The protein-contact fingerprint wins; feeding the pose into a ligand-only model actively hurts it.** ProLIF's docked CYP2D6 poses recover **Glu216 and Asp301** directly — the literature basic-nitrogen anchors 09's ligand-geometry descriptors failed to express — alongside the Phe cluster lining the site (Phe120 82% hit rate). `docking_score` (smina affinity, one column) also beats ECFP alone. `unimol_docked` significantly *underperforms* its own free-conformer control, confirming the probe: Uni-Mol cannot productively use a pose.

**But this does not beat what's already shipped.** 05's `pubchem`-pretrained chemprop (ST-RAE 0.9251, reused OOF) still beats every LGBM arm here (1.043–1.065). PLIF helps within LGBM; it does not close the gap to the incumbent.

**Chemprop promotion was structurally blocked for docking-derived descriptors, not merely unimplemented — fixed same day.** `aux_training.run_cv_pretrained`'s `descriptor_columns` path computed `pharmacophore.descriptors()` on the **pretraining corpus itself** — valid for 09's SMILES-derived geometry, not for PLIF/`docking_score`, which need a pose docked against the CYP2D6 receptor that the public PubChem corpus never had. Failed with `ColumnNotFoundError` only *after* a full ~1h pharmacophore-descriptor computation over the 8,061-compound corpus, since the function never validated the descriptor source up front. Fixed by adding `pretrain_descriptors_from_smiles: bool = True` — `False` zero-pads the pretrain-stage block instead of computing it, purely to keep `--checkpoint`'s FFN width loadable. Regression-tested (`tests/test_auxiliary.py`) by reproducing the exact crash at the old default and confirming it disappears with the flag set.

**Atom-level PLIF via chemprop's `--atom-features-path`, added the same session.** `plif.atom_features` builds a multi-hot per-heavy-atom interaction-type vector (not a single binary flag — 5.3% of contacted CYP2D6 atoms carry more than one interaction type at once) from ProLIF's `fp.ifp` per-atom contact records, mapped onto chemprop's own heavy-atom indexing via `Chem.AddHs`'s index-preserving order (confirmed directly, not assumed). **`--atom-features-path` only works with a single combined `--data-path` file plus `--splits-file`** — confirmed against chemprop 2.2.1's source (`build_splits`/`load_input_feats_and_descs`): the flag is read once per training call and applied identically to whichever CSV is currently being parsed, so it cannot pair one array with the separate train/val/test files every other `fit`/`predict` in `graph_models.py` uses. `ChempropMultitargetModel._fit_with_atom_features` is the new single-file/splits-file code path this required; it also means atom features need an **explicit** `smiles_val`/`y_val` (no internal random split, since a `.npz`'s row order can't be reordered along with it).

**Chemprop promotion run, same day: neither molecule-level nor atom-level PLIF beats the incumbent.**

| arm | mechanism | ST-RAE |
|:--|:--|--:|
| `pubchem` (05, reused OOF) | incumbent, no PLIF/docking input | **0.9251** |
| `pubchem_plif` | chemprop, PLIF as `--descriptors-columns`, zero-padded pretrain | 0.9665 |
| `chemprop_atom_plif` | chemprop, PLIF as `--atom-features-path`, trained from scratch | 1.0023 |
| `pubchem_docking_score` | chemprop, smina affinity as `--descriptors-columns`, zero-padded pretrain | 1.0057 |
| best LGBM arm (`plif`) | — | 1.0430 |

Alignment against 05's OOF checked directly (1,497 shared fold/compound rows, 0 `y_true` mismatches), so the `pubchem` row is a legitimate number to beat, not voided.

**Molecule-level PLIF is chemprop's best non-incumbent arm, and warm-starting is doing real work.** `pubchem_plif` beats every LGBM arm including its own `plif` (0.9665 vs 1.0430) and beats `pubchem_docking_score`, the same mechanism with a weaker descriptor — consistent with 05/09's general finding that a learned, pretrained encoder plus a good molecule-level feature outperforms either alone. It still falls 0.0414 short of `pubchem`, so the extra descriptor columns did not help beyond what a plain pretrained encoder already gets from ECFP-equivalent structural signal — the same shape of result 09 found for pharmacophore descriptors, this time not even reversing sign.

**Atom-level PLIF underperforms its own molecule-level twin, and the likely cause is the missing warm-start, not the feature.** `chemprop_atom_plif` (1.0023) trained from scratch — no PubChem-pretrained encoder is compatible with `--atom-features-path`'s input width, so this arm never got 05's biggest lever (pretraining helps only with the encoder unfrozen, and there was no pretrained encoder to unfreeze here). Whether atom-level attribution beats molecule-level pooling once warm-started is a different, unanswered question; this result cannot distinguish "the representation doesn't help" from "no encoder pretrain was available to receive it."

**Conclusion: ship `pubchem` as-is.** No docking-derived chemprop arm built this session beats 05's incumbent. `plif` remains the best evidence that the protein matters (it wins decisively within LGBM), but every route tried to carry that signal into the model that would actually ship — molecule-level descriptors, atom-level features, with or without a fresh pretrain — landed short of a model that never saw the protein at all. A from-scratch atom-level warm-start bridge (pretraining a `--atom-features-path`-shaped encoder on zero-padded public atom features, mirroring the molecule-level fix) is the one variant not yet tried, and is the natural next step if this is revisited.

### 12_final_placement (2026-09-19, scored 2026-09-20) — per-endpoint placement, and the solve's first failed check

Shipped the interim submission with **no model change** — `05`'s `pubchem` arm, still the incumbent after 06, 08, 09, 10 and 11 each failed to beat it. Three changes over 07, all on placement: `intersect` with two curves instead of a single-vector sign guess, the ST-RAE optimum instead of the R2 optimum, and **per-endpoint gating** on measured `recoverable` (threshold 0.05; there is a 10x gap between the two groups so no threshold between them changes the decision).

**Scored: macro ST-RAE 0.7040 → 0.6578, rank 84/205.** TDI unchanged from 04 as shipped: macro MCC 0.2111, rank 94 (vs 0.2483 CV — ordinary OOF-threshold optimism).

| endpoint | regime | ST-RAE | R2 | expected | verdict |
|:--|:--|--:|--:|:--|:--|
| CYP1A2 | placed, ST-RAE optimum | **0.6256** | 0.4147 | 0.42–0.51 | missed by 0.115 |
| CYP2C9 | unplaced (gated) | 0.5567 | 0.5868 | 0.5567 | exact |
| CYP2D6 | reuse 07's vector | 0.8902 | 0.2230 | 0.8902 | exact |
| CYP3A4 | unplaced (gated) | 0.5586 | 0.6102 | 0.5586 | exact |
| macro | — | **0.6578** | 0.4587 | 0.607–0.629 | missed |

**The three controls passed exactly**, four decimals, across submissions weeks apart. **The live half is stable and scores are comparable across submissions** — the one property everything else in this section assumes, now measured rather than hoped for. Gating is also vindicated directly: CYP2C9/CYP3A4 held their scores where 07's uniform placement had cost them 0.041 and 0.114.

**Placement direction right, magnitude wrong by ~2x.** CYP1A2 went 0.6887 unplaced → 0.6256, its best score across all three submissions. But the simulator overshot by +0.203, outside the +0.040/+0.128 range its own back-test spanned. **Bracketing a forecast by a previously observed error assumes that error is bounded by what you have seen; with two points it is not.** Treat `strae_optimal_placement`'s simulated score as a direction, never a magnitude — it has now failed twice.

**The three-vector solve fails its first real check, and this is the important finding.** A third scored, differently-placed CYP1A2 vector makes `intersect` overdetermined — the first point at which the solved population can be *checked* rather than merely computed. It does not survive:

| vectors | solved mean | solved sd | disagreement |
|:--|--:|--:|--:|
| 05 + 07 | 4.378 | 1.080 | 0.0029 |
| 05 + 12 | 4.222 | 1.660 | 0.0029 |
| 07 + 12 | 4.736 | 1.670 | 0.0036 |
| **all three** | 4.183 | 1.100 | **0.2804** |

Every *pair* agrees to 0.003 while landing on populations 0.5 apart in centre and 0.6 in spread; adding the third vector pushes disagreement to 0.2804, at the ~0.3 ceiling above which this section already says the solve must not be used. **Pairwise agreement was never evidence of correctness** — two curves always intersect somewhere, and a tight `disagreement` on two vectors measures only that a solution exists, not that it is the right one. The same blind spot as CYP2D6's ρ scan, which also showed near-zero disagreement at every ρ while the centre slid a full pIC50 unit. **`disagreement` is only informative once the system is overdetermined.** Do not quote a two-vector solve's tightness as confidence again.

The practical reading: the solved *population* is not identified to better than ~0.5 pIC50 on centre, but the *placement direction* it implies has now paid on the board three times. Use it to choose a direction and gate which endpoints move; do not treat its moments as measured constants.

**CYP1A2 still has headroom.** Board Spearman 0.7423 → Pearson 0.758 → R2 ceiling 0.574, against 0.4147 scored: **0.160 still recoverable**, the only endpoint where that is true (CYP2C9 and CYP3A4 are at 97% and 96% of ceiling). Its three measured placements, narrowest scoring best: sd 0.826 → 0.6887, sd 1.075 → 0.6559, sd 0.648 → **0.6256**. Centre moved too, so spread and centre are not separated by these three points.

## Auxiliary and public data

**Structural exclusion is keyed on the InChIKey connectivity block, never canonical SMILES.** `external.inchikey_skeleton` / `skeleton_keys`, used by every `exclude_smiles` path in `aux_training`. SMILES distinguishes stereoisomers, tautomers and salts that are the *same compound* for leakage purposes, so it under-removes. Measured: rekeying collapses 451 records in Veith CYP2D6, 140 in ChEMBL's, 47 in Tox21's, and switching the union matrix removed **184 more compounds (1,083 labels)**. On current snapshots the two keys agree on *blind-set* overlap (7 either way) — luck, not a property to rely on, since 960 source records match a *training* compound only under the skeleton key. 05's shipped corpus audits clean under the stricter key. Rows RDKit can parse but not serialise to InChI keep a null key and are **retained** — silently discarding public training data is the worse failure.

**Public sources.** Three downloaders writing dated snapshots under `data/external/` (`cyp.external`, `cyp.chembl`, `cyp.tox21`). `aux_training.full_union_matrix` assembles **24,718 compounds × 16 heads × 125,193 labels**, each source on its own head at its own scale — never merged into a scored column, which is what makes heterogeneous sources safe.

| source | compounds | what it uniquely supplies |
|:--|--:|:--|
| Veith qHTS pIC50 | ~12,900 | the base corpus |
| Veith `max_response` | same | efficacy at 100% coverage vs ~50% for potency |
| ChEMBL 37 (5 isoforms) | 41,403 | **14,899 new skeletons**; no low-end range |
| Tox21 P450-Glo (3 isoforms) | 4,893 | **~2,400 explicit inactives per isoform** |

Two blog claims did **not** reproduce: Tox21's median pIC50 is **4.92–5.07** here (not 4.76), so its value is the confirmed negatives, not a lower median; and ChEMBL's cross-protocol disagreement is **0.18–0.50 log units**, not the feared >1.

**`max_response` is the cheapest real gain** — 100% coverage vs ~50%, doubling pretraining labels from 38,375 to 77,602, and it carries the inactive half of the library our hit-enriched labels never reach. Strongest correlation with challenge pIC50 on CYP2D6 (−0.569). `external.load_pubchem` returns it clipped to `constants.MAX_RESPONSE_CLIP` — a fixed window, not a percentile, because the positive tail differs by two orders of magnitude across isoforms. `aux_training.public_union_matrix` adds one head per isoform.

**The challenge's own unused arms are the cheapest lever.** `aux_training.challenge_auxiliary_matrix` — 12 heads, 22,081 labels, no download, same lab and protocol. TDI-condition pIC50 lifts CYP3A4 coverage 2,335 → **3,583 (+53%)** and brings 1,240 new compounds. Emax's *direct-inhibition* variant correlates **+0.770 Spearman** with CYP2D6 potency vs −0.39 to −0.46 elsewhere. The blog's 0.555 is the same quantity's **Pearson** (0.5546 measured here) — state which correlation you mean whenever quoting it.

## Multitask vs single-task

Every harness in `models.py` trains one model per endpoint; `multitask.py` is the controlled alternative. The multitask arm always runs against its single-task twin, never as a replacement.

**Folds must be shared — a correctness issue.** 1,309 of 4,905 compounds carry more than one endpoint, so per-endpoint `scaffold_splits` can put a compound in CYP3A4's training set and CYP2D6's test set at once. `cv.shared_scaffold_folds` assigns folds over the union; `cv.fold_assignment_splits` applies them per endpoint. Sharing folds is also what makes `evaluation.paired_bootstrap` between arms legitimate.

**"Multitask" means three different things** (`multitask.STRATEGY`):

- `stacked` (trees, TFMs): one model on all 6,525 rows, endpoint one-hot encoded. Without the indicators the model learns the average of four targets.
- `native` (Macau): genuine sparse `(n_compounds, 4)` factorization; missing entries absent, not imputed.
- `multitarget` (Chemprop/CheMeleon): one D-MPNN, four heads, unmeasured cells **empty, not zero** (a 0.0 reads as a real measurement of an inactive compound). Only 41 of 4,905 compounds have all four endpoints, so dropping incomplete rows is not an option.

**TDI gets its own harness** — `run_cv_stacked_classification`, `run_cv_multitarget_classification`, `run_cv_multitask_classification` (dispatcher, keyed on `STRATEGY_CLASSIFICATION`) and `run_cv_singletask_classification` (the shared-fold control). No `native` strategy (Macau has no classification formulation), which is why `models.CLASSIFIER_FACTORIES` omits it too. The regression runners cannot be reused: `_oof_records` casts `y_true` to float and carries `y_lower`/`y_upper` a boolean label lacks.

**Classification OOF frames carry `y_prob` alongside `y_pred`**, and `y_pred` is always exactly `y_prob >= threshold`, so a threshold sweep reproduces the harness's labels without a 25-fold refit. A threshold tuned on OOF is honest for *choosing* but mildly optimistic as a *reported* score.

**Both arms must get identical model settings.** `run_cv_singletask_classification` takes `**model_kwargs` and raises on an unknown attribute rather than silently dropping it. An arm trained for more epochs than its twin measures the epochs.

## Lessons carried from the PXR challenge

Hard-won on a completed blind challenge ([repo](https://github.com/adlvdl/pxr_challenge), [blog](https://www.delavega.ai/blog.html)). They cost real leaderboard places.

- **Do not rank models on differences the data cannot resolve.** In PXR three finalists spanned 0.0039 MAE and the ordering reversed between phases, costing five places. Use `evaluation.paired_bootstrap`, `holm_bonferroni` for many variants, `mcs.make_mcs_grid` for the across-methods view. A grid with few stars means the CV cannot support a ranking.
- **Regression to the mean is the dominant failure mode.** Already visible here. Check `evaluation.bias_by_potency_bin` on every run.
- **Calibration is cheap insurance.** Dismissed in PXR CV at 0.0001 MAE; won the blind set. Cross-fit (`calibration.crossfit_calibrate`) for evaluation; fit on all OOF only for final test predictions.
- **Ensembling suppresses catastrophic errors** rather than improving typical accuracy — which is what ST-RAE rewards. Track `ensemble.catastrophic_rate`.
- **HPO pays off for tree models, not neural nets.** XGBoost went 0.64 → 0.52 MAE in PXR; graph models barely moved.
- **What PXR missed and cost the most:** public data beyond the challenge release, and 3D/structural information — most of the top ten used docking or cofolding.

## Conventions

- New featurizers go in `fingerprints.py`, registered in `_FP_REGISTRY`. New models go in `models.py`'s `MODEL_FACTORIES`. Classification gets its own factories (`models.CLASSIFIER_FACTORIES`), harness (`models.run_cv_classification`) and comparison table (`evaluation.compare_methods_classification`) rather than branching the regression versions — the OOF schemas differ and the metrics are unrelated. `models.MajorityBaseline` is the classification analogue of `MeanBaseline`: MCC = 0.0 by construction.
- **Exploration and plotting live in four modules** (see `notebooks/02_chemical_space.py`): `embedding.py` (UMAP/t-SNE plus the per-endpoint panel grid), `similarity.py` (Tanimoto, NN coverage, activity cliffs), `mmp.py` (the `mmpdb` shell-out and network plot), `scaffolds.py` (scaffold decomposition). `interactive.py` holds the Altair+RDKit hover pattern and is deliberately **not** imported by `cyp/__init__.py` (it pulls in marimo; `import cyp` must work without the notebook extra). Add to these rather than re-deriving a plot in a notebook.
- **UMAP on fingerprints uses Jaccard, not Euclidean** — Jaccard on a bit vector is 1 − Tanimoto; Euclidean on 4096 bits is dominated by heavy-atom count. `embedding.add_umap` defaults to it and binarises counts first.
- **A Tanimoto threshold is meaningless without naming its fingerprint.** `similarity.CLIFF_SIM_THRESHOLDS` (MACCS ≥ 0.8, ECFP4 ≥ 0.4); check against `similarity.similarity_percentiles` on the current snapshot — a threshold above a fingerprint's p99.9 selects noise.
- **`cv.murcko_scaffold` and `scaffolds.decompose` answer different questions.** Murcko: one scaffold per molecule, for grouping folds. The scaffold *network*: every ring system and core, for coverage. A compound can be Murcko-novel while every ring in it is well represented.
- **Progress bars count folds, not cells.** Every CV harness takes `on_fold` (`cv.FoldCallback`, invoked as `on_fold(fold, n_folds)`); notebooks pass a closure ticking `mo.status.progress_bar`. A 5x5 run is 25 fits per (method, endpoint) and Chemprop takes ~90s/fold, so a per-cell bar sits motionless for ~37 minutes — the point at which someone kills a working run. Cached branches advance by the folds they skipped. `cv.report_fold` swallows callback exceptions.
- **Timings go to `experiments/<notebook>/timings.csv` as each measurement is taken**, via `cyp.timings.record`. Caching means a timer inside the training branch measures nothing on a rerun, and a run that dies partway still leaves what finished. `timings.summary` for per-method totals; `timings.extrapolate` projects a quick-mode run onto the full 5×5, which is how you decide affordability *before* starting. Re-timing replaces a row rather than appending.
- **Run long sweeps fold-major, cache per fold.** Outer loop = fold, inner = (method, arm); key the cache on `(method, arm, fold)`. The CV runners take `folds=[...]` for this. The failure mode matters more than speed: method-major caching killed at hour six leaves some methods complete and the rest missing, so no comparison is possible; fold-major leaves every method complete through fold *k* — a smaller but fully usable paired comparison. `test_fold_slices_reassemble_into_the_whole_run` pins that ordering cannot change results.
- **Cache CV, not the final fit.** Cache OOF to `experiments/<notebook>/cache/`, keyed by whatever varies the result. Do **not** cache the fit-on-everything-and-predict step: it is cheap and a separate cache silently goes stale against a retrained CV cache — a real bug caught here, where a submission used calibration parameters from before a CV retrain.
- **Prefer `mcs.reference_forest_plot` and `mcs.paired_arm_plot` over the heatmap past ~8 methods.** Effect size on x with CI, one row per method, colour carrying the verdict. They scale because each method takes a row rather than a row *and* a column, and they answer "what should I ship, and is it better than the incumbent?" `paired_arm_plot` is the multitask one (one row per **model**, each arm against its twin). All three are driven by the same `rm_tukey_hsd` — except `paired_arm_plot`'s p-values are uncorrected paired t-tests, deliberately most permissive; read its intervals for shape and settle significance with `paired_bootstrap` + `holm_bonferroni`.
- **`mcs.make_mcs_grid` sizes itself to the method count; do not pass `figsize`.** A fixed size degrades into unreadable overlapping text by ~12 methods and fails *silently*, writing the PNG either way. Geometry derives from `mcs.CELL_IN` outward; `top_n` trims a large grid. Two bugs not to reintroduce: a hidden axes still claims its share of `subplots_adjust` width (hence `ncol = min(len(titles), ...)`), and `sns.heatmap(square=True)` shrinks the matrix rather than growing the box.
- Every notebook plot is saved as a PNG under `experiments/<notebook_name>/` (`dpi=300, bbox_inches="tight"`), not left as inline output — see `mcs.make_mcs_grid`'s `save_path` for the pattern.
- One marimo notebook per run: `notebooks/NN_name.py` (pure `.py`). Anything reused belongs in `src/cyp/`.
- A notebook writes diagnostics to `experiments/<notebook_name>/` and submissions to `submissions/<notebook_name>/<date>/`, deriving the name from `Path(__file__).stem`. Both tracks (`activity.csv`, `tdi.csv`) share one dated subfolder. Every submission folder gets a `PROVENANCE.md` recording model, calibration, CV protocol, data snapshot, **and an expected-performance table (per endpoint plus macro) for both tracks** — computed fresh for the exact submitted method/calibration. This is what lets you check the interim reveal against a number instead of a surprise.
- Save OOF as `oof.parquet` in the experiment directory — every comparison, calibration and ensemble step reads that schema (`cv.oof_frame`).
- Large purely-derived caches (no training) are gitignored; caches that took model fits are committed. Derived summaries and figures are always committed.
- **Leaderboard snapshots go to `data/leaderboard/YYYYMMDD/leaderboards.csv`** via `scripts/snapshot_leaderboard.py` (long format: `board` live/interim, `track`, `scope` endpoint/overall, `is_ours`). Filter `board`, not `scope`, for live vs interim. Ranks are only comparable within one snapshot; the field grew 205 → 257 between 2026-09-19 and 2026-10-21.
- **Submission builders live in `notebooks/build_*.py`** (`build_tdi_ensemble_submission.py`, `build_pool_b_submission.py`, `build_pooled_seed_submission.py`), run as scripts, refuse to overwrite an existing dated folder, and call `submission.check` before writing the provenance. Final-fit seeds are explicit (`graph_models` passes `--pytorch-seed`/`--data-seed` only when a script patches them in, as `19` and `20` do), and seed means are cached under `experiments/19_seed_spread/`.

## Environment traps on this machine (16 GB Apple Silicon)

**OpenMP load order — `import cyp` must come first.** `lightgbm` (`lib_lightgbm.dylib`), `scikit-learn` and `torch` each vendor an OpenMP runtime; whichever loads first wins. If LightGBM is not first, computing a fingerprint then fitting LightGBM **segfaults** — no catchable exception. `src/cyp/__init__.py` imports lightgbm first to claim the slot; keep it at the top. `KMP_DUPLICATE_LIB_OK=TRUE` does **not** fix it. The conflict is bidirectional: once LightGBM's runtime is loaded, an in-process TabICL/TabPFN fit segfaults, which is why every torch model runs in a subprocess (`tabular_models.predict_subprocess` / `run_cv_subprocess`, `graph_models.chemeleon_embed`, the Chemprop CLI, `graph_models.device`). Do not "simplify" any into an in-process call. **Any new torch module joins this list** — `src/cyp/pka.py` (vendored MolGpKa) hit this directly (exit 139) and runs every prediction through a subprocess.

**A torch model fitted in-process after Macau deadlocks silently** — no exception, no CPU use, no log line. The 2026-09-13 run lost 14.5 hours to this: the multitask arm correctly used `predict_subprocess` but the single-task arm called `MODEL_FACTORIES["tabicl"]()` directly after Macau ran in the same process. `_warn_if_openmp_conflict` emits a `RuntimeWarning`, but a warning is not a fix. If a run stalls at 0% CPU, sample the process and look for `kmp_flag_64::wait` in `libomp.dylib`.

**The two TFMs need separate memory budgets.** Measured at n=2335, PCA-reduced from 2048-dim CheMeleon:

| | features | n_est | peak RSS | time | corr |
|:--|--:|--:|--:|--:|--:|
| TabICL | 128 | 2 | 3.89 GB | 5s | — |
| TabICL | 128 | 8 | 10.96 GB | 35s | — |
| TabPFN | 128 | 2 | 4.50 GB | 10s | 0.158 |
| TabPFN | **192** | **2** | **6.58 GB** | **9s** | **0.232** |
| TabPFN | 256 | 4 | 8.44 GB | 22s | 0.240 |

TabICL's memory scales hard with *ensemble size* and the extra buys nothing (corr 0.999 across configs); TabPFN's scales with *features* and it genuinely uses them (0.158 at 128 → 0.232 by 192, then plateaus). Hence `TABICL_MAX_FEATURES=128`/`TABICL_N_ESTIMATORS=2` vs `TABPFN_MAX_FEATURES=192`/`TABPFN_N_ESTIMATORS=2`, enforced by `check_tabicl_budget`/`check_tabpfn_budget`. **There is no `MemoryError` to catch** — TabICL at 2048×8 swapped the machine to a standstill and the run reported nothing. Raise limits only while watching RSS. For scale: Chemprop ~2 GB per fit, trees under 1 GB.

**TabPFN's token must be the API key, not a session JWT.** `TABPFN_TOKEN` in `.env` (gitignored); `tabular_models.load_env` loads it. The value is the API key from ux.priorlabs.ai, looking like `tabpfn_sk_...`. A browser JWT decodes as valid but gets a 401 from `api.priorlabs.ai/protected/`, and TabPFN reports that identically to having no token. If TabPFN claims no token while one is set, check that endpoint's HTTP status first.

**Force Chemprop to CPU: `CYP_CHEMPROP_DEVICE=cpu`** (see `graph_models.device`). On MPS, per-fold time went 100s, 91s, 97s then rose to ~2000s and **plateaued there permanently** (~20x). On CPU: 80.7, 75.7, 76.4, 75.9, 79.6, 75.8, 76.1s — no drift, and faster than MPS ever was, since a 408K-parameter model at batch 64 never gives the GPU enough work to amortise launch overhead. Fold-major interleaving does **not** prevent the degradation, but it makes it *diagnosable* — with other methods running between Chemprop fits, you can see that some slow folds hit every method at once while the Chemprop climb is specific and permanent.

**Machine contention is a separate, self-recovering problem.** The clearest evidence: `majority_singletask`, plain `np.mean` on a boolean array, took 3.6s against a 0.1–0.2s baseline (18–36x) — a method with no computation cannot have a genuine slow fold. Two confirmed causes, neither universal: Spotlight/`mdworker` CPU activity, and a depressed kernel `ApplePassthroughPPM` thermal budget with *no* competing process (self-inflicted throttling after ~9h of sustained load). Diagnosis only works **live** — the responsible process has usually exited by the time anyone asks afterward.

`timings.flag_contention(path)` classifies a finished log: any unit ≥`CONTENTION_FLOOR_SECONDS` (200s) **and** ≥`CONTENTION_MULTIPLE` (3x) its method's 25th-percentile baseline, with the `majority_singletask` numbers pinned in `tests/test_timings.py`. One known false positive: a pretrained arm's first fold pays a real one-time pretraining cost (`aux_training.run_cv_pretrained`) that nothing in the CSV schema marks. `scripts/watch_contention.sh` (`make watch-contention TIMING=...`, run in a second terminal) is the live half, snapshotting `ps`, `pmset -g therm` and a `ThermalPowerBudget` log grep the moment a unit is flagged. Written for bash 3.2 — no associative arrays.

**Every `cmd | head` or `cmd | tail` inside a `set -eo pipefail` script needs an explicit fallback.** `watch_contention.sh` silently died on its own first real detection, twice: `head` closing its read end sends `ps` a SIGPIPE, and `pipefail` turns that into the pipeline's status, killing the whole script with no error text. It only fired inside `snapshot()`, i.e. only on an actual flagged row, which is why it looked fine in short manual tests. A consumer that stops reading early is not an error in any normal sense but reads as one under `pipefail`. Found by tracing a real reproduction (`bash -x` plus a manual `trap ... EXIT`) — synthetic tests with too little history could not reproduce it.

**A regression chemprop checkpoint silently corrupts a classification fine-tune.** `load_and_use_pretrained_model_scalers` applies the checkpoint's `UnscaleTransform` to the new targets whenever `--checkpoint` is passed, and there is no flag to skip it. With mismatched head counts it raises (`operands could not be broadcast together with shapes (250,2) (1,3)`), and only at fine-tune time — after a pretrain that runs ~40 minutes on the full corpus. With *matching* head counts it is worse: training completes, nothing warns, and `predict` returns values on the **pIC50 scale** instead of probabilities (measured: a "probability" of **−0.525**), which downstream reads as a model that learned nothing. The fix is that both stages must be the same task type: `aux_training.binarise_pretraining_matrix` converts a pIC50 panel to binary so the pretrain is classification and no `UnscaleTransform` is ever written. `run_cv_pretrained_classification` checks the head count before the fold assignment. Both pinned in `tests/test_auxiliary.py`. Note `--task-type` is one model-wide flag — chemprop has no per-head task type, so a single model cannot mix continuous auxiliary heads with binary TDI heads.

**numpy 2.2.x silently zeroes the TDI track.** Under numpy 2.2.6 a TabICL classification fit on ~2,400+ rows returns **all-NaN probabilities** and raises nothing — the fit reports success, `predict_subprocess` exits 0, and every downstream MCC reads 0.000, which looks exactly like "the model learned nothing". Bisected directly: clean at numpy 2.4.6 and 2.5.3, 100% NaN at 2.2.6, same code, same folds, same seed; CYP2D6 (~1,200 training rows) is unaffected, so it presents as one endpoint mysteriously failing. **`unimol_tools` pins `numpy<2.3.0`**, so installing it downgrades numpy and breaks TabICL for the whole repo — and `make setup` runs `uv sync --all-extras`, which would pull it in. It is therefore **not in any extra**; `pyproject.toml` pins `numpy>=2.4` and `uv sync` fails outright if both are requested. To re-run `11`'s Uni-Mol arm, install it into a throwaway venv and point `src/cyp/unimol.py`'s subprocess at that interpreter. This cost a full 25-fold TDI run: `13`'s first attempt scored its own incumbent at macro MCC 0.068 against a known 0.248, which read as a bad baseline rather than a broken dependency.

**gnina cannot run on this machine — CUDA-only.** The sole v1.3.3 release asset is a 1.96 GB CUDA-12.8 Linux binary and every documented build path needs `nvcc`, against an arm64 Mac with no NVIDIA GPU. `smina` (`brew install brewsci/bio/smina`) substitutes, being the Vina fork gnina itself derives from; the only capability lost is gnina's CNN pose rescoring. See `src/cyp/docking.py`.

**ProLIF segfaults (exit 138, no traceback) on an unprotonated crystal PDB.** MDAnalysis infers bond orders when converting to RDKit, and that inference dies outright on a structure with no hydrogens — which every raw PDB download is. Fix: protonate with `obabel -h -p 7.4` before ProLIF ever sees the receptor (`docking.prepare_receptor`'s `_rec_h.pdb` output exists for exactly this).

**`unimol_tools` forks worker processes internally, so a caller without a `__main__` guard fork-bombs** — many Python processes at high CPU, with the real error ("An attempt has been made to start a new process before the current process has finished its bootstrapping phase") buried in output. Distinguish from the OpenMP deadlock above by CPU: that one sits at 0% in `kmp_flag_64::wait`, this one runs hot. Every call goes through a subprocess anyway (torch), and the subprocess script carries the guard (`src/cyp/unimol.py`).

**Docking timing on this machine is unreliable to the point of needing chunked execution.** smina per-100-molecule chunk (`docking.dock_in_chunks`) varied 390s–11,662s (30x) for identical-complexity molecules across a single continuous run, with **zero thermal throttling detected** (`pmset -g therm` and the kernel `ThermalPowerBudget` log checked repeatedly, never fired) — confirmed as ordinary desktop contention, `mediaanalysisd` (Photos indexing) caught live at 66–94% CPU at two of the slow chunks. An early clean chunk repeatedly produced estimates that inflated 2–4x by completion; the estimate only stabilised after roughly two-thirds of the chunks had landed. Budget a docking run at a multiple of the median-chunk extrapolation, not the extrapolation itself, and re-quote after each new chunk rather than anchoring on the first.
