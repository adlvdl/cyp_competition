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

**With two scored submissions `intersect` solves three endpoints** — CYP1A2 4.378/1.080, CYP2C9 4.871/1.200, CYP3A4 4.856/1.340, all `disagreement` under 0.005.

**CYP2D6 does not solve, and the failure is a finding.** Submission 87's board R2 (0.223) exceeds the ceiling implied by its own Spearman (0.361 → pearson 0.376 → ceiling 0.141), so `intersect` correctly refuses. The only consistent reading is that CYP2D6's true blind Pearson is **at least 0.472** — the bivariate-normal conversion underestimates by ~30%, because weak correlations are where a skewed, truncated pIC50 distribution departs furthest from normality. **Treat `pearson_from_spearman` as a lower bound on CYP2D6, not a point estimate.**

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
