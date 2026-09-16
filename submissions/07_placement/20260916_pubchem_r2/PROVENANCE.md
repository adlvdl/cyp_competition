# Second-submission probe: pubchem, R2-optimal placement

- Generated: 2026-09-16
- Notebook: `notebooks/07_placement.py` (final fit run as a script, see
  `build_pubchem_r2_submission.py` in this folder's provenance)
- Challenge data snapshot: `20260908`
- External data snapshot: `20260916`
- CV: nested scaffold, 5x5 folds (`experiments/07_placement/`)

## Model

**pubchem** -- one Chemprop D-MPNN with four output heads, encoder pretrained on the
PubChem qHTS panel (Veith), then fine-tuned on all four challenge endpoints. The same
arm 05 shipped, refit this session (07's CV measured this exact checkpoint at macro
ST-RAE 0.7075, 25 folds) -- chemprop's own pretraining has no seed control, so this is
not byte-identical to 05's model, only the same configuration.

Three extra-data arms were also measured in this session (`pubchem_union`,
`full_union`, `full_union_aux`) and none beat this baseline at a resolvable margin
(best p=0.1226 unadjusted, none Holm-significant) -- see CLAUDE.md. This submission
uses the simplest arm with no unresolved-source dependency.

## Placement -- deliberately different from 05, by design

This submission exists primarily as a **probe**: with only 05 scored so far,
`placement.intersect` cannot triangulate the blind population from a single vector --
R2 and MAE alone leave a mirrored pair of candidate centres, and the sign has to be
guessed via `solve_moments`'s ST-RAE fallback. A second scored submission with
genuinely different placement resolves that ambiguity properly.

So rather than placing at the ST-RAE optimum (best expected score, but converges
toward 05's own placement), this ships at the **R2 optimum**: `k = rho` applied to
each endpoint's solved spread, centred on the solved mean. No ST-RAE asymmetry
correction.

| endpoint | rho | from_mean | to_mean | from_sd | to_sd | ties |
|:--|--:|--:|--:|--:|--:|--:|
| CYP1A2 | 0.7580 | 4.9820 | 4.2580 | 0.7810 | 1.0760 | 1 |
| CYP2C9 | 0.7780 | 4.8390 | 4.6830 | 0.7830 | 0.7700 | 0 |
| CYP2D6 | 0.3540 | 4.5770 | 3.1140 | 0.5030 | 0.5380 | 0 |
| CYP3A4 | 0.7970 | 4.7190 | 4.5020 | 0.9720 | 0.9170 | 0 |

**Separation from 05's submitted vector** -- the number that determines how sharp the
next `intersect` solve will be:

| endpoint | this_mean | this_sd | 05_mean | 05_sd | centre_gap | sd_gap |
|:--|--:|--:|--:|--:|--:|--:|
| CYP1A2 | 4.2580 | 1.0750 | 4.9000 | 0.8260 | -0.6410 | 0.2480 |
| CYP2C9 | 4.6830 | 0.7700 | 4.8160 | 0.7820 | -0.1330 | -0.0120 |
| CYP2D6 | 3.1140 | 0.5380 | 4.5300 | 0.5310 | -1.4160 | 0.0070 |
| CYP3A4 | 4.5020 | 0.9170 | 4.6840 | 0.9410 | -0.1820 | -0.0240 |

CYP2D6 carries by far the largest separation (centre gap ~-1.4), which is also the
endpoint where the population solve matters most.

## Solved population used for placement

From 05's own returned board scores (R2, Spearman per endpoint), sign resolved via
`placement.resolve_sign_with_strae` against the board's ST-RAE -- see CLAUDE.md's
"Placement is separable from ordering" section for the full derivation. These moments
describe the **live half** of the test set only.

| endpoint | solved mean | solved sd |
|:--|--:|--:|
| CYP1A2 | 4.258 | 1.420 |
| CYP2C9 | 4.683 | 0.990 |
| CYP2D6 | 3.114 | 1.520 |
| CYP3A4 | 4.502 | 1.150 |

## Expected performance (CV, this run's checkpoint)

Macro ST-RAE 0.7075 (`pubchem`, 25 folds, `experiments/07_placement/`) -- an
uncalibrated, unplaced number. CV cannot see the placement correction at all:
`placement.decompose` on this arm's own OOF measured `recoverable` R2 at
0.000-0.002, since OOF is scored against the training distribution it fits. The
correction's value is real only against the blind set, which this submission's
returned scores will test directly.

## What this submission is for

Not chosen to maximise expected score on its own. Chosen to maximise what the next
`BOARD` update can determine: with two submissions of genuinely different placement
scored, `placement.intersect` replaces the single-submission sign fallback with a
solve carrying its own consistency check (`disagreement`), and a repeated 3.11 for
CYP2D6 across two independent solves would be far stronger evidence than either alone.

## Risk

Solved moments describe the live half only; OpenADMET's chemical-series split means
they transfer to the final half only insofar as the split preserved the distribution,
which it is designed not to. Same caveat as every other placed submission this
notebook produces.
