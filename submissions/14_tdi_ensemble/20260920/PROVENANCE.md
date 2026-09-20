# TDI submission — Caruana ensemble of four arms

- Generated: 2026-09-20
- Builder: `notebooks/build_tdi_ensemble_submission.py` (run as a script)
- Selection: `notebooks/14_tdi_ensemble.py`, arms from `notebooks/13_tdi_pretrained.py`
- Challenge data snapshot: `20260908`
- External data snapshot: `20260916` (PubChem corpus, pretraining only)
- CV: nested scaffold, 5x5 shared folds (`cv.shared_scaffold_folds`, seed 42)
- Track: TDI only. **The regression track is unchanged** — the live activity
  submission remains `submissions/12_final_placement/20260919/activity.csv`.

## Model

A **Caruana-weighted blend of four arms**, each fitted on 100% of the labelled data
and predicting the 750 blind compounds. Decision threshold **0.25** for both
isoforms.

| arm | representation | CYP3A4 weight | CYP2D6 weight |
|:--|:--|--:|--:|
| `tabicl_singletask` | frozen CheMeleon (2048-d), one model per isoform | 0.538 | 0.571 |
| `chemprop_pubchem_frozen` | D-MPNN, PubChem-pretrained encoder, frozen | 0.231 | 0.171 |
| `chemprop_pubchem` | D-MPNN, PubChem-pretrained encoder, unfrozen | 0.205 | 0.229 |
| `chemprop_scratch` | D-MPNN, no pretraining | 0.026 | 0.029 |

`tabicl_singletask` is `04`'s shipped incumbent; the three chemprop arms are new in
`13`. Weights are refit on all 25 folds for shipping and match `14`'s held-out means
to within 0.02 at every position (e.g. CYP3A4 `tabicl` 0.538 vs 0.541), which is the
evidence that the selection is not fitting fold noise.

## Expected performance

Out-of-sample **with respect to the ensemble weights**: weights fitted on four outer
repeats, scored on the fifth, rotated (`14`). Selecting and scoring weights on the
same predictions would measure the selection, not the ensemble.

| endpoint | expected MCC | incumbent (`04`) | diff | Holm |
|:--|--:|--:|--:|:--|
| CYP3A4 | 0.3944 | 0.3738 | +0.0206 | **significant** (CI [0.0106, 0.0308], p=0.0000) |
| CYP2D6 | 0.1356 | 0.1192 | +0.0165 | not significant (CI [−0.0004, 0.0338], p=0.056) |
| **MA (macro)** | **0.2650** | 0.2465 | **+0.0197** | **significant** (CI [0.0113, 0.0283], p=0.0000) |

**The macro gain is carried by CYP3A4.** CYP2D6 moves the same direction on half the
data and does not resolve. Do not read this submission as an improvement on CYP2D6.

**A uniform mean of the same four arms was the control** and gains only +0.0100
(p=0.073, fails Holm). The weighting is doing the work, not the averaging — which is
why a fitted object is being shipped rather than a parameter-free mean.

## Why an ensemble at all

Three single-model hypotheses for the TDI gap were tested and rejected first:
threshold retuning (+0.003/+0.005 available), a two-stage model through the pIC50
shift (oracle MCC 0.69/0.90 but loses to a direct classifier), and a pretrained
encoder (`13`: transfers at +0.075 over from-scratch, still 0.039 behind TabICL).

The ensemble works here because the arms are **decorrelated** — pairwise correlation
of OOF probabilities runs 0.30–0.62, and three of the four are a different model
family from the incumbent. For contrast, `07`'s four regression arms correlate at
r ≈ 0.94 and would not support this.

## Caveats on the record

**Threshold-optimistic, concentrated on CYP3A4.** 0.25 was tuned on OOF. `04`
measured that discount directly: its OOF said CYP3A4 0.374 and the board returned
0.305, while CYP2D6 landed almost exactly (0.119 vs 0.117). Expect the same shape —
the CYP3A4 number above is the one to discount.

**The blind set predicts more positive than training**, as `04` documented and
verified as genuine rather than a bug: CYP3A4 45.6% positive here against a 21.3%
training rate (`04`'s submission: 48.8%), CYP2D6 27.2% against 21.6%. Two real
causes — the blind set is hit-enriched, and TabICL's in-context behaviour shifts
between an 80% fold fit and a 100% final fit.

**91.3% label agreement with `04`'s submission** (1,370 of 1,500), so this is an
incremental improvement on the shipped incumbent rather than a different model.

**This does not explain the gap to the leaderboard.** The visible contender scores
macro MCC 0.4025 against our board 0.2111; this buys ~0.02 in CV. The ensemble was
the last of four hypotheses and the only one that paid, so the remaining difference
is unidentified — most plausibly their encoder corpus (a D-MPNN pretrained on
ChEMBL/TDC/Polaris plus a Novartis-surrogate CYP panel), which this repo cannot
replicate.

## Reproducing

```bash
CYP_CHEMPROP_DEVICE=cpu uv run python notebooks/build_tdi_ensemble_submission.py
```

Requires `13`'s pretrain checkpoints under `.chemprop_pretrain/chemprop_pubchem{,_frozen}/`
— the script raises rather than silently training a different model if they are
absent. The final fit is deliberately **not cached** (CLAUDE.md: cache CV, never the
final fit).
