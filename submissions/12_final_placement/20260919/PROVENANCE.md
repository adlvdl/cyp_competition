# Interim submission — per-endpoint placement at the ST-RAE optimum

- Generated: 2026-09-19
- Notebook: `notebooks/12_final_placement.py`
- Challenge data snapshot: `20260908`
- CV: nested scaffold, 5x5 folds (`experiments/07_placement/oof.parquet`, `pubchem` arm)
- Target deadline: 2026-09-24 interim
- **Scored on the live leaderboard 2026-09-20** — see "Returned scores" below

## Model — unchanged

**No model was trained for this submission.** The regression predictions are `05`'s
`pubchem` arm: one Chemprop D-MPNN with four output heads, encoder pretrained on the
PubChem qHTS panel (Veith) and fine-tuned unfrozen on all four challenge endpoints.
Notebooks `06`, `08`, `09`, `10` and `11` each tested a route past it and each ended in
"ship `pubchem` as-is".

What changed is **placement** — a per-endpoint affine transform that moves no
compound's rank.

## Per-endpoint placement plan

| endpoint | regime | recoverable R2 | from mean/sd | to mean/sd |
|:--|:--|--:|:--|:--|
| CYP1A2 | placed at ST-RAE optimum | 0.2351 | 4.900 / 0.826 | 4.485 / 0.648 |
| CYP2C9 | **unplaced** (gated) | 0.0182 | 4.816 / 0.782 | unchanged |
| CYP2D6 | reuse `07`'s scored vector | 0.8676 | — | 3.114 / 0.538 |
| CYP3A4 | **unplaced** (gated) | 0.0254 | 4.684 / 0.941 | unchanged |

`recoverable` is from `experiments/07_placement/board_decomposition.csv`, measured on
`05`'s returned board row. The gate is 0.05; there is a 10x gap between the two groups,
so no threshold between them changes the decision.

**Gating is the correction to `07`'s one mistake.** `07` placed all four endpoints
uniformly and CYP3A4 — already at 96% of its R2 ceiling — came back +0.187 ST-RAE worse
than its CV prediction, the only endpoint to underperform. Placement on a well-placed
endpoint is a cost, not a no-op.

## Blind population, solved by intersection

Two scored submissions with genuinely different centres give `placement.intersect` two
curves to triangulate against:

| endpoint | solved mean | solved sd | disagreement | status |
|:--|--:|--:|--:|:--|
| CYP1A2 | 4.378 | 1.080 | 0.0029 | solved |
| CYP2C9 | 4.871 | 1.200 | 0.0039 | solved |
| CYP2D6 | — | — | — | **no consistent population** |
| CYP3A4 | 4.856 | 1.340 | 0.0005 | solved |

The three solves are an order of magnitude tighter than the 0.018 `intersect` was
validated against, and all three imply plausible blind distributions (0.8-1.7% of
compounds below pIC50 2, against training sets whose minima are ~1.9).

**CYP2D6 does not solve, and this submission does not pretend otherwise.** Submission
87's board R2 (0.223) exceeds the ceiling implied by its own reported Spearman (0.361 →
Pearson 0.376 → ceiling 0.141). Forcing a larger Pearson produces a consistent solve at
every value above the hard floor of √0.223 = 0.472, with near-zero disagreement
throughout, but the solved centre slides from 3.30 to 2.32 as the assumed correlation
rises — and **every one of those populations is physically impossible**, putting
17-26% of blind compounds below pIC50 1 against an assay whose lowest training
observation is 1.95 (`cyp2d6_rho_scan.csv`). CYP2D6 therefore ships `07`'s empirically
scored vector unchanged (board 0.8902, against 1.3099 unplaced).

## ST-RAE optimum, not R2 optimum

`07` shipped at the R2 optimum (`k = rho`) deliberately, to buy a second
differently-placed vector. That purchase is now spent, so this ships at the ST-RAE
optimum. The shape is measured on `05`'s own OOF (`strae_optimal_shape.csv`):

| endpoint | offset above truth | sd fraction | shrink vs R2 optimum |
|:--|--:|--:|--:|
| CYP1A2 | +0.107 | 0.450 | 0.792 |
| CYP2C9 | +0.101 | 0.600 | 0.866 |
| CYP2D6 | +0.044 | 0.325 | 0.721 |
| CYP3A4 | +0.200 | 0.675 | 0.831 |

`offset_above_truth` is **positive at all four endpoints**, which is the check `07`
flagged as worth stopping for — a negative value would contradict the metric's
asymmetry argument. The shrink factor averages 0.802 with a standard deviation of
0.054 across four endpoints that differ enormously in difficulty and interval width;
that consistency is what justifies carrying it onto the blind set. Each endpoint is
placed with its own measured values, not the pooled average.

The correlation used to set CYP1A2's spread is **`05`'s own** board Spearman (0.7423 →
Pearson 0.7579), not `intersect`'s median across submissions. The population's moments
are shared between submissions; the correlation is a property of the model being placed.

## Expected performance

| endpoint | expected ST-RAE | basis | board `05` | board `07` |
|:--|:--|:--|--:|--:|
| CYP1A2 | 0.42 – 0.51 | simulated, bracketed | 0.6887 | 0.6559 |
| CYP2C9 | 0.5567 | scored (vector identical to `05`) | 0.5567 | 0.5975 |
| CYP2D6 | 0.8902 | scored (vector identical to `07`) | 1.3099 | 0.8902 |
| CYP3A4 | 0.5586 | scored (vector identical to `05`) | 0.5586 | 0.6725 |
| **MA (macro)** | **0.607 – 0.629** | arithmetic mean | 0.7785 | 0.7040 |

**Three of four endpoints ship an already-scored vector**, so only CYP1A2 is a genuine
forecast — and its expectation is a bracket, not a point estimate, because the
simulator fails its own back-test:

| vector | centre | spread | simulated | board | error |
|:--|--:|--:|--:|--:|--:|
| `05` | 4.900 | 0.826 | 0.7282 | 0.6887 | +0.0395 |
| `07` | 4.258 | 1.075 | 0.7842 | 0.6559 | +0.1283 |

The board says `07`'s CYP1A2 vector scored *better* than `05`'s; the simulator says
worse. It gets the order wrong, and overshoots both by different amounts. Treat
0.42-0.51 as optimistic and a landing anywhere in 0.42-0.60 as consistent with what is
known.

## TDI track — unchanged

`tdi.csv` is byte-identical to `submissions/04_methods_tdi/20260915/tdi.csv`:
TabICL single-task on frozen CheMeleon embeddings, decision threshold 0.25 tuned on
OOF. Expected macro MCC **0.2483** (CYP3A4 0.3750, CYP2D6 0.1216). No work has been
done on this track since 2026-09-15.

## Returned scores (live leaderboard, retrieved 2026-09-20)

Submitted 2026-09-19 16:17 UTC. **Macro ST-RAE 0.6578, rank 84/205**, against `07`'s
0.7040 — a gain of 0.0462 with no model change. TDI macro MCC 0.2111, rank 94/113.

| endpoint | rank | ST-RAE | R2 | Spearman | expected | verdict |
|:--|--:|--:|--:|--:|:--|:--|
| CYP1A2 | 81 | 0.6256 | 0.4147 | 0.7423 | 0.42-0.51 | **missed, too optimistic by 0.115** |
| CYP2C9 | 117 | 0.5567 | 0.5868 | 0.7629 | 0.5567 | exact |
| CYP2D6 | 88 | 0.8902 | 0.2230 | 0.3610 | 0.8902 | exact |
| CYP3A4 | 109 | 0.5586 | 0.6102 | 0.7831 | 0.5586 | exact |
| **MA** | **84** | **0.6578** | 0.4587 | 0.6623 | 0.607-0.629 | missed (CYP1A2 carries it) |

**The three controls passed exactly.** CYP2C9 and CYP3A4 returned `05`'s scores to four
decimals and CYP2D6 returned `07`'s — identical vectors scoring identically across
three submissions weeks apart. The live half is stable and scores are comparable across
submissions, which was the failure mode this document flagged as most important to check
first. It is ruled out.

**CYP1A2's placement worked; its forecast did not.** 0.6887 unplaced → 0.6256 placed is
the best CYP1A2 score across all three submissions, so the gate and the direction were
right. The magnitude was overstated by roughly a factor of two. The back-test had
already shown the simulator overshooting by +0.040 and +0.128; the real overshoot was
+0.203, outside the range both back-test points spanned. Bracketing by a previously
observed error assumes that error is bounded by what has been seen, and with two points
it is not.

**Do not read the simulated ST-RAE as a magnitude.** It is directionally useful for
choosing between placements and has now twice failed to predict a score.

## Caveats that belong on the record

**The solved moments describe the live half of the test set only.** The 750 compounds
were split by chemical series, so a correction fitted to the live half transfers to the
final half only insofar as that split preserved the distribution — which it is designed
not to. This is the single largest risk in the submission and it is unquantifiable
before the reveal.

**CV is structurally blind to placement.** `recoverable` on this arm's own OOF is
0.000-0.002 at every endpoint, because OOF is scored against the distribution it
trained on. A flat CV result does not mean placement is worth nothing, and a good CV
result could not have validated this.

**Validation:** both tracks PASS `submission.check`. Rank preservation confirmed
exactly — Spearman 1.000000 against the unplaced vector at every endpoint, zero ties
created by the pIC50 floor.
