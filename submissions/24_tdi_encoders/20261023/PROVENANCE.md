# TDI: six-arm encoder blend + 22's regression

- Generated: 2026-10-23
- Script: `notebooks/build_tdi_encoders_submission.py`
- Evidence: notebook `23_tdi_encoders` (CV), `16`/`19b` (encoders, seed spread)
- Regression: `22_pooled_seed_ensemble/20261023/activity.csv`, copied unchanged
- CV: nested scaffold 5x5, shared folds with `13`/`14`; threshold fixed at 0.25
- **Not yet scored on the board.**

## TDI model

Per-isoform Caruana blend (MCC at 0.25) of six arms, each fitted on 100% of the labelled rows,
every blind-set member a seed mean:

- `tabicl_singletask` (CheMeleon), `tabicl_monroe`, `tabicl_atom_jepa`: TabICL, PCA 128, 2 estimators, 5 seeds
- `chemprop_scratch`, `chemprop_pubchem`, `chemprop_pubchem_frozen`: `13`'s arms, 3 seeds each

Weights, fitted on all 25 folds of single-seed OOF:

| endpoint | tabicl_singletask | tabicl_monroe | tabicl_atom_jepa | chemprop_scratch | chemprop_pubchem | chemprop_pubchem_frozen |
|---|---|---|---|---|---|---|
| CYP3A4 | 0.2220 | 0.2860 | 0.2380 | 0.0000 | 0.1270 | 0.1270 |
| CYP2D6 | 0.1820 | 0.3640 | 0.1820 | 0.0000 | 0.1820 | 0.0910 |

## What changed against 17's TDI file (= 14's ensemble)

| isoform | positive_rate_new | positive_rate_17 | label_agreement_with_17 |
|---|---|---|---|
| CYP3A4 | 0.4890 | 0.4560 | 0.9110 |
| CYP2D6 | 0.2800 | 0.2720 | 0.8560 |

## Expected performance (CV, macro MCC, weights fitted off the held-out repeat)

| method | CYP2D6_is_TDI | CYP3A4_is_TDI | macro |
|---|---|---|---|
| caruana_P1 | 0.1683 | 0.4208 | 0.2945 |
| caruana_P0 | 0.1356 | 0.3944 | 0.2650 |
| tabicl_singletask | 0.1192 | 0.3738 | 0.2465 |

`caruana_P1` is this submission's blend, `caruana_P0` is 14's. The gain is +0.028 (CI 0.019 to 0.037,
Holm-significant) and survives dropping the 225 compounds that appear in Monroe's PCBA pretraining
corpus (+0.027). Two Atom-JEPA conformer failures were imputed with the mean embedding.

**Do not quote a board score from this.** 14's blend scored 0.265 in CV and 0.2028 on the live board,
so a threshold-tuned OOF figure is optimistic, and the live half is small. Seed-averaging is not in
the CV table, and the weights were fitted on single-seed members.
