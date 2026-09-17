# Vendored third-party code

## `cyp_challenge_tutorial/`

Scoring and validation modules copied from the official
[OpenADMET/CYP-Challenge-Tutorial](https://github.com/OpenADMET/CYP-Challenge-Tutorial),
which is itself a port of the challenge scoring backend.

- **Upstream commit:** `858ae63ce79934113bccdb7fc65467de5f7b1935` (2026-08-19)
- **Fetched:** 2026-09-08
- **License:** see `cyp_challenge_tutorial/LICENSE` (upstream's own terms)

Vendored so that local scoring and submission validation match the leaderboard exactly
and work offline. `src/cyp/metrics.py` and `src/cyp/submission.py` import from here.

**Treat these files as read-only.** To pick up upstream changes, re-download rather
than editing:

```bash
D=vendor/cyp_challenge_tutorial
B=https://raw.githubusercontent.com/OpenADMET/CYP-Challenge-Tutorial/main
for f in evaluation/config.py evaluation/custom_scoring_functions.py \
         evaluation/evaluate_predictions.py evaluation/utils.py \
         validation/__init__.py validation/activity_validation.py \
         validation/tdi_validation.py LICENSE; do
  curl -sfL -o "$D/$f" "$B/$f"
done
```

Then update the commit hash above, run `make test`, and check whether
`evaluation/config.py` changed any endpoint names (which would require matching edits
in `src/cyp/constants.py`).

The `__init__.py` files in `cyp_challenge_tutorial/` and `cyp_challenge_tutorial/evaluation/`
are added locally to make the packages importable; upstream does not ship them.

The tutorial's notebooks are not vendored — read them upstream.

## `molgpka/`

Per-atom pKa prediction, adapted from [Xundrug/MolGpKa](https://github.com/Xundrug/MolGpKa)
(Pan et al., *J. Med. Chem.* 2021) — used by `notebooks/10_pka_features.py` to weight
CYP2D6's basic-nitrogen feature by predicted protonation rather than a binary
sp3-nitrogen heuristic.

- **Upstream commit:** `4dc8352` (2024-01-11, latest on `master`; repo appears unmaintained
  since)
- **Fetched:** 2026-09-16
- **License:** MIT (`molgpka/LICENSE.md`) — note the file's copyright line reads
  "Copyright (c) 2017-Present OpenNMT", which is a copy-paste error in upstream's own
  repo, not ours. The MIT terms are otherwise unambiguous; cite Pan et al. 2021, not
  OpenNMT.

**This entry is an adaptation, not a pin — unlike `cyp_challenge_tutorial/` above.**
Upstream's `GCNNet` depends on `torch_geometric.nn.GCNConv`/`GlobalAttention` and
`torch_scatter.scatter_add`. `torch_scatter` is a compiled extension pinned to an exact
torch build with no reliable prebuilt wheel for Apple Silicon, which would mean a
from-source compile for a single ~60-line layer. `molgpka/net.py` reimplements that
layer's exact math (symmetric-normalised GCN propagation, gated attention pooling) in
plain `torch.Tensor` ops instead. The two `.pth` weight files are upstream's own,
unmodified — parameter names in the rewrite match upstream's exactly, and both load
with zero missing/unexpected keys.

Because this is a rewrite rather than a copy, there is no upstream artifact to diff
against for correctness. Verified instead against chemical plausibility (nicotine's
pyrrolidine N predicts pKa 8.42 against an experimental 8.02; `tests/test_pka.py`
pins known amine/acid examples) — a different and weaker guarantee than
`cyp_challenge_tutorial/`'s bit-for-bit match to the scoring backend. Do not add
another vendor entry this way without a comparably strong verification step.

`ionization_sites.py` (candidate site detection via `smarts_pattern.tsv`, upstream's
own 143-pattern table) and `descriptor.py` (atom featurisation) are copied near-verbatim,
fixing one bug found in the original: `ionization_group.py` resolved its SMARTS file
path from the process's working directory rather than the module's own location
(upstream issue #9), which only worked when a script happened to run from
`./MolGpKa/src`. Fixed here to resolve relative to `__file__`.

Not vendored: the training pipeline, `protonate.py` (upstream issue #20 — raises
`OverflowError` on essentially every carboxylic acid against a recent RDKit; not
needed for per-atom pKa prediction), `GATNet`/`MPNNNet` (unused by inference;
`MPNNNet` does not run in the original repo either — it references undefined names),
and the ~193MB of training data and benchmark sets.
