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
