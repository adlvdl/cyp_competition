"""Build the TDI ensemble submission from notebook 14's result.

Run as a script, not a notebook:

    CYP_CHEMPROP_DEVICE=cpu uv run python notebooks/build_tdi_ensemble_submission.py

## What this does

Fits each of `13`'s four arms on **100% of the labelled data** (CV holds out folds;
a submission must not), predicts the 750 blind compounds, blends them with the
Caruana weights refit on the full OOF, and thresholds at 0.25.

## Deliberately not cached

CLAUDE.md's rule: cache CV, never the final fit. A stale final-fit cache silently
disagrees with a retrained CV cache -- a real bug caught in this repo, where a
submission shipped calibration parameters from before a CV retrain. One fit is
cheap next to 25 folds.

## The weights are refit here, on all 25 folds

Notebook 14 fits weights on four outer repeats and scores on the fifth, rotated,
because a score has to be out-of-sample with respect to its own weights. That is
the right protocol for *measuring* the ensemble and the wrong one for *shipping*
it: the shipped weights should use every fold, exactly as the shipped models use
every compound. The 14 protocol already established the weights are stable across
held-out repeats (tabicl 0.49-0.61 across all ten), so refitting on the union is
not expected to move them much -- and the script prints both so the difference is
visible rather than assumed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cyp import constants as C  # noqa: E402
from cyp import data, ensemble, submission  # noqa: E402
from cyp.graph_models import ChempropMultitargetModel  # noqa: E402
from cyp.tabular_models import predict_subprocess  # noqa: E402

#: Fixed at `04` and `13`'s optimum. Not re-tuned here: the ensemble must be
#: compared against its members at the same operating point, and re-tuning on the
#: full OOF after choosing the ensemble on it would compound the optimism.
THRESHOLD = 0.25

#: Same settings every arm was measured with in `13`. An arm shipped with different
#: settings than it was validated with is not the arm that was validated.
CHEMPROP_KWARGS = {"epochs": 50, "pretrain_epochs": 30}

ARMS = (
    "tabicl_singletask",
    "chemprop_scratch",
    "chemprop_pubchem",
    "chemprop_pubchem_frozen",
)


def _chemeleon_features(frames: dict[str, pl.DataFrame], test_smiles: list[str]):
    """Frozen CheMeleon embeddings for the TabICL arm, train and test.

    `04`'s cached train embeddings are reused (row order verified identical, both
    come from `data.tdi_training_frame`); the test embeddings are computed here,
    since no previous notebook embedded the blind set for TDI.
    """
    from cyp.graph_models import chemeleon_embed

    cache = PROJECT_ROOT / "experiments" / "04_methods_tdi" / "cache"
    test_cache = PROJECT_ROOT / "experiments" / "14_tdi_ensemble" / "chemeleon_test.npy"
    test_cache.parent.mkdir(parents=True, exist_ok=True)

    train = {}
    for endpoint, frame in frames.items():
        path = cache / f"chemeleon_tdi_{endpoint.split('_')[0]}.npy"
        embedding = np.load(path)
        assert embedding.shape[0] == frame.height, (
            f"{endpoint}: {embedding.shape[0]} embedding rows vs {frame.height} frame rows"
        )
        train[endpoint] = embedding

    # The blind-set embedding is an input, not a fit, so caching it is safe -- it
    # depends only on the test SMILES, which never change within a snapshot.
    if test_cache.exists():
        test = np.load(test_cache)
        assert test.shape[0] == len(test_smiles)
    else:
        test, _ = chemeleon_embed(test_smiles, test_smiles[:1])
        np.save(test_cache, test)
    return train, test


def main() -> None:
    test = data.load_test()
    test_smiles = test["SMILES"].to_list()
    frames = {f"{iso}_is_TDI": data.tdi_training_frame(iso) for iso in C.TDI_ISOFORMS}
    endpoints = list(frames)

    print(f"Fitting on all labelled data; predicting {len(test_smiles)} blind compounds.")
    for endpoint, frame in frames.items():
        print(f"  {endpoint}: {frame.height} rows, {int(frame['y_true'].sum())} positive")

    # ── Arm 1: TabICL on frozen CheMeleon, one model per isoform ────────────────
    chemeleon_train, chemeleon_test = _chemeleon_features(frames, test_smiles)
    probabilities: dict[str, dict[str, np.ndarray]] = {arm: {} for arm in ARMS}

    print("\n[1/4] tabicl_singletask")
    for endpoint, frame in frames.items():
        probability, _, _ = predict_subprocess(
            chemeleon_train[endpoint],
            frame["y_true"].to_numpy(),
            chemeleon_test,
            kind="tabicl",
            task="classification",
            seed=42,
        )
        assert not np.isnan(probability).any(), (
            f"{endpoint}: TabICL returned NaN -- check numpy>=2.4 (see CLAUDE.md)"
        )
        probabilities["tabicl_singletask"][endpoint] = probability
        print(f"  {endpoint}: mean p {probability.mean():.3f}")

    # ── Arms 2-4: multitarget chemprop, from scratch and from each pretrain ─────
    wide = None
    for endpoint in endpoints:
        block = frames[endpoint].select(
            "Molecule_Name", "SMILES", pl.col("y_true").cast(pl.Float64).alias(endpoint)
        )
        wide = (
            block
            if wide is None
            else wide.join(block, on=["Molecule_Name", "SMILES"], how="full", coalesce=True)
        )
    wide = wide.sort("Molecule_Name")

    rng = np.random.default_rng(42)
    order = rng.permutation(wide.height)
    n_val = max(1, int(0.1 * wide.height))
    val_wide, fit_wide = wide[order[:n_val]], wide[order[n_val:]]

    for index, arm in enumerate(ARMS[1:], start=2):
        print(f"\n[{index}/4] {arm}")
        pretrain_dir = (
            None if arm == "chemprop_scratch" else C.PROJECT_ROOT / ".chemprop_pretrain" / arm
        )
        if pretrain_dir is not None and not (pretrain_dir / "model_0" / "best.pt").exists():
            raise FileNotFoundError(
                f"{arm}: no pretrain checkpoint at {pretrain_dir}. Run notebook 13 first "
                "-- the submission must ship the arm that was validated."
            )

        kwargs = dict(CHEMPROP_KWARGS)
        if pretrain_dir is None:
            kwargs.pop("pretrain_epochs")
        model = ChempropMultitargetModel(
            targets=endpoints,
            pred_type="classification",
            pretrain_dir=pretrain_dir,
            freeze_encoder=(arm == "chemprop_pubchem_frozen"),
            **kwargs,
        )
        model.fit(
            fit_wide["SMILES"].to_list(),
            fit_wide.select(endpoints).to_numpy(),
            smiles_val=val_wide["SMILES"].to_list(),
            y_val=val_wide.select(endpoints).to_numpy(),
        )
        predicted = model.predict(test_smiles)
        for position, endpoint in enumerate(endpoints):
            column = np.asarray(predicted[:, position], dtype=float)
            assert not np.isnan(column).any(), f"{arm}/{endpoint}: NaN predictions"
            assert 0.0 <= column.min() and column.max() <= 1.0, (
                f"{arm}/{endpoint}: predictions outside [0,1] "
                f"({column.min():.3f}-{column.max():.3f}) -- a regression pretrain "
                "checkpoint leaks its UnscaleTransform (see CLAUDE.md)"
            )
            probabilities[arm][endpoint] = column
            print(f"  {endpoint}: mean p {column.mean():.3f}")

    # ── Weights: refit on the full OOF ──────────────────────────────────────────
    from sklearn.metrics import matthews_corrcoef

    oof = pl.read_parquet(PROJECT_ROOT / "experiments" / "13_tdi_pretrained" / "oof.parquet")
    held_out = pl.read_csv(PROJECT_ROOT / "experiments" / "14_tdi_ensemble" / "caruana_weights.csv")

    print("\nCaruana weights (refit on all 25 folds), against 14's held-out mean:")
    weights: dict[str, dict[str, float]] = {}
    for endpoint in endpoints:
        slice_ = oof.filter(pl.col("endpoint") == endpoint)
        candidates = {
            arm: slice_.filter(pl.col("method") == arm)
            .sort(["fold", "Molecule_Name"])["y_prob"]
            .to_numpy()
            for arm in ARMS
        }
        truth = (
            slice_.filter(pl.col("method") == ARMS[0])
            .sort(["fold", "Molecule_Name"])["y_true"]
            .to_numpy()
        )
        fitted = ensemble.caruana_selection(
            candidates,
            truth,
            lambda t, p: float(matthews_corrcoef(t, p >= THRESHOLD)),
            maximize=True,
            seed=0,
        )
        weights[endpoint] = fitted
        previous = held_out.filter(pl.col("endpoint") == endpoint).select(ARMS).mean().row(0)
        print(f"  {endpoint}")
        for arm, mean_weight in zip(ARMS, previous, strict=True):
            print(f"    {arm:26} full {fitted[arm]:.3f}   14 held-out mean {mean_weight:.3f}")

    # ── Blend, threshold, write ─────────────────────────────────────────────────
    predictions = {}
    print("\nBlended blind-set predictions:")
    for endpoint in endpoints:
        blended = sum(weights[endpoint][arm] * probabilities[arm][endpoint] for arm in ARMS)
        labels = blended >= THRESHOLD
        predictions[f"{endpoint.split('_')[0]}_is_TDI"] = labels
        print(
            f"  {endpoint}: mean p {blended.mean():.3f}, "
            f"{labels.sum()}/{len(labels)} positive ({100 * labels.mean():.1f}%)"
        )

    out_dir = PROJECT_ROOT / "submissions" / "14_tdi_ensemble" / "20260920"
    path = submission.build_tdi_submission(predictions, out_dir / "tdi.csv")
    ok = submission.check(path, track="tdi")
    print(f"\nWrote {path.relative_to(PROJECT_ROOT)} -- validation {'PASSED' if ok else 'FAILED'}")

    pl.DataFrame([{"endpoint": e, **weights[e]} for e in endpoints]).write_csv(
        out_dir / "weights.csv"
    )


if __name__ == "__main__":
    main()
