"""Build the TDI submission from notebook 23's six-arm blend.

Run as a script, not a notebook:

    CYP_CHEMPROP_DEVICE=cpu uv run python notebooks/build_tdi_encoders_submission.py

## What this ships

TDI: a Caruana blend of six arms, thresholded at 0.25 (`04`, `13`, `14` and `23` all fix it):

- `tabicl_singletask`  TabICL on frozen CheMeleon (`04`'s incumbent)       5 seeds
- `tabicl_monroe`      TabICL on frozen Monroe embeddings (`23`)             5 seeds
- `tabicl_atom_jepa`   TabICL on frozen Atom-JEPA embeddings (`23`)          5 seeds
- `chemprop_scratch`, `chemprop_pubchem`, `chemprop_pubchem_frozen` (`13`)   3 seeds each

Regression: `22`'s activity.csv, copied unchanged.

## Final fit, seed means

Every arm is fitted on 100% of the labelled rows. Seed means follow `19`/`19b`: TabICL is not
deterministic across seeds and Chemprop's torch/data seeds are only set where a script patches
them in. Weights come from `23`'s single-seed OOF refit on all 25 folds, so the blind members
are slightly smoother than the members the weights were fitted for (unmeasured). Per-seed blind
probabilities are cached under `experiments/24_tdi_seed_fits/` purely so a crash does not repeat
an hour of Chemprop; delete that folder to force a refit.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cyp  # noqa: E402,F401
from cyp import constants as C  # noqa: E402
from cyp import data, ensemble, graph_models, submission  # noqa: E402
from cyp.graph_models import ChempropMultitargetModel  # noqa: E402
from cyp.tabular_models import predict_subprocess  # noqa: E402
from sklearn.metrics import matthews_corrcoef  # noqa: E402

EXP = PROJECT_ROOT / "experiments"
FITS = EXP / "24_tdi_seed_fits"
SUB22 = PROJECT_ROOT / "submissions" / "22_pooled_seed_ensemble" / "20261023"
SUB17_TDI = PROJECT_ROOT / "submissions" / "17_regression_ensemble" / "20261020" / "tdi.csv"
THRESHOLD = 0.25
TABICL_SEEDS = (42, 0, 1, 2, 3)
CHEMPROP_SEEDS = (0, 1, 2)
CHEMPROP_KWARGS = {"epochs": 50, "pretrain_epochs": 30}
TABICL_ARMS = ("tabicl_singletask", "tabicl_monroe", "tabicl_atom_jepa")
CHEMPROP_ARMS = ("chemprop_scratch", "chemprop_pubchem", "chemprop_pubchem_frozen")
ARMS = (*TABICL_ARMS, *CHEMPROP_ARMS)


def _load_23():
    spec = importlib.util.spec_from_file_location("n23", PROJECT_ROOT / "notebooks" / "23_tdi_encoders.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _seeded_cli(seed: int):
    original = graph_models._run_chemprop_cli

    def run(args: list[str]) -> None:
        if args and args[0] == "train":
            args = [*args, "--pytorch-seed", str(seed), "--data-seed", str(seed)]
        original(args)

    return original, run


def main() -> None:
    FITS.mkdir(parents=True, exist_ok=True)
    n23 = _load_23()
    test = data.load_test()
    test_smiles = test["SMILES"].to_list()
    frames = {f"{iso}_is_TDI": data.tdi_training_frame(iso) for iso in C.TDI_ISOFORMS}
    endpoints = list(frames)
    print(f"Fitting on all labelled data; predicting {len(test_smiles)} blind compounds.", flush=True)

    # ── TabICL arms: features ───────────────────────────────────────────────────
    train_feats = n23.tdi_features(frames)  # monroe / atom_jepa, identical to the CV features
    chem_cache = EXP / "04_methods_tdi" / "cache"
    train_feats["tabicl_singletask"] = {
        e: np.load(chem_cache / f"chemeleon_tdi_{e.split('_')[0]}.npy") for e in endpoints
    }
    blind_dir = EXP / "17_regression_ensemble" / "blind_embeddings"
    blind_feats = {
        "tabicl_singletask": np.load(EXP / "14_tdi_ensemble" / "chemeleon_test.npy"),
        "tabicl_monroe": np.load(blind_dir / "monroe.npy"),
        "tabicl_atom_jepa": np.load(blind_dir / "atom_jepa.npy")[:, -1, :],
    }
    for arm, x in blind_feats.items():
        assert x.shape[0] == len(test_smiles), (arm, x.shape)
    for arm in TABICL_ARMS:
        for e in endpoints:
            assert train_feats[arm][e].shape[0] == frames[e].height, (arm, e)

    probs: dict[str, dict[str, np.ndarray]] = {a: {} for a in ARMS}
    for arm in TABICL_ARMS:
        for e in endpoints:
            per_seed = []
            for s in TABICL_SEEDS:
                path = FITS / f"{arm}_{e}_{s}.npy"
                if not path.exists():
                    p, _, _ = predict_subprocess(
                        train_feats[arm][e],
                        frames[e]["y_true"].to_numpy(),
                        blind_feats[arm],
                        kind="tabicl",
                        task="classification",
                        seed=s,
                    )
                    p = np.asarray(p, dtype=float)
                    assert not np.isnan(p).any(), f"{arm}/{e}/{s}: NaN (numpy>=2.4? see CLAUDE.md)"
                    np.save(path, p)
                per_seed.append(np.load(path))
            probs[arm][e] = np.mean(per_seed, axis=0)
            print(f"{arm} {e}: mean p {probs[arm][e].mean():.3f}", flush=True)

    # ── Chemprop arms ───────────────────────────────────────────────────────────
    wide = None
    for e in endpoints:
        block = frames[e].select("Molecule_Name", "SMILES", pl.col("y_true").cast(pl.Float64).alias(e))
        wide = block if wide is None else wide.join(block, on=["Molecule_Name", "SMILES"], how="full", coalesce=True)
    wide = wide.sort("Molecule_Name")
    order = np.random.default_rng(42).permutation(wide.height)
    n_val = max(1, int(0.1 * wide.height))
    val_wide, fit_wide = wide[order[:n_val]], wide[order[n_val:]]

    for arm in CHEMPROP_ARMS:
        pretrain_dir = None if arm == "chemprop_scratch" else C.PROJECT_ROOT / ".chemprop_pretrain" / arm
        if pretrain_dir is not None and not (pretrain_dir / "model_0" / "best.pt").exists():
            raise FileNotFoundError(f"{arm}: no pretrain checkpoint at {pretrain_dir}; run notebook 13 first")
        per_seed = []
        for s in CHEMPROP_SEEDS:
            path = FITS / f"{arm}_seed{s}.npy"
            if not path.exists():
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
                original, patched = _seeded_cli(s)
                graph_models._run_chemprop_cli = patched
                try:
                    model.fit(
                        fit_wide["SMILES"].to_list(),
                        fit_wide.select(endpoints).to_numpy(),
                        smiles_val=val_wide["SMILES"].to_list(),
                        y_val=val_wide.select(endpoints).to_numpy(),
                    )
                finally:
                    graph_models._run_chemprop_cli = original
                pred = np.asarray(model.predict(test_smiles), dtype=float)
                assert not np.isnan(pred).any(), f"{arm}/{s}: NaN predictions"
                assert pred.min() >= 0.0 and pred.max() <= 1.0, (
                    f"{arm}/{s}: outside [0,1] -- a regression pretrain checkpoint leaks its UnscaleTransform"
                )
                np.save(path, pred)
            per_seed.append(np.load(path))
            print(f"{arm} seed {s} done", flush=True)
        mean = np.mean(per_seed, axis=0)
        for i, e in enumerate(endpoints):
            probs[arm][e] = mean[:, i]
            print(f"{arm} {e}: mean p {mean[:, i].mean():.3f}", flush=True)

    # ── Weights: Caruana on all 25 folds of 23's OOF ───────────────────────────
    oof = pl.read_parquet(EXP / "23_tdi_encoders" / "oof.parquet")
    held = pl.read_csv(EXP / "23_tdi_encoders" / "caruana_weights_P1.csv")
    keys = ["fold", "Molecule_Name"]
    weights: dict[str, dict[str, float]] = {}
    print("\nCaruana weights (all 25 folds) vs 23's held-out mean:")
    for e in endpoints:
        s = oof.filter(pl.col("endpoint") == e)
        cands = {a: s.filter(pl.col("method") == a).sort(keys)["y_prob"].to_numpy() for a in ARMS}
        y = s.filter(pl.col("method") == ARMS[0]).sort(keys)["y_true"].to_numpy()
        w = ensemble.caruana_selection(
            cands, y, lambda t, p: float(matthews_corrcoef(t, p >= THRESHOLD)), maximize=True, seed=0
        )
        weights[e] = w
        hm = held.filter(pl.col("endpoint") == e).group_by("model").agg(pl.col("weight").mean())
        hd = dict(zip(hm["model"], hm["weight"], strict=True))
        print(f"  {e}: " + "  ".join(f"{a} {w[a]:.3f} ({hd.get(a, 0.0):.3f})" for a in ARMS))

    # ── Blend, threshold, write ─────────────────────────────────────────────────
    labels: dict[str, np.ndarray] = {}
    print("\nBlended blind-set predictions:")
    for e in endpoints:
        blended = sum(weights[e][a] * probs[a][e] for a in ARMS)
        lab = blended >= THRESHOLD
        labels[e] = lab
        print(f"  {e}: mean p {blended.mean():.3f}, {lab.sum()}/{len(lab)} positive ({100 * lab.mean():.1f}%)")
    dest = PROJECT_ROOT / "submissions" / "24_tdi_encoders" / date.today().strftime("%Y%m%d")
    if dest.exists():
        raise FileExistsError(f"{dest} exists; refusing to overwrite a submission folder")
    dest.mkdir(parents=True)
    tdi_path = submission.build_tdi_submission(
        {f"{e.split('_')[0]}_is_TDI": labels[e] for e in endpoints}, dest / "tdi.csv"
    )
    shutil.copy(SUB22 / "activity.csv", dest / "activity.csv")
    ok_t, ok_a = submission.check(tdi_path, track="tdi"), submission.check(dest / "activity.csv", track="activity")
    print(f"\nvalidation: tdi {'PASSED' if ok_t else 'FAILED'}, activity {'PASSED' if ok_a else 'FAILED'}")
    if not (ok_t and ok_a):
        raise SystemExit("validation failed; nothing should be submitted")

    # Agreement with the TDI file currently on the board (17's, = 14's ensemble)
    new = pl.read_csv(tdi_path).sort("Molecule_Name")
    old = pl.read_csv(SUB17_TDI).sort("Molecule_Name")
    agree = {c: float((new[c] == old[c]).mean()) for c in new.columns if c.endswith("_is_TDI")}
    pos_new = {c: float(new[c].cast(pl.Float64).mean()) for c in agree}
    pos_old = {c: float(old[c].cast(pl.Float64).mean()) for c in agree}
    print("label agreement with 17's TDI file:", agree)

    pl.DataFrame(
        [{"endpoint": e, **{a: round(weights[e][a], 4) for a in ARMS}} for e in endpoints]
    ).write_csv(dest / "weights.csv")
    summary = pl.read_csv(EXP / "23_tdi_encoders" / "summary.csv")
    keep = summary.filter(pl.col("method").is_in(["caruana_P1", "caruana_P0", "tabicl_singletask"]))
    keep.write_csv(dest / "expected_performance.csv")
    (dest / "PROVENANCE.md").write_text(_provenance(weights, keep, agree, pos_new, pos_old))
    print(f"\nWrote {dest}")


def _provenance(weights, expected: pl.DataFrame, agree, pos_new, pos_old) -> str:
    def md(frame: pl.DataFrame) -> str:
        cols = frame.columns
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        lines += ["| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |" for row in frame.iter_rows()]
        return "\n".join(lines)

    w = pl.DataFrame([{"endpoint": e.split("_")[0], **{a: round(weights[e][a], 3) for a in ARMS}} for e in weights])
    pos = pl.DataFrame(
        [{"isoform": c.split("_")[0], "positive_rate_new": round(pos_new[c], 3),
          "positive_rate_17": round(pos_old[c], 3), "label_agreement_with_17": round(agree[c], 3)} for c in agree]
    )
    return f"""# TDI: six-arm encoder blend + 22's regression

- Generated: {date.today().isoformat()}
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

{md(w)}

## What changed against 17's TDI file (= 14's ensemble)

{md(pos)}

## Expected performance (CV, macro MCC, weights fitted off the held-out repeat)

{md(expected)}

`caruana_P1` is this submission's blend, `caruana_P0` is 14's. The gain is +0.028 (CI 0.019 to 0.037,
Holm-significant) and survives dropping the 225 compounds that appear in Monroe's PCBA pretraining
corpus (+0.027). Two Atom-JEPA conformer failures were imputed with the mean embedding.

**Do not quote a board score from this.** 14's blend scored 0.265 in CV and 0.2028 on the live board,
so a threshold-tuned OOF figure is optimistic, and the live half is small. Seed-averaging is not in
the CV table, and the weights were fitted on single-seed members.
"""


if __name__ == "__main__":
    main()
