"""Chemprop D-MPNN and the CheMeleon foundation backbone.

Ported from the PXR repo's `marimo_notebooks/4_ml_optimization_2.py`, where these
lived as notebook cells; they are modules here because CLAUDE.md's conventions put
anything reused across notebooks in `src/cyp`. PXR's retrospective ranks CheMeleon as
the single best-performing model family it tried -- it won every CV comparison
outright -- which is why this is the highest-expected-value item in PLAN.md.

## Why the CLI, not the Python API

`train()` and `predict()` shell out to the `chemprop` binary rather than calling the
Python API in-process. This is not stylistic. Running many CV folds through the
Python API inside one kernel exhausts the MPS allocator on Apple Silicon and stalls
between folds (PXR lost real time to this, documented in its "Stalling the M4 engine"
post). A subprocess gets a fresh allocator per fold, and `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`
removes the ceiling that triggers macOS memory-pressure stalls.

The same reasoning drives `chemeleon_embed` and `device`: PyTorch ships its own
OpenMP runtime, and once torch is imported into a process a later LightGBM fit
**segfaults** on macOS -- a hard crash, not an exception. Since a comparison notebook
fits both, nothing in this module may import torch into the parent. Every torch touch
here happens in a child process, `device()` included.

## Interface

`run_cv` in `models.py` featurizes once up front and calls `fit(X, y)` on a matrix.
Graph models consume SMILES instead, so `SmilesModelMixin` marks the classes that
need raw SMILES and `models.run_cv` checks for it -- see `models._needs_smiles`.
Chemprop also wants an explicit validation set for early stopping, which is what
`cv.scaffold_splits(..., p_val=0.1)` exists to supply.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl


class SmilesModelMixin:
    """Marker: this model's `fit`/`predict` take SMILES strings, not a feature matrix.

    `models.run_cv` checks for this to decide whether to hand a fold its fingerprint
    rows or its SMILES column. A marker class rather than a constructor flag because
    the distinction is a fixed property of the model type, not a per-instance choice.
    """

    requires_smiles: bool = True


#: Cached device string, so the detection subprocess runs at most once per session.
_DEVICE: str | None = None

#: Device detection, run in a child process. Importing torch into the *parent* is
#: precisely what must not happen -- see the module docstring's OpenMP note -- and
#: this function is called on every Chemprop fit, so doing it inline would poison
#: any notebook that also fits LightGBM.
_DEVICE_SCRIPT = (
    "import torch;"
    "print('cuda' if torch.cuda.is_available() else "
    "('mps' if torch.backends.mps.is_available() else 'cpu'))"
)


def device() -> str:
    """Best available accelerator for PyTorch: cuda, then mps, then cpu.

    Detected in a subprocess and cached. This looks like overkill for a two-line
    check, but `import torch` in this process would make every later LightGBM fit
    segfault (module docstring), and this function runs on every Chemprop fit.
    Falls back to "cpu" if detection fails -- a slow run beats a crashed one.
    """
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    try:
        result = subprocess.run(
            [sys.executable, "-c", _DEVICE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=120,
        )
        _DEVICE = result.stdout.strip() if result.returncode == 0 else "cpu"
    except (subprocess.SubprocessError, OSError):
        _DEVICE = "cpu"
    return _DEVICE or "cpu"


# ── Chemprop CLI plumbing ───────────────────────────────────────────────────────

#: Resolve the chemprop binary from the same venv as the running interpreter, so a
#: notebook launched via `uv run` gets that environment's chemprop and not a stray
#: one earlier on PATH.
CHEMPROP_BIN = Path(sys.executable).parent / "chemprop"

#: All CLI output is appended here rather than printed: chemprop is extremely
#: verbose and 25 folds of it would bury a notebook.
CHEMPROP_LOG = Path("logs/chemprop_cli.log")


def _write_smiles_csv(
    smiles: list[str],
    targets: np.ndarray | None,
    path: Path,
    target_col: str,
    sample_weight: np.ndarray | None = None,
) -> None:
    """Write a chemprop input CSV: a smiles column, an optional target column, and an
    optional per-datapoint `weight` column (chemprop's native loss weighting)."""
    columns: dict[str, list] = {"smiles": list(smiles)}
    if targets is not None:
        columns[target_col] = np.asarray(targets).flatten().tolist()
    if sample_weight is not None:
        columns["weight"] = np.asarray(sample_weight).flatten().tolist()
    pl.DataFrame(columns).write_csv(path)


def _run_chemprop_cli(args: list[str]) -> None:
    """Run the chemprop CLI, logging to file; raise with a log tail on failure."""
    CHEMPROP_LOG.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(CHEMPROP_BIN), *args]
    # Removing the MPS allocator ceiling prevents the macOS memory-pressure stalls
    # that appear between folds on Apple Silicon.
    env = {**os.environ, "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0"}
    with open(CHEMPROP_LOG, "a") as log:
        log.write(f"\n{'=' * 60}\nCMD: {' '.join(cmd)}\n{'=' * 60}\n")
        result = subprocess.run(cmd, stdout=log, stderr=log, text=True, env=env)
    if result.returncode != 0:
        tail = "\n".join(CHEMPROP_LOG.read_text().splitlines()[-30:])
        raise RuntimeError(
            f"chemprop CLI failed (exit {result.returncode}). Log: {CHEMPROP_LOG}\n{tail}"
        )


class ChempropModel(SmilesModelMixin):
    """Chemprop v2 D-MPNN trained from scratch, via the CLI.

    Checkpoints go to a fixed directory that is cleared on every `fit` call, so disk
    does not accumulate across 25 folds.

    Transfer learning: call `pretrain` once on an auxiliary dataset before the CV
    loop; `fit` then detects the checkpoint and passes `--checkpoint` to warm-start
    the encoder. `freeze_encoder=True` additionally locks the message-passing weights
    so only the FFN head moves. This is the hook for PLAN.md's single-concentration
    screen experiment -- PXR found auxiliary-assay multitask training *hurt*, so that
    needs to be run as a genuine experiment against an unpretrained control, not
    assumed to help.
    """

    def __init__(
        self,
        pred_type: str = "regression",
        model_dir: Path | None = None,
        pretrain_dir: Path | None = None,
        freeze_encoder: bool = False,
        from_foundation: str | None = None,
        epochs: int = 50,
        pretrain_epochs: int = 50,
        message_hidden_dim: int = 300,
        depth: int = 3,
        dropout: float = 0.0,
        ffn_hidden_dim: int = 300,
        ffn_num_layers: int = 2,
        batch_size: int = 64,
        init_lr: float = 1e-4,
        max_lr: float = 1e-3,
        final_lr: float = 1e-4,
    ) -> None:
        """
        Args:
            pred_type: "regression" (direct inhibition) or "classification" (TDI).
            model_dir: Where fine-tuning checkpoints are written; cleared each `fit`.
            pretrain_dir: Where `pretrain` checkpoints persist. *Not* cleared between
                folds, so one pretrained encoder is reused across the whole CV run.
            freeze_encoder: Freeze message-passing weights during fine-tuning.
                Ignored when no pretrain checkpoint exists.
            from_foundation: Foundation backbone to warm-start from, e.g. "CHEMELEON".
                The CLI downloads and caches weights to ~/.chemprop on first use.
            epochs: Max training epochs for `fit`.
            pretrain_epochs: Max training epochs for `pretrain`.
            message_hidden_dim: MPNN message-passing hidden dimension.
            depth: Number of message-passing steps.
            dropout: Dropout after each message-passing and FFN layer.
            ffn_hidden_dim: Feed-forward head hidden dimension.
            ffn_num_layers: Number of FFN layers.
            batch_size: Mini-batch size.
            init_lr: Initial LR for the one-cycle schedule.
            max_lr: Peak LR for the one-cycle schedule.
            final_lr: Final LR for the one-cycle schedule.
        """
        if pred_type not in ("regression", "classification"):
            raise ValueError("pred_type must be 'regression' or 'classification'")
        tmp = Path(tempfile.gettempdir())
        # Default directories are namespaced by backbone so a scratch run and a
        # CheMeleon run in the same session cannot overwrite each other's checkpoints.
        suffix = (from_foundation or "scratch").lower()
        self.pred_type = pred_type
        self.model_dir = model_dir or tmp / f"chemprop_{suffix}_model"
        self.pretrain_dir = pretrain_dir or tmp / f"chemprop_{suffix}_pretrain"
        self.freeze_encoder = freeze_encoder
        self.from_foundation = from_foundation
        self.epochs = epochs
        self.pretrain_epochs = pretrain_epochs
        self.message_hidden_dim = message_hidden_dim
        self.depth = depth
        self.dropout = dropout
        self.ffn_hidden_dim = ffn_hidden_dim
        self.ffn_num_layers = ffn_num_layers
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.target_col: str = "target"

    def _base_train_args(self, target_col: str) -> list[str]:
        """CLI args shared between `pretrain` and `fit`."""
        args = [
            "--smiles-columns",
            "smiles",
            "--target-columns",
            target_col,
            "--task-type",
            self.pred_type,
            "--accelerator",
            device(),
            "--message-hidden-dim",
            str(self.message_hidden_dim),
            "--depth",
            str(self.depth),
            "--dropout",
            str(self.dropout),
            "--ffn-hidden-dim",
            str(self.ffn_hidden_dim),
            "--ffn-num-layers",
            str(self.ffn_num_layers),
            "--batch-size",
            str(self.batch_size),
            "--init-lr",
            str(self.init_lr),
            "--max-lr",
            str(self.max_lr),
            "--final-lr",
            str(self.final_lr),
        ]
        if self.from_foundation:
            args += ["--from-foundation", self.from_foundation]
        return args

    def pretrain(
        self,
        smiles_train: list[str],
        y_train: np.ndarray,
        smiles_val: list[str],
        y_val: np.ndarray,
        target_col: str = "pretrain_target",
    ) -> ChempropModel:
        """Pretrain on an auxiliary dataset, saving a checkpoint `fit` warm-starts from.

        Call once *before* the CV loop: `pretrain_dir` is not cleared between folds,
        so every fold fine-tunes from the same encoder. A second `pretrain` call
        overwrites the previous checkpoint.
        """
        tmp = Path(tempfile.gettempdir())
        train_csv, val_csv = tmp / "cyp_cp_pre_train.csv", tmp / "cyp_cp_pre_val.csv"
        _write_smiles_csv(smiles_train, y_train, train_csv, target_col)
        _write_smiles_csv(smiles_val, y_val, val_csv, target_col)
        if self.pretrain_dir.exists():
            shutil.rmtree(self.pretrain_dir)
        _run_chemprop_cli(
            [
                "train",
                "--data-path",
                str(train_csv),
                str(val_csv),
                str(val_csv),
                *self._base_train_args(target_col),
                "--epochs",
                str(self.pretrain_epochs),
                "--save-dir",
                str(self.pretrain_dir),
            ]
        )
        train_csv.unlink(missing_ok=True)
        val_csv.unlink(missing_ok=True)
        return self

    def fit(
        self,
        smiles_train: list[str],
        y_train: np.ndarray,
        smiles_val: list[str] | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        target_col: str = "target",
    ) -> ChempropModel:
        """Train (or fine-tune) on the target dataset.

        Chemprop's CLI requires a validation path for early stopping. When no
        validation set is passed, a random 10% of the training rows is held out --
        acceptable for the final fit-on-everything, but for CV you should pass the
        `val` frame that `cv.scaffold_splits(..., p_val=0.1)` yields, so the early-
        stopping set respects the scaffold grouping like the test fold does.
        """
        self.target_col = target_col
        smiles_train = list(smiles_train)
        y_train = np.asarray(y_train).flatten()

        if smiles_val is None or y_val is None:
            rng = np.random.default_rng(42)
            idx = rng.permutation(len(smiles_train))
            n_val = max(1, int(0.1 * len(idx)))
            val_idx, train_idx = idx[:n_val], idx[n_val:]
            smiles_val = [smiles_train[i] for i in val_idx]
            y_val = y_train[val_idx]
            if sample_weight is not None:
                sample_weight = np.asarray(sample_weight)[train_idx]
            smiles_train = [smiles_train[i] for i in train_idx]
            y_train = y_train[train_idx]

        tmp = Path(tempfile.gettempdir())
        train_csv, val_csv = tmp / "cyp_cp_train.csv", tmp / "cyp_cp_val.csv"
        # Chemprop needs the weight column present in every split when it is used at
        # all, so the validation rows get a uniform weight of 1.
        val_w = None if sample_weight is None else np.ones(len(smiles_val))
        _write_smiles_csv(smiles_train, y_train, train_csv, target_col, sample_weight)
        _write_smiles_csv(smiles_val, np.asarray(y_val), val_csv, target_col, val_w)

        if self.model_dir.exists():
            shutil.rmtree(self.model_dir)

        args = [
            "train",
            "--data-path",
            str(train_csv),
            str(val_csv),
            str(val_csv),
            *self._base_train_args(target_col),
            "--epochs",
            str(self.epochs),
            "--save-dir",
            str(self.model_dir),
        ]
        if sample_weight is not None:
            args += ["-w", "weight"]

        # An explicit pretrain checkpoint takes precedence over --from-foundation:
        # warm-starting from our own auxiliary run is a stronger prior than the
        # generic backbone when both are available.
        pretrain_ckpt = self.pretrain_dir / "model_0" / "best.pt"
        if pretrain_ckpt.exists():
            args += ["--checkpoint", str(pretrain_ckpt)]
            if self.freeze_encoder:
                args.append("--freeze-encoder")

        _run_chemprop_cli(args)
        train_csv.unlink(missing_ok=True)
        val_csv.unlink(missing_ok=True)
        return self

    def predict(self, smiles_test: list[str]) -> np.ndarray:
        """Run inference via `chemprop predict`. Returns a 1-D array.

        For `pred_type="classification"` these are positive-class probabilities, not
        labels -- thresholding is the caller's job, and for TDI the threshold should
        be tuned for MCC rather than left at 0.5 (see PLAN.md item 5).
        """
        tmp = Path(tempfile.gettempdir())
        test_csv, pred_csv = tmp / "cyp_cp_test.csv", tmp / "cyp_cp_preds.csv"
        model_pt = self.model_dir / "model_0" / "best.pt"
        _write_smiles_csv(list(smiles_test), None, test_csv, self.target_col)
        _run_chemprop_cli(
            [
                "predict",
                "--test-path",
                str(test_csv),
                "--model-path",
                str(model_pt),
                "--preds-path",
                str(pred_csv),
            ]
        )
        preds = pl.read_csv(pred_csv)[self.target_col].to_numpy()
        test_csv.unlink(missing_ok=True)
        pred_csv.unlink(missing_ok=True)
        return np.asarray(preds, dtype=float).flatten()


class ChempropChemeleonModel(ChempropModel):
    """Chemprop D-MPNN fine-tuned from the CheMeleon foundation backbone.

    Identical to `ChempropModel` but always passes `--from-foundation CHEMELEON`, so
    the encoder starts from CheMeleon's pretrained weights instead of random init.
    The CLI downloads and caches those weights to ~/.chemprop/chemeleon_mp.pt on
    first use. This is the model that won every PXR CV comparison.
    """

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("from_foundation", "CHEMELEON")
        super().__init__(**kwargs)


# ── CheMeleon embeddings as a featurizer ────────────────────────────────────────
#
# Beyond fine-tuning, CheMeleon's frozen 2048-dim embedding is a *representation*:
# any downstream model can consume it in place of a fingerprint. That matters more
# here than it did in PXR, because the tabular-foundation-model literature
# (arXiv 2604.16123) finds TabPFN+CheMeleon substantially outperforms TabPFN+Morgan
# on molecular property prediction -- so this is the featurizer the TFM comparison
# in 03_methods.py should be run on, not ECFP.

_CHEMELEON_SCRIPT = "\n".join(
    [
        "import os, json, sys, numpy as np",
        # Two OpenMP runtimes (torch's libkmp and sklearn's) in one process abort on
        # macOS. This script only ever runs as a child process, so the override is
        # scoped to it and never touches the parent kernel.
        "os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'",
        "from pathlib import Path",
        "import torch",
        "from chemprop import featurizers",
        "from chemprop import nn as chemnn",
        "from chemprop.models import MPNN",
        "from chemprop.nn import RegressionFFN",
        "from chemprop.data import BatchMolGraph",
        "from rdkit.Chem import MolFromSmiles",
        "smiles_file, out_train, out_test = sys.argv[1], sys.argv[2], sys.argv[3]",
        "with open(smiles_file) as f:",
        "    data = json.load(f)",
        "mp_path = Path.home() / '.chemprop' / 'chemeleon_mp.pt'",
        "ckpt = torch.load(mp_path, weights_only=True)",
        "mp = chemnn.BondMessagePassing(**ckpt['hyper_parameters'])",
        "mp.load_state_dict(ckpt['state_dict'])",
        # The RegressionFFN head is never used -- `fingerprint()` reads out the encoder
        # before it -- but MPNN requires a head to construct.
        "model = MPNN(mp, chemnn.MeanAggregation(), RegressionFFN(input_dim=mp.output_dim))",
        "model.eval()",
        "feat = featurizers.SimpleMoleculeMolGraphFeaturizer()",
        "def embed(smiles):",
        "    bmg = BatchMolGraph([feat(MolFromSmiles(s)) for s in smiles])",
        "    with torch.no_grad():",
        "        return model.fingerprint(bmg).numpy(force=True)",
        "np.save(out_train, embed(data['train']))",
        "np.save(out_test,  embed(data['test']))",
    ]
)


def chemeleon_embed(
    smiles_train: list[str],
    smiles_test: list[str],
    prefix: str = "chemeleon",
) -> tuple[np.ndarray, np.ndarray]:
    """Frozen CheMeleon embeddings for two SMILES lists, via an isolated subprocess.

    Train and test are embedded in one call because the subprocess launch and the
    checkpoint load dominate the cost -- doing them separately roughly doubles it for
    no benefit. The embedding is frozen, so there is no leakage in embedding both at
    once: no label ever enters this computation.

    Requires the CheMeleon checkpoint at ~/.chemprop/chemeleon_mp.pt, which
    `ChempropChemeleonModel` downloads on its first run.

    Args:
        smiles_train: SMILES for the training set.
        smiles_test: SMILES for the test set.
        prefix: Temp-file prefix, so concurrent calls do not collide.

    Returns:
        `(X_train, X_test)`, each of shape (n, 2048).
    """
    tmp = Path(tempfile.gettempdir())
    script_path = tmp / "cyp_chemeleon_embed.py"
    script_path.write_text(_CHEMELEON_SCRIPT)

    smi_file = tmp / f"{prefix}_smiles.json"
    train_file, test_file = tmp / f"{prefix}_train", tmp / f"{prefix}_test"
    smi_file.write_text(json.dumps({"train": list(smiles_train), "test": list(smiles_test)}))

    result = subprocess.run(
        [sys.executable, str(script_path), str(smi_file), str(train_file), str(test_file)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"CheMeleon subprocess failed:\n{result.stderr}")

    X_train = np.load(f"{train_file}.npy")
    X_test = np.load(f"{test_file}.npy")
    for p in (smi_file, Path(f"{train_file}.npy"), Path(f"{test_file}.npy")):
        p.unlink(missing_ok=True)
    return X_train, X_test


class ChempropMultitargetModel(ChempropModel):
    """Chemprop D-MPNN with one output head per endpoint, trained on all at once.

    The multitask counterpart to `ChempropModel`. Chemprop masks missing targets in
    its loss, so the training CSV can carry an empty cell wherever a compound was
    never measured for an isoform -- no imputation, and no need to restrict training
    to the handful of compounds measured for everything (only 41 of 4,905 have all
    four). The message-passing encoder is shared across every endpoint while each
    head learns only from its own labels, which is the mechanism by which a weak
    endpoint might benefit from CYP3A4's much larger row count.

    Whether that actually helps is the open question -- PXR found multitask hurt --
    so this exists to be compared against the single-task arm on identical folds, not
    to replace it. See `multitask.run_cv_multitarget`.
    """

    def __init__(self, targets: list[str], **kwargs) -> None:
        """
        Args:
            targets: Endpoint names, in the column order used by `fit` and returned
                by `predict`. Stored so both agree without the caller re-specifying.
            **kwargs: Forwarded to `ChempropModel`.
        """
        if not targets:
            raise ValueError("targets must name at least one endpoint")
        super().__init__(**kwargs)
        self.targets = list(targets)

    def _base_train_args(self, target_col: str) -> list[str]:
        """Same as the single-target version but naming every target column.

        `target_col` is ignored: the column names come from `self.targets`, since a
        multi-target run has no single one.
        """
        args = [
            "--smiles-columns",
            "smiles",
            "--target-columns",
            *self.targets,
            "--task-type",
            self.pred_type,
            "--accelerator",
            device(),
            "--message-hidden-dim",
            str(self.message_hidden_dim),
            "--depth",
            str(self.depth),
            "--dropout",
            str(self.dropout),
            "--ffn-hidden-dim",
            str(self.ffn_hidden_dim),
            "--ffn-num-layers",
            str(self.ffn_num_layers),
            "--batch-size",
            str(self.batch_size),
            "--init-lr",
            str(self.init_lr),
            "--max-lr",
            str(self.max_lr),
            "--final-lr",
            str(self.final_lr),
        ]
        if self.from_foundation:
            args += ["--from-foundation", self.from_foundation]
        return args

    def _write_wide_csv(self, smiles: list[str], y: np.ndarray, path: Path) -> None:
        """Write a multi-target CSV, leaving unmeasured cells empty.

        Empty is deliberate and load-bearing: Chemprop reads a blank cell as "no
        label" and masks it out of the loss, whereas a 0.0 would be read as a real
        measurement of an extremely inactive compound and would poison the head.
        """
        columns: dict[str, list] = {"smiles": list(smiles)}
        y = np.asarray(y, dtype=float)
        for i, target in enumerate(self.targets):
            column = y[:, i]
            columns[target] = [None if np.isnan(v) else float(v) for v in column]
        pl.DataFrame(columns).write_csv(path)

    def fit(
        self,
        smiles_train: list[str],
        y_train: np.ndarray,
        smiles_val: list[str] | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        target_col: str = "target",
    ) -> ChempropMultitargetModel:
        """Train on a `(n_compounds, n_targets)` matrix with NaN for unmeasured.

        `sample_weight` is not supported here: Chemprop's `-w` weights a datapoint,
        not a (datapoint, target) pair, so it cannot express per-endpoint weighting
        and would silently mean something different from the single-task case.
        """
        if sample_weight is not None:
            raise ValueError(
                "sample_weight is not supported for multi-target training: chemprop "
                "weights datapoints, not (datapoint, target) pairs."
            )

        y_train = np.asarray(y_train, dtype=float)
        if y_train.ndim != 2 or y_train.shape[1] != len(self.targets):
            raise ValueError(f"y_train must be (n, {len(self.targets)}), got {y_train.shape}")

        smiles_train = list(smiles_train)
        if smiles_val is None or y_val is None:
            rng = np.random.default_rng(42)
            idx = rng.permutation(len(smiles_train))
            n_val = max(1, int(0.1 * len(idx)))
            val_idx, train_idx = idx[:n_val], idx[n_val:]
            smiles_val = [smiles_train[i] for i in val_idx]
            y_val = y_train[val_idx]
            smiles_train = [smiles_train[i] for i in train_idx]
            y_train = y_train[train_idx]

        tmp = Path(tempfile.gettempdir())
        train_csv, val_csv = tmp / "cyp_cp_mt_train.csv", tmp / "cyp_cp_mt_val.csv"
        self._write_wide_csv(smiles_train, y_train, train_csv)
        self._write_wide_csv(smiles_val, np.asarray(y_val, dtype=float), val_csv)

        if self.model_dir.exists():
            shutil.rmtree(self.model_dir)

        _run_chemprop_cli(
            [
                "train",
                "--data-path",
                str(train_csv),
                str(val_csv),
                str(val_csv),
                *self._base_train_args(target_col),
                "--epochs",
                str(self.epochs),
                "--save-dir",
                str(self.model_dir),
            ]
        )
        train_csv.unlink(missing_ok=True)
        val_csv.unlink(missing_ok=True)
        return self

    def predict(self, smiles_test: list[str]) -> np.ndarray:
        """Predict every endpoint at once. Returns `(n_compounds, n_targets)` in
        `self.targets` order."""
        tmp = Path(tempfile.gettempdir())
        test_csv, pred_csv = tmp / "cyp_cp_mt_test.csv", tmp / "cyp_cp_mt_preds.csv"
        model_pt = self.model_dir / "model_0" / "best.pt"
        pl.DataFrame({"smiles": list(smiles_test)}).write_csv(test_csv)

        _run_chemprop_cli(
            [
                "predict",
                "--test-path",
                str(test_csv),
                "--model-path",
                str(model_pt),
                "--preds-path",
                str(pred_csv),
            ]
        )

        preds = pl.read_csv(pred_csv)
        missing = [t for t in self.targets if t not in preds.columns]
        if missing:
            raise RuntimeError(
                f"chemprop predictions missing target columns {missing}; got {preds.columns}"
            )
        out = preds.select(self.targets).to_numpy().astype(float)

        test_csv.unlink(missing_ok=True)
        pred_csv.unlink(missing_ok=True)
        return out
