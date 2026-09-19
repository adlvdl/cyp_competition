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
post). A subprocess gets a fresh allocator per fold, and the MPS watermark is capped
rather than removed -- see `_run_chemprop_cli` for why `0.0` was actively harmful
across a long sequence of fits.

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

    ## Overriding with CYP_CHEMPROP_DEVICE

    Set `CYP_CHEMPROP_DEVICE=cpu` to force the accelerator. This exists because MPS
    degrades badly across a long block of Chemprop fits on this machine, and it does
    so *progressively then permanently*: measured on the 04 TDI run (2026-09-14/15),
    `chemprop_singletask` went 100s, 91s, 97s per fold for three folds, then rose to
    1109s and 1791s and plateaued around 2000s for every fold after -- a ~20x
    penalty that never recovered. `chemprop_multitask` showed the same shape. Other
    methods in the same interleaved run (LightGBM, XGBoost, TabICL) stayed flat
    throughout, so this is specific to the sustained MPS workload rather than
    machine-wide contention.

    Interleaving methods within a fold did not prevent it, which rules out the
    simplest mitigation. Forcing CPU trades a higher fixed per-fit cost for a rate
    that does not drift, which is the better deal over 25 folds.
    """
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE

    override = os.environ.get("CYP_CHEMPROP_DEVICE", "").strip()
    if override:
        _DEVICE = override
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
    """Run the chemprop CLI, logging to file; raise with a log tail on failure.

    ## The MPS degradation this guards against

    In the 2026-09-12 full run, chemprop slowed monotonically across consecutive
    folds -- 0.55, 1.63, 7.79, 6.57 seconds per compound across four endpoints, a
    ~14x degradation on comparable work -- while TabICL and TabPFN, which also run
    torch per fold, showed no trend. The original mitigation set
    `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`, which *removes* the allocator ceiling
    entirely. That is the opposite of what is wanted across many sequential fits:
    with no watermark, MPS never reclaims its cache, so pressure accumulates from
    one fold to the next until the process is thrashing.

    `0.0` is therefore replaced by a real ratio, and `PYTORCH_MPS_ALLOCATOR_POLICY`
    asks the allocator to release unused blocks back to the system between runs.
    Each call is already its own process; what was missing was letting that process
    give memory back.
    """
    CHEMPROP_LOG.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(CHEMPROP_BIN), *args]
    env = {
        **os.environ,
        # A real ceiling, not 0.0 -- an unbounded cache is what let pressure build
        # across folds. Both watermarks are set explicitly: PyTorch derives the low
        # watermark as 2x the high one by default, so a high of 0.7 yields a low of
        # 1.4 and chemprop dies with "invalid low watermark ratio 1.4". Both must
        # be <= 1.0.
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.5",
        "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.4",
    }
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

    #: pred_type -> chemprop --task-type, for the uncertainty-bearing regression
    #: heads. "regression" stays plain regression unless an uncertainty method needs
    #: a different head shape (mve, evidential); ensemble and dropout both read out
    #: of an ordinary regression head, so they do not appear here.
    _UNCERTAINTY_TASK_TYPE = {
        "mve": "regression-mve",
        "evidential": "regression-evidential",
    }

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
        uncertainty_method: str | None = None,
        ensemble_size: int = 1,
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
            uncertainty_method: None (point predictions only), "mve", "evidential",
                "ensemble" or "dropout". "mve"/"evidential" change the FFN head and
                loss function (`_UNCERTAINTY_TASK_TYPE`) and must be requested at
                training time. "ensemble" requires `ensemble_size > 1` and reads
                disagreement among those checkpoints. "dropout" needs no training-time
                change -- it resamples an ordinary model with dropout left on at
                inference -- so it can be requested against a model trained with
                `uncertainty_method=None`, including an already-fitted checkpoint.
            ensemble_size: Number of checkpoints chemprop trains in one `train` call.
                Only meaningful (and only multiplies fit cost) when
                `uncertainty_method="ensemble"`; left at 1 otherwise.
        """
        if pred_type not in ("regression", "classification"):
            raise ValueError("pred_type must be 'regression' or 'classification'")
        if uncertainty_method not in (None, "mve", "evidential", "ensemble", "dropout"):
            raise ValueError(
                "uncertainty_method must be one of None, 'mve', 'evidential', "
                f"'ensemble', 'dropout' -- got {uncertainty_method!r}"
            )
        if uncertainty_method == "ensemble" and ensemble_size < 2:
            raise ValueError("uncertainty_method='ensemble' needs ensemble_size >= 2")
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
        self.uncertainty_method = uncertainty_method
        self.ensemble_size = ensemble_size
        self.target_col: str = "target"

    def _task_type(self) -> str:
        """`--task-type`, switched to the uncertainty-bearing head (mve/evidential)
        when `uncertainty_method` calls for one; plain otherwise. Shared by both the
        single- and multi-target `_base_train_args`, so the mapping only lives once."""
        uncertainty_task_type = self._UNCERTAINTY_TASK_TYPE.get(self.uncertainty_method)
        if self.pred_type == "regression" and uncertainty_task_type is not None:
            return uncertainty_task_type
        return self.pred_type

    def _checkpoint_args(self, pretrain_ckpt: Path) -> list[str]:
        """`--checkpoint`, repeated once per ensemble member when warm-starting an
        ensemble from a single pretrained checkpoint.

        Chemprop's `train_model` reads `--ensemble-size` from `len(--checkpoint
        paths)` whenever `--checkpoint` is given at all (`chemprop/cli/train.py`,
        `train_model`): passing the checkpoint once silently collapses
        `ensemble_size=4` down to 1, with only a log warning -- no error, so it is
        easy to lose an entire ensemble arm to a single warm-started member without
        noticing (this is exactly what happened while building notebook 06: an
        `ensemble` fit trained with `ensemble_size=1` under the hood, and the later
        `predict --uncertainty-method ensemble` call raised because chemprop refuses
        ensemble uncertainty from a single checkpoint). Passing the same path
        `ensemble_size` times gives every member the same warm start -- they still
        diverge from there via each member's own random seed and data shuffling,
        same as an ensemble trained from scratch would.
        """
        # argparse's --checkpoint takes nargs="+": one flag, N space-separated paths.
        # Repeating the flag itself (`--checkpoint a --checkpoint b`) would silently
        # keep only the last occurrence, which is the same failure mode this method
        # exists to avoid.
        n = self.ensemble_size if self.uncertainty_method == "ensemble" else 1
        return ["--checkpoint", *([str(pretrain_ckpt)] * n)]

    def _base_train_args(self, target_col: str) -> list[str]:
        """CLI args shared between `pretrain` and `fit`."""
        args = [
            "--smiles-columns",
            "smiles",
            "--target-columns",
            target_col,
            "--task-type",
            self._task_type(),
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
        if self.uncertainty_method == "ensemble":
            args += ["--ensemble-size", str(self.ensemble_size)]
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
            args += self._checkpoint_args(pretrain_ckpt)
            if self.freeze_encoder:
                args.append("--freeze-encoder")

        _run_chemprop_cli(args)
        train_csv.unlink(missing_ok=True)
        val_csv.unlink(missing_ok=True)
        return self

    #: uncertainty_method -> chemprop --uncertainty-method value at *predict* time.
    #: "mve" and "evidential" read out of the head trained for them
    #: (_UNCERTAINTY_TASK_TYPE); "evidential-total" is the combined
    #: aleatoric+epistemic variance -- the single number comparable to MVE's, rather
    #: than the epistemic/aleatoric split this comparison has no use for. "ensemble"
    #: and "dropout" work with an ordinary regression head.
    _UNCERTAINTY_PREDICT_METHOD = {
        "mve": "mve",
        "evidential": "evidential-total",
        "ensemble": "ensemble",
        "dropout": "dropout",
    }

    def _model_path_args(self) -> list[str]:
        """`--model-paths`, pointed at the checkpoint dir so an ensemble's several
        `model_i/best.pt` files are all picked up (chemprop discovers them from a
        directory); a non-ensemble run has exactly one such file."""
        return ["--model-paths", str(self.model_dir)]

    def predict(
        self, smiles_test: list[str], return_uncertainty: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Run inference via `chemprop predict`. Returns a 1-D array, or
        `(y_pred, y_unc)` when `return_uncertainty=True`.

        For `pred_type="classification"` these are positive-class probabilities, not
        labels -- thresholding is the caller's job, and for TDI the threshold should
        be tuned for MCC rather than left at 0.5 (see PLAN.md item 5).

        `y_unc` is chemprop's native uncertainty output: a variance for "mve" and
        "evidential", the ensemble's inter-checkpoint variance for "ensemble", and
        the MC-dropout variance for "dropout" -- always a variance, never a stdev, to
        match how `evaluation.py`'s calibration diagnostics (ENCE etc.) expect it.
        Requesting this without `uncertainty_method` set raises, since there is
        nothing chemprop can report.
        """
        if return_uncertainty and self.uncertainty_method is None:
            raise ValueError("return_uncertainty=True needs uncertainty_method set at construction")
        tmp = Path(tempfile.gettempdir())
        test_csv, pred_csv = tmp / "cyp_cp_test.csv", tmp / "cyp_cp_preds.csv"
        _write_smiles_csv(list(smiles_test), None, test_csv, self.target_col)
        cli_args = [
            "predict",
            "--test-path",
            str(test_csv),
            *self._model_path_args(),
            "--preds-path",
            str(pred_csv),
        ]
        if return_uncertainty:
            cli_args += [
                "--uncertainty-method",
                self._UNCERTAINTY_PREDICT_METHOD[self.uncertainty_method],
            ]
        _run_chemprop_cli(cli_args)
        preds_df = pl.read_csv(pred_csv)
        preds = preds_df[self.target_col].to_numpy()
        test_csv.unlink(missing_ok=True)
        pred_csv.unlink(missing_ok=True)
        y_pred = np.asarray(preds, dtype=float).flatten()
        if not return_uncertainty:
            return y_pred
        unc = preds_df[f"{self.target_col}_unc"].to_numpy()
        return y_pred, np.asarray(unc, dtype=float).flatten()


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

    def __init__(
        self,
        targets: list[str],
        descriptor_columns: list[str] | None = None,
        atom_feature_width: int | None = None,
        **kwargs,
    ) -> None:
        """
        Args:
            targets: Endpoint names, in the column order used by `fit` and returned
                by `predict`. Stored so both agree without the caller re-specifying.
            descriptor_columns: Names for extra molecule-level descriptors (e.g.
                `pharmacophore.DESCRIPTOR_COLUMNS`) to concatenate onto the learned
                representation before the FFN head, via chemprop's own
                `--descriptors-columns`. When set, `fit` and `predict` both require a
                `descriptors` array of shape `(n_compounds, len(descriptor_columns))`
                -- chemprop needs the same columns present, by name, in the training,
                validation and test CSVs, or the FFN's input width would not match
                between fit and predict. None (default) trains a plain model with no
                extra columns, unchanged from before this argument existed.
            atom_feature_width: Width of a per-atom extra feature vector (e.g.
                `plif.atom_features`' multi-hot interaction-type columns) to
                concatenate onto each atom's own learned representation before
                message passing, via chemprop's `--atom-features-path`. Unlike
                `descriptor_columns`, this is a *molecule-graph* input, not a CSV
                column, so `fit`/`predict` take a path to a `.npz` file
                (`plif.atom_features_npz`'s output) rather than an in-memory array
                -- chemprop reads atom features directly from that file, one array
                per CSV row in order. Set here purely so the encoder's message-
                passing layer is built with the right input width; the actual data
                is supplied per call. None (default) is unchanged from before this
                argument existed.
            **kwargs: Forwarded to `ChempropModel`.
        """
        if not targets:
            raise ValueError("targets must name at least one endpoint")
        super().__init__(**kwargs)
        self.targets = list(targets)
        self.descriptor_columns = list(descriptor_columns) if descriptor_columns else None
        self.atom_feature_width = atom_feature_width

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
            self._task_type(),
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
        if self.uncertainty_method == "ensemble":
            args += ["--ensemble-size", str(self.ensemble_size)]
        if self.descriptor_columns:
            args += ["--descriptors-columns", *self.descriptor_columns]
        return args

    def _check_descriptors(
        self, descriptors: np.ndarray | None, n_compounds: int
    ) -> np.ndarray | None:
        """Validate `descriptors` against `descriptor_columns`, in both directions.

        Both a model built with `descriptor_columns` fed no array, and one built
        without it fed an array, would otherwise fail silently: chemprop would either
        train a plain model no one asked for, or ignore an array whose columns were
        never named. Both are the kind of mismatch that only shows up as a confusing
        result days later, so this raises immediately instead.
        """
        if self.descriptor_columns is None:
            if descriptors is not None:
                raise ValueError(
                    "descriptors were given but this model has no descriptor_columns "
                    "set -- pass descriptor_columns to __init__ to use them"
                )
            return None
        if descriptors is None:
            raise ValueError(
                f"this model was built with descriptor_columns={self.descriptor_columns} "
                "and requires a matching `descriptors` array"
            )
        descriptors = np.asarray(descriptors, dtype=float)
        if descriptors.shape != (n_compounds, len(self.descriptor_columns)):
            raise ValueError(
                f"descriptors must have shape ({n_compounds}, "
                f"{len(self.descriptor_columns)}), got {descriptors.shape}"
            )
        return descriptors

    def _check_atom_features_path(
        self, atom_features_path: Path | None, n_compounds: int
    ) -> Path | None:
        """Validate an atom-features `.npz` against `atom_feature_width`, in both
        directions -- the file-based counterpart to `_check_descriptors`.

        Only the row count and the first array's column count are checked, not
        every array's shape: chemprop's own CLI raises on a genuinely malformed
        file, and eagerly loading every array here (a docking-derived one can be
        tens of thousands of atoms across a full endpoint) would burn the memory
        this check exists to avoid spending twice.
        """
        if self.atom_feature_width is None:
            if atom_features_path is not None:
                raise ValueError(
                    "atom_features_path was given but this model has no "
                    "atom_feature_width set -- pass atom_feature_width to __init__ "
                    "to use it"
                )
            return None
        if atom_features_path is None:
            raise ValueError(
                f"this model was built with atom_feature_width="
                f"{self.atom_feature_width} and requires a matching "
                "atom_features_path"
            )
        with np.load(atom_features_path) as archive:
            if len(archive.files) != n_compounds:
                raise ValueError(
                    f"{atom_features_path} has {len(archive.files)} arrays, expected "
                    f"one per compound ({n_compounds})"
                )
            first = archive["arr_0"]
            if first.ndim != 2 or first.shape[1] != self.atom_feature_width:
                raise ValueError(
                    f"{atom_features_path}'s arrays must be (n_atoms, "
                    f"{self.atom_feature_width}), got an array shaped {first.shape}"
                )
        return atom_features_path

    def _write_wide_csv(
        self,
        smiles: list[str],
        y: np.ndarray,
        path: Path,
        sample_weight: np.ndarray | None = None,
        descriptors: np.ndarray | None = None,
    ) -> None:
        """Write a multi-target CSV, leaving unmeasured cells empty.

        The empty cell is the point: Chemprop reads a blank as "no label" and masks
        it out of the loss, whereas a 0.0 would be read as a real measurement of an
        extremely inactive compound and would poison the head.
        """
        columns: dict[str, list] = {"smiles": list(smiles)}
        y = np.asarray(y, dtype=float)
        for i, target in enumerate(self.targets):
            column = y[:, i]
            columns[target] = [None if np.isnan(v) else float(v) for v in column]
        if sample_weight is not None:
            columns["weight"] = np.asarray(sample_weight, dtype=float).flatten().tolist()
        if descriptors is not None:
            for i, name in enumerate(self.descriptor_columns):
                columns[name] = descriptors[:, i].tolist()
        pl.DataFrame(columns).write_csv(path)

    def pretrain_wide(
        self,
        smiles_train: list[str],
        y_train: np.ndarray,
        smiles_val: list[str],
        y_val: np.ndarray,
        descriptors_train: np.ndarray | None = None,
        descriptors_val: np.ndarray | None = None,
    ) -> ChempropMultitargetModel:
        """Pretrain on an auxiliary multi-target matrix, keeping the same head shape.

        The inherited `pretrain` writes a single-target CSV, which cannot warm-start a
        four-head model: chemprop would build a one-output FFN and the checkpoint
        would not load into the fine-tuning architecture. This writes the same wide
        format `fit` uses, so the pretrained encoder *and* head transfer.

        Call once before the CV loop. `pretrain_dir` survives between folds, so every
        fold fine-tunes from the same encoder -- and because the auxiliary data is
        external to the challenge, the same checkpoint is legitimately reusable across
        folds without leaking a fold's own labels.

        `descriptors_train`/`descriptors_val` are required (and only accepted) when
        this model was built with `descriptor_columns`, for a reason that is easy to
        miss: `--checkpoint` restores the FFN's weights whole, including its input
        width. A checkpoint pretrained without descriptor columns has a narrower FFN
        than a fine-tuning model built with them, and chemprop fails the load with a
        matrix-shape error rather than adapting the width -- confirmed directly against
        the chemprop CLI, not inferred. So the descriptor columns have to be present
        at *both* stages, on the same names, even though the pretraining corpus has no
        genuine use for them beyond keeping the architecture loadable.
        """
        y_train = np.asarray(y_train, dtype=float)
        y_val = np.asarray(y_val, dtype=float)
        if y_train.ndim != 2 or y_train.shape[1] != len(self.targets):
            raise ValueError(f"y_train must be (n, {len(self.targets)}), got {y_train.shape}")
        descriptors_train = self._check_descriptors(descriptors_train, len(smiles_train))
        descriptors_val = self._check_descriptors(descriptors_val, len(smiles_val))

        tmp = Path(tempfile.gettempdir())
        train_csv, val_csv = tmp / "cyp_cp_mt_pre_train.csv", tmp / "cyp_cp_mt_pre_val.csv"
        self._write_wide_csv(list(smiles_train), y_train, train_csv, descriptors=descriptors_train)
        self._write_wide_csv(list(smiles_val), y_val, val_csv, descriptors=descriptors_val)

        if self.pretrain_dir.exists():
            shutil.rmtree(self.pretrain_dir)

        _run_chemprop_cli(
            [
                "train",
                "--data-path",
                str(train_csv),
                str(val_csv),
                str(val_csv),
                *self._base_train_args("ignored"),
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
        descriptors: np.ndarray | None = None,
        descriptors_val: np.ndarray | None = None,
        atom_features_path: Path | None = None,
    ) -> ChempropMultitargetModel:
        """Train on a `(n_compounds, n_targets)` matrix with NaN for unmeasured.

        `sample_weight` weights a *compound*, not a (compound, endpoint) pair, because
        that is the only thing Chemprop's `-w` can express. A compound measured for
        three isoforms therefore carries one weight covering all three. Per-endpoint
        weighting is not available here at all, so a caller that needs it has to fall
        back to single-task models -- see `aux_training.run_cv_selection_corrected`,
        which averages its per-endpoint propensities for exactly this reason.

        `descriptors`/`descriptors_val` are required (and only accepted) when this
        model was built with `descriptor_columns` -- see `_check_descriptors`. When
        `smiles_val`/`y_val` are omitted and an internal split is taken, `descriptors`
        is split the same way so every row's descriptor stays paired with its own
        compound.

        `atom_features_path` is the equivalent requirement for `atom_feature_width`,
        but takes a **different code path through chemprop's CLI**, confirmed
        directly against the installed version (2.2.1) rather than assumed:
        `--atom-features-path` is read once per training invocation and applied
        identically to every split chemprop parses, so it only works when
        `--data-path` names a **single combined file** that chemprop splits
        internally via `--splits-file` -- the three-separate-train/val/test-file
        convention every other `fit` in this module uses cannot pair an atom-feature
        array with more than one of those files at once (each file has its own row
        count, and the array must match whichever CSV chemprop is currently
        parsing). `smiles_train`/`smiles_val` are therefore concatenated into one
        CSV here, with a `--splits-file` JSON recording which rows are which,
        instead of the two-file pattern above. This means, unlike `descriptors`,
        `atom_features_path` **requires an explicit `smiles_val`/`y_val`** --
        chemprop's own internal splitter (used when they are omitted) would put
        rows in an order this method cannot predict, and the atom-features array's
        row order has to be decided before calling it.
        """
        y_train = np.asarray(y_train, dtype=float)
        if y_train.ndim != 2 or y_train.shape[1] != len(self.targets):
            raise ValueError(f"y_train must be (n, {len(self.targets)}), got {y_train.shape}")

        smiles_train = list(smiles_train)
        descriptors = self._check_descriptors(descriptors, len(smiles_train))
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float).flatten()
            if len(sample_weight) != len(smiles_train):
                raise ValueError(
                    f"sample_weight must have one entry per compound: got "
                    f"{len(sample_weight)} for {len(smiles_train)} compounds"
                )

        if atom_features_path is not None and (smiles_val is None or y_val is None):
            raise ValueError(
                "atom_features_path requires an explicit smiles_val/y_val split -- "
                "see the docstring for why an internal random split cannot be used "
                "with atom features"
            )

        if smiles_val is None or y_val is None:
            rng = np.random.default_rng(42)
            idx = rng.permutation(len(smiles_train))
            n_val = max(1, int(0.1 * len(idx)))
            val_idx, train_idx = idx[:n_val], idx[n_val:]
            smiles_val = [smiles_train[i] for i in val_idx]
            y_val = y_train[val_idx]
            smiles_train = [smiles_train[i] for i in train_idx]
            y_train = y_train[train_idx]
            if sample_weight is not None:
                sample_weight = sample_weight[train_idx]
            if descriptors is not None:
                descriptors_val = descriptors[val_idx]
                descriptors = descriptors[train_idx]
        else:
            descriptors_val = self._check_descriptors(descriptors_val, len(smiles_val))
            # Existence checked here too, not only inside `_fit_with_atom_features`:
            # a model built with `atom_feature_width` but given no
            # `atom_features_path` would otherwise silently take the plain
            # two-file training path below and train without atom features at all
            # -- exactly the "confusing result days later" `_check_descriptors`
            # exists to prevent for descriptors. The shape check happens later,
            # against `len(smiles_train) + len(smiles_val)`
            # (`_fit_with_atom_features` does it): the array covers both splits
            # combined, which `smiles_val`'s count alone cannot validate.
            if self.atom_feature_width is not None and atom_features_path is None:
                raise ValueError(
                    f"this model was built with atom_feature_width="
                    f"{self.atom_feature_width} and requires a matching "
                    "atom_features_path"
                )
            if self.atom_feature_width is None and atom_features_path is not None:
                raise ValueError(
                    "atom_features_path was given but this model has no "
                    "atom_feature_width set -- pass atom_feature_width to __init__ "
                    "to use it"
                )

        if self.model_dir.exists():
            shutil.rmtree(self.model_dir)

        if atom_features_path is not None:
            self._fit_with_atom_features(
                smiles_train,
                y_train,
                smiles_val,
                np.asarray(y_val, dtype=float),
                target_col,
                sample_weight,
                atom_features_path,
            )
            return self

        tmp = Path(tempfile.gettempdir())
        train_csv, val_csv = tmp / "cyp_cp_mt_train.csv", tmp / "cyp_cp_mt_val.csv"
        # Chemprop needs the weight column in every split once it is used at all, so
        # the validation rows get a uniform 1.0 -- early stopping should measure fit
        # quality, not the reweighting.
        val_weight = None if sample_weight is None else np.ones(len(smiles_val))
        self._write_wide_csv(smiles_train, y_train, train_csv, sample_weight, descriptors)
        self._write_wide_csv(
            smiles_val, np.asarray(y_val, dtype=float), val_csv, val_weight, descriptors_val
        )

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

        # Same precedence rule as the single-target `fit`: our own auxiliary
        # checkpoint outranks the generic foundation backbone when both exist.
        pretrain_ckpt = self.pretrain_dir / "model_0" / "best.pt"
        if pretrain_ckpt.exists():
            args += self._checkpoint_args(pretrain_ckpt)
            if self.freeze_encoder:
                args.append("--freeze-encoder")

        _run_chemprop_cli(args)
        train_csv.unlink(missing_ok=True)
        val_csv.unlink(missing_ok=True)
        return self

    def _fit_with_atom_features(
        self,
        smiles_train: list[str],
        y_train: np.ndarray,
        smiles_val: list[str],
        y_val: np.ndarray,
        target_col: str,
        sample_weight: np.ndarray | None,
        atom_features_path: Path,
    ) -> None:
        """The single-combined-file training path `atom_features_path` requires.

        `atom_features_path` must already cover exactly `smiles_train + smiles_val`,
        in that concatenated order -- this does not slice or reorder it, since doing
        so would mean reading and rewriting a `.npz` that may hold tens of thousands
        of per-atom arrays for no benefit over building it in the right order once.
        `plif.atom_features_npz`'s `names_order`/`smiles_order` are exactly this
        concatenated order when the caller builds them that way.
        """
        import json

        n_train, n_val = len(smiles_train), len(smiles_val)
        self._check_atom_features_path(atom_features_path, n_train + n_val)

        tmp = Path(tempfile.gettempdir())
        combined_csv = tmp / "cyp_cp_mt_atomfeat_combined.csv"
        splits_json = tmp / "cyp_cp_mt_atomfeat_splits.json"

        combined_weight = None
        if sample_weight is not None:
            # Validation rows get uniform weight 1.0, same convention as the
            # two-file path -- early stopping should measure fit quality, not
            # whatever reweighting the training rows carry.
            combined_weight = np.concatenate([sample_weight, np.ones(n_val)])
        self._write_wide_csv(
            smiles_train + smiles_val,
            np.vstack([y_train, y_val]),
            combined_csv,
            combined_weight,
            descriptors=None,  # descriptor_columns + atom_features_path together
            # is not a combination this repo has needed yet; _check_descriptors
            # already rejects a descriptors array with no descriptor_columns set,
            # so a caller trying to combine them gets a clear error, not a silent
            # drop of one or the other.
        )
        # Chemprop's test split is required by the CLI but plays no role here --
        # `fit` never reads chemprop's own test-set metrics, only the saved
        # checkpoint -- so it is the same rows as validation rather than a third
        # carve-out this method has no use for.
        splits = [
            {
                "train": list(range(n_train)),
                "val": list(range(n_train, n_train + n_val)),
                "test": list(range(n_train, n_train + n_val)),
            }
        ]
        splits_json.write_text(json.dumps(splits))

        args = [
            "train",
            "--data-path",
            str(combined_csv),
            "--splits-file",
            str(splits_json),
            *self._base_train_args(target_col),
            "--atom-features-path",
            str(atom_features_path),
            "--epochs",
            str(self.epochs),
            "--save-dir",
            str(self.model_dir),
        ]
        if sample_weight is not None:
            args += ["-w", "weight"]

        pretrain_ckpt = self.pretrain_dir / "model_0" / "best.pt"
        if pretrain_ckpt.exists():
            args += self._checkpoint_args(pretrain_ckpt)
            if self.freeze_encoder:
                args.append("--freeze-encoder")

        _run_chemprop_cli(args)
        combined_csv.unlink(missing_ok=True)
        splits_json.unlink(missing_ok=True)

    def predict(
        self,
        smiles_test: list[str],
        return_uncertainty: bool = False,
        descriptors: np.ndarray | None = None,
        atom_features_path: Path | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict every endpoint at once. Returns `(n_compounds, n_targets)` in
        `self.targets` order, or `(y_pred, y_unc)` of that same shape when
        `return_uncertainty=True` -- see `ChempropModel.predict` for what `y_unc`
        means per method (always a variance).

        `descriptors` is required when this model was built with `descriptor_columns`
        -- the FFN's input width was fixed at fit time to include them, so predicting
        without them would not merely give a worse answer, it would not run at all.
        `atom_features_path` is the equivalent requirement for `atom_feature_width`.
        """
        if return_uncertainty and self.uncertainty_method is None:
            raise ValueError("return_uncertainty=True needs uncertainty_method set at construction")
        smiles_test = list(smiles_test)
        descriptors = self._check_descriptors(descriptors, len(smiles_test))
        atom_features_path = self._check_atom_features_path(atom_features_path, len(smiles_test))
        tmp = Path(tempfile.gettempdir())
        test_csv, pred_csv = tmp / "cyp_cp_mt_test.csv", tmp / "cyp_cp_mt_preds.csv"
        test_columns: dict[str, list] = {"smiles": smiles_test}
        if descriptors is not None:
            for i, name in enumerate(self.descriptor_columns):
                test_columns[name] = descriptors[:, i].tolist()
        pl.DataFrame(test_columns).write_csv(test_csv)

        cli_args = [
            "predict",
            "--test-path",
            str(test_csv),
            *self._model_path_args(),
            "--preds-path",
            str(pred_csv),
        ]
        if self.descriptor_columns:
            cli_args += ["--descriptors-columns", *self.descriptor_columns]
        if atom_features_path is not None:
            cli_args += ["--atom-features-path", str(atom_features_path)]
        if return_uncertainty:
            cli_args += [
                "--uncertainty-method",
                self._UNCERTAINTY_PREDICT_METHOD[self.uncertainty_method],
            ]
        _run_chemprop_cli(cli_args)

        preds = pl.read_csv(pred_csv)
        missing = [t for t in self.targets if t not in preds.columns]
        if missing:
            raise RuntimeError(
                f"chemprop predictions missing target columns {missing}; got {preds.columns}"
            )
        out = preds.select(self.targets).to_numpy().astype(float)

        test_csv.unlink(missing_ok=True)
        pred_csv.unlink(missing_ok=True)

        if not return_uncertainty:
            return out

        unc_cols = [f"{t}_unc" for t in self.targets]
        missing_unc = [c for c in unc_cols if c not in preds.columns]
        if missing_unc:
            raise RuntimeError(
                f"chemprop predictions missing uncertainty columns {missing_unc}; "
                f"got {preds.columns}"
            )
        unc = preds.select(unc_cols).to_numpy().astype(float)
        return out, unc

    def predict_ensemble_members(self, smiles_test: list[str]) -> np.ndarray:
        """Each ensemble checkpoint's own prediction, unaveraged.

        Only meaningful for `uncertainty_method="ensemble"`: chemprop's own
        `predict` averages the checkpoints into one number and reports their
        variance as `_unc`, which is enough to rank compounds by confidence but not
        enough to build anything other than a uniform average from the ensemble --
        e.g. an inverse-variance-weighted blend needs each member's own prediction.
        Chemprop writes those to a `..._individual.csv` sidecar next to the normal
        predictions file (`{target}_model_{i}` columns) whenever more than one
        checkpoint is used; this reads that file instead of the averaged one.

        Returns `(n_compounds, n_targets, ensemble_size)`.
        """
        if self.uncertainty_method != "ensemble":
            raise ValueError(
                "predict_ensemble_members only applies to uncertainty_method='ensemble', "
                f"got {self.uncertainty_method!r}"
            )
        tmp = Path(tempfile.gettempdir())
        test_csv, pred_csv = tmp / "cyp_cp_mt_test.csv", tmp / "cyp_cp_mt_preds.csv"
        individual_csv = pred_csv.with_name(f"{pred_csv.stem}_individual{pred_csv.suffix}")
        pl.DataFrame({"smiles": list(smiles_test)}).write_csv(test_csv)

        _run_chemprop_cli(
            [
                "predict",
                "--test-path",
                str(test_csv),
                *self._model_path_args(),
                "--preds-path",
                str(pred_csv),
                "--uncertainty-method",
                "ensemble",
            ]
        )

        preds = pl.read_csv(individual_csv)
        member_cols = {
            target: [f"{target}_model_{i}" for i in range(self.ensemble_size)]
            for target in self.targets
        }
        missing = [c for cols in member_cols.values() for c in cols if c not in preds.columns]
        if missing:
            raise RuntimeError(
                f"chemprop individual predictions missing columns {missing}; got {preds.columns}"
            )
        out = np.stack(
            [preds.select(member_cols[t]).to_numpy().astype(float) for t in self.targets], axis=1
        )
        test_csv.unlink(missing_ok=True)
        pred_csv.unlink(missing_ok=True)
        individual_csv.unlink(missing_ok=True)
        return out
