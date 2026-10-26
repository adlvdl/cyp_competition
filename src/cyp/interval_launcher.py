"""Chemprop CLI entry point that adds the ST-RAE interval loss.

Run as a script, never imported by `cyp/__init__.py`: it imports torch and chemprop,
which must stay out of any process that also fits LightGBM (CLAUDE.md, OpenMP traps).
`graph_models._run_chemprop_cli` runs `CHEMPROP_BIN` as a subprocess, so pointing that
at a shim that execs this file is enough to train with the new loss; nothing else in
the training path changes.

## Why a launcher and not a flag

ST-RAE scores a prediction as the distance outside its compound's credible interval,
and zero inside it. The exact training analogue is the hinge

    loss = relu(lo - pred) + relu(pred - hi)

but chemprop's CLI only offers losses that see one target value per cell (its
`bounded-*` losses take a one-sided censor, not an interval). The interval itself is
not in the training file. It is, however, almost a function of the label alone:
isotonic regression of the band half-widths on the pIC50 recovers it at R^2 0.93-0.97
(Lizard Wizard's report, re-measured by `interval_loss.band_table`). So the loss rebuilds
each cell's band from its own target value, using a lookup table written next to the
training run.

Chemprop trains on targets standardised with a per-task mean and standard deviation, so
the loss undoes that scaling to read pIC50, looks up the band, and re-applies it to compare
against the (scaled) prediction. The statistics are recomputed here exactly as chemprop
computes them (NaN-aware mean and population standard deviation of the first `--data-path`
file; checked against a saved model's `output_transform` in `tests/test_interval_loss.py`).
Under `--checkpoint` chemprop instead scales by the checkpoint's own `output_transform`, so
that is read from the checkpoint file.

## Two behaviours of chemprop this works around

1. `--checkpoint` rebuilds the model from the saved file and **silently ignores
   `--loss-function`**: the fine-tune trains on whatever criterion the checkpoint was saved
   with (MSE for `pubchem`). Left alone, an "interval" fine-tune would be an MSE fine-tune.
   `_apply_loss_after_checkpoint_load` wraps `MPNN.load_from_file` so the requested loss
   replaces both the training criterion and the validation metric that early stopping and
   best-checkpoint selection read. It prints a line to stderr when it fires, which the CLI
   test asserts.
2. Heads with no entry in the band table (the pooled model's auxiliary and public heads)
   train on plain L1, so a multi-head model can use the interval loss on its scored heads only.

## Environment

    CYP_INTERVAL_TABLE   path to the .npz from `interval_loss.write_band_table`
    CYP_INTERVAL_TIEBREAK  weight of an added |pred - target| term for
                           `interval-mae-tb` (default 0.1)

Two losses are registered: `interval-mae` (the pure hinge) and `interval-mae-tb` (hinge
plus a small L1 pull toward the point estimate, so predictions inside the band are not
left to drift).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
from chemprop.nn.metrics import ChempropMetric, LossFunctionRegistry


def _arg_values(argv: list[str], flag: str) -> list[str]:
    """Values following `flag` up to the next option."""
    i = argv.index(flag) + 1
    out: list[str] = []
    while i < len(argv) and not argv[i].startswith("-"):
        out.append(argv[i])
        i += 1
    return out


def train_stats(argv: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Target names and the (mean, std) chemprop standardises them with."""
    targets = _arg_values(argv, "--target-columns")
    if "--checkpoint" in argv:
        # Fine-tuning: targets are scaled by the checkpoint's output transform.
        ckpt = torch.load(
            _arg_values(argv, "--checkpoint")[0], map_location="cpu", weights_only=False
        )
        state = ckpt["state_dict"]
        mean = state["predictor.output_transform.mean"].numpy().reshape(-1)
        std = state["predictor.output_transform.scale"].numpy().reshape(-1)
        if len(mean) != len(targets):
            raise ValueError(
                f"checkpoint has {len(mean)} heads, {len(targets)} target columns given"
            )
        return targets, mean, std
    train_csv = _arg_values(argv, "--data-path")[0]
    y = pd.read_csv(train_csv)[targets].to_numpy(dtype=float)
    std = np.nanstd(y, axis=0)
    # sklearn's StandardScaler (what chemprop uses) leaves a zero-variance column unscaled.
    std = np.where(std == 0.0, 1.0, std)
    return targets, np.nanmean(y, axis=0), std


def _interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear lookup, clamped to the grid ends."""
    x = x.clamp(float(xp[0]), float(xp[-1]))
    i = torch.bucketize(x, xp).clamp(1, len(xp) - 1)
    x0, x1, f0, f1 = xp[i - 1], xp[i], fp[i - 1], fp[i]
    return f0 + (x - x0) / (x1 - x0) * (f1 - f0)


class IntervalMAE(ChempropMetric):
    """Distance outside the compound's credible interval, in standardised units.

    Heads absent from the band table fall back to plain L1.
    """

    tiebreak: float = 0.0

    def __init__(self, task_weights=1.0, stats=None, table=None, **kwargs):
        super().__init__(task_weights)
        names, mean, std = stats if stats is not None else train_stats(sys.argv)
        tab = (
            table
            if table is not None
            else np.load(os.environ["CYP_INTERVAL_TABLE"], allow_pickle=True)
        )
        known = [str(t) for t in tab["targets"]]
        banded = [n in known for n in names]
        if not any(banded):
            raise ValueError(f"none of {names} has a band table entry ({known})")
        lo = np.zeros((len(names), len(tab["grid"])))
        hi = np.zeros_like(lo)
        for k, n in enumerate(names):
            if banded[k]:
                lo[k], hi[k] = tab["lo_off"][known.index(n)], tab["hi_off"][known.index(n)]
        self.register_buffer("banded", torch.as_tensor(banded).view(1, -1))
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).view(1, -1))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).view(1, -1))
        self.register_buffer("grid", torch.as_tensor(tab["grid"], dtype=torch.float32))
        self.register_buffer("lo_off", torch.as_tensor(lo, dtype=torch.float32))
        self.register_buffer("hi_off", torch.as_tensor(hi, dtype=torch.float32))

    def _calc_unreduced_loss(self, preds, targets, *args) -> torch.Tensor:
        y = targets * self.std + self.mean  # pIC50
        lo = torch.stack(
            [_interp(y[:, k], self.grid, self.lo_off[k]) for k in range(y.shape[1])], dim=1
        )
        hi = torch.stack(
            [_interp(y[:, k], self.grid, self.hi_off[k]) for k in range(y.shape[1])], dim=1
        )
        lo_s, hi_s = (y - lo - self.mean) / self.std, (y + hi - self.mean) / self.std
        hinge = torch.relu(lo_s - preds) + torch.relu(preds - hi_s)
        if self.tiebreak:
            hinge = hinge + self.tiebreak * (preds - targets).abs()
        return torch.where(self.banded, hinge, (preds - targets).abs())


@LossFunctionRegistry.register("interval-mae")
class IntervalMAEPure(IntervalMAE):
    pass


@LossFunctionRegistry.register("interval-mae-tb")
class IntervalMAETiebreak(IntervalMAE):
    tiebreak = float(os.environ.get("CYP_INTERVAL_TIEBREAK", "0.1"))


def _apply_loss_after_checkpoint_load(argv: list[str]) -> None:
    """Make `--loss-function` take effect under `--checkpoint` (chemprop ignores it there)."""
    if "--checkpoint" not in argv or "--loss-function" not in argv:
        return
    name = _arg_values(argv, "--loss-function")[0]
    from chemprop.models.model import MPNN

    original = MPNN.load_from_file.__func__

    def load_from_file(cls, *args, **kwargs):
        model = original(cls, *args, **kwargs)
        weights = (
            [float(w) for w in _arg_values(argv, "--task-weights")]
            if "--task-weights" in argv
            else [1.0] * model.n_tasks
        )
        criterion = LossFunctionRegistry[name](task_weights=weights)
        model.predictor.criterion = criterion
        # Chemprop rebuilds the model from its saved hyperparameters when it reloads the best
        # checkpoint, so the criterion has to be recorded there too. Without this the rebuilt
        # model has the default criterion and rejects the state dict's loss buffers.
        model.predictor.hparams["criterion"] = criterion
        model.metrics[-1] = criterion.clone()
        print(f"interval_launcher: criterion after checkpoint load = {name}", file=sys.stderr)
        return model

    MPNN.load_from_file = classmethod(load_from_file)


if __name__ == "__main__":
    from chemprop.cli.main import main

    _apply_loss_after_checkpoint_load(sys.argv)
    main()
