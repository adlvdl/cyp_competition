"""Tests for the ST-RAE interval loss used to train chemprop (`interval_loss`, `interval_launcher`).

The launcher runs inside the chemprop subprocess, so a wrong sign or a mismatch between
its scaling and chemprop's would not raise: the model would simply train against the
wrong target and report a plausible loss. These pin the arithmetic and the one
end-to-end property worth having, that the CLI accepts the registered loss name.
"""

from __future__ import annotations

import importlib.util
import subprocess

import numpy as np
import polars as pl
import pytest
import torch

from cyp import interval_loss

spec = importlib.util.spec_from_file_location("interval_launcher", interval_loss.LAUNCHER)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def _loss(cls, preds, targets, mean=5.0, std=2.0, lo=0.5, hi=1.0):
    table = {
        "targets": np.array(["a"]),
        "grid": np.array([1.0, 9.0]),
        "lo_off": np.array([[lo, lo]]),
        "hi_off": np.array([[hi, hi]]),
    }
    metric = cls(stats=(["a"], np.array([mean]), np.array([std])), table=table)
    return metric._calc_unreduced_loss(
        torch.tensor(preds, dtype=torch.float32).view(-1, 1),
        torch.tensor(targets, dtype=torch.float32).view(-1, 1),
    ).view(-1)


def test_zero_inside_the_band_and_linear_outside_in_scaled_units():
    # label 7.0 -> scaled 1.0; band [6.5, 8.0] -> scaled [0.75, 1.5]
    out = _loss(launcher.IntervalMAEPure, [0.75, 1.0, 1.5, 2.5, 0.25], [1.0] * 5)
    assert out.tolist() == pytest.approx([0.0, 0.0, 0.0, 1.0, 0.5])


def test_tiebreak_pulls_toward_the_point_estimate_inside_the_band():
    pure = _loss(launcher.IntervalMAEPure, [1.2], [1.0])
    tb = _loss(launcher.IntervalMAETiebreak, [1.2], [1.0])
    assert pure.item() == 0.0 and tb.item() == pytest.approx(0.1 * 0.2, rel=1e-4)


def test_band_table_recovers_a_label_determined_band():
    rng = np.random.default_rng(0)
    y = rng.uniform(2.0, 8.0, 800)
    half = 1.5 - 0.15 * (y - 2.0)  # wide for weak, tight for potent
    frame = pl.DataFrame({"y_true": y, "y_lower": y - half, "y_upper": y + 0.5 * half})
    table = interval_loss.band_table({"a": frame})
    quality = interval_loss.band_fit_quality({"a": frame}, table)
    assert quality["r2_lo"][0] > 0.99 and quality["r2_hi"][0] > 0.99


def test_chemprop_cli_accepts_the_registered_loss(tmp_path):
    """Three epochs on twelve molecules: the loss name parses, trains and saves a model."""
    smiles = [
        "CCO",
        "CCN",
        "CCC",
        "CCCl",
        "c1ccccc1",
        "CC(=O)O",
        "CCOC",
        "CCS",
        "CCBr",
        "c1ccncc1",
        "CC(C)O",
        "CCCC",
    ]
    rng = np.random.default_rng(1)
    y = rng.uniform(4, 7, len(smiles))
    csv = tmp_path / "t.csv"
    pl.DataFrame({"smiles": smiles, "a": y}).write_csv(csv)
    frame = pl.DataFrame({"y_true": y, "y_lower": y - 0.4, "y_upper": y + 0.4})
    table = interval_loss.write_band_table(
        interval_loss.band_table({"a": frame}), tmp_path / "band.npz"
    )
    shim = interval_loss.write_shim(tmp_path / "chemprop_interval")
    result = subprocess.run(
        [
            str(shim),
            "train",
            "--data-path",
            str(csv),
            str(csv),
            str(csv),
            "--smiles-columns",
            "smiles",
            "--target-columns",
            "a",
            "--task-type",
            "regression",
            "--epochs",
            "3",
            "--accelerator",
            "cpu",
            "--loss-function",
            "interval-mae",
            "--save-dir",
            str(tmp_path / "m"),
        ],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "CYP_INTERVAL_TABLE": str(table)},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert any((tmp_path / "m").rglob("best*.pt")) or any((tmp_path / "m").rglob("*.pt"))


def test_heads_without_a_band_fall_back_to_plain_l1():
    table = {
        "targets": np.array(["a"]),
        "grid": np.array([1.0, 9.0]),
        "lo_off": np.array([[0.5, 0.5]]),
        "hi_off": np.array([[1.0, 1.0]]),
    }
    metric = launcher.IntervalMAEPure(
        stats=(["a", "aux"], np.array([5.0, 5.0]), np.array([2.0, 2.0])), table=table
    )
    preds = torch.tensor([[2.5, 2.5]])  # both heads predict 2.5 against a scaled target of 1.0
    targets = torch.tensor([[1.0, 1.0]])
    out = metric._calc_unreduced_loss(preds, targets).view(-1)
    # "a": 2.5 is 1.0 above the scaled upper bound of 1.5. "aux": plain |2.5 - 1.0|.
    assert out.tolist() == pytest.approx([1.0, 1.5])


def _tiny_files(tmp_path):
    smiles = [
        "CCO", "CCN", "CCC", "CCCl", "c1ccccc1", "CC(=O)O", "CCOC", "CCS", "CCBr", "c1ccncc1",
        "CC(C)O", "CCCC", "CCCO", "CCCN", "c1ccccc1O", "CC(C)C", "CCCCC", "c1ccccc1C", "CCOCC",
        "CCCS",
    ]  # fmt: skip
    rng = np.random.default_rng(3)
    a, b = rng.uniform(3, 8, len(smiles)), rng.uniform(4, 6, len(smiles))
    b[::3] = np.nan
    csv = tmp_path / "t.csv"
    pl.DataFrame({"smiles": smiles, "a": a, "b": b}).write_csv(csv)
    frame = pl.DataFrame({"y_true": a, "y_lower": a - 0.4, "y_upper": a + 0.4})
    table = interval_loss.write_band_table(
        interval_loss.band_table({"a": frame}), tmp_path / "band.npz"
    )
    return csv, table, np.column_stack([a, b])


def _train(shim, csv, table, tmp_path, save, *extra):
    import os

    return subprocess.run(
        [str(shim), "train", "--data-path", str(csv), str(csv), str(csv), "--smiles-columns",
         "smiles", "--target-columns", "a", "b", "--task-type", "regression", "--epochs", "3",
         "--accelerator", "cpu", "--save-dir", str(tmp_path / save), *extra],
        capture_output=True, text=True,
        env={**os.environ, "CYP_INTERVAL_TABLE": str(table)},
    )  # fmt: skip


def test_scaler_matches_chemprop_and_checkpoint_finetune_honours_the_loss(tmp_path):
    """Two facts the whole checkpoint path rests on, both measured against real chemprop.

    1. The NaN-aware mean / population std this launcher assumes is what chemprop saves as
       the model's `output_transform`, for a fresh run and read back from a checkpoint.
    2. Under `--checkpoint`, chemprop ignores `--loss-function` unless the launcher patches
       it in; the launcher says so on stderr when the patch fires.
    """
    csv, table, y = _tiny_files(tmp_path)
    shim = interval_loss.write_shim(tmp_path / "chemprop_interval")
    first = _train(shim, csv, table, tmp_path, "pre", "--loss-function", "mae")
    assert first.returncode == 0, first.stderr[-2000:]
    ckpt = next((tmp_path / "pre").rglob("best.pt"))
    names, mean, std = launcher.train_stats(
        ["--target-columns", "a", "b", "--checkpoint", str(ckpt)]
    )
    assert names == ["a", "b"]
    assert mean == pytest.approx(np.nanmean(y, axis=0), rel=1e-5)
    assert std == pytest.approx(np.nanstd(y, axis=0), rel=1e-5)
    assert launcher.train_stats(["--data-path", str(csv), "--target-columns", "a", "b"])[1] == (
        pytest.approx(mean, rel=1e-5)
    )

    tuned = _train(
        shim,
        csv,
        table,
        tmp_path,
        "ft",
        "--checkpoint",
        str(ckpt),
        "--loss-function",
        "interval-mae",
    )
    assert tuned.returncode == 0, tuned.stderr[-2000:]
    assert "criterion after checkpoint load = interval-mae" in tuned.stderr

    # The fine-tuned model must also reload for prediction, which is where a criterion
    # recorded on the live model but not in its hyperparameters would fail.
    import os

    predicted = subprocess.run(
        [str(shim), "predict", "--test-path", str(csv), "--smiles-columns", "smiles",
         "--model-paths", str(tmp_path / "ft"), "--preds-path", str(tmp_path / "p.csv"),
         "--accelerator", "cpu"],
        capture_output=True, text=True,
        env={**os.environ, "CYP_INTERVAL_TABLE": str(table)},
    )  # fmt: skip
    assert predicted.returncode == 0, predicted.stderr[-2000:]
    assert pl.read_csv(tmp_path / "p.csv").height == 20
