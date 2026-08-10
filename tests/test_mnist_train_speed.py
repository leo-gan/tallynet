"""MNIST training speed: old float-sum kernel vs packed SIMD.

Needs ``data/MNIST`` (copy from ``binary-optimizers``) and torchvision.
Skipped otherwise. Same writeback; only the tally decode differs.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import pytest
import torch

from tallynet.models import TallyMLP
from tallynet.native import load_native
from tallynet.ref_old import old_forward, simd_forward, train_step
from tallynet.writeback import TallyWriteback

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _mnist_available() -> bool:
    raw = DATA_DIR / "MNIST" / "raw"
    return (raw / "train-images-idx3-ubyte").is_file()


def _median_ms(fn, *, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


@pytest.fixture(scope="module")
def mnist_batch():
    if load_native() is None:
        pytest.skip("native SIMD popcount library not built")
    if not _mnist_available():
        pytest.skip(f"MNIST not found under {DATA_DIR} (copy from binary-optimizers)")
    try:
        from tallynet.data import mnist_loaders
    except ImportError:
        pytest.skip("torchvision not installed (uv sync --extra train)")
    torch.manual_seed(0)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    train_loader, _ = mnist_loaders(DATA_DIR, batch_size=128, num_workers=0)
    x, y = next(iter(train_loader))
    return x, y


def test_mnist_train_step_simd_faster_than_old_kernel(mnist_batch):
    x, y = mnist_batch
    model = TallyMLP(hidden_dim=128, tally_width=256, encoder="majority", ln_mode="none")
    wb = TallyWriteback(model.tally_layers(), opt="adam", lr=1e-3)

    t_old = _median_ms(
        lambda: train_step(model, wb, x, y, old_forward),
        warmup=2,
        runs=8,
    )
    t_simd = _median_ms(
        lambda: train_step(model, wb, x, y, simd_forward),
        warmup=2,
        runs=8,
    )
    assert t_simd < t_old, (
        f"SIMD train step {t_simd:.2f} ms is not faster than old kernel {t_old:.2f} ms"
    )
    model.assert_binary_invariants()


def test_mnist_train_step_old_and_simd_both_finite(mnist_batch):
    x, y = mnist_batch
    model = TallyMLP(hidden_dim=128, tally_width=256, encoder="majority", ln_mode="none")
    wb = TallyWriteback(model.tally_layers(), opt="adam", lr=1e-3)
    loss_old = train_step(model, wb, x, y, old_forward)
    loss_simd = train_step(model, wb, x, y, simd_forward)
    assert loss_old > 0.0 and loss_simd > 0.0
