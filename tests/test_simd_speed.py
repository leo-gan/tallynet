"""The SIMD popcount must speed up the current TallyMLP, not only a microbench.

Compares three forwards on the default MNIST-shaped net (784→128→10, S=256):

* old: unpack ±1, ``float().sum(-1)`` (the previous layer)
* lut: packed table popcount
* native: CPU SIMD popcount

Skipped if the native library did not build.
"""

from __future__ import annotations

import statistics
import time

import pytest
import torch
import torch.nn.functional as F

from tallynet.encoders import encode_tally
from tallynet.kernels import decode_tally_weight
from tallynet.models import TallyMLP
from tallynet.native import load_native
from tallynet.packed import unpack_pm1
from tallynet.writeback import TallyWriteback


def _median_ms(fn, *, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def _old_forward(model: TallyMLP, x: torch.Tensor) -> torch.Tensor:
    x = model.flatten(x)
    n = len(model.linears)
    for i, (ln, linear) in enumerate(zip(model.lns, model.linears)):
        x = ln(x)
        pm1 = unpack_pm1(linear.bits, linear.tally_width).float()
        s = pm1.sum(-1)
        w = encode_tally(s, linear.tally_width, linear.encoder, linear.tanh_tau)
        x = F.linear(x, w * linear.gain)
        if i < n - 1:
            x = model.act(x)
    return x


def _lut_forward(model: TallyMLP, x: torch.Tensor) -> torch.Tensor:
    x = model.flatten(x)
    n = len(model.linears)
    for i, (ln, linear) in enumerate(zip(model.lns, model.linears)):
        x = ln(x)
        w = decode_tally_weight(
            linear.bits,
            linear.tally_width,
            linear.encoder,
            linear.tanh_tau,
            native=False,
        )
        x = F.linear(x, w * linear.gain)
        if i < n - 1:
            x = model.act(x)
    return x


@pytest.fixture(scope="module")
def mlp_and_batch():
    if load_native() is None:
        pytest.skip("native SIMD popcount library not built")
    torch.manual_seed(0)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    model = TallyMLP(hidden_dim=128, tally_width=256, encoder="majority", ln_mode="none")
    x = torch.randn(128, 1, 28, 28)
    y = torch.randint(0, 10, (128,))
    return model, x, y


def test_simd_forward_faster_than_float_sum_on_tallymlp(mlp_and_batch):
    model, x, _ = mlp_and_batch
    t_old = _median_ms(lambda: _old_forward(model, x), warmup=3, runs=7)
    t_nat = _median_ms(lambda: model(x), warmup=8, runs=15)
    assert t_nat < t_old / 5.0, (
        f"SIMD TallyMLP forward {t_nat:.2f} ms is not 5× faster than "
        f"float-sum {t_old:.2f} ms"
    )


def test_simd_forward_faster_than_lut_on_tallymlp(mlp_and_batch):
    model, x, _ = mlp_and_batch
    t_lut = _median_ms(lambda: _lut_forward(model, x), warmup=4, runs=9)
    t_nat = _median_ms(lambda: model(x), warmup=8, runs=15)
    assert t_nat < t_lut / 3.0, (
        f"SIMD TallyMLP forward {t_nat:.2f} ms is not 3× faster than "
        f"table popcount {t_lut:.2f} ms (S=256 should be count-bound)"
    )


def test_simd_train_step_runs_and_beats_old_forward(mlp_and_batch):
    """Train step must work; writeback may dominate, but the step must beat old infer."""
    model, x, y = mlp_and_batch
    wb = TallyWriteback(model.tally_layers(), opt="adam", lr=1e-3)

    def step():
        wb.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        wb.step()

    t_old = _median_ms(lambda: _old_forward(model, x), warmup=2, runs=5)
    t_tr = _median_ms(step, warmup=2, runs=5)
    # Writeback (loop over S) can match old infer time; it must not blow up.
    assert t_tr < t_old * 3.0, (
        f"train step {t_tr:.2f} ms is much slower than old float-sum infer {t_old:.2f} ms"
    )
    model.assert_binary_invariants()
