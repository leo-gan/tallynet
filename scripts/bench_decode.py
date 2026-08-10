"""Time packed popcount vs the old float-sum, and time the current TallyMLP."""

from __future__ import annotations

import statistics
import time

import torch
import torch.nn.functional as F

from tallynet.encoders import encode_tally
from tallynet.kernels import decode_tally_weight
from tallynet.layers import TallyLinear
from tallynet.models import TallyMLP
from tallynet.native import load_native
from tallynet.packed import pack_pm1, popcount_packed, unpack_pm1
from tallynet.writeback import TallyWriteback


def _median_ms(fn, *, warmup: int = 5, runs: int = 15) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def _old_mlp_forward(model: TallyMLP, x: torch.Tensor) -> torch.Tensor:
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


def _lut_mlp_forward(model: TallyMLP, x: torch.Tensor) -> torch.Tensor:
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


def main() -> None:
    load_native(verbose=True)
    torch.manual_seed(0)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))

    print("\n# Isolated decode, 1024×1024 groups")
    for S in (8, 64, 256):
        pm1 = torch.randint(0, 2, (1024, 1024, S), dtype=torch.int8) * 2 - 1
        packed = pack_pm1(pm1)
        t_old = _median_ms(lambda: pm1.float().sum(-1), warmup=3, runs=12)
        t_nat = _median_ms(lambda: popcount_packed(packed, S, native=True), warmup=3, runs=12)
        t_lut = _median_ms(lambda: popcount_packed(packed, S, native=False), warmup=3, runs=12)
        print(
            f"S={S:3d}  float_sum={t_old:7.2f} ms  native={t_nat:7.2f} ms  "
            f"lut={t_lut:7.2f} ms  pack={packed.numel() / pm1.numel():.3f}× bytes"
        )

    layer = TallyLinear(1024, 1024, tally_width=8, compute="float32")
    x = torch.randn(32, 1024)
    t_layer = _median_ms(lambda: layer(x), warmup=5, runs=20)
    print(f"TallyLinear 1024×1024 S=8 batch=32: {t_layer:.2f} ms")

    print("\n# Current net: TallyMLP 784→128→10 (MNIST demo shape)")
    for hidden, S, batch, label in (
        (128, 256, 128, "default"),
        (128, 8, 128, "S=8"),
        (512, 256, 128, "wider"),
    ):
        model = TallyMLP(hidden_dim=hidden, tally_width=S, encoder="majority", ln_mode="none")
        xb = torch.randn(batch, 1, 28, 28)
        yb = torch.randint(0, 10, (batch,))
        t_old = _median_ms(lambda: _old_mlp_forward(model, xb), warmup=2, runs=7)
        t_lut = _median_ms(lambda: _lut_mlp_forward(model, xb), warmup=3, runs=9)
        t_nat = _median_ms(lambda: model(xb), warmup=6, runs=15)
        print(
            f"{label:8s} hidden={hidden} S={S} batch={batch}  "
            f"old={t_old:7.2f}  lut={t_lut:7.2f}  simd={t_nat:7.2f} ms  "
            f"({t_old / t_nat:.0f}× vs old, {t_lut / t_nat:.1f}× vs lut)"
        )
        wb = TallyWriteback(model.tally_layers(), opt="adam", lr=1e-3)

        def step() -> None:
            wb.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            wb.step()

        t_tr = _median_ms(step, warmup=2, runs=7)
        print(f"{'':8s} train+writeback {t_tr:7.2f} ms")


if __name__ == "__main__":
    main()
