"""Time packed popcount vs the old float-sum of ±1 bits."""

from __future__ import annotations

import time

import torch

from tallynet.layers import TallyLinear
from tallynet.native import load_native
from tallynet.packed import pack_pm1, popcount_packed


def main() -> None:
    load_native(verbose=True)
    torch.manual_seed(0)
    for S in (8, 64, 256):
        pm1 = torch.randint(0, 2, (1024, 1024, S), dtype=torch.int8) * 2 - 1
        packed = pack_pm1(pm1)
        for _ in range(3):
            pm1.float().sum(-1)
            popcount_packed(packed, S, native=True)
            popcount_packed(packed, S, native=False)
        def bench(fn, n=20):
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            return (time.perf_counter() - t0) / n * 1e3
        t_old = bench(lambda: pm1.float().sum(-1))
        t_nat = bench(lambda: popcount_packed(packed, S, native=True))
        t_lut = bench(lambda: popcount_packed(packed, S, native=False))
        print(
            f"S={S:3d}  float_sum={t_old:7.2f} ms  native={t_nat:7.2f} ms  "
            f"lut={t_lut:7.2f} ms  pack={packed.numel()/pm1.numel():.3f}× bytes"
        )
    layer = TallyLinear(1024, 1024, tally_width=8, compute="float32")
    x = torch.randn(32, 1024)
    for _ in range(5):
        layer(x)
    t0 = time.perf_counter()
    for _ in range(30):
        layer(x)
    print(f"TallyLinear 1024×1024 S=8 batch=32: {(time.perf_counter()-t0)/30*1e3:.2f} ms")


if __name__ == "__main__":
    main()
