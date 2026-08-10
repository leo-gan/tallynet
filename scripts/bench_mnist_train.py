"""Compare MNIST training speed: old float-sum kernel vs packed SIMD.

Uses ``data/MNIST`` (copy from binary-optimizers). Same writeback on both
paths; only tally decode changes.

    uv run --extra train python scripts/bench_mnist_train.py
    uv run --extra train python scripts/bench_mnist_train.py --epoch
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from tallynet.data import mnist_loaders
from tallynet.models import TallyMLP
from tallynet.native import load_native
from tallynet.ref_old import old_forward, simd_forward, train_step
from tallynet.writeback import TallyWriteback

ROOT = Path(__file__).resolve().parents[1]


def _median_ms(fn, *, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def _time_epoch(model, wb, loader, forward) -> tuple[float, float]:
    n = 0
    loss_sum = 0.0
    t0 = time.perf_counter()
    for x, y in loader:
        loss_sum += train_step(model, wb, x, y, forward) * y.size(0)
        n += y.size(0)
    return time.perf_counter() - t0, loss_sum / max(1, n)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--epoch", action="store_true", help="also time one full train epoch each")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if load_native() is None:
        raise SystemExit("native SIMD kernel not built (./scripts/build_native.sh)")

    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    train_loader, _ = mnist_loaders(args.data_dir, batch_size=args.batch_size)
    x0, y0 = next(iter(train_loader))

    print(
        f"# MNIST train speed  hidden={args.hidden_dim} batch={args.batch_size}  "
        f"native={load_native() is not None}"
    )
    print(
        f"{'S':>5}  {'old ms/step':>12}  {'SIMD ms/step':>12}  {'speedup':>8}  "
        f"{'old epoch s':>12}  {'SIMD epoch s':>12}"
    )

    for S in (8, 256):
        model = TallyMLP(
            hidden_dim=args.hidden_dim,
            tally_width=S,
            encoder="majority",
            ln_mode="none",
        )
        wb = TallyWriteback(model.tally_layers(), opt="adam", lr=1e-3)
        t_old = _median_ms(
            lambda: train_step(model, wb, x0, y0, old_forward),
            warmup=2,
            runs=args.steps,
        )
        t_simd = _median_ms(
            lambda: train_step(model, wb, x0, y0, simd_forward),
            warmup=2,
            runs=args.steps,
        )
        e_old = e_simd = float("nan")
        if args.epoch:
            m1 = TallyMLP(
                hidden_dim=args.hidden_dim,
                tally_width=S,
                encoder="majority",
                ln_mode="none",
            )
            m2 = TallyMLP(
                hidden_dim=args.hidden_dim,
                tally_width=S,
                encoder="majority",
                ln_mode="none",
            )
            m2.load_state_dict(m1.state_dict())
            wb1 = TallyWriteback(m1.tally_layers(), opt="adam", lr=1e-3)
            wb2 = TallyWriteback(m2.tally_layers(), opt="adam", lr=1e-3)
            e_old, _ = _time_epoch(m1, wb1, train_loader, old_forward)
            e_simd, _ = _time_epoch(m2, wb2, train_loader, simd_forward)
        print(
            f"{S:5d}  {t_old:12.2f}  {t_simd:12.2f}  {t_old / t_simd:7.2f}×  "
            f"{e_old:12.1f}  {e_simd:12.1f}"
        )


if __name__ == "__main__":
    main()
