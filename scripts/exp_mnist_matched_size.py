#!/usr/bin/env python3
"""Iso-memory MNIST: float MLP vs TallyMLP under matched training footprint.

Decisions (autonomous defaults):
- Match **analytic** training bytes: weights + grads + Adam + activations.
- Spend Tally memory savings on **width** (max hidden under budget); grid S.
- float32 baseline only; depth=1; no bias; no LayerNorm.
- Outputs CSV under artifacts/experiments/.

    uv run --extra train python scripts/exp_mnist_matched_size.py
    uv run --extra train python scripts/exp_mnist_matched_size.py --scale
    uv run --extra train python scripts/exp_mnist_matched_size.py --quick
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from tallynet.budget import (
    float_mlp_budget,
    max_tally_hidden_under_budget,
    tally_mlp_budget,
)
from tallynet.data import accuracy, mnist_loaders
from tallynet.models import FloatMLP, TallyMLP
from tallynet.native import load_native, status as native_status
from tallynet.writeback import TallyWriteback

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RunResult:
    arm: str
    h_nn_budget: int
    hidden_dim: int
    tally_width: int
    depth: int
    seed: int
    epochs: int
    batch_size: int
    best_test_acc: float
    final_test_acc: float
    final_train_loss: float
    budget_total: int
    budget_weights: int
    budget_grads: int
    budget_optimizer: int
    budget_activations: int
    n_params: int
    n_groups: int
    seconds: float
    native_cpu: int


def train_float(
    *,
    hidden_dim: int,
    depth: int,
    train_loader,
    test_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[float, float, float, float]:
    torch.manual_seed(seed)
    model = FloatMLP(hidden_dim=hidden_dim, depth=depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = 0.0
    final_acc = 0.0
    final_loss = 0.0
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        loss_sum = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * y.size(0)
            n += y.size(0)
        final_loss = loss_sum / max(1, n)
        final_acc = accuracy(model, test_loader, device)
        best = max(best, final_acc)
    return best, final_acc, final_loss, time.perf_counter() - t0


def train_tally(
    *,
    hidden_dim: int,
    tally_width: int,
    depth: int,
    train_loader,
    test_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    seed: int,
    encoder: str,
    decoder: str,
) -> tuple[float, float, float, float]:
    torch.manual_seed(seed)
    model = TallyMLP(
        hidden_dim=hidden_dim,
        tally_width=tally_width,
        encoder=encoder,  # type: ignore[arg-type]
        ln_mode="none",
        depth=depth,
    ).to(device)
    wb = TallyWriteback(
        model.tally_layers(),
        opt="adam",
        lr=lr,
        decoder=decoder,  # type: ignore[arg-type]
    )
    best = 0.0
    final_acc = 0.0
    final_loss = 0.0
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        loss_sum = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            wb.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            wb.step()
            loss_sum += float(loss.item()) * y.size(0)
            n += y.size(0)
        final_loss = loss_sum / max(1, n)
        final_acc = accuracy(model, test_loader, device)
        model.assert_binary_invariants()
        best = max(best, final_acc)
    return best, final_acc, final_loss, time.perf_counter() - t0


def _print_plan(rows: list[dict]) -> None:
    print("\n# Size-matched configs")
    print(
        f"{'h_nn':>6}  {'budget_kb':>10}  {'S':>4}  {'h_t':>6}  "
        f"{'tally_kb':>10}  {'nn_params':>10}  {'tally_bits':>12}  {'bit/param':>9}"
    )
    for r in rows:
        print(
            f"{r['h_nn']:6d}  {r['budget_total']/1024:10.1f}  {r['S']:4d}  {r['h_t']:6d}  "
            f"{r['tally_total']/1024:10.1f}  {r['nn_params']:10d}  {r['tally_bits']:12d}  "
            f"{r['tally_bits']/max(1,r['nn_params']):9.1f}"
        )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "experiments")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--h-nn", type=int, nargs="+", default=[64, 128])
    p.add_argument("--s-values", type=int, nargs="+", default=[8, 32, 256])
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--encoder", type=str, default="majority")
    p.add_argument("--decoder", type=str, default="density")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--scale",
        action="store_true",
        help="Budget ladder H_nn=[64,128,256,512,1024], S=[8,32] (find gap plateau)",
    )
    p.add_argument(
        "--gap-plateau-pp",
        type=float,
        default=0.3,
        help="Gap change (percentage points) below this counts as plateau",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="H_nn=[64], S=[8,256], seeds=[0], epochs=2",
    )
    args = p.parse_args(argv)

    if args.quick:
        args.h_nn = [64]
        args.s_values = [8, 256]
        args.seeds = [0]
        args.epochs = 2
    elif args.scale:
        args.h_nn = [64, 128, 256, 512, 1024]
        args.s_values = [8, 32]

    device = torch.device(args.device)
    load_native()
    nat = native_status()
    print(
        f"# exp_mnist_matched_size  device={device}  native_cpu={nat['cpu']}  "
        f"epochs={args.epochs}  batch={args.batch_size}  seeds={args.seeds}"
    )

    train_loader, test_loader = mnist_loaders(args.data_dir, batch_size=args.batch_size)

    plan: list[dict] = []
    for h_nn in args.h_nn:
        nn_b = float_mlp_budget(
            hidden_dim=h_nn, batch_size=args.batch_size, depth=args.depth
        )
        for S in args.s_values:
            h_t = max_tally_hidden_under_budget(
                nn_b.total,
                tally_width=S,
                batch_size=args.batch_size,
                depth=args.depth,
            )
            if h_t is None:
                print(f"warning: no tally hidden fits under budget for H_nn={h_nn} S={S}")
                continue
            t_b = tally_mlp_budget(
                hidden_dim=h_t,
                tally_width=S,
                batch_size=args.batch_size,
                depth=args.depth,
            )
            plan.append(
                {
                    "h_nn": h_nn,
                    "S": S,
                    "h_t": h_t,
                    "budget_total": nn_b.total,
                    "tally_total": t_b.total,
                    "nn_params": nn_b.n_params,
                    "tally_bits": t_b.n_params,
                    "nn_budget": nn_b,
                    "tally_budget": t_b,
                }
            )
    _print_plan(plan)

    results: list[RunResult] = []
    native_flag = 1 if nat["cpu"] else 0

    # Unique float baselines per (h_nn, seed)
    done_float: set[tuple[int, int]] = set()

    for item in plan:
        h_nn = item["h_nn"]
        nn_b = item["nn_budget"]
        for seed in args.seeds:
            key = (h_nn, seed)
            if key not in done_float:
                print(f"\n>> float  H={h_nn}  seed={seed}")
                best, final, loss, secs = train_float(
                    hidden_dim=h_nn,
                    depth=args.depth,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    device=device,
                    epochs=args.epochs,
                    lr=args.lr,
                    seed=seed,
                )
                results.append(
                    RunResult(
                        arm="float",
                        h_nn_budget=h_nn,
                        hidden_dim=h_nn,
                        tally_width=0,
                        depth=args.depth,
                        seed=seed,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        best_test_acc=best,
                        final_test_acc=final,
                        final_train_loss=loss,
                        budget_total=nn_b.total,
                        budget_weights=nn_b.weights,
                        budget_grads=nn_b.grads,
                        budget_optimizer=nn_b.optimizer,
                        budget_activations=nn_b.activations,
                        n_params=nn_b.n_params,
                        n_groups=nn_b.n_groups,
                        seconds=secs,
                        native_cpu=native_flag,
                    )
                )
                print(
                    f"   best_acc={best:.4f}  final_acc={final:.4f}  "
                    f"loss={loss:.4f}  {secs:.1f}s"
                )
                done_float.add(key)

            h_t = item["h_t"]
            S = item["S"]
            t_b = item["tally_budget"]
            print(f"\n>> tally  H_nn_budget={h_nn}  H_t={h_t}  S={S}  seed={seed}")
            best, final, loss, secs = train_tally(
                hidden_dim=h_t,
                tally_width=S,
                depth=args.depth,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                seed=seed,
                encoder=args.encoder,
                decoder=args.decoder,
            )
            results.append(
                RunResult(
                    arm=f"tally_S{S}",
                    h_nn_budget=h_nn,
                    hidden_dim=h_t,
                    tally_width=S,
                    depth=args.depth,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    best_test_acc=best,
                    final_test_acc=final,
                    final_train_loss=loss,
                    budget_total=t_b.total,
                    budget_weights=t_b.weights,
                    budget_grads=t_b.grads,
                    budget_optimizer=t_b.optimizer,
                    budget_activations=t_b.activations,
                    n_params=t_b.n_params,
                    n_groups=t_b.n_groups,
                    seconds=secs,
                    native_cpu=native_flag,
                )
            )
            print(
                f"   best_acc={best:.4f}  final_acc={final:.4f}  "
                f"loss={loss:.4f}  {secs:.1f}s"
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_csv = args.out_dir / f"mnist_matched_size_{stamp}.csv"
    fieldnames = list(asdict(results[0]).keys()) if results else []
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))
    # also write a stable latest path
    latest = args.out_dir / "mnist_matched_size_latest.csv"
    with latest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))

    print("\n# Summary (mean best_test_acc ± std over seeds)")
    print(
        f"{'arm':<12}  {'h_nn':>5}  {'H':>5}  {'S':>4}  "
        f"{'acc_mean':>9}  {'acc_std':>8}  {'params':>12}  {'budget_kb':>10}"
    )
    # group by arm + h_nn_budget + hidden + S
    groups: dict[tuple, list[RunResult]] = {}
    for r in results:
        k = (r.arm, r.h_nn_budget, r.hidden_dim, r.tally_width)
        groups.setdefault(k, []).append(r)
    means: dict[tuple, float] = {}
    for k, rs in sorted(groups.items()):
        accs = [r.best_test_acc for r in rs]
        mean = statistics.mean(accs)
        means[k] = mean
        std = statistics.stdev(accs) if len(accs) > 1 else 0.0
        r0 = rs[0]
        print(
            f"{r0.arm:<12}  {r0.h_nn_budget:5d}  {r0.hidden_dim:5d}  {r0.tally_width:4d}  "
            f"{mean:9.4f}  {std:8.4f}  {r0.n_params:12d}  {r0.budget_total/1024:10.1f}"
        )

    # Gap vs float by budget: does the tally deficit keep shrinking?
    float_by_h: dict[int, float] = {}
    for (arm, h_nn, _hid, _s), mean in means.items():
        if arm == "float":
            float_by_h[h_nn] = mean

    tally_arms = sorted({k[0] for k in means if k[0].startswith("tally")})
    if float_by_h and tally_arms:
        print("\n# Gap vs float (tally_acc - float_acc, percentage points)")
        print(
            f"{'arm':<12}  {'h_nn':>5}  {'float_acc':>10}  {'tally_acc':>10}  "
            f"{'gap_pp':>8}  {'Δgap_pp':>8}"
        )
        for arm in tally_arms:
            prev_gap: float | None = None
            plateau_at: int | None = None
            rows_arm = sorted(
                (k for k in means if k[0] == arm),
                key=lambda k: k[1],
            )
            for (_arm, h_nn, _hid, _s) in rows_arm:
                if h_nn not in float_by_h:
                    continue
                f_acc = float_by_h[h_nn]
                t_acc = means[(_arm, h_nn, _hid, _s)]
                gap_pp = (t_acc - f_acc) * 100.0
                dgap = "" if prev_gap is None else f"{gap_pp - prev_gap:+.2f}"
                if (
                    prev_gap is not None
                    and plateau_at is None
                    and abs(gap_pp - prev_gap) < args.gap_plateau_pp
                ):
                    plateau_at = h_nn
                print(
                    f"{arm:<12}  {h_nn:5d}  {f_acc:10.4f}  {t_acc:10.4f}  "
                    f"{gap_pp:8.2f}  {dgap:>8}"
                )
                prev_gap = gap_pp
            if plateau_at is not None:
                print(
                    f"  → {arm}: gap change < {args.gap_plateau_pp:.2f} pp "
                    f"starting at H_nn={plateau_at} (plateau / stop improving)"
                )
            else:
                print(
                    f"  → {arm}: no plateau under threshold "
                    f"{args.gap_plateau_pp:.2f} pp within this ladder"
                )

    print(f"\nwrote {out_csv}")
    print(f"wrote {latest}")


if __name__ == "__main__":
    main()
