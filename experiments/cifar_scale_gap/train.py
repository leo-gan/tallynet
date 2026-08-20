#!/usr/bin/env python3
"""Iso-shape CIFAR-10: float MLP vs TallyMLP, same H. Test if the gap shrinks.

The starting ladder is not the answer. After a run the script prints
EXPAND H / EXPAND S / STOP. Use --expand-h / --expand-s to add the next
rung; finished cells in the latest CSV are skipped.

    uv run --extra train python experiments/cifar_scale_gap/train.py --quick
    uv run --extra train python experiments/cifar_scale_gap/train.py --scale
    uv run --extra train python experiments/cifar_scale_gap/train.py --expand-h
    uv run --extra train python experiments/cifar_scale_gap/train.py --expand-s
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn.functional as F

from tallynet.budget import float_mlp_budget, tally_mlp_budget
from tallynet.data import accuracy, cifar10_loaders
from tallynet.models import FloatMLP, TallyMLP
from tallynet.native import load_native, status as native_status
from tallynet.repro import resolved_args, seed_all, write_manifest
from tallynet.writeback import TallyWriteback

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[1]
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from analysis import DELTA_PP, PRIMARY_S, SCALE_H, decide_expand  # noqa: E402

CIFAR_IN_DIM = 32 * 32 * 3
N_CLASSES = 10
EXPERIMENT = "cifar_scale_gap"
LATEST_NAME = "latest.csv"


@dataclass
class RunResult:
    arm: str
    hidden_dim: int
    tally_width: int
    depth: int
    seed: int
    epochs: int
    batch_size: int
    best_test_acc: float
    final_test_acc: float
    final_train_acc: float
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


def _cell_key(r: RunResult) -> tuple:
    return (r.arm, r.hidden_dim, r.tally_width, r.seed, r.epochs)


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
) -> tuple[float, float, float, float, float]:
    seed_all(seed)
    model = FloatMLP(
        hidden_dim=hidden_dim,
        depth=depth,
        in_dim=CIFAR_IN_DIM,
        n_classes=N_CLASSES,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = 0.0
    final_acc = 0.0
    final_loss = 0.0
    final_train_acc = 0.0
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        correct = 0
        n = 0
        last = epoch == epochs - 1
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * y.size(0)
            n += y.size(0)
            if last:
                correct += int((logits.argmax(dim=1) == y).sum().item())
        final_loss = loss_sum / max(1, n)
        if last:
            final_train_acc = correct / max(1, n)
        final_acc = accuracy(model, test_loader, device)
        best = max(best, final_acc)
    return best, final_acc, final_train_acc, final_loss, time.perf_counter() - t0


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
) -> tuple[float, float, float, float, float]:
    seed_all(seed)
    model = TallyMLP(
        hidden_dim=hidden_dim,
        tally_width=tally_width,
        encoder=encoder,  # type: ignore[arg-type]
        ln_mode="none",
        depth=depth,
        in_dim=CIFAR_IN_DIM,
        n_classes=N_CLASSES,
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
    final_train_acc = 0.0
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        correct = 0
        n = 0
        last = epoch == epochs - 1
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            wb.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            wb.step()
            loss_sum += float(loss.item()) * y.size(0)
            n += y.size(0)
            if last:
                correct += int((logits.argmax(dim=1) == y).sum().item())
        final_loss = loss_sum / max(1, n)
        if last:
            final_train_acc = correct / max(1, n)
        final_acc = accuracy(model, test_loader, device)
        model.assert_binary_invariants()
        best = max(best, final_acc)
    return best, final_acc, final_train_acc, final_loss, time.perf_counter() - t0


def _print_plan(rows: list[dict]) -> None:
    print("\n# Iso-shape configs (budget logged, not matched)")
    print(
        f"{'H':>6}  {'S':>4}  {'nn_params':>10}  {'tally_bits':>12}  "
        f"{'float_kb':>10}  {'tally_kb':>10}"
    )
    for r in rows:
        print(
            f"{r['H']:6d}  {r['S']:4d}  {r['nn_params']:10d}  {r['tally_bits']:12d}  "
            f"{r['nn_b'].total/1024:10.1f}  {r['t_b'].total/1024:10.1f}"
        )


def _load_csv(path: Path) -> list[RunResult]:
    if not path.is_file():
        return []
    out: list[RunResult] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            kwargs = {}
            for fld in fields(RunResult):
                raw = row[fld.name]
                typ = fld.type
                if typ in (int, "int"):
                    kwargs[fld.name] = int(float(raw))
                elif typ in (float, "float"):
                    kwargs[fld.name] = float(raw)
                else:
                    kwargs[fld.name] = raw
            out.append(RunResult(**kwargs))
    return out


def _write_csv(path: Path, rows: list[RunResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [fld.name for fld in fields(RunResult)]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    tmp.replace(path)


def _checkpoint(latest: Path, rows: list[RunResult]) -> None:
    """Flush after every cell so a killed run can resume."""
    _write_csv(latest, rows)


def _means(results: list[RunResult]) -> dict[tuple, float]:
    groups: dict[tuple, list[float]] = {}
    for r in results:
        groups.setdefault((r.arm, r.hidden_dim, r.tally_width), []).append(
            r.best_test_acc
        )
    return {k: statistics.mean(v) for k, v in groups.items()}


def _tables(means: dict[tuple, float]) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    float_by_h = {h: acc for (arm, h, _s), acc in means.items() if arm == "float"}
    tally_by_hs = {
        (h, s): acc
        for (arm, h, s), acc in means.items()
        if arm.startswith("tally")
    }
    return float_by_h, tally_by_hs


def _print_summary(results: list[RunResult], delta_pp: float) -> None:
    print("\n# Summary (mean best_test_acc ± std over seeds)")
    print(
        f"{'arm':<12}  {'H':>5}  {'S':>4}  "
        f"{'acc_mean':>9}  {'acc_std':>8}  {'train_loss':>10}  "
        f"{'train_acc':>9}  {'params':>12}  {'budget_kb':>10}"
    )
    groups: dict[tuple, list[RunResult]] = {}
    for r in results:
        groups.setdefault((r.arm, r.hidden_dim, r.tally_width), []).append(r)
    means = {}
    for k, rs in sorted(groups.items()):
        accs = [r.best_test_acc for r in rs]
        mean = statistics.mean(accs)
        means[k] = mean
        std = statistics.stdev(accs) if len(accs) > 1 else 0.0
        r0 = rs[0]
        tr_loss = statistics.mean(r.final_train_loss for r in rs)
        tr_acc = statistics.mean(r.final_train_acc for r in rs)
        print(
            f"{r0.arm:<12}  {r0.hidden_dim:5d}  {r0.tally_width:4d}  "
            f"{mean:9.4f}  {std:8.4f}  {tr_loss:10.4f}  {tr_acc:9.4f}  "
            f"{r0.n_params:12d}  {r0.budget_total/1024:10.1f}"
        )

    float_by_h, _tally = _tables(means)
    tally_arms = sorted({k[0] for k in means if k[0].startswith("tally")})
    if float_by_h and tally_arms:
        print("\n# Gap vs float (tally_acc - float_acc, percentage points)")
        print(
            f"{'arm':<12}  {'H':>5}  {'float_acc':>10}  {'tally_acc':>10}  "
            f"{'gap_pp':>8}  {'Δgap_pp':>8}  {'Δtally_pp':>10}"
        )
        for arm in tally_arms:
            prev_gap: float | None = None
            prev_tally: float | None = None
            rows_arm = sorted((k for k in means if k[0] == arm), key=lambda k: k[1])
            for (_arm, h, _s) in rows_arm:
                if h not in float_by_h:
                    continue
                f_acc = float_by_h[h]
                t_acc = means[(_arm, h, _s)]
                gap_pp = (t_acc - f_acc) * 100.0
                dgap = "" if prev_gap is None else f"{gap_pp - prev_gap:+.2f}"
                dt = "" if prev_tally is None else f"{(t_acc - prev_tally) * 100:+.2f}"
                print(
                    f"{arm:<12}  {h:5d}  {f_acc:10.4f}  {t_acc:10.4f}  "
                    f"{gap_pp:8.2f}  {dgap:>8}  {dt:>10}"
                )
                prev_gap = gap_pp
                prev_tally = t_acc

    longest = max((r.seconds for r in results), default=0.0)
    decision = decide_expand(
        float_by_h=float_by_h,
        tally_by_hs={
            (h, s): acc
            for (arm, h, s), acc in means.items()
            if arm.startswith("tally")
        },
        delta_pp=delta_pp,
        longest_cell_sec=longest,
    )
    print()
    for line in decision.summary_lines():
        print(line)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--out-dir", type=Path, default=EXP_DIR / "results")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--h-values", type=int, nargs="+", default=[128, 256])
    p.add_argument("--s-values", type=int, nargs="+", default=[PRIMARY_S])
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--encoder", type=str, default="majority")
    p.add_argument("--decoder", type=str, default="density")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--scale",
        action="store_true",
        help="Starting width ladder H=128..2048 at S=32",
    )
    p.add_argument(
        "--expand-h",
        action="store_true",
        help="Next H rung from the latest CSV (no-op if STOP)",
    )
    p.add_argument(
        "--expand-s",
        action="store_true",
        help="Next S rung / S-cross from the latest CSV (no-op if STOP)",
    )
    p.add_argument("--delta-pp", type=float, default=DELTA_PP)
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-run cells even if they are already in the latest CSV",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="H=[128], S=[32], seeds=[0], epochs=2",
    )
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = p.parse_args(argv)

    latest = args.out_dir / LATEST_NAME
    prior = [] if args.no_resume else _load_csv(latest)
    pairs: list[tuple[int, int]] | None = None

    if args.quick:
        args.h_values = [128]
        args.s_values = [PRIMARY_S]
        args.seeds = [0]
        args.epochs = 2
    elif args.scale:
        args.h_values = list(SCALE_H)
        args.s_values = [PRIMARY_S]
    elif args.expand_h or args.expand_s:
        if not prior:
            raise SystemExit(f"no results at {latest}; run --scale first")
        means = _means(prior)
        float_by_h, tally_by_hs = _tables(means)
        longest = max((r.seconds for r in prior), default=0.0)
        decision = decide_expand(
            float_by_h=float_by_h,
            tally_by_hs=tally_by_hs,
            delta_pp=args.delta_pp,
            longest_cell_sec=longest,
        )
        print("\n".join(decision.summary_lines()))
        wanted: list[tuple[int, int]] = []
        if args.expand_h:
            if not decision.expand_h or decision.next_h is None:
                print("no H rung to run")
            else:
                wanted.append((decision.next_h, PRIMARY_S))
        if args.expand_s:
            if not decision.expand_s or not decision.next_s_values or decision.s_h is None:
                print("no S rung to run")
            else:
                wanted.extend((decision.s_h, s) for s in decision.next_s_values)
        if not wanted:
            return
        pairs = wanted

    device = torch.device(args.device)
    load_native()
    nat = native_status()
    print(
        f"# cifar_scale_gap  device={device}  native_cpu={nat['cpu']}  "
        f"epochs={args.epochs}  batch={args.batch_size}  seeds={args.seeds}"
    )

    train_loader, test_loader = cifar10_loaders(args.data_dir, batch_size=args.batch_size)

    if pairs is None:
        pairs = [(h, s) for h in args.h_values for s in args.s_values]

    plan: list[dict] = []
    for h, S in pairs:
        nn_b = float_mlp_budget(
            hidden_dim=h,
            batch_size=args.batch_size,
            depth=args.depth,
            in_dim=CIFAR_IN_DIM,
            n_classes=N_CLASSES,
        )
        t_b = tally_mlp_budget(
            hidden_dim=h,
            tally_width=S,
            batch_size=args.batch_size,
            depth=args.depth,
            in_dim=CIFAR_IN_DIM,
            n_classes=N_CLASSES,
        )
        plan.append(
            {
                "H": h,
                "S": S,
                "nn_params": nn_b.n_params,
                "tally_bits": t_b.n_params,
                "nn_b": nn_b,
                "t_b": t_b,
            }
        )
    _print_plan(plan)

    results: list[RunResult] = list(prior)
    done = {_cell_key(r) for r in results}
    native_flag = 1 if nat["cpu"] else 0
    done_float: set[tuple[int, int, int]] = set()

    for item in plan:
        h = item["H"]
        nn_b = item["nn_b"]
        for seed in args.seeds:
            fk = ("float", h, 0, seed, args.epochs)
            if (h, seed, args.epochs) not in done_float and fk not in done:
                print(f"\n>> float  H={h}  seed={seed}")
                best, final, tr_acc, loss, secs = train_float(
                    hidden_dim=h,
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
                        hidden_dim=h,
                        tally_width=0,
                        depth=args.depth,
                        seed=seed,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        best_test_acc=best,
                        final_test_acc=final,
                        final_train_acc=tr_acc,
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
                done.add(fk)
                _checkpoint(latest, results)
                print(
                    f"   best_acc={best:.4f}  final_acc={final:.4f}  "
                    f"train_acc={tr_acc:.4f}  loss={loss:.4f}  {secs:.1f}s"
                )
            elif fk in done:
                print(f"\n>> float  H={h}  seed={seed}  (skip, already in CSV)")
            done_float.add((h, seed, args.epochs))

            S = item["S"]
            t_b = item["t_b"]
            tk = (f"tally_S{S}", h, S, seed, args.epochs)
            if tk in done:
                print(f"\n>> tally  H={h}  S={S}  seed={seed}  (skip, already in CSV)")
                continue
            print(f"\n>> tally  H={h}  S={S}  seed={seed}")
            best, final, tr_acc, loss, secs = train_tally(
                hidden_dim=h,
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
                    hidden_dim=h,
                    tally_width=S,
                    depth=args.depth,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    best_test_acc=best,
                    final_test_acc=final,
                    final_train_acc=tr_acc,
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
            done.add(tk)
            _checkpoint(latest, results)
            print(
                f"   best_acc={best:.4f}  final_acc={final:.4f}  "
                f"train_acc={tr_acc:.4f}  loss={loss:.4f}  {secs:.1f}s"
            )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_csv = args.out_dir / f"{stamp}.csv"
    _write_csv(out_csv, results)
    _write_csv(latest, results)
    extra = {"csv": out_csv.name, "n_rows": len(results)}
    write_manifest(
        args.out_dir / f"{stamp}.manifest.json",
        experiment=EXPERIMENT,
        argv=raw_argv,
        resolved=resolved_args(args),
        extra=extra,
    )
    write_manifest(
        args.out_dir / "latest.manifest.json",
        experiment=EXPERIMENT,
        argv=raw_argv,
        resolved=resolved_args(args),
        extra=extra,
    )

    _print_summary(results, args.delta_pp)
    print(f"\nwrote {out_csv}")
    print(f"wrote {latest}")
    print(f"wrote {args.out_dir / 'latest.manifest.json'}")


if __name__ == "__main__":
    main()
