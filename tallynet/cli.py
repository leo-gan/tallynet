"""CLI entry points."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from tallynet.data import accuracy, mnist_loaders
from tallynet.models import TallyMLP
from tallynet.writeback import TallyWriteback


def train_mnist(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train TallyMLP on MNIST (demo)")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--tally-width", type=int, default=256)
    p.add_argument(
        "--encoder",
        type=str,
        default="majority",
        choices=["fixed", "tanh", "signed_sqrt", "majority"],
    )
    p.add_argument("--opt", type=str, default="adam", choices=["sgd", "sgd_m", "adam"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--decoder", type=str, default="density", choices=["density", "thresholded", "sign_noise"])
    p.add_argument("--p-noise", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_loader, test_loader = mnist_loaders(args.data_dir, batch_size=args.batch_size)
    model = TallyMLP(
        hidden_dim=args.hidden_dim,
        tally_width=args.tally_width,
        encoder=args.encoder,  # type: ignore[arg-type]
        ln_mode="none",
    ).to(device)

    lr = args.lr
    if args.opt in ("sgd", "sgd_m") and lr == 1e-3:
        lr = 0.1  # historical default for SGD cells

    wb = TallyWriteback(
        model.tally_layers(),
        opt=args.opt,  # type: ignore[arg-type]
        lr=lr,
        decoder=args.decoder,  # type: ignore[arg-type]
        p_noise=args.p_noise,
        ln_params=model.ln_parameters() or None,
    )

    best = 0.0
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            wb.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            wb.step()
            loss_sum += float(loss.item()) * y.size(0)
            n += y.size(0)
        train_loss = loss_sum / max(1, n)
        test_acc = accuracy(model, test_loader, device)
        model.assert_binary_invariants()
        best = max(best, test_acc)
        print(
            f"epoch {epoch:3d}  loss={train_loss:.4f}  "
            f"test_acc={test_acc:.4f}  best={best:.4f}  "
            f"flip={wb.last_flip_frac:.4f}"
        )
    print(f"done in {time.perf_counter() - t0:.1f}s  best_test_acc={best:.4f}")


if __name__ == "__main__":
    train_mnist()
