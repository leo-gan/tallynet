"""Optional dataset helpers (requires torchvision extra)."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader


def mnist_loaders(
    data_dir: str | Path = "data",
    batch_size: int = 128,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    try:
        from torchvision import datasets, transforms
    except ImportError as e:
        raise ImportError(
            "MNIST loaders require torchvision. Install with: "
            "pip install 'tallynet[train]' or uv sync --extra train"
        ) from e

    data_dir = Path(data_dir)
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_ds = datasets.MNIST(str(data_dir), train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(str(data_dir), train=False, download=True, transform=tfm)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


@torch.no_grad()
def accuracy(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += y.numel()
    model.train()
    return correct / max(1, total)
