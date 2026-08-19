"""Dataset helpers. Skip when torchvision or local files are missing."""

from __future__ import annotations

from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _has_mnist() -> bool:
    return (DATA_DIR / "MNIST" / "raw" / "train-images-idx3-ubyte").is_file()


def _has_cifar10() -> bool:
    return (DATA_DIR / "cifar-10-batches-py" / "data_batch_1").is_file()


def test_mnist_loaders_shapes():
    pytest.importorskip("torchvision")
    if not _has_mnist():
        pytest.skip("MNIST not in data/")
    from tallynet.data import mnist_loaders

    train, test = mnist_loaders(DATA_DIR, batch_size=8)
    x, y = next(iter(train))
    assert x.shape == (8, 1, 28, 28)
    assert y.shape == (8,)
    x_t, y_t = next(iter(test))
    assert x_t.shape[1:] == (1, 28, 28)
    assert y_t.ndim == 1


def test_cifar10_loaders_shapes():
    pytest.importorskip("torchvision")
    if not _has_cifar10():
        pytest.skip("CIFAR-10 not in data/")
    from tallynet.data import cifar10_loaders

    train, test = cifar10_loaders(DATA_DIR, batch_size=8)
    x, y = next(iter(train))
    assert x.shape == (8, 3, 32, 32)
    assert y.shape == (8,)
    x_t, y_t = next(iter(test))
    assert x_t.shape[1:] == (3, 32, 32)
    assert y_t.ndim == 1
