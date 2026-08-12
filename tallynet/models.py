"""Small TallyNet models (architecture demos)."""

from __future__ import annotations

from typing import List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from tallynet.encoders import EncoderName
from tallynet.layers import TallyLinear

LNMode = Literal["none", "no_affine", "affine"]
ActName = Literal["relu", "squared_relu"]


class SquaredReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x).square()


def _make_ln(num_features: int, mode: LNMode) -> nn.Module:
    if mode == "none":
        return nn.Identity()
    if mode == "no_affine":
        return nn.LayerNorm(num_features, elementwise_affine=False)
    if mode == "affine":
        return nn.LayerNorm(num_features, elementwise_affine=True)
    raise ValueError(f"Unknown ln_mode: {mode}")


class FloatMLP(nn.Module):
    """Flatten → Linear → Act → … → Linear → logits (no bias).

    Topology mirrors ``TallyMLP`` so iso-memory experiments share depth/width axes.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        depth: int = 1,
        activation: ActName = "relu",
        in_dim: int = 28 * 28,
        n_classes: int = 10,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.hidden_dim = hidden_dim
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.flatten = nn.Flatten()
        self.act: nn.Module = (
            SquaredReLU() if activation == "squared_relu" else nn.ReLU()
        )
        self.linears = nn.ModuleList()
        d = in_dim
        for _ in range(depth):
            self.linears.append(nn.Linear(d, hidden_dim, bias=False))
            d = hidden_dim
        self.linears.append(nn.Linear(d, n_classes, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        n = len(self.linears)
        for i, linear in enumerate(self.linears):
            x = linear(x)
            if i < n - 1:
                x = self.act(x)
        return x


class TallyMLP(nn.Module):
    """Flatten → [LN] → TallyLinear → Act → … → TallyLinear → logits.

    Default shape matches the MNIST toy used in the parent research line:
    784 → hidden → 10 with tally-coded linears.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        tally_width: int = 256,
        encoder: EncoderName = "fixed",
        tanh_tau: float = 0.0,
        ln_mode: LNMode = "none",
        depth: int = 1,
        activation: ActName = "relu",
        in_dim: int = 28 * 28,
        n_classes: int = 10,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.hidden_dim = hidden_dim
        self.tally_width = tally_width
        self.encoder = encoder
        self.ln_mode = ln_mode
        self.in_dim = in_dim
        self.n_classes = n_classes

        self.flatten = nn.Flatten()
        self.act: nn.Module = (
            SquaredReLU() if activation == "squared_relu" else nn.ReLU()
        )
        self.lns = nn.ModuleList()
        self.linears = nn.ModuleList()

        d = in_dim
        for _ in range(depth):
            self.lns.append(_make_ln(d, ln_mode))
            self.linears.append(
                TallyLinear(
                    d,
                    hidden_dim,
                    tally_width=tally_width,
                    encoder=encoder,
                    tanh_tau=tanh_tau,
                )
            )
            d = hidden_dim
        self.lns.append(_make_ln(d, ln_mode))
        self.linears.append(
            TallyLinear(
                d,
                n_classes,
                tally_width=tally_width,
                encoder=encoder,
                tanh_tau=tanh_tau,
            )
        )

    def tally_layers(self) -> List[TallyLinear]:
        return [m for m in self.linears if isinstance(m, TallyLinear)]

    def ln_parameters(self) -> List[nn.Parameter]:
        return [p for ln in self.lns for p in ln.parameters()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        n = len(self.linears)
        for i, (ln, linear) in enumerate(zip(self.lns, self.linears)):
            x = ln(x)
            x = linear(x)
            if i < n - 1:
                x = self.act(x)
        return x

    @torch.no_grad()
    def assert_binary_invariants(self) -> None:
        for layer in self.tally_layers():
            p = layer.bits
            assert p.dtype == torch.uint8, p.dtype
            assert p.shape[-1] == layer.nbytes
            layer.enforce_binary_()
            pm1 = layer.bits_pm1()
            uniq = set(torch.unique(pm1).tolist())
            assert uniq.issubset({-1, 1}), uniq
