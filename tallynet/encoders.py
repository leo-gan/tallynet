"""Tally encoders: map bit-bag tally (sum / count) → scalar weight."""

from __future__ import annotations

from typing import Literal

import torch

EncoderName = Literal["fixed", "tanh", "signed_sqrt", "majority"]


def encode_tally(
    s: torch.Tensor,
    tally_width: int,
    encoder: EncoderName = "fixed",
    tanh_tau: float = 0.0,
) -> torch.Tensor:
    """Map sum ``s = Σ a_k`` over a bag of ``tally_width`` ±1 bits to a weight.

    The encoder input is the tally only (equivalently: popcount of +1 bits).
    Bit positions have no place value.
    """
    S = float(tally_width)
    if encoder == "fixed":
        return s / S
    if encoder == "tanh":
        tau = tanh_tau if tanh_tau > 0 else max(S / 2.0, 1.0)
        return torch.tanh(s / tau)
    if encoder == "signed_sqrt":
        return torch.sign(s) * torch.sqrt(s.abs() / S)
    if encoder == "majority":
        w = torch.sign(s)
        return torch.where(w == 0, torch.ones_like(w), w)
    raise ValueError(f"unknown encoder: {encoder}")


def tally_from_bits(bits: torch.Tensor) -> torch.Tensor:
    """Sum of unpacked ±1 bits along the last dimension.

    Prefer ``tallynet.packed.tally_from_packed`` for the storage format.
    """
    return bits.float().sum(dim=-1)
