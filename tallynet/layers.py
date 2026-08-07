"""TallyLinear: tally-coded multi-bit linear layer."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tallynet.encoders import EncoderName, encode_tally


class TallyLinear(nn.Module):
    """Linear layer with tally-coded weights.

    Storage: int8 bit bag ``bits`` of shape ``[out, in, S]`` with values in {±1}.
    Forward: ``w = enc(sum(bits))``, then ``y = x @ (w * gain).T``.

    Gradients attach to a leaf ``w`` (detached from storage) so a continuous
    optimizer can produce Δ on the tally weight; writeback into bits is separate
    (see ``tallynet.writeback``). Storage remains the discrete bit bag — no FP
    master weight tensor is kept as the parameter of record.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tally_width: int = 256,
        encoder: EncoderName = "fixed",
        tanh_tau: float = 0.0,
        weight_scale: bool = True,
    ):
        super().__init__()
        if tally_width < 1:
            raise ValueError(f"tally_width must be >= 1, got {tally_width}")
        self.in_features = in_features
        self.out_features = out_features
        self.tally_width = tally_width
        self.encoder: EncoderName = encoder
        self.tanh_tau = float(tanh_tau)
        gain = 1.0 / math.sqrt(in_features) if weight_scale else 1.0
        self.register_buffer("gain", torch.tensor(gain, dtype=torch.float32))

        init = torch.randint(0, 2, (out_features, in_features, tally_width))
        init = (init * 2 - 1).to(torch.int8)
        self.register_buffer("bits", init)

        self._w: Optional[torch.Tensor] = None

    @property
    def last_weight_grad(self) -> Optional[torch.Tensor]:
        if self._w is None:
            return None
        return self._w.grad

    def tally(self, bits_f: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return sum of ±1 bits per matrix entry, shape ``[out, in]``."""
        if bits_f is None:
            bits_f = self.bits.float()
        return bits_f.sum(dim=-1)

    def tally_weight(self, bits_f: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encoded scalar weight per matrix entry, shape ``[out, in]``."""
        s = self.tally(bits_f)
        return encode_tally(s, self.tally_width, self.encoder, self.tanh_tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bits_f = self.bits.detach().float()
        s = bits_f.sum(dim=-1)
        w = encode_tally(s, self.tally_width, self.encoder, self.tanh_tau)
        # Leaf so ∂L/∂w is available for continuous optimizers + writeback.
        w_param = w.detach().requires_grad_(True)
        self._w = w_param
        return F.linear(x, w_param * self.gain)

    @torch.no_grad()
    def enforce_binary_(self) -> None:
        p = self.bits
        p.copy_(torch.where(p >= 0, torch.ones_like(p), -torch.ones_like(p)))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"tally_width={self.tally_width}, encoder={self.encoder}"
        )
