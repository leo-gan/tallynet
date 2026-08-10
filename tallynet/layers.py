"""TallyLinear: tally-coded multi-bit linear layer."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from tallynet.encoders import EncoderName, encode_tally
from tallynet.kernels import ComputeName, decode_tally_weight, tally_linear
from tallynet.packed import (
    bytes_per_group,
    clear_padding_,
    pack_pm1,
    popcount_packed,
    random_packed,
    unpack_pm1,
)


class TallyLinear(nn.Module):
    """Linear layer with tally-coded weights.

    Storage: packed bits ``bits`` of shape ``[out, in, ceil(S/8)]`` as ``uint8``.
    Each of the ``S`` bits is one parameter (``1`` → ``+1``, ``0`` → ``-1``).
    Forward: popcount → ``w = enc(tally)`` → ``y = x @ (w * gain).T``.

    Gradients attach to a leaf ``w`` (detached from storage) so a continuous
    optimizer can produce Δ on the tally weight; writeback into bits is separate
    (see ``tallynet.writeback``). Storage remains the packed bits — no FP
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
        compute: ComputeName = "auto",
    ):
        super().__init__()
        if tally_width < 1:
            raise ValueError(f"tally_width must be >= 1, got {tally_width}")
        self.in_features = in_features
        self.out_features = out_features
        self.tally_width = tally_width
        self.encoder: EncoderName = encoder
        self.tanh_tau = float(tanh_tau)
        self.compute: ComputeName = compute
        gain = 1.0 / math.sqrt(in_features) if weight_scale else 1.0
        self.register_buffer("gain", torch.tensor(gain, dtype=torch.float32))
        self.register_buffer(
            "bits",
            random_packed(out_features, in_features, tally_width),
        )
        self._w: Optional[torch.Tensor] = None

    @property
    def nbytes(self) -> int:
        return bytes_per_group(self.tally_width)

    @property
    def num_bits(self) -> int:
        return self.out_features * self.in_features * self.tally_width

    @property
    def last_weight_grad(self) -> Optional[torch.Tensor]:
        if self._w is None:
            return None
        return self._w.grad

    def bits_pm1(self) -> torch.Tensor:
        """Unpack to ``int8`` ``±1``, shape ``[out, in, S]`` (debug / tests)."""
        return unpack_pm1(self.bits, self.tally_width)

    @torch.no_grad()
    def set_bits_pm1_(self, bits_pm1: torch.Tensor) -> None:
        packed = pack_pm1(bits_pm1.to(device=self.bits.device))
        if packed.shape != self.bits.shape:
            raise ValueError(f"packed shape {tuple(packed.shape)} != {tuple(self.bits.shape)}")
        self.bits.copy_(packed)

    def popcount(self) -> torch.Tensor:
        """Number of ``+1`` bits per group, ``int32 [out, in]``."""
        return popcount_packed(self.bits, self.tally_width)

    def tally(self, bits_f: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Signed tally ``2*popcount - S`` per group, shape ``[out, in]``.

        If ``bits_f`` is given it is treated as unpacked ``±1`` with last dim ``S``
        (old path, for tests). Otherwise the packed buffer is counted.
        """
        if bits_f is not None:
            return bits_f.float().sum(dim=-1)
        return (self.popcount() * 2 - self.tally_width).to(torch.float32)

    def tally_weight(self, bits_f: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encoded scalar weight per group, shape ``[out, in]``."""
        if bits_f is not None:
            s = self.tally(bits_f)
            return encode_tally(s, self.tally_width, self.encoder, self.tanh_tau)
        return decode_tally_weight(
            self.bits, self.tally_width, self.encoder, self.tanh_tau
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        holder: list[torch.Tensor] = []
        y = tally_linear(
            x,
            self.bits.detach(),
            self.tally_width,
            encoder=self.encoder,
            tanh_tau=self.tanh_tau,
            gain=float(self.gain.item()),
            compute=self.compute,
            w_out=holder,
        )
        self._w = holder[0] if holder else None
        return y

    @torch.no_grad()
    def enforce_binary_(self) -> None:
        clear_padding_(self.bits, self.tally_width)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"tally_width={self.tally_width}, nbytes={self.nbytes}, "
            f"encoder={self.encoder}, compute={self.compute}"
        )
