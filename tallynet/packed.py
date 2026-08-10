"""Pack ±1 bits into bytes and count them.

Storage: ``uint8[..., ceil(S/8)]``. Bit ``k`` of a group lives in byte ``k // 8``,
bit ``k % 8`` (low bit first). ``1`` means ``+1``, ``0`` means ``-1``. Unused bits
in the last byte are kept 0 so a popcount of the bytes equals the tally of ``+1``.
"""

from __future__ import annotations

from typing import Optional

import torch

from tallynet.native import native_popcount

_LUT: Optional[torch.Tensor] = None


def bytes_per_group(tally_width: int) -> int:
    if tally_width < 1:
        raise ValueError(f"tally_width must be >= 1, got {tally_width}")
    return (tally_width + 7) // 8


def last_byte_mask(tally_width: int) -> int:
    rem = tally_width & 7
    return 0xFF if rem == 0 else (1 << rem) - 1


def _popcount_lut_table(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    global _LUT
    if _LUT is None:
        _LUT = torch.tensor([i.bit_count() for i in range(256)], dtype=torch.int16)
    return _LUT.to(device=device, dtype=dtype)


def pack_pm1(bits_pm1: torch.Tensor) -> torch.Tensor:
    """Pack last-dim ``±1`` (or ``{0,1}``) int tensor to ``uint8`` bytes."""
    if bits_pm1.dim() < 1:
        raise ValueError("bits_pm1 must have a last dimension of length S")
    S = bits_pm1.shape[-1]
    nbytes = bytes_per_group(S)
    ones = bits_pm1 > 0
    packed = torch.zeros(*bits_pm1.shape[:-1], nbytes, dtype=torch.uint8, device=bits_pm1.device)
    for k in range(S):
        bit = ones[..., k].to(torch.uint8)
        packed[..., k >> 3] |= bit << (k & 7)
    return packed


def unpack_pm1(packed: torch.Tensor, tally_width: int) -> torch.Tensor:
    """Unpack to ``int8`` ``±1`` of last-dim ``tally_width``."""
    S = int(tally_width)
    nbytes = packed.shape[-1]
    if nbytes != bytes_per_group(S):
        raise ValueError(f"packed last dim {nbytes} != ceil({S}/8)")
    out = torch.empty(*packed.shape[:-1], S, dtype=torch.int8, device=packed.device)
    for k in range(S):
        bit = (packed[..., k >> 3] >> (k & 7)) & 1
        out[..., k] = torch.where(bit.bool(), torch.ones_like(bit, dtype=torch.int8), -torch.ones_like(bit, dtype=torch.int8))
    return out


def random_packed(
    out_features: int,
    in_features: int,
    tally_width: int,
    *,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    nbytes = bytes_per_group(tally_width)
    packed = torch.randint(
        0,
        256,
        (out_features, in_features, nbytes),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    packed[..., -1] &= last_byte_mask(tally_width)
    return packed


def popcount_lut(packed: torch.Tensor, tally_width: int) -> torch.Tensor:
    """Portable popcount: 256-entry table, summed over bytes."""
    nbytes = packed.shape[-1]
    if nbytes != bytes_per_group(tally_width):
        raise ValueError(f"packed last dim {nbytes} != ceil({tally_width}/8)")
    table = _popcount_lut_table(packed.device, torch.int16)
    last = packed[..., -1] & last_byte_mask(tally_width)
    if nbytes == 1:
        return table[last.long()].to(torch.int32)
    head = table[packed[..., :-1].long()].sum(dim=-1)
    return (head + table[last.long()]).to(torch.int32)


def popcount_packed(packed: torch.Tensor, tally_width: int, *, native: bool = True) -> torch.Tensor:
    """Popcount of the first ``tally_width`` bits. Native SIMD/CUDA if available."""
    if native:
        out = native_popcount(packed, tally_width)
        if out is not None:
            return out
    return popcount_lut(packed, tally_width)


def tally_from_packed(packed: torch.Tensor, tally_width: int, *, native: bool = True) -> torch.Tensor:
    """Signed tally ``s = (#plus) - (#minus) = 2*popcount - S``, float32."""
    pop = popcount_packed(packed, tally_width, native=native)
    return (pop * 2 - tally_width).to(torch.float32)


@torch.no_grad()
def flip_packed_bits_(
    packed: torch.Tensor,
    tally_width: int,
    want_plus: torch.Tensor,
    want_minus: torch.Tensor,
    p: torch.Tensor,
) -> int:
    """Stochastic flips on packed bits. ``p``, ``want_*`` are per-group, shape ``[out, in]``.

    Returns the number of bits flipped.
    """
    S = int(tally_width)
    flipped = 0
    p = p.to(dtype=torch.float32)
    for k in range(S):
        byte_i = k >> 3
        mask = 1 << (k & 7)
        byte = packed[..., byte_i]
        is_plus = (byte & mask) != 0
        eligible = (want_plus & ~is_plus) | (want_minus & is_plus)
        if not bool(eligible.any()):
            continue
        roll = torch.rand(p.shape, device=p.device, dtype=torch.float32)
        flip = eligible & (roll < p)
        n = int(flip.sum().item())
        if n:
            xor = flip.to(torch.uint8) * mask
            packed[..., byte_i] = byte ^ xor
            flipped += n
    return flipped


def clear_padding_(packed: torch.Tensor, tally_width: int) -> None:
    packed[..., -1] &= last_byte_mask(tally_width)
