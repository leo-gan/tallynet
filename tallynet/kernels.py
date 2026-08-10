"""Decode packed tallies and multiply.

CPU: SIMD popcount (or LUT). GPU: ``__popc`` when the native extension built
with CUDA. The multiply is a stock GEMM: ``F.linear`` in float32/bfloat16, or
``torch._int_mm`` for the optional int8 inference path.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn.functional as F

from tallynet.encoders import EncoderName, encode_tally
from tallynet.packed import tally_from_packed

ComputeName = Literal["auto", "float32", "bfloat16", "int8"]


def decode_tally_weight(
    packed: torch.Tensor,
    tally_width: int,
    encoder: EncoderName = "fixed",
    tanh_tau: float = 0.0,
    *,
    native: bool = True,
) -> torch.Tensor:
    s = tally_from_packed(packed, tally_width, native=native)
    return encode_tally(s, tally_width, encoder, tanh_tau)


def _pick_compute(x: torch.Tensor, compute: ComputeName) -> ComputeName:
    if compute != "auto":
        return compute
    if x.is_cuda and torch.cuda.is_bf16_supported():
        return "bfloat16"
    return "float32"


def linear_int8_majority(
    x: torch.Tensor,
    w_pm1: torch.Tensor,
    gain: float,
) -> torch.Tensor:
    """Per-tensor int8 on ``x``, exact ``±1`` weights, ``torch._int_mm``.

    Inference only. ``w_pm1`` is ``[out, in]`` in ``{-1, +1}``.
    """
    if x.dim() != 2:
        raise ValueError("int8 path expects 2D x [batch, in]")
    scale = x.detach().abs().amax().clamp_min(1e-8) / 127.0
    x_i8 = torch.clamp(torch.round(x / scale), -128, 127).to(torch.int8)
    w_i8 = w_pm1.to(torch.int8).contiguous()
    # _int_mm(a, b): a[M,K] int8, b[K,N] int8 -> int32[M,N]
    acc = torch._int_mm(x_i8.contiguous(), w_i8.t().contiguous())
    return acc.to(dtype=x.dtype) * scale * gain


def tally_linear(
    x: torch.Tensor,
    packed: torch.Tensor,
    tally_width: int,
    *,
    encoder: EncoderName = "fixed",
    tanh_tau: float = 0.0,
    gain: float = 1.0,
    compute: ComputeName = "auto",
    native: bool = True,
    w_out: Optional[list[torch.Tensor]] = None,
) -> torch.Tensor:
    """``y = x @ (enc(tally) * gain).T``.

    When grads are enabled, ``w`` is a float leaf (for writeback) and the GEMM
    is float32 or bfloat16. ``compute='int8'`` is inference-only (majority).
    """
    orig_shape = x.shape
    if x.dim() > 2:
        x2 = x.reshape(-1, orig_shape[-1])
    else:
        x2 = x

    kind = _pick_compute(x2, compute)
    if kind == "int8":
        if torch.is_grad_enabled():
            raise RuntimeError("compute='int8' is inference-only; disable grad or use float32/bfloat16")
        if encoder != "majority":
            raise RuntimeError("compute='int8' currently supports encoder='majority' only")
        s = tally_from_packed(packed, tally_width, native=native)
        w_pm1 = torch.where(s == 0, torch.ones_like(s), torch.sign(s))
        y = linear_int8_majority(x2, w_pm1, gain)
        return y.reshape(*orig_shape[:-1], packed.shape[0])

    w = decode_tally_weight(packed, tally_width, encoder, tanh_tau, native=native)
    w_param = w.detach().requires_grad_(True)
    if w_out is not None:
        w_out.append(w_param)

    dtype = torch.bfloat16 if kind == "bfloat16" else torch.float32
    y = F.linear(x2.to(dtype), (w_param * gain).to(dtype))
    y = y.to(dtype=x.dtype)
    return y.reshape(*orig_shape[:-1], packed.shape[0])
