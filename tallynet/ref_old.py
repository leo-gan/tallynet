"""Reference *old* tally path: unpack ±1 and ``float().sum(-1)``.

Used only to compare against the packed SIMD kernel. Not a supported API.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from tallynet.encoders import encode_tally
from tallynet.models import TallyMLP
from tallynet.packed import unpack_pm1
from tallynet.writeback import TallyWriteback

ForwardFn = Callable[[TallyMLP, torch.Tensor], torch.Tensor]


def old_forward(model: TallyMLP, x: torch.Tensor) -> torch.Tensor:
    """Previous layer: unpack every bit to float and sum. Attaches ``_w`` for writeback."""
    x = model.flatten(x)
    n = len(model.linears)
    for i, (ln, linear) in enumerate(zip(model.lns, model.linears)):
        x = ln(x)
        pm1 = unpack_pm1(linear.bits, linear.tally_width).float()
        s = pm1.sum(-1)
        w = encode_tally(s, linear.tally_width, linear.encoder, linear.tanh_tau)
        w_param = w.detach().requires_grad_(True)
        linear._w = w_param
        x = F.linear(x, w_param * linear.gain)
        if i < n - 1:
            x = model.act(x)
    return x


def simd_forward(model: TallyMLP, x: torch.Tensor) -> torch.Tensor:
    return model(x)


def train_step(
    model: TallyMLP,
    wb: TallyWriteback,
    x: torch.Tensor,
    y: torch.Tensor,
    forward: ForwardFn,
) -> float:
    """One Adam/SGD writeback step. Returns the loss value."""
    wb.zero_grad()
    loss = F.cross_entropy(forward(model, x), y)
    loss.backward()
    wb.step()
    return float(loss.item())
