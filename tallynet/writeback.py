"""Continuous step on tally weights + stochastic flip writeback into bit bags.

This is a *training utility*, not the architecture. TallyNet's claim is the
weight representation; writeback is one way to keep storage discrete while
stepping a continuous Δ on the encoded weight.
"""

from __future__ import annotations

from typing import Iterable, List, Literal, Optional, Sequence

import torch
import torch.nn as nn

from tallynet.layers import TallyLinear
from tallynet.packed import flip_packed_bits_

OptName = Literal["sgd", "sgd_m", "adam"]
DecoderName = Literal["density", "thresholded", "sign_noise"]


class TallyWriteback:
    """Adam/SGD on tally weights; map Δ → flip probabilities; flip bits in place.

    Not a ``torch.optim.Optimizer``: works on buffer-backed bit bags.
    """

    def __init__(
        self,
        layers: Sequence[TallyLinear],
        *,
        opt: OptName = "adam",
        lr: float = 1e-3,
        momentum: float = 0.9,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        decoder: DecoderName = "density",
        alpha: float = 10.0,
        p_min: float = 0.0,
        p_max: float = 0.25,
        p_noise: float = 0.001,
        threshold: float = 1e-3,
        ln_params: Optional[Iterable[nn.Parameter]] = None,
        ln_lr: float = 1e-2,
        freeze_bits: bool = False,
    ):
        self.layers: List[TallyLinear] = list(layers)
        self.opt_name = opt
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)
        self.decoder = decoder
        self.alpha = float(alpha)
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.p_noise = float(p_noise)
        self.threshold = float(threshold)
        self.freeze_bits = bool(freeze_bits)
        self._step = 0

        self._m: dict[int, torch.Tensor] = {}
        self._v: dict[int, torch.Tensor] = {}

        ln_list = list(ln_params) if ln_params is not None else []
        self.ln_optimizer: torch.optim.Optimizer | None
        if ln_list:
            self.ln_optimizer = torch.optim.SGD(ln_list, lr=ln_lr)
        else:
            self.ln_optimizer = None

        self.last_flip_frac: float = 0.0
        self.last_delta_abs_mean: float = 0.0
        self.last_grad_abs_mean: float = 0.0

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer._w = None
        if self.ln_optimizer is not None:
            self.ln_optimizer.zero_grad(set_to_none=True)

    def _ensure_state(self, layer: TallyLinear, like: torch.Tensor) -> None:
        key = id(layer)
        if key not in self._m:
            self._m[key] = torch.zeros_like(like)
        if self.opt_name == "adam" and key not in self._v:
            self._v[key] = torch.zeros_like(like)

    def _adam_delta(self, layer: TallyLinear, g: torch.Tensor, step: int) -> torch.Tensor:
        self._ensure_state(layer, g)
        key = id(layer)
        m = self._m[key]
        v = self._v[key]
        m.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
        v.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)
        m_hat = m / (1.0 - self.beta1**step)
        v_hat = v / (1.0 - self.beta2**step)
        return -self.lr * m_hat / (v_hat.sqrt() + self.eps)

    def _flip_prob(self, delta: torch.Tensor) -> torch.Tensor:
        abs_d = delta.abs()
        if self.decoder == "density":
            p = (self.alpha * abs_d).clamp(self.p_min, self.p_max)
            return (p + self.p_noise).clamp(0.0, 1.0)
        if self.decoder == "thresholded":
            p = torch.where(
                abs_d > self.threshold,
                torch.full_like(abs_d, self.p_max),
                torch.full_like(abs_d, self.p_noise),
            )
            return p.clamp(0.0, 1.0)
        if self.decoder == "sign_noise":
            return torch.full_like(abs_d, self.p_noise).clamp(0.0, 1.0)
        raise ValueError(self.decoder)

    @torch.no_grad()
    def step(self) -> float:
        """Apply writeback; return fraction of bits flipped."""
        flipped = 0
        total = 0
        delta_abs_sum = 0.0
        delta_abs_n = 0
        grad_abs_sum = 0.0
        grad_abs_n = 0

        if self.opt_name == "adam":
            self._step += 1
            step_t = self._step
        else:
            step_t = 0

        for layer in self.layers:
            g = layer.last_weight_grad
            if g is None:
                continue

            grad_abs_sum += float(g.abs().sum().item())
            grad_abs_n += g.numel()

            if self.opt_name == "adam":
                delta = self._adam_delta(layer, g, step_t)
            elif self.opt_name == "sgd":
                delta = -self.lr * g
            elif self.opt_name == "sgd_m":
                self._ensure_state(layer, g)
                m = self._m[id(layer)]
                m.mul_(self.momentum).add_(g)
                delta = -self.lr * m
            else:
                raise ValueError(self.opt_name)

            delta_abs_sum += float(delta.abs().sum().item())
            delta_abs_n += delta.numel()

            if self.freeze_bits:
                total += layer.num_bits
                continue

            p = self._flip_prob(delta)
            want_plus = delta > 0
            want_minus = delta < 0
            n_flip = flip_packed_bits_(
                layer.bits,
                layer.tally_width,
                want_plus,
                want_minus,
                p,
            )
            flipped += n_flip
            total += layer.num_bits
            layer.enforce_binary_()

        if self.ln_optimizer is not None:
            self.ln_optimizer.step()

        self.last_flip_frac = flipped / max(1, total)
        self.last_delta_abs_mean = delta_abs_sum / max(1, delta_abs_n)
        self.last_grad_abs_mean = grad_abs_sum / max(1, grad_abs_n)
        return self.last_flip_frac
