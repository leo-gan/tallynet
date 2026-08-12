"""Analytic training-memory budgets for float vs Tally MLPs.

Counts weights + gradients + Adam state + a simple activation estimate.
This is the matching criterion for iso-memory experiments (not a profiler).
"""

from __future__ import annotations

from dataclasses import dataclass

from tallynet.packed import bytes_per_group


@dataclass(frozen=True)
class TrainBudget:
    """Byte breakdown for one training step's resident model+graph estimate."""

    weights: int
    grads: int
    optimizer: int
    activations: int
    n_params: int  # float weights or tally bits
    n_groups: int  # connections (same for both at matched shape)
    meta: dict

    @property
    def total(self) -> int:
        return self.weights + self.grads + self.optimizer + self.activations

    def as_dict(self) -> dict:
        return {
            "weights": self.weights,
            "grads": self.grads,
            "optimizer": self.optimizer,
            "activations": self.activations,
            "total": self.total,
            "n_params": self.n_params,
            "n_groups": self.n_groups,
            **{f"meta_{k}": v for k, v in self.meta.items()},
        }


def mlp_group_counts(
    *,
    hidden_dim: int,
    depth: int = 1,
    in_dim: int = 28 * 28,
    n_classes: int = 10,
) -> list[tuple[int, int]]:
    """List of (in_features, out_features) for each linear, matching TallyMLP/FloatMLP."""
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    layers: list[tuple[int, int]] = []
    d = in_dim
    for _ in range(depth):
        layers.append((d, hidden_dim))
        d = hidden_dim
    layers.append((d, n_classes))
    return layers


def n_groups(
    *,
    hidden_dim: int,
    depth: int = 1,
    in_dim: int = 28 * 28,
    n_classes: int = 10,
) -> int:
    return sum(i * o for i, o in mlp_group_counts(
        hidden_dim=hidden_dim, depth=depth, in_dim=in_dim, n_classes=n_classes
    ))


def activation_bytes(
    *,
    hidden_dim: int,
    batch_size: int,
    depth: int = 1,
    in_dim: int = 28 * 28,
    n_classes: int = 10,
    elem_bytes: int = 4,
) -> int:
    """Rough activations kept for backward: layer inputs + pre-activations.

    depth hidden blocks: for each, save LN/input (feature d) and linear output (H);
    final linear input (H) and logits (n_classes). Conservative, float-sized.
    """
    b = batch_size
    bytes_ = 0
    d = in_dim
    for _ in range(depth):
        bytes_ += b * d * elem_bytes  # input to linear
        bytes_ += b * hidden_dim * elem_bytes  # linear out / relu input
        d = hidden_dim
    bytes_ += b * d * elem_bytes  # input to final linear
    bytes_ += b * n_classes * elem_bytes  # logits
    return bytes_


def float_mlp_budget(
    *,
    hidden_dim: int,
    batch_size: int,
    depth: int = 1,
    in_dim: int = 28 * 28,
    n_classes: int = 10,
    elem_bytes: int = 4,
    adam: bool = True,
) -> TrainBudget:
    """float32 (default) MLP: W + grad + Adam(m,v) + activations. No bias."""
    g = n_groups(
        hidden_dim=hidden_dim, depth=depth, in_dim=in_dim, n_classes=n_classes
    )
    w = g * elem_bytes
    grads = g * elem_bytes
    opt = (2 * g * elem_bytes) if adam else 0
    acts = activation_bytes(
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        depth=depth,
        in_dim=in_dim,
        n_classes=n_classes,
        elem_bytes=elem_bytes,
    )
    return TrainBudget(
        weights=w,
        grads=grads,
        optimizer=opt,
        activations=acts,
        n_params=g,
        n_groups=g,
        meta={
            "kind": "float",
            "hidden_dim": hidden_dim,
            "depth": depth,
            "batch_size": batch_size,
            "elem_bytes": elem_bytes,
            "adam": int(adam),
        },
    )


def tally_mlp_budget(
    *,
    hidden_dim: int,
    tally_width: int,
    batch_size: int,
    depth: int = 1,
    in_dim: int = 28 * 28,
    n_classes: int = 10,
    elem_bytes: int = 4,
    adam: bool = True,
) -> TrainBudget:
    """TallyMLP training peak estimate (current writeback path).

    - weights: packed bits (ceil(S/8) bytes per group)
    - grads: float grad on group leaf ``w`` (not per bit)
    - optimizer: Adam m,v per group
    - plus resident float ``w`` leaf during the step (counted in grads+optimizer
      path as an extra full group tensor under ``weights``-adjacent peak)

    Peak group tensors during a step: bits + w + w.grad + m + v
    = bits + (1 + 1 + 2_if_adam) * groups * elem_bytes for the continuous side,
    with w counted once under a dedicated peak term folded into grads slot as
    w+grad and optimizer as m+v. Explicitly:

      bits + w + grad + m + v
    """
    g = n_groups(
        hidden_dim=hidden_dim, depth=depth, in_dim=in_dim, n_classes=n_classes
    )
    nbytes = bytes_per_group(tally_width)
    bits = g * nbytes
    # Continuous side during train: decoded w (resident for autograd), grad, Adam
    w_leaf = g * elem_bytes
    grads = g * elem_bytes
    opt = (2 * g * elem_bytes) if adam else 0
    acts = activation_bytes(
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        depth=depth,
        in_dim=in_dim,
        n_classes=n_classes,
        elem_bytes=elem_bytes,
    )
    return TrainBudget(
        weights=bits + w_leaf,  # packed storage + temporary/resident w leaf
        grads=grads,
        optimizer=opt,
        activations=acts,
        n_params=g * tally_width,
        n_groups=g,
        meta={
            "kind": "tally",
            "hidden_dim": hidden_dim,
            "tally_width": tally_width,
            "depth": depth,
            "batch_size": batch_size,
            "elem_bytes": elem_bytes,
            "adam": int(adam),
            "bytes_per_group": nbytes,
        },
    )


def max_tally_hidden_under_budget(
    budget_bytes: int,
    *,
    tally_width: int,
    batch_size: int,
    depth: int = 1,
    in_dim: int = 28 * 28,
    n_classes: int = 10,
    adam: bool = True,
    max_hidden: int = 8192,
) -> int | None:
    """Largest hidden_dim whose tally budget is <= budget_bytes, or None."""
    lo, hi = 1, max_hidden
    best: int | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        b = tally_mlp_budget(
            hidden_dim=mid,
            tally_width=tally_width,
            batch_size=batch_size,
            depth=depth,
            in_dim=in_dim,
            n_classes=n_classes,
            adam=adam,
        )
        if b.total <= budget_bytes:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best
