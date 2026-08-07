"""Minimal forward pass — no training, no dataset."""

import torch

from tallynet import TallyLinear, TallyMLP, encode_tally


def main() -> None:
    torch.manual_seed(0)
    layer = TallyLinear(16, 8, tally_width=32, encoder="fixed")
    x = torch.randn(4, 16)
    y = layer(x)
    print("TallyLinear out:", tuple(y.shape))
    print("tally weight range:", float(layer.tally_weight().min()), float(layer.tally_weight().max()))

    # Same tally ⇒ same weight
    s = torch.tensor([2.0, -2.0, 0.0])
    print("encode fixed:", encode_tally(s, 4, "fixed").tolist())

    m = TallyMLP(hidden_dim=32, tally_width=16, encoder="majority")
    logits = m(torch.randn(2, 1, 28, 28))
    print("TallyMLP logits:", tuple(logits.shape))
    m.assert_binary_invariants()
    print("binary invariants ok")


if __name__ == "__main__":
    main()
