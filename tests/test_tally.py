"""Unit tests for TallyNet (no dataset required)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tallynet.encoders import encode_tally
from tallynet.layers import TallyLinear
from tallynet.models import TallyMLP
from tallynet.writeback import TallyWriteback


def test_encode_fixed_and_majority():
    s = torch.tensor([[-4.0, 0.0, 4.0]])
    w = encode_tally(s, tally_width=4, encoder="fixed")
    assert torch.allclose(w, torch.tensor([[-1.0, 0.0, 1.0]]))
    w_m = encode_tally(s, tally_width=4, encoder="majority")
    assert w_m[0, 0].item() == -1.0
    assert w_m[0, 1].item() == 1.0  # tie → +1
    assert w_m[0, 2].item() == 1.0


def test_encode_signed_sqrt_and_tanh():
    s = torch.tensor([4.0, -4.0])
    w = encode_tally(s, tally_width=16, encoder="signed_sqrt")
    assert w[0].item() > 0 and w[1].item() < 0
    w_t = encode_tally(s, tally_width=16, encoder="tanh", tanh_tau=8.0)
    assert w_t.abs().max().item() <= 1.0 + 1e-5


def test_forward_backward_weight_grad_shape():
    layer = TallyLinear(16, 8, tally_width=32, encoder="fixed")
    x = torch.randn(4, 16)
    y = layer(x)
    assert y.shape == (4, 8)
    y.sum().backward()
    assert layer.last_weight_grad is not None
    assert layer.last_weight_grad.shape == (8, 16)


def test_double_flip_identity():
    layer = TallyLinear(4, 2, tally_width=8)
    before = layer.bits.clone()
    pm1 = layer.bits_pm1()
    layer.set_bits_pm1_(-pm1)
    layer.set_bits_pm1_(pm1)
    assert torch.equal(layer.bits, before)


def test_writeback_preserves_pm1():
    torch.manual_seed(0)
    layer = TallyLinear(8, 4, tally_width=16, encoder="fixed")
    wb = TallyWriteback([layer], opt="sgd", lr=1.0, alpha=100.0, p_max=0.5)
    x = torch.randn(8, 8)
    wb.zero_grad()
    loss = layer(x).pow(2).mean()
    loss.backward()
    flip = wb.step()
    layer.enforce_binary_()
    uniq = set(layer.bits_pm1().unique().tolist())
    assert uniq.issubset({-1, 1})
    assert 0.0 <= flip <= 1.0


def test_freeze_bits():
    torch.manual_seed(1)
    layer = TallyLinear(8, 4, tally_width=8)
    wb = TallyWriteback([layer], freeze_bits=True, lr=1.0, alpha=100.0)
    before = layer.bits.clone()
    x = torch.randn(4, 8)
    wb.zero_grad()
    layer(x).sum().backward()
    wb.step()
    assert torch.equal(layer.bits, before)


def test_model_train_step_opts_and_decoders():
    for opt_name in ("sgd", "sgd_m", "adam"):
        for dec in ("density", "thresholded", "sign_noise"):
            m = TallyMLP(hidden_dim=16, tally_width=8, ln_mode="none")
            wb = TallyWriteback(m.tally_layers(), opt=opt_name, decoder=dec, lr=0.1)
            x = torch.randn(2, 1, 28, 28)
            y = torch.tensor([0, 1])
            wb.zero_grad()
            loss = F.cross_entropy(m(x), y)
            loss.backward()
            wb.step()
            m.assert_binary_invariants()


def test_encoders_on_model():
    for enc in ("fixed", "tanh", "signed_sqrt", "majority"):
        m = TallyMLP(hidden_dim=8, tally_width=8, encoder=enc)
        out = m(torch.randn(2, 1, 28, 28))
        assert out.shape == (2, 10)


def test_same_tally_same_weight():
    """Unordered multiset: only tally matters."""
    layer = TallyLinear(1, 1, tally_width=4, encoder="fixed", weight_scale=False)
    # two +1, two -1 → tally sum 0 → w=0 regardless of positions
    layer.set_bits_pm1_(torch.tensor([[[1, 1, -1, -1]]], dtype=torch.int8))
    w1 = layer.tally_weight().item()
    layer.set_bits_pm1_(torch.tensor([[[-1, 1, -1, 1]]], dtype=torch.int8))
    w2 = layer.tally_weight().item()
    assert abs(w1 - w2) < 1e-6
    assert abs(w1) < 1e-6
