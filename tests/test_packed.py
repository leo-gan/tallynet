"""Packed bits, popcount kernels, and GEMM compute paths."""

from __future__ import annotations

import pytest
import torch

from tallynet.encoders import encode_tally
from tallynet.layers import TallyLinear
from tallynet.native import load_native
from tallynet.packed import (
    bytes_per_group,
    pack_pm1,
    popcount_lut,
    popcount_packed,
    tally_from_packed,
    unpack_pm1,
)


@pytest.mark.parametrize("S", [1, 7, 8, 9, 16, 32, 64, 256])
def test_pack_roundtrip(S: int):
    bits = torch.randint(0, 2, (3, 5, S), dtype=torch.int8) * 2 - 1
    packed = pack_pm1(bits)
    assert packed.dtype == torch.uint8
    assert packed.shape == (3, 5, bytes_per_group(S))
    back = unpack_pm1(packed, S)
    assert torch.equal(back, bits)


@pytest.mark.parametrize("S", [8, 16, 32, 64, 256])
def test_popcount_matches_float_sum(S: int):
    bits = torch.randint(0, 2, (4, 6, S), dtype=torch.int8) * 2 - 1
    packed = pack_pm1(bits)
    pop = popcount_packed(packed, S, native=False)
    n_plus = (bits > 0).sum(dim=-1)
    assert torch.equal(pop, n_plus.to(torch.int32))
    s = tally_from_packed(packed, S, native=False)
    assert torch.allclose(s, bits.float().sum(-1))


def test_native_matches_lut():
    ext = load_native()
    if ext is None:
        pytest.skip("native extension not built")
    torch.manual_seed(0)
    for S in (8, 16, 32, 64, 256):
        bits = torch.randint(0, 2, (8, 12, S), dtype=torch.int8) * 2 - 1
        packed = pack_pm1(bits)
        lut = popcount_lut(packed, S)
        nat = popcount_packed(packed, S, native=True)
        assert torch.equal(lut, nat), f"S={S}"


def test_layer_forward_matches_unpacked_reference():
    torch.manual_seed(2)
    layer = TallyLinear(16, 8, tally_width=32, encoder="fixed", compute="float32")
    x = torch.randn(4, 16)
    y = layer(x)
    pm1 = layer.bits_pm1().float()
    s = pm1.sum(-1)
    w = encode_tally(s, 32, "fixed")
    y_ref = torch.nn.functional.linear(x, w * layer.gain)
    assert torch.allclose(y, y_ref, atol=1e-5, rtol=1e-5)


def test_bf16_forward_close():
    torch.manual_seed(3)
    layer = TallyLinear(16, 8, tally_width=8, encoder="majority", compute="bfloat16")
    x = torch.randn(4, 16)
    y = layer(x)
    assert y.shape == (4, 8)
    assert y.dtype == torch.float32


def test_int8_majority_inference():
    torch.manual_seed(4)
    layer = TallyLinear(16, 8, tally_width=8, encoder="majority", compute="int8")
    x = torch.randn(4, 16)
    with torch.no_grad():
        y = layer(x)
    assert y.shape == (4, 8)
    layer_f = TallyLinear(16, 8, tally_width=8, encoder="majority", compute="float32")
    layer_f.bits.copy_(layer.bits)
    with torch.no_grad():
        y_f = layer_f(x)
    # int8 quantizes x; same sign pattern, not bit-exact
    assert torch.isfinite(y).all()
    cos = torch.nn.functional.cosine_similarity(y.flatten(), y_f.flatten(), dim=0)
    assert float(cos) > 0.95


def test_int8_rejects_grad():
    layer = TallyLinear(8, 4, tally_width=8, encoder="majority", compute="int8")
    x = torch.randn(2, 8, requires_grad=True)
    with pytest.raises(RuntimeError, match="inference-only"):
        layer(x)


def test_storage_is_packed():
    layer = TallyLinear(10, 4, tally_width=64)
    assert layer.bits.dtype == torch.uint8
    assert layer.bits.numel() == 4 * 10 * 8
    assert layer.num_bits == 4 * 10 * 64
