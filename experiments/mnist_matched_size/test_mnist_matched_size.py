"""Smoke for the iso-memory MNIST study (no full train)."""

from pathlib import Path

from tallynet.budget import (
    float_mlp_budget,
    max_tally_hidden_under_budget,
    tally_mlp_budget,
)


def test_iso_memory_tally_fits_and_has_more_bits():
    nn = float_mlp_budget(hidden_dim=64, batch_size=128, depth=1)
    h_t = max_tally_hidden_under_budget(nn.total, tally_width=8, batch_size=128)
    assert h_t is not None
    t = tally_mlp_budget(hidden_dim=h_t, tally_width=8, batch_size=128)
    assert t.total <= nn.total
    assert t.n_params > nn.n_params


def test_train_script_lives_here():
    here = Path(__file__).resolve().parent
    text = (here / "train.py").read_text()
    assert 'EXPERIMENT = "mnist_matched_size"' in text
