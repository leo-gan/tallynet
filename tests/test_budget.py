"""Training budget helpers for iso-memory experiments."""

from tallynet.budget import (
    float_mlp_budget,
    max_tally_hidden_under_budget,
    n_groups,
    tally_mlp_budget,
)


def test_n_groups_depth1():
    # 784*128 + 128*10
    assert n_groups(hidden_dim=128, depth=1) == 784 * 128 + 128 * 10


def test_float_budget_adam_is_4x_weights():
    b = float_mlp_budget(hidden_dim=64, batch_size=128, depth=1)
    # W+G+m+v = 4 * weights when adam and no other
    assert b.weights == b.n_groups * 4
    assert b.grads == b.weights
    assert b.optimizer == 2 * b.weights
    assert b.total == b.weights + b.grads + b.optimizer + b.activations


def test_tally_fits_more_params_under_same_budget():
    nn_b = float_mlp_budget(hidden_dim=64, batch_size=128)
    h_t = max_tally_hidden_under_budget(
        nn_b.total, tally_width=8, batch_size=128
    )
    assert h_t is not None
    t_b = tally_mlp_budget(hidden_dim=h_t, tally_width=8, batch_size=128)
    assert t_b.total <= nn_b.total
    assert t_b.n_params > nn_b.n_params


def test_tally_budget_increases_with_s_bits():
    a = tally_mlp_budget(hidden_dim=32, tally_width=8, batch_size=64)
    b = tally_mlp_budget(hidden_dim=32, tally_width=256, batch_size=64)
    assert b.n_params > a.n_params
    assert b.weights > a.weights  # more packed bytes (+ same w leaf)
