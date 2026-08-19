"""Smoke tests for the CIFAR scale-gap experiment (no full train)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from tallynet.budget import float_mlp_budget, n_groups, tally_mlp_budget
from tallynet.models import FloatMLP, TallyMLP

_DIR = Path(__file__).resolve().parent


def _load_local(name: str):
    import sys

    path = _DIR / f"{name}.py"
    mod_name = f"cifar_scale_gap_{name}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_analysis = _load_local("analysis")
decide_expand = _analysis.decide_expand

CIFAR_IN = 32 * 32 * 3


def test_models_cifar_shapes():
    x = torch.randn(4, 3, 32, 32)
    f = FloatMLP(hidden_dim=32, depth=1, in_dim=CIFAR_IN)
    t = TallyMLP(hidden_dim=32, tally_width=8, depth=1, in_dim=CIFAR_IN, encoder="majority")
    assert f(x).shape == (4, 10)
    assert t(x).shape == (4, 10)
    t.assert_binary_invariants()


def test_budget_iso_shape_same_groups():
    h = 128
    fb = float_mlp_budget(hidden_dim=h, batch_size=128, in_dim=CIFAR_IN)
    tb = tally_mlp_budget(hidden_dim=h, tally_width=32, batch_size=128, in_dim=CIFAR_IN)
    assert fb.n_groups == tb.n_groups
    assert tb.n_params == fb.n_groups * 32
    assert n_groups(hidden_dim=h, in_dim=CIFAR_IN) == 3072 * h + h * 10


def test_expand_h_when_tally_and_gap_still_move():
    float_by_h = {128: 0.40, 256: 0.45, 512: 0.50}
    # gap -8, -6, -4 pp — still closing 2 pp per doubling; tally still up
    tally = {(128, 32): 0.32, (256, 32): 0.39, (512, 32): 0.46}
    d = decide_expand(float_by_h=float_by_h, tally_by_hs=tally)
    assert d.expand_h and d.next_h == 1024
    assert d.expand_s  # S cross missing
    assert not d.stop


def test_stop_when_width_and_s_plateau():
    float_by_h = {128: 0.50, 256: 0.52, 512: 0.53, 1024: 0.535, 2048: 0.538}
    # tally acc almost flat at the top; gap stuck ~-3 pp
    tally = {
        (128, 32): 0.45,
        (256, 32): 0.49,
        (512, 32): 0.500,
        (1024, 32): 0.503,
        (2048, 32): 0.505,
        (2048, 8): 0.500,
        (2048, 128): 0.506,
        (2048, 256): 0.507,
    }
    d = decide_expand(float_by_h=float_by_h, tally_by_hs=tally)
    assert not d.expand_h
    assert not d.expand_s
    assert d.stop


def test_s_cross_then_double():
    float_by_h = {512: 0.50}
    tally = {(512, 32): 0.40}
    d = decide_expand(float_by_h=float_by_h, tally_by_hs=tally)
    assert d.expand_s
    assert set(d.next_s_values) == {8, 128, 256}
    assert d.s_h == 512

    tally2 = {
        (512, 8): 0.38,
        (512, 32): 0.40,
        (512, 128): 0.43,
        (512, 256): 0.46,
    }
    d2 = decide_expand(float_by_h=float_by_h, tally_by_hs=tally2)
    assert d2.expand_s
    assert d2.next_s_values == (512,)


def test_csv_roundtrip(tmp_path):
    train = _load_local("train")
    RunResult = train.RunResult
    _load_csv = train._load_csv
    _write_csv = train._write_csv

    row = RunResult(
        arm="tally_S32",
        hidden_dim=128,
        tally_width=32,
        depth=1,
        seed=0,
        epochs=2,
        batch_size=128,
        best_test_acc=0.4,
        final_test_acc=0.39,
        final_train_acc=0.42,
        final_train_loss=1.5,
        budget_total=1000,
        budget_weights=100,
        budget_grads=100,
        budget_optimizer=200,
        budget_activations=600,
        n_params=394240,
        n_groups=12320,
        seconds=1.25,
        native_cpu=1,
    )
    path = tmp_path / "latest.csv"
    _write_csv(path, [row])
    loaded = _load_csv(path)
    assert len(loaded) == 1
    assert loaded[0].arm == "tally_S32"
    assert loaded[0].hidden_dim == 128
    assert loaded[0].tally_width == 32
    assert loaded[0].best_test_acc == 0.4
    assert loaded[0].n_params == 394240


def test_h_cap_is_inconclusive_not_optimum():
    hs = [128, 256, 512, 1024, 2048, 4096, 8192]
    float_by_h = {h: 0.40 + 0.02 * i for i, h in enumerate(hs)}
    tally = {(h, 32): 0.20 + 0.02 * i for i, h in enumerate(hs)}
    d = decide_expand(float_by_h=float_by_h, tally_by_hs=tally)
    assert not d.expand_h
    assert d.next_h is None
    assert any("cap" in r.lower() for r in d.reasons)
