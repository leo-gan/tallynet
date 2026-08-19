"""Run-manifest helper used by experiment scripts."""

from __future__ import annotations

import json

import torch

from tallynet.repro import resolved_args, seed_all, write_manifest


def test_seed_all_is_deterministic():
    seed_all(0)
    a = torch.rand(4)
    seed_all(0)
    b = torch.rand(4)
    assert torch.equal(a, b)


def test_write_manifest_records_experiment_and_argv(tmp_path):
    class Args:
        def __init__(self) -> None:
            self.epochs = 20
            self.data_dir = tmp_path / "data"

    path = tmp_path / "latest.manifest.json"
    write_manifest(
        path,
        experiment="cifar_scale_gap",
        argv=["--scale"],
        resolved=resolved_args(Args()),
        extra={"n_rows": 0},
    )
    data = json.loads(path.read_text())
    assert data["experiment"] == "cifar_scale_gap"
    assert data["argv"] == ["--scale"]
    assert data["resolved"]["epochs"] == 20
    assert data["resolved"]["data_dir"] == str(tmp_path / "data")
    assert "torch" in data
    assert "git" in data
    assert data["extra"]["n_rows"] == 0
