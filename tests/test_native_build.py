"""Prebuilt kernel search path and explicit build."""

from __future__ import annotations

from pathlib import Path

import pytest

from tallynet.native import (
    CPU_SONAME,
    build_native,
    find_library,
    load_native,
    reset,
    status,
)


def test_status_after_default_load():
    load_native()
    info = status()
    assert "search" in info
    assert info["enabled"] is True


def test_build_to_explicit_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    if load_native() is None:
        pytest.skip("g++ cannot build the CPU kernel here")
    dest = tmp_path / "kernels"
    built = build_native(out_dir=dest, march="x86-64-v3", force=True)
    assert built["cpu"] is not None
    assert built["cpu"].is_file()
    assert built["cpu"].name == CPU_SONAME
    monkeypatch.setenv("TALLYNET_KERNEL_DIR", str(dest))
    reset()
    path = find_library(CPU_SONAME)
    assert path == built["cpu"]
    reset()
    assert load_native() is not None
    assert status()["cpu_path"] == str(built["cpu"])
