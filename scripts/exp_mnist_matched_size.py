#!/usr/bin/env python3
"""Moved to experiments/mnist_matched_size/train.py. Shim for old commands."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_NEW = Path(__file__).resolve().parents[1] / "experiments" / "mnist_matched_size" / "train.py"
print(
    "note: scripts/exp_mnist_matched_size.py moved to "
    "experiments/mnist_matched_size/train.py",
    file=sys.stderr,
)
sys.argv[0] = str(_NEW)
runpy.run_path(str(_NEW), run_name="__main__")
