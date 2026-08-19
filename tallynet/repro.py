"""Seeding and run manifests for experiment scripts.

Not part of the layer. Experiments import this from the library so they
stay independent of each other.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def seed_all(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_state(repo: Path) -> dict[str, Any]:
    def _run(args: list[str]) -> str | None:
        try:
            r = subprocess.run(
                args, cwd=repo, check=True, capture_output=True, text=True
            )
            return r.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    dirty = _run(["git", "status", "--porcelain"])
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "dirty": None if dirty is None else bool(dirty),
    }


def resolved_args(args: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in vars(args).items():
        if isinstance(val, Path):
            out[key] = str(val)
        elif isinstance(val, list):
            out[key] = [str(v) if isinstance(v, Path) else v for v in val]
        else:
            out[key] = val
    return out


def write_manifest(
    path: Path,
    *,
    experiment: str,
    argv: list[str],
    resolved: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import tallynet
    from tallynet.native import status as native_status

    try:
        import torchvision

        tv = torchvision.__version__
    except ImportError:
        tv = None

    manifest: dict[str, Any] = {
        "experiment": experiment,
        "created": datetime.now(timezone.utc).isoformat(),
        "argv": argv,
        "resolved": resolved,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "tallynet": getattr(tallynet, "__version__", None),
        "torch": torch.__version__,
        "torchvision": tv,
        "git": git_state(Path(__file__).resolve().parents[1]),
        "native": native_status(),
        "cwd": str(Path.cwd()),
    }
    if extra:
        manifest["extra"] = extra
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest
