"""Optional native popcount kernels (CPU SIMD, CUDA __popc).

Built on first use with ``g++`` / ``nvcc`` into a small shared library (no
Python headers). Set ``TALLYNET_NATIVE=0`` to force the pure-PyTorch LUT path.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch

_CPU: Optional[ctypes.CDLL] = None
_CUDA: Optional[ctypes.CDLL] = None
_TRIED = False


def native_enabled() -> bool:
    return os.environ.get("TALLYNET_NATIVE", "1") not in ("0", "false", "False")


def _cache_dir() -> Path:
    d = Path(__file__).resolve().parent.parent / ".kernel_cache"
    d.mkdir(exist_ok=True)
    return d


def _compile_cpu(src: Path, so: Path) -> bool:
    if so.is_file() and so.stat().st_mtime >= src.stat().st_mtime:
        return True
    cmd = [
        "g++",
        "-O3",
        "-shared",
        "-fPIC",
        "-march=native",
        "-fopenmp",
        str(src),
        "-o",
        str(so),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return so.is_file()
    except (OSError, subprocess.CalledProcessError):
        # Retry without OpenMP (toolchain may lack libgomp).
        cmd = [c for c in cmd if c != "-fopenmp"]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return so.is_file()
        except (OSError, subprocess.CalledProcessError):
            return False


def _compile_cuda(src: Path, so: Path) -> bool:
    if so.is_file() and so.stat().st_mtime >= src.stat().st_mtime:
        return True
    nvcc = os.environ.get("NVCC", "nvcc")
    cmd = [
        nvcc,
        "-O3",
        "--compiler-options",
        "-fPIC,-fvisibility=hidden",
        "-shared",
        str(src),
        "-o",
        str(so),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return so.is_file()
    except (OSError, subprocess.CalledProcessError):
        return False


def _load_libs() -> None:
    global _CPU, _CUDA, _TRIED
    if _TRIED:
        return
    _TRIED = True
    if not native_enabled():
        return
    here = Path(__file__).resolve().parent / "csrc"
    cache = _cache_dir()
    cpu_src = here / "popcount_cpu.cpp"
    if cpu_src.is_file():
        cpu_so = cache / "libtallynet_popcount_cpu.so"
        if _compile_cpu(cpu_src, cpu_so):
            lib = ctypes.CDLL(str(cpu_so))
            lib.tallynet_popcount_cpu.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int32,
                ctypes.c_int32,
            ]
            lib.tallynet_popcount_cpu.restype = None
            _CPU = lib
    if torch.cuda.is_available():
        cu_src = here / "popcount_cuda.cu"
        if cu_src.is_file():
            cu_so = cache / "libtallynet_popcount_cuda.so"
            if _compile_cuda(cu_src, cu_so):
                lib = ctypes.CDLL(str(cu_so))
                lib.tallynet_popcount_cuda.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int64,
                    ctypes.c_int32,
                    ctypes.c_int32,
                ]
                lib.tallynet_popcount_cuda.restype = ctypes.c_char_p
                _CUDA = lib


def load_native(*, verbose: bool = False) -> Optional[object]:
    """Load CPU (and CUDA, if available) libraries. Returns the CPU lib or None."""
    _load_libs()
    if verbose:
        print(f"tallynet native cpu={_CPU is not None} cuda={_CUDA is not None}", file=sys.stderr)
    return _CPU


def native_popcount(packed: torch.Tensor, tally_width: int) -> Optional[torch.Tensor]:
    _load_libs()
    packed = packed.contiguous()
    nbytes = int(packed.shape[-1])
    n = packed.numel() // nbytes
    out_sizes = packed.shape[:-1]
    if packed.is_cuda:
        if _CUDA is None:
            return None
        out = torch.empty(out_sizes, dtype=torch.int32, device=packed.device)
        err = _CUDA.tallynet_popcount_cuda(
            ctypes.c_void_p(packed.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int64(n),
            ctypes.c_int32(nbytes),
            ctypes.c_int32(int(tally_width)),
        )
        if err:
            return None
        return out
    if _CPU is None:
        return None
    if packed.device.type != "cpu":
        return None
    out = torch.empty(out_sizes, dtype=torch.int32)
    _CPU.tallynet_popcount_cpu(
        ctypes.c_void_p(packed.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int64(n),
        ctypes.c_int32(nbytes),
        ctypes.c_int32(int(tally_width)),
    )
    return out
