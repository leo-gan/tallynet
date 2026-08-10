"""Optional native popcount kernels (CPU SIMD, CUDA __popc).

Search order for prebuilt libraries:

1. ``$TALLYNET_KERNEL_DIR``
2. ``tallynet/lib/`` (install / wheel)
3. ``.kernel_cache/`` (local JIT)

If none is found, the CPU library is compiled into ``.kernel_cache/``
(or ``$TALLYNET_KERNEL_DIR``). Set ``TALLYNET_NATIVE=0`` to force the
pure-PyTorch table. ``TALLYNET_MARCH`` selects ``-march`` (default
``native``; use ``x86-64-v3`` for portable AVX2 artifacts).

    python -m tallynet.native build --out tallynet/lib --march x86-64-v3
    python -m tallynet.native info
"""

from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import torch

CPU_SONAME = "libtallynet_popcount_cpu.so"
CUDA_SONAME = "libtallynet_popcount_cuda.so"

_CPU: Optional[ctypes.CDLL] = None
_CUDA: Optional[ctypes.CDLL] = None
_TRIED = False


def native_enabled() -> bool:
    return os.environ.get("TALLYNET_NATIVE", "1") not in ("0", "false", "False")


def package_dir() -> Path:
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    return package_dir().parent


def default_cache_dir() -> Path:
    return repo_root() / ".kernel_cache"


def kernel_search_dirs() -> list[Path]:
    dirs: list[Path] = []
    env = os.environ.get("TALLYNET_KERNEL_DIR")
    if env:
        dirs.append(Path(env).expanduser().resolve())
    dirs.append(package_dir() / "lib")
    dirs.append(default_cache_dir())
    return dirs


def find_library(soname: str) -> Optional[Path]:
    for d in kernel_search_dirs():
        p = d / soname
        if p.is_file():
            return p
    return None


def march_flag() -> str:
    return os.environ.get("TALLYNET_MARCH", "native")


def _compile_cpu(src: Path, so: Path, *, march: str, openmp: bool) -> bool:
    so.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        os.environ.get("CXX", "g++"),
        "-O3",
        "-shared",
        "-fPIC",
        f"-march={march}",
        str(src),
        "-o",
        str(so),
    ]
    if openmp:
        cmd[4:4] = ["-fopenmp"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return so.is_file()
    except (OSError, subprocess.CalledProcessError):
        if not openmp:
            return False
        return _compile_cpu(src, so, march=march, openmp=False)


def _compile_cuda(src: Path, so: Path) -> bool:
    so.parent.mkdir(parents=True, exist_ok=True)
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


def _bind_cpu(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    lib.tallynet_popcount_cpu.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.tallynet_popcount_cpu.restype = None
    return lib


def _bind_cuda(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    lib.tallynet_popcount_cuda.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.tallynet_popcount_cuda.restype = ctypes.c_char_p
    return lib


def build_native(
    *,
    out_dir: Optional[Path] = None,
    march: Optional[str] = None,
    cuda: bool = False,
    force: bool = False,
) -> dict[str, Optional[Path]]:
    """Compile kernels into ``out_dir``. Returns paths that exist."""
    dest = Path(out_dir) if out_dir is not None else default_cache_dir()
    dest.mkdir(parents=True, exist_ok=True)
    isa = march or march_flag()
    csrc = package_dir() / "csrc"
    cpu_src = csrc / "popcount_cpu.cpp"
    cu_src = csrc / "popcount_cuda.cu"
    cpu_so = dest / CPU_SONAME
    cu_so = dest / CUDA_SONAME
    built: dict[str, Optional[Path]] = {"cpu": None, "cuda": None}

    def stale(src: Path, so: Path) -> bool:
        return force or (not so.is_file()) or src.stat().st_mtime > so.stat().st_mtime

    if cpu_src.is_file() and stale(cpu_src, cpu_so):
        if _compile_cpu(cpu_src, cpu_so, march=isa, openmp=True):
            built["cpu"] = cpu_so
    elif cpu_so.is_file():
        built["cpu"] = cpu_so

    want_cuda = cuda or (os.environ.get("TALLYNET_CUDA", "0") in ("1", "true", "True"))
    if want_cuda and cu_src.is_file() and stale(cu_src, cu_so):
        if _compile_cuda(cu_src, cu_so):
            built["cuda"] = cu_so
    elif cu_so.is_file():
        built["cuda"] = cu_so

    reset()
    return built


def reset() -> None:
    global _CPU, _CUDA, _TRIED
    _CPU = None
    _CUDA = None
    _TRIED = False


def _load_libs() -> None:
    global _CPU, _CUDA, _TRIED
    if _TRIED:
        return
    _TRIED = True
    if not native_enabled():
        return

    cpu_path = find_library(CPU_SONAME)
    if cpu_path is None:
        cache = Path(os.environ["TALLYNET_KERNEL_DIR"]).expanduser() if os.environ.get("TALLYNET_KERNEL_DIR") else default_cache_dir()
        built = build_native(out_dir=cache, march=march_flag(), cuda=False)
        cpu_path = built.get("cpu")
        # build_native() calls reset(); mark this load as in progress again
        _TRIED = True

    if cpu_path is not None:
        try:
            _CPU = _bind_cpu(cpu_path)
        except OSError:
            _CPU = None

    if torch.cuda.is_available():
        cu_path = find_library(CUDA_SONAME)
        if cu_path is None:
            cache = Path(os.environ["TALLYNET_KERNEL_DIR"]).expanduser() if os.environ.get("TALLYNET_KERNEL_DIR") else default_cache_dir()
            built = build_native(out_dir=cache, march=march_flag(), cuda=True)
            cu_path = built.get("cuda")
            _TRIED = True
        if cu_path is not None:
            try:
                _CUDA = _bind_cuda(cu_path)
            except OSError:
                _CUDA = None


def load_native(*, verbose: bool = False) -> Optional[Any]:
    """Load CPU (and CUDA, if available) libraries. Returns the CPU lib or None."""
    _load_libs()
    if verbose:
        info = status()
        print(
            f"tallynet native cpu={info['cpu']} cuda={info['cuda']} "
            f"cpu_path={info['cpu_path']} cuda_path={info['cuda_path']}",
            file=sys.stderr,
        )
    return _CPU


def status() -> dict[str, Any]:
    _load_libs()
    return {
        "enabled": native_enabled(),
        "cpu": _CPU is not None,
        "cuda": _CUDA is not None,
        "cpu_path": str(find_library(CPU_SONAME) or ""),
        "cuda_path": str(find_library(CUDA_SONAME) or ""),
        "march": march_flag(),
        "search": [str(p) for p in kernel_search_dirs()],
    }


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


def main(argv: Optional[list[str]] = None) -> None:
    raise SystemExit(_main(argv))


def _main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="tallynet-native", description="Build or inspect TallyNet kernels")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="compile CPU (and optional CUDA) kernels")
    b.add_argument("--out", type=Path, default=None, help="output directory (default: .kernel_cache)")
    b.add_argument("--march", default=None, help="g++ -march (default: $TALLYNET_MARCH or native)")
    b.add_argument("--cuda", action="store_true", help="also try nvcc")
    b.add_argument("--force", action="store_true", help="rebuild even if .so exists")

    sub.add_parser("info", help="print which libraries would load")

    args = p.parse_args(argv)
    if args.cmd == "build":
        out = args.out or default_cache_dir()
        built = build_native(out_dir=out, march=args.march, cuda=args.cuda, force=args.force)
        for kind, path in built.items():
            print(f"{kind}: {path or '(missing)'}")
        if built.get("cpu") is None:
            return 1
        return 0
    info = status()
    for k, v in info.items():
        print(f"{k}: {v}")
    return 0 if info["cpu"] or not info["enabled"] else 1


if __name__ == "__main__":
    main()
