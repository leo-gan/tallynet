# Install, build, and deploy

TallyNet is a Python package plus a small **C++** popcount library (optional CUDA). Python still runs without the `.so` (table fallback). The SIMD kernel is what you want for experiments.

## One-command install (developer machine)

Needs Python ≥ 3.12, `g++`, and (for the demo) a working venv.

```bash
./scripts/install.sh           # venv + package + CPU kernel
./scripts/install.sh --train   # also torchvision (MNIST)
./scripts/install.sh --cpu-torch   # PyTorch CPU wheels (no NVIDIA)
```

Then:

```bash
source .venv/bin/activate
python -m tallynet.native info
pytest -q
```

Or `make install` / `make test`.

## What gets built

| File | Role |
|------|------|
| `tallynet/csrc/popcount_cpu.cpp` | AVX2 / `popcnt` kernel, C ABI |
| `tallynet/csrc/popcount_cuda.cu` | `__popc` kernel (optional) |
| `.kernel_cache/libtallynet_popcount_cpu.so` | local JIT / `make native` |
| `tallynet/lib/*.so` | copied into wheels by `scripts/package.sh` |

Search order at import: `$TALLYNET_KERNEL_DIR` → `tallynet/lib/` → `.kernel_cache/` → compile.

```bash
./scripts/build_native.sh
./scripts/build_native.sh --out tallynet/lib --march x86-64-v3
./scripts/build_native.sh --cuda --force
python -m tallynet.native build --out .kernel_cache --march native
```

| Variable | Meaning |
|----------|---------|
| `TALLYNET_NATIVE=0` | never load/compile; use the Python table |
| `TALLYNET_MARCH` | `g++ -march` (default `native`; CI uses `x86-64-v3`) |
| `TALLYNET_KERNEL_DIR` | where to find/write `.so` files |
| `TALLYNET_CUDA=1` | try `nvcc` when building |
| `CXX` / `NVCC` | compiler overrides |

**Do not ship `-march=native` artifacts.** That is only for the box you compile on. Release / CI kernels use `x86-64-v3` (AVX2 + POPCNT, Haswell and later).

## Deployable artifacts

```bash
./scripts/package.sh
```

Writes:

- `dist/tallynet-*.whl` and `dist/tallynet-*.tar.gz`
- `artifacts/native/libtallynet_popcount_cpu.so`

Install the wheel on another Linux x86_64 machine with a compatible glibc:

```bash
pip install dist/tallynet-*.whl
python -m tallynet.native info
```

The sdist has the C++ sources; the first import compiles if no `.so` is on the search path.

## CI / CD

GitHub Actions:

| Workflow | When | Artifacts |
|----------|------|-----------|
| [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) | push / PR to `main` | `test-report-*.xml`, `python-packages` (wheel+sdist), `native-kernels-linux-x86_64` |
| [`.github/workflows/release.yml`](../.github/workflows/release.yml) | tag `v*` | same files on the GitHub Release |

Download CI kernels from the Actions run → Artifacts, or set `TALLYNET_KERNEL_DIR` to that folder.

## MNIST data

`data/` is gitignored. Copy a local set (no download):

```bash
cp -a ../binary-optimizers/data/MNIST data/
tallynet-mnist --data-dir data --epochs 5
```

## Makefile

```text
make install      # scripts/install.sh
make native       # compile kernels
make test         # pytest + junit in artifacts/
make wheel        # portable wheel + kernel
make clean
```
