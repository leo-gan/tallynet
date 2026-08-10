# TallyNet

Weights stored as bits. Each **parameter is one bit** (`+1` or `-1`). A group of `S` bits on one connection is a **tally weight**: we count how many are `+1` and use that count in the layer.

```text
packed bits  →  popcount  →  turn count into a number  →  stock GEMM
```

This repo is about **that way of storing weights**. Training helpers exist so demos run; they are not the claim.

**Goal:** use less memory to train, so more weights fit on the computer you already have, and the model can be smarter. The bet is that a TallyNet trained on a given machine can beat a normal network that has to fit in the same memory. See [docs/GOAL.md](docs/GOAL.md). Names: [docs/TALLYNET_NAMING.md](docs/TALLYNET_NAMING.md). SIMD / popcount kernels: [docs/SIMD.md](docs/SIMD.md).

## Install

Uses **[uv](https://docs.astral.sh/uv/)** (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
# venv + package + C++ popcount kernel
./scripts/install.sh
./scripts/install.sh --train        # + torchvision (MNIST)
./scripts/install.sh --cpu-torch    # PyTorch CPU wheels

# or by hand
uv sync --extra train --extra dev
./scripts/build_native.sh
```

Full install / wheel / CI notes: [docs/INSTALL.md](docs/INSTALL.md).

PyTorch CPU wheels: if install is slow or wrong platform, install `torch` / `torchvision` from the [PyTorch index](https://pytorch.org/get-started/locally/) first (`uv pip install torch --index-url https://download.pytorch.org/whl/cpu`).

Bits are stored packed (`uint8`). The first tally uses a small native popcount library (CPU SIMD; CUDA `__popc` when the driver works). Set `TALLYNET_NATIVE=0` to force the Python table. `TallyLinear(..., compute="bfloat16")` or `compute="int8"` (majority, inference) picks the GEMM.

GitHub Actions builds portable kernels (`-march=x86-64-v3`) and uploads **wheels**, **`.so` files**, and **test reports** as artifacts. Tags `v*` also publish a GitHub Release.

## Quick start (library)

```python
import torch
from tallynet import TallyLinear, TallyMLP

layer = TallyLinear(64, 32, tally_width=128, encoder="fixed")
y = layer(torch.randn(8, 64))

model = TallyMLP(hidden_dim=128, tally_width=256, encoder="majority")
logits = model(torch.randn(4, 1, 28, 28))  # MNIST-shaped
```

## Train MNIST demo

```bash
# downloads MNIST into ./data
uv run tallynet-mnist --epochs 5 --encoder majority --opt adam --tally-width 256
# or
uv run python -m tallynet.cli --epochs 5
```

Storage stays discrete (`bits` in `{±1}`); the demo steps Adam/SGD on the encoded weight and writes back with stochastic bit flips (`TallyWriteback`).

## Tests

```bash
uv run pytest -q
# no dataset required
```

## Layout

```
tallynet/           # library
  csrc/             # C++ / CUDA popcount
  native.py         # load / JIT / CLI
  ...
scripts/
  install.sh        # venv + kernels
  build_native.sh
  package.sh        # wheel + portable .so
.github/workflows/  # CI test + artifacts; tag release
docs/INSTALL.md
```

## Relation to `binary-optimizers`

This repo was **extracted and renamed** from research in [`binary-optimizers`](../binary-optimizers) (unary link / Swarm ladder: sum encoder, multi-bit per entry). That project also studies discrete optimizers and place-value registers. **TallyNet** is a clean home for the **weight representation** only.

| Concept (old) | TallyNet |
|---------------|----------|
| Swarm / agents | Bit bag / bits |
| Swarm size \(S\) | Tally width \(S\) |
| Link value | Tally weight \(w\) |
| `UnaryLinkLinear` | `TallyLinear` |
| `BitSwarmLinear` | Legacy name in parent repo |

## License

MIT (see `LICENSE` if present; otherwise all rights reserved by authors until set).
