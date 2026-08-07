# TallyNet

**Tally-coded multi-bit weights** for neural nets.

Each matrix entry stores a bag of \(S\) equal \(\pm 1\) bits. The **tally** (how many are \(+1\), or equivalently the sum) is encoded into the scalar used in the matmul—unary multi-bit storage without place-value digits.

```text
bits [out, in, S]  →  tally  →  enc(tally) →  w  →  matmul
```

This repository is **architecture-first**. Training helpers exist so demos run; they are not the product claim. Naming rationale: [docs/TALLYNET_NAMING.md](docs/TALLYNET_NAMING.md).

## Install

```bash
cd tallynet
# with uv (recommended)
uv sync --extra train --extra dev

# or pip
pip install -e ".[train,dev]"
```

PyTorch CPU wheels: if install is slow or wrong platform, install `torch` / `torchvision` from the [PyTorch index](https://pytorch.org/get-started/locally/) first.

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
  encoders.py       # enc(tally) maps
  layers.py         # TallyLinear
  models.py         # TallyMLP
  writeback.py      # optional continuous step + bit flips
  data.py / cli.py  # MNIST demo
docs/
  TALLYNET_NAMING.md
  ARCHITECTURE.md
tests/
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
