# Experiment: CIFAR-10 scale-gap (BitNet transfer)

**Path:** `experiments/cifar_scale_gap/`  
**Runner:** `experiments/cifar_scale_gap/train.py`  
**Raw results:** `experiments/cifar_scale_gap/results/latest.csv` (+ `latest.manifest.json`)

## Question

On CIFAR-10, as the network gets wider, does TallyNet’s test-accuracy deficit vs a standard float MLP **shrink** (BitNet-style), or does it **plateau away from zero** the way it did on MNIST?

This is **iso-shape** (same `H`, same depth), not iso-memory. The MNIST study (`experiments/mnist_matched_size/`) already lost the memory-budget bet. Memory is logged here, not matched.

We are not replicating BitNet (Wang et al. 2023). We ask whether the same qualitative pattern — gap vs full precision narrows as size grows — appears for TallyNet vs float MLP on CIFAR-10.

## Why CIFAR

MNIST float is already 97–98%; the gap froze near −2.5 pp from `H=512`. A flat CIFAR-10 MLP still has headroom (parent-repo cells ~0.38–0.51), so both arms can keep improving as width grows.

## Architecture

Flatten → Linear(3072→`H`) → ReLU → Linear(`H`→10). No bias, no LayerNorm, depth 1. `TallyMLP` / `FloatMLP` with `in_dim=3072`. Encoder `majority` (each connection’s forward value is `±1`). No TallyConv.

## Starting grid

| Knob | Value |
|------|--------|
| Data | `data/cifar-10-batches-py` (gitignored). Standard CIFAR-10 mean/std, no aug. |
| Match | Same `H`, depth, epochs, lr |
| Width | `H ∈ {128, 256, 512, 1024, 2048}` |
| `S` | `{32}` first. Then a cross at the best `H`: `{8, 32, 128, 256}` |
| Train | Adam `lr=1e-3`, batch 128, 20 epochs, seeds `{0,1,2}` |
| Device | CPU default |

The first table is a **start**. If the peak is not in view, **move the limits**.

## Finding the optimum

`δ = 0.5` pp.

| Quantity | Optimum |
|----------|---------|
| Tally acc vs `H` (fixed `S`) | first doubling that gains `< δ` |
| Tally acc vs `S` (fixed `H`) | first doubling that gains `< δ` |
| Gap vs `H` (BitNet) | first two successive doublings that change the gap by `< δ` |

After `--scale` the runner prints `EXPAND H` / `EXPAND S` / `STOP`.

- **Width:** if Tally acc still rose `≥ δ` on the last doubling **or** the gap still closed `≥ δ`, next `H ← 2H` (2048→4096→8192). Always train float at the new `H`.
- **`S`:** if only `S=32`, the `S` optimum is unknown — run the cross. If the largest `S` still gains `≥ δ` (or still closes the gap), next `S ← 2S` (256→512→1024). Attach the `S` search at the Tally-acc peak `H`, or at max `H` if width is still climbing.
- **Caps:** `H ≤ 8192`, `S ≤ 1024`, or one cell `> 4 h`. Hitting a cap with no plateau is **inconclusive/cap**, not an optimum.

`--expand-h` / `--expand-s` run only the next rung and **append** to the latest CSV.

## Reading (pre-registered)

Gap = (tally − float) in percentage points, mean best test acc over seeds. Evaluate at the largest `H` where both arms have a plateau-or-cap, not at a fixed 2048.

- **Support (weak):** gap moves toward 0 on at least two successive doublings, each closing `≥ δ`.
- **Support (strong):** at the `H` optimum, `|gap| < 1` pp.
- **Reject (MNIST-like):** two successive doublings change the gap by `< δ` **and** `|gap| > 2` pp.
- **Inconclusive:** hard cap with no plateau, or train loss shows one arm starved.

Always report train loss / train acc so a leftover gap can be labeled underfit vs capacity.

## Non-claims

- Not a BitNet replication (MLP vs Transformer, ~0.4–6M connections vs 125M–30B, writeback vs STE).
- Not the iso-memory project bet.
- Not a competitive CIFAR model (flat MLP, no aug).
- A residual gap can be the discrete trainer. An STE ±1 control is a follow-up.

## Reproduce

```bash
# smoke
uv run --extra train python experiments/cifar_scale_gap/train.py --quick

# starting width ladder at S=32
uv run --extra train python experiments/cifar_scale_gap/train.py --scale

# only if the decision line says so
uv run --extra train python experiments/cifar_scale_gap/train.py --expand-h
uv run --extra train python experiments/cifar_scale_gap/train.py --expand-s

# expand-rule unit tests (no full train)
uv run pytest -q experiments/cifar_scale_gap
```

Outputs stay in `experiments/cifar_scale_gap/results/` (`latest.csv`, `latest.manifest.json`). Do not write into another experiment’s folder.
