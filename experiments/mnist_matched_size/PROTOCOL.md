# Experiment: MNIST float NN vs TallyNet (matched training size)

**Path:** `experiments/mnist_matched_size/`  
**Runner:** `experiments/mnist_matched_size/train.py`  
**Raw results:** `experiments/mnist_matched_size/results/latest.csv` (+ `latest.manifest.json`)

## Question

On MNIST, with the **same training memory budget** (weights + gradients + Adam state + activations), does TallyNet beat a regular float MLP on test accuracy?

Also: as budget grows, does the Tally–float **gap keep closing**, and **where does it stop improving**?

## Setup (short)

| Item | Choice |
|------|--------|
| Task | MNIST, batch 128, 5 epochs, Adam `lr=1e-3`, 3 seeds |
| Float arm | `FloatMLP` 784→`H`→10, no bias, float32 |
| Tally arm | `TallyMLP` same depth, packed bits, writeback Adam, `encoder=majority` |
| Match | Analytic budget: W + grad + Adam + activations |
| Tally under budget | Max hidden width that still fits |
| Budget ladder | float `H ∈ {64, 128, 256, 512, 1024}` |
| Tally `S` | `{8, 32}` on the ladder (`S=256` only in an earlier small run) |
| Device | CPU; native SIMD popcount on |
| Plateau rule | gap change &lt; **0.3 pp** between successive budgets |

**Note:** Today’s Tally train path still holds a float group weight `w` plus Adam **per group**. Iso-memory width stays near the float net; large `S` adds bits *inside* groups and shrinks width.

## Size-matched configs (scale ladder)

| Float `H` | Budget (KB) | `S` | Tally `H` | Tally bits | vs float params |
|----------:|------------:|----:|----------:|---------------:|----------------:|
| 64 | 1255 | 8 | 60 | 381k | ~7.5× |
| 64 | 1255 | 32 | 51 | 1.30M | ~25× |
| 128 | 2113 | 8 | 121 | 769k | ~7.6× |
| 128 | 2113 | 32 | 103 | 2.62M | ~26× |
| 256 | 3829 | 8 | 242 | 1.54M | ~7.6× |
| 256 | 3829 | 32 | 207 | 5.26M | ~26× |
| 512 | 7261 | 8 | 484 | 3.07M | ~7.6× |
| 512 | 7261 | 32 | 415 | 10.5M | ~26× |
| 1024 | 14125 | 8 | 968 | 6.15M | ~7.6× |
| 1024 | 14125 | 32 | 831 | 21.1M | ~26× |

## Results — accuracy by budget

Mean **best test accuracy** ± std over seeds `{0,1,2}`:

| Budget `H` | float | tally `S=8` | tally `S=32` |
|----------:|------:|------------:|-------------:|
| 64 | **0.9704 ± 0.0019** | 0.9243 ± 0.0015 | 0.9336 ± 0.0028 |
| 128 | **0.9757 ± 0.0010** | 0.9388 ± 0.0023 | 0.9436 ± 0.0024 |
| 256 | **0.9774 ± 0.0019** | 0.9450 ± 0.0032 | 0.9528 ± 0.0020 |
| 512 | **0.9799 ± 0.0011** | 0.9498 ± 0.0042 | 0.9554 ± 0.0026 |
| 1024 | **0.9810 ± 0.0005** | 0.9537 ± 0.0025 | 0.9557 ± 0.0010 |

## Results — gap vs float (does it keep closing?)

Gap = (tally − float) in percentage points. **Δgap** = change from previous budget (positive = closer to float).

### Best arm: `S=32`

| `H` | float acc | tally acc | gap (pp) | Δgap (pp) |
|----:|----------:|----------:|---------:|----------:|
| 64 | 0.9704 | 0.9336 | −3.68 | — |
| 128 | 0.9757 | 0.9436 | −3.21 | **+0.47** |
| 256 | 0.9774 | 0.9528 | −2.46 | **+0.75** |
| 512 | 0.9799 | 0.9554 | −2.46 | **+0.00** |
| 1024 | 0.9810 | 0.9557 | −2.53 | −0.07 |

**Plateau:** from **`H=512`**, gap stops improving (change &lt; 0.3 pp). Doubling again to 1024 does not help; gap is essentially flat (~−2.5 pp).

### `S=8`

| `H` | float acc | tally acc | gap (pp) | Δgap (pp) |
|----:|----------:|----------:|---------:|----------:|
| 64 | 0.9704 | 0.9243 | −4.61 | — |
| 128 | 0.9757 | 0.9388 | −3.69 | +0.92 |
| 256 | 0.9774 | 0.9450 | −3.23 | +0.46 |
| 512 | 0.9799 | 0.9498 | −3.01 | +0.22 |
| 1024 | 0.9810 | 0.9537 | −2.73 | +0.28 |

**Plateau (0.3 pp rule):** also flagged from **`H=512`** (steps &lt; 0.3 pp). Absolute gap still creeps down slowly; by 1024 it is still ~−2.7 pp and never crosses float.

## Takeaways

1. **Float wins at every budget** on this protocol; Tally never catches up within `H≤1024`.
2. **Gap closes with budget only up to a point.** For best Tally (`S=32`), closing **stops around `H=256→512`**; beyond that the gap is stuck near **−2.5 pp**.
3. Tally **accuracy** still edges up with width (`S=32`: 0.934 → 0.956), but **float rises too**, so the *relative* deficit plateaus.
4. **`S=32` stays better than `S=8`** at every budget on the ladder.
5. Extra budget alone (same train recipe) is **not enough** for Tally to match float under this memory accounting; the limit looks algorithmic / capacity, not “need a bit more H”.

## Reproduce

```bash
# full budget ladder (H=64..1024, S=8,32)
uv run --extra train python experiments/mnist_matched_size/train.py --scale

# original small grid (includes S=256)
uv run --extra train python experiments/mnist_matched_size/train.py

# smoke
uv run --extra train python experiments/mnist_matched_size/train.py --quick

uv run pytest -q experiments/mnist_matched_size
```

Outputs stay in `experiments/mnist_matched_size/results/` (`latest.csv`, `latest.manifest.json`). Do not write into another experiment’s folder.
