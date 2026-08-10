# How a TallyNet layer works

See [TALLYNET_NAMING.md](TALLYNET_NAMING.md) for names. **One bit is one parameter.** A tally weight is a **group of `S` bits** on one connection. The layer turns that group into one number by counting.

## Weights

For each connection `(i, j)`:

1. Store `S` bits, each `+1` or `-1`. Each bit is a parameter. In memory they are **packed**: `bits` is `uint8 [out, in, ceil(S/8)]` (`1` → `+1`, `0` → `-1`).
2. Count them with a popcount (CPU SIMD or CUDA `__popc` when the native extension built; otherwise a 256-entry table). `s = 2 * popcount - S`.
3. Turn the count into a number `w` with a small rule (below).
4. Multiply with a stock GEMM: `F.linear` in float32 or bfloat16 (training). Optional `compute='int8'` (majority, inference) uses `torch._int_mm`.

All bits in a group count the same. Same count ⇒ same `w`. The layer’s parameter count is `out × in × S`, not `out × in`.

### Rules that turn a count into a number

| Name | Rule |
|------|------|
| `fixed` | `s / S` |
| `majority` | sign of `s` (a tie becomes `+1`) |
| `tanh` | `tanh(s / τ)` |
| `signed_sqrt` | sign of `s`, times `sqrt(|s| / S)` |

## Code map

| Name | Role |
|------|------|
| `TallyLinear` | Linear layer that stores bits this way |
| `TallyMLP` | Small network built from `TallyLinear` |
| `encode_tally` | The rules in the table above |
| `packed` / `native` | Pack bits; popcount kernels |
| `TallyWriteback` | Optional training helper: step a trainer on `w`, then flip bits |

## Not part of “the architecture”

- Which trainer you pick (SGD vs Adam)
- How often bits flip
- Which dataset or how long you train

Those live in `writeback` and `cli` as demos.
