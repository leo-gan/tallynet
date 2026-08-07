# TallyNet architecture

See also [TALLYNET_NAMING.md](TALLYNET_NAMING.md).

## Weight representation

For each matrix entry \((i,j)\):

1. Store **tally width** \(S\) bits \(a_k \in \{\pm 1\}\) (buffer `bits[i,j,:]`).
2. Compute **tally** \(s = \sum_k a_k\) (equivalently: popcount of \(+1\)).
3. Encode \(w_{ij} = \mathrm{enc}(s)\).
4. Use \(w_{ij}\) (optionally scaled by \(1/\sqrt{\mathrm{fan\_in}}\)) in the matmul.

Bits are **equal**: no place-value roles. Same tally ⇒ same \(w\) (unordered multiset).

### Built-in encoders

| Name | Formula |
|------|---------|
| `fixed` | \(s/S\) |
| `majority` | \(\mathrm{sign}(s)\) (ties → \(+1\)) |
| `tanh` | \(\tanh(s/\tau)\) |
| `signed_sqrt` | \(\mathrm{sign}(s)\sqrt{|s|/S}\) |

## Module map

| Symbol | Role |
|--------|------|
| `TallyLinear` | Tally-coded linear layer |
| `TallyMLP` | Small MLP built from `TallyLinear` |
| `encode_tally` | Encoder functions |
| `TallyWriteback` | Optional training: Adam/SGD on \(w\) + bit flips |

## What is *not* architecture

- Continuous optimizer choice (SGD vs Adam)
- Flip probability schedule / noise floor
- Dataset or train budget protocol

Those live under `writeback` and `cli` as demos.
