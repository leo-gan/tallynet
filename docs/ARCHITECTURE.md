# How a TallyNet layer works

See [TALLYNET_NAMING.md](TALLYNET_NAMING.md) for names. **One bit is one parameter.** A tally weight is a **group of `S` bits** on one connection. The layer turns that group into one number by counting.

## Weights

For each connection `(i, j)`:

1. Store `S` bits, each `+1` or `-1` (`bits[i, j, :]`). Each bit is a parameter.
2. Count them: `s = sum of the bits` (same as “how many are `+1`”).
3. Turn the count into a number `w` with a small rule (below).
4. Use `w` in the usual layer multiply (optionally scaled by `1 / sqrt(number of inputs)`).

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
| `TallyWriteback` | Optional training helper: step a trainer on `w`, then flip bits |

## Not part of “the architecture”

- Which trainer you pick (SGD vs Adam)
- How often bits flip
- Which dataset or how long you train

Those live in `writeback` and `cli` as demos.
