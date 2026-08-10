# SIMD and bit kernels for TallyNet

**Date:** 2026-08-09  
**Scope:** why popcount hardware helps, what we built, and what we measured on the current net.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the layer, [GOAL.md](GOAL.md) for the memory bet. This note is only about **compute**: counting bits and multiplying.

---

## The job is two steps, not one

For each connection:

1. Load `S` bits (one bit = one parameter).
2. **Count** how many are `+1` (a popcount if we store `0/1`).
3. Turn that count into one number `w`.
4. Do `y += x * w` as a normal multiply.

That is **not** a BitNet / XNOR–popcount matmul. Mixing the two leads to the wrong kernel.

| | TallyNet | Binary / BitNet GEMM |
|--|--|--|
| What you count | The **`S` bits of one weight** | `xnor(activation, weight)` along the inner dimension |
| How often | Once per connection | Once per multiply in the inner product |
| Then | A **normal** (or INT8) matmul with `w` | The count *is* the inner product |

Hardware popcount is for step 2. It does not replace step 4 unless you also binarize activations and use majority.

```text
packed bits [out, in, ceil(S/8)]
        │
        ▼
   popcount  ←  SIMD / __popc  (this note)
        │
        ▼
   w = enc(tally)     one number per group, not per bit
        │
        ▼
   stock GEMM         F.linear (float32 / bfloat16)
                      or torch._int_mm (majority, inference)
```

---

## Why SIMD / GPU bit ops help

The old code stored `int8` `±1` and did `bits.float().sum(-1)`. For a group of size `S` that:

- turns every bit into a 32-bit float
- adds `S` floats

A 1024×1024 layer at `S=256` is **256 million** floats just to count. The count itself is a popcount of packed bytes. That is what CPUs and GPUs already do well.

| Group size `S` | What counts one group | On this class of chip |
|----------------|------------------------|------------------------|
| 8 | popcount of one byte | AVX2 nibble LUT (`pshufb`); AVX-512 **`VPOPCNTB`** if present |
| 16 / 32 / 64 | popcount of 2 / 4 / 8 bytes | `POPCNT` / `__builtin_popcountll`; AVX-512 `VPOPCNTW/D/Q` |
| 256 | 32 byte popcounts, summed | same, 32 bytes per group |

`VPOPCNTB` (AVX-512 BITALG) is a perfect match for `S=8`: one 512-bit load is 64 tallies. **This laptop (i7-12800H) has no AVX-512** — 12th-gen consumer chips dropped it. The CPU kernel uses AVX2 `pshufb` + scalar `popcnt` instead. Same idea, fewer groups per instruction.

CUDA has `__popc` / `__popcll`. We ship that kernel; this machine’s driver is too old to run it.

After the count you still need a matmul. On a GPU, a fused

```text
acc += x[j] * enc(popc(bits[i, j]))
```

runs on CUDA cores. Large GEMMs want **tensor cores**. So the first design is: **popcount front-end + stock GEMM**, not one heroic bit-GEMM.

| Situation | Better kernel |
|--|--|
| Large batch, compute-bound | Decode once, then BF16/INT8 GEMM |
| Small batch / memory-bound | Fused popcount+FMA (later) |
| Training | Float/BF16 GEMM on **group** values so writeback can see `∂L/∂w` |
| Majority inference | Optional `torch._int_mm` |

Other hardware that actually fits:

- **Packing** — first 8× on storage; popcount instructions are wasted on `int8 ±1`.
- **AMX / AVX-512 VNNI / INT8 tensor cores** — after the count, `w` is a small integer (`0…S`).
- **`S+1` distinct values** — a later kernel can bucket columns by tally (not built yet).
- **Writeback** — bit flips are XOR + RNG, not SIMD popcount. Tensor cores do not help.

What does **not** help much: calling this “XNOR-popcount hardware”; fusing popcount into FP32 GEMM and expecting tensor-core speed; defaulting the first kernel to `S=256` if you care about `VPOPCNTB` mapping 1:1 (`S=8`/`16` is the friendly knob).

---

## What is in the tree

| Piece | Where |
|--|--|
| Packed `uint8` storage, `1` → `+1` | `tallynet/packed.py`, `TallyLinear.bits` |
| Table popcount (always works) | `popcount_lut` |
| CPU SIMD library (`g++ -march=native`) | `tallynet/csrc/popcount_cpu.cpp` |
| CUDA `__popc` (built when CUDA runs) | `tallynet/csrc/popcount_cuda.cu` |
| Loader | `tallynet/native.py` (`TALLYNET_NATIVE=0` forces the table) |
| GEMM | `tallynet/kernels.py` — `F.linear` or `_int_mm` |
| Packed writeback | `tallynet/writeback.py` |

The native lib is a C ABI `.so` (ctypes). It does not need Python headers. First use compiles into `.kernel_cache/`.

---

## Measured on the current net

The current net is **TallyMLP**: MNIST-shaped `784 → hidden → 10`, default `hidden=128`, `S=256`, majority. Same machine as this note (CPU only, 4 threads). Median of repeated forwards.

| Setup | Old (`±1` float-sum) | Table popcount | **SIMD popcount** | vs old | vs table |
|--|--:|--:|--:|--:|--:|
| Default demo, `hidden=128`, `S=256`, batch 128 | 153 ms | 10.6 ms | **0.39 ms** | **~400×** | **~27×** |
| Same, `S=8` | 1.85 ms | 0.32 ms | **0.33 ms** | ~6× | ~1× |
| Wider, `hidden=512`, `S=256`, batch 128 | 888 ms | 37 ms | **1.2 ms** | **~730×** | **~31×** |

At default `S=256`, **counting used to be the whole forward**. SIMD makes decode cheap; the remaining ~0.4 ms is mostly the small GEMM.

At `S=8` the old float-sum is already small, and SIMD vs table is a wash — the multiply dominates.

**Training** (forward + backward + packed writeback, default demo): **~123 ms**. Writeback still walks `S` bit positions per group. The SIMD kernel speeds the **forward count**, not the flip loop. At `S=256` that loop is the next bottleneck.

Isolated decode, 1024×1024 weights (from `scripts/bench_decode.py`):

| `S` | Old float-sum | SIMD | Packed size vs `int8 ±1` |
|--|--:|--:|--:|
| 8 | 5.9 ms | 0.09 ms | 1/8 |
| 64 | 27 ms | 0.58 ms | 1/8 |
| 256 | 100 ms | 0.71 ms | 1/8 |

Re-run:

```bash
python scripts/bench_decode.py
pytest -q tests/test_simd_speed.py
```

The speed test **fails** if native is built and the default TallyMLP is not clearly faster than float-sum and the table. If `g++` cannot build the `.so`, that test skips; correctness tests still use the table.

---

## What this does *not* claim

- Inference-time weight size (0.125 GB / 1B bits) is not training memory. See [GOAL.md](GOAL.md).
- Activations are unchanged. SIMD does not shrink them.
- We did not build a fused tally-GEMM or a tensor-core inner loop.
- AVX-512 `VPOPCNTB` is the right instruction when the chip has it; this box does not.

---

## Next, if experiments need more speed

1. Packed writeback without a Python loop over `S` (training, large `S`).
2. Fused popcount+FMA for tiny batches.
3. Bucketed GEMM that uses only `S+1` distinct `w` values.
4. CUDA path on a machine whose driver matches the PyTorch wheel.
