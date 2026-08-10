# What this project is for

**Date:** 2026-08-09

This note says **why TallyNet exists**. How a layer is built is in [ARCHITECTURE.md](ARCHITECTURE.md). What we call things is in [TALLYNET_NAMING.md](TALLYNET_NAMING.md).

---

## The idea in one breath

In TallyNet, **one weight is one bit** (`+1` or `-1`). That bit is one parameter of the model.

Several bits that sit on the same connection are **counted together**. That group is the **tally weight**. The layer uses the count (how many are `+1`), not each bit on its own.

Because each weight is a bit, not a full number, **training should need less memory**. On the same computer you can fit **more weights**. More weights can mean a **smarter** model. We think a TallyNet trained on a given machine can **beat a normal network that has to live in the same memory**.

That last sentence is a **bet**, not a result. This project exists to test it.

---

## How we count

| In a normal network | In TallyNet |
|---------------------|-------------|
| One parameter is one number (often 16 or 32 bits of memory). | One parameter is **one bit**. |
| The number is what the layer multiplies with. | Bits are stored. A **group of S bits** is counted. That count becomes the number the layer multiplies with. |
| “1 billion parameters” means 1 billion numbers. | “1 billion parameters” means **1 billion bits**. |

`S` is how many bits share one connection. We call `S` the **tally width**. It is a group size, not “how wide the network is.”

Do not count the *groups* and call that the size of the model. The size of the model is the number of **bits**.

---

## Why memory decides how smart the model can be

While you train, the computer has to hold:

- the weights
- the gradients (how to change the weights)
- extra numbers some trainers keep (Adam keeps two extra full numbers for every weight)
- activations (values from the forward pass, needed to go backward)

That often fills the GPU long before you run out of compute. So the network you *can* train is the one that **fits**, not the one you wish you had.

Many “binary” methods still keep a full number for every weight *during training*, and only use bits when they run the model later. The saved model is small. **Training is not.**

TallyNet’s difference:

- what you store and train is the **bits**
- the number used in the layer is just **the count of a group**, made when needed
- you should not also keep a full number for every bit

A normal weight often costs 16–32 bits, plus extra trainer numbers. A TallyNet weight costs **1 bit**. Same memory, more weights. That is the path to a smarter model on the same machine.

---

## 1 billion vs 1 billion

Same number of parameters: **1 billion numbers** vs **1 billion bits**.

| | Normal 1B | TallyNet 1B |
|--|-----------|-------------|
| What “1B” means | 1 billion numbers | 1 billion bits |
| How many groups / connections | 1 billion (one number each) | 1 billion ÷ `S` (a tally weight is a group of `S` bits) |

Two kinds of memory:

| Piece | What it is | Depends on bit vs number? |
|-------|------------|---------------------------|
| **Weights** (and trainer extras) | The model, plus gradients and Adam extras while training | **Yes.** This is what TallyNet changes. |
| **Activations** | Layer outputs you keep. Training keeps them to go backward. Inference keeps a working set (and, for a language model, a KV cache). | **No**, not if the network shape and data batch are the same. |

The tables that only show 4 GB / 0.125 GB / 16 GB / 2.1 GB are **weight (and trainer) memory**. They left activations out on purpose: at the **same** width, depth, batch, and sequence length, activations cost the **same** on both sides, so they cancel in a 1B-vs-1B *difference*. They do **not** cancel in **peak** memory, and they grow if extra bits are used to make a wider or deeper net. They are added back in “Activations and peak memory” below.

**0.125 GB is not a training number.** It is the size of 1 billion packed bits — the **saved model** and the **inference-time size of the weights**. Training is larger. The old label “what we want” meant this bits-only figure; it is listed under inference below, not under training.

### Saved model and inference-time weights

This is what you must keep in memory to **hold the model** and to **run it** (inference). No gradients. No Adam extras.

| Model | Memory for 1B parameters |
|-------|--------------------------:|
| Normal, 32-bit numbers | **4.0 GB** |
| Normal, 16-bit numbers | **2.0 GB** |
| Normal, 8-bit numbers | **1.0 GB** |
| TallyNet, bits packed tightly | **0.125 GB** |
| TallyNet as the code is today (one byte per bit) | **1.0 GB** |

Packed TallyNet is **32 times smaller** than a 32-bit 1B model (0.125 GB vs 4 GB). That does not depend on `S`. `S` only changes how those 1 billion bits are grouped.

If inference **builds a full number for every group** in order to multiply, peak weight memory is `0.125 + 4/S` GB, not 0.125 GB. The 0.125 GB figure is the bits alone — the resident model at inference if that extra tensor is not kept.

| Group size `S` | Groups in a 1B-bit model | Bits + one 32-bit number per group |
|----------------|-------------------------:|-----------------------------------:|
| 8 | 125 million | 0.625 GB |
| 16 | 62.5 million | 0.375 GB |
| 32 | 31 million | 0.250 GB |
| 256 | 3.9 million | 0.141 GB |

### Training

Training is **not** 0.125 GB. You still hold the bits, plus gradients, plus whatever extras the trainer keeps.

A common way to train 1 billion normal weights with Adam needs about **16 GB** (the weights, their gradients, and two extra numbers each).

TallyNet always stores the bits. Training *may* also build a temporary number per **group**, and Adam *may* keep extra numbers per **group** — not per bit.

| How we train 1B TallyNet bits | Memory (about) |
|------------------------------|----------------|
| Packed bits + a simple trainer (no Adam extras), with a number and a gradient per group | 0.125 + 8/`S` GB |
| Packed bits + Adam on each group (today’s trainer idea, bits packed) | 0.125 + 16/`S` GB |
| **Code today:** one byte per bit + Adam on each group | 1.0 + 16/`S` GB |

With a few group sizes:

| `S` | Packed + simple trainer | Packed + Adam | Code today | vs 16 GB normal + Adam |
|----:|------------------------:|--------------:|-----------:|------------------------|
| 8 | 1.1 | 2.1 | 3.0 | ~7× smaller with Adam on groups |
| 16 | 0.63 | 1.1 | 2.0 | ~14× |
| 32 | 0.38 | 0.63 | 1.5 | ~26× |
| 256 | 0.16 | 0.19 | 1.1 | ~85× |

These are **planned numbers**, not a benchmark we have run. Even the code as it is now, with 1 billion **bits**, uses much less **training** memory than 1 billion **numbers**. Bigger `S` makes Adam cheaper because Adam runs on fewer groups. It does not change how many parameters you have.

There is no honest “training = bits only” row. Bits only is inference (and the saved file).

### Same 16 GB: how many parameters fit

Two different 16 GB questions. Do not mix them.

**Inference / saved weights** — 16 GB of packed bits:

| What you hold in 16 GB | Parameters that fit | vs 1B normal numbers (4 GB file) |
|------------------------|--------------------:|----------------------------------|
| Normal 32-bit numbers | 4.0 billion numbers | 4× |
| TallyNet, packed bits (**inference-time weight size**) | **128 billion bits** | **128×** vs 1B numbers; **32×** vs the 4 GB a 1B 32-bit file uses |

**Training** — 16 GB is about what 1 billion normal weights + Adam need. How many TallyNet bits can you **train** in that same 16 GB?

| What you train in 16 GB | Parameters that fit | vs 1B normal + Adam |
|-------------------------|--------------------:|---------------------|
| Normal 32-bit + Adam | 1.0 billion numbers | 1× |
| Normal 16-bit + Adam | ~1.3 billion numbers | 1.3× |
| TallyNet, packed + Adam, `S = 8` | ~7.5 billion bits | ~7.5× |
| TallyNet, packed + Adam, `S = 32` | ~26 billion bits | ~26× |
| TallyNet, packed + Adam, `S = 256` | ~85 billion bits | ~85× |
| Code today, `S = 8` | ~5 billion bits | ~5× |
| Code today, `S = 256` | ~15 billion bits | ~15× |

The project’s bet is about **training** on the same machine: more bits trained, better model. The 128 billion figure is **inference** (how many packed bits fit in 16 GB). It is not how many you can train.

You can spend extra bits on **more connections** or on **larger groups**. Example if you **train** ~7.5 billion bits with packed Adam at `S = 8` (the 16 GB training row above):

| Group size | Bits you can train in 16 GB (packed + Adam) | Connections |
|------------|--------------------------------------------:|------------:|
| 8 | ~7.5 billion | ~0.94 billion |
| 32 | ~26 billion | ~0.80 billion |
| 256 | ~85 billion | ~0.33 billion |

The bet is that those extra **trained** weights, on the same machine, make a **better** model than the 1 billion normal weights that machine could otherwise train.

Those “how many bits fit” rows still ignore activations. They are an upper bound on the **weight budget**. Once activations are in the same 16 GB, fewer extra bits fit — see below.

### Activations and peak memory

**Why they were omitted above.** Activations are not stored as tally bits. They are the numbers flowing through the net. At matched shape they are the same for a normal 1B and a TallyNet 1B. The first tables isolate the part TallyNet can shrink. That is not the same as peak memory on the GPU.

**Peak** is weights (and trainer) **plus** activations.

Activations scale with **batch × sequence × width × depth** (training must keep a history; inference can throw away a layer after it is used, but a language model still keeps a KV cache that grows with sequence length). They do **not** scale with `S` or with “bit vs 32-bit weight.”

#### Example shape (stated so the arithmetic can be checked)

About 1 billion weights: 20 layers, width 2048, 16 heads, MLP width 8192.  
Training: batch 2, sequence 2048, 16-bit activations, **no** rematerialization (keep what the backward pass needs).  
Inference: batch 1, sequence 2048, 16-bit activations, one layer at a time, plus a KV cache.

One 16-bit tensor of shape `[batch, seq, width]` at train batch 2 is  
`2 × 2048 × 2048 × 2 bytes = 16 MB`.

**Training activations, one layer** (kept for backward):

| Tensor | Shape (plain words) | Size |
|--------|---------------------|-----:|
| About nine width-sized copies (residuals, Q, K, V, …) | 9 × 16 MB | 144 MB |
| Attention scores | batch × heads × seq × seq | 256 MB |
| MLP hidden | batch × seq × MLP width | 64 MB |
| **One layer** | | **~464 MB** |
| **20 layers** | | **~9.3 GB** |

If you **rematerialize** (keep only each layer’s input, recompute the rest on the way back):  
`20 × 16 MB ≈ 0.3 GB`. That is a training trick, not part of TallyNet.

**Inference activations** (this example):

| Piece | Size |
|-------|-----:|
| Working set (one layer; attention scores dominate) | ~0.13 GB |
| KV cache (20 layers × K and V) | ~0.32 GB |
| **Inference activations** | **~0.45 GB** |

Longer context scales the KV cache linearly. At sequence 32k it is ~5 GB — already larger than 1B packed bits.

#### Peak, same 1B shape

| | Weights + trainer | Activations | **Peak** |
|--|------------------:|------------:|---------:|
| **Inference** | | | |
| 1B normal 32-bit numbers | 4.0 | 0.45 | **4.5 GB** |
| 1B TallyNet packed bits | 0.125 | 0.45 | **0.58 GB** |
| 1B TallyNet packed bits, also build a 32-bit number per group, `S=8` | 0.625 | 0.45 | **1.1 GB** |
| **Training, no rematerialization** | | | |
| 1B normal + Adam | 16.0 | 9.3 | **25.3 GB** |
| 1B TallyNet packed + Adam, `S=8` | 2.1 | 9.3 | **11.4 GB** |
| 1B TallyNet, code today, `S=8` | 3.0 | 9.3 | **12.3 GB** |
| **Training, with rematerialization** | | | |
| 1B normal + Adam | 16.0 | 0.3 | **16.3 GB** |
| 1B TallyNet packed + Adam, `S=8` | 2.1 | 0.3 | **2.4 GB** |

Read the table this way:

- **Same shape, inference:** activations are a small add-on here; the big cut is still the weights (4.5 GB → 0.58 GB). At long context the KV cache can dwarf packed weights.
- **Same shape, training, no rematerialization:** activations are ~9 GB on **both** sides. TallyNet still wins (25 GB → 11 GB), but it is about **2×**, not 7×. The 7× figure was weights+trainer only.
- **Same shape, training, rematerialization:** activations shrink to ~0.3 GB; the weight/trainer cut shows up almost in full (16 GB → 2.4 GB).

TallyNet does **not** shrink activations. Any claim about “how many more parameters fit on the same GPU” must leave room for them.

#### Same GPU, including activations

Take a GPU that can train the **example** 1B normal net **without** rematerialization: about **25 GB** peak (16 + 9.3).

If TallyNet keeps the **same shape**, peak drops to ~11 GB. The free ~14 GB can go to a larger batch (more activation memory, same weights) or a larger net (more weights **and** more activations).

You **cannot** spend all 14 GB on extra bits as if activations stayed 9.3 GB. Wider or deeper nets make activations grow. A rough bound: if you only grow depth and activations stay ~0.46 GB per layer, each new layer costs that plus its own weights. Extra width is more expensive, because every activation tensor grows.

So the earlier “~7.5 billion bits in 16 GB” row is an **upper bound on the weight budget alone**. With activations, the number of extra bits you can train on that GPU is smaller. How much smaller depends on batch, sequence, and whether you rematerialize — not on TallyNet’s bit packing.

### A fair slide

```text
Weights only (1B parameters) — activations not included
  Inference / saved
    1B normal numbers                     ████  4 GB
    1B TallyNet bits, packed              ▏     0.125 GB
  Training (weights + trainer)
    1B normal + Adam                      ████████████████  16 GB
    1B TallyNet packed + Adam, S=8        ██    2.1 GB

Peak, same example shape (weights + activations)
  Inference (seq 2048)
    1B normal                             ████▌ 4.5 GB
    1B TallyNet packed                    █     0.58 GB
  Training, no rematerialization
    1B normal + Adam                      █████████████████████████  25 GB
    1B TallyNet packed + Adam, S=8        ███████████  11 GB
```

**Wrong:** putting 0.125 GB on the training line. That number is inference-time (and saved) weight size.

**Wrong:** calling the weight-only 16 GB or 2.1 GB “peak training memory.” Add activations.

**Wrong:** calling “1B TallyNet” a network with 1 billion *connections*. That would be `S` billion bits — a bigger model, not 1B vs 1B.

---

## How this is supposed to help

1. Store bits. Each bit is a weight.
2. Group `S` bits on each connection (that group is the tally weight).
3. Count how many are `+1`.
4. Turn that count into the number the layer uses.

Why that helps:

- A bit is much smaller than a normal number, so more weights fit.
- The trainer can keep extra numbers **per group**, not per bit.
- You do not need a second full-number copy of every weight.
- `S` is a choice: larger groups give finer counts; more groups give a wider or deeper net.

---

## What would count as success

Success is a **measured** comparison on the same machine, not “we wrote a layer.”

**It works if:**

- Training really uses less memory per weight.
- On a fixed memory limit, a TallyNet with more weights scores better than the largest normal net that fits.
- The extra bits are doing something — a bigger TallyNet is actually better.

**It fails if:**

- Training still keeps a full number for every bit, so memory does not drop.
- Extra bits do not improve the score.
- Other memory (activations, temporary numbers) eats the savings. TallyNet does not shrink activations; see the peak-memory example.
- A simple binary baseline already does as well, with the same memory.

The small demos in this repo (MNIST, the current trainer) are tools for the test. They are not the test.

---

## What the code does today (honestly)

| In the code now | What that means |
|-----------------|-----------------|
| Bits are packed as `uint8` (`ceil(S/8)` bytes per group) | A parameter is 1 bit in storage. 1 billion parameters take **0.125 GB** packed. |
| Count uses a popcount kernel (CPU SIMD or CUDA `__popc`, with a table fallback) | Decode no longer turns every bit into a float. |
| The multiply is a stock GEMM on the **group** values (`F.linear`, or int8 majority at inference) | Training still builds one number per group (not per bit). That tensor must not stick around after the step. |
| The trainer (Adam or SGD) keeps extra numbers **per group** | That is fine. We must not do it **per bit**. |

None of this changes the goal. It is the gap between “the idea is written down” and “bigger, smarter models on the same computer.”

---

## What this project is not

- Not a claim that we invented a new chip instruction.
- Not a new brand of optimizer.
- Not “beat every model ever,” with unlimited memory.
- Not proven just because the layer exists.
- Not a license to count connections and call them parameters.

---

## One paragraph you can reuse

TallyNet stores every weight as one bit. A group of bits on the same connection is a tally weight: we count how many are `+1` and use that count in the layer. The goal is to need less memory to train, so more weights fit on the computer you already have, and the model can be smarter. The bet is that a TallyNet trained on a given machine will beat a normal network that has to fit in the same memory.

---

## Changes

| Date | Change |
|------|--------|
| 2026-08-09 | First version of the goal. |
| 2026-08-09 | Added 1B vs 1B memory numbers. |
| 2026-08-09 | Count bits, not connections: one bit is one parameter; a tally weight is a group of bits. |
| 2026-08-09 | Rewrote in plain language. |
| 2026-08-09 | 0.125 GB labeled as **inference / saved weights**, not training. Removed the “what we want = bits only” training row. |
| 2026-08-09 | Explained why activations were omitted; added peak-memory example (same 1B shape, with and without rematerialization). |
| 2026-08-09 | Packed storage and popcount kernels are in the tree; honesty table updated. |
