# What we call things

**Date:** 2026-08-09

This note is only about **names**: what a weight is, what a tally weight is, and why the project is called TallyNet. It is not about how to train.

The idea started in a sibling repo (`binary-optimizers`). You do not need that repo to use TallyNet.

---

## The names we use

| We say | We mean |
|--------|---------|
| **TallyNet** | This family of networks. |
| **Bit** / **weight** / **parameter** | One stored `+1` or `-1`. **This is one parameter.** The size of the model is the number of bits. |
| **Tally weight** | A **group of S bits** on one connection. They are counted together. When the layer runs, that group becomes one number (the count, turned into a value). A tally weight is **not** one parameter. It is a group of parameters. |
| **Tally** | How many bits in the group are `+1` (same information as adding the `+1`/`-1` values). |
| **Tally width** `S` | How many bits are in one group. Not “how wide the network is.” |
| **TallyLinear** | A linear layer that works this way. |

In one sentence:

> Each parameter is one bit. `S` bits on the same connection are a tally weight. We count them and use that count in the layer.

\[
w
=
\mathrm{enc}(\text{tally of }S\text{ bits}),
\qquad
\text{tally}
=
\text{number of }{+}1\text{s}
\]

Each of those `S` bits is a parameter. `w` is the group, turned into one number for the multiply. We do not store `w` as the real weight.

---

## Easy to mix up

| This | Is not |
|------|--------|
| A **bit** (one weight) | A digit inside some other weight |
| A **tally weight** (a group of bits) | One stored parameter |
| That **group** | A **batch of training examples** |

A layer with `out` outputs and `in` inputs stores `out × in × S` parameters and has `out × in` tally weights.

---

## What happens in the layer

```text
one bit  (+1 or -1)          ← one weight, one parameter
    │
    │  S bits share one connection
    ▼
tally weight  (the group)
    └── count how many are +1
          └── turn the count into a number
                └── that number is used in the layer
```

- Every bit in the group counts the same. They are not “the 1s place, the 2s place, the 4s place.”
- Only the **count** matters. Two groups with the same number of `+1`s are the same for the layer.
- A group of size `S` can only make `S+1` different counts.
- This is a way to **store weights**. It is not a committee of whole networks, and it is not several models trained at once.

How we *update* the bits (Adam, flipping bits, and so on) is a training choice. The name TallyNet does not include that.

---

## Why “tally”

- The main step is a **count** of equal bits.
- A tally mark is just another mark; no bit is worth more than another.
- Majority (“which side wins”) and a share (`count / S`) both read naturally as a tally.
- The name is not already used for a well-known kind of net.
- In code, `TallyLinear` and `tally_width=S` are easy to read.
- It names **how weights are stored**, not how they are trained.

We do **not** call this:

| Name | Why not |
|------|---------|
| BitSwarm / Swarm | Sounds like particle swarm; old name. |
| Popcount BitNet | Sounds like BitNet, and “popcount” already means a different hardware trick. |
| Popcount QAT | QAT is a training method, not a way to store weights. |
| Committee network | That already means a vote among whole models. |
| BallotNet | Informal; also a CUDA word. |
| Thermometer coding | Cousin idea; those bits are often ordered. Ours are not. |

“Popcount” is fine as a *description* (“the tally is a popcount of the group”). It is a bad *product name*.

---

## Words to prefer

| Prefer | Avoid |
|--------|--------|
| TallyNet | Swarm net |
| Bit, weight, parameter (all the same thing) | Digit, agent, sub-weight |
| Tally weight, group of bits | Calling a connection “a parameter” |
| Tally width `S` | Swarm size |
| Tally, count | Using only “popcount” as the name |
| Group of training examples | Bare “batch” when you mean a group of bits |

**How to count a model:** TallyNet size = number of **bits**. Normal-net size = number of **numbers**. “1B vs 1B” means 1 billion numbers vs 1 billion bits, not 1 billion connections on each side.

---

## Short text you can reuse

**Short:**

> TallyNet stores every parameter as one bit. A group of bits on one connection is a tally weight; we count them and use the count in the layer.

**A bit longer:**

> A normal layer treats one number as one parameter. TallyNet treats one bit as one parameter. Bits are grouped. The number the layer uses is a function of how many bits in the group are `+1`. How we train is a separate choice.

**Do not say, just from the name:**

- We invented a new hardware popcount.
- This is a new BitNet.
- This is a committee of models.
- This is a particle-swarm optimizer.
- A tally weight is one parameter. (It is a **group** of parameters.)

---

## Changes

| Date | Change |
|------|--------|
| 2026-08-07 | Chose the name TallyNet. |
| 2026-08-07 | Copied into this repo. |
| 2026-08-09 | One bit is one parameter. A tally weight is a group of bits. “1B” counts bits. |
| 2026-08-09 | Rewrote in plain language. |
