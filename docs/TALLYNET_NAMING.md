# TallyNet — naming rationale

**Status:** direction decision (architecture naming)  
**Date:** 2026-08-07  
**Scope:** **network / weight representation only** — not optimizers, not training loops.

This document records **why** this project uses the name **TallyNet** (and related terms: tally-coded weights, `TallyLinear`).

Origin: naming discussion and unary multi-bit experiments in the sibling research repo `binary-optimizers` (not required to use TallyNet).

---

## 1. Decision

| Item | Choice |
|------|--------|
| **Brand / family name** | **TallyNet** |
| **Weight representation** | **Tally-coded multi-bit weights** |
| **Layer type** | **`TallyLinear`** |
| **Width hyperparameter** | **Tally width** \(S\) (number of equal bits per matrix entry) |
| **What we are naming** | How each matrix entry is **stored and decoded** into a scalar for the matmul |
| **What we are not naming here** | SGD/Adam, flip writeback, STE, BitNet training recipes |

**One-sentence architecture claim:**

> Each matrix entry stores a fixed bag of \(S\) equal \(\pm 1\) bits; the **tally** (how many are \(+1\), or equivalently the sum) is mapped by an encoder to the scalar used in the linear map.

\[
w_{ij}
=
\mathrm{enc}\bigl(\mathrm{tally}(a_{ij,1},\ldots,a_{ij,S})\bigr),
\qquad
\mathrm{tally}
=
\#\{k : a_{ij,k}=+1\}
\quad\text{(or } s=\sum_k a_{ij,k}\text{)}
\]

---

## 2. What TallyNet *is* (architecture)

### 2.1 Core objects

| Term | Meaning |
|------|---------|
| **Bit** | One stored unit \(\in \{\pm 1\}\) (or \(0/1\) if packed) on a matrix entry |
| **Tally width** \(S\) | Number of bits **per matrix entry** (not network width in neurons) |
| **Tally** | Count of \(+1\)s (equivalent information to sum of \(\pm 1\) bits) |
| **Tally encoder** | \(w = \mathrm{enc}(\mathrm{tally})\): e.g. fixed \(s/S\), majority \(\mathrm{sign}(s)\), \(\tanh\), … |
| **Tally weight** | Scalar \(w\) actually used in the matmul for that entry |
| **Tally-coded linear** | Linear map whose weights are tally-coded |

### 2.2 Forward sketch

```text
matrix entry (i, j)
  └── S equal bits  a₁…a_S ∈ {±1}
        └── tally  (popcount of +1  /  sum)
              └── enc(tally) → w_ij
                    └── used in y = x W
```

Implications:

- Bits on one entry have **equal place value** (unary), not \(2^i\) roles.
- Only the **count** matters for \(w\): patterns with the same tally share the same weight.
- At most \(S+1\) distinct tallies (hence at most that many distinct \(w\) for a monotone encoder).
- This is a **weight parameterization**, not an ensemble of full networks.

### 2.3 Explicitly out of this name

| Out of TallyNet-the-architecture | Why |
|----------------------------------|-----|
| Place-value / binary **register** weights | Different coding family |
| Classical **committee machines** (ensemble of nets) | Wrong scale (models vs bits) |
| **BitNet** as a product lineage | Different quant + STE + latent-\(W\) recipe |
| **QAT** as the name | Training regime, not representation |
| Discrete **optimizer** brands | Separate research axis |

Training utilities (`TallyWriteback`: Adam/SGD on \(w\) + stochastic bit flips) may be used in demos; they are **not** part of the architecture name.

---

## 3. Why “Tally” / “TallyNet”

| Reason | Detail |
|--------|--------|
| **Matches the math** | Defining operation is a **count** of equal bits → a scalar. |
| **Encodes unary structure** | Tally marks are equal units (no place-value roles inside the bit bag). |
| **Supports multi-level and majority** | Majority = who wins; \(s/S\) = soft tally / vote share. |
| **Low literature collision** | No established DL family named TallyNet (unlike committee machines, BitNet, PSO “swarm”). |
| **Avoids hardware misread** | “Popcount” in industry usually means **XNOR–popcount matmul**, not “weight = f(count of private bits).” |
| **Works in code** | `TallyLinear`, `tally_width=S`, `encoder=` are readable APIs. |
| **Separates architecture from optimizer** | Answers *how weights are represented*, not *how they are stepped*. |

---

## 4. Why not the other candidates

| Candidate | Verdict | Main problem |
|-----------|---------|--------------|
| **Bitswarm / BitSwarm** | Legacy | PSO collision; optimizer-colored |
| **Popcount BitNet** | Reject as brand | Impersonates BitNet; popcount = XNOR GEMM slang |
| **Popcount QAT** | Reject as brand | QAT is a training method |
| **Committee network** | Reject as brand | **Committee machine** = ensemble of models |
| **BallotNet** | Demoted | Informal; CUDA `__ballot__` |
| **Thermometer coding** | Related, not default brand | Often **ordered** prefix bits; we use **unordered** tally |

### Popcount — how to use it

| Do | Don’t |
|----|--------|
| “The tally is the popcount of the bit bag” | Brand “PopcountNet” / “Popcount BitNet” |
| Hardware notes: packed sum / `__popc` | Claim novelty is inventing XNOR–popcount |

### Closest relatives (for papers)

| Relative | Relationship |
|----------|----------------|
| Unary / multi-bit binary weights | Same broad family |
| ABC-Net–style multi-binary bases | Multi-bit binary; usually weighted bases |
| BitNet / BNNs | Low-bit linears; different default storage |
| Thermometer / unary codes | Digital-design cousins; note unordered tally |

---

## 5. Preferred terminology

| Prefer | Avoid (in new text) |
|--------|---------------------|
| TallyNet | Swarm net (as brand) |
| Tally-coded weight | Effective weight (unless mapping legacy) |
| Tally width \(S\) | Swarm size (unless mapping legacy) |
| Tally / tally count | Bare “popcount” as the only name |
| Tally encoder | Unnamed “decode” |
| Bit (on an entry) | Agent |
| Matrix entry | Connection (ambiguous) |

---

## 6. Public phrasing

**Elevator:**

> **TallyNet** uses **tally-coded weights**: each parameter is a small bag of equal bits whose **tally** is encoded into a multi-level scalar for the matmul—unary multi-bit storage without place-value digits.

**Slightly longer:**

> Standard linear layers store one real (or one quantized) number per matrix entry. TallyNet stores **\(S\) equal binary bits** per entry and defines the entry’s value as a function of their **tally** (count of \(+1\)). Capacity is the number of distinct counts (\(S+1\)), not binary place values. This is a **weight representation** choice; training rules are specified separately.

**Do not claim under the architecture name alone:**

- “We invented popcount hardware.”
- “This is a new BitNet.”
- “This is a committee machine / ensemble.”
- “This is a particle-swarm optimizer.”

---

## 7. Changelog

| Date | Change |
|------|--------|
| 2026-08-07 | Initial naming rationale; adopt **TallyNet** / tally-coded weights |
| 2026-08-07 | Seeded into standalone `tallynet` repository |
