# What this study may claim, and the nearest thing it may not

**This file is the source the write-up is drafted from.** Not a summary of the
results — a ledger of *sentences*, each paired with the sentence one paraphrase
away that the data does not support.

It exists because the failure it guards against is not a wrong number. Every
number here can be correct and the paper still be wrong, if a caveat recorded
in `limitations.md` is dropped when someone compresses 900 lines into a
paragraph. Caveats do not survive summarising; **the claim and its boundary
have to be one sentence, written once, and copied rather than rephrased.**

The rule for using this file: if a sentence in the write-up is not in the
"supported" column, it is not licensed by this study's data — regardless of
how obviously true it seems.

---

## GLA / linear attention

**Measured 2026-09-07. Verdict: DROP, failed gate 1.** `docs/gla_arm_decision.md`.

| | |
|---|---|
| **Supported** | *Substituting ungated GLA into softmax-trained Qwen2.5 weights at inference produces output that does not depend on the context at all: **1 distinct prediction across 100 distinct 2048-token contexts**, against 100/100 for the dense control on the same examples in the same process.* |
| **Not supported** | *Linear attention degrades on long-context retrieval.* |
| **Also not supported** | *Ungated GLA scores near zero on retrieval.* |

**Why the second is a different claim.** It is about an architecture; ours is
about a *substitution*. Qwen2.5's weights were trained under softmax
attention and its projections were never optimised for a recurrent state. A
natively linear-attention-trained model has different weights, and this study
measured none of them. Licensing it would need RWKV, Mamba, or a
GLA-trained checkpoint on the same tasks — a different experiment.

**Why the third is a different claim, and this is the one the measurement
added.** A *score* implies the task was attempted. Gate 1 failed: the output
is context-independent, so the model never engaged the task at all. Reporting
"near zero" would describe performance on a task that was never undertaken.
The pre-registered gate ordering — both 1 and 2 failing must report gate 1 —
is what preserved this distinction; it was written before the number existed
precisely because a small number invites the performance reading.

**Second boundary, inside the first.** The measured configuration is
**ungated** (`gate_source="ungated"`, g = 0, decay factor exactly 1, nothing
forgotten). Full retention did not help, which rules out forgetting as the
cause. A real GLA layer has a learned forget-gate projection and Qwen2.5 has
none to borrow, so the supported sentence is about ungated GLA and the
qualifier is not optional.

**Timing is unaffected.** A kernel's throughput does not depend on the values
in its gate, only shape and dtype. Stage 2's GLA latency and its bounded O(1)
state stand. *GLA is faster and its state is bounded* is a timing claim and
holds; *GLA is as accurate* is an accuracy claim and no measurement of it
exists. Stage 3 ships with two backends, dense and block-sparse.

---

## Block-sparse accuracy

**Three bands measured (2048/4096/8192), n=300 per cell, standard errors
0.0–2.1 points.** Report absolutes alongside deltas: the dense baseline is
not flat across length, so a delta alone hides which side moved.

| | |
|---|---|
| **Supported** | *With an oracle importance ranking, the sparsity a task tolerates is a property of the **task**, not of the kernel: `niah_single` is intact to 0.75 in every band (98.3 / 99.7 / 99.7 vs 100.0 dense), while `niah_multikey` has already lost ground at 0.5 (89.3 / 88.7 / 71.0 vs 97.0 / 95.0 / 86.0 dense) and falls to 41.7 at 0.75 by 8192.* |
| **Not supported** | *Block-sparse attention holds dense accuracy to 0.5 sparsity.* |

**Why the second is a different claim, three times over.** It generalises
across tasks, across lengths, and against a fixed reference. Across tasks:
`niah_single` survives 0.75 and `niah_multikey` is down 7.7 points at 0.5 —
one "holds to 0.5" sentence averages those into a number describing neither.
Across lengths: `niah_multikey` at 0.75 goes 73.0 → 72.3 → **41.7**, so
tolerance is length-dependent as well as task-dependent. And the reference
moves: **dense itself degrades**, 97.0 → 95.0 → 86.0 on `niah_multikey`, so
part of what looks like sparse degradation at 8192 is the baseline falling.

**The 0.9 arm at 2048 is a grid artifact, not a sparsity result.** It scored
0.0 / 63.3 / 26.3 there and 38.3 / 93.7 / 73.0 at 4096. With
`block_size=128`, 0.9 sparsity retains 1.6 blocks at 2048, 3.2 at 4096, 6.4
at 8192 — at 2048 there is almost no budget to be right with, whatever the
ranking. A test was stated before 8192 ran: *if the budget explanation holds,
0.9 keeps climbing.* **It plateaued** (38.3 → 35.3). So the threshold reading
survives — 2048@0.9 is below a floor — and the smooth "more blocks, more
accuracy" story does not.

**The oracle qualifier is load-bearing, and stronger than a ceiling.** See
the dedicated section below.

**Second qualifier:** sparsity is applied **during prefill only**; generation
runs dense over the cache (`decode_backend` on every row).

---

## The oracle can put sparse ABOVE dense, and that is contamination

The single most important thing measured this session, and it is bad news for
the headline rather than good.

| | |
|---|---|
| **Supported** | *`block_sparse` at 0.75 scores **above** the dense baseline on `vt` in all three bands: +10.8 (86.8 vs 76.0), +5.9 (90.6 vs 84.7), **+14.6** (85.1 vs 70.5), against standard errors of 1.0–1.4 on n=300.* |
| **Not supported** | *Sparse attention improves accuracy on variable tracking.* |

At 8192 that gap is roughly **nine standard errors**, same direction, three
independent bands. It is not noise, and it survived being withheld from this
file at one band and again at two.

**Discarding computation cannot improve a model.** What can is where the mask
came from. The ranking is derived from the full attention scores, so the mask
concentrates attention on blocks that dense attention identified as important
but does not preferentially attend to. On a chain-of-assignments task that is
a denoiser.

If that is the mechanism, **no deployable method reproduces it** — a cheap
estimator computed from partial information does not know which blocks
matter. The oracle here is not standing in for a deployable estimator; it is
supplying information the deployable estimator cannot have. The mechanism is
proposed; the effect and its size are measured. Separating them needs the
same grid under a genuinely cheap `score_source`, which does not exist yet.

**This changes the status of the oracle caveat.** It had been "accuracy here
is an upper bound" — a bound on how good sparse can look. It is stronger: the
oracle can make sparse look *better than dense*, which no deployable method
could achieve. That is a different kind of contamination and belongs in its
own statement, not as a footnote to the bound.

**It also narrows every Stage 4 matched budget.** The protocol certifies
non-inferiority to dense; where the oracle pushes sparse above dense, part of
what clears the bar is oracle-supplied. A matched budget licenses *"non-
inferior to dense when the mask is chosen with full knowledge of the
attention scores"*, never *"this budget is free"*. Carried in
`analysis/matched.py`'s docstring so it reaches the code that computes it.

## Cross-architecture timing

| | |
|---|---|
| **Supported** | *On the 11 cells measured on both sm_80 and sm_89, the backend ranking is stable except where noted.* |
| **Not supported** | *The backend ranking is architecture-independent.* |

Eleven cells, one of which carries the claim. See `limitations.md`, "The
cross-architecture claim rests on 11 cells".

---

## Absolute accuracy numbers

| | |
|---|---|
| **Supported** | *Backend A scores X against backend B's Y on identical inputs.* |
| **Not supported** | *Backend A scores X on RULER.* |

Tasks are RULER's *algorithm* re-implemented with a dependency-free filler
haystack (`haystack_mode` on every row), not RULER's benchmark distribution.
Between-backend comparisons on identical inputs are valid; comparison to
published RULER numbers is not. See `limitations.md`, "Task construction is
RULER's algorithm, not RULER's benchmark".

---

## How to add to this file

A row belongs here when the honest claim and the appealing claim differ by a
qualifier that a reader would not miss. If the qualifier is obvious from the
supported sentence alone, it belongs in `limitations.md` instead — this file
is for the ones that get lost.
