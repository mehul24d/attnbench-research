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

| | |
|---|---|
| **Supported** | *Substituting GLA into a softmax-trained model at inference time does not preserve retrieval.* |
| **Not supported** | *Linear attention degrades on long-context retrieval.* |

**Why the second is a different claim.** It is about an architecture; ours is
about a *substitution*. Qwen2.5's weights were trained under softmax
attention, and its `q/k/v` projections were never optimised for a recurrent
state. A model trained natively with linear attention has different weights,
and this study measured none of them. The result says what happens when a
mechanism is swapped underneath weights fitted to another one — which is a
real and useful thing to know, and not the same thing.

**What would license the broader claim:** a natively linear-attention-trained
model of comparable size (RWKV, Mamba, a GLA-trained checkpoint) evaluated on
the same tasks. That is a different experiment with a different budget, and it
is not in this study's scope.

**Second boundary, inside the first.** GLA here runs **ungated** —
`gate_source="ungated"`, `g = 0`, decay factor exactly 1. A real GLA layer has
a learned forget-gate projection, and Qwen2.5 has none to borrow. So even the
supported sentence is about *ungated* GLA, and the qualifier is not optional:

| | |
|---|---|
| **Supported** | *Ungated GLA, substituted into softmax-trained weights, ...* |
| **Not supported** | *GLA, substituted into softmax-trained weights, ...* |

This one is carried by the data itself rather than by prose: every linear row
records `gate_source`, and `AccuracyResult` refuses to be constructed without
it (`schema.py`). A reader holding only the parquet can ask which gate ran.
The earlier version of this arm — a synthetic gate with a 1.24-token memory
horizon — is why that column exists: see `silent_failure_patterns.md` #17.

**Timing is unaffected by all of the above.** A kernel's throughput does not
depend on the values in its gate, only on shape and dtype, so Stage 2's GLA
latency numbers stand whatever the accuracy arm's verdict. The claims split
cleanly: *GLA is faster and its state is bounded* is a timing claim and holds;
*GLA is as accurate* is an accuracy claim and does not.

---

## Block-sparse accuracy

| | |
|---|---|
| **Supported** | *At 2048 tokens, with an oracle importance ranking: single-needle retrieval is intact at 0.75 sparsity (98.3 vs 100.0 dense) and degraded but non-zero at 0.9 (63.3); multi-key retrieval is already down at 0.5 (89.3 vs 97.0) and is zero at 0.9.* |
| **Not supported** | *Block-sparse attention holds dense accuracy to 0.5 sparsity.* |

**Why the second is a different claim, twice over.** It generalises across
tasks and across lengths, and the banked data supports neither generalisation.
Across tasks: the sparsity a task tolerates is a property of the *task*, not
of the kernel — `niah_single` survives 0.75 and `niah_multikey` has already
lost 7.7 points at 0.5. A single "holds to 0.5" sentence averages those into a
number describing neither. Across lengths: **only the 2048 band is measured.**
Bands 4096 and 8192 are unrun, and the 16384/32768 bands are unbooked, so
every sentence above carries "at 2048" until they are not.

A third observation is deliberately left out of the supported column: `vt`
scores *higher* under sparsity than dense (76.0 → 78.1 → 86.8). One band, one
seed, no error bars, and no mechanism proposed — it is an observation to
check, not a finding, and it is exactly the shape of thing a summary would
promote into "sparsity improves variable tracking".

The oracle qualifier is the whole claim. The ranking comes from a dense
softmax pass that costs more than the attention it is used to skip — an upper
bound on what any deployed cheap estimator could achieve, with the estimator's
own cost excluded from every latency number. Carried on the row as
`score_source`. See `limitations.md`, "Importance scores are an upper bound".

Second qualifier: sparsity is applied **during prefill only**; generation runs
dense over the cache (`decode_backend` on every row).

---

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
