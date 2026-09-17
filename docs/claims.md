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

## End-to-end speedup, which is the point of the whole study

**Measured 2026-09-07 from Stage 3's `generate()` latency, n=900 per arm.**
Stage 5's dedicated timing does not exist yet; this instrument is weaker in
stated ways (no warmup control, scoring pass excluded, no prefill/decode
split) and the run order biases *against* the finding — the dense arm runs
first in every band, so warmup penalises the baseline.

> **These are the measured numbers, and they were measured under two
> confounds Stage 5 later found — an unmatched decode kernel between arms,
> and unequal generation length on `vt`. Read
> ["The dominance result was measured under two confounds"](#the-dominance-result-was-measured-under-two-confounds)
> below before quoting any figure in this section.** The headline **1.06×**
> survives normalization unchanged; the operating point behind it and the
> dominance counts do not.

| | |
|---|---|
| **Supported, as measured** | *At batch 1 with prefill-only sparsity and dense decode, the best end-to-end speedup any accuracy-matched block-sparse operating point achieves is **1.06×** — `vt`, 0.5 sparsity, 4096. On `niah_single` every matched point is **slower** than dense (0.8–0.9×) and dominated by it on both axes.* |
| **Superseded by the correction** | *…and that best point is `vt`/0.5/4096.* Normalized, the best point is `niah_single`/0.9/8192, at the same 1.057×. *…and every `niah_single` point is dominated.* Normalized, 9 of 15 are. |
| **Not supported** | *Block-sparse attention is slower than dense.* |
| **Also not supported** | *Block-sparse attention gives a 1.24× speedup.* |

**Why the second is a different claim.** It drops the regime, and the regime
is doing all the work. Sparsity is applied during **prefill only**; decode
runs dense over the cache in both arms. At batch 1 generating ~30 tokens,
decode is the larger share of end-to-end cost and sparsity does not touch
it — so this measures a setting in which prefill sparsity attacks the smaller
part of the bill *by construction*. That is the regime a single-stream
interactive deployment runs in, which is why it is worth measuring, but a
batched or long-generation regime is a different measurement this study has
not made.

**Why the third is a different claim, and this is the study's thesis.** 1.24×
is a **kernel** number, at 90% sparsity, from Stage 2. End-to-end, at the
operating points that actually preserve accuracy, it becomes **≤1.06× and
usually <1.0×**. A kernel speedup is not an end-to-end speedup, and the
distance between them is the gap this study exists to measure. Reporting the
kernel figure as a system result is the same regime-vs-unit error that has
already appeared three times in this project's own analysis.

**16 of 31 accuracy-matched sparse operating points are dominated by the
dense baseline** on both axes. All 15 `niah_single` points are dominated.
The survivors are all `vt` — and `vt` is the `oracle_sensitive` task, so most
of what keeps them on the frontier is the oracle rather than the sparsity.
Only **two distinct operating points** are genuinely faster than dense.

This is recorded as data, not prose: `dominated_by_dense` on every Stage 6
row, and the dense baseline carried as a point (`is_dense_reference`) so a
reader of `pareto.parquet` cannot see a tidy frontier without seeing that the
baseline beats most of it.

### The dominance result was measured under two confounds

**Stage 5 decomposed the end-to-end totals and found confounds inside them.**
This is the case for Stage 5 in one sentence: *an end-to-end number cannot
surface a confound that lives inside it.*

**Confound 1 — the decode kernel.** `block_sparse` has no decode path, so
`decode_backend_for` falls back to `DENSE_DECODE_BACKEND = "sdpa_math"`. The
dense arm is `sdpa_flash` and decodes through **itself**. Sparsity is
prefill-only, so decode is dense in both arms — through *different kernels*,
on the phase that dominates the bill at batch 1.

| band | dense decode (`sdpa_flash`) | sparse decode (`sdpa_math`) | penalty | share of the sparse arm's total |
|---|---|---|---|---|
| 2048 | 34.80 ms/token | 42.90–43.16 | **+23.3 to +24.0%** | 17.4% |
| 4096 | 34.60 ms/token | 42.04–42.57 | **+21.5 to +23.0%** | 14.9% |
| 8192 | 34.16 ms/token | 54.94–55.92 | **+60.8 to +63.7%** | **28.8%** |

Recipe for the last column, stated because it moved: the largest value over
that band's sparse operating points of `(n_own − 1) × penalty / measured_ms`.
It read `18.2 / 15.5 / 29.2` until 2026-09-12, computed with `n ×` rather
than `(n − 1) ×`. Recomputed from `results/stage5/phases.parquet` and
`results/stage6/decode_corrected.parquet`.

**Confound 2 — unequal generation length, which was already in the measured
result.** On `vt` the arms do not generate the same number of tokens: dense
35.2–38.8, sparse 27.7–34.6. Their mean end-to-end latencies were never
comparable. An arm that stops earlier finishes sooner for reasons that have
nothing to do with attention speed. `niah_single` is clean — every arm
generates exactly 14.0.

Correcting **only** the decode kernel gives 6/31 dominated and a best speedup
of 1.26×. **Both of those are artifacts of confound 2** and neither is
reported. (They read 0/31 and 1.28× until 2026-09-12; recomputed under the
`(n − 1)` identity from era-matched phases. The point is unchanged and the
direction is the same — correcting one confound alone overstates.) The unconfounded quantity holds both arms to the same decode kernel
*and* the same generation length.

#### DERIVED, NOT MEASURED

`normalized` is a model — `prefill(band, sparsity) + (n−1) × decode_dense(band)` —
built from Stage 5 phases taken on **random token ids at exactly the band
length**, n=10 reps, not on the RULER prompts (4000–8196 tokens inside the
8192 band) and not at n=900. It reconstructs what the arms *would* have cost
under a matched decode kernel. `results/stage6/decode_corrected.parquet`
carries `measured_ms` beside `decode_corrected_ms` and `normalized_ms` on
every row, so a corrected number cannot travel without what it was corrected
from.

**Read `normalized_ms`, not `decode_corrected_ms`.** The middle column
corrects the decode kernel and leaves the generation-length confound in, so
it is inflated on `vt` — it is an intermediate, kept only so the two
corrections can be seen apart. It is also only meaningful when the phases it
was built from are from the same decode era as the rows: fed post-2026-09-08
phases, where both arms already decode through `sdpa_flash`, the penalty is
0.29–0.75 ms/token instead of 8–22 and the column becomes a no-op wearing the
name of a correction. That is what the file contained from 2026-09-08 to
2026-09-12. `decode_confound.correct` now refuses that pairing outright and
`decode_penalty_ms` is on every row, so the size of what was removed is
visible without recomputing it.

#### What survives the correction, and what does not

| claim | status |
|---|---|
| *The best end-to-end speedup at any accuracy-matched point is ≤1.06×* | **SURVIVES.** Measured best 1.057× (`vt`/4096/0.5); normalized best **1.059×** (`niah_single`/8192/0.9). The normalized figure read 1.057× until 2026-09-12, when it was recomputed under the `(n − 1)` identity from era-matched phases; ≤1.06× is unaffected. |
| *…and that point is `vt`, 0.5 sparsity, 4096* | **DOES NOT SURVIVE.** Normalized, the best point is **`niah_single`, 0.9 sparsity, 8192**. The number is unchanged and the operating point behind it is different. |
| *16 of 31 matched sparse points are dominated by dense* | **DOES NOT SURVIVE.** Normalized: **12 of 31**. |
| *All 15 `niah_single` points are dominated* | **DOES NOT SURVIVE.** Normalized: **9 of 15**. The six that leave are all at 8192. |
| *The survivors are all `vt`* | **DOES NOT SURVIVE.** Six `niah_single` points at 8192 survive normalization. |
| *A 1.24× kernel speedup becomes ≤1.06× end-to-end* | **SURVIVES**, and is strengthened — the gap is now measured against a matched-kernel baseline rather than one carrying a handicap. |
| *`vt`'s matched budgets are `oracle_sensitive`* | **UNTOUCHED.** Nothing here bears on the oracle. |

**Seven points leave the dominated set**, every one of them at 8192 —
sparsity's prefill saving only becomes visible at the longest band measured:

| task | band | sparsity | measured | normalized | dense (normalized) |
|---|---|---|---|---|---|
| niah_single | 8192 | 0.50 | 1415.1 | 1120.8 | 1138.4 |
| niah_single | 8192 | 0.75 | 1360.8 | 1096.1 | 1138.4 |
| niah_single | 8192 | 0.90 | 1342.6 | 1075.5 | 1138.4 |
| vt | 8192 | 0.90 | 2155.7 | 1921.3 | 1984.2 |

*The `normalized` and `dense` columns above were `1155.0 / 1130.3 / 1109.6`
and `1173.1 / 2020.3` until 2026-09-12. Those were computed with
`prefill + n × decode_step`; the identity is `(n − 1)`, because the prefill
forward emits the first token's logits. The dominance verdicts, the count of
seven, and which points move are unchanged — the correction shifts every
column by one decode step (~34 ms) in the same direction.*

**Three points enter it** — `vt`, 2048, 0.5 sparsity, at all three epsilons.
They were on the frontier only because that arm generated 27.8 tokens against
dense's 35.2. Recorded because a correction that could only ever free points
is a correction nobody checked.

| | |
|---|---|
| **Supported** | *Under a matched decode kernel and matched generation length, **12 of 31** accuracy-matched sparse operating points remain dominated by dense, and the best speedup is **1.059×**, at `niah_single`/8192/0.9. Sparsity's end-to-end benefit is real, small, and confined to the longest band measured.* |
| **Not supported** | *Block-sparse beats dense once you correct for the decode kernel.* Correcting only that gives 1.26×, which is confound 2 talking. |
| **Not supported** | *The measured Stage 6 numbers are wrong.* They are correct measurements of a system in which one arm decodes through a slower kernel. That is a real property of this harness, and the qualifier is the regime, not an error bar. |

`grid_configs.py` names confound 1 in its own docstring — *"a row labelled
`sdpa_math` and one labelled `sdpa_flash` are the same class and different
kernels, which is exactly the confound `SDPABackend` exists to remove"* — and
the decode fallback reintroduces it **between arms**. The choice was recorded
on every row the whole time. Nothing compared the two arms' decode backends
until the phases were measured apart.

#### The unconfounded end-to-end comparison is not obtainable on two of three tasks

`DENSE_DECODE_BACKEND` was changed to `sdpa_flash` on 2026-09-08, so future
runs are unconfounded on the kernel. **Re-running Stage 3 to get an
unconfounded end-to-end number does not work**, and the reason is a
methodological finding rather than a budget one.

The effect to resolve is the prefill gap: **18–63 ms**. The re-run would have
to resolve it against the per-example spread of the paired dense-minus-sparse
latency difference:

| task | sd of the paired difference | n for a 5 ms standard error |
|---|---|---|
| `niah_single` | 6.6–17.8 ms | **2–13** |
| `vt` | 320–750 ms | 4,100–22,500 |
| `niah_multikey` | 641–1008 ms | 16,400–**40,700** |

| | |
|---|---|
| **Supported** | *On tasks whose output length varies per example, an unconfounded end-to-end comparison of a prefill-sized effect is **not obtainable at any affordable sample size**: generation-length variance exceeds the effect by an order of magnitude. For `niah_multikey` it would take ~40,000 examples per cell.* |
| **Not supported** | *We did not measure it because it was too expensive.* At n=300 it is not underpowered by a factor of two; it is the wrong instrument. |

`niah_single` is measurable at **n=13** for the same reason it was clean of
the generation-length confound: fixed 14-token output. So the instrument
works exactly where output length is fixed, and fails everywhere else.

**Why this matters beyond this study.** The distance between a kernel
speedup and an end-to-end speedup is what this study exists to measure. This
result says something about *why that distance is hard to measure honestly*:
end-to-end wall-clock, the natural instrument for the system-level number,
has variance from generation length that swamps prefill-sized effects on any
variable-length workload. A kernel microbenchmark has no such variance, which
is part of why it is the number that gets reported. Closing the gap honestly
requires per-phase decomposition — not a bigger end-to-end sample.

#### MEASURED 2026-09-08: the correction was right, and the penalty was the kernel

Both measurements above were run. `DENSE_DECODE_BACKEND` is now `sdpa_flash`.

**1. The decode penalty is gone, not reduced.** Threshold pre-registered at
5% residual before the data existed:

| band | dense | sparse was (`sdpa_math`) | sparse now (`sdpa_flash`) | penalty was | penalty now |
|---|---|---|---|---|---|
| 2048 | 34.96 | 42.90–43.16 | 34.34–34.58 | +23.3 to +24.0% | **−1.8 to −1.1%** |
| 4096 | 34.58 | 42.04–42.57 | 34.36–34.72 | +21.5 to +23.0% | **−0.6 to +0.4%** |
| 8192 | 31.25 | 54.94–55.92 | 31.41–32.60 | +60.8 to +63.7% | **+0.5 to +4.3%** |

Worst residual 4.3%, inside the pre-registered 5%. The premise the analytical
correction rested on — that the whole gap was the kernel choice — **holds**.
Precision caveat: the decode estimator is a two-point slope whose noise at
8192 is ~3% (silent_failure_patterns #26), so "removed" means
"indistinguishable from dense to within 3–4%", not "equal".

**2. The analytical correction predicted the measurement to within 2.4%.**
`niah_single`, n=100 paired, matched decode kernel, 14 tokens generated by
every arm:

| band | sparsity | measured speedup | predicted | error | score vs dense |
|---|---|---|---|---|---|
| 8192 | 0.90 | **1.062×** | 1.057× | **+0.5%** | 99.0 vs 100.0 |
| 8192 | 0.75 | **1.044×** | 1.037× | +0.7% | **100.0 vs 100.0** |
| 8192 | 0.50 | 0.998× | 1.015× | −1.7% | 100.0 vs 100.0 |
| 4096 | 0.75 | 0.997× | 1.000× | −0.3% | 100.0 vs 100.0 |
| 4096 | 0.50 | 0.968× | 0.992× | −2.4% | 100.0 vs 100.0 |
| 2048 | 0.75 | 0.993× | 0.989× | +0.4% | 99.0 vs 100.0 |
| 2048 | 0.50 | 0.984× | 0.986× | −0.2% | 100.0 vs 100.0 |

The predicted best operating point — `niah_single`/0.9/8192 — **is** the
measured best. Paired bootstrap, n=100: **+66.8 ms saved, 95% CI [+65.5,
+68.0]**, significant. The CI is tight for the same reason this task is
measurable at all: every arm generates exactly 14 tokens.

**Control:** the dense arm, which did not change, reproduced across sessions
and hosts to within **0.9%** at every band (+8.9 / −2.0 / +5.2 ms on 593 /
737 / 1146 ms).

| | |
|---|---|
| **Supported, MEASURED** | *With a matched decode kernel, the best end-to-end speedup at an accuracy-preserving operating point is **1.062×** (`niah_single`, 0.9 sparsity, 8192), and **1.044× at zero accuracy cost** (0.75 sparsity, same band, 100.0 vs 100.0). Below 8192 no sparsity is faster than dense.* |
| **Not supported** | *Block-sparse is faster than dense.* At 2048 and 4096 every point is at or below 1.0×, and at 8192 sparsity 0.5 is 0.998×. The result is confined to high sparsity at the longest band measured. |
| **Not supported** | *1.06× was always the right number.* It was the right number by coincidence: the measured pre-correction headline was 1.057× at `vt`/0.5/4096, and the unconfounded 1.062× is at `niah_single`/0.9/8192. Same magnitude, different operating point, different task, and the survivors are no longer confined to the `oracle_sensitive` one. |

#### MEASURED 2026-09-08 at 16384: the length dependence continues and steepens

`niah_single`, n=100, matched `sdpa_flash` decode, 14 tokens generated by
every arm. All three sparsities are faster than dense and **all three score
100.0**:

| sparsity | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|
| 0.50 | 0.984× / 100.0 | 0.968× / 100.0 | 0.998× / 100.0 | **1.071× / 100.0** |
| 0.75 | 0.993× / 99.0 | 0.997× / 100.0 | 1.044× / 100.0 | **1.140× / 100.0** |
| 0.90 | 0.992× / 62.0 | 1.005× / 93.0 | 1.062× / 99.0 | **1.186× / 100.0** |

*(speedup vs dense / RULER score; dense scores 100.0 at every band)*

**Both axes move together, monotonically, along the 0.9 row:** speedup
0.992 → 1.005 → 1.062 → **1.186**, accuracy 62.0 → 93.0 → 99.0 → **100.0**.
Two measurements, neither designed to produce a length dependence, agreeing
on one. Paired bootstrap at 16384, n=100: 0.9 sparsity saves **+329 ms, 95%
CI [+322.0, +337.3]**.

| | |
|---|---|
| **Supported** | *Block-sparse attention's end-to-end benefit rises with context length and is **1.186× at 16384 at zero accuracy cost** (100.0 vs 100.0). Below 8192 it is at or below 1.0×. The mechanism that makes sparsity affordable at long context is the same one that makes it accurate there.* |
| **Not supported** | *Block-sparse gives a 1.19× speedup.* One task, one band, batch 1, prefill-only sparsity, oracle-derived masks. Every one of those qualifiers is load-bearing. |
| **Not supported** | *The trend will continue at 32768.* Three rising points do not establish an asymptote. 32768 is unmeasured and is the band where memory, not arithmetic, may dominate. |

**The estimator costs ~36× what the sparsity saves, and that belongs here,
not in limitations.** These masks are chosen with full knowledge of the
attention scores. Producing that ranking is a dense attention pass measured
at **11.4 s per example at 16384**, against the **330 ms** the resulting
sparsity saves:

| band | scoring pass | best sparsity saves | ratio |
|---|---|---|---|
| 2048 | 288 ms | — (nothing is faster) | undefined |
| 4096 | 929 ms | — | undefined |
| 8192 | 3.25 s | 67 ms | **49×** |
| 16384 | 11.8 s | 328 ms | **36×** |

| | |
|---|---|
| **Supported** | *The 1.186× is what block-sparse achieves **given** an oracle ranking whose own cost exceeds the saving by ~36×. No deployable estimator in this study reproduces that ranking, and none is measured.* |
| **Not supported** | *Block-sparse attention delivers 1.186× at 16384.* Not as a system. It delivers that **given a mask nobody can afford to compute**, and the ratio does not obviously improve with length — it is 49× at 8192 and 36× at 16384. |

The ratio narrowing from 49× to 36× is the only sign that scale might help,
and two points is not a trend. **This sits beside the headline because a
reader who takes 1.186× and leaves the 36× behind has the study backwards** —
the speedup is the upper bound a real estimator would have to approach from
below, while also being cheap, which nothing here demonstrates is possible.

#### MEASURED 2026-09-08 at 32768: speedup continues, accuracy turns over, oracle cost does not close

The full grid. `niah_single`, matched `sdpa_flash` decode, 14 tokens from
every arm; n=100 through 16384, n=50 at 32768 (sized from the measured paired
sd of 32–44 ms, which needs n=19 for a 10 ms standard error).

| sparsity | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|
| 0.50 | 0.984× / 100.0 | 0.968× / 100.0 | 0.998× / 100.0 | 1.071× / 100.0 | **1.098× / 100.0** |
| 0.75 | 0.993× / 99.0 | 0.997× / 100.0 | 1.044× / 100.0 | 1.140× / 100.0 | **1.321× / 100.0** |
| 0.90 | 0.992× / 62.0 | 1.005× / 93.0 | 1.062× / 99.0 | 1.186× / 100.0 | **1.404× / 98.0** |

Three separate things happen at 32768, and they do not all point the same way.

**1. The speedup trend continues and steepens.** At 0.75 sparsity:
0.993 → 0.997 → 1.044 → 1.140 → **1.321×**, at 100.0 accuracy in every band
from 4096 up. Paired bootstrap at 32768, n=50: **+1091 ms saved, 95% CI
[+1075, +1107]**. The gain is prefill: dense prefill 4018 ms against 2723 ms
at 0.9 sparsity.

**2. Accuracy at 0.9 stops improving — but does NOT measurably decline.**
62.0 → 93.0 → 99.0 → 100.0 → 98.0. The final point is **one flipped example
out of 50**: paired bootstrap on the accuracy difference gives −2.00 with 95%
CI **[−6.00, +0.00]**, which includes zero.

| | |
|---|---|
| **Supported** | *Accuracy at 0.9 sparsity reaches ceiling (100.0) by 16384 and is at ceiling or indistinguishable from it at 32768.* |
| **NOT supported** | *Accuracy turns over / declines at 32768.* One example in 50, CI touching zero. An earlier draft of this section stated the turnover as fact and built a two-axes argument on it; that was a single retrieval failure read as a trend. |
| **NOT supported either** | *Speed and accuracy are one mechanism.* Accuracy saturates at 100.0 from 16384, so beyond that band the data **cannot distinguish** "accuracy would keep rising if it could" from "accuracy has stopped". The convergence claim is untestable past 16384, not refuted. |

What survives is narrower and holds: the convergence of the two trends is
established over **2048–16384**, where both moved and neither was at ceiling.
The 0.75 row holds 100.0 throughout and is the operating point to quote.

**3. The oracle ratio flattens and does not close.** Scoring cost against the
best latency it buys: **249× → 49× → 36× → 35×** at 4096 / 8192 / 16384 /
32768. Measured scoring at 32768 is **44.6 s per example** against 1286 ms
saved. The ratio improved by 5× from 4096 to 8192 and has moved 4% since
16384. Whatever asymptote it has is around 35×, not 1×.

| | |
|---|---|
| **Supported** | *Block-sparse prefill attention is faster than dense on an **A100 at 8192 and above** — **1.201× at 16384/0.75, 1.282× at 16384/0.9** — **but only with a vectorised mask builder, which the reference implementation does not have.** With the reference builder the same configurations are 0.633× and 0.956×. The difference is one function: ~91% of the reference builder's cost is Python interpreter overhead in an unvectorised per-query-block loop.* See "The A100 reversal is CPU mask construction". |
| **Supported** | *On an **NVIDIA L4 (sm_89)**, block-sparse attention reaches **1.321× end-to-end at 32768 with no accuracy loss** (100.0 vs 100.0, 0.75 sparsity), and the benefit grows monotonically with context length across five bands.* **The card is not a detail of this sentence.** On an A100 the same configuration is 0.475× with the reference builder — and the cause is the builder, not the card. |
| **Not supported** | *Block-sparse attention is slower than dense on an A100.* This was the published claim on 2026-09-16 and it is wrong as a statement about the method. It is true only of the reference mask builder, and it inverts when that builder is replaced — measured end-to-end, same process, same scores, only the builder changed, with bitwise-identical model outputs. |
| **Supported** | *The oracle scoring pass costs ~35× the latency it saves, and that ratio stops improving after 8192. The speedup is an upper bound no measured estimator approaches.* |
| **Supported** | *The oracle requirement is **not** a small-model artifact. Against MInference's mean-pool estimator on `niah_multikey` at 16384, the oracle's advantage **widens** with scale — +32/+40/+19 points at 1.5B against **+39/+67/+76** at 7B — because better representations help the dense-softmax oracle far more than the estimator. At 0.9 sparsity the oracle retains 95% of dense at 7B while the estimator retains 16%.* |
| **Not supported** | *Scale will close the oracle-versus-estimator gap.* The opposite, on the one axis point available, and on the harder of the two discriminating tasks. `vt`'s gap does close at 7B (+8.6 → −0.6), so the two tasks differ in sign and a task-averaged summary hides the one that matters. |
| **Not supported** | *Block-sparse attention delivers 1.321× at 32768.* Not as a system. It delivers that **given a mask nobody can afford to compute**, whose cost is 44.6 s per example against the 1286 ms saved — 35×, a ratio that has moved 4% since 16384. |
| **Not supported** | *Higher sparsity is always better at long context.* 0.9 buys 1.404× against 0.75's 1.321×, for an accuracy difference of one example in 50 that the CI cannot separate from zero. At that margin 0.75 is the defensible pick, not because 0.9 is worse but because nothing here shows it is not. |
| **Not supported** | *Accuracy and speed both improve with length, across the whole grid.* Both moved together over 2048–16384. From 16384 accuracy is at ceiling, so 32768 cannot test the claim either way. |

**The reconciliation identity, corrected the same day, closes at 32768 to
+0.0% / +0.0% / +0.1% / +0.3%** across the four arms — against +1.7% to +9.5%
all-positive under the old `n * step` form. That is the `(n − 1)` fix
confirmed by a measurement independent of the intercept check that found it.

### THE STUDY'S CONCLUSION

Everything above resolves into one sentence, and both halves are load-bearing:

> **On an NVIDIA L4, block-sparse attention with oracle-derived masks achieves up to 1.32×
> end-to-end at no accuracy cost at 32K context, and computing the oracle
> costs roughly 35× the latency it saves.**

The second clause is not a caveat on the first. It is the finding. A measured
advantage purchased with a mask nobody can afford to compute is not an
advantage any deployed system has.

**The break-even bar this sets.** To make the operating point profitable, a
deployable importance estimator would have to produce a good-enough ranking
for **under ~3% of the scoring pass's cost** (1/35), while preserving enough
of the oracle's ordering to keep accuracy at ceiling. This study measures
neither half of that: no cheap estimator was implemented, and none is
evaluated. `score_source` on every row is `dense_softmax_fp32` precisely so
that the day a cheap estimator exists it enters as a new value rather than a
silent change of meaning.

**The ratio's own trend is the discouraging part.** 249× → 49× → 36× → 35× at
4096 / 8192 / 16384 / 32768. It improved fivefold from 4096 to 8192 and 4%
since 16384. Longer context does not rescue it: scoring and the saving grow
at similar rates, so the ratio flattens near 35 rather than heading toward 1.
The gap is structural, not a small-scale artifact.

**Why this is the useful form of the result.** The literature's kernel
numbers (1.24× at 90% sparsity here, Stage 2) and this study's end-to-end
numbers differ by regime, and that gap is what the study set out to measure.
The measurement now has both ends: **a kernel speedup of 1.24×, an
oracle-masked end-to-end speedup of up to 1.32× at 32K, and an estimator cost
of 35× the saving standing between that and any deployment.** Reporting the
first without the third is the error this study exists to document.

#### What the study's central finding becomes

The finding was: *sparse is dominated by dense at matched accuracy; best
end-to-end speedup 1.06×.* With the kernel confound removed, it is
**length-dependent rather than flat**:

| band | does dense dominate? | best sparse speedup |
|---|---|---|
| 2048 | **yes** — every sparsity at or below 1.0× | 0.993× |
| 4096 | **yes** — every sparsity at or below 1.0× | 0.997× |
| 8192 | **no** | **1.062×** (0.9), **1.044× at no accuracy cost** (0.75) |

| | |
|---|---|
| **Supported** | *At batch 1 with prefill-only sparsity and a decode kernel matched across arms, dense dominates block-sparse at 2048 and 4096 and **does not** at 8192, where 0.75 sparsity is **1.044× faster at identical accuracy** (100.0 vs 100.0) and 0.9 sparsity is 1.062× for one point of accuracy. The benefit appears only at the longest band measured.* |
| **Superseded** | *Sparse is dominated by dense at matched accuracy.* True at 2048 and 4096; false at 8192. Stating it flatly reports the short-context result as the general one. |
| **Still not supported** | *Sparsity pays off end-to-end.* One band, high sparsity, one task, 1.04–1.06×. That is a real effect and a small one. |

**This aligns with the accuracy result rather than sitting beside it.** Stage
3 found sparsity tolerance *rising* with context length (block-sparse at 0.9
scored 26.3 at 2048, 73.0 at 4096, 69.7 at 8192 on the three-task mean; on
`niah_single` alone, 62.0 / 93.0 / 99.0 at n=100). The latency result now has
the same shape and the same direction: **longer context is where sparsity
both keeps its accuracy and starts to pay.** Two independent measurements
agreeing on a length dependence is a stronger claim than either alone, and
neither was designed to produce it.

#### The analytical method was validated where validation was possible

The correction for `vt` and `niah_multikey` cannot be checked directly — that
is the whole point of the section above. But the *method* was checked, on the
one task where checking is possible, and it held:

- Predicted vs measured speedup on all seven `niah_single` points: **errors
  of 0.2–2.4%**.
- On the operating point that matters most, **0.5%** (predicted 1.057×,
  measured 1.062×).
- The predicted best operating point **is** the measured best.

So the `vt` and `niah_multikey` corrected figures are not unverifiable
models. They are outputs of a procedure that was verified against measurement
wherever verification was available, and they should be read with that weight
— stronger than "modelled", weaker than "measured", and explicitly labelled
`normalized_ms` in the data either way.

**`vt` and `niah_multikey` stay analytical, permanently** — the sample size
their generation-length variance demands is not obtainable, and that is a
statement about the instrument, not the budget.

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

## The oracle requirement DOES survive a change of model scale — it gets worse

Measured 2026-09-17, the companion to the section below and the opposite
result. Same model (Qwen2.5-7B-Instruct), same band (16384), same tasks, n=100,
`clocks_locked=True`, `git_dirty=False` on both sides; only the scorer differs
(`dense_softmax_fp32` vs `minference_meanpool`, MInference 1.0 Algorithm 3).
The dense arm builds no mask and consults no scores, so the two runs compute
it identically — and did, to **0.0 points** on both tasks. Every gap below is
therefore signal, not run-to-run noise.

**Oracle minus cheap, `niah_multikey`:**

| sparsity | 1.5B | **7B** |
|---|---|---|
| 0.50 | +32.0 | **+39.0** |
| 0.75 | +40.0 | **+67.0** |
| 0.90 | +19.0 *(cheap at floor, 1.0)* | **+76.0** |

**The gap widens at every sparsity.** The 1.5B figure at 0.9 is
floor-compressed — the cheap arm scored 1.0 and could not go lower — so the
true 1.5B gap there is larger than +19.0 and the comparison at that row
understates rather than overstates the widening.

**This is not the cheap estimator failing to improve.** It improves
substantially with scale in absolute terms: 34.0 → 57.0, 19.0 → 27.0, 1.0 →
15.0. The oracle simply improves far more. As a share of each model's own
dense score:

| sparsity | 1.5B oracle | 1.5B cheap | 7B oracle | 7B cheap |
|---|---|---|---|---|
| 0.50 | 100% | 52% | 100% | 59% |
| 0.75 | 89% | 29% | 98% | 28% |
| 0.90 | 30% | 2% | **95%** | **16%** |

At 0.9 the oracle goes from retaining 30% of dense to retaining **95%**, while
the estimator goes from 2% to 16%. **Better representations are something the
dense-softmax oracle can exploit and the mean-pool estimator largely cannot**,
so scale widens the distance between what is achievable and what is
affordable.

**`vt` goes the other way, and the two tasks together are the mechanism.**

| sparsity | 1.5B gap | 7B gap |
|---|---|---|
| 0.50 | +3.2 | +1.6 |
| 0.75 | +5.4 | +0.4 |
| 0.90 | +8.6 | **−0.6** |

On `vt` the gap closes to nothing at 7B and inverts trivially at 0.9. So the
estimator is adequate for an easy retrieval pattern and inadequate for a hard
one, and scale sharpens that division rather than softening it — consistent
with the capacity-versus-difficulty account in Sparse Frontier (arXiv:2504.17768v2)
Appendix D.4. **A task-averaged summary would report the gap shrinking and
hide the task where it nearly quadrupled.**

**What this does to the study's deployability caveat: it makes it permanent
rather than provisional.** The natural hope after the sparsity-collapse result
— that the oracle requirement is a small-model artifact that scale would
dissolve — is refuted on the one axis point available. The honest statement is
that *the oracle requirement is not a small-model artifact, and on the
evidence here it is worse at scale, on hard retrieval.*

**Scope, in the same terms this document applies everywhere else.** Two model
sizes, one family, one band, one estimator. The estimator is MInference's
mean-pool, not every cheap estimator; a different one could behave differently,
and this measures the class only through one member. What is established is
that *this* gap does not close between 1.5B and 7B.

---

## The accuracy collapse at high sparsity does not survive a change of model scale

Measured 2026-09-17. Oracle scoring on both sides (`score_source =
dense_softmax_fp32`), `seq_len=16384`, n=100 per (task, cell), clocks locked
on the 7B rows, `git_dirty=False` on both. Each model's sparse cells are
re-based on **its own** dense control, so "the bigger model is better at the
task" is separated from "sparsity costs less at scale".

| task | sparsity | Qwen2.5-1.5B | Qwen2.5-7B |
|---|---|---|---|
| `niah_multikey` | dense | 66.0 | 96.0 |
| | 0.50 | +0.0 | +0.0 |
| | 0.75 | −7.0 | −2.0 |
| | **0.90** | **−46.0** | **−5.0** |
| `vt` | dense | 78.8 | 85.8 |
| | 0.50 | +9.4 | +0.6 |
| | 0.75 | +12.8 | +3.2 |
| | 0.90 | +12.0 | +6.0 |

**At 90% sparsity, multi-key retrieval costs 46 points at 1.5B and 5 points at
7B.** The 7B model answers 91 of 96 correctly while attending to a tenth of
its blocks. The collapse that looks like a property of block-sparse attention
at 1.5B is substantially a property of the 1.5B model.

This is the accuracy analogue of the card finding, and it carries the same
caveat, for the same reason: **a result is scoped by the axes it was varied
across.** One model at each of two scales is two points, not a scaling law,
and the two models are the same family — Qwen2.5 — so architecture is held
fixed along with everything else that travels with it. What can be said is
that the 1.5B collapse is not reproduced at 7B, which is enough to stop the
collapse being reported as a general property.

**`vt` moves the other way and is worth stating separately.** Sparsity *helps*
verbatim-retrieval at both scales — +12.0 at 1.5B and +6.0 at 7B at 0.9 — so
the two tasks do not merely differ in magnitude, they differ in sign. A
summary that averages across tasks would report a small net effect and hide
both.

**What this does not establish.** Nothing here measures 32768, where the
1.5B's own headline speedup was largest, and nothing here re-measures the
cheap estimator at 7B — at 1.5B the oracle beat the cheap arm by 32–40 points
on `niah_multikey`, and whether that gap is also scale-dependent is untested.

---

## The speedup does not survive a change of card

Measured 2026-09-16 on an A100-SXM4-80GB (sm_80) under DWS Flex Start, clocks
locked, `git_dirty=False`, against the L4 (sm_89) numbers above. Both cards
were pinned by the same policy to **85% of their maximum SM clock** (L4
2040→1740 MHz, A100 1410→1200 MHz), so this is a like-for-like comparison at
the same fraction of each card's ceiling.

**Prefill speedup vs dense, on the same card (>1 means sparse wins):**

| band | card | 0.50 | 0.75 | 0.90 |
|---|---|---|---|---|
| 16384 | L4 | 1.108 | 1.194 | 1.258 |
| 16384 | **A100** | **0.394** | **0.615** | **0.951** |
| 32768 | L4 | 1.014 | **1.373** | 1.475 |
| 32768 | **A100** | **0.281** | **0.475** | **0.817** |

**CORRECTED 2026-09-16, same day, by a kernel-level sweep.** The first version
of this section said "the block-sparse kernel does not exploit the A100" and
attributed the reversal to the kernel. **That was an inference from end-to-end
prefill timings presented as a mechanism, and it is wrong.** Stage 2 on A100,
batch 1, matched geometry (32 q-heads / 8 kv / 128 dim), measures the kernel
with the mask already built:

| seq | card | flash ms | bs 0.75 ms | flash ÷ bs |
|---|---|---|---|---|
| 4096 | L4 | 2.049 | 1.750 | 1.17× |
| 4096 | **A100** | 0.841 | 1.175 | **0.72×** |
| 8192 | L4 | 9.473 | 3.762 | 2.52× |
| 8192 | **A100** | 2.960 | 1.969 | **1.50×** |
| 16384 | **A100** | 11.363 | 5.386 | **2.11×** |

**The block-sparse kernel BEATS flash attention on the A100** — by 1.50× at
8192 and 2.11× at 16384. End-to-end at those same configurations it loses
(0.735× and 0.615×). The penalty therefore lives **outside the attention
kernel**, in per-call work the kernel never sees — mask conversion is the
known candidate, already measured on L4 as a constant per-call tax.

What survives from the original claim is narrower and still true: **flash
gains more from the better card than block-sparse does** — 3.20× against
1.91× at 8192, 2.44× against 1.49× at 4096. So the sparse kernel's *margin*
shrinks on the A100 (2.52× → 1.50× at 8192). But a shrinking margin is not a
lost race, and the race is lost somewhere else.

### The same comparison at the model's real head geometry

Measured 2026-09-17 on the same A100, Stage 0/1 re-run first so the cells are
licensed (`block_sparse` passes 27/27 at `12:2`, including cross-backend
checks at both 8192 and 16384 against an independent implementation).

The table above is at `head_layouts` of `(32,32)` / `(32,8)`. **Qwen2.5-1.5B
is `(12,2)`**, and that mismatch is what invalidated an earlier attempt to
scale kernel cells by 28 layers. Re-measured at the model's own geometry,
batch 1, block_size 128:

| seq | sparsity | flash ms | bs ms | flash ÷ bs | per-call saving | × 28 layers |
|---|---|---|---|---|---|---|
| 4096 | 0.5 | 0.509 | 1.065 | **0.48×** | −0.556 | **−15.6 ms** |
| 4096 | 0.75 | 0.509 | 0.971 | **0.52×** | −0.462 | −12.9 ms |
| 4096 | 0.9 | 0.509 | 0.889 | **0.57×** | −0.380 | −10.6 ms |
| 8192 | 0.5 | 1.570 | 1.544 | 1.02× | +0.026 | +0.7 ms |
| 8192 | 0.75 | 1.570 | 1.230 | 1.28× | +0.340 | +9.5 ms |
| 8192 | 0.9 | 1.570 | 1.133 | 1.39× | +0.437 | +12.2 ms |
| 16384 | 0.5 | 4.434 | 3.373 | 1.31× | +1.061 | +29.7 ms |
| 16384 | 0.75 | 4.434 | 2.266 | 1.96× | +2.168 | +60.7 ms |
| 16384 | 0.9 | 4.434 | 1.795 | **2.47×** | +2.639 | +73.9 ms |

**The direction of the kernel finding survives the geometry correction; the
magnitude does not.** At matched sparsity 0.75 the ratio is lower at every
band than the 32-head sweep reported — 0.52× vs 0.72× at 4096, 1.28× vs 1.50×
at 8192, 1.96× vs 2.11× at 16384. The 32-head geometry **overstated** the
kernel's advantage at the geometry that actually runs.

That cuts against the explanation, not for it. The kernel's contribution to a
forward at 16384/0.75 is +60.7 ms, not the +167.4 ms the 32-head cells
implied — so whatever accounts for the end-to-end loss has to account for
*more* of it, from a kernel that saves *less*.

**A result that only appears at the real geometry:** at 4096 the block-sparse
kernel is **slower than flash at every sparsity** (0.48–0.57×), a kernel-level
loss with no mask-construction involved. At 12 query heads and 2 KV heads the
4096 problem is too small to amortise the kernel's fixed work. The 32-head
sweep showed 0.72× here, close enough to parity to read as noise; at the real
geometry it is an unambiguous loss and it is the short-context boundary of the
kernel's usefulness on this card.

**The 28× multiplier in the last column is arithmetic, not a measurement.**
It is printed because it makes the scale legible, and it must not be added to
a separately measured mask-construction cost to predict an end-to-end gap:
that composition has now been wrong in both directions (undershooting by
~2.5× with laptop construction timings, overshooting by 1.7–2.5× with the
instance's own). The end-to-end number is measured end-to-end.

**The magnitude of that overhead is NOT quantified here.** The obvious
arithmetic — scale the per-call kernel saving by the model's 28 layers and
compare to the end-to-end delta — is invalid, because the Stage 2 grid times
32 query heads while Qwen2.5-1.5B has 12. The two datasets establish the
*sign* and the *location* of the penalty, not its size. A matched-geometry
run would be needed for that.

**The obvious alternative explanation was checked and rejected.** "Sparse is
slow on the new card" is exactly what a missing sm_80 kernel would look like.
It is not that:

- Stage 0 on A100 records `block_sparse` **supported** on 180 configs
  (`claimed=True / actual=supported`). The `unsupported` rows are `mask kind
  causal not wired`, which is expected — block-sparse does not serve plain
  causal.
- Stage 1 on A100 passes **90 of 90** block-sparse correctness cells against
  the float64 reference.
- The kernel responds to sparsity **more** steeply on A100 than on L4 — 65.7%
  spread across 0.5→0.9 at 32768 against the L4's 31.2%. A fallback path would
  not track sparsity at all, let alone better.

So the kernel is present, correct, and doing real sparse work. It loses a race
against a much better dense baseline.

| | |
|---|---|
| **Supported** | *The end-to-end block-sparse prefill speedup is **hardware-conditional**. On an L4 it reaches 1.373× at 32768/0.75; on an A100 the same configuration is 0.475×, and sparse is slower than dense at every band and sparsity measured.* |
| **Supported** | *At the kernel level the block-sparse kernel is FASTER than flash on A100 at long context — **at the model's own `(12,2)` head geometry, 1.28× at 8192 and 1.96× at 16384 (sparsity 0.75), rising to 2.47× at 16384/0.9**. The end-to-end reversal is therefore caused by per-call work outside the attention kernel, not by the kernel. Earlier figures of 1.50× / 2.11× came from a 32-head sweep and **overstated** the advantage at the geometry that runs.* |
| **Supported** | *At 4096 the same kernel is **slower** than flash at every sparsity measured (0.48–0.57×) at the model's real geometry, with no mask construction involved. The kernel's usefulness on this card has a short-context floor.* |
| **Not established** | *That the kernel saving and the mask-construction cost compose into the observed end-to-end gap.* Composing them has been wrong in both directions — undershooting by ~2.5× with laptop construction timings and overshooting by 1.7–2.5× with the instance's own, from the same arithmetic with a strictly better input. The end-to-end effect is measured end-to-end (item 4) rather than predicted. |
| **Supported** | *Flash attention gains more from the better card than block-sparse does: 3.20× against 1.91× at 8192. The sparse kernel's margin shrinks on A100 but does not invert.* |
| **Not supported** | *The overhead outside the kernel is N milliseconds.* Not computable from these two datasets: Stage 2 times 32 query heads, the model has 12, so per-layer scaling between them is invalid. Sign and location only. |
| **Not supported** | *Block-sparse attention is useless on A100.* One kernel (`block_sparse_attn`, block size 128), prefill only, batch 1, one model. A better-optimised sparse kernel is not excluded by this. |
| **Not supported** | *The A100 result invalidates the L4 result.* Both are correct measurements of their own hardware. What is invalid is any sentence that states either number without naming the card. |

**No banked corroboration existed when the first version of this section was
written.** The A100 Stage 2 sweep contained **no `block_sparse` rows at all**
(`mask: ['causal']` only) — not a capability limit, but a flag: `run_sweep.py`
builds sparse cells only when `--mask-source random` is passed, and that
session never passed it. Writing a mechanism without it was the error; the
107-cell sweep that corrected this took **two minutes** of GPU.

---

## Positioning against Sparse Frontier

This study's contribution is defined relative to Sparse Frontier, and the
sentences differ sharply depending on which of their claims is being answered.

| | |
|---|---|
| **Supported** | *Sparse Frontier establishes accuracy-vs-sparsity trade-offs without measuring realised wall-clock speedup on the hardware. This study measures it, on two cards, and finds the answer is hardware-conditional: 1.321x at 32768 at 0.75 sparsity on an L4 with no accuracy loss, 0.475x on an A100 — and on the L4 the importance oracle producing that accuracy costs 35x the latency it saves.* |
| **Not supported** | *This study contradicts Sparse Frontier.* It does not. It measures a quantity they scope out, on one sparse family (block-sparse), in one regime (prefill), on one model size (1.5B). Where the two overlap, they agree. |
| **Not supported** | *Sparse attention does not pay off.* Prefill-only, block-sparse-only, oracle-masked, 1.5B. See the scope banner at the top of `limitations.md`. Their own positive results are strongest in regimes this study excludes by construction — decode sparsity, and large-batch serving, which per their Appendix B.3 is a decode phenomenon because weights load once per forward pass regardless of batch. |

**The batch axis is not a gap.** Their Appendix B.3 states that for prefilling
all cost components scale linearly with batch size, so the attention-to-total
ratio stays constant. A prefill sparsity result is therefore batch-invariant
**by their own model**, and this study's batch=1 measurements do not need a
batch-size caveat. The large-batch regime where sparse attention pays is
decode, which is out of scope. One limitation, not two.

**Their block size is unreachable here, and the bias is in the safe
direction.** Block-Sparse-Attention hardcodes 128 and flex's 64 is
shared-memory-capped on sm_89, so this study cannot measure the block size
their sweep finds optimal. Accuracy at a given sparsity is therefore
**conservative** relative to their optimum, which biases the matched budgets
toward understating sparse attention rather than overstating it.

**The cheap estimator is not a substitute for the oracle — measured, not
argued.** The obvious rebuttal to "the oracle costs 35x the latency it saves"
is "then use the cheap estimator a deployed system would use." That was run
as its own arm on 2026-09-16: MInference's mean-pool estimator (arXiv:
2407.02490, Algorithm 3) against the dense-softmax oracle, identical
examples, one band (16384), n=100 per task, `git_dirty=False`, both arms on
one host with clocks locked.

| task | dense | oracle 0.50 / 0.75 / 0.90 | cheap 0.50 / 0.75 / 0.90 |
|---|---|---|---|
| `niah_multikey` | 66.0 | 66.0 / 59.0 / 20.0 | **34.0 / 19.0 / 1.0** |
| `vt` | 78.8 | 88.2 / 91.6 / 90.8 | 85.0 / 86.2 / 82.2 |
| `niah_single` | 100.0 | 100.0 / 100.0 / 100.0 | 100.0 / 100.0 / 100.0 |

The dense reference is identical in both arms to the decimal (66.0 / 100.0 /
78.8), which is the internal control that the two runs are comparable.

| | |
|---|---|
| **Supported** | *At 16384 with 128-token blocks and a head-uniform budget, MInference's mean-pool estimator loses 32 to 40 accuracy points against a dense-softmax oracle on the one task in this set with headroom, and 3 to 9 points on `vt`, with the gap widening as sparsity rises.* |
| **Not supported** | *Cheap importance estimation does not work.* One estimator, one band, one model, one block size — and that block size is the known confound below. |
| **Not supported** | *This reproduces Sparse Frontier's Block-Sparse.* It does not: theirs is 16x16 with per-head selection and binary-search top-k. Three declared divergences below. |

**The block-size divergence is a threat to exactly this experiment, and it
cuts against the cheap arm.** Block-Sparse-Attention hardcodes 128; Sparse
Frontier's ablation selects 16. Pooling dilution scales with block size — a
block mean underranks a block whose mass sits in a few tokens, and a 128-token
block has eight times as many tokens to dilute across as a 16-token one. The
cheap estimator is therefore handicapped here **relative to the spec it was
taken from**, and the gap above is an upper bound on the gap at their block
size, not an estimate of it. This was identified as a threat before the arm
was built, not after the result came in. Anyone citing the table must carry
this sentence with it.

**Three declared divergences from Sparse Frontier's Block-Sparse:** block
granularity (128 here, 16x16 theirs — above); head uniformity (one shared
budget here, per-head adaptive selection theirs); and sink preservation, which
was a genuine defect here until 2026-09-16 and is now fixed to match their
"always preserve the first key block and the diagonal."

**One sparse family, not sparse attention.** Their finding is that
Vertical-Slash is best for retrieval and block-sparse for high-dispersion
tasks. This study implements only block-sparse, so every conclusion is about
block-sparse and none generalises to sparse attention as a class.

**They concede the gap explicitly, and this is the strongest positioning
available.** Sparse Frontier's Limitations section (arXiv:2504.17768**v2**,
27 Jan 2026, third paragraph) states:

> we report hardware-agnostic computational costs (FLOPs and memory access)
> rather than wall-clock timings

The rest of that paragraph, paraphrased: they justify the choice on the
grounds that wall-clock numbers are implementation- and hardware-specific,
and that FLOPs and memory access are the portable quantity. That is a
defensible methodological decision, not an oversight — which is exactly why
this study is positioned as *measuring a quantity they scoped out*, never as
correcting them. The supported row above stands on their own sentence.

**Pin the revision.** v1 (April 2025) and v2 (January 2026) differ
substantially; cite v2 or the quotation may not be there. Any claim in this
file about what Sparse Frontier says refers to **v2** unless it names v1.

---

## How to add to this file

A row belongs here when the honest claim and the appealing claim differ by a
qualifier that a reader would not miss. If the qualifier is obvious from the
supported sentence alone, it belongs in `limitations.md` instead — this file
is for the ones that get lost.
