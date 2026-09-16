# Limitations

> **Drafting the write-up? Start from `docs/claims.md`, not here.** This file
> records *why* each limit exists and is written to be read in full. That one
> records the *sentences* — each supported claim beside the near-paraphrase it
> must not become — because the qualifiers below are what get dropped when 900
> lines are compressed into a paragraph.


---

## SCOPE, before anything else: this study measures sparse PREFILL

Every result in this study is about **prefill**. Decode runs dense in every
arm, by construction (`GenerationResult.decode_backend` is `sdpa_math` for
every `block_sparse` row). So:

> **What is supported:** block-sparse *prefill* attention, at long context,
> against dense prefill on the same hardware.
>
> **What is NOT supported:** any claim about sparse attention in general, or
> about sparse *inference* as a system. "Sparse attention is dominated by
> dense" is broader than this data, and is not a sentence this study may
> write.

Decode sparsity is a **different mechanism with different constraints**, not a
config change: it selects pages or tokens against a KV cache rather than
estimating block importance over a full prompt, and the published evidence is
that it tolerates *higher* sparsity than prefill does. It is out of scope
here, deliberately, and the boundary is the same one Sparse Frontier's own
taxonomy draws between the two axes. A reader who wants the decode answer will
not find it here, and should not read the prefill answer as standing in for
it.

The full reasoning is in **Sparsity is applied during prefill only** below;
this banner exists because that section was previously 131 lines in and the
scope it sets is the first thing a reader needs.

---

Things a reader of this study's numbers needs to know before comparing them
to anything else. Each entry names what is affected and what is not.

## Accuracy scoring is more permissive than Sparse Frontier's

Our accuracy figures use RULER's own metric functions, vendored verbatim
(`attnbench/_vendor/ruler/scoring.py`): `string_match_all` and
`string_match_part`, which score by **substring containment**:

```python
1.0 if r.lower() in pred.lower() else 0.0
```

A prediction counts as correct if the expected string appears anywhere in it.
A verbose, hedged, or partly-wrong answer that happens to contain the right
value scores full marks.

The Sparse Frontier codebase (`PiotrNawrot/sparse-frontier`, Apache 2.0)
re-implements the same task families with substantially stricter scoring: it
regex-extracts numbered answers in a required format
(`r'(\d+)\.\s*The answer for\s+([\w-]+)\s+is:?\s+(.+?)(?:\.|$)'`), normalises
whitespace and casing, and requires an **exact match** on both key and value,
with the answer index lining up.

**Consequence: our absolute accuracy will read higher than theirs, and the
two are not comparable.** A number from this study should never be placed
next to a published Sparse Frontier number, or a published RULER leaderboard
number, as though they measured the same thing.

**What this does not affect.** Every backend in a given cell is scored by the
identical function on identical inputs, so *backend-versus-backend*
comparison -- which is the entire point of this study, and what
`analysis/matched.py`'s non-inferiority tests operate on -- is unaffected. A
permissive metric raises all backends together; it does not favour dense over
sparse or vice versa. The matched-accuracy conclusions stand on their own
terms.

The stricter metric would be a reasonable thing to add later as a second
scoring column, since it needs no new data -- only a re-score of stored
predictions.

### And `vt`'s prompt carries another model's chat markup

Found 2026-09-06, while settling the generation contract. The `vt` prompt ends:

```
Question: Find all variables that are assigned the value 55280 in the text
above. [/INST] Answer: According to the chain(s) of variable assignment in
the text above, 5 variables are assgined the value 55280, they are:
```

`[/INST]` is Mistral / Llama-2 chat markup, vendored verbatim from RULER,
being fed to **Qwen2.5-1.5B-Instruct**, whose markup is `<|im_start|>` /
`<|im_end|>`. RULER templates per model; this study does not, and applies no
chat template at all (see `docs/stage3_generation_decision.md` for why adding
one now would be a second deviation compounding on the first).

**Deliberately not patched.** Changing a prompt mid-project is worse than
carrying a known defect: it would split the dataset into pre- and post-change
halves that cannot be pooled, for a fix whose benefit is unmeasured.

**What it cannot do.** Manufacture a difference *between* backends. Every
backend sees the identical prompt, so the between-backend comparison -- the
study's actual claim -- is untouched, exactly as with the permissive scoring
above.

**What it can do, and this is the part to watch.** Depress `vt`'s absolute
accuracy for everyone. That matters beyond comparability, because Stage 4's
non-inferiority protocol needs headroom: if the dense baseline scores near the
floor on `vt`, there is little room for sparsity to *degrade* into, and the
matched test loses the ability to distinguish "sparsity is harmless here" from
"nothing could have shown a difference here". A ceiling effect and a floor
effect break a non-inferiority test in the same way.

**Prediction recorded in advance, to be checked at Segment 1.** If `vt`'s
dense-baseline accuracy comes back materially below `niah_single` and
`niah_multikey`, the markup mismatch is the first explanation to test -- not a
finding about variable tracking being intrinsically hard. Written down now
because a prediction made after seeing the number is not a prediction, and
because "vt is hard for these models" is exactly the plausible-sounding
conclusion this project keeps having to guard against.

## Task construction is RULER's algorithm, not RULER's benchmark

Two substitutions, both forced by dependencies deliberately not added
(`attnbench/_vendor/ruler/VENDORED.md`):

- **Haystack**: `"noise"` (a repeated filler sentence) or `"needle"`
  (needle-shaped decoys) instead of RULER's `"essay"` mode, which needs NLTK
  plus a separately-downloaded Paul Graham essay corpus. A synthetic haystack
  is more homogeneous than prose, which may make retrieval easier or harder
  in ways not measured here.
- **Needle type**: `uuids`/`numbers` instead of RULER's `words`, which needs
  the `wonderwords` package.

Both raise `NotImplementedError` rather than silently substituting, and
`AccuracyResult.haystack_mode` records which was used on every row.

## Common-words-extraction (CWE) is not implemented

The task needs a large natural-vocabulary word list. RULER and Sparse
Frontier both take a `wonderwords` dependency for it; Sparse Frontier reaches
into that package's **private** API (`random_word._get_words_from_text_file`),
which would be fragile to vendor. Not implemented rather than approximated.

## A different NIAH design exists and was considered

Sparse Frontier's NIAH has **no natural-language haystack at all** --
distractors are synthetic key-value pairs indistinguishable in form from the
needles, so the model cannot use surface features to locate the target. That
is arguably a cleaner retrieval test than a needle sitting in visibly
different filler.

It was deliberately **not** adopted, for one specific reason: it places
needles by uniform shuffle and so has **no depth control**. RULER's NIAH
varies needle depth, and depth is exactly the axis on which a sparse
attention pattern that drops mid-context tokens would be expected to fail.
For a study about sparsity, losing that axis costs more than the cleaner
distractor design gains. Adding it as a *second* task was also rejected on
cost -- it would add Stage 3 GPU-hours the budget does not have spare.

## Sparsity is applied during prefill only

Stage 3 generates text, and generation has two phases with very different
costs. **Sparse attention is applied to the prefill; generation runs full
attention over the KV cache.**

**Why.** The scoring pass ranks the *prompt's* blocks against each other. A
token being generated does not exist when that ranking is computed, so a
sparse decode needs a mask row for it that was never scored. Three ways to
supply one were considered:

| | what it does | why not |
|---|---|---|
| reuse the last scored row | treats generated tokens as belonging to the final prompt block | defensible on adjacency, but the approximation sits exactly where sparsity's harm would show. "Sparse attention degraded the answer" and "our mask extrapolation degraded the answer" become inseparable |
| re-score incrementally | computes the new token's ranking against cached keys | more faithful, and what a deployed method does -- but it puts the estimator inside the measured path, which the section below explicitly excludes. Stage 3 would then measure two different things depending on phase |
| **dense decode** (chosen) | sparse prefill, full attention while generating | states exactly what was tested, with no approximation to caveat |

**Why the cost argument settles it.** At 8192 context with the measured
generation lengths (8-33 tokens; see
`docs/stage3_generation_decision.md`), decode is **under 1% of total
compute**. Sparsifying it saves nothing measurable and adds a confound to the
part that can be measured. Prefill is where the quadratic cost lives and where
every sparse-attention method aims.

**What the claim becomes.** "Does sparse *prefill* change the answer" — not
"does sparse inference change the answer". That is narrower, and it is the
axis the compute argument says matters. Sparse Frontier separates prefill and
decode sparsity as distinct axes for the same reason.

**It is recorded per row, not just here.** `GenerationResult` carries
`prefill_backend` and `decode_backend`, and they differ for every
block_sparse row (`block_sparse` / `sdpa_math`). A reader does not have to
know this section exists to see which backend produced a row's text — and
`AttentionBackend.supports_decode()` makes the fallback explicit rather than
an except-clause: a backend with no decode path refuses to pick one silently,
because a silent fallback would make this design indistinguishable from a bug.

## Importance scores are an upper bound, and their cost is excluded

`SwappableAttentionModel.compute_importance_scores` runs a full dense
softmax pass to derive block importance. A deployed sparse method would use
a cheap approximate estimator instead. So block_sparse accuracy here is an
**upper bound** on what a real deployment achieves, and the estimator's own
cost is excluded from every latency number.

This is deliberate -- it isolates the kernel under test -- but it is the same
move this study criticises in LongCA-bench's random-mask methodology, so it
is recorded on every row as `score_source="dense_softmax_fp32"` rather than
left in a docstring. A future cheap-estimator variant becomes a new
`score_source` value, not a silent change in meaning.

### The oracle is not only a ceiling. It can put sparse ABOVE dense.

Stated as "an upper bound" this reads as a bound on how good sparse can be
made to look. Measured on 2026-09-07 it is stronger than that, and the
difference is qualitative rather than one of degree.

**`block_sparse` at sparsity 0.75 beats the dense baseline on `vt` in all
three measured bands** -- +10.8 at 2048 (86.8 vs 76.0), +5.9 at 4096 (90.6 vs
84.7), **+14.6 at 8192** (85.1 vs 70.5) -- against standard errors of 1.0-1.4
points on n=300 cells. At 8192 that is roughly nine standard errors, in the
same direction, on three independent bands.

Discarding computation cannot improve a model. What can is *where the mask
comes from*: the ranking is derived from the full attention scores, so the
mask concentrates attention on blocks that dense attention itself identified
as important but does not preferentially attend to. On a task that requires
following a chain of assignments, that acts as a denoiser -- it suppresses
distractor blocks the dense model was still spending probability mass on.

If that mechanism is right, **no deployable method can reproduce this
result**, because a cheap estimator computed from partial information does
not know which blocks matter. The oracle is not standing in for a deployable
estimator here; it is supplying information the deployable estimator cannot
have.

Proposed mechanism, not a demonstrated one. What is demonstrated is the
effect and its size. Distinguishing it would need the same grid under a
genuinely cheap estimator, which is a `score_source` this study does not yet
have.

**Consequence for Stage 4 (`analysis/matched.py`).** The matched-accuracy
protocol certifies "the smallest sparsity budget whose accuracy is
non-inferior to dense". Where the oracle can push sparse *above* dense, part
of what clears that bar is oracle-supplied. The protocol stays valid -- it
measures exactly what it claims on the data it is given -- but what it
certifies is narrower than "this budget is free". It is: *this budget is
non-inferior to dense **when the mask is chosen with full knowledge of the
attention scores***. That qualifier belongs in every statement of a matched
budget, not only here.

## Batching does not help, and the reason is not the obvious one

Measured at 16384 on an L4 (re-measured 2026-09-03 with token-exact sizing,
16383 real tokens): memory is **not** the constraint -- batch=8 fits at
10.96 GiB (sdpa_flash) / 11.71 GiB (GLA) of ~22 usable, and batch=12 now fits
too at ~15-16 GiB. Per-example wall time is **flat** once the kernels are
warm: sdpa_flash 1.543 / 1.543 / 1.563 / 1.613 s/example across batch
2->12, and GLA 1.384 -> 1.398 across batch 1->2. At 16K tokens a single
example already saturates the GPU, so the workload is compute-bound and the
weight-reuse argument behind batching does not apply.

This means the batch=1 architectural limit in `compute_importance_scores`
costs nothing in throughput -- a batch-aware rewrite would buy nothing. See
`attnbench/accuracy/batch_scaling.py`.

**The prefill result is batch-invariant by Sparse Frontier's own model, and
this is the stronger argument.** The measurement above is empirical and
L4-specific; their Appendix B.3 makes the general case. For *prefilling*, all
cost components scale linearly with batch size, so the attention-to-total
ratio stays constant and a prefill sparsity result does not depend on the
batch it was measured at. Decoding is the exception, and the reason is
specific: weights load once per forward pass regardless of batch, so the
attention share grows with batch, which is why a large-batch regime can make
sparse decode pay when sparse prefill would not.

**Consequence for this study.** There is no batch hole in the prefill result.
The large-batch regime in which sparse attention becomes favourable is a
**decode** phenomenon, and decode is out of scope (see the scope banner at the
top of this file). Those are one limitation, not two, and conflating them
overstates the gap.

**A warning about how nearly this was reported backwards.** The first version
of this probe had no warmup pass, so each backend's first measured call also
paid CUDA context setup and, for GLA, a full Triton JIT compile. Two
*identical* batch=1 GLA calls in one process measured 5.394 s and 1.384 s --
a 3.9x gap between a measurement and its own repeat. Because the cold call is
always the smallest batch, and the smallest batch is the numerator of the
speedup ratio, the probe confidently reported `gla 4.28x -- batching helps`:
the exact opposite of the truth, from otherwise-correct data, with no error
raised anywhere. The tell was in the raw totals rather than the summary --
GLA's batch=2 run took *less total wall time* than batch=1 (4.436 s vs
6.831 s) while doing twice the work, which no batching effect can produce.
The probe now runs and records a warmup call per backend, and
`batching_speedup` excludes it. Any future timing added to this study should
be assumed to need the same treatment until shown otherwise.

## Context lengths are exact token counts (since the sizing fix)

Earlier revisions sized haystacks by a words-per-unit constant, so a grid
`seq_len` of 16384 produced ~19821 real tokens and every FLOPs/hour estimate
built on the nominal figure ran ~20% low. Contexts are now binary-searched
against the target model's real tokenizer
(`attnbench/accuracy/sizing.py`), so a grid length means what it says, to
within ~1%.

**Results produced before that change are not directly comparable** to ones
after it at the same nominal length, because the underlying contexts were
~21% longer. Throughput measurements (effective TFLOPS) are unaffected --
they were computed at each example's own measured length, so they were always
self-consistent -- but any *grid total* derived from nominal lengths was low.

## Two sparse arms, and why the second one exists

Stage 2 now runs two block-sparse backends: `block_sparse`
(mit-han-lab/Block-Sparse-Attention) and `flex`
(`torch.nn.attention.flex_attention`). This is not redundancy.

BSA's block size is hardcoded to 128 *inside* `block_sparse_attn_func` -- it
is not a configuration knob. Stage 2's grid runs `block_sizes=(64, 128)`, and
`sweep.build_cells` drops cells a backend does not claim, so before flex was
wired up the sweep planned 216 sparse cells at 128 and **zero at 64**. Half
the block-size axis had no sparse arm at all, which is invisible in a results
file: a cell nobody could run looks exactly like a cell nobody asked for.

FlexAttention also needs no build step -- it ships with torch -- so a failed
BSA build session no longer means zero sparse timing results. Where the two
overlap (block_size=128) they are a genuine cross-check of the same nominal
pattern through two independent kernels.

Caveats carried forward: flex's block-mask representation can OOM at long
sequence lengths, and its `torch.compile` step must be warmed outside the
timed region or the first measurement times the compiler rather than the
kernel -- the same class of error as the GLA JIT warmup above.

### flex-sparse and flex-dense are not tiled alike

Stage 2 segment 1 measured flex block-sparse **0/72** on an L4. Inductor picks
exactly one candidate config when `max_autotune` is off -- BLOCK_M=BLOCK_N=128
at head_dim=128, the only head_dim in the block-sparse grid -- and at that tile
size `block_size=64` cannot be lowered at all (64 % 128 != 0, and it raises
rather than skipping because there is only one candidate) while
`block_size=128` needs 114688 B of shared memory against sm_89's 101376 B.

The backend therefore pins `kernel_options={"BLOCK_M": 64, "BLOCK_N": 64}` for
block_sparse, which divides both block sizes and roughly halves shared memory.
**It does not pin them for dense**, which lowers fine at the default and
already has measured rows on this card.

So a flex-dense row and a flex-sparse row at the same shape were produced by
kernels with different tile sizes. Comparing them to each other conflates the
sparsity effect with a tiling effect. Compare flex-sparse against
flex-sparse across sparsity levels, and against the *other* backends at the
same shape -- not against flex-dense. This is a constraint of the card, not a
choice: without the override there is no flex-sparse measurement to compare
with at all.

### The mask is built once per config, not once per call

Until 2026-09-04, `FlexAttentionBackend.forward` called `create_block_mask` /
`to_flex_block_mask` on **every invocation**, which put mask construction
inside Stage 2's timed region. A real deployment builds a BlockMask once and
reuses it across calls and layers, so this was a tax no user would pay, and it
dominated exactly the cells where it matters least: segment 1 measured flex at
4.22 useful TFLOPS at seq_len=1024/batch=1 against FA2's 47.9 on the same
card, with a latency floor pinned near 2.0 ms at every shape -- the signature
of a constant addend, not of a slow kernel. It inflated `peak_memory_mb` too,
since the builder materialises a dense `Q_LEN x KV_LEN` bool before reducing it
to the block grid.

**Every flex row in `results/stage2/segment_20260903_seg1/` predates the fix
and is not comparable to later flex rows.** They are kept as evidence, not as
measurements.

### cuDNN fused attention faults the device above 8192 -- on sm_89 AND sm_80

`sdpa_cudnn` is excluded above `seq_len=8192` and its cells are recorded with
`status="illegal_memory_access"`, not omitted.

Observed 2026-09-04 on an L4 (driver 580.173.02, torch 2.9.1+cu129, CUDA
12.9). The Stage 0 probe reached the 16384 band; **nine backends completed all
84 configs in it and `sdpa_cudnn` wrote zero**, then the process died:

```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
NVRM: Xid (PCI:0000:00:03): 31, pid=1338, name=python3
  MMU Fault: ENGINE GRAPHICS GPC2 GPCCLIENT_T1_3
  faulted @ 0x77aa_5e201000, FAULT_PDE ACCESS_TYPE_VIRT_READ
```

An earlier session that reached 32768 died identically; it was not isolated
per-backend, so it counts as corroboration rather than a second independent
observation.

**Confirmed on Ampere, 2026-09-05.** The A100-SXM4-80GB session reproduced it
at the same band with the same signature -- Xid 31, MMU fault, GPU board serial
1322522008730, recorded in `results/a100/serial_console.log`. That makes it
**four machines and two architectures** (sm_89 and sm_80), same driver
(580.173.02) and torch (2.9.1+cu129) on all of them.

So the fault is **not** a property of Ada, and the section heading has been
corrected accordingly. What it is a property of is not yet established: driver
and torch were held constant across all four observations, so it could equally
be a cuDNN version issue rather than a silicon one. That distinction needs a
second driver or torch build to separate, and this study has not run one.
Stating it as architecture-general is what the evidence supports; stating it as
hardware-independent is not.

**A prior claim here was wrong and is worth recording.** During the A100
session it was reported that the fault "was gone on Ampere", on the strength of
a count of zero `illegal_memory_access` rows. The cells had not run -- the
length cap excluded them -- so the zero was an absence of attempts, not an
absence of faults. `docs/silent_failure_patterns.md` general hazard:
*distinguish "the check passed" from "the check ran."*

8192 is where the line is drawn because it is the longest band cuDNN actually
completed (84/84) — not a round number. Only `cudnn` is affected: `flash`,
`efficient` and `math` all completed 16384, which is also why the declaration
is per-instance. All four SDPA variants share one class, so a class-level
declaration would have deleted three backends' worth of long-context coverage
to work around one.

**Why it is not caught and retried instead.** An illegal memory access
corrupts the CUDA context for the whole process. `try/except` around the call
is useless — the damage is to the context, not the Python frame — so every
subsequent CUDA call in that process fails, which is exactly how it presented:
the traceback pointed at an unrelated `torch.cuda.empty_cache()`, with torch
itself warning that "CUDA kernel errors might be asynchronously reported at
some other API call". The only safe handling is not to launch it. Subprocess
isolation per cell would also work and was rejected on scope: it is a process
pool plus IPC plus timeout handling, and it would pay CUDA context creation on
every one of ~1400 cells, adding minutes of overhead and noise to the very
timing measurements the sweep exists to collect.

**The cost is one data point in a well-covered family.** Dense-exact at 16384
and 32768 still has `fa2`, `sdpa_flash`, `sdpa_efficient` and `sdpa_math`.

**The finding is worth more than the cells.** A shipping cuDNN kernel that
works at 8192 and reads unmapped memory at 16384 on the same card is a result
about a production kernel, reproducible, with a kernel-level Xid to back it.
It is also the starkest form of the hardware-conditional behaviour this study
is about: not a ranking that reorders between cards, but a kernel that works
at one length and faults at another on one card.

### flex-sparse is length-capped on sm_89, and that is a hardware finding

The 64×64 override above is applied **only at `seq_len <= 1024`**
(`_BLOCK_SPARSE_KERNEL_OPTIONS_MAX_SEQ`). Above it, no override is passed and
inductor's default tile applies — which on this card at `head_dim=128` cannot
lower `block_size=64` at all and exceeds shared memory at `block_size=128`. So
**flex block-sparse is effectively unavailable above 1024 on an L4.**

Two reasons, and the second is the honest one:

1. 1024 is the literal extent of the evidence. The 2026-09-04 diagnostic ran
   under a 30-minute cap and bought exactly two configs, both at 1024, both
   agreeing with the float64 oracle (0.008404 at block_size 64, 0.008983 at
   128).
2. The next session died to an **Xid 31 MMU fault** — `ENGINE GRAPHICS GPC2 …
   FAULT_PDE ACCESS_TYPE_VIRT_READ`, a GPU page fault — and the last configs
   logged before it were `seq_len=32768`. Forcing non-default triton tiles at
   32× the verified length is a plausible cause. It is deliberately left
   unconfirmed: a session to test it would cost real money to distinguish two
   outcomes that would be acted on identically, since this cap ships either
   way.

The underlying constraint is not a scoping convenience. sm_89 gives 101376 B
of shared memory per block, and flex's default block-sparse kernel at
`head_dim=128` asks for 114688 B. That is a **real limit of this
architecture**, and it means the block-size axis of Stage 2 has a sparse arm
only at short lengths on an L4 — reportable as a finding about the card, not
merely as missing coverage. A second architecture (H100, 227 KB shared memory
per SM) would very likely not have it, which makes it a cross-architecture
result worth stating rather than a hole to apologise for.

**Tested on Ampere, 2026-09-05, and the prediction holds — but only half of
it.** The A100-SXM4-80GB (164 KB shared memory per SM) was probed at every
band:

| `block_size` | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|
| 128 | supported | supported | supported | supported | supported |
| 64 | supported | error | error | error | error |

So the ceiling **is** lifted, decisively, and for the reason predicted: at
`block_size=128` the kernel that would not fit in sm_89's 100 KB fits in
sm_80's 164 KB, and flex block-sparse runs to 16384. Verified, not merely
launched — `masked_exact` against the oracle through 4096, and
`cross_backend_pair` through 16384.

The `block_size=64` failures are a **different constraint that Ampere does not
touch**, and the error says so plainly:

```
LoweringException: ValueError: Q and KV block size must be divisible by
BLOCK_M and BLOCK_N. We got Q_BLOCK_SIZE=64 and KV_BLOCK_SIZE=64.
```

That is inductor's tiling, not the card's shared memory, and it is the exact
condition `_BLOCK_SPARSE_KERNEL_OPTIONS` exists to override — an override this
code applies only at `seq_len <= 1024`. The 2048+ column is therefore **our
cap, not the hardware**, on both architectures. Whether raising it works on
sm_80 is now a cheap question rather than an expensive one, but it changes what
the sweep measures and has not been changed here.

**Two claims made earlier in the A100 session were wrong and are corrected
here.** That "Ampere does not lift the sparse-arm ceiling" — it does, at
`block_size=128`. And that flex block-sparse fails "on the inductor tiling
constraint, not shared memory" — both constraints are real, they bind at
different `block_size` values, and only one of them is architectural. The
lesson is the same in both cases: two failures with the same surface
(`error` in a probe cell) had different causes, and one summary sentence was
made to cover both.

### The suite could not have caught this, and now can

Every other test in this project asserts *correctness* — outputs, masks,
shapes, dtypes, determinism, causality. None asserted anything about **what is
inside the timed region**. For a study whose entire output is timing
measurements that is a structural gap, and it is why the mask sat in flex's
hot path for as long as the backend existed: the suite was green throughout,
because the answers were right.

`tests/test_timed_region_setup.py` now asks the question for every registered
backend — is this work done once, or per call? It detects two things a backend
cannot hide: factory ops (`torch.ones/zeros/arange/…`) sized by `seq_len`,
which take no input tensor and so depend only on cfg; and calls to the named
view constructors (`BlockSparseMask.to_*`, `create_block_mask`), which take a
tensor and are therefore invisible to the first check.

Run against the code as it stood, it immediately found a **second** instance:
`NaiveAttention` was allocating `torch.ones(S, S).triu(1)` on every causal
call and re-running `to_dense_bool()` plus a `~` on every block-sparse call.
Fixed the same way. The effect is far smaller than flex's — naive is O(S²) in
its own right, so mask construction is a bounded *factor* rather than the
unbounded *addend* that pinned flex to a 2 ms floor — but it is a change to
the timed region and is recorded as one in `canary.BASELINE_CHANGES` rather
than assumed negligible.

Backends needing CUDA or an extension (`fa2`, `gla`, `block_sparse`, `sage`,
`xformers`) report as NOT_EXERCISED on a workstation rather than passing
silently; the same test covers them on the instance, where the suite is a
per-session precondition.

## Sub-block causality: the oracle was leaking

`BlockSparseMask.active` is a grid over BLOCKS, so it can mark the diagonal
block active but cannot express the triangle inside it. `to_dense_bool`
originally expanded the grid without applying causality, so a query could
attend to up to `block_size - 1` strictly future keys within its own diagonal
block: measured at seq_len=256/block_size=64, 8064 such (query, key) pairs,
63 future keys for the worst-case query.

`backends/block_sparse.py` passes `is_causal=mask.causal` to the real kernel
and therefore masked exactly those positions, so the correctness oracle and
the kernel under test computed different things by construction. A gate
comparing them would have reported a mismatch that looked like BSA's fault.
Any accuracy taken through the oracle path would have been optimistic by
whatever a 63-token lookahead is worth on a retrieval task.

Fixed in `to_dense_bool` (a `tril` when `causal`) and matched in
`to_flex_block_mask` (an elementwise `q_idx >= kv_idx` inside `mask_mod`).
`to_block_sparse_attn_mask` deliberately does neither: BSA takes causality as
a separate argument, so encoding it in the block grid would mask twice.

## The study is inference-focused: the backward pass is out of scope

**Decided, not overlooked.** `SweepGrid.passes` contains `fwd_bwd`, and Stage 0
finds **914 supported `fwd_bwd` cells** at ≤16384 — more than the 1000 `fwd`
ones. None of them will be measured, because none of them can be licensed:
Stage 1 has no backward correctness check, so no `fwd_bwd` cell has a pass, and
`build_cells` rejects them all.

**Why not just build the check.** A backward check is not a variant of the
forward one. It needs a float64 oracle for three gradients rather than one
output, a tolerance calibration per gradient (dQ, dK and dV do not accumulate
error alike), a decision about how to grade a `structural`-family backend whose
backward is a different function again, and a memory budget roughly 3.5× the
forward's against a card the forward check already declines at 32768. That is
session-scale work with its own failure modes, not an afternoon.

**What it costs, stated plainly.** LongCA-bench's own conclusion is that the
**backward pass is the major bottleneck across sparse kernels**. Excluding it
means this study cannot speak to training-time cost at all, and a reader should
not extrapolate the forward results to it — sparse kernels that win on the
forward may well lose on the backward, which is precisely the finding
LongCA-bench reports.

**Why it is nonetheless defensible.** The study becomes an inference-focused
one, which is coherent with where it already lives: the Stage 3 accuracy track
is inference-only (RULER, prefill and decode), the decode regime exists in
`AttnConfig`, and the hardware-conditional claim is about serving. An
inference study that says so is a different thing from a general study with a
silent hole in it.

`fwd_bwd` stays in the grid and in the config hash rather than being deleted,
so a future backward segment appends to this dataset instead of colliding
with it.

## The sparse arm's verification ceiling is a property of the card, not the study

**The honest statement, restated 2026-09-05 after the A100 session.** How far
`block_sparse` can be verified is **architecture-dependent**, and the earlier
version of this section stated an sm_89 result as if it were a property of the
method:

| | oracle (`masked_exact`) | peer (`cross_backend_pair`) | inferred only |
|---|---|---|---|
| **L4, sm_89, 23 GiB** | ≤ 4096 | — | > 4096 |
| **A100, sm_80, 80 GiB** | ≤ **8192** | **16384** | > 16384 |

Both terms that moved are hardware. The float64 oracle needs 16 GiB at 8192,
which an 80 GB card has and a 23 GB card does not. And `flex` lowers
block-sparse at `block_size=128` on sm_80, restoring the second opinion that
sm_89's shared-memory limit removed — see the flex-sparse section above.

**Corrected 2026-09-16 after the H100 session: the ceiling is structural, and
it stops moving here.** Reading the L4 -> A100 step as "hardware moved it" was
right about that step and wrong as a trend. Hopper, with the same 80 GiB and a
newer architecture, lands in exactly the same place:

| | oracle (`masked_exact`) | peer (`cross_backend_pair`) | inferred only |
|---|---|---|---|
| **L4, sm_89, 23 GiB** | ≤ 4096 | — | > 4096 |
| **A100, sm_80, 80 GiB** | ≤ 8192 | 16384 | > 16384 |
| **H100, sm_90, 80 GiB** | ≤ **8192** | **16384** | > 16384 |

Not one band further out. The H100 Stage 1 table makes the reason explicit:
of its 55 rows that could not be verified at all, **54 are block-sparse-masked
and only 36 are a memory limit.** The counts by reason:

    54  fa2             declines this config: no block sparse
    54  sage            declines this config: no block sparse
    36  sdpa_efficient  declines this config: no block sparse
    36  naive           OOM at this shape

Only `naive`'s is a size problem. The dense backends' refusal is a
**capability gap**: they do not implement block-sparse masks, so they are not
references that a bigger card makes available — they are references that do
not exist. Above 4096 with a block-sparse mask, `block_sparse` and `flex` are
each other's only possible witnesses, and wherever one of them declines the
config there is no witness at any VRAM.

So the sparse arm's verification ceiling is **not hardware-limited any more**.
Buying a larger card was the correct move once and will not work twice; what
would move it is a third block-sparse implementation, which is the thing the
"second sparse backend" paragraph below already argues is not worth a session
— and that argument is now stronger, because the pair verdict it would produce
is exactly the class of evidence this table already has at 16384.

**A knock-on from the sm_90 backward fault.** `block_sparse` is absent from
the H100 32768 band entirely, because its backward kernel faults there
(`b=16, hkv=32, fwd_bwd` — see the runbook). That costs more than its own
rows: `flex`'s 18 block-sparse configs at 32768 are all recorded unverifiable,
because the one backend that could have been their peer was not in the band.
A missing backend removes its own measurements and every verdict that depended
on it as a witness.

**16384 is a pair, not a panel, and the rows say so.** The
`cross_backend_pair` verdicts at 16384 carry their own caveat text: *"agrees
with 1 independent implementations (flex) within atol=0.04; NOT verified
against a float64 oracle, which would need 1024.0 GiB for this config. WEAKER
EVIDENCE: only 2 implementations survived this shape, below the 3 this study
normally requires, so a shared bug between two kernels would not be caught
here."* Two block-sparse kernels agreeing can still share a *convention* error
— the block-grid causality bug of 2026-09-03 is exactly that shape — which is
what oracle agreement rules out and peer agreement does not. So 16384 is
better than inference and weaker than verification, and it is recorded as its
own `check_kind` rather than folded into either neighbour.

**Why the constraints bind at all.** Three converge, and **any one of them
failing differently leaves a usable reference** — which is precisely what
Ampere demonstrates, by relaxing two of the three:

| candidate reference | why it cannot serve above 4096 |
|---|---|
| the five dense kernels (fa2, sdpa_×4) | they *decline* `block_sparse` — `claims_support` returns `(False, "no block sparse")`. Run anyway they ignore the mask and compute plain causal attention, which is a confident answer about a different function |
| `naive` (the float64 oracle) | needs 16 GiB at 8192, **64 GiB at 16384, 256 GiB at 32768**. Against a 23 GiB L4 that caps it at 4096; against an 80 GiB A100 it reaches 8192 and stops there. The wall is the same wall, one band further out |
| `flex` | on sm_89, cannot lower block-sparse above 1024: 114688 B of shared memory required against a 101376 B limit. **On sm_80 (164 KB/SM) it lowers at `block_size=128` at every band to 16384**, which is what supplies the pair verdicts above |

**Why a second sparse backend was considered and rejected.** FlashInfer would
need building, is forward-only, and would very likely meet its own sm_89
constraints — a session spent to find out. More fundamentally, it would not
buy what it appears to: two sparse implementations agreeing can share a
*convention* error (the block-grid causality bug of 2026-09-03 is exactly that
shape), which is precisely what agreement with a float64 oracle rules out and
agreement with a peer does not.

**What the inference rests on.** The kernel's correctness is not length-
dependent in any way observed here: BSA agreed with the naive oracle to
8.09e-03 (bf16 tolerance 2e-2) given a real mask, and cross-backend divergence
in the segment 1 data *decreased* with length rather than accumulating. The
inference is stated so a reader can weigh it, not buried.

**Answered on other hardware, 2026-09-05.** This section used to end by asking
whether `flex` lowers block-sparse on an A100, and to predict that if it did,
the sparse arm would gain "real cross-backend verification at 8192 and 16384".
It does lower, and the prediction was half right in a way worth keeping: 8192
turned out **better** than hoped (the oracle itself fits, so it is
`masked_exact`, not merely cross-backend) and 16384 **weaker** (one peer, not
a panel — `cross_backend_pair`).

**And this is itself a cross-architecture result.** Which kernels can be
*verified*, not just which run fast, depends on the card. That is a claim about
methodology on real hardware, and it is more useful than the flat limitation it
replaces: a study that reports "sparse is unverified above 4096" is describing
its rental budget, while one that reports the ceiling per architecture is
describing the problem.

**One caveat that travels with these rows.** `max_rel_err` on `masked_exact`
verdicts reaches 3.3e4 while `max_abs_err` stays at 1.3e-2, because masked
positions put near-zero values in the denominator. The pass/fail decision keys
off absolute error for exactly that reason. `max_rel_err` needs a resolution
floor before it is reportable — the same defect as `CANARY_MIN_LATENCY_MS`,
still open.

## Clocks are not locked on every host, and the ratios say so

`nvidia-smi -lgc` needs root and fails on most rental hosts — including the
A100 rented for the second architecture. Two options existed, and both have a
cost:

- **Gate at measurement time.** Refuse to record timing on unlocked clocks.
  This would delete the second architecture entirely, and with it the
  hardware-conditional finding the study exists to produce.
- **Flag at analysis time.** Record `clocks_locked` per row and carry it onto
  every derived comparison.

The second, deliberately. Unlocked-clock variance is real but bounded; losing
an architecture is not recoverable. So the flag rides onto `Speedup`,
`ArchitectureComparison` and `CanaryDrift`, and it is reduced rather than
averaged: **a ratio is as noisy as its noisier half**, so one unlocked side
makes the whole ratio unlocked, and one unlocked host makes its architecture's
mean unlocked.

`ArchitectureComparison.caveat()` is harshest where the result is most
interesting. A *flip* — the backend winning on one architecture and losing on
another — is this study's headline claim, and the canary's drift tolerance is
5%, which is the same order as unlocked-clock variance. A flip between 0.98
and 1.02 on unlocked clocks is not evidence of anything, and the caveat says
so in those words.

**What this does not do:** it does not veto. An unlocked comparison is still
produced and the canary still fires on unlocked data. Flagging that became
suppressing would hide exactly what these checks exist to find.

The raw readings (`sm_clock_mhz`, `mem_clock_mhz`, `persistence_mode`) stay
unconsulted on purpose — nothing can act on 1710 MHz versus 1695, and a check
with no decision behind it is worse than none, because it looks like coverage.

**And `sm_clock_mhz` could not rescue `clocks_locked` even if consulted.** The
obvious repair — derive the flag from the reading rather than accepting it as
an unvalidated parameter nobody passes — does not work with what is recorded.
The column holds **one sample, taken when the provenance stamp is captured**,
not a statistic over the timed region. Across the whole dataset:

| host | `sm_clock_mhz` |
|---|---|
| A100 (sm_80) | 1410 |
| L4 ×3 (sm_89) | 210 |

210 MHz is the L4's **idle** clock; it runs a kernel near 2040. So the reading
does not describe the measurement at all, and a single sample cannot separate
"locked at this value" from "happened to be at this value" in either direction.
Deriving `clocks_locked` from it would manufacture confidence rather than
measure it. The honest fix needs sampling *during* the run, which needs
hardware. Until then the flag is `False` everywhere, on both architectures —
which is at least the symmetric case: both halves of every cross-architecture
ratio carry the same variance, so a reader discounts them uniformly instead of
mistaking a difference in measurement quality for a difference in silicon.


## The cross-architecture claim rests on 11 cells, and one of them carries it

Stage 2 now has two architectures with real data on both -- Ada (L4, sm_89)
and Ampere (A100-SXM4-80GB, sm_80). The headline result is that the GLA/FA2
crossover moves: FA2 wins everywhere on both cards up to 4096, GLA wins from
8192 on the A100, and on the L4 FA2 is still ahead at 8192 and only loses at
16384. One doubling earlier on Ampere.

**What that rests on, stated exactly.** `scripts/run_cross_arch_analysis.py`
restricts to (seq_len, batch) cells measured on *both* cards. There are **11**
of them out of 17, and the winner differs in **exactly one**: seq_len=8192,
batch=1, where GLA is 1.26x faster on the A100 and 19% slower on the L4.

**And that cell is the one cell in the crossover table the instrument cannot
resolve** -- by 0.007. Its A100 side is a 2.16 ms kernel, where this dataset's
own cross-host evidence puts the uncertainty at 27.2%; the margin is 26.5%.
27 of the 28 crossover cells clear their band; this is the one that does not,
and it is the one the headline rested on.

The verdict on each card is separately resolvable, at different batches:

| card | batch | FA2 ms | GLA ms | margin | resolution | verdict |
|---|---|---|---|---|---|---|
| A100 | 1 | 2.73 | 2.16 | 0.265 | 0.272 | **unresolvable** |
| A100 | 4 | 10.68 | 8.36 | 0.277 | 0.133 | gla |
| A100 | 16 | 42.31 | 32.65 | 0.296 | 0.133 | gla |
| L4 | 1 | 9.19 | 10.97 | 0.162 | 0.133 | fa2 |

So the disagreement is real and stands on resolvable measurements on both
sides -- at unmatched batch. What makes that tolerable is that the A100's
ratio barely moves with batch at 8192 (1.265 / 1.277 / 1.296, a 2.4% spread
across a 15x range of kernel time), so batch is not what carries the result.
It is stated here rather than folded into the matched table because it is an
unmatched comparison supporting a claim, which is the thing the rest of this
analysis refuses.

Four things about it that a reader should have:

- **Above 4096, only batch=1 is shared.** The A100 covers batches 1, 4 and 16
  at 8192 and 16384; the L4 covers batch 1. So the claim above 4096 is a
  batch-1 claim. The A100's own batch 4 and 16 rows agree with its batch 1 row
  (1.28x and 1.30x at 8192), which is supporting evidence and not the same as
  matched evidence.
- **32768 exists on the L4 only.** The A100 session was capped at 16384 after
  a cuDNN Xid 31 crash consumed part of it. The L4's 32768 numbers
  (GLA 3.65-3.70x faster) have no Ampere counterpart and are excluded from the
  matched table.
- **Clocks are unlocked on both** (see above). Symmetric, so the ratios are
  uniformly noisy rather than differently controlled -- but 1.26 against 0.84
  is a 50% gap, not a 5% one.
- **Verification depth also differs by card**, which is a second
  hardware-conditional result rather than a footnote to this one -- see "The
  sparse arm's verification ceiling is a property of the card".
- **The environment is genuinely common:** driver 580.173.02, torch
  2.9.1+cu129, torch_cuda 12.9, triton 3.5.1 on all four hosts, and the timed
  region byte-identical in AST after `code_identity.restrict_to_reference_code`
  drops the 172 pre-hoist seg1 rows for `flex` and `naive`. This is the part of
  the comparison that is strongest, and it was not free -- see
  `attnbench/analysis/code_identity.py`.

**19 comparisons flip; NONE clears the resolution its own latencies support.**
`ArchitectureComparison.flips()` counts a backend winning on one card and
losing on the other, and at threshold 1.0 that includes 0.999 against 1.001.
An earlier version of this section reported 4 clearing a flat 5% bar. That bar
was a guess, and the dataset contains a measurement of the right one: the L4
was rented three times, and 75 (backend, config) pairs were measured on more
than one of those hosts with identical driver, torch and triton.

| shorter latency | pairs | median spread | max spread |
|---|---|---|---|
| < 3 ms | 54 | 1.8% | **27.2%** |
| 3-5 ms | 0 | -- | -- |
| 5-10 ms | 7 | 2.0% | 11.3% |
| 10-20 ms | 7 | 5.0% | 13.3% |
| > 20 ms | 7 | 3.3% | 6.2% |

The median is low everywhere, which is exactly why an unfloored analysis looks
healthy; the tail is what disqualifies a claim. Two of the four the 5% bar had
certified were at 1024, where **the two L4 hosts straddle parity by
themselves** -- `fa2` at 1.031 and 0.811, `sdpa_cudnn` at 0.895 and 1.004. The
"flip" was between an average of those and the A100.

Flips are structurally the hardest thing here to resolve: a flip requires one
side near parity by definition, and near parity is where the margin is
smallest relative to the noise. The crossover survives the same standard
because FA2/GLA ratios run from 0.11 to 3.7 and sit far from parity.

The five apparent `flex` flips show the shape plainly: flex sits at 1.00-1.03
against sdpa_flash on the L4, so any A100 value below 1 registers. The real
statement about flex is that it is 15-19% slower than sdpa_flash on the A100
and level with it on the L4 -- a magnitude difference, not a reversal.

**Marginal summaries of this dataset are not reportable.** Any median over
seq_len is refused by `analysis.composition`, because OOM attrition changes the
batch composition between bands and the resulting number can move opposite to
every cell inside it. See `docs/silent_failure_patterns.md` instance 13. The
per-cell tables are the result; there is no valid one-number version of them.

## Stage 1 "passed" means five different things

One column, five meanings. Every correctness row records `check_kind`, which
has **no default** — a default would let a weaker verdict be constructed as
`"exact"` and read that way forever after.

| `check_kind` | applies to | what a pass certifies |
|---|---|---|
| `exact` | dense backends, seq_len ≤ 4096 | agreement with a float64 naive softmax oracle within the dtype's tolerance |
| `masked_exact` | sparse backends | the same comparison, with the oracle given the **same** block-sparse mask |
| `cross_backend` | any dense/sparse backend, seq_len > 4096 | agreement among **three** independent implementations at float32 tolerance. **No oracle was consulted.** |
| `cross_backend_pair` | as above, where the shape cost us a reference | agreement between **two** implementations only. Weaker again — a bug shared by both would not be caught. **No oracle, and no third opinion.** |
| `structural` | linear backends (GLA) | finite output, correct shape and dtype, determinism, and causality. **Not numerical agreement of any kind.** |

**Why there is no oracle above 4096.** A float64 naive reference materialises
the full S×S score matrix: at Stage 2's geometry (batch 1, 32 heads) that is
4 GiB at 4096, 16 GiB at 8192, **64 GiB at 16384 and 256 GiB at 32768**,
against a 23 GiB card. The oracle cannot be allocated at the lengths this
study is actually about. The alternatives were capping Stage 2 at 4096 (which
deletes the subject matter) or inheriting a 4096 pass upward (which assumes
precisely what Stage 1 exists to measure — that numerical error does not
accumulate with length).

Three agreeing implementations, not two: two kernels sharing a bug is
plausible — a common upstream, a shared CUTLASS path — while three from
different authors is much less so. A single disagreeing backend is recorded
as a **finding**, not skipped: it means one of three implementations is wrong
at that shape, and which one is a question worth answering.

The measured agreement error is recorded per row, and every row carries its
`seq_len`. So cross-backend divergence **as a function of length** is readable
as a result in its own right — numerical fidelity at long context is one of
the gaps this study targets, not merely a gate to clear.

### Cross-backend verification is itself memory-bounded, and runs out first

The oracle's 23 GiB ceiling is well known here. The cross-backend check has
one too, and it went unmodelled until it took down three consecutive rented
sessions. The check holds, concurrently: Q, K and V (no reference can free
them, since every reference needs them); the tested output; one reference
output; the transient GQA expansion of K and V to the query head count inside
a reference's forward; and three float32 batch-slices for the comparison.
`gates.cross_backend_bytes` computes that term by term — deliberately not as a
single headroom factor, because the oracle's first bound *was* a guess (a
`seq_len ≤ 4096` cutoff that assumed batch 1) and was wrong by 16× at batch 16.

Against a 12 GiB budget on a ~22 GiB card, the resulting picture:

| shape (32 heads, head_dim 128, bf16) | concurrent tensors | verdict |
|---|---|---|
| 8192, batch 16, GQA 32:8 | 5.4 GiB | runs |
| 16384, batch 16, GQA 32:8 | 10.8 GiB | runs |
| 32768, batch 16, GQA 32:8 | **21.5 GiB** | **declined in advance** |

**A declined check is a result, not a gap.** "Cross-backend verification is
infeasible at 32768/batch 16 on 24 GB" is a true statement about what this
study can verify on this hardware, and it is recorded as such: `passed=False`
with a detail that names the size needed, the budget, and — explicitly —
that nothing was run and the row is *not* a verdict on the backend. The
alternative is finding out by OOM, which costs the run rather than the cell.

Consequence: at the largest shapes there is **no verification path at all**.
The float64 oracle needs 256 GiB, cross-backend needs 21.5, the card has 22.
Those cells are honestly unverifiable here, and the constraint is the card,
not the kernels.

**A reference that cannot lower is not a verdict on the backend under test.**
The 2026-09-04 Stage 1 reported 66 failures, every one of them
`reference flex raised InductorError` — 54 from the sm_89 shared-memory limit
and 12 from the BlockMask/tile divisibility mismatch, both of them flex's own
documented limits above, and in every case flex was a *reference* rather than
the backend being checked. `UnsupportedConfig` and `OutOfMemoryError` from a
reference were already handled as "one fewer opinion"; a device-capability
limit is the same thing arriving under a different exception type, and
treating it as a hard failure cost block_sparse 42 of its 78 cells — the
sparse arm, at exactly the lengths this study is about.

The classification is deliberately narrow: it matches the two known *messages*,
never the exception type. `except InductorError: skip` would be a hatch that
silently downgraded the evidence behind every cell a genuinely broken reference
touched, with no trace, since the cell would pass with one fewer opinion and
no reason to look.

**Where a reference is lost rather than the whole check.** Three
implementations are offered; sometimes fewer survive the shape — one OOMs
running, the comparison itself cannot be allocated, or a reference hits a
device limit. Insisting on three
there would delete the long bands to protect a standard the *hardware* made
unreachable, so two agreeing implementations pass under `cross_backend_pair`,
carrying the weakness on the row rather than in this file. The distinction
that keeps that honest is **attempted-and-lost versus never-offered**: a
caller who simply supplies two references still gets a hard failure, because
nothing was tried and no hardware limit was reached. That is a study-design
gap, and letting it wear the same label as a memory-forced downgrade would
hide the one case where three-way agreement is actually within reach.

**Why linear backends get no numerical check.** Gated Linear Attention is not
an approximation of softmax attention; it is a different function. Graded
against the exact oracle it produced `max_abs_err=1.42e+01, failed=6/6` —
confirmation that two different computations differ, which was known in
advance. A reference GLA implementation was considered and rejected: it would
either be the same `fla` code path under test (circular — certifies nothing)
or a reimplementation (a fresh source of bugs, with no way to tell which of
the two was wrong when they disagreed).

The structural properties are chosen to catch real bugs, and the load-bearing
one is **causality**: perturbing token *t* must not change any output before
*t*. A linear-attention kernel leaking future information would be badly
broken in a way no throughput measurement reveals, and the corruption would
reach Stage 3 as plausible accuracy numbers.

**Consequence for a reader.** A `passed=True` row is not self-describing.
Any statement of the form "all backends passed correctness" must name the
`check_kind` distribution behind it, or it silently equates a float64 oracle
comparison with a structural sanity check.

### And a fifth thing a pass has to mean: that Stage 2 can run it

A correctness verdict licenses a timing cell. That licence is void if the
verdict was earned by code the sweep will not execute — which is not
hypothetical. On 2026-09-03 Stage 1 certified `flex` block-sparse 72/72 with
genuine agreement (0.013–0.021) while Stage 2 could not lower a single one of
those cells on the same card, because `torch._dynamo` had hit its recompile
limit during the probe and silently run `flex_attention` **eagerly** from then
on. The eager path materialises scores and works at any block size; the
compiled path does not exist on an L4 at head_dim=128. See
`docs/silent_failure_patterns.md` instance 8.

`gates.check_for_family` now wraps every check in `compile_guard.guard()`. If
a fallback is detected, `passed` is forced to False and the reason written into
`detail` as `VOID (...)`. `check_kind` and `max_abs_err` are **preserved**:
what was attempted is still the honest label for the row, and the measured
agreement is still a real number. It is the licence that is withdrawn, not the
measurement.

Detection is sticky at process scope, because the dynamo warning fires once —
at the crossing — and every call after it is silent. Per-call detection alone
would void the one cell that happened to cross and clear the hundreds that
followed, which is the exact inverse of the truth.

## An open discrepancy: prefill + decode do not sum to the measured total

**Recorded 2026-09-07, unresolved, deliberately not reconciled.**

Dense `generate()` at 8192 costs **62.1 ms per generated token** end-to-end
(n=900, clocks locked). The session's slope-based decode measurement put a
decode step at **55.5 ms** at that band. Those together leave ~6.6 ms per
token for an 8192-token prefill amortised over ~30 tokens — about 200 ms
total — against roughly **580 ms** implied by the model's own FLOPs at the
measured 42 TFLOPS prefill throughput.

The three numbers do not close, and one of them means something other than
what it has been used for.

**No reconciliation is offered here on purpose.** This project has made the
regime-vs-unit error three times (kernel vs whole-model TFLOPS; GLA's 1.7
against FA2's 61.8; prefill TFLOPS applied to decode), and each time the
plausible arithmetic fix was wrong because a quantity was being read in the
wrong regime. The prior strongly favours a fourth instance of that over a
slip in the multiplication. Guessing which input is misinterpreted, and
adjusting it until the sum closes, would produce a decomposition that agrees
with itself and with nothing else.

**What does not depend on it:** the end-to-end comparison. Dense and sparse
were measured the same way on the same rows, so the ~30% gap and the
domination result stand regardless of how the total divides internally.

**What does:** any per-phase attribution — "sparsity saves X% of prefill",
"decode is Y% of the bill". Stage 5's instrumented timing, with the phases
measured separately rather than inferred by subtraction, is what settles it.
Until then no phase split should be quoted from this data.

### Update 2026-09-12: Stage 5 settled it for the dense arm and reopened it for the sparse one

Stage 5 measured the phases separately, so the split above is no longer
inferred by subtraction. Three of the four banked reconciliation sets close
without systematic bias under the corrected `(n − 1)` identity:

| set | signs | sign-test p | mean residual | closes |
|---|---|---|---|---|
| `results/stage5/` | 10+/14− | 0.541 | +6.5 ms | 24/24 |
| `results/stage5_ols/` | 15+/13− | 0.851 | −105.3 ms | 19/28 |
| `results/stage5_32768/` | 4+/0− | 0.125 | +5.1 ms | 4/4 |
| **`results/stage5_flashdecode/`** | **3+/21−** | **0.00028** | **−143.7 ms** | **15/24** |

**`results/stage5_flashdecode/` carries a systematic bias and it is not
diagnosed.** The sign test is decisive — 21 of 24 residuals share a sign at
p = 0.00028 — and the identity **understates** the observed total. Nine of
its 24 cells fall outside the 10% tolerance.

The structure is worth recording precisely, because it is a fact about the
data rather than a hypothesis about the cause:

- **Every non-closing cell is `block_sparse`.** All eight dense
  (`sdpa_flash`) cells close, at both generation lengths.
- **Every non-closing cell is at the long generation length** (n = 28.7–33.6).
  All twelve cells at n = 8 close, including every `block_sparse` one, whose
  residuals there run −3.3 to +0.9 ms.
- **The residual is a near-constant fraction of the observed total within a
  band**: −0.16 at 2048 (all three sparsities), −0.14 at 4096, −0.29 at 8192.
  In absolute terms 205–701 ms.

So the phase model reconstructs `block_sparse` to within a few ms for eight
generated tokens and misses by a fixed proportion for thirty, at every band
and every sparsity.

**No cause is offered, and that is deliberate.** This project has made the
regime-vs-unit error four times, and each time an arithmetic fix that made
the sum close was wrong because a quantity was being read in the wrong
regime. A near-constant fractional residual has several plausible
explanations — per-call overhead that scales with generation, a decode step
measured under different cache occupancy than generation produces, a mask or
cache cost paid per call rather than per config — and this data cannot
choose between them. Naming one would produce a decomposition that agrees
with itself and with nothing else.

**What this does NOT touch, stated plainly because the number is quotable:**
no published speedup. The headline figures — 1.321× at 32768, 1.186× at
16384, the five-band trend, the oracle ratios — are **wall-clock end-to-end
measurements**, dense and sparse timed the same way on the same rows. They
are not derived from this identity and do not depend on it closing. The
`normalized_ms` column *is* identity-derived, but it is built from
`results/stage5/` (10+/14−, p = 0.54, unbiased), not from this set.

**What it does touch:** any per-phase attribution drawn from
`stage5_flashdecode` specifically, and confidence in the phase model's
transferability across generation lengths generally. The model was validated
at the lengths Stage 5 measured; this says it should not be extrapolated to
a generation length it was not checked at without re-checking.

The finding is now readable from the parquet rather than from a terminal:
every row of every `reconciliation.parquet` carries `bias_detected`,
`bias_sign_test_p`, `bias_n_positive`, `bias_n_negative`,
`bias_mean_residual_ms` and `bias_direction`. It was a printed line until
2026-09-12, which is how a 24-of-24 same-sign result survived unexamined for
a day in 2026-09-07 — see silent_failure_patterns #27.

## Known debt: the analysis provenance stamp is written and never read

**Recorded 2026-09-12. Small, unbuilt, and the fifteenth variation on one
theme.**

Every derived artifact now carries `analysis_tool`, `analysis_git_commit`,
`analysis_git_dirty`, `analysis_host` and `analysis_timestamp` — fourteen
files across Stages 4, 5, 6, 7 and `cross_arch`, all currently clean at a
single commit. **Nothing consults any of it.**

That is the 2026-09-04 situation exactly (silent_failure_patterns #10): a
stamp that was recorded correctly, on every row, and read by no gate. The
measurement stamp has `provenance.stamp_integrity_problems`, ten tests, and a
caller in `sweep.py`. The analysis stamp has none of that, and
`provenance.stamp_analysis` has no test of its own.

**The specific consumer that is missing:** a check that refuses to join two
analysis outputs whose `analysis_git_commit` differs, and refuses any row
with `analysis_git_dirty=True`, mirroring what
`cross_arch.load_segments` already does for measurement stamps against
`git_commit`. Stage 7 reads Stage 6's parquet; Stage 6 reads Stage 5's;
nothing establishes that those were produced by one checkout. On 2026-09-12
the four `reconciliation.parquet` files were stamped `analysis_git_dirty=True`
for several hours — accurately, from a repair run on an uncommitted tree —
and were only noticed because they were read by hand.

Not built because it is one more loop on a study whose remaining work is
writing, not measuring. Recorded so it is known debt rather than a surprise.

## FlashAttention-3 is deliberately excluded, not unavailable

**Decided 2026-09-12, before the H100 session was booked.**

A reader will notice that the study benchmarks FA2 and not FA3 on Hopper, and
the reason is a scope judgement rather than a capability gap. It is recorded
here so the absence is not read as an oversight.

**FA3 is sm90-only.** It has no sm_80 or sm_89 path, so it cannot appear in
the cross-architecture comparison — which is the entire purpose of measuring
a second and third architecture. Adding it would give a faster dense baseline
on one card and a backend that exists nowhere else in the grid, extending the
study's cross-architecture claim by **zero cells**.

**And it is not merely a build.** No FA3 backend exists in this repository:
the registry holds `block_sparse`, `fa2`, `flex`, `gla`, `naive`, `sage`,
`sdpa` and `xformers`. Adding FA3 means a new `AttentionBackend` subclass, a
`Capability` declaring `min_compute_capability=9.0`, Stage 0 capability rows,
Stage 1 correctness gating against the float64 reference, and its own entry in
the exclusion rules — plus a separate compile session from flash-attn's
`hopper` branch, which the v5 image does not contain.

**What this means for the claims.** No sentence in `claims.md` is about FA3,
and none should be. In particular, *"FA2 is the fastest dense attention
implementation on H100"* is **not supported** by this study and is not
claimed: FA3 exists, is designed for exactly that hardware, and was not
measured. The supported form is the one already in the ledger — backend A
against backend B on identical inputs, among the backends actually run.

## Stage 1 on sm90: `passed=False` means two different things

The H100 Stage 1 table has 558 rows: 492 pass, 66 fail. Reading that as a 88%
pass rate would be wrong, because the 66 are two unrelated things:

| | count | what it means |
|---|---|---|
| **unverifiable** | 55 | no reference implementation could run the shape, so nothing was compared. `max_abs_err` is NaN. |
| **numerical disagreement** | 11 | a real comparison ran and the error exceeded tolerance. |

A row that was never checked and a row that was checked and disagreed both
land in the table as `passed=False`. The distinction is recoverable -- an
unverifiable row has `max_abs_err = NaN` and a `detail` naming every reference
that declined -- but nothing in the schema states it, and a consumer grouping
on `passed` will merge them.

This is the mirror image of the `cross_arch` gap recorded elsewhere, where a
backend missing from one architecture silently shrinks the join and an absence
reads as **agreement**. Here an absence reads as **disagreement**. Both are the
same underlying error: a missing measurement being typed as a verdict rather
than as missing.

On the other side the table is clean: **all 492 passes have a real measured
error, none NaN.** A pass always means a comparison actually happened.

### What the 55 unverifiable rows are, and why a bigger card cannot fix them

54 of 55 are `block_sparse`-masked. The reasons the references declined:

    54  fa2             declines this config: no block sparse
    54  sage            declines this config: no block sparse
    36  sdpa_efficient  declines this config: no block sparse
    36  naive           OOM at this shape

Only `naive` is a memory limit. The others are a **capability gap**: the dense
backends do not implement block-sparse masks at all, so there is no reference
to compare a block-sparse kernel against at any context length, on any card.
The 80GB H100 restores the float64 oracle at some shapes where the 24GB L4
lost it, but it cannot manufacture a second block-sparse implementation. Above
4096 with a block-sparse mask, `block_sparse` and `flex` are each other's only
possible witnesses, and where one of them declines the config there is no
witness at all.

### The 11 genuine disagreements are all `naive`, and that is expected

All 11 are `naive` in bfloat16, `max_abs_err` 2.46e-02 to 3.17e-02, at
seq_len 1024-4096. Every fused kernel passes the same configs at ~1.29e-02.

`naive` accumulates the softmax and the value matmul in the input dtype;
FlashAttention-family kernels accumulate in fp32 internally regardless of
input dtype. So the "reference-shaped" implementation is the least accurate
one in the table, by roughly 2.5x. This is a property of the implementation,
not a defect found by the gate -- but it is worth stating plainly, because
"naive" reads as "trustworthy baseline" and here it is the opposite.

## The A100 and H100 correctness tables are 40+ commits apart

Comparing them as an architecture difference is invalid without a commit
check, and the first thing such a comparison shows is an artifact:

    A100  d2d8ceb (2026-09-05)  543 rows, 30 of them check_kind="structural" (all gla)
    H100  9c05dfd (2026-09-16)  558 rows, ZERO structural rows

`gla` produces no correctness rows at all on H100 -- 504/504 `unsupported`.
That is not sm90. It is `b12974b` (2026-09-07), "gla: the forget gate was
random noise, and it made 900 rows meaningless", which made the backend
decline rather than synthesise a forget gate with a ~1.24 token memory
horizon. The A100 run predates it.

So the 30 A100 structural rows are precisely the rows that commit exists to
stop producing. A cross-architecture reader seeing "30 on A100, 0 on H100"
would conclude sm90 lost a capability, when what actually happened is that the
A100 table was measured before a correctness fix landed.

`sage` is similarly absent on H100 (`cfg.quant_scheme must be set`) -- a
configuration gap, not an architecture one.

**The rule.** Any cross-architecture claim must first establish that the
tables being compared were produced at the same commit, or enumerate what
changed between them. Three backends are currently missing from the H100
table for three unrelated non-hardware reasons: `xformers` (version pin
`>=2.7.1,<=2.8.2` vs installed 2.8.3.post1), `gla` (deliberate decline),
`sage` (unset config). None of the three is an sm90 finding, and all three
would look like one.

## Stage 5 absolute latencies do not transfer across sessions; speedups do

The 2026-09-16 replicate measured both long bands from scratch on a second L4.
The two sessions agree very differently depending on what is read out of them:

| band | absolute prefill times | speedup built from them |
|---|---|---|
| 16384 | +1.03% to +2.26% | **−0.19%** |
| 32768 | +0.13% to +1.08% | **−0.01%** |

Every absolute moved in the same direction by a similar fraction, and the
ratio did not move. That is a common-mode scale factor — a different physical
card in a different thermal state — and a ratio divides it out while a latency
carries it.

**The practical rule.** Quote Stage 5 **speedups** across sessions. Do not
quote Stage 5 **absolute milliseconds** across sessions without saying which
session produced them, because a 2% session-to-session shift in the absolutes
is normal and carries no information about the kernels. Every headline in this
study is a ratio, so this costs nothing that is currently published — but the
absolutes are in the parquet, they look quotable, and nothing in the file
warns that they are session-scoped. This is that warning.

**Scope of the thermal check.** During the replicate the card was observed
executing at 1200 MHz against a 1740 MHz lock. The concern was differential
throttling: the dense arm runs longer per call, soaks more thermal budget, and
would bias the ratio it appears in. Both arms moved together on both bands, so
for **these** measurements the throttling is common-mode and the ratios are
safe. That is an empirical result about this pair of sessions, not a general
guarantee about L4 thermal behaviour, and a future band with a much longer
per-call time would need the check repeated rather than assumed.

**One cell carries no signal.** At 32768, sparsity 0.5 measures 1.0049× —
indistinguishable from dense — and its absolute moved +1.08% between sessions
against the dense arm's +0.13%, eight times as much as the thing it is
divided by. Do not quote that cell.

## The attention sink is not forced, and it confounds the random-vs-importance comparison

**Found 2026-09-16, before building the cheap-estimator arm, by reading
`masks.py` and then measuring against 1,107 cached oracle score tensors.**

Sparse Frontier's Block-Sparse (their Appendix A.1.1) **always preserves** two
things outside the sparsity budget: the attention sink (the first key block)
and the diagonal (local context). This study preserves only one of them.

**The diagonal is preserved, in both arms.** `random_block_mask` and
`importance_block_mask` both call `active.fill_diagonal_(True)`, and
`_candidate_rows` excludes the diagonal from every candidate list, so it is
granted free rather than spent from budget. The two arms are symmetric here
and the random-vs-importance comparison is not distorted by local context.

**The sink is not preserved, in either arm.** Key block 0 is an ordinary
off-diagonal candidate. It survives only if it wins a top-k. Measured:

| sparsity | query blocks keeping the sink, oracle | random |
|---|---|---|
| 0.50 | 84.8% | 49.1% |
| 0.75 | 76.2% | 24.3% |
| **0.90** | **65.0%** | **7.5%** |

**It is a ranking outcome, not a budget artifact.** At 0.9 sparsity on a
309-block example, 217 of 308 query blocks drop the sink, and **212 of those
had a non-zero budget and ranked the sink out of it.** Only 5 were the
structural budget<=0 rows (qb 1..5, which get the diagonal alone). So the
oracle is actively scoring the sink below other blocks for roughly two-thirds
of query rows at high sparsity.

**The likely mechanism is pooling dilution.** The sink's attention mass is
concentrated on token 0, and these scores are **block means** over 128 tokens,
which spreads that mass across the block. The sink block's observed rank among
a row's causal candidates is 3rd, 36th, 3rd, 26th and 66th at query blocks 10,
40, 100, 200 and 300 — never reliably first, and worse as the candidate set
grows. That is consistent with dilution, but it is **not directly measured**:
the score cache stores already-pooled block scores, so the within-block
concentration cannot be recovered from it. Stated as the probable cause, not a
demonstrated one. It also explains why Sparse Frontier forces the sink rather
than trusting a pooled score to find it — a forced sink is a correction for
exactly this pooling artifact.

### What this affects

**1. Accuracy at high sparsity is depressed for a reason unrelated to the
importance ranking.** Dropping the attention sink is independently known to be
destructive. Any accuracy number at 0.9 sparsity — including the
`niah_multikey` collapse — is a measurement of *this mask construction*, not
of importance-guided sparsity as Sparse Frontier defines it.

**2. The random-versus-importance comparison is confounded, and the direction
is knowable.** Random retains the sink 7.5% of the time at 0.9 sparsity
against the oracle's 65.0% — an 8.7x enrichment. So the oracle is partly
winning that comparison by incidentally preserving sinks, not purely by
ranking quality. **Any claim that the oracle beats random by margin M
overstates the ranking's contribution by an unmeasured amount.**

**3. It is a divergence from the method being compared against.** This study's
block-sparse is not Sparse Frontier's block-sparse, and the gap is in the
conservative direction for accuracy.

### What it does not affect

Every **timing** number. Masks of a given sparsity have the same block count
whether the sink is forced or not, so kernel latency, the Stage 2 slice, the
conversion tax and every speedup are untouched. This is an accuracy-side
finding only.

### The fix, and what it costs

Forcing column 0 alongside the diagonal is a two-line change to both mask
constructors. The cost is not the change, it is that **every accuracy number
at every sparsity would need regenerating**, and the oracle-versus-random
comparison would need re-running to separate ranking quality from sink
preservation. Not done yet; recorded here so no number is read as though it
had been.
