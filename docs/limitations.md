# Limitations

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

### cuDNN fused attention faults the device above 8192 on sm_89

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
observation. **Confirm at 16384 whenever a future session touches that band.**

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

## The sparse arm is exactly-verified to 4096, and inferred above it

**The honest statement.** `block_sparse` is verified against a float64 naive
oracle (`masked_exact`, same mask on both sides) at every seq_len up to and
including 4096, across both block sizes. Above 4096 there is no numerical
verification at all, and its correctness is *inferred from the kernel's
length-independence* rather than measured.

That is narrower than "unverified", and it is a consequence of hardware, not
of methodology. Three independent constraints converge, and **any one of them
failing differently would have left a usable reference**:

| candidate reference | why it cannot serve above 4096 |
|---|---|
| the five dense kernels (fa2, sdpa_×4) | they *decline* `block_sparse` — `claims_support` returns `(False, "no block sparse")`. Run anyway they ignore the mask and compute plain causal attention, which is a confident answer about a different function |
| `naive` (the float64 oracle) | needs 16 GiB at 8192, **64 GiB at 16384, 256 GiB at 32768** against a 23 GiB card |
| `flex` | cannot lower block-sparse above 1024 on sm_89: 114688 B of shared memory required against a 101376 B limit |

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

**Open on other hardware.** Whether `flex` lowers block-sparse on an A100
(80 GB, sm_80) is untested. If it does, it restores a third opinion and gives
the sparse arm real cross-backend verification at 8192 and 16384 on at least
one architecture — see `docs/stage2_plan.md`, A100 session.

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
