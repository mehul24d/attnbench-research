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

## Stage 1 "passed" means four different things

One column, four meanings. Every correctness row records `check_kind`, which
has **no default** — a default would let a weaker verdict be constructed as
`"exact"` and read that way forever after.

| `check_kind` | applies to | what a pass certifies |
|---|---|---|
| `exact` | dense backends, seq_len ≤ 4096 | agreement with a float64 naive softmax oracle within the dtype's tolerance |
| `masked_exact` | sparse backends | the same comparison, with the oracle given the **same** block-sparse mask |
| `cross_backend` | any dense/sparse backend, seq_len > 4096 | agreement among **three** independent implementations at float32 tolerance. **No oracle was consulted.** |
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
