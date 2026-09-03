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
