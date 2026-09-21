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

## SCOPE, second: every speedup in this study names a card, or it is wrong

Added 2026-09-16, after Stage 5 ran on a second architecture and the sign
flipped.

> **What is supported:** block-sparse prefill is faster than dense **on an
> NVIDIA L4 (sm_89)**, up to 1.373× at 32768/0.75.
>
> **What is NOT supported:** that it is faster anywhere else. On an
> A100-SXM4-80GB (sm_80) the same configuration is **0.475×** — sparse is
> slower than dense at every band and every sparsity measured.

The cause is the dense baseline. Moving L4 → A100 at 32768, flash attention
gets **3.63×** faster while the block-sparse kernel gets **1.26×** at 0.75 and
**1.01×** at 0.5. Block-sparse wins on the L4 by beating a dense baseline that
is weak there; where dense attention is well optimised, there is nothing left
to take. Full numbers and the not-a-missing-kernel check are in `claims.md`,
"The speedup does not survive a change of card."

**Why this was not visible earlier.** Stages 0, 1 and 2 span three
architectures, and Stage 2's *kernel* microbenchmarks on A100 contain no
block-sparse rows at all (`mask: ['causal']` only). Every end-to-end and
phase-decomposition number in this study came from one card until this date.
A hardware-conditional result is invisible to a single-hardware study, and it
does not announce itself: the L4 numbers are correct, reproducible, and were
replicated on a second L4 to within 0.19%. **Replication on the same
architecture cannot detect a claim that is true of that architecture.**

---

## The A100 reversal is CPU mask construction, and it is an implementation cost

Measured 2026-09-17, mostly without a GPU. The Stage 5 finding that
block-sparse prefill reverses on A100 has a located cause, and it is not the
attention kernel.

**Per-forward budget, A100, sparsity 0.75, 28 layers — and why it does not
work.** Both inputs have since been remeasured properly: the kernel column at
the model's real `(12,2)` geometry, and the construction column on the
instance's own CPU rather than a laptop.

| seq | kernel ×28 (12:2) | construct ×28 (laptop) | construct ×28 (instance) | net (laptop) | net (instance) | Stage 5 observed |
|---|---|---|---|---|---|---|
| 4096 | −12.9 | 17.6 | 68.5 | −30.5 | −81.4 | **−39.7** |
| 8192 | +9.5 | 53.1 | 196.2 | −43.6 | −186.7 | **−68.8** |
| 16384 | +60.7 | 175.8 | 628.3 | −115.1 | −567.6 | **−274.3** |

**The sign is right in every cell and the magnitude is wrong in both
directions.** With laptop construction timings the composition undershoots the
observed gap; with the instance's own CPU — a strictly more accurate input,
measured on the machine the numbers came from — it overshoots by 2.1–2.7×.
Replacing one measured term with a better measured term made the prediction
worse, which is the signature of an error in the model rather than in its
terms.

So this composition is **not evidence for the mechanism**, and it is kept here
as a record of an approach that failed rather than as support. What the
mechanism rests on instead is the located, directly measured facts below —
that construction is CPU-bound, accelerator-invariant, non-parallelising and
interpreter-dominated — plus an end-to-end run that changes the builder and
nothing else.

**The term that decides the sign does not scale with the accelerator.**
`masks.py` builds masks on CPU by convention, once per layer per forward:

| seq_len | blocks | laptop ms/call | **instance ms/call** | laptop ×28 | **instance ×28** |
|---|---|---|---|---|---|
| 4096 | 32 | 0.625 | **2.447** | 17.5 | **68.5** |
| 8192 | 64 | 1.890 | **7.008** | 52.9 | **196.2** |
| 16384 | 128 | 6.418 | **22.440** | 179.7 | **628.3** |
| 32768 | 256 | 23.092 | **80.539** | 646.6 | **2255.1** |

O(seq_len²), and bit-identical on both cards. **The instance CPU is uniformly
~3.7× slower than the laptop** (a2-ultragpu-1g, 12 vCPU, 6 torch threads), so
every construction figure published from laptop timings understates the cost
on the hardware that actually ran the measurement by that factor. At 32768 the
real per-forward construction cost is **2.26 seconds**, not the 0.65 s
originally recorded. At 8192 the L4's kernel saves
159.9 ms against a 52.9 ms construction cost and wins; the A100's kernel saves
27.8 ms against **the same** 52.9 ms and loses. The GPU got ~3× faster and the
tax did not move. **The net sign is set by how fast the GPU is relative to a
fixed CPU term**, which predicts the reversal is *worse* on H100.

**It does not parallelise.** 1→8 torch threads gives 1.01× / 0.97× / 1.05× at
8192 / 16384 / 32768. So instance vCPU count does not confound the card
comparison (g2-standard-8 has 8, a2-ultragpu-1g has 12); only single-core
clock differs, a much smaller effect.

**"As implemented" is load-bearing — this is an engineering cost, not a
property of sparse attention.** Profiling `importance_block_mask` at 32768:
the Python loop body is 0.212 s per 10 calls while every torch op inside it
totals ~0.013 s. **~91% is interpreter overhead, not tensor work.** The
structure is a per-query-block loop that fancy-indexes with a Python *list*,
then writes results back through an inner `for j in top.tolist(): active[qb,
kvs[j]] = True` — roughly 8,000 scalar tensor assignments per mask at 32768.
Vectorising to a single batched top-k and one scatter would collapse it. A
reader should take this as "the available implementation is CPU- and
interpreter-bound", not "block-sparse attention is inherently CPU-bound".

**The counterfactual, measured rather than predicted.** A minimal
vectorisation of `importance_block_mask` — one masked `argsort`, one
`scatter_` to get ranks, one comparison against a per-row budget, replacing
the per-query-block loop and its inner scalar-assignment loop — was written
solely to answer "would the reversal survive?". It is **not a contribution and
not proposed as a replacement**; the reference implementation's numbers stand
exactly as measured. It produces **bit-identical `active` matrices** across 36 synthetic
configurations (4 seq_lens × 3 sparsities × 3 seeds, zero mismatches) —
**but not on real scores, and that check was misleading.** Against the model's
own cached scores the two builders disagree on a small number of cells per
layer. The cause is fp16 tie-breaking, not a ranking difference: `score_cache`
stores fp16, ~3.0% of candidates at `block_size=128` share a value with
another candidate, and where a tie group straddles the top-k boundary the
reference (1e-9 jitter) and the vectorised builder (stable argsort) keep
different, equal-scoring blocks. `torch.rand` produces essentially no ties, so
the 36-config check exercised the one regime in which the two provably agree.
See `docs/silent_failure_patterns.md` pattern 39.

What holds on real scores, and is what the timing comparison actually
requires, is that the two builders agree on the **per-row active count**
(identical kernel work) and the **per-row kept-score multiset** (identical
ranking). Both are asserted before any timing, and the assertion is checked
against negative controls so the relaxation is not a deletion.

| seq_len | reference | vectorised | speedup | ref ×28 | vec ×28 |
|---|---|---|---|---|---|
| 4096 | 0.630 ms | 0.047 ms | 13.5× | 17.6 | 1.3 |
| 8192 | 1.896 ms | 0.075 ms | 25.3× | 53.1 | 2.1 |
| 16384 | 6.279 ms | 0.319 ms | 19.7× | 175.8 | 8.9 |
| 32768 | 21.962 ms | 1.325 ms | 16.6× | **614.9** | **37.1** |

Per-forward budget at sparsity 0.75, reference versus vectorised:

| seq | card | kernel ×28 | net (ref) | net (vec) | |
|---|---|---|---|---|---|
| 4096 | L4 | 8.4 | −9.3 | **+7.1** | flips |
| 4096 | A100 | −9.3 | −27.0 | −10.7 | **stays lost** |
| 8192 | L4 | 159.9 | +106.8 | +157.8 | — |
| 8192 | A100 | 27.8 | **−25.3** | **+25.7** | **flips** |
| 16384 | A100 | 167.4 | **−8.5** | **+158.4** | **flips** |

**So the A100 reversal disappears at 8192 and 16384 under vectorised
construction, and does not at 4096** — where the A100's sparse kernel is
genuinely slower than flash (0.72× at the kernel level). That floor is a real
kernel result and survives the correction.

The claim this licenses: **the end-to-end speedup requires either a dense
baseline weak enough to beat, or a vectorised mask builder — and the reference
implementation has neither.** That is a statement about the implementation
landscape, not about sparse attention.

**The end-to-end run, 2026-09-17.** Measured rather than composed: one
process, one model instance, one score tensor, arms interleaved, changing only
which function `masks.mask_for` calls. A100, clocks locked, `git_dirty=False`.

| band | sparsity | reference builder | **vectorised builder** |
|---|---|---|---|
| 8192 | 0.5 | 0.544× | **0.968×** |
| 8192 | 0.75 | 0.750× | **1.023×** |
| 8192 | 0.9 | 0.973× | **1.058×** |
| 16384 | 0.5 | 0.405× | **1.090×** |
| 16384 | 0.75 | 0.633× | **1.201×** |
| 16384 | 0.9 | 0.956× | **1.282×** |

**The A100 reversal is the mask builder, not the card.** With the reference
implementation block-sparse loses at every cell measured; with a vectorised
builder it wins clearly at all three 16384 cells, reaching **1.282× at
16384/0.9**. At 8192 it is parity, not a win: 0.5 is a loss (0.968×), and
0.75's +2.3% sits inside the 1.8–5.8% session-to-session spread the reference
arm shows at 8192 between the 2026-09-16 and 2026-09-17 hosts. *(This read
"wins at five of six" until the 2026-09-19 audit; the sentence counted cells,
not cells that clear the floor.)* 32768 was not measured with the vectorised
builder. The dense
control — which builds no mask, so the builder cannot touch it — moved −0.1%
at 8192 and +0.0% at 16384 between the two settings, so nothing else changed.

**The outputs are bitwise identical**, all 8 cells, both bands: `max|Δlogit| =
0.0`, argmax token unchanged. The tie-break disagreement the guard reports (2
cells in 1 row at 16384, max gap 5.239e-10, inside the reference's own 1e-9
jitter) costs nothing observable. Same masks by every measure that matters,
~30× cheaper construction, and the change in latency is attributable to the
builder alone.

**This retires the disjunctive form of the claim.** The earlier phrasing was
"the end-to-end speedup requires either a dense baseline weak enough to beat,
or a vectorised mask builder." The first disjunct is gone: against `sdpa_flash`
on an A100 — the strongest dense baseline in this study, on the faster of the
two cards — the vectorised builder wins outright. What remains is one
condition, and it is an engineering gap rather than a hardware property.

**A caveat on this run's inputs.** It scores seeded random token ids, so its
score distribution shows 0.0% exact ties, against 3.0% measured on real RULER
examples at `block_size=128`. For latency this is immaterial — block-sparse
cost depends on mask *density*, which sparsity fixes, and the per-row active
counts are asserted identical between builders. But the tie-rate figures from
this run describe random input and must not be quoted interchangeably with the
real-example figures elsewhere in this file.

**What is NOT available as a saving:** reusing one mask across layers. Each
layer builds from its own scores (`state.scores[self.layer_idx]`), so the 28
masks genuinely differ and 27 of them are not redundant. The amortisation is
inside a single construction, not across them.

**Geometry caveat.** The kernel column comes from Stage 2, whose grid is
`head_layouts = ((32,32),(32,8))` at head_dim 128, while Qwen2.5-1.5B is
(12, 2). Per-layer scaling between the two is invalid, so the table above
establishes the **sign and the location** of the penalty, not its size. An
earlier estimate of 459.6 ms computed that way was withdrawn. A separate
double-count was also withdrawn: `timing.measure()` takes the mask as a
parameter, so Stage 2's numbers already contain the per-call *conversion*
tax, and adding it again as an external term counted it twice.

---

## The decode-step decomposition is ill-conditioned at short context

The Stage 5 decode step is an **OLS intercept** over `max_new_tokens ∈
[1,2,4,8,16]`, not a direct measurement, and the fit's intercept disagrees
with the separately measured prefill by an amount that shrinks with context:

| band | L4 | A100 |
|---|---|---|
| 2048 | −27.7% | (flagged) |
| 4096 | −13.0% | **−38.1%** |
| 8192 | −5.7% | — |
| 16384 | −2.8% | — |

Total time is not linear in tokens generated at short context, so "the decode
step" is not one number there. The effect is present on **both** cards, shrinks
monotonically on both, and is larger on the faster one — consistent with fixed
per-call overhead being a bigger fraction of a smaller total.

**What it does not touch:** the scoring phase, which is measured directly
rather than fitted, so the cross-architecture scoring ratios (2.52× at 2048
rising to 4.45× at 32768) are unaffected. **What it does touch:** any
cross-architecture *decode* comparison at 2048 or 4096, which should not be
quoted to more precision than a ±30% intercept supports.

This was nearly published as an A100-specific finding. The A100 run emitted the
warnings and the L4 log showed none — but that log covered only 16384 and
32768, the two bands where the effect is smallest on either card. The banked
L4 parquet had the answer all along. A difference in *what happened to be
logged*, read as a difference in *what the hardware did*.

---

> **Sparse Frontier means v2 throughout this file.** Every reference below to
> Sparse Frontier's text -- their Limitations, Appendix A.1.1 (Block-Sparse
> estimator), Appendix B.3 (batch scaling), Appendix D.4 (model size) -- is to
> **arXiv:2504.17768v2, 27 Jan 2026**. v1 (April 2025) and v2 differ
> substantially, and an appendix letter from one revision does not reliably
> name the same content in the other. Any claim here that is v1-specific says
> so explicitly.

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

### The oracle is not only a ceiling. It can put sparse ABOVE dense — mostly WITHDRAWN 2026-09-20

Stated as "an upper bound" this reads as a bound on how good sparse can be
made to look. Measured on 2026-09-07 it looked stronger than that, and the
difference looked qualitative rather than one of degree. **Most of the effect
did not survive re-measurement.** The section is kept because the reasoning
built on it is what the correction has to reach — the same treatment
`claims.md` gives it under "The oracle can put sparse ABOVE dense".

**What was measured, and is withdrawn.** `block_sparse` at sparsity 0.75 beat
the dense baseline on `vt` in all three bands -- +10.8 at 2048 (86.8 vs 76.0),
+5.9 at 4096 (90.6 vs 84.7), **+14.6 at 8192** (85.1 vs 70.5) -- against
standard errors of 1.0-1.4 points on n=300 cells. At 8192 that was roughly
nine standard errors, in the same direction, on three independent bands.

**Two causes, both measured, neither about attention.**

1. **The sink.** Those masks did not force kv block 0 (see "The attention sink
   is not forced" below). Audit S1a re-measured the same 100 examples with it
   forced: **+3.2 / +0.8 / −2.4** at 2048 / 4096 / 8192. The margin does not
   survive at any band at 0.75.
2. **Stopping.** `vt` scores recall over five names under a 40-token cap, and
   the unforced-sink sparse arms stopped early -- 31-183 caps per 300 against
   dense's 214-279. An arm that stops before restating the chain scores full
   on a short list while dense runs into the cap. See "`vt`'s
   sparse-above-dense gap is substantially a STOPPING effect at 1.5B, and not
   at 7B".

Reproducing across three bands tested neither of those, which is why three
agreeing bands felt like confirmation. **Nine standard errors of a quantity
that was not what it was taken for.**

**What survives, and it is narrow.** At 16384 with the sink forced, `vt`
sparse is still above dense -- +9.4 / +12.8 / +12.0 at 0.5 / 0.75 / 0.9 -- but
among pairs that stopped the same way that falls to +2.4 / +4.4 / +8.1, and at
7B the conditioned margin is +0.4 / +2.7 / +6.8. So a residual positive margin
at high sparsity survives conditioning, on both models; the large gross
margins do not.

**The mechanism proposed for it.** Discarding computation cannot improve a
model. What can is *where the mask comes from*: the ranking is derived from
the full attention scores, so the mask concentrates attention on blocks that
dense attention itself identified as important but does not preferentially
attend to. On a task that requires following a chain of assignments, that
would act as a denoiser -- suppressing distractor blocks the dense model was
still spending probability mass on.

**That account is now one of three**, and it is the only one with a testable
prediction. The others are block structure suiting multi-hop tracking (Sparse
Frontier's reading) and whatever the stopping conditioning cannot remove
cleanly, since stopping is a mediator rather than a covariate. **Nothing here
separates them**, and the residual they are competing to explain is 6-8 points
at one band, not the 14.6 this section was written around.

*Until 2026-09-21 this section continued: "If that mechanism is right, **no
deployable method can reproduce this result**, because a cheap estimator
computed from partial information does not know which blocks matter." That
inference was drawn from the withdrawn margins. It is also no longer the state
of the evidence: the same grid under a genuinely cheap estimator has since
been run -- `score_source="minference_meanpool"`,
`results/accuracy_forced_sink_cheap/` and `results/s9_7b_cheap_16384/` -- and
on `vt` at 16384 the cheap arm is +6.2 / +7.4 / +3.4 above dense, i.e. it
reproduces a positive margin rather than failing to. The sentence claimed an
impossibility that the measurement contradicts.*

**Consequence for Stage 4 (`analysis/matched.py`).** The matched-accuracy
protocol certifies "the smallest sparsity budget whose accuracy is
non-inferior to dense". Where the oracle can push sparse *above* dense, part
of what clears that bar is oracle-supplied. The protocol stays valid -- it
measures exactly what it claims on the data it is given -- but what it
certifies is narrower than "this budget is free". It is: *this budget is
non-inferior to dense **when the mask is chosen with full knowledge of the
attention scores***. That qualifier belongs in every statement of a matched
budget, not only here.

**The qualifier survives the withdrawal; its reach shrank.** It was drawn from
margins that were mostly sink and stopping artifacts, so it now rests on the
conditioned residual at 16384 (+8.1 at 1.5B, +6.8 at 7B) rather than on
+14.6 at 8192. The mechanism by which oracle knowledge could flatter a matched
budget is unchanged and the wording above is unchanged; what changed is how
much of the published margin it is being asked to explain.

**Visible in the rebuilt Stage 4.** `oracle_sensitive` is derived from "at
least one sparsity level EXCEEDED dense beyond noise", so the withdrawal moves
the flag itself: in `results/stage4/matched_budgets.parquet`, `vt` at 2048 is
`False` at all three epsilons where the pre-fix file
(`results/_superseded/stage4_prefix_sink/`) had it `True`. Six of nine `vt`
cells still carry it. A caveat that is recorded as a column rather than as
prose moves with the data, which is the argument for recording it that way.

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
silently.

> **WITHDRAWN 2026-09-20. This paragraph ended "…the same test covers them on
> the instance, where the suite is a per-session precondition." Neither half
> of that is supported by the session record.**
>
> Instrumented on a workstation, the test exercises **4 of 11** backend/mask
> pairs — `flex` and `naive` only. `block_sparse` and `sdpa`, which are the
> two arms of every end-to-end comparison in this study, are among the seven
> it does not reach.
>
> The claim that the instance covers them rests on two session logs, and both
> record the test **failing**:
>
> | log | outcome |
> |---|---|
> | `results/stage5/stage5.log:5` | `FAILED test_no_backend_rebuilds_setup_inside_the_timed_region` |
> | `results/h100_20260916_stage0_partial/logs/suite.log:56` | same test, same verdict |
>
> **No log in this repository records it passing on a GPU.** Which backend
> offended, and whether the failure was environmental, is not recorded
> anywhere — not in a commit message, not in this file, not in
> `silent_failure_patterns.md`.
>
> "Per-session precondition" is also not what the harness does:
> `results/stage5/stage5.log` prints `suite rc=1` on line 2 and proceeds to
> phase timing on line 7. And the session that produced the banked
> `results/stage5/phases.parquet` —
> `results/gpu_session_20260907_stage5/stage5.log`, 65 lines — **contains no
> suite record at all**. That file is the source of every `normalized_ms`
> value, which is the latency axis of Stage 6 and the entire Stage 7 decision
> map.
>
> **What is still true.** The two defects this test was written for — flex's
> `create_block_mask` and `NaiveAttention`'s `torch.ones(S, S).triu(1)` — were
> real, were found by it, and were fixed; both backends are among the four it
> does exercise, and `test_the_detector_catches_a_deliberately_reintroduced_rebuild`
> confirms the detector still fires. What is withdrawn is the coverage claim
> for the CUDA backends. **Whether `block_sparse` or `sdpa` rebuild setup
> inside Stage 2's and Stage 5's timed regions is, as of today, unverified in
> either direction.**
>
> Resolving it needs one instance session that runs the suite, captures the
> full failure output rather than the summary line, and records the verdict
> per backend.

### RESOLVED 2026-09-20, and the answer is worse than "unverified"

That session ran (`attnbench-l4-s7-20260920-2158`, audit item S8). Both
backends now have a verdict and neither is clean.

**`block_sparse` does rebuild setup inside the timed region.** The detector
reports `setup ops per call [1, 1, 1]` — a conversion on every call, not just
the first. The call is
[`backends/block_sparse.py:104`](../attnbench/backends/block_sparse.py#L104):

    base_blockmask = mask.to_block_sparse_attn_mask(b, h, device=q.device)

and `BlockSparseMask.__post_init__` **requires `active` to live on the CPU**
(`masks.py:41`, `masks.py:58`). So that line is a host-to-device copy of the
whole block grid, executed once per forward, inside the region Stage 2 times.
A real deployment builds the kernel's mask once and reuses it across calls and
layers — which is the exact argument this detector was written to make about
`create_block_mask`, where the same shape cost a 10× error in a reported
number.

**`sdpa` could not be observed even on CUDA**, reporting `not runnable here:
UnsupportedConfig` on the `causal` config. It is one arm of every end-to-end
comparison in the study, and its timed region has therefore been checked
**nowhere** — not on the workstation, not on the instance.

**The tax is one-sided, and that is the load-bearing fact.** S14 established that the SDPA kernels this study actually measures — `sdpa_math` and `sdpa_flash` — rebuild nothing per call (`per_call = [0, 0, 0]`, on causal and block_sparse alike). So the per-call copy depresses the numerator of every block_sparse-over-dense ratio and never touches the denominator. **Every reported block_sparse speedup is a lower bound on the true one**, arithmetically, independent of how large the tax turns out to be. S11 measured the magnitude (closed 2026-09-21); it did not change the sign.

**What this does and does not mean for the published numbers.** The magnitude
is measured (register item S11, closed 2026-09-21, A100-SXM4-80GB, p50 29.7-34.6
us per call) and the bytes are small — 16 KB at 16384/128, 64 KB at
32768 — so against a millisecond-scale kernel the tax is a fraction of a
percent (0.24-1.9% depending on the cell), and proportionally largest at the
short seq_lens Stage 2 also sweeps. The **direction** is known and it is the
conservative one: `block_sparse` is being measured slower than it is, so every
reported `block_sparse`-over-dense speedup is **understated**, not inflated.
That is the right way round to be wrong, and it is still wrong. S11 has
measured it: no `block_sparse` timing number in this study is free of a
per-call setup cost, and no claim rests on a checked timed region for `sdpa`
at all.

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

**CORRECTED 2026-09-19 — the paragraph above is out of date and no longer
describes the data.** It was true when written (before 2026-09-06, when
`lock_clocks` could not succeed and `capture()` was never passed the outcome;
`silent_failure_patterns.md` #16). Since the 2026-09-06 fix, the flag is an
observed outcome, and the banked data splits by **stage**, not by card:

| `clocks_locked` | what |
|---|---|
| **True** | every Stage 3 accuracy/latency run from 2026-09-06 (L4) · every Stage 5 phase file, both L4 sessions and the A100 · the 7B oracle and cheap arms (A100) · the vectorised end-to-end (A100) |
| **False** | every Stage 0/1 probe and correctness file, all cards · **every Stage 2 kernel sweep, all cards** — `run_sweep` has never attempted the lock (commit `a157f03` stopped it claiming otherwise) |

So the "symmetric case" argument now holds only for Stage 2 — which is where
it matters, because the cross-architecture ranking and the A100 kernel ratios
at (12,2) are Stage 2 numbers, and both halves of those are unlocked. The
end-to-end headlines are locked on both cards. The single-sample limitation
of `sm_clock_mhz` (now `sm_clock_mhz_at_capture`) is unchanged: a locked row
records that the lock call succeeded, not the clock during the timed region,
and the 2026-09-16 L4 replicate observed 1200 MHz against a 1740 MHz lock.


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

**`results/stage5_flashdecode/` carried a systematic bias. Diagnosed
2026-09-19, from banked data, at no GPU cost: it is the decode-kernel
confound entering the reconciliation through its `--observed` input.**

The original record (below, unchanged in substance) described the structure
and deliberately offered no cause. That restraint was right — two cheap
explanations turned out false — and the structure it recorded is what made
the diagnosis possible.

*What was recorded 2026-09-12:* 3+/21−, p = 0.00028, mean −143.7 ms, 9 of 24
cells outside tolerance. Every non-closing cell `block_sparse`; every one at
the long generation length (n ≈ 29–34); all twelve n = 8 cells closing within
−3.3 to +0.9 ms.

**Diagnosis, in four links, each checked rather than inferred:**

1. **The long-generation "observed" totals are not from the flashdecode run.**
   `results/stage3_flashdecode/` holds only `niah_single` with `n_generated`
   fixed at 14. The reconciliation's n ≈ 30 rows match the *original* Stage 3
   bands (`stage3_s1`, `stage3_s1b`) to every printed decimal, in both
   `n_generated` and `latency_ms`.
2. **In those original bands every `block_sparse` row decoded through
   `sdpa_math`; every dense row through `sdpa_flash`.** This is the decode
   confound already documented in `claims.md` ("two confounds"). So a Stage 5
   phase model measured with the sparse arms decoding through *flash* was
   reconciled against totals in which they decoded through *math*.
3. **The magnitude matches.** The measured math-minus-flash decode-step gap
   (from `results/stage5/`, which timed the sparse decode the math way)
   accounts for **83–98%** of the per-step residual in every cell — 8.3 vs
   7.3 ms at 2048, 7.9 vs 7.4 at 4096, 24.5 vs 23.4 at 8192 — and reproduces
   the residual's shape (flat from 2048 to 4096, ~3× at 8192), which neither
   rejected hypothesis did.
4. **A matched-regime reconciliation closes.** The same Stage 5 phases
   reconciled against `stage3_flashdecode` — flash-decode on both sides,
   n = 14 — close in **12 of 12** cells, residuals −38 to +2 ms (≤ 3.3%),
   sign test **p = 0.146**.

**Two hypotheses tested and rejected on the way**, recorded because both are
the natural first guess:

- *The importance scoring pass inside the timer.* It is outside it
  (`generation.generate_one` scores before `t0`), and its cost scales
  288 / 929 / 3,258 ms against a residual that is flat from 2048 to 4096.
- *fp16 scores from a cache hit making mask construction slower than Stage
  5's fp32 cache misses.* A hit does return fp16 and a miss fp32 — a real
  asymmetry — but the full 28-layer mask path differs by ±3 ms between the
  two dtypes, against ~220 ms needed.

**What remains unexplained:** 2–17% of the per-step residual, and a
non-significant negative lean in the matched reconciliation (9 of 12 cells
negative, mean −9.8 ms). The two decode steps in link 3 come from different
sessions, and host-to-host variance of that size is documented elsewhere in
this file, but it is not *shown* to be the cause here.

**A conclusion drawn from the undiagnosed bias was wrong and is withdrawn.**
The earlier text said the result reduced "confidence in the phase model's
transferability across generation lengths generally." It does not.
**Generation length was perfectly confounded with decode kernel** in the
failing set — every long-generation `block_sparse` row decoded through
`sdpa_math`, every short one through flash — so the variable the text blamed
was standing in for the variable that mattered. In a matched regime the model
transfers to n = 14 without bias.

**What this does NOT touch, unchanged:** no published speedup. The headline
figures are wall-clock end-to-end measurements, not identity-derived, and
`normalized_ms` is built from `results/stage5/` (10+/14−, p = 0.54), which
reconciles math-decode phases against math-decode totals — a matched regime,
which is exactly why it was unbiased all along.

**What stops it recurring:** `scripts/run_phase_timing.py` now refuses
`--observed` rows whose `decode_backend` differs from the decode kernel the
phase model measures for that arm.

The finding is now readable from the parquet rather than from a terminal:
every row of every `reconciliation.parquet` carries `bias_detected`,
`bias_sign_test_p`, `bias_n_positive`, `bias_n_negative`,
`bias_mean_residual_ms` and `bias_direction`. It was a printed line until
2026-09-12, which is how a 24-of-24 same-sign result survived unexamined for
a day in 2026-09-07 — see silent_failure_patterns #27.

## Resolved 2026-09-19: the analysis provenance stamp is now read

**`provenance.load_derived` refuses a derived input whose stamp is absent,
dirty, spans more than one commit, or names a different tool than the
consumer expects.** It is wired into `run_decode_confound.py` and
`run_decision_map.py --corrected`, the two scripts that consume derived
files. Run over all 16 stamped parquets in `results/`, 15 passed and one
did not: **`results/stage7/decision_map.parquet`, all 27 rows derived from
a dirty tree at `67af7c4`** — the published decision map. Regenerated at a
clean commit from the same inputs, its content was identical in every row
and column, so the dirt was immaterial; that is now measured rather than
inferred, and the banked file carries a clean stamp.

**Not built, and stated so it is not mistaken for done:** refusing to
*join* two derived inputs produced at different commits. No script
currently joins two derived files, so there is nothing to guard yet; the
first one that does should call `load_derived` on both and compare
`analysis_git_commit`.

*Original entry, 2026-09-12:*

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

This is the mirror image of a `cross_arch` gap, where a backend missing from
one architecture silently shrank the join and an absence read as
**agreement**. That one was fixed on 2026-09-19: `cross_arch.coverage_gaps`
now names every excluded pair by backend and the analysis writes
`coverage.parquet` — on the banked Stage 2 segments, 49 pairs across 7
backends, none wholly absent from an architecture. (The sentence that used to
stand here said the gap was "recorded elsewhere"; it was not, anywhere.) Here an absence reads as **disagreement**. Both are the
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

### What it does not affect — CORRECTED 2026-09-19: it affects timing too

*This subsection originally read: "Every **timing** number. Masks of a given
sparsity have the same block count whether the sink is forced or not, so
kernel latency, the Stage 2 slice, the conversion tax and every speedup are
untouched. This is an accuracy-side finding only." **That is false**, and it
was false when written: the fix changed the budget, not only the candidates.*

The per-row budget is `round((1 - sparsity) * len(candidates))`. Removing
kv_block 0 from the candidate list shrinks `len` by one, so the budget drops
by about `1 - sparsity` blocks — and then the sink is added back free. The net
is roughly `sparsity` extra blocks per row. Counted exactly (the count depends
only on the rule, not on the scores; cross-checked against real 28-layer
masks at 2048):

| band | 0.50 | 0.75 | 0.90 |
|---|---|---|---|
| 2048 | +9.2% (0.559 → 0.610) | +23.9% (0.338 → 0.419) | **+53.8%** (0.191 → 0.294) |
| 4096 | +5.4% | +14.7% | +35.0% (0.152 → 0.205) |
| 8192 | +2.9% | +8.3% | +21.8% |
| 16384 | +1.5% | +4.4% | +12.2% (0.113 → 0.127) |
| 32768 | +0.8% | +2.3% | +6.6% |

*(active blocks, forced-sink vs pre-fix; realised causal density in parentheses)*

**What this does and does not invalidate.**

- **No published number is wrong.** Every measurement was taken on the masks
  of its own era, and every row pairs a timing with the accuracy of the same
  masks. The L4 headline (1.321× at 32768/0.75, 100.0 vs 100.0) is a correct
  measurement of the pre-fix configuration.
- **The current code does not reproduce the L4 configuration.** Re-running
  any pre-2026-09-16 cell today builds a denser mask — by 2.3% at the L4
  headline cell, by 54% at 2048/0.9.
- **Comparisons across the fix are not like-for-like, and the bias is one
  direction.** Everything on the L4 before 2026-09-16 (Stage 2, Stage 3,
  Stage 5 and its replicate) is pre-fix; everything on the A100 from
  2026-09-16 on (Stage 5, the 2026-09-16 and 2026-09-17 kernel sweeps, the
  vectorised end-to-end) is post-fix. So every L4-versus-A100 sparse number
  compares a sparser L4 mask against a denser A100 one, and **understates the
  A100's sparse arm**. The conclusions survive it: the kernel cross-card
  ratios move by at most the block excess (bs gain 1.91× at 8192 could be at
  most ~2.07×, still below flash's 3.20×), and the A100 reversal at 16384/0.75
  (0.615×) is far outside a 4.4% block excess. The A100 kernel's "slower than
  flash at every sparsity at 4096" was measured at realised densities of
  0.559 / 0.339 / 0.205, not 0.5 / 0.25 / 0.1 — removing the whole 35% excess
  at 0.9 would still leave it slower (0.889 ms → at best ~0.66 ms against
  flash's 0.509).
- **"Sparsity 0.9" names a nominal budget, not a density.** At short bands the
  two are now far apart. Any sparsity axis plotted from post-fix data should
  use the realised density `BlockSparseMask` records.

### Why this makes the block-size limitation causal, not incidental

Elsewhere this file records that block size 16 is unreachable here — the
kernel hardcodes 128 and flex's 64 is shared-memory-capped — and treats that
as a precision loss. The sink finding shows it is more specific than that.

Sparse Frontier's ablation selects **16x16** blocks because smaller blocks
consistently performed better. The mechanism above says why, in at least one
concrete case: **a block mean underranks a block whose mass is concentrated in
a few tokens, and the dilution scales with block size.** At 16 tokens the
sink's mass is spread over 8x fewer positions than at 128, so a fine-grained
estimator ranks the sink higher on its own and the forcing rule matters less.

So coarse blocks do not merely lose resolution uniformly. They
**systematically underrank concentrated-mass blocks**, and the attention sink
is the canonical instance of exactly that. A study operating at 128 is
therefore *more* dependent on hand-forced structural rules than one operating
at 16, and its unforced results degrade in a specific, predictable direction
rather than just noisily.

This also bounds what forcing the sink buys: it repairs the one
concentrated-mass block that is known in advance. Any *other* block whose mass
is similarly concentrated is still underranked at 128 and there is no rule
naming it.

**Stated as a directional bias, because that is what it is.** At 128-token
granularity this study's importance ranking systematically underweights any
input whose important information is *concentrated* rather than *distributed*
— and for retrieval tasks that is most inputs, since a needle occupies a few
tokens inside one block. So accuracy under an importance mask here is
**conservative** for concentrated-information inputs, by an amount that grows
with block size and that no result in this study measures. It is not a
uniform precision loss that averages out; it has a sign, and the sign is
against sparse attention.

That is also the argument that makes Sparse Frontier's 16x16 choice
**principled rather than tuned**: finer blocks are not merely more accurate,
they are less biased in this specific direction, which is why their ablation
finds smaller consistently better rather than better-up-to-a-point.

**The estimator this study adds inherits the same bias.** MInference's
Block-Sparse estimation (arXiv:2407.02490, Algorithm 3) mean-pools Q and K
before scoring, so it dilutes concentrated mass exactly as the oracle's
block-mean does — see `minference_meanpool_scores`. Running both arms at 128
therefore compares two rankings that share this bias, which is what makes the
comparison between them clean and what makes neither of them an estimate of
what a 16-granularity method would achieve.

### The fix, and what it costs

**Fixed 2026-09-16.** `_candidate_rows` now excludes kv_block 0 from every
candidate list and both constructors set `active[:, 0] = True`, so the sink is
granted free exactly as the diagonal already was. Verified: sink retention is
100% in both arms at every sparsity, nesting still holds, and the suite is
green (870 passed). Five regression tests were added in
`tests/test_masks_determinism.py`, including one that scores the sink **last**
so a pass proves it is granted outside the budget rather than merely winning
it — nothing in the suite had tested the sink either way, which is why the
defect survived every previous green run.

**Realised density is now above nominal**, because two blocks per row are free
instead of one: measured **0.508 / 0.262 / 0.114** at 32768 against nominal
0.50 / 0.25 / 0.10. The reference implementation binary-searches k to hit a
target exactly; this study does not. *(Added 2026-09-19: those three figures
are the long-band case. The excess grows as context shrinks — 0.610 / 0.419 /
0.294 at 2048 — see the table under "What it does not affect" above, which is
where the "~1%" this paragraph originally claimed turned out to be wrong.)* It
is in the conservative direction for speedup.

> *This line read 0.506 / 0.260 / 0.111 until 2026-09-21, and named no band.
> Those three numbers reconcile with no band the study runs: rebuilding the
> masks gives 0.515 / 0.273 / 0.128 at 16384 and 0.508 / 0.262 / 0.114 at
> 32768, and the density is deterministic given band and sparsity — the budget
> fixes how many blocks are active, only which ones vary — so this is not a
> draw-to-draw difference. The correction is small, in the same direction, and
> changes no conclusion; it is made because an unattributable measurement is
> the thing this file is about, and three digits that belong to no band are
> exactly that.*

**Every accuracy number predating this change is superseded, not deleted.**
The unforced-sink rows are retained and marked, because the forced-versus-
unforced comparison *is* the measurement of what the sink was worth — probably
the cleanest demonstration this study contains of why a mask-construction
detail that papers put in an appendix is load-bearing. Regeneration is
pending.

**Status 2026-09-20: done, and here is what is left.** Audit item S1a re-ran
2048, 4096 and 8192 at n=100 with the sink forced, same example ids, decode
pinned to the banked `sdpa_math`, dense arm as a hard canary (300/300 at every
band) — `results/s1a/accuracy_all_bands.parquet`. Stage 4, Stage 6 and the
Stage 7 decision map were rebuilt from it and promoted to
`results/stage4|6|7`; the pre-fix versions are under `results/_superseded/`.
See `claims.md`, "The derived stages rebuilt on forced-sink accuracy".

**Still not regenerated:** `stage3_flashdecode` (the sparse-decode-through-
`sdpa_flash` comparison at 2048–8192) and `stage3_32768`. Both remain era 1.
The 16384 band was already post-fix (1.5B oracle and cheap, 7B oracle and
cheap) and its 1.5B oracle arm has since moved again, to era 3, under audit
S7.

*This block read "still pending" and listed Stage 4, Stage 6 and the Stage 7
decision map as not regenerated until 2026-09-21. They were rebuilt on
2026-09-20 — the day after the status was written, and a day before it was
read by an audit that had to check the directories to find out. It also said
that at 16384 post-fix `niah_multikey` at 0.5 is +0.0 against dense: that is
the era-2 cell, and S7 re-measured it at −1.0 under the fixed tie-break. Each
is inside the other's CI.*

---

## The oracle ranks a signal fp16 cannot fully separate, and it gets worse at finer blocks

Found 2026-09-17 while investigating why a vectorised mask builder disagreed
with the reference on real data (`docs/silent_failure_patterns.md` pattern
39). Measured directly on banked oracle score tensors — Qwen2.5-1.5B,
`seq_len=16384`, all 28 layers, candidates only (strictly-lower triangle;
the causally-masked region is zero by construction and is not a selection
candidate).

| `block_size` | n_blocks | exactly zero | **tied with another candidate** | worst layer |
|---|---|---|---|---|
| 64 | 256 | 0.0% | **17.1%** | 37.3% |
| **128** (this study) | 128 | 0.0% | **3.0%** | 10.0% |
| 256 | 64 | 0.0% | **0.4%** | 1.0% |
| 512 | 32 | 0.0% | **0.1%** | 0.4% |

**Nothing underflows in the interior rows, and the mechanism there is fp16
*mantissa* collisions rather than underflow:** `score_cache.save` stores fp16,
and in a row of 256 candidates the pooled probabilities average ~0.004, where
adjacent distinct values land on the same fp16 representable number. Where the
top-k boundary falls inside such a group, which member gets kept is decided by
the tie-break, not by the score.

> **CORRECTED 2026-09-20. This paragraph read "Nothing underflows. The
> exact-zero rate is 0.0% at every block size", and that is true only of the
> row it was measured on.** The table above was taken on a tensor whose final
> query block is full. It is not, for the **final query block of a prompt that
> does not land on a block boundary** — which is 95–100% of every banked
> accuracy row (`accuracy_forced_sink`: 300/300; `stage3_16384`: 100/100).
> `_scoring_forward_chunked` zero-pads that block's query rows to a full
> `block_size` before pooling, so its pooled means are divided by the padding
> and a large share fall below fp16's representable range. Measured across
> every banked oracle score tensor, candidates of the final query block that
> are **exactly 0.0**:
>
> | n_blocks | final block | exactly zero |
> |---|---|---|
> | 16 / 32 / 64 / 128 / 256 | full | **0.0%** |
> | 17 | 1–128 real rows | **15.0%** |
> | 65 | 1–128 real rows | **52.8%** |
> | 309 | 1–128 real rows | **7.7%** |
>
> So the exact-zero rate is 0.0% exactly where `seq_len` divides evenly and
> substantial everywhere else. In that row the top-k is decided almost
> entirely by the tie-break rather than by the score, which is what made
> instance 45 reach a nesting break instead of staying a curiosity. The
> mantissa-collision account above is still correct for the interior; it is
> not the whole story.

**The direction is the opposite of what coarse-block dilution would predict.**
Tie density falls roughly 6x per doubling of `block_size`. Coarser blocks pool
more probability mass into each cell, pushing values up into a region where
fp16 has resolution to spare; *finer* blocks spread the same mass thinner
until neighbouring blocks become indistinguishable in storage. So this is not
the sink-dilution mechanism recorded elsewhere in this file appearing in a new
place — it runs the other way, and the two should not be folded together.

**What it means for the block-size threat.** At the study's `block_size=128`
the affected fraction is 3.0% of candidates (10.0% in the worst layer), which
is small but not nothing. The interesting extrapolation is downward: a
`block_size=16` arm — the configuration this document already flags as the
sharpest threat to the headline result — would sit well above the 17.1%
measured at 64. Any future fine-block arm should therefore **re-measure this
before interpreting its accuracy**, because a meaningful share of its mask
would be selected arbitrarily rather than by importance.

**This is a storage decision, not a property of sparse attention.** Caching
scores in fp32 costs 2x disk (the grid's largest entry goes from 64 MiB to
128 MiB) and removes the effect entirely. Like the CPU mask-construction
finding, the honest phrasing is *as implemented*: the oracle is degraded by a
cache dtype chosen for space, not by anything intrinsic to importance-based
block selection.

**A claim made in conversation before this was measured was wrong**, and is
recorded here rather than quietly dropped: the tie structure was first
described as "~62% of pooled fp16 scores round to exact zero," which would
have made this a signal-loss story about coarse pooling. That figure came
from a synthetic softmax and counted the causally-masked upper triangle. The
real rate is 0.0%, the real mechanism is mantissa collision, and the real
block-size dependence points the other way.

### The sparsity arms rank from different precision within a run — measured, and *that* does not break nesting

> **Read the heading narrowly.** The precision split is not what broke
> nesting; a sparsity-dependent tie-break was (instance 45, corrected
> 2026-09-20, boxed below). The two were found in the same place and are
> different defects, and the measurements in this section remain correct
> for the question they asked.

Found by the 2026-09-19 single-measurement audit (item S5 -- see
[`docs/audit_register.md`](audit_register.md), which records that this
item cleared the question it asked and that the extrapolation made from
it did not survive); nothing had
recorded it for accuracy. `generation.generate_one` fetches scores per
(arm, example), and arms run dense → 0.5 → 0.75 → 0.9 (confirmed from row
order in `stage3_s1b`, `accuracy_forced_sink` and the 7B run). On a cold
cache the **0.5 arm ranks from the fp32 tensor** a miss returns, and **0.75
and 0.9 rank from the fp16 copy** `score_cache.save` wrote. Every oracle and
cheap accuracy run in this study has that structure. The timing side of the
same asymmetry (±3 ms of mask construction) is recorded under the
flashdecode diagnosis; this is the ranking side.

The threat was nesting: the ladder is a ladder because the 0.75 mask is a
subset of the 0.5 mask, built from one ranking. Two rankings do not
guarantee it. Measured on CPU with the real model and the real mask path
(`scripts/check_score_dtype_asymmetry.py`; output in
`results/diagnostics/20260919_score_dtype/`), Qwen2.5-1.5B, block 128, 28
layers, both the current forced-sink rule and the pre-fix rule the banked
2048–8192 accuracy used:

| band | examples | rule | fp32≠fp16 at 0.5 | at 0.75 | at 0.9 | **nesting breaks as run** |
|---|---|---|---|---|---|---|
| 2048 | 9 | pre-fix | 0.073% of active blocks | 0.017% | 0 | **0 / 252 layer-masks** |
| 2048 | 9 | forced sink | 0.086% | 0.014% | 0 | **0 / 252** |
| 4096 | 3 | pre-fix | 0.111% | 0.031% | 0 | **0 / 84** |
| 4096 | 3 | forced sink | 0.097% | 0.040% | 0.044% | **0 / 84** |
| 8192 | 3 | pre-fix | 0.175% | 0.122% | 0.045% | **0 / 84** |
| 8192 | 3 | forced sink | 0.194% | 0.120% | 0.022% | **0 / 84** |

"Nesting breaks as run" counts blocks in the fp16 0.75/0.9 mask absent from
the fp32 0.5 mask; the same count within a single ranking is the control and
is 0 by construction (it was). The fp16 reorderings are confined to near-tie
groups at a top-k boundary, and the 0.75 and 0.9 boundaries sit well inside
the 0.5 set, so no reordering reached across.

**What this licenses.** At 2048 and 4096 the accuracy-versus-sparsity curve is
a ladder as run, and the 0.5 arm's mask differs from what a single fp16
ranking would give by about one block in a thousand. That is far below
anything an n=100–300 accuracy cell resolves.

**What it does not.** It is measured, not proved: a tie group straddling the
0.5 and 0.75 boundaries at once in a short row could in principle break it.
**8192 was added 2026-09-20** (3 examples, both rules): the fp32/fp16
disagreement roughly doubles from 4096, to 0.18–0.19% of active blocks at
0.5 and up to 55 of 84 layer-masks, and nesting still holds in all of them.
So the disagreement grows with length, as the thinner probabilities predict,
and still stays confined to the boundaries. **16384 was not sampled**; its
fp16 tie density is 3.0% of candidates (the section above), and there a
structural argument was carrying the claim rather than a sample.

> **That structural argument was wrong, and the thing it was covering for was
> a real bug — see instance 45 (2026-09-20).** The argument assumed the three
> arms break a tie the same way, so that a tie group could reorder without
> reaching across a budget boundary. They did not: `importance_block_mask`
> drew its jitter *after* a `budget <= 0` early return, and `budget` depends
> on `sparsity`, so the arms ran off different positions in one shared
> generator. Where scores tied, each arm broke the tie differently and nesting
> failed — 70 of 112 layer-masks at `n_blocks=65` on banked oracle tensors.
> Fixed in `masks.py` by drawing unconditionally before any sparsity-dependent
> branch; the same sweep now reports **0 breaks in 30,968 layer-masks across
> 1,106 banked examples**, at every block count including 128 and 256.
>
> **The fix proposed in the next sentence would not have fixed it.** Returning
> the fp16 round-trip on a miss makes every arm rank from one *tensor*; the
> defect was that they did not share a *tie-break*. The reproduction above
> used a single fp16 tensor for all three sparsities and still broke.
>
> **What this costs the banked data.** Re-running the fixed builder over every
> banked score tensor changes **5.6% of (layer, sparsity) masks** and 0.23% of
> all active blocks, concentrated in the ragged final block — **0.0% where
> `seq_len` divides evenly**, 0.92% at `n_blocks=17`, 1.67% at 65. **Both
> figures are pools, and both understate where the change concentrates.** The
> 5.6% is flagged as such below; the 0.23% was not until 2026-09-21, and it is
> the same defect — the zero-rate configurations dominate the denominator, so
> the pooled fraction describes the bands where nothing moved. The third term
> was in the commit message of `5cc3a40` and was dropped when the figure was
> quoted here, which is precisely the term that makes the pooling visible.
> Untied rows are bit-identical, because
> a different 1e-9 perturbation cannot reorder distinct scores. So the banked
> accuracy rows are **not bit-reproducible under the current code**, and
> whether any score moved is **measured — S7, closed 2026-09-21**, on the one
> GPU band that matters, 16384, the headline band and the one with the most
> ragged final blocks (76 of 100 examples have ≤16 real tokens there): dense
> canary 200/200 identical, score canary 200/200 score tensors bit-identical,
> and `niah_multikey` moved **66/59/20 → 65/53/17** (−6.0 at 0.75, 95% CI
> [−12.0, −1.0], excludes zero), while `niah_single` held at 100.0 despite
> 14/2/2 of its texts changing underneath it. One published number moved.

The remaining per-arm precision asymmetry is untouched by that fix. The
remedy for it, if one is ever wanted, is to return the fp16 round-trip on a
miss too, so every arm ranks from one tensor. It was **deliberately not made
before audit item S1a**: S1a re-runs the pre-fix bands to isolate the sink,
and must reproduce the originals' per-arm precision (a cold cache) to do
that.

### Which mask era each banked accuracy file belongs to

**There are now three, and a replicate of any banked number will disagree
across an era boundary for a reason that is not drift.** Stating that eras
exist is not enough — a canary comparison that straddles one fails with no
indication why, and the next person spends a session finding out. So the
assignment is per file:

| era | mask rule | banked files |
|---|---|---|
| **1. pre-sink** | kv block 0 is an ordinary candidate; only the diagonal is free | `stage3_s1`, `stage3_s1b`, `stage3_16384`, `stage3_32768`, `stage3_flashdecode`, and the `gpu_session_2026090[678]*` copies of each |
| **2. forced-sink** | kv block 0 granted free (`37675a0`, 2026-09-16); jitter drawn **after** the budget check | `accuracy_forced_sink`, `accuracy_forced_sink_cheap`, `s7_7b_16384`, `s9_7b_cheap_16384`, `s1a/accuracy_band2048`, `s1a/accuracy_bands2`, `s1a/accuracy_all_bands`, `sink_control` (`importance_randfree`) |
| **3. post-jitter** | as era 2, plus jitter drawn unconditionally (instance 45, 2026-09-20) | `s7_jitter` — 16384, `niah_single` and `niah_multikey` only, n=100. **Supersedes the same cells of `accuracy_forced_sink`.** That file's `vt` cells at 16384, and every other band, remain era 2. |

**One file is era 3, and it is a partial band.** `s7_jitter` covers 16384 at
`niah_single` and `niah_multikey`; `vt` was deliberately excluded, because its
stopping confound would make a flip unattributable. So `accuracy_forced_sink`
at 16384 is now **split by task**: its `niah_*` cells are superseded and its
`vt` cells are the only measurement there is. Do not read that file as one
population — see "Which mask era each banked accuracy file belongs to" above.
(The composition rule in `analysis/composition.py` is a separate matter
and does not cover this: it balances facet levels within an aggregation
and cannot see a mask rule. The era check is `analysis/eras.py`, and it
reaches this file only through the comparison scripts, not through
`aggregate()`.)

Every other banked accuracy row is era 1 or era 2, so a bit-exact replicate of
any published number outside that one cell is still not obtainable from `HEAD`.

**A correction that breaks a comparison's internal consistency is worse than
a known, labelled staleness.** The instinct on finding era 3 is to bring the
nearest era-2 file up to date. Resist it where the file is one arm of a
multi-arm comparison. The published oracle-versus-estimator row compares four
arms — 1.5B oracle, 1.5B cheap, 7B oracle, 7B cheap — and all four being era 2
made it a *correct* statement about a mask rule the code no longer builds. S7
moved one of them. Re-running the 1.5B cheap arm to restore that pair would
put an era-3 gap beside an era-2 gap and make the scale comparison cross-era,
which `scripts/run_scale_comparison.py` refuses from `git_commit` via
`analysis/eras.py`. *(This clause read "which is precisely what `analysis/composition.py` refuses" until 2026-09-21. That was wrong: composition refuses on facet balance and has no concept of a commit, so nothing refused a cross-era comparison at all. `scripts/run_scale_comparison.py` and `scripts/run_scorer_comparison.py` now do, from `git_commit` via `analysis/eras.py`, unless `--cross-era era2:era3` is passed with a stated reason.)* **All arms of a
comparison move together or none of them do**, and "none, clearly labelled"
is a legitimate disposition — the era label is the fix. See
`audit_register.md`, item S12.

**How to tell without this table.** Both boundaries are commits and
`git_commit` decides both. Era 1 is any commit that is not a descendant of the
sink fix `37675a0`; era 3 is any commit that is a descendant of the jitter fix
`5cc3a40`; era 2 is what lies between. Each boundary commit belongs to the era
it opens. Verified against the real history on 2026-09-21: the partition is
exact for all thirteen registered commits, and `attnbench/analysis/eras.py`
derives it with `git merge-base --is-ancestor` rather than restating it.

> **This paragraph said "Era 2 and 3 split on date only — the jitter fix
> carries no schema change, so a row cannot be assigned between them from its
> own contents", and called that "a deliberate limitation being recorded
> rather than a gap". It was wrong, and wrong in the direction that made the
> project look less careful than it is.** `5cc3a40` is an ordinary commit, so
> the same mechanism this paragraph uses one sentence earlier for era 1/2
> settles era 2/3. The substitute it offered does not even work: `179c894`,
> `44ab65c` and `33598b4` are era 2 and share 2026-09-20 with `39e1d6d`, which
> is era 3, so a date rule at day granularity misfiles three of four. The
> claim was then copied verbatim into `analysis/eras.py` when that module was
> written on 2026-09-21 — one day after the guard against copying stale
> statements was built, and by the same hand. Nobody re-derived it from the
> commit graph until an audit was told to.

**Why there is still no `mask_rule` column.** Not because the information is
unavailable — it is, from `git_commit`. Because adding the column now would
stamp only future rows and leave every banked row unlabelled, which is the
asymmetry that makes a half-populated provenance field worse than none (see
`provenance.stamp_onto`). That argument stands on its own; the availability
argument never did.

**Size of the era-2 → era-3 difference. It is strongly length-dependent, and
the aggregate hides that.** Rebuilding every banked score tensor under both
rules:

| band | layer-masks changed | |
|---|---|---|
| 2048 | 1288 / 75 600 | 1.7% |
| 4096 | 487 / 8 400 | 5.8% |
| 8192 | 2969 / 8 400 | 35.3% |
| **16384** | 62 / 84 | **73.8%** |
| 32768 | 250 / 252 | **99.2%** |
| *all bands pooled* | *5175 / 92 904* | *5.6%* |

> **The pooled 5.6% is not a summary of this table; it is an artifact of its
> composition.** 81% of the banked score tensors are 2048-band, where 1.7% of
> masks move, so the pooled figure describes the short bands and says nothing
> about the long ones. It was the first number measured, and it was quoted
> here and in the commit message for instance 45 before the per-band split was
> computed. **Pooling across a differently-composed population is the failure
> this project has now been caught by three times** (pattern #13;
> `analysis/composition.py` exists because of the first two). Corrected 2026-09-20; the commit message
> of `5cc3a40` still carries the pooled figure and cannot be amended.

**Confirmed at 16384 on 200 real examples**, independent of the sample above:
the banked forced-sink run used a cold cache, so its 0.75 and 0.9 arms ranked
from the fp16 tensors that survive in GCS, and those masks are therefore
reproducible byte-for-byte. Rebuilt under both rules, `niah_single` and
`niah_multikey`, n=100 each:

| | 0.75 | 0.9 |
|---|---|---|
| layers changed per example, median of 28 | 25–27 | 13–19 |
| examples with **no** change | **0 / 200** | **0 / 200** |
| active blocks differing | 0.28–0.60% | 0.13–0.28% |

74.6% of layer-masks over the 200, against 73.8% from the single banked
tensor — the two agree, which is the check that the GCS sample is
representative.

**Two things follow.** The block-level perturbation stays small at every band;
what rises with length is how *many* masks contain one, because longer
contexts pool thinner probabilities into more candidates per row and so tie
more often in fp16. And **no example at 16384 can be excluded from a re-run**:
every one of 200 has at least one layer whose mask moved, so the fraction of
the cell that could flip is 100% and there is no cheap subset. Whether any
score actually moves is **measured — audit item S7, closed 2026-09-21**:
dense canary 200/200 identical, score canary 200/200 bit-identical, and
`niah_multikey` moved 66/59/20 → 65/53/17 (−6.0 at 0.75, 95% CI [−12.0, −1.0]).
One published number moved.

**Era 1 → era 2 is much larger** and is documented above: up to 44 points on
individual cells, which is why Stages 4/6/7 were rebuilt.

**What this means for the dense canary.** `scripts/check_dense_canary.py`
compares the dense arm, which builds no mask and is therefore **era-invariant**
— it is expected to reproduce 300/300 across all three eras, and did across
1→2. A canary failure is still a real signal. A *sparse* arm disagreeing across
an era boundary is not.

### Changing the sparse arms' decode kernel moves their text, not their scores

Measured 2026-09-19 from banked data, while designing audit item S1a. The
only pair of runs that differs in nothing but the sparse arms' decode kernel
is `niah_single` at 2048/4096/8192, n=100, all pre-sink-fix: `stage3_s1`/`s1b`
(sparse decode `sdpa_math`) against `stage3_flashdecode` (sparse decode
`sdpa_flash`). The dense arm decodes through `sdpa_flash` in both.

| | |
|---|---|
| dense predictions identical across the two runs | **300 / 300** |
| sparse predictions whose text differs | **80 / 900** (3–14% per cell) |
| — of which the answer span (text before the first period) differs | **5** |
| — of which correctness changed | **1** (2048/0.9: 63 → 62) |
| sparse-minus-dense delta identical | **8 of 9 cells**; 2048/0.9 moves −37 → −38 |

**What this establishes.** On `niah_single` the published sparse-minus-dense
deltas are decode-invariant to one example in 900. The text differences are
almost all *after* the answer — the free continuation inside the 14-token cap,
where a small logit difference early compounds over the remaining tokens.

**What it does not establish.**

- *That sparse arms are more sensitive to numerical path than dense ones.* It
  is tempting, and it is not tested. The dense arm's kernel never changed
  between these runs, so "dense did not move" is expected, not a comparison.
  A dense-arm decode switch was never measured.
- *Decode invariance on `niah_multikey` or `vt`.* No banked pair isolates the
  decode kernel on either. Both have longer outputs and scores well below
  ceiling, so there is more room for a kernel change to move correctness.
  This is the untested third item. **It feeds no current claim**: rows 67 and
  103 of `claims.md` came from `sdpa_math`-decoded runs and S1a replays that
  regime, and every end-to-end row at 2048–8192 is `niah_single`. It would
  matter the first time anyone quotes current-code (`sdpa_flash`-decoded)
  accuracy for those two tasks at those bands.

**Why S1a pins decode anyway.** The per-example comparison in S1a is exact,
not statistical, and a text-level change from the kernel on 3–14% of
examples would sit inside it. `--pin-fallback-decode-backend sdpa_math`
replays the regime, only values in `DENSE_DECODE_BACKEND_HISTORY` are
accepted, every row carries `decode_pinned=True`, and a pinned run refuses to
resume into an unpinned output. S1a's rows are therefore **sink-corrected
accuracy comparable to the banked numbers, not current-code accuracy**.

### `vt`'s sparse-above-dense gap is substantially a STOPPING effect at 1.5B, and not at 7B

Found 2026-09-19 from the `stop_reason` column, after S1a's band 2048 showed
the `vt` margin collapsing from +11.8 to +3.2 under a forced sink. Measured
by `scripts/run_vt_stopping_analysis.py` over every banked `vt` run, paired
within each run (`results/diagnostics/vt_stopping.parquet`).

`vt` is scored by recall over five variable names under a 40-token cap. An
arm that emits a bare name list and stops scores full; an arm that restates
the assignment chain (`VAR A = VAR B, ...`) runs into the cap before naming
all five and loses the rest. **That is a property of the output, not of the
attention** — and the arms did not stop alike. Before the sink fix, the
sparse arms hit the cap **31–183** times per 300 against dense's 214–279, and
generated 25.9–35.4 tokens against dense's 35.2–38.8.

*Corrected 2026-09-20. This read "63–158 caps" and "25.9–30.0 tokens", which
are the **2048 band's** min and max quoted as if they covered all nine
pre-fix cells. From `results/diagnostics/vt_stopping.parquet`, `sparse_cap`
across those cells is 31, 63, 70, 85, 103, 149, 158, 177, 183 and
`sparse_tok` is 25.9–35.4. Dense's 214–279 was right. The direction of the
argument is unaffected — the widest sparse value, 183, is still below the
narrowest dense one, 214 — but the true range is nearly three times wider,
and it hides that 8192/0.75 capped 183 times against dense's 279 rather than
stopping early in any strong sense.*

Paired sparse-minus-dense, all pairs / pairs that stopped the same way /
pairs where both hit the cap:

| run | 0.50 | 0.75 | 0.90 |
|---|---|---|---|
| 1.5B 16384 forced sink | +9.4 / **+2.4** / +3.8 | +12.8 / **+4.4** / +7.2 | +12.0 / +8.1 / +12.4 |
| 1.5B 16384 cheap | +6.2 / +1.5 / +3.9 | +7.4 / +3.6 / +6.3 | +3.4 / +9.4 / +14.3 |
| 7B 16384 forced sink | +0.6 / +0.4 / +0.5 | +3.2 / +2.7 / +2.9 | +6.0 / +6.8 / +7.1 |
| 7B 16384 cheap | −1.0 / −1.3 / −1.4 | +2.8 / +2.9 / +2.9 | +6.6 / +6.9 / +7.0 |
| 1.5B 2048 forced sink | −0.2 / −0.2 / 0.0 | +3.2 / +3.0 / +4.2 | −4.6 / −5.8 / −5.2 |

**At 1.5B most of the margin at 0.5 and 0.75 is stopping** — +9.4 becomes
+2.4, +12.8 becomes +4.4. **At 7B there is no stopping component**: all three
views agree within noise, and the cap counts are nearly equal (92 dense
against 80–85 sparse, versus 68 against 43–51 at 1.5B). That is the
prediction the smaller 7B `vt` gap already implied, confirmed on the
mechanism rather than the outcome.

**What does not go away.** The cap-cap gaps at 0.75 and 0.9 stay positive
with CIs excluding zero, on both models. So stopping is a large part of the
`vt` effect and not all of it.

**The adjustment is post-treatment, and that bounds what it can show.**
Sparsity changes stopping and stopping changes the score, so `stop_reason` is
a mediator, not a covariate. Conditioning on it estimates neither the total
effect nor a clean direct effect, and it can select in either direction: at
8192/0.9 pre-fix the gap is −0.9 over all pairs and **+18.3** among same-stop
pairs, a 19-point move from the conditioning alone. The clean experiment is a
scorer or cap that does not reward stopping early, which is a re-scoring
question, not a GPU one.

**This is the accuracy-side twin of confound 2.** Unequal generation length
between arms is already recorded here as a *latency* confound (`claims.md`,
"Confound 2 — unequal generation length"). It reaches the *scores* too, by
the same mechanism, and that was not recorded until now.

**A third explanation for `vt`, and the only one with a testable mechanism.**
The other two on record are the oracle acting as a denoiser (`claims.md`) and
block structure suiting multi-hop tracking (Sparse Frontier). Both are
inferences from the outcome; this one predicted where the effect would be
absent (7B) and was checked there.

### Forcing the sink reshuffles masks; it does not only repair them

From the S1a re-measurement (2026-09-20), across all 2,700 sparse rows at
2048/4096/8192, same examples, sink forced as the only change:

| | |
|---|---|
| examples flipped wrong → right | **299** |
| examples flipped right → wrong | **345** |
| cells with a net *negative* shift | 8192/0.9 `niah_multikey` (**−10**, 6 up / 16 down); 2048/0.75 `niah_single` (−1) |
| cells with a positive net that still flip both ways | e.g. 4096/0.9 `niah_multikey`, 23 up / 20 down |

(The raw flip counts are dominated by `vt`, where `correct` means a perfect
5-of-5 recall and the forced-sink arms shifted to a longer output format —
see the stopping section above. Mean scores there mostly rise while exact
correctness falls.)

**Why this is worth stating rather than smoothing over.** The natural reading
of the sink fix is "the mask was missing the block that matters, and now it
is not". A repair of that kind should help almost monotonically. What the
data shows is a reshuffle with a favourable bias: most cells improve, some
degrade, and examples move in both directions inside nearly every cell.

Two hypotheses fit that: the fix adds `sparsity` blocks per row (+53.8% of
active blocks at 2048/0.9) and an extra block changes what the model sees, or
the sink is genuinely important on average while occasionally displacing
something better.

**The control separating them was run on 2026-09-20** (`mask_source=
"importance_randfree"`, `scripts/sink_control_run.sh`, 25 min, ₹33). It grants
one *arbitrary* off-diagonal block per row instead of kv 0, with the budget
computed over the same candidate count, so per-row block counts are identical
— 40 per layer in both arms at 2048/0.9, against 26 pre-fix. Same examples,
decode pinned to `sdpa_math`, dense canary 300/300.

| task | dense | no free block | random free | sink free | **sink − random** [95% CI] |
|---|---|---|---|---|---|
| `niah_single` | 100.0 | 63.0 | 69.0 | 97.0 | **+28.0** [+19, +37] |
| `niah_multikey` | 98.0 | 0.0 | 9.0 | 44.0 | **+35.0** [+25, +45] |
| `vt` | 75.0 | 25.8 | 39.0 | 70.4 | **+31.4** [+24, +38] |

**The sink is doing real work.** Density accounts for 6–13 points of the
34–45-point gain; the sink's identity accounts for 28–35, with CIs far from
zero on all three tasks. So the earlier reading — that the non-monotonicity
pointed at "an extra block" rather than "the right block" — is **wrong as a
summary**: it is the right block, and granting it still costs a minority of
examples their previous answer. Both facts stand, and the reshuffle is the
smaller of the two effects.

This also makes the study's own sink-forcing rule an evidenced choice rather
than an imported convention: Sparse Frontier's Appendix A.1.1 forces the sink,
this study now has its own measurement of what that is worth, and the two
agree.
