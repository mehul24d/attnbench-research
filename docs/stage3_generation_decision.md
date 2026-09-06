# Stage 3 generation: stopping rule, decode mode, and what it costs

**Status: proposed, awaiting confirmation. Nothing is built against it yet.**

Stage 3 scores *text* — `ruler.score(task, predicted, expected)` compares
strings — and this repository has never generated any. There is no `.generate(`,
no `decode(`, no `max_new_tokens` anywhere in it. So the generation contract has
to be decided before the KV cache is built, because the cache is built against
it.

## 1. Answer lengths, measured

200 examples per task, 40 from each of the five bands, tokenised with the real
`Qwen/Qwen2.5-1.5B-Instruct` tokenizer:

| task | n | min | median | p95 | max |
|---|---|---|---|---|---|
| `niah_single` | 200 | 7 | 7 | 7 | **7** |
| `niah_multikey` | 200 | 27 | 32 | 35 | **36** |
| `vt` | 200 | 12 | 16 | 18 | **20** |

**Answer length does not vary with `seq_len`** for any task — `niah_single` is
exactly 7 tokens in all 200 samples (a fixed 7-digit number), and the other two
vary only through BPE splitting of the same fixed-shape payload. So one cap per
task is correct, and it does not need a per-band table.

This supersedes the three single-example figures quoted earlier (7 / 33 / 16),
which were one sample each.

## 2. Per-task caps

Proposed: **2× the observed maximum**, per task.

| task | max observed | cap |
|---|---|---|
| `niah_single` | 7 | **14** |
| `niah_multikey` | 36 | **72** |
| `vt` | 20 | **40** |

Two adjustments to the caps sketched in conversation (16 / 66 / 32), which came
off the single-sample figures: `niah_multikey`'s real max is 36, not 33, so 2×
is 72; and `vt`'s max is 20, so 32 is only 1.6× — 40 keeps the rule uniform.

**The risk is one-sided, which is why a generous cap is cheap.** RULER's NIAH
metric is substring match, so a model that emits the right answer and then
rambles still scores 1.0 — over-generating costs decode steps and nothing else.
Under-capping produces a *false negative*: a correct answer scored wrong. A flat
32-token budget would truncate `niah_multikey` on essentially every example
(median 32, max 36) and report it as an accuracy collapse.

## 3. EOS will almost never fire, and that changes the stopping rule

This is the part that was not visible from the plan.

The prompts are **completion-style, not chat**. They end mid-sentence:

```
...What is the special magic number for 04b5665a-... mentioned in the
provided text? The special magic number for 04b5665a-... is
```

No chat template is applied anywhere in the repo, and there is no assistant
turn. Qwen2.5-1.5B-Instruct's EOS is `<|im_end|>`, emitted at the end of an
assistant turn — so on this prompt shape it will essentially never be produced.

**Consequence:** an "EOS or cap" rule is in practice a "cap" rule. Every example
would hit its cap, truncation rate would read ~100% for every backend, and the
diagnostic that was supposed to distinguish backends would carry no signal at
all. That is a metric that looks like coverage and is not — the shape this
project keeps finding.

**Proposal: stop on EOS *or newline* or cap.** All three task templates expect a
single-line answer, and RULER's own inference harness uses stop-words for
exactly this reason (its `pred/` directory was not vendored — only the
generators and the scorer were — so there is no in-repo authority to copy, and
the choice has to be recorded here instead of inherited).

With a newline stop the expected step counts are answer length + 1 — **8, 33, 17
tokens** — and the cap becomes a genuine backstop rather than the normal exit.
Truncation then means something: a backend that hits its cap is one that failed
to terminate, which is a real difference between backends and worth recording.

**Not proposed: applying a chat template.** It would change the prompt every
published RULER number was produced against, on top of scoring that is already
more permissive than theirs. Two deviations compound in a way one does not.

**Flagged, not fixed:** `vt`'s prompt contains a literal `[/INST]`, Mistral /
Llama-2 chat markup vendored verbatim from RULER, being fed to a Qwen model
whose markup is `<|im_start|>`. RULER templates per model and we do not. It
affects all backends identically so it cannot produce a spurious *difference*
between them, which is why it is recorded rather than silently patched — but it
does mean `vt` absolute scores are not comparable to published ones.

## 4. Greedy, and enforced

Deterministic decode: `do_sample=False`, no temperature, no top-p. Sampling
noise would swamp the effect being measured — whether sparse attention changes
the answer — and would make a re-run disagree with itself.

Enforced rather than intended: a test asserts the same (example, backend, cfg)
decodes to byte-identical text twice, the same discipline
`test_masks_determinism.py` applies to mask construction.

## 5. What travels on the row

`AccuracyResult` gains two fields, for the same reason `check_kind` and
`score_source` exist — the standard a verdict was produced under travels with
the verdict:

- **`stop_reason`**: `"eos" | "newline" | "cap" | "error"`. A closed set, typed
  like `MaskKind`.
- **`n_generated`**: tokens actually produced.

Truncation rate per backend is then a query, not a reconstruction. If sparse
backends hit their caps more often than dense, **that is a finding** — a real
failure-to-terminate difference — and not a scoring artefact. Without these
fields the two are indistinguishable after the fact.

## 6. What it costs

Expected decode steps per example, with the newline stop: 8 / 33 / 17, mean
**19.3** across the three tasks.

**Without a KV cache** — the current state, where `SwappedAttention` accepts
`past_key_values` and ignores it — every step is a full forward over the whole
context, so the measured phase multiplies by ~20.3:

| | measured | scoring | total |
|---|---|---|---|
| as estimated (1 forward/example) | 4.37 h | 8.99 h | **13.35 h** |
| no KV cache | ~88.7 h | 8.99 h | **~97.7 h** |

**With a KV cache**, a decode step attends against cached keys and runs its
projections and MLP for one position instead of `seq_len` of them, so `k` steps
cost roughly `k / seq_len` of a prefill:

| band | steps / seq_len | decode tax |
|---|---|---|
| 2048 | 19.3 / 2048 | 0.94% |
| 4096 | 19.3 / 4096 | 0.47% |
| 8192 | 19.3 / 8192 | 0.24% |

**Under 1% on the measured half. The 13.35 h estimate survives intact.**

### CORRECTION, 2026-09-06: that table is wrong, and by ~20x

The FLOPs ratio above is arithmetically correct and the conclusion drawn from
it is not, because `k / seq_len` is only a *time* ratio if decode and prefill
run at the same throughput. **They do not, and they cannot.**

A batch-1 decode step reads all **3.09 GB** of Qwen2.5-1.5B's bf16 weights to
produce one token. On an L4 (~300 GB/s, ~80% achieved) that is a **~13 ms
floor** — before any attention over the cache, before any Python dispatch.
Priced at the prefill's measured 42.2 TFLOPS, the same step's 3.09 GFLOP
"takes" **0.074 ms**.

**177×.** Prefill at 8192 tokens does ~8192 FLOPs per byte of weight read and
is compute-bound. A batch-1 decode step does 2, and is bandwidth-bound. One
throughput number cannot describe both regimes.

This is the same shape as two errors already recorded in this project —
attention-kernel TFLOPS read as whole-model TFLOPS (a 42% phantom speedup),
and GLA's 1.7 "TFLOPS" against FA2's 61.8. Every time, the unit matched and
the regime did not.

**The corrected cost**, from `scripts/reestimate_stage3.py`, modelled from
bandwidth rather than FLOPs:

| band | prefill h | decode h (floor) | tax |
|---|---|---|---|
| 2048 | 0.45 | 0.32 | **+70%** |
| 4096 | 0.95 | 0.32 | +34% |
| 8192 | 2.09 | 0.33 | +16% |
| 16384 | 4.77 | 0.36 | +7.5% |
| 32768 | 5.10 | 0.14 | +2.7% |
| **total** | **13.35** | **1.47** | **+11%** |

**The tax is largest where prefill is cheapest, and that is structural.** The
weight-read term does not depend on context, so decode costs about the same
per row at 2048 as at 32768 while prefill grows quadratically. The 2048 band
pays more to generate ~58 tokens per row than to attend 2048 of them.

**The answer is a bracket, not a number**, because this implementation's
per-step dispatch cost is unknown — an eager HF forward per step, a custom
attention module per layer, no CUDA graphs:

| scenario | grid decode | total | S1 wall |
|---|---|---|---|
| expected stops, bandwidth floor | 1.47 h | 14.82 h | 5.96 h |
| expected stops, +15 ms/step | 3.04 h | 16.39 h | 7.05 h |
| every example to its cap, floor | 3.19 h | 16.54 h | 7.10 h |
| every example to its cap, +15 ms/step | 6.60 h | 19.96 h | 9.46 h |

**What closes it: one measured decode step**, which
`scripts/time_one_accuracy_example.py` now takes before the grid commits. Two
minutes on the instance collapses a 5-hour range.

**What it changes about the plan:** S1 as booked was 4.99 h wall against a
five-hour window. It is now 5.96 h at best and 9.46 h at worst. The
segmentation is redone in `docs/stage3_segmentation.md`.

## 7. What the cache buys beyond cost

`backends/base.py` already defines `KVCacheState` and a `decode_step` interface
that nothing currently calls. Wiring it makes Stage 3 measure something Stage 2
structurally cannot: **GLA's decode state is fixed-size and recurrent while the
dense backends' cache grows with context.** That is the bounded-versus-unbounded
memory comparison, measured on a real model at real context lengths, and it is
invisible to a kernel benchmark that times a single forward pass.

## 8. Open, and deliberately not decided here

Whether the *first* generated token alone would suffice for `niah_single` (its
answer is a fixed 7-digit number, and the first token is highly diagnostic).
It would cut that task's decode cost by ~8×, and it would also change what is
being scored from "produces the answer" to "starts to produce the answer".
Not worth the ambiguity at a <1% decode tax; revisit only if the cache turns
out harder than expected.
