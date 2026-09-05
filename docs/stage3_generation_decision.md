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

**Under 1% on the measured half. The 13.35 h estimate survives intact** — which
is the whole argument for building the cache rather than working around it.

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
