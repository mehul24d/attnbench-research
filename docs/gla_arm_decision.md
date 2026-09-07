# Does GLA get an accuracy arm? The rule, written before the measurement

**Status: PRE-REGISTERED, 2026-09-07. Unmeasured.** The thresholds below were
fixed before the experiment ran, and the reason is that the expected result is
an ambiguous small number. A near-zero score is compatible with three very
different states of the world, and choosing between them after seeing the
number is not a decision, it is a rationalisation.

## What happened, in one paragraph

Stage 3 segment 1 ran GLA and got 0.0 on every task. The cause was that
`linear.py` synthesizes GLA's forget gate from `cfg.key()` — random
`logsigmoid(randn)`, mean −0.81 per step, a **1.24-token memory horizon**.
That is correct for Stage 2, where a kernel's throughput cannot depend on the
values in its gate, and fatal anywhere the output is read. See
`docs/silent_failure_patterns.md` #17.

**Qwen2.5 has no gate projection to borrow**, so there is no correct gate to
supply. This is a scope boundary, not a bug with a fix.

## The one experiment

Run GLA with `gate_source="ungated"` — `g = 0`, decay factor exactly 1,
nothing ever forgotten. That is a **real mechanism** (ungated linear
attention), just not the one Qwen's weights were trained for. It answers the
only question that decides the arm: *with the state retained in full, does the
output depend on the context at all?*

**Cost: ~5 minutes.** `niah_single`, `n=100`, `seq_len=2048`, one config. It
rides at the head of the next session rather than being booked for.

`niah_single` is the right task to decide on: its answer is a random 7-digit
number appearing exactly once in the context, so the probability of emitting
it without retrieving it is ~1e-7. **A non-zero score cannot happen by luck.**

## The positive control runs in the same process

The dense arm (`sdpa_flash`) is measured on the identical examples in the same
invocation, and must clear all three gates. If it does not, the harness is
broken and **the GLA result is not interpretable at all** — a failing control
is not evidence against GLA. This is the same discipline as
`test_the_determinism_check_resolves_one_ulp`: assert the mechanism, and
separately assert the mechanism is observable.

## The three gates, in order

Each is checked on the 100 `niah_single` predictions. The first failure stops
the evaluation and determines what we report.

| # | gate | threshold | dense arm gets |
|---|---|---|---|
| 1 | **input-dependence** — `nunique(predicted) / n` | **≥ 0.90** | 1.00 |
| 2 | **format validity** — fraction of predictions containing a 7+ digit run | **≥ 0.50** | ~1.00 |
| 3 | **retrieval** — mean RULER `niah_single` score | **≥ 20.0** | 100.0 |

### Why these numbers

**Gate 1 at 0.90.** The synthetic-gate failure scored 15/300 = 0.05. The dense
arm scores 1.00. There is no plausible mechanism that lands between 0.05 and
0.90, so the threshold sits in empty space and is not a tuning knob. Failing
it means the context is still not reaching the computation.

**Gate 2 at 0.50.** This is what separates *bad at the task* from *not doing
the task*. A model that has retained the context but cannot retrieve from it
still emits an answer-shaped thing — a wrong 7-digit number. Token salad does
not. Half is deliberately lenient: the question is whether answer-shaped
output happens at all, not how often.

**Gate 3 at 20.0.** The load-bearing one, and it is about **cost**, not
significance. Chance is ~1e-7, so even 5/100 would be overwhelming evidence of
real retrieval. But 2048 is the *easiest* band, and NIAH retrieval falls with
context length — an arm scoring below 20 here will score ~0 at 4096 through
32768, contributing one constant number at roughly 0.35 h per band. Twenty is
the point below which the remaining grid buys nothing.

## The decision, fully determined in advance

| outcome | verdict | what the study reports |
|---|---|---|
| all three pass | **KEEP the arm** | GLA accuracy across the grid, captioned: ungated, and run on weights trained for softmax attention. |
| 1 and 2 pass, 3 fails | **DROP the arm** | A documented negative: *the substitution is coherent — output is input-dependent and answer-shaped — and retrieves nothing.* This is a real result about mechanism substitution, and it is **not** a claim about linear attention in general, because the weights were never trained for it. Reported in `limitations.md`, not as a headline. |
| 1 or 2 fails | **DROP the arm** | GLA is **timing-only**. Stage 2's kernel results stand; there is no accuracy comparison to make, and none is reported. |
| dense control fails any gate | **NO VERDICT** | The harness is broken. Fix it and re-run; do not interpret the GLA numbers. |

In every DROP case, `configs_by_backend` loses its `gla` entry for Stage 3 and
the remaining segments shrink by roughly 0.35 h per band.

## What this rule refuses to allow

Reporting a near-zero GLA score as *"linear attention degrades on long-context
retrieval."* That claim needs weights trained for linear attention. What this
study can say is narrower and true: *substituting GLA into a softmax-trained
model at inference time does not preserve retrieval* — which is a statement
about substitution, and is the honest form of the finding.

That sentence is not left here to be found. It is the first entry in
**`docs/claims.md`**, the ledger the write-up is drafted from: each supported
sentence paired with the paraphrase one qualifier away that the data does not
license. A caveat in a limitations file survives until someone compresses the
study into a paragraph; a sentence written once and copied rather than
rephrased survives that.
