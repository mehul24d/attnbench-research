# Invalid rows in this results set

**900 rows with `backend == "gla"` are INVALID and must be excluded from
every analysis.** They are retained rather than deleted so the exclusion is
auditable and the count is checkable.

Reason: GLA's forget gate was synthesized from `cfg.key()` (random
`logsigmoid(randn)`, mean -0.81/step), giving a memory horizon of 1.24 tokens.
The recurrent state had forgotten the prompt long before the question. The
rows carry fluent text, real latencies and valid provenance, and score 0.0.

Evidence: 15 distinct predictions across 300 distinct niah_single prompts
(the dense arm produced 300 of 300). The first generated token — from the
prefill logits, before any decode step — was already near-constant.

See docs/silent_failure_patterns.md #17. `GatedLinearAttention` now refuses to
run without a `gate_source` named explicitly.

## The exclusion is decided, not provisional (2026-09-07)

When this file was written it was open whether a corrected gate could
regenerate these rows. The pre-registered rule closed it:
`results/stage3_s1b/gla_arm_verdict.json` — **DROP, failed gate 1**. Run
ungated (`g = 0`, decay factor exactly 1, nothing forgotten), GLA produced
**1 distinct prediction across 100 distinct contexts** against 100/100 for the
dense control in the same process. Full retention did not help, so forgetting
was not the cause and no gate setting regenerates these rows.

Stage 3 therefore has **no linear arm**. GLA remains a Stage 2 timing result —
a kernel's throughput does not depend on the values in its gate — and these
900 rows stay excluded permanently rather than pending a re-run.
See docs/gla_arm_decision.md and docs/claims.md.

Valid rows in this file: 3600 (['block_sparse', 'sdpa_flash']).
