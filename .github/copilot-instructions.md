# Review instructions

For repository reviews involving numerical results, derived metrics, gates,
provenance, benchmarks, or analysis, apply `docs/copilot_review_plan.md`.

Use the plan's categories 1-17 to identify relevant integrity risks. Review
findings should be grounded in code and line references, with particular
attention to silent failures that could change a number reported in
`docs/claims.md` without downstream detection. Treat documented limitations as
boundaries to verify, not as bugs to rediscover.
