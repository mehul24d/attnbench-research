# H100 flex-start attempts (FA3 correctness gate, then P4)

One row per attempt, granted or not, appended in order and never edited
(plan Section 8 calendar, 31 Oct 2026: "every attempt logged"). The times
are the launcher's output, in UTC.

| # | Submitted (UTC) | Zone | Outcome | Waited (s) | Instance | Session / note |
|---|---|---|---|---|---|---|
| 1 | 2026-10-31T12:37:43Z | us-central1-a | Not granted: the insert ended `ZONE_RESOURCE_POOL_EXHAUSTED` at 12:45:11Z. The VM showed `STOPPING` at about 12:41Z, which the launcher reported as "could not be followed (failed)", not as no capacity; it cancelled, confirmed the instance gone, and tried no other zone. Never RUNNING, nothing billed (cleanup check clean) | about 450 | `attnbench-h100-gate-20261031-1807` | FA3 gate (`--gate-only`), commit `f3ceb1b`, cap Rs 500; launched on the researcher's yes of 31 Oct 2026 |
