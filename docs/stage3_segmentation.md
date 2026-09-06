# Stage 3 segmentation plan

**REVISED 2026-09-06.** The first version of this plan cut three segments
against a 13.35 h compute estimate. That estimate priced one forward pass per
row; a Stage 3 row is a prefill plus greedy decode, and decode is
**bandwidth-bound where prefill is compute-bound**, so it cannot be costed
from the prefill's TFLOPS. See the correction in
`docs/stage3_generation_decision.md` §6. The bands below are re-priced and
the segments re-cut.

**14.85 h did not fit one sitting and the corrected figure fits less.** The
standing rule is that no instance is left running at the end of a turn, so
Stage 3 runs in several sessions. This is the plan for how, written before
the first booking rather than improvised at the second.

## Cost per band

Prefill from `scripts/reestimate_stage3.py` (all inputs measured on an L4,
session 4, commit b6ed63b). Decode modelled from memory bandwidth, and
reported as a **bracket** because this implementation's per-step dispatch
cost is not yet measured — an eager HF forward per step, a custom attention
module per layer, no CUDA graphs.

| seq_len | n | scoring h | prefill measured h | decode h (floor → worst) | band total |
|---|---|---|---|---|---|
| 2048 | 300 | 0.28 | 0.17 | 0.32 → 1.48 | **0.77 → 1.93** |
| 4096 | 300 | 0.59 | 0.36 | 0.32 → 1.49 | **1.27 → 2.44** |
| 8192 | 300 | 1.31 | 0.77 | 0.33 → 1.51 | **2.42 → 3.60** |
| 16384 | 300 | 3.03 | 1.74 | 0.36 → 1.57 | **5.13 → 6.34** |
| 32768 | 100 | 3.78 | 1.32 | 0.14 → 0.56 | **5.24 → 5.66** |
| **total** | | **8.99** | **4.37** | **1.47 → 6.60** | **14.82 → 19.96** |

The floor assumes every example stops at a newline where the measured answer
lengths say it will, and that decode achieves 80% of the L4's 300 GB/s. The
worst case assumes every example runs to its per-task cap and pays 15 ms/step
of dispatch on top of the bandwidth floor.

**Decode cost is nearly flat per row**, because the 3.09 GB weight read
dominates and does not depend on context. So the tax lands hardest on the
cheapest bands: **+70% at 2048, +2.7% at 32768.**

## The segments

Cut on band boundaries. Bands are independent, the score cache is keyed per
`(model, task, example_id, seq_len)`, and a band-aligned segment end leaves no
partially-scored band to reconstruct.

| | bands | compute (floor → worst) | + overhead | wall | ~₹ |
|---|---|---|---|---|---|
| **S1** | 2048, 4096 | 2.04 → 4.37 h | 1.5 h | **3.54 → 5.87 h** | 280–470 |
| **S2** | 8192 | 2.42 → 3.60 h | 1.5 h | **3.92 → 5.10 h** | 310–410 |
| **S3** | 16384 | 5.13 → 6.34 h | 1.5 h | **6.63 → 7.84 h** | 530–630 |
| **S4** | 32768 | 5.24 → 5.66 h | 1.5 h | **6.74 → 7.16 h** | 540–570 |
| | | 14.82 → 19.96 h | 6.0 h | **20.82 → 25.96 h** | **1,670–2,080** |

Four segments, not three: the previous S1 (2048+4096+8192) is now 5.96–9.46 h
wall and no longer fits a five-hour window under any scenario worth booking
against.

**₹1,670–2,080 of ~₹26,600 remaining, expiring 2026-12-01.** The extra segment
costs ₹120 in overhead. Money is not the constraint; four supervised windows
before that date is.

## S1 has a decision point in it, deliberately

**The first thing S1 runs after the gates is
`scripts/time_one_accuracy_example.py`, which now measures a decode step.**
That single measurement collapses a five-hour range in the grid estimate,
and it takes about two minutes.

- **Measured step ≈ the 13 ms floor** → the floor column holds. S1 finishes
  2048 and 4096 in ~3.5 h wall, and S2–S4 are planned against real numbers.
- **Measured step ≫ the floor** → the worst column is closer. Run 2048 only
  (~2.6 h wall including overhead), bank it, and re-cut the remaining bands
  before booking again.

This is a go/no-go on 4096 taken 15 minutes into the session on a
measurement, not a gamble taken at hour four. It is the whole reason S1 is
the small segment rather than the big one.

**Ascending order, still.** The cheap bands bank first, so an interruption
costs the expensive tail rather than the whole run. And 32768 is the
`directional` point (n=100, ~66% power at epsilon=5, flagged as such in
`stage3_grid.yaml`) — the least valuable band, sitting last, where it is the
natural thing to drop if the calendar tightens.

**Not a mid-band cut.** 16384 at 6.63–7.84 h wall is the longest sitting here
and splitting it would help, but it puts a cut inside a band, which is the one
place the score cache has to survive to avoid repaying 3.03 h of scoring.
Revisit only if the measured decode step lands in the worst column.

## An open prediction, to be tested at S3

**`sdpa_math` is projected to OOM at 16384 on a 23 GiB L4. That is arithmetic,
not a measurement, and it is written here so it gets checked rather than
assumed.**

Measured on 2026-09-06 at 8192: `sdpa_math` peaked at **10.17 GiB**, because
the math kernel materialises the S x S attention matrix. The term is
quadratic, so 16384 projects to ~40 GiB and 32768 to ~160 GiB.

It is *not* the reason the dense arm is `sdpa_flash`. That rests on two
measured/structural arguments which stand without it: the grid's hour estimate
was anchored on a `sdpa_flash` throughput figure, and a dense baseline that
changed kernel partway through the grid would split the accuracy data into two
differently-composed halves.

Why it is a prediction and not a fact: **this project has been wrong twice
about exactly this kind of memory arithmetic** -- the oracle length cutoff,
which was off by 16x at batch 16, and the device-aware budget, which was
verified only against a simulated card. A quadratic projection from one
measured point is the same shape as both.

**Test at S3, on hardware that is already up:** one `sdpa_math` prefill at
16384 with `torch.cuda.max_memory_allocated()`. It costs seconds. Record the
answer here either way -- a confirmed OOM is worth knowing, and so is a
projection that was 4x pessimistic.

## What must be carried between segments

**`accuracy.parquet` — hard requirement.** `runner.load_done_keys` reads it to
skip completed cells. If it is not on the next instance before the run starts,
S2 re-runs all of S1. Sync it down before every teardown and back up before
every start.

**The score cache — 4.09 GiB, insurance.** Re-verified 2026-09-06 against the
grid as it stands today, and no longer a table anyone has to trust:
`score_cache.cache_bytes_for_grid` computes it, and
`test_score_cache_still_fits_the_disk_the_plan_budgeted` asserts the total, so
a grid edit that outgrows the disk fails on CPU rather than at hour four.
Nothing about decode changes it — the cache holds prefill importance scores.
Sizes at `finest_block_size=128` with Qwen2.5-1.5B's 28 layers / 2 KV heads,
fp16:

| seq_len | n_blocks | MiB/example | examples | GiB |
|---|---|---|---|---|
| 2048 | 16 | 0.027 | 900 | 0.02 |
| 4096 | 32 | 0.109 | 900 | 0.10 |
| 8192 | 64 | 0.438 | 900 | 0.38 |
| 16384 | 128 | 1.750 | 900 | 1.54 |
| 32768 | 256 | 7.000 | 300 | 2.05 |

Band-aligned segments consume each band's cache within the segment that
builds it, so this is not needed for ordinary progress. It is needed if a
segment dies *mid-band*: without it, restarting 16384 repays 3.03 h of
scoring; with it, minutes. 4 GiB is cheap for that.

**The commit — now enforced.** `runner.check_code_continuity` refuses to
resume into a checkpoint written at a different commit, or from/into a dirty
tree. An accuracy score does not depend on which machine produced it — which
is why `load_done_keys` correctly ignores host — but it depends entirely on
the code that produced it: the attention swap in `accuracy/model.py`, the
example generation and scoring in `accuracy/ruler.py`, the mask, the kernel.

Three sessions across several days is exactly the window in which a repository
changes. Stage 2's equivalent hazard was found only *after* four segments had
been measured, and cost a per-backend AST analysis to disentangle
(`analysis.code_identity`). Pin the commit at S1 and check it out for S2 and
S3; the guard fails loudly if that is forgotten.

Whole-commit equality here, not Stage 2's per-backend fingerprinting. There
the question was "may these independently-motivated segments be joined?", and
a blanket answer was wrong because it was yes for seven backends and no for
two. Here the segments are one experiment split for scheduling, so any change
to the code that produced a row makes the remainder a different run.

## Per-session checklist

1. `bash scripts/gcp_cleanup_check.sh` — confirm nothing is already running.
2. Create on-demand `g2-standard-8` + L4 in `asia-south1` from
   `attnbench-env-v5-20260905`. **Not Spot**: at 5–7 h per segment one
   preemption costs more than the discount saves.
3. Deploy source by moving history to the pinned commit (never by copying a
   source tree — see `docs/silent_failure_patterns.md` instance 10).
4. Upload `accuracy.parquet` and the score cache from the previous segment.
5. Run the test suite as a precondition, then the Stage 0/1 gates.
6. **Run `scripts/time_one_accuracy_example.py`** — it measures a decode step
   against the bandwidth floor and reports the grid's decode hours from that
   measurement instead of the bracket. On S1 this is the go/no-go above; on
   later segments it confirms the number the segment was booked against.
7. Run the segment's bands.
8. Sync `accuracy.parquet` and the score cache down **before** teardown.
9. Tear down. Re-run `gcp_cleanup_check.sh`.
10. Record minutes and estimated spend in `docs/spend_ledger.md`.

## Budget

See the segment table above: ₹1,670–2,080 of ~₹26,600 remaining, expiring
**2026-12-01**. Money is not the constraint; four supervised windows before
that date is.
