# Stage 3 segmentation plan

**14.85 h does not fit in one supervised sitting**, and the standing rule is
that no instance is left running at the end of a turn. So Stage 3 runs in
three sessions. This is the plan for how, written before the first booking
rather than improvised at the second.

Compute cost per band, from `scripts/reestimate_stage3.py` (all inputs
measured, none assumed):

| seq_len | n | scoring h | measured h | total h |
|---|---|---|---|---|
| 2048 | 300 | 0.28 | 0.17 | 0.45 |
| 4096 | 300 | 0.59 | 0.36 | 0.95 |
| 8192 | 300 | 1.31 | 0.77 | 2.09 |
| 16384 | 300 | 3.03 | 1.74 | 4.77 |
| 32768 | 100 | 3.78 | 1.32 | 5.10 |
| **total** | | **8.99** | **4.37** | **13.35** |

Scoring is 67% of the cost. That matters for what has to survive between
segments — see "What must be carried" below.

## The three segments

Cut on band boundaries. Bands are independent, the score cache is keyed per
`(model, task, example_id, seq_len)`, and a band-aligned segment end leaves no
partially-scored band to reconstruct.

| | bands | compute | + overhead | wall | ~₹ |
|---|---|---|---|---|---|
| **S1** | 2048, 4096, 8192 | 3.49 h | 1.5 h | **4.99 h** | 400 |
| **S2** | 16384 | 4.77 h | 1.5 h | **6.27 h** | 500 |
| **S3** | 32768 | 5.10 h | 1.5 h | **6.60 h** | 530 |
| | | 13.35 h | 4.5 h | **17.86 h** | **1,430** |

**The segmentation tax is 3.0 h / ₹240** against a single 14.85 h session —
two extra rounds of instance creation, image boot, source deploy, pre-flight
suite and Stage 0/1 gates. That is the price of every session ending with
nothing running, and it is worth paying.

**Ascending order, deliberately.** The cheap bands bank first, so an
interruption costs the expensive tail rather than the whole run. And 32768 is
the `directional` point (n=100, ~66% power at epsilon=5, flagged as such in
`stage3_grid.yaml`) — the least valuable band, sitting last, where it is the
natural thing to drop if the calendar tightens before 2026-12-01.

**Not two segments.** The alternative — splitting mid-16384 into 7.4 h and
9.0 h halves — saves 1.5 h and produces two sittings longer than any run this
project has supervised. It also puts a cut inside a band, which is the one
place the score cache has to survive to avoid repaying 3.03 h of scoring.

## What must be carried between segments

**`accuracy.parquet` — hard requirement.** `runner.load_done_keys` reads it to
skip completed cells. If it is not on the next instance before the run starts,
S2 re-runs all of S1. Sync it down before every teardown and back up before
every start.

**The score cache — 4.09 GiB, insurance.** Sizes computed from the pinned grid
at `finest_block_size=128` with Qwen2.5-1.5B's 28 layers / 2 KV heads, fp16:

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
6. Run the segment's bands.
7. Sync `accuracy.parquet` and the score cache down **before** teardown.
8. Tear down. Re-run `gcp_cleanup_check.sh`.
9. Record minutes and estimated spend in `docs/spend_ledger.md`.

## Budget

₹1,430 of ~₹26,600 remaining, expiring **2026-12-01**. Money is not the
constraint; three supervised 5–7 h windows before that date is.
