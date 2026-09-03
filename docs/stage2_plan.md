# Stage 2 — multi-session kernel sweep on L4

Stage 2 is the study's core timing deliverable and is **unblocked now**: the
dense, linear and flex-sparse arms all run without Block-Sparse-Attention.

---

## BLOCKER — the repository has no commits

**Stage 2 must not start until this is fixed.** The git repository enclosing
this project is rooted at the user's **home directory**, and its `main` branch
has zero commits. `git rev-parse HEAD` fails there, but git echoes the
unresolved argument to *stdout* before writing its error to stderr, so
`provenance` captured the literal string `"HEAD"` and stored it as a commit.

Every row written on 2026-09-03 therefore carries `git_commit="HEAD"` and
`git_dirty=True` (the latter from thousands of unrelated files in `$HOME`).

`provenance._git_state` now validates the 40-hex SHA format and anchors git to
this package's directory, so it records `None` rather than a plausible
non-answer. But **None is still not a commit**, and every integrity check
below depends on having a real one:

```bash
cd ~/Desktop/research/attnbench_scaffold
git init                       # its own repo, not $HOME's
git add -A && git commit -m "attnbench: baseline before Stage 2"
python3 -c "from attnbench import provenance; print(provenance.capture().git_commit)"
# must print a 40-hex SHA, not None and not HEAD
```

Until that prints a SHA, `cross_arch.load_segments` refuses to join segments
and `load_stage1_pass_set(at_commit=...)` cannot verify anything. Both have
explicit escape hatches (`allow_unverified=True`, `at_commit=None`) which
exist so a run *can* proceed — but a Stage 2 run using them produces results
whose provenance cannot be checked afterwards, which for a multi-session sweep
is most of the value.

---

## Scope and sizing

| | |
|---|---|
| Cells runnable now (dense + linear + flex) | **1008** |
| Iterations per cell | 40 (`WARMUP=10` + `REPS=30`) |
| Total attention FLOPs | ~860 PFLOP |
| Estimated wall time | **~6 h at 40 TFLOPS effective; bracket 4–12 h** |
| Cells at 32768 | 168 |
| BSA adds later | 288 cells, all block_size=128 |

Per-backend: `flex` 504, then `fa2` / `gla` / `naive` / `sdpa_{cudnn,
efficient, flash, math}` at 72 each. Flex is half the sweep because it is the
only backend covering block_size=64 as well as 128.

This does not fit one session. Plan for **three sessions of ~2 h**, or two
longer ones.

---

## Ordering: shortest sequence length first

Run 1024 → 2048 → 4096 → 8192 → 16384 → 32768.

Cost scales as O(S²), so the six lengths are wildly unequal: the 32768 cells
alone are most of the sweep. Shortest-first banks the largest number of
completed cells per hour, so an interrupted session loses the least — and two
instances have already become unreachable mid-work in this project.

This ordering is only safe because **session 4's 32K FA2 dense baseline
answers the memory question in advance**. Without it, leaving the longest
cells to last would mean discovering a 32K memory wall at the end of the third
session. Do not reorder to shortest-first if that baseline has not been run.

Within a length, `shuffled()` randomises visiting order so thermal drift
cannot correlate with backend identity. That is existing behaviour and must
not be disabled to make the ordering "tidier".

Expect `sdpa_math` and `naive` to OOM at 32768 — they materialise the full
O(S²) score matrix (69.66 GiB requested, observed in the first validation
session). Those are **recorded results**, not failures.

---

## Segment files: one per session, append-only

Each session writes its **own** results file:

```
results/stage2/segment_<YYYYMMDD>_<host>.parquet
```

`check_host_continuity` refuses to resume into a checkpoint written on a
different host, which is correct — every rented machine is a new host, and a
latency ratio spanning two machines is invalid. Segments are joined only at
analysis time by `cross_arch.load_segments`.

**Rules, all enforced in code:**

1. **Append-only. Never rewrite a segment.** If a cell needs re-measuring it
   goes into a *new* segment with the reason recorded, and the analysis layer
   decides which wins. Overwriting a row in place destroys the audit trail and
   leaves no way to discover that it happened.
2. **Duplicate `(config_key, backend, host)` is an error, not a merge.**
   `load_segments` raises and lists the offending groups. Last-wins would hide
   whether a duplicate was a deliberate re-measure or the same checkpoint
   joined twice — opposite situations wanting opposite treatment.
3. **All segments must share one git commit.** A code change between segments
   makes them different experiments. `allow_mixed_commits=True` exists but
   must be a deliberate, visible choice in the calling code.
4. **Every segment carries a `segment_digest`** — an order-independent
   SHA-256 over the sorted `(config_key, backend, host, latency)` tuples.
   Row order, column order and parquet encoding are incidental and do not
   change it; a changed measurement does.

---

## Per-session preconditions

Run in this order. Each is cheap; the sweep is not.

```bash
python3 -m pytest tests/ -q                    # includes the AST call-site check
bash scripts/gcp_cleanup_check.sh              # expect "Clean"
python3 -c "from attnbench import provenance; print(provenance.capture().git_commit)"
```

Then, **on the instance, before any sweep cell runs**:

1. **Stage 1 correctness at the current commit.**
   `load_stage1_pass_set(path, at_commit=<sha>)` — a pass recorded at another
   commit raises `Stage1CommitError` rather than being inherited. A pass
   describes the code that produced it: the 2026-09-03 `to_dense_bool`
   causality fix changed what a causal block-sparse mask *means*, so every
   block_sparse pass recorded before it describes a different function.
2. **Canary against the previous session** (`analysis/canary.py`):
   ```python
   assert_no_canary_drift(previous_segment_df, this_session_canary_df)
   ```
   Runs the canary cells (seq_len 1024 and 4096) and compares **within-host
   ratios**, not raw latencies — a ratio is normalised by its own machine, so
   re-renting different hardware of the same architecture does not trip it,
   while a driver update or image rebuild does. Tolerance 5%, against an
   observed repeat-measurement spread under 2%.

   **Do not widen the tolerance or edit `CANARY_SEQ_LENS` to make a firing
   canary pass.** That resets the baseline and makes the check decorative.
3. **GPU exclusivity** (`provenance.assert_exclusive`) — existing gate.

---

## Block-Sparse-Attention: one attempt, then move on

Session 4 attempts the BSA build once. **If it fails, do not retry and do not
debug it on billed hardware** — start Stage 2 and let BSA join later.

BSA is worth far less than it was before flex landed. It contributes 288 of
1404 cells, *all* at block_size=128, where flex already runs the same nominal
pattern. It has gone from "the only sparse arm" to "a second kernel
cross-checking a block size that is already covered". Resume is keyed on
`(config_key, backend, host)`, so adding those cells later costs only those
cells.

If BSA does land, its cells are a genuine independent check of flex at 128 —
two unrelated kernels on one mask object.

---

## What the integrity machinery is actually defending against

Not tampering. Every incident behind these checks was self-inflicted, and four
of them happened in a single day (2026-09-03):

- a correctness oracle computing a **different function** than the kernel
  under test (`to_dense_bool` permitting 63 future tokens per query)
- a metric reporting **JIT compilation as batching benefit** (`gla 4.28x —
  batching helps`, for a workload where batching does nothing)
- a **guard that passed without ever running** (`git_commit` recording the
  string `"HEAD"`)
- a **test whose edit never applied**, reporting success for an experiment
  that had not been performed

Every one produced a plausible number that was wrong, and none raised an
error. The checks above exist to make that class of result either impossible
to produce or impossible to mistake for a clean one. The recurring tell is
the same in all four cases: **a check nobody has watched fail is a guess.**
Each guard in this plan has been observed failing when deliberately broken,
and the tests that do so are in the suite.

---

## Abort conditions

- the canary fires and the cause is not identified
- Stage 1 passes cannot be verified at the current commit
- duplicate cells appear in a join
- any step fails twice
- an OOM kill appears in the serial log
