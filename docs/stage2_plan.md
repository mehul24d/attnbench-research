# Stage 2 — multi-session kernel sweep on L4

Stage 2 is the study's core timing deliverable and is **unblocked now**: the
dense, linear and flex-sparse arms all run without Block-Sparse-Attention.

---

## Provenance precondition — RESOLVED 2026-09-03

This project is now its own git repository, rooted at `attnbench_scaffold`,
with a real baseline commit (`eee2c6b32cf4`). `provenance.capture()` records a
40-hex SHA and `git_dirty=False` on a clean tree.

It previously sat inside an accidental repository rooted at the user's **home
directory** — a clone of an unrelated project. `git rev-parse HEAD` failed
there, and because git echoes the unresolved argument to stdout before writing
its error to stderr, provenance captured the literal string `"HEAD"` and stored
it as a commit. See `docs/silent_failure_patterns.md` instance 3.

**Results written before eee2c6b32cf4 carry `git_commit="HEAD"`** and cannot be
joined with anything measured after it except via `allow_unverified=True`.
That includes the 2026-09-03 batch-scaling and anchor measurements. They remain
valid as measurements — the number is what the machine did — but their
provenance is not verifiable, and they must not be silently mixed into a Stage
2 join.

Before each session, confirm:

```bash
cd ~/Desktop/research/attnbench_scaffold
python3 -c "from attnbench import provenance; print(provenance.capture().git_commit)"
# must print a 40-hex SHA, not None and not HEAD
git status --short        # empty: results should be measured from committed code
```

A dirty tree is not fatal — `git_dirty` records it — but a Stage 2 segment
measured from uncommitted code cannot be reproduced from its own commit field,
which defeats the purpose of recording one.

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

   **Segment 2 must pass `rebased_backends=frozenset({"flex", "naive"})`.**
   Both had work hoisted out of their timed region on 2026-09-04 — flex moved
   4.22 → 40.72 useful TFLOPS at 1024/b1 — so their ratios against segment 1
   have moved for a reason the canary was never meant to detect. Firing there
   would be a false alarm, and a false alarm teaches a reader to ignore the
   check. Excluding them is not a loophole: every name must have a
   `canary.BaselineChange` entry naming the commit and the reason, or the call
   raises. `sdpa_*`, `fa2`, `gla` and `block_sparse` are unchanged and must
   still be compared.
3. **GPU exclusivity** (`provenance.assert_exclusive`) — existing gate.
4. **Stage 1 must reproduce the 2026-09-04 flex diagnostic**, after the probe
   and **before the sweep**:
   This ran as `scripts/check_stage1_against_diagnostic.py` against each
   segment's `probe/correctness.parquet`, non-zero exit meaning stop. The
   script was removed after the last segment; the comparison it made is in
   `tests/test_diagnostic_agreement.py`, which runs in the ordinary suite. The diagnostic path (a script calling
   `check_for_family` directly) and the pipeline path (`run_probe.py` over the
   full grid) must agree, or something differs between them and it is worth
   knowing at the cost of one comparison rather than 504 cells.

   The check can **name** its failure rather than merely detecting one, because
   eager and compiled flex leave different numerical fingerprints at the same
   config and seed — 0.0157/0.0169 against 0.0084/0.0090, consistently ~1.9×
   apart. A run reporting the eager values has fallen back again, and the tool
   says `EAGER_SIGNATURE` instead of a generic mismatch someone would have to
   diagnose from scratch a second time. It is a spot-check on a process-scoped
   property (2 of 72 cells), not grid coverage — see the module docstring.

### Segment 2 carries two obligations from segment 1

- **Stage 1's existing pass table contains 72 void flex entries** — passes
  earned by the eager fallback. They must be re-earned under `compile_guard`,
  not inherited.
- **Segment 1's 84 flex rows stay void** and are re-measured through the full
  pipeline. The 2026-09-04 diagnostic numbers are *diagnostics*: they bypassed
  the Stage 1 pass table deliberately to fit a 30-minute cap, and live in
  `results/diagnostics/`, never `results/stage2/`. The other 219 rows from
  segment 1 stand — `flex` is the only backend that calls `torch.compile`.

### The A100 session carries one cheap check that is not about timing

**Does `flex` lower block-sparse on sm_80?** On the L4 it cannot: 114688 B of
shared memory required against a 101376 B limit, which is why flex is
unavailable as a cross-backend reference and why the sparse arm has no
numerical verification above 4096 (see `docs/limitations.md`). An A100 has
164 KB of shared memory per SM and 80 GB of device memory, so both of the
constraints that bind on the L4 may not bind there.

Run it **before** the sweep, as a precondition-style probe, not after:

1. One `block_sparse` config at 8192 and one at 16384, through
   `check_for_family` with the full reference set.
2. If flex lowers, the sparse arm gains real `cross_backend` verification at
   8192 and 16384 **on at least one architecture** — a materially stronger
   claim than the L4 result, and one that costs minutes.
3. If it does not, record the sm_80 numbers next to the sm_89 ones. "Flex
   block-sparse is shared-memory-bound on two generations" is a finding about
   a shipping PyTorch feature, not a gap.

Either outcome is worth recording. Neither changes the timing plan.

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

Not tampering. Every incident behind these checks was self-inflicted, and
**six** are now recorded in `docs/silent_failure_patterns.md` — including the
one that would have made this integrity layer decorative (`git_commit`
recording the literal string `"HEAD"` on every row, so a consistency check
over them would pass while verifying nothing).

Read that file before relying on any check here. Its through-line is the rule
this plan depends on: **a check nobody has watched fail is a guess.** Every
guard referenced above has been deliberately broken once, observed failing,
and restored, and the tests that do so are in the suite.

---

## Abort conditions

- the canary fires and the cause is not identified
- Stage 1 passes cannot be verified at the current commit
- duplicate cells appear in a join
- any step fails twice
- an OOM kill appears in the serial log
