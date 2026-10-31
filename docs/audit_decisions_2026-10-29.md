# Audit C2 to C4 fixes: decisions for the plan (29 Oct 2026)

Branch `cloud/C18-2026-10-29`. This file proposes the plan wording and the
Appendix A lines for each decision the fixes carry out or needed. It edits
nothing in `docs/RESEARCH_PLAN.md`; the integration session applies the
edits in one batch with the researcher's yes. Lines marked **(open)** are
choices this branch made that the researcher has not confirmed.

Line numbers are those of `docs/RESEARCH_PLAN.md` on this branch (main at
`c1f8029`).

## 1. One sync gap and a planned wait (C2 finding 1)

**Done.** `predict/commit.py` holds `REQUIRED_SYNC_GAP_S = 60` (the gap and
the clock skew the analysis allows, one number) and
`PLANNED_SYNC_WAIT_S = REQUIRED_SYNC_GAP_S + 30`. The P2 and P4 runners
import both and hold for 90 s. The shakedown runners sync no prediction
and set no gap. The launch cards say 90 s.

**(open)** A start exactly 60 s after the sync is still refused: the check
stays `sync + skew < start`, as the runners' `gap > 60 s`. The plan says
"at least 60 s", so the wording should match the code.

- Proposed wording, Section 5.1 (line 121) and the 25 Oct line (641):
  "the timed workload starts more than 60 s after the prediction file and
  its sha256 are synced; a session waits 90 s (60 s plus a 30 s margin, so
  the store's whole-second creation time and the upload never eat the
  gap); the analysis allows 60 s of clock skew; both numbers are
  `predict/commit.py`'s."
- Appendix A: "**29 Oct 2026.** Sync gap (researcher's decision): one
  source for the 60 s gap and skew (`commit.REQUIRED_SYNC_GAP_S`); every
  session runner waits the gap plus 30 s (`PLANNED_SYNC_WAIT_S`); a start
  must be more than 60 s after the sync (Section 5.1, T2.11)."

## 2. The training set is exactly the pairings banked first (C2 finding 2)

**Done.** `load_committed` requires `banked_at`: every pairing banked so
far, or `{}` when none is. It refuses:

- a pairing banked before the write but left out of the training set;
- a pairing named but banked after the write (as before);
- a pairing banked within the clock skew of the write, which cannot be
  ordered **(open)**;
- the pairing's own ground truth banked before its prediction.

- Proposed wording, Section 5.4, "Fitting the baselines" (line 204): add
  "The analysis checks both directions: every pairing banked before the
  prediction was written is in its training set, and none banked after it
  is."
- Appendix A: "**29 Oct 2026.** Training set (researcher's decision): the
  committed baselines' training pairings must equal the pairings banked
  before the prediction, in both directions; a pairing banked within the
  allowed skew of the write is refused as unorderable (Section 5.4, T2.11)."

## 3. Scoring reads only a loaded prediction (C2 finding 3)

**Done.** `commit.require_loaded` refuses a prediction with no checked sync
time (`synced_at` None, from `verify_bytes`). `residuals` calls it, and the
H1 and T3.x tables go through `residuals`. The paper pipeline already
called `check_loaded`. The queue and density sensitivities still take
either reader: they recompute, they do not score.

- Proposed wording, T2.11 row (line 405): "Scoring (H1, residuals, the
  T3.x tables) refuses any prediction not read through `load_committed`."
- Appendix A: "**29 Oct 2026.** Scoring reads only predictions loaded
  through `load_committed` with a checked sync time (K9; T2.11, T2.15)."

## 4. Throttle detection (C3 finding 1)

**Done.**

- The software power cap cannot be put in a flag set (`RECORDED_ONLY`); it
  is recorded on every block.
- `ground_truth.Controls` refuses a counters-only `BlockSampler`.
- `ground_truth.Controls` also refuses any flag set that lacks one of the
  researcher's four bits.
- Tests inject hardware slowdown, both thermal slowdowns and the hardware
  power brake, and each one flags.
- The NVML reader is unchanged and UNRUN.
- The ground-truth test rigs now sample inside the block.
- `test_process_sampler`'s "a session that asks for the cap gets it" now
  expects a refusal.

**(open)** `t228_steps.NullSampler`, which measures the sampler's own cost,
is not a `BlockSampler`, so `Controls` still accepts it. Its runs are a cost
measurement, not ground truth.

- Proposed wording, T2.13 row (line 407): "The detector is the sampler
  process; a counters-only sampler is refused, because no counter shows
  hardware slowdown and the power counter holds the cap with the brake.
  The software power cap is recorded and can never flag."
- Appendix A: "**29 Oct 2026.** Throttle detection (researcher's decision):
  the session's controls refuse a counters-only sampler and any flag set
  missing hardware slowdown, either thermal slowdown or the hardware power
  brake; the software power cap is recorded and cannot be chosen to flag
  (Section 5.3, T2.13; restates the 27 Oct 2026 line)."

## 5. Every ground-truth row carries five fields (C3 finding 2)

**Done.** The new module `attnbench/gt_schema.py` (torch-free) holds the
fields and the builder kinds that C10 introduced:

- `context_length` and `batch`: the prompt's.
- `mask_density` and `executed_density`: as defined in `densities.py`.
- `builder_kind`: `"cpu"` or `"gpu"`.

How the fields are set and checked:

- A dense row has no mask, executes every block and has no builder kind.
- A deployable row must name a builder kind.
- `time_arms` takes a required density probe and checks every row before
  returning.
- The session's probe, `kernel_densities`, runs one untimed forward per arm
  after every block. It reads the executed density at the kernel boundary
  and the mask density from the backend's own `last_layer_density`.
- The figure readers refuse a row without the fields, and refuse a stated
  cell whose length or batch the row contradicts.
- C10 reads `BUILDER_KINDS` from `gt_schema`.
- The shakedown arms name their builder kinds: the oracle builds on the
  host (`"cpu"`); the mean-pool and both XAttention arms build on the device
  (`"gpu"`).

**(open)** Only some backends keep `last_layer_density`: the XAttention
backend, the backend without asserts, the compiled wrappers and the
precomputed-mask backend. The block-sparse backend used by the oracle and
mean-pool arms may not. If it does not, `kernel_densities` refuses the row
rather than guessing. The T3.1 preflight must confirm this, or the
block-sparse backend must start recording it.

- Proposed wording, Section 5.3 step 7 (line 176): "Every row carries its
  context length, batch, mask density, executed density and builder kind
  (`gt_schema`); a row without them is refused by the harness and by every
  reader."
- Status for T2.13 (line 407) and T2.20 (line 414): the claim that the
  harness puts both densities on its rows is true on this branch, not on
  main. The integration session updates both rows.
- Appendix A: "**29 Oct 2026.** Ground-truth rows (researcher's decision):
  every row carries context length, batch, mask density, executed density
  and builder kind ("cpu" or "gpu"; none for dense), checked by
  `gt_schema` in the harness and the readers (Section 5.3, T2.13, T2.20,
  T7.1)."

## 6. The dense control belongs to the cell (C3 finding 3)

**Done.** `dense_selection.check_control` takes `key`, which is the
prediction key or its K2 cell. It refuses a choice recorded for another
card, context length or batch. The card is the first field of the
code-path label, so it must equal the device name the trace reports. The
Shakedown 2 step passes the card and prompt it ran on.

- Proposed wording, T2.16 row (line 410): "`check_control` takes the
  prediction key and refuses a choice recorded for another card, length or
  batch."
- Appendix A: "**29 Oct 2026.** K2 control (researcher's decision): the dense
  control is checked against the choice recorded for the same card,
  context length and batch as the prediction key (Section 5.5, T2.16)."

## 7. Waits, mechanisms and "same sequence" (C4 findings 1 and 3)

**Done.**

- An implicit `cudaFree` wait is flagged (`trace_flags`) and not costed, and
  `to_operations` leaves it out.
- The flag goes into the banked sequence record as `"flags"` (checked on
  read), the sync map, and the arms-table column.
- The event synchronise and the blocking copies from the host, which the
  recursion has priced since C1, are no longer marked unpriced.
- `op_sequence.same_sequence` is the strict definition. It compares the
  ordered (operation class, stream, blocking or not, payload class) and
  ignores kernel names, aten operators and layer labels.
- `compare` keeps its fuller key for T2.4 and T2.18, and
  `structural_probe.py` is untouched.

The lock file states `scored_pairings` per hypothesis, with H0 and H2
required. The lock report prints each list, and a lock without them is not
complete.

**(open)** Choices inside the strict key:

- Operation class: the kind, with a synchronisation's mechanism and a
  copy's direction.
- Stream: its index in order of first use, not the device's stream id.
- Blocking: a synchronisation blocks; nothing else does.
- Payload classes: none, up to 8 B, up to 4 KiB, up to 1 MiB, above 1 MiB.
- The looser, secondary definition is still undefined.
- The implicit-wait flag reaches the banked sequence record, the sync map
  and the arms table, but not the prediction file, whose priced operations
  simply leave the wait out. If "the results" should also mean the
  prediction file, its cells need a flags field.

- Proposed wording, Section 5.1 (line 121, after "a looser one"): "The
  strict definition compares, in order, each operation's class (its kind,
  with a synchronisation's mechanism and a copy's direction), its stream
  (by order of first use), whether the host blocks on it, and its payload
  class (none, up to 8 B, up to 4 KiB, up to 1 MiB, above); kernel names,
  aten operators and layer labels are not compared. The pairings each
  hypothesis is scored on are listed in the lock file and printed with
  the lock table."
- Proposed wording, T2.3 row (line 397): replace "Waits the recursion does
  not price yet (a blocking copy from the host, an event synchronise) are
  named and refused; an implicit wait in `cudaFree` is flagged" with "A
  blocking copy from the host and an event synchronise are priced (25 Oct
  2026); an implicit wait in `cudaFree` is flagged in the record and the
  arms table and is not costed."
- Appendix A, line 1: "**29 Oct 2026.** Strict "same sequence" (researcher's
  decision): the ordered operation class, stream, blocking or not and
  payload class; kernel names ignored. The scored pairings per hypothesis
  are printed in the lock table (Section 5.1, T2.3, T4.1)."
- Appendix A, line 2: "**29 Oct 2026.** The implicit `cudaFree` wait is
  flagged, not refused, and the flag reaches the banked record; the three
  mechanisms added on 25 Oct 2026 are no longer marked unpriced (T2.3)."
- The C4 audit's plan-wording test (`tests/audit_C4_op_sequence.py`,
  finding 3) passes once the Section 5.1 sentence above is in the plan.

## 8. The mask bank checks its manifest (C4 finding 2)

**Done.** The manifest gains `calls_sha256`, a hash over the call entries.
`load` recomputes every call's density from its mask with its stated
causal flag. It refuses a manifest whose fields do not follow from the
masks, whose shape does not fit the packed bits, or that has no
`calls_sha256`. No bank is committed yet, so none becomes unreadable.

- Proposed wording, T2.4 row (line 398): add "The bank's manifest is
  checked on load: each call's density is recomputed from its mask, and a
  hash covers the density and causal fields."
- Appendix A: "**29 Oct 2026.** Mask bank: the manifest's density and causal
  fields are hashed and rechecked against the masks on load (T2.4)."

## 9. Portable `mktemp` in the deploy script

**Done.** `gcp_deploy_source.sh` makes its bundle in a directory from a
template that BSD and GNU `mktemp` both accept, and the trap removes it.
Before this, GNU refused `-t attnbench-deploy`, so the deploy stopped
before any gcloud call. On macOS, the file `mktemp` made was left behind.
It is tested under strict fakes of both implementations. No plan wording
is needed. A `docs/limitations.md` annotation, if the integration session
keeps one for Linux deploys, is its call.

## What the integration session also needs

- **Merges.** This branch merged C6, C7, C10 and C15 onto C5, then current
  `origin/main` (`c1f8029`, which brings the T2.13 sampler process, the
  strict stamp and Shakedown 2). None conflicted.
- **UNRUN tests.** These import torch; here they ran only against an
  import-only stand-in, never in the repository:
  - `test_audit_fixes_ground_truth.py` (23 tests);
  - `test_audit_fixes_mask_bank.py` (7);
  - the migrated `test_ground_truth_controls.py`,
    `test_ground_truth_timed_region.py`, `test_precomputed_masks.py`,
    `test_dense_selection.py`, `test_components_mask_bank.py` and
    `test_t228_shakedown.py`.
  
  The Mac session runs them with torch.
- **Off-repo items.** Record the push of this branch in the permission logs
  (rule 11).
