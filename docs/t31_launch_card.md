# Launch card: T3.1, the P1 session

Prepared 28 Oct 2026; rewritten 1 Nov 2026 after Shakedown 6 and the
researcher's rulings that followed it; updated 2 Nov 2026 after the first
session's crash and its diagnostic, and again after the third session's
out-of-memory stop. Deploys from `main` at the merge of
`t31-step5-memory-2026-11-02` or later. **The session starts from step 1
and runs all of step 5 again:** none of the third session's 31 cells is
reused (researcher's ruling, 2 Nov 2026), so every row comes from one code
identity. Launchable only on the researcher's
yes; nothing here runs on its own. **Stop at the first failure:** the session
runs once, the runner stops at the first step that fails, and the instance
is torn down. Nothing is fixed and resumed inside this session; a fix and a
new session need their own yes.

## What runs

`scripts/t31_session.py`, steps in `scripts/t31_steps.py`, values in
`configs/t31/p1_session.json` with its null working values filled from
`docs/lock_value_table.md` (marked "working, pre-lock", P1 only). Eleven
steps, in order, each in its own process with its own log
(`results/t31_p1_<UTC>/<NN>_<step>/step.log`), each synced to GCS as it ends
(`run_phase.sh`, phase `t31_<NN>_<step>`).

The grid: lengths 2048, 4096, 8192, 16384 and 32768 at batch 1, and 16384 at
batch 4 for dense only; dense, the oracle and mean-pool at 0.25 and 0.5, and
the two XAttention arms at their calibrated density. That is 36 cells. The
ground truth runs three compile groups: eager (all five arms), compiled
default and compiled capture (mean-pool and XAttention, each with the eager
dense control). It runs 10 clean rounds per cell and 5 dense-selection
rounds; the rerun budget is 0 and the floor is derived in the session.

Minutes are from `scripts/t31_projection.py` (low / central / high). Steps
1 to 4 are the third session's, which ran them on this grid; step 5 is that
session's 93 minutes plus what it did not reach; steps 6 to 11 are scaled
from Shakedown 6. Steps 6 to 11 have not run on a GPU, so their minutes are
the least certain.

| # | Step | Task | Minutes |
|---|---|---|---|
| 1 | `preflight` | T2.1 probe `start`, T2.20 fields, config of record; the kernel contract | 1.6 |
| 2 | `link` | T2.5 row, T2.6 Q, mechanism table per length | 2.7 |
| 3 | `dense_select` | T2.16 record per (length, batch); the session floor | 3.6 |
| 4 | `probe` | T2.18, every arm, at every grid length (durations discarded) | 13.9 |
| 5 | `host_components` | T2.2 per cell, each cell in its own process under a 26 GiB RSS ceiling: the recording, the null pass (instrumented), the bare prices with the per-layer residual, which the prediction reads | 119 / 139 / 170 |
| 6 | `device_components` | T2.4 replay per cell; the priced cells | 9 / 10 / 15 |
| 7 | `predict` | T2.11: write, sync, read back; the receipt | 1 |
| 8 | `hold` | until sync + 90 s (`PLANNED_SYNC_WAIT_S`), on the instance clock | 1.5 |
| 9 | `ground_truth` | T2.13, one process per compile group, under the recompile gate | 55 / 77 / 124 |
| 10 | `verify` | `load_committed` against the earliest ground-truth stamp | 1 |
| 11 | `postflight` | T2.1 probe `end` | 2 |

Step 5's minutes (low / central / high): the third session's 93.3 for 31
cells and the first 24 minutes of the 32nd; the rest of that cell,
`oracle_d0.5_L32768`, 10 / 20 / 35; the four mean-pool and XAttention cells
at 32768, 4 / 8 / 12; a fresh process per cell (imports, CUDA, the model
load), 12 / 18 / 30.

## What changed since the third session (2 Nov 2026)

The third session (`results/t31_p1_s3/`, ₹270, 149 min) passed steps 1 to 4,
the kernel contract clean on the L4 (memcheck 130 errors with the library's
own buffer, 0 with the fix; the real prompt bitwise 0 on 28 layers). Step 5
ran 31 of its 36 cells; in the 32nd, `oracle_d0.5_L32768`, the kernel's
out-of-memory killer took the step's process at 31.6 GB, during its isolated
replay's trace export.

**Measured on the Mac from the banked cells** (one read per process, peak
RSS above the process's start):

| What | Holds about | Measured on |
|---|---|---|
| Reading an isolated trace in full (`json.load`, `_events`) | 2.7 × the file | the oracle's 2048 to 16384 cells (33 MB to 827 MB) |
| Reading the recording's trace (`parse_trace`, `launch_check`), each | 4.9 × the file | `oracle_d0.5_L16384` (295 MB) |
| The profiler before and during its export | 2.2 × the trace it writes | a CPU-only synthetic run, 722 MB trace |
| Writing the compact stream | under 0.1 GB, at any size | the 3.3 GB `oracle_d0.25_L32768` trace, 49 s |
| Reading the compact stream | 330 bytes per kept event (0.8 × the full trace) | `oracle_d0.25_L16384` |

The oracle's traces grow with the length: its isolated trace is 3.1 GB at
`d0.25_L32768`, and `d0.5_L32768`'s projects to 5.7 GB (its recording is 1.85
times as large). So both causes held. **Memory built up across cells:** the
process held 18.9 GB during the 32nd cell's recording, more than any one read
of that phase needs. **And the 32nd cell was large on its own:** its export
alone needs about 12 GiB, and reading its trace in full about 15 GiB, so
alone it would have peaked near 20 GiB. Neither cause alone exceeded the
host; together they did.

**The fix** (researcher's ruling, 2 Nov 2026):
- **A process per cell.** Step 5 runs each cell as `t31_steps.py
  host-cell-one` under `attnbench/resource_guard.py`, which samples the
  child's RSS and kills it at the ceiling (26 GiB of the host's 31; a
  session config may set `host_cell_rss_ceiling_gb`). The cell then fails
  alone, with its record, and the step stops as at any failure.
- **The isolated trace as a compact stream.** It is read one event at a time
  (`attnbench/components/trace_stream.py`). Of it, the
  `isolated_host.events.jsonl` stream keeps what the profile reads, and the
  full trace is deleted. Each kept event has its operation class, its name,
  its thread, its timing and its index. The stream is read once, for both
  the isolated prices and the chain check. On the banked cells it gives the
  same launches, calls, flushes and prices as the full trace, and the same
  chain check.
- **Each cell's manifest** (`cell_manifest.json`) holds its code identity,
  the instance spec, the status, and the peak RSS, sampled and the child's
  own.
- **Resume per cell.** A cell resumes only where its manifest says it passed
  under the same commit, the same clean tree, the same libraries and the
  same instance. Any other cell's directory is cleared and the cell runs
  afresh. This session does not use resume (no `--from`); the third
  session's cells are not reused.
- **The bare prices** are one cache for the step (`bare_price_cache.pkl`),
  as when the cells shared a process.

**What is not compacted:**
- **The recording's own trace** stays whole. The sequence parser reads its
  Python frames: the call instance of an extension's kernel, such as the
  block-sparse call, comes from them. The launch check reads its shapes
  (ruling, 1 Nov 2026), and `recording.json` pins the trace by sha256. Its
  profiler settings are unchanged.
- **The isolated replay's profiler** already records no stacks, shapes,
  memory or flops.
- **No full trace is kept for Figure 1.** It reads T2.27's own recording,
  not a T3.1 trace.

**Projected peak RSS per cell** (GiB): the process's base plus the cell's
largest phase. The base on the L4 is not measured; it is taken as 3 to 6
GiB. The first cells' manifests will show it.

| Cell | Largest phase | Above base | Peak | Against 26 |
|---|---|---|---|---|
| `oracle_d0.5_L32768` | the isolated trace's export (5.7 GB) | 12.2 | 15 to 18 | 8 to 11 GiB under |
| | (reading it in full, before the fix) | (15.4) | (18 to 21) | |
| `oracle_d0.25_L32768` | the isolated trace's export (3.1 GB) | 6.6 | 10 to 13 | |
| `oracle_d0.5_L16384` | the isolated trace's export (1.5 GB) | 3.3 | 6 to 9 | |
| dense, mean-pool, XAttention, any length | traces of 33 MB or less | under 0.2 | 3 to 6 | |

The profiler's figure is from a CPU-only run. With CUDA activity it may hold
more per event. Even if it held twice as much, `oracle_d0.5_L32768` would
peak at 30 GiB. That is over the ceiling, so the guard would stop that cell
alone and record it.

## What changed since the second session (2 Nov 2026)

The second session (`results/t31_p1_s2/`, ₹22, `asia-northeast1-c` after a
stockout in zone a) stopped in step 1, in the kernel contract. memcheck's
positive control saw no error with the library's own buffer: without padding,
a read past one allocation that lands in the next one is not an error. memcheck
now pads every allocation by 512 bytes (`--padding`).

Every other part of the contract passed:
- **Kernel against the reference:** 0.0156.
- **Padding and tail values:** they changed nothing bitwise on the synthetic
  masks or on the real prompt (28 layers, with the library's own buffer run
  and compared).

So the effect is crash-only, and the `qa_1` re-check is dropped.

## What changed since the first session (2 Nov 2026)

The first session (1 Nov 2026, `results/t31_p1/`, ₹53) passed steps 1 to 3
and stopped in step 4 at XAttention's first full recording at 2048 with a
CUDA illegal memory access. The diagnostic (`docs/t31_xa_diag_card.md`,
₹78) located it in Block-Sparse-Attention's forward kernel:
- **The read.** The kernel reads its int32 row mask one entry past any fully
  selected row, and for the last row of the last head one entry past the
  buffer. In the source (`flash_blockmask.h` at the pinned `49d6c39`), the
  value read is discarded. The read faults only when the address past the
  buffer is unmapped.
- **Exposure.** XAttention reaches the read in every banked prefill, at
  layers 8, 16, 18 and 19 at least; no banked mean-pool or oracle mask does
  (`scripts/bsa_row_exposure.py`).
- **The fix, in the adapter.** The library is unchanged. Every kernel call
  now gets a buffer with a row of -1 after it (`attnbench/bsa_contract.py`,
  `row-mask-tail-1`). That adds one fill kernel of one row per call, for the
  oracle, mean-pool and XAttention alike. Rows record the library's version,
  its interface file's sha256 and the fix.
- **The kernel contract, in step 1**, before any arm runs. Any failure
  refuses the session. It checks:
  - **the pinned library:** version 0.0.2 and the interface file's sha256;
  - **the conversion:** the fix's rows equal the library's;
  - **the output:** the kernel against a reference masked attention (fp32,
    tolerance 0.02), at 1951 and 4096 tokens on 12 heads, with every head's
    last row and one middle row fully selected;
  - **that the tolerance can see one block** removed from the mask;
  - **that the value past a row is unused:** the output is bitwise unchanged
    when the row after the buffer holds 0 or the last block instead of -1;
  - **that the padding changes nothing:** the output is bitwise unchanged
    between the library's buffer and the fixed one, on a mask whose stray
    reads stay inside the buffer, as the banked results' did for heads 0
    to 10 (`padded_vs_unpadded`, the number the earlier results rest on);
  - **the same on a real prompt:** XAttention's prefill on the grid's first
    2048 prompt (`qa_1`, 1951 tokens), in a process of its own. It runs the
    fix twice (the two runs must be bitwise equal, or no bitwise comparison
    holds), then with 0 and with the last block after the buffer. A second
    process runs the library's own buffer. Every layer's output must be
    bitwise equal to the fix's. If that second process faults, the fault is
    the known read: it is recorded, not refused, and the result is not clean;
  - **memcheck:** one kernel call at 1951 under `compute-sanitizer` with
    PyTorch's caching allocator off. With the fix it must report no errors.
    With the library's buffer it must report some: a check that cannot see
    the read proves nothing.

## What changed since Shakedown 6

- **Step 5, the bare timer's entry.** A custom op's re-issues were counted
  as calls, which refused the oracle in Shakedown 6. They are no longer
  counted (`5375b07`); on the instance, with the fix applied, the oracle
  aligned at 2048 and passed the chain gate at 5.5%.
- **Step 5, once-only host pricing.** XAttention's null pass was refused at a
  `where` with a Python number. Each pass call is now priced once, inside the
  recorded call that nests it. This was tested on Shakedown 6's own sequence,
  with a conservation test, and the CPU dry run now reproduces the nesting.
- **Step 5, the per-layer residual.** The arm's own Python between calls,
  measured per layer from the prefill's span under the tape with real kernels
  and added back to the bare prices. A layer whose spread is over T4.1 item
  34 (working 10%), or where the device fell behind, labels the cell. It is
  never refused.
- **Step 5, speed.** The isolated trace is read by bisection: the oracle's
  8192-token trace took 4 s, down from over 27 minutes, and that alone ended
  Shakedown 6's zoo. A null pass takes about half its former time.
- **Step 9, the kernel interface outside Dynamo for every arm.** Compiled
  mean-pool recompiled per call under the default settings in Shakedown 6.
  The oracle, mean-pool and XAttention now call the kernel through
  `block_sparse.kernel_interface`, one declared break under both settings.
  The recompile gate (0 compile events after warm-up) refuses the cell
  otherwise, which stops this session.

## The instance

| | |
|---|---|
| Card | 1 × NVIDIA L4, `g2-standard-8` (32 GB RAM) |
| Zone | `asia-northeast1-a`; fallback `asia-northeast1-c` (plan Section 5.2), with `GCP_ZONE` and `ZONE` both set, only on the researcher's word (used on 2 Nov 2026, when zone a stocked out) |
| Provisioning | STANDARD (on-demand) |
| Image | `attnbench-env-v6-20260917` (set `ATTNBENCH_IMAGE`; the launcher defaults to v5) |
| Bootstrap | `scripts/gcp_bootstrap_instance.sh NAME asia-northeast1-a`, with XAttention |
| Rate | ₹108.64/h all-in planning rate (`GCP_RATE_INR_HR=108.64`) |
| Launch window | by 9 Dec 2026; within the credit by 30 Nov 2026 |

While this VM exists, the H100 retry loop running on the Mac skips its rounds
(one GPU session at a time). It resumes by itself after the delete.

## Time and cost

| | Hours | ₹ at 108.64/h |
|---|---|---|
| Setup: launch, deploy, preflight, write gate, bootstrap (Shakedown 6: 6 min) | 0.1 | 11 |
| Steps 1 to 11, central (low 3.5 h, high 5.6 h) | 4.2 | 459 |
| Teardown: results pull, delete, cleanup check | 0.13 | 14 |
| **Expected, central** (low 3.8 h ₹408, high 5.8 h ₹635) | **4.5** | **484** |
| In-guest halt (`GCP_HALT_MINUTES=420`): GPU billing stops | 7.0 | 760 |
| Hard delete (`GCP_MAX_RUN=7h30m`) | 7.5 | disk only after the halt |

**Cost cap: ₹760**, the halt. The projection's central case is inside the
expected 5 h. Its high case (5.8 h) is over that, but inside the cap. Each
failure from step 6 on would cost a whole session, 2 to 3 hours and ₹100 to
₹300. There is no "fixed and resumed" row: the session stops at the
first failure.

## The commands

On the Mac, at the launch commit, with a clean tree:

```bash
export ATTNBENCH_GCP_PROJECT=...          # this study's project, never the gcloud default
export CLOUDSDK_CORE_PROJECT="$ATTNBENCH_GCP_PROJECT"
export ATTNBENCH_IMAGE=attnbench-env-v6-20260917
export GCP_HALT_MINUTES=420 GCP_MAX_RUN=7h30m GCP_RATE_INR_HR=108.64
NAME=attnbench-l4-t31-$(date +%Y%m%d-%H%M); ZONE=asia-northeast1-a

bash scripts/gcp_cleanup_check.sh                       # expect "Clean"
bash scripts/gcp_launch_l4.sh "$NAME"                    # type 'launch'
bash scripts/gcp_deploy_source.sh "$NAME" "$ZONE"
bash scripts/gcp_preflight_instance.sh "$NAME" "$ZONE" --quarantine
bash scripts/gcp_verify_gcs_writable.sh "$NAME" "$ZONE"
bash scripts/gcp_bootstrap_instance.sh "$NAME" "$ZONE"
```

On the instance, from `~/attnbench_scaffold`, once:

```bash
python3 scripts/t31_session.py          # stops at the first failure; no --from in this session
```

Back on the Mac, when it ends (passed or stopped):

```bash
bash scripts/gcp_session_elapsed.sh "$NAME" "$ZONE" 108.64
bash scripts/gcp_teardown_session.sh "$NAME" "$ZONE" results/t31_p1_s4   # sessions 1 to 3: results/t31_p1, results/t31_p1_s2, results/t31_p1_s3
bash scripts/gcp_cleanup_check.sh
```

## The prediction and the clock

- Steps 1 to 6 time microbenchmarks, kernels, null passes, bare calls and
  the tape span only, never the composed workload (plan Section 5.1). The
  residual's span runs the arm's prefill under the tape before the
  prediction is written; it is a component measurement, as the null pass is.
- Step 7 writes the file and its sha256, uploads both with no-clobber,
  reads them back, and records the store's creation time (`synced_at`).
- Step 8 waits until the instance clock reads `synced_at + 90 s`.
- Each ground-truth process checks the same condition itself before its
  first forward and records `clock_not_before` there.
- Step 10 runs the analysis's own refusal in the session, so a late
  prediction is found before teardown.

## Before launch, on the Mac

The CPU dry run, which must pass. It needs x-attention cloned outside the
repo at `XATTN_COMMIT`, and transformers 4.46.0 (the L4's) in a venv over
the system torch:

```bash
export ATTNBENCH_XATTN_DIR=/path/to/x-attention
python3 scripts/t31_session.py --cpu-boundary --session /tmp/t31_cpu
python3 scripts/t228_steps.py density --cpu-boundary --out /tmp/density --length 1024
python3 scripts/t228_steps.py pricing_zoo --cpu-boundary --out /tmp/zoo --length 1024
python3 scripts/t227_steps.py graph-breaks-one --cpu-boundary --arms meanpool --setting default --out /tmp/gb --gb-length 1024
python3 scripts/t31_projection.py
```

## The CPU dry run (2 Nov 2026, after the third session)

Behind the CPU boundary (transformers 4.46.0, torch 2.13.0), on the step-5
fix before its commit, the rehearsal passed steps 1 to 8 in 9.4 minutes:
- **Step 5** ran its 15 rehearsal cells, each in its own `host-cell-one`
  process, in 127 s. Every cell wrote its manifest. The peaks were 0.5 to 1.0
  GiB against the 26 GiB ceiling. For each cell, the guard's sampled peak and
  the child's own figure were within 100 MiB of each other; the manifest
  keeps both and their maximum.
- **Compaction:** each cell kept `isolated_host.events.jsonl` (for the
  oracle at 2048, 8,826 of 9,736 events, 0.7 MB of 3.1 MB), and no full
  isolated trace was left.
- **Signatures:** the step timed 711, one cache across the cells.
- **Step 6** read the per-cell output and passed.
- **Resume:** the tree was dirty (uncommitted), so no cell could have resumed,
  and none did.

The CPU's timings and peaks mean nothing for the L4's.

## The CPU dry run (2 Nov 2026)

At the launch branch's head, behind the CPU boundary, the rehearsal (steps 1
to 8) passed in 6.6 minutes. Step 1 ran the kernel contract in 15 s, whose
CPU parts behave as follows:
- **The walk emulation.** With the library's layout it reads past the buffer
  (the positive control); with the fix it does not.
- **The output parts.** They ran against the stub kernel, which ignores the
  row buffer, so they only exercise the code. Kernel against the reference:
  0.008. One block removed changes the reference by 1.29, well outside the
  0.02 tolerance. Tail-value effect and padded against unpadded are both 0.
- **The real prompt.** Both processes ran: the padded variants, and the
  library's own buffer. On the CPU the model is the random two-layer stand-in
  and the prompt is random; on the L4 it is `qa_1` at 1951 tokens. Control,
  tail values and padded against unpadded were all 0.
- **memcheck** is not run: there is no CUDA device.

Every recording's code path names the fix (`bsa_row_mask_fix`). The kernel
itself, the tolerance on the L4 and memcheck are the GPU's to check, in step
1 of the session.

## The CPU dry run (1 Nov 2026)

On the Mac, behind `attnbench/cpu_boundary.py` (transformers 4.46.0,
x-attention at `XATTN_COMMIT`), at the launch branch's head, everything
passed:
- **The rehearsal, steps 1 to 8.** In step 5 every cell was recorded,
  null-passed, bare-priced and given its residual. Each XAttention cell
  priced its `scalar_tensor` calls inside their `where` (10 per cell, 8 for
  the assert-free variant), the nesting that refused Shakedown 6.
- **`density`.** Both compiled groups made 0 compile events in 12 prefills
  after warm-up.
- **`pricing_zoo`.**
- **The graph-break reports**, each arm under both settings. Mean-pool and
  the oracle show exactly the declared break, at `block_sparse.py
  kernel_interface`, and dense shows none. The XAttention arms show the
  declared break and one more at `Xattention.py:58`: Dynamo cannot trace the
  stub device's properties object there. The L4's own report in Shakedown 6
  had no such break.

The CPU's timings, residuals, ratios and labels mean nothing: a CPU call runs
its compute.

## Known before launch

- **Steps 5 to 11 have not run on a GPU for a sparse arm.** In Shakedown 6
  step 5 priced dense at 2048 and the oracle with the entry fix applied at
  run time. Once-only pricing, the residual and the speed-ups are tested on
  the CPU and on Shakedown 6's banked sequence and traces, not on the L4.
  The first failure stops the session.
- **The recompile gate is the proof of the kernel exclusion.** Behind the
  CPU boundary the stub sizes a tensor from its data, so a traced interface
  shows its break and is refused as another code path. The L4's per-call
  recompile itself cannot be reproduced on a CPU.
- **The oracle's host work grows with length.** It builds its mask on the
  host: 21,507 host operators per prefill at 2048 and 125,751 at 8192. At
  that rate it reaches about 735,000 at 32768, which makes recording and
  isolated-replay traces of over 1 GB. The third session found the limit:
  step 5 held one process for every cell, and the killer took it in
  `oracle_d0.5_L32768`. Step 5 now runs a process per cell under a ceiling,
  with the isolated trace compacted (above).
- **Step 6 still runs every cell in one process** (not in the ruling;
  proposed, not built). Per cell it reads the recording's trace when the
  sequence has flagged memsets: about 5 GiB for `oracle_d0.5_L32768`. Where a
  cell's graph replay does not match the whole sequence, it also runs the
  isolated replay of the original operators and reads that trace in full
  (`isolated_original.json`). At the oracle's 32768 cells, that read is the
  12 to 15 GiB of step 5's before the fix. A step-6 out-of-memory stop would
  cost a session.
- **The residual's device check.** A layer where the device fell behind the
  host is labelled, and its span may hold launch-queue waits. Expect labels
  at 32768.
- **The density figures and the floor are working, pre-lock values**, and
  the rows are P1's pre-lock data (plan Section 5.3).
- **The H100 loop** on the Mac skips its rounds while this VM exists. It is
  restarted from `main` with Spot on only on the researcher's own yes;
  until then the old Flex-only loop runs.
- **What step 1 decides about earlier results** (researcher's ruling, 2 Nov
  2026). Clean (`clean` in step 1's result) means the padding changed nothing,
  bitwise, on every synthetic mask and on the real prompt. If clean, the
  T1.1 `qa_1` re-check is dropped, an Appendix A line records the effect as
  crash-only, and the limitations note takes the measured result. A
  difference refuses the session in step 1, and the re-check stays. The
  second session measured clean, and all three were done on 2 Nov 2026.
- **XAttention again.** If the contract passes and XAttention still fails in
  the session, the session stops there (first failure). The fallback ruled on
  2 Nov 2026 then applies: T3.1 runs with XAttention quarantined, "not run,
  crash under investigation".
- **Teardown waits for the home logs** (step 2b): it refuses to delete until
  every `~/*.log` is pulled. A VM whose SSH is dead needs
  `TEARDOWN_DELETE_WITHOUT_HOME_LOGS='<why>'`.
