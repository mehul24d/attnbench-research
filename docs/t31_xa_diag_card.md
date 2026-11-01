# Launch card: the XAttention crash diagnostic (after T3.1's first session)

Prepared 2 Nov 2026 on the researcher's ruling after T3.1's first P1
session. Launchable only on the researcher's yes; nothing here runs on its
own. One L4 session, cap ₹120, in-guest halt at 60 minutes. No T3.1 step
runs, and nothing timed is kept.

## What was observed

T3.1's first session (1 Nov 2026, `results/t31_p1/`) stopped in step 4.
Dense, the oracle and mean-pool were recorded at all five grid lengths, and
XAttention's three probes (20480, 28672, 32768). XAttention's full recording
at 2048 then raised "CUDA error: an illegal memory access" at x-attention
`utils.py:145` (`find_blocks_chunked`). Observed under our harness at
`c38e196` (x-attention `e379887`); the cause is under investigation. The
error is asynchronous: the kernel that faulted may be an earlier one.

## What the code says (static review, 2 Nov 2026)

- **Line 145 cannot make the bad index itself.** Its index tensor comes from
  `torch.sort` over the last dimension of the same-shaped input, and the
  `torch.where` beside it can only set entries to 0. So the fault is most
  likely an earlier kernel, reported at the next launch.
- **On the L4 the estimator is all PyTorch operators.** x-attention turns
  Triton off on any device whose name lacks "100". So the one non-PyTorch
  kernel in an XAttention layer is `block_sparse_attn_func`.
- **The failing prefill was not under the profiler.** It was the recorder's
  untraced warm-up (`record_prefill_trace`, before the profiler opens),
  inside the recorder's mask capture, which keeps each layer's mask alive.
- **`Harness.ready` had just run one 2048 prefill of the same arm** and
  synchronised it without error. Two identical calls, one clean and one
  faulting, point to a placement-dependent out-of-bounds access: the same
  stray address is harmless in one memory layout and unmapped in another.
  The allocator's layout differs between the two calls, and between T3.1
  and Shakedown 6, whose earlier arms stopped at 16384 rather than 32768.
- **No adapter state crosses recordings.** Every recording builds a new
  backend. The threshold cache holds one row per layer and is not sized by
  length. The three kernel tensors (query lengths, key lengths, head-mask
  type) are built on every call, as upstream builds them.
- **The assert-strip hook was not on the path.** The frame is the shipped
  `xattn` package with its asserts. The hook serves only its alias, and only
  the assert-free arm, which runs after XAttention, installs it. Its asserts
  in `find_blocks_chunked` also come after line 145.

## What runs

`scripts/t31_xa_diag.py session`. Each case runs in a process of its own
and is synced to GCS as its own phase (`t31xa_<case>`). Every case writes
`events.jsonl`, which records each recording's start and end and the
XAttention layer and length last entered, fsynced per line. Each case also
writes a `case.json` with its status (reproduced, clean, other error, or cut
by the deadline) and the traceback.

| Case | What | Expected min |
|---|---|---|
| C | T3.1's step-4 order: dense, the oracle and mean-pool (probes, then the 5 grid lengths), XAttention (its probes, then the grid), then the assert-free arm. `CUDA_LAUNCH_BLOCKING=1`, so the faulting kernel is named at its own launch | 10 to 20 |
| B | XAttention's three probes, then its grid lengths from 2048 | 3 to 5 |
| A | XAttention alone at 2048, recorded three times | 2 |
| D | C's order with no profiler: each recording runs the same prefills, synchronises and mask capture, untraced | 8 to 10 |
| E | C's order without the assert-free arm, with the assert-strip hook held off (any install fails the case) | 8 to 10 |

`compute-sanitizer --tool memcheck` runs once, with PyTorch's caching
allocator off, so that each access is checked against its own tensor's
bounds. It runs on the smallest of A and B that reproduced, as soon as A has
run (before D and E, which repeat C), with a 10-minute limit. C, D and E are
too long for that limit. A memcheck error names the kernel and the address
even where the stray access does not fault.

**Proposed (on the yes only):** `--sanitize-a-anyway`, memcheck on A when
neither A nor B reproduced. An out-of-bounds access can land in mapped
memory and not fault, so A under memcheck can find it without reproducing
the crash.

**How the cases separate the hypotheses:**
- **C reproduces and names a kernel.** The fault is located: either
  `block_sparse_attn_func` or a PyTorch operator.
- **B or A reproduces.** The cause is in XAttention's own sequence, and
  memcheck runs on the smaller case.
- **Only C, D and E reproduce.** The earlier arms' state matters: the
  allocator layout, or the mask capture's references. D rules the profiler
  in or out, and E rules out the assert-free arm and the hook.
- **Nothing reproduces.** That is itself a finding about how often it
  happens (one failure in T3.1). The fallback below then applies.

## The instance

| | |
|---|---|
| Card | 1 × NVIDIA L4, `g2-standard-8` (32 GB RAM), `asia-northeast1-a`, STANDARD |
| Image | `attnbench-env-v6-20260917`; bootstrap with XAttention |
| Rate | ₹108.64/h all-in planning rate |
| In-guest halt | `GCP_HALT_MINUTES=60` (₹109) |
| Hard delete | `GCP_MAX_RUN=66m` (₹120; disk only after the halt) |
| Deadline | no case or sanitizer starts after 48 min from boot; each case is cut at it |

**Cost cap: ₹120.** Expected: setup 7 min, cases and sanitizer to the
deadline, then teardown 5 to 7 min. That is about 55 min, or ₹100. If C runs
long, the deadline cuts E first, then D.

## The commands

On the Mac, at the launch commit, with a clean tree:

```bash
export ATTNBENCH_GCP_PROJECT=...          # this study's project, never the gcloud default
export CLOUDSDK_CORE_PROJECT="$ATTNBENCH_GCP_PROJECT"
export ATTNBENCH_IMAGE=attnbench-env-v6-20260917
export GCP_HALT_MINUTES=60 GCP_MAX_RUN=66m GCP_RATE_INR_HR=108.64
NAME=attnbench-l4-xadiag-$(date +%Y%m%d-%H%M); ZONE=asia-northeast1-a

bash scripts/gcp_cleanup_check.sh
bash scripts/gcp_launch_l4.sh "$NAME"                    # type 'launch'
bash scripts/gcp_deploy_source.sh "$NAME" "$ZONE"
bash scripts/gcp_preflight_instance.sh "$NAME" "$ZONE" --quarantine
bash scripts/gcp_verify_gcs_writable.sh "$NAME" "$ZONE"
bash scripts/gcp_bootstrap_instance.sh "$NAME" "$ZONE"
```

On the instance, from `~/attnbench_scaffold`, under `nohup`, with the
session and driver logs in `~/`:

```bash
MIN=$(( 48 - ( $(date +%s) - $(date -d "$(uptime -s)" +%s) ) / 60 ))
python3 scripts/t31_xa_diag.py session --deadline-min "$MIN" > ~/t31xa_session.log 2>&1
```

Back on the Mac, when it ends:

```bash
bash scripts/gcp_session_elapsed.sh "$NAME" "$ZONE" 108.64
bash scripts/gcp_teardown_session.sh "$NAME" "$ZONE" results/t31_xa_diag
bash scripts/gcp_cleanup_check.sh
```

## Teardown now waits for the home logs

T3.1's first session lost `~/t31_session.log` and `~/t31_driver.log` to the
delete (researcher's ruling, 2 Nov 2026). `gcp_teardown_session.sh` now has a
step 2b. It lists every `~/*.log` on the instance with its size, copies each
into `<results>/home_logs/`, and counts a copy only if it is at least the
listed size. If any log is missing, it refuses the delete with exit 3, and
the instance stays up, bounded by its halt and hard delete. An instance with
dead SSH is deleted by re-running with
`TEARDOWN_DELETE_WITHOUT_HOME_LOGS='<why>'`, and the reason is written to
`home_logs_missing.txt`. A self-halted VM has no SSH either, so it needs the
override as well.

## The CPU dry run (2 Nov 2026)

On the Mac, behind `attnbench/cpu_boundary.py` (transformers 4.46.0,
x-attention at `XATTN_COMMIT`), at the launch branch's head:

```bash
export ATTNBENCH_XATTN_DIR=/path/to/x-attention
python3 scripts/t31_xa_diag.py session --cpu-boundary --session /tmp/xa_dry --deadline-min 120 --sanitize-a-anyway
```

It passed in 19 minutes on the CPU boundary's grid (2048 and 4096, with
XAttention's probes at 20480, 28672 and 32768):
- Every case was clean and ran in its own process: C 22 of 22 recordings,
  B 5, A 3, D 22 and E 17.
- E held the hook off throughout, the only case to do so. Behind the CPU
  boundary the boundary itself installs the hook, and E takes it off first.
- The sanitizer chose A (`--sanitize-a-anyway`). It did not run, since
  compute-sanitizer is not on the Mac.

The unit tests (`tests/test_t31_xa_diag.py`) cover the case orders, the
environments, the deadline, the sanitizer's choice and placement, D's
prefills and synchronises, E's guard and the layer watch. Three mutations
were each watched to fail.

## Results (1 Nov 2026 UTC, `attnbench-l4-xadiag-20261102-0131`, 43 min, ₹78)

Run from `09f2c11` with `--sanitize-a-anyway`, as approved, and banked in
`results/t31_xa_diag/`. Observed under our harness, with x-attention at
`e379887` and block-sparse-attn 0.0.2. The cause is under investigation,
and the kernel source has not yet been read.

| Case | Outcome |
|---|---|
| C | Reproduced at recording 24, XAttention's first full recording (1951 tokens), layer 8. With `CUDA_LAUNCH_BLOCKING=1` the error is raised in block-sparse-attn's `fwd_block` (`block_sparse_attn_func`), not in `find_blocks_chunked` |
| B | Clean (8 of 8) |
| A | Clean (3 of 3) |
| D | Reproduced at the same recording, with no profiler (layer 9 entered: the error is reported at the next launch) |
| E | Reproduced at the same recording, with the hook held off |
| memcheck on A | 130 errors. Every printed one is an invalid 4-byte global read in `flash_fwd_block_kernel` (head dim 128, 64×64 tiles), in grid block (30, 0, 11), the last 64-row query tile of head 11, the last head. Each is just past the end of a 12,288-byte allocation, the size of the library's int32 row block mask for this call, (1, 12, 16, 16) |

**Two fits with the banked masks:**
- **A full last row.** `convert_blockmask_row_reverse` lists each row's
  selected key blocks and pads with `-1`. A row with all 16 blocks selected
  has no `-1` after it, so one read past the row leaves the buffer at the
  last head.
- **Layer 8.** In A's and B's masks, layer 8 is the first layer whose last
  head has its last row fully selected (also layers 16, 18, 19, 20, 26 and
  27). Layer 8 is where C faulted.

The read faults only when the address past the buffer is unmapped. That
depends on the allocator's layout, which the earlier arms' 32768 work sets:
A and B do not fault, and C, D and E do.

**Open:**
- **Whether the value read is used.** For heads 0 to 10 the same read lands
  in the next head's first row, inside the buffer, where memcheck cannot see
  it. If the kernel uses that value, it would add a block to the last query
  tile. This also bears on mean-pool and the oracle, which call the same
  kernel.
- **The kernel source**, to confirm the read.

## Fallback (researcher's ruling, 2 Nov 2026)

If there is no root cause by 4 Nov, T3.1 runs with XAttention quarantined as
"not run, crash under investigation". The other arms and steps 5 to 8 then
bank, and XAttention runs in a session of its own once fixed. The cost is a
partial prediction file to reconcile.
