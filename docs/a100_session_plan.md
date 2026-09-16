# A100 session 1 — second architecture

Draft for approval. Nothing here has been run.

The point of this session is the **hardware-conditional claim**: every result
so far describes one card. Until a second architecture exists, "backend X wins
on an L4 but loses elsewhere" is a hypothesis with one data point.

---

## What is fixed before booking

| | |
|---|---|
| machine type | `a2-ultragpu-1g` — 12 vCPU, 170 GB, 1× A100 **80 GB** |
| zone | **`asia-southeast1-c` only** — re-verified 2026-09-16: `-a` and `-b` do not offer the shape |
| provisioning | **DWS Defined Duration first, Spot as fallback.** On-demand `NVIDIA_A100_80GB_GPUS` is 0 in every region; `PREEMPTIBLE_NVIDIA_A100_80GB_GPUS` is 1 in asia-southeast1 only |
| rate (verified 2026-09-16) | DWS Defined Duration GPU **$2.277654/h**; Spot GPU **$2.648100/h**; on-demand $4.846072/h. All-in Spot = GPU + 12 × $0.023340 core + 170 × $0.003128 RAM = **$3.4599/h ≈ ₹321/h** |
| quota | all gating metrics confirmed at 12/12/1 — see below |
| image | `attnbench-l4-image-v4-20260903`, **global**, `storageLocations: ['asia']` |

> **UNBLOCKED 2026-09-16 — read this before the BLOCKED note below.** The
> blocker described next was cleared by the 2026-09-05 rebuild and this
> document was never updated. `attnbench-env-v5-20260905` is a plain **disk**
> image (`gcloud compute images list`, not `machine-images list`): 200 GB,
> `storageLocations: ['asia']`, no `guestAccelerators` binding and no
> `disks[0].interface` to override, because a disk image cannot carry either.
> It is the image every L4 session since has booted from. No rebuild is
> needed.
>
> **The BLOCKED note is kept rather than deleted, because the failure it
> illustrates is not the image lock.** A document describing a *resolved*
> constraint reads exactly like one describing a *live* constraint — same
> confident tone, same specifics, same absence of any signal that the world
> moved. Acting on this page as written would have sent a session into an
> unnecessary image rebuild, at cost, to fix something already fixed. The
> only defence is to re-verify a blocker against the live API before honouring
> it, which is what happened here and is cheap: two read-only calls.
>
> Anything below this line dated 2026-09-05 or earlier is a claim about the
> world on that date, not a claim about the world now.

> **BLOCKED 2026-09-05.** The v4 machine image cannot boot an A2 at all: it
> records `guestAccelerators: nvidia-l4` (replaceable, never clearable) and
> `disks[0].interface: NVME` (not overridable), so it boots `g2` alone. Four
> creates confirmed this; none reached a capacity check. The session cannot
> run until a plain **disk** image exists — see
> `docs/machine_image_family_lock.md`. Capture it in the next G2 session, from
> the running instance, with `gcloud compute images create --force`.

Quota verified live, 2026-09-05: `A2_CPUS` 12, `PREEMPTIBLE_CPUS` 12,
`PREEMPTIBLE_NVIDIA_A100_80GB_GPUS` 1, `CPUS` 100, `CPUS_ALL_REGIONS` 32,
`GPUS_ALL_REGIONS` 1. Nothing is short.

**Spot pricing must be confirmed at launch, not assumed.** Every cost in this
document is provisional until the launcher prints the real rate and I state it
back. On-demand `a2-ultragpu-1g` is roughly 5× the L4's hourly rate, and Spot
is a discount on that — an unverified guess here is exactly the kind of number
that should not drive a booking.

---

## Pre-session work (off billed time, before anything boots)

1. **Launcher: fail clean on a single zone.** `gcp_launch_compile_session.sh`
   takes a zone fallback list and tries each in turn. Here there is exactly one
   zone, so the loop has nothing to retry into and looping is worse than
   useless — it turns one refusal into a slow one. It must attempt once,
   report the stockout, and exit non-zero. The existing preflight
   (refuse-if-an-instance-exists) stays.
2. **Capacity cannot be checked in advance.** There is no read-only API that
   answers "is there Spot A100 capacity in `asia-southeast1-c` right now"; the
   only test is an attempt, and an attempt that succeeds *is* the booking. So
   the honest sequence is: attempt, and if refused, stop and try later. Do not
   book five hours of attention against a resource whose availability is
   unknown until the moment it is created.
3. **~~A2 override flags.~~ FALSE — this is what blocked the session.** The
   claim that both properties "must be overridden at create time" and that this
   "is a flag change, not a rebuild" was wrong on both counts.
   `guestAccelerators` can be *replaced* but never *cleared*, and
   `disks[0].interface: NVME` cannot be overridden at all. What is needed is a
   plain disk image, captured with `gcloud compute images create --force` from
   a running G2 — see `docs/machine_image_family_lock.md`. Still not a rebuild,
   but not a flag change either.

---

## Session order — irreplaceable work first

Ordered so that a **Spot preemption costs the tail, not the head**. Each phase
checkpoints before the next begins.

### Phase 0 — clock lock (first, ~2 min)

`provenance.lock_clocks()` immediately after boot, before any measurement.

**This project has had root nowhere.** Both L4 sessions ran unlocked, and
`clocks_locked=True` has never appeared in a single result row. If it works
here, every A100 row is better-controlled than anything else in the dataset.

Record the outcome either way — it is a fact about Spot instances worth
knowing, and it decides how the cross-architecture comparison must be worded
(see *Deliverables*).

### Phase 1 — the wheel gate (~10 min, decides 3 h vs 6 h)

`flash-attn` and `Block-Sparse-Attention` were compiled on the L4, and
`docs/retry_session_runbook.md` records the flags actually used:
`FLASH_ATTN_CUDA_ARCHS=80;90` and `BLOCK_SPARSE_ATTN_CUDA_ARCHS=80;90`.
**sm_80 was in both builds**, so the gate is now expected to pass rather than
hoped to.

(An earlier draft of this plan said the wheels were "compiled for sm_89". That
was inferred from the card they were built on rather than read from the build
flags, and it is wrong.)

It still runs, and the 15-minute decision below still stands. A recorded build
flag is not a loaded kernel: `no kernel image is available for execution on the
device` passes every paper check and fails at first launch, which is the exact
shape this project keeps meeting. Image metadata cannot answer it; only an
import plus a forward can.

```
python3 -c "import torch; print(torch.cuda.get_device_capability())"   # expect (8, 0)
python3 -c "import flash_attn, block_sparse_attn"
# then a trivial forward on each, because import success does not imply the
# kernel loads -- the extension may import and fail at first launch
```

The forward matters as much as the import. A module that imports cleanly and
raises `no kernel image is available for execution on the device` on first
call is the exact silent-failure shape this project keeps meeting.

**Decision recorded in advance — do not improvise:**

| gate result | action |
|---|---|
| both import **and** run | proceed to Phase 2. Session is ~3 h. |
| either fails, **and elapsed < 15 min** | **rebuild in-session.** flash-attn ~2h12m + BSA ~50m = **3h02m**, then continue. Session is ~6 h. |
| either fails, **and elapsed ≥ 15 min** | tear down, rebuild in a dedicated session |

Rebuilding in-session is preferred at the 15-minute mark because the instance
is already up and single-zone Spot capacity is not guaranteed to exist
tomorrow. Past 15 minutes the arithmetic stops working against a 6 h cap.

Both builds have known guards from the compile-session runbook (`ptxas`
arch scoping, the `MAX_JOBS` memory ceiling). If the rebuild path is taken,
set `TORCH_CUDA_ARCH_LIST=8.0` — **not** `8.0;8.9`, which doubles compile time
for an architecture this session will never touch.

### Phase 2 — Stage 0 (~45–60 min)

Full length grid, `--max-seq 32768`. Expected to run **longer** than the L4's
Stage 0, not shorter: 80 GB means far fewer cells OOM early and cheaply, so
more of them actually execute.

Expected shape, from the L4's 1126 supported / 332 OOM at `fwd`: roughly
**1458 supported** on 80 GB. Attrition falling is the finding.

### Phase 3 — Stage 1 (~30–45 min)

Two things get *stronger* here, both consequences of memory:

**The float64 oracle is restored at three config classes** that the L4 could
only reach by cross-backend agreement:

| config | oracle needs | L4 (8.15 GiB budget) | A100 (29.6 GiB budget) |
|---|---|---|---|
| 2048 / batch 16 | 16.0 GiB | no | **fits** |
| 4096 / batch 4 | 16.0 GiB | no | **fits** |
| 8192 / batch 1 | 16.0 GiB | no | **fits** |

Those rows move from `check_kind="cross_backend"` to `"exact"` — a genuine
upgrade in verification strength, on the same kernels.

**Cross-backend at 32768/batch 16 stops being declined.** It needs 21.5 GiB,
which the L4's 12.12 GiB budget refused in production and the A100's
44.0 GiB accommodates.

**Done, in `da8ca90`.** The budgets are no longer absolute constants sized for
a 22 GiB card: they are fractions of `torch.cuda.get_device_properties(0)
.total_memory` (`ORACLE_BUDGET_FRACTION = 0.37`,
`CROSS_BACKEND_BUDGET_FRACTION = 0.55`), giving 8.15/12.12 GiB on an L4 and
29.6/44.0 GiB on an A100 — and every existing L4 verdict is asserted unchanged.
The budget that produced a verdict travels on the result row as
`memory_budget_bytes`, so a decline can be reproduced without knowing which
card ran it.

### Phase 4 — sweep (~55–65 min)

`--pass-kind fwd`, full length grid.

**`naive` must be capped.** At 2.8 TFLOP/s it accounts for **5.67 h of an
8.2 h** naive-inclusive estimate — because on 80 GB it stops OOMing at long
lengths and actually runs. Its timing is not a result; "materialising the
score matrix is slow" is not a finding. Cap it to the lengths the L4 also
measured, purely for comparability, and spend the memory on Phase 3 instead.

Excluding `naive` and `sdpa_math`: 1115 candidate cells, ~2.26 h at an assumed
2.5× L4 throughput; at the L4's 29% licensing rate, ~323 cells and ~0.65 h.
Licensing should be *higher* here because both verification paths fit better,
so plan **0.9–1.1 h**.

The 2.5× throughput assumption is the softest number in this plan. It is a
peak-FLOPs ratio, not a measurement, and Phase 4 replaces it with real data.

### Phase 5 — the three probes (~10 min, last)

Ordered by destructiveness. **cuDNN goes last, after everything is
checkpointed**, because an Xid 31 corrupts the CUDA context process-wide and
cannot be caught.

1. **Does `flex` lower block-sparse on sm_80?** The L4 fails at 114688 B
   required vs 101376 B available; an A100 has 164 KB per SM. If it lowers,
   the sparse arm gains real cross-backend verification at 8192 and 16384 on
   one architecture — which `docs/limitations.md` currently records as
   impossible on this hardware. Two configs.
2. **Does the float64 oracle reach further than modelled?** Phase 3 answers
   the modelled cases; this checks the boundary empirically at 16384/batch 1
   (**64 GiB** — predicted *not* to fit against a 29.6 GiB budget). A prediction that holds
   is worth as much as one that fails.
3. **Does the cuDNN fault reproduce on Ampere?** Three L4 confirmations exist,
   84 rows each, on three separate machines. `SDPABackend._CUDNN_FAULTS_ABOVE`
   excludes it above 8192; this deliberately lifts that for two configs at
   16384. **If sm_80 is clean, that is a strongly architecture-conditional
   result** — a shipping kernel that faults on Ada and not Ampere — and it is
   worth more than the two cells it costs. The exclusion must be lifted
   *narrowly and temporarily*, never by editing the capability default.

---

## Deliverables

- Stage 0/1/sweep tables at one commit, `git_dirty=False`, deployed by
  `gcp_deploy_source.sh` (bundle + instance-side commit verification).
- `clocks_locked` recorded on every row, whatever it turns out to be.
- The first **cross-architecture comparison** in the project:
  `compare_across_architectures` over L4 + A100 ratios, with
  `ArchitectureComparison.caveat()` on every flip.
- Three probe results, each recorded whether positive or negative.

**The clock asymmetry must be stated, not just flagged.** If Phase 0 succeeds,
this session produces locked-clock A100 ratios that will be compared against
unlocked-clock L4 ratios. Ratios are computed strictly within host, so the
comparison is *valid* — but the two sides are not measured to the same
precision, and `unlocked_architectures: ['NVIDIA L4']` will say so on every
comparison.

`ArchitectureComparison.asymmetrically_controlled` now distinguishes this case
from a uniformly-unlocked one and `caveat()` states it in words, because
`unlocked_architectures: ['NVIDIA L4']` on its own reads as "this comparison
is unlocked" and invites discounting both halves equally. The asymmetry — that
one half is better controlled than the other, and the difference between them
is therefore partly measurement quality — is the thing a reader cannot recover
from the number. The writeup should still say it plainly; the machinery is
there so that forgetting to is not silent.

This matters more than it looks because the canary measured roughly a **6%
host-to-host spread** across unlocked L4 hosts, which is larger than
`DRIFT_TOLERANCE` and the same order as the effects being claimed.

---

## What this session cannot do

- **No canary.** Different `gpu_name` means `cross_arch` treats the A100 as a
  second architecture, correctly — not as drift. There is no drift check here
  until a second A100 session exists, so **A100 numbers are single-sourced** in
  a way the L4 numbers no longer are. Accepted for session one, recorded as a
  limitation.
- **No `fwd_bwd`.** Unchanged: no backward correctness check exists, so nothing
  licenses those cells on any architecture.
- **No same-session repeat.** One instance, one pass. Run-to-run spread on the
  A100 is unmeasured.

---

## Abort conditions

Stop and tear down, do not debug on billed hardware:

- **Stockout.** Single zone, no fallback. Report and try later.
- **Wheel gate fails at ≥ 15 min elapsed** — see the table above.
- **Any step fails twice.** Standing rule, unchanged.
- **Preemption.** Not an abort: band-major checkpointing survives it, and
  `check_host_continuity` will refuse to resume into the same file on a new
  host — which is correct. The recovery path is a **new segment file**, joined
  at analysis time by `cross_arch`, never a resume into the old one.
- **Clock lock fails.** Not an abort. Record it and continue; the
  analysis-time flag was built for exactly this case.

## Budget

Provisional, pending the real Spot rate at launch:

| branch | duration | hard cap |
|---|---|---|
| wheels work | ~3 h | 240 min |
| rebuild in-session | ~6 h | 400 min |

The cap must be set for the branch actually taken. A 400-minute cap on the
3-hour path is a forgotten instance waiting to happen; a 240-minute cap on the
rebuild path kills the build at hour four.
