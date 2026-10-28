# T6.4: the FA3 backend and wheel build

Written 27 Oct 2026 as a plan. The three decisions were taken on 28 Oct 2026 and the
backend, build script, import check and launcher are built (Section 8); nothing is launched.
The H100 launch path is built (`scripts/gcp_launch_h100.sh`, `scripts/_flex_request.sh`,
`tests/test_flex_request.py`). This note plans the other half of the H100 work:
FlashAttention-3 (FA3), which the plan names as a dense-baseline candidate on the
H100 (Section 5.5, control K2).

Every number is sourced. Where a figure is an estimate or has not been seen on this
project, it says so. Three decisions are the researcher's and are listed at the end.

## 1. Why FA3 is needed

K2 picks the fastest *correct* dense kernel per card, context length and batch.
The H100 candidates are SDPA flash, SDPA cuDNN, SDPA memory-efficient, FA2 and FA3.
FA3 is the Hopper-native kernel, so without it an H100 speedup is measured against a
possibly weaker baseline. The archived frontier pre-registration put the same point as
"sparse savings would be overstated" and resolved it as "FA3 included, never cut; a
failed build makes H100 claims upper bounds" (`docs/archive/estimator_frontier_preregistration.md`,
the K2 text and its resolved-objection row 8). `docs/limitations.md` ("FlashAttention-3 is
deliberately excluded", 12 Sep 2026) records the older decision to leave it out and
what adding it would take. That note predates the plan; the plan's K2 text now
includes FA3, and the plan wins.

## 2. The backend (small, CPU-testable)

A new `attnbench/backends/fa3.py`, written like `FlashAttention2` in `impls.py`:

- `@register`, name `fa3`, family `dense_exact`, `min_compute_capability=(9, 0)`,
  `supports_gqa=True`. Forward only. The head dimensions it declares come from the
  pinned source, not from memory.
- `_import_check` imports the module the Hopper build installs. The upstream README
  says it imports as `flash_attn_interface` (the `hopper/` directory), and a mirror
  shows `from flash_attn_3 import flash_attn_interface`. Which one the pinned commit
  installs is read from its `setup.py`, not assumed.
- `forward` transposes (B, H, S, D) to (B, S, H, D) as FA2 does, calls the kernel with
  `causal` set from the config, and unpacks the result. Some FA3 versions return
  `(out, softmax_lse)`; the pinned commit decides. A wrong unpack passes shape checks on
  one version and fails on another, so the test uses a stand-in with both return shapes.
- Sliding-window masks: declared unsupported unless the pinned source supports a window
  argument in the form the harness uses.

Tests, all CPU, none needing an H100: the registry holds `fa3` and
`tests/test_backend_capability_declarations.py` still passes; `is_available()` is False
without CUDA 9.0; a recorder stand-in for `flash_attn_interface` receives the transposed
tensors and the causal flag; both return shapes unpack; an unsupported config raises
`UnsupportedConfig` with the reason. The stand-in pattern is the one used for
`bsa_prefill` and the XAttention backend.

Gate work that needs the H100 and so waits for a session: the float64 correctness check
of `attnbench/gates.py` on fixed inputs (K2's "runs correctly"), the capability rows, and
the exclusion rules. The correctness tolerance is fixed at lock, as K2 says.

The `fa3` name also has to enter the places that list backends by name (the dense-baseline
selection and the exclusion rules). Those call sites are listed by grep when the work
starts, not from memory.

No source is copied (rule 8). FA3 is installed and called. It is BSD-3. Whether it needs
a row in `docs/vendor_register.md` is the register's own rule: a row is for reproduced
lines, and none are reproduced. A credit line in the README is still owed.

## 3. The wheel

**What is built.** `Dao-AILab/flash-attention`, `hopper/`, at one pinned commit, as a wheel.
It needs an H100 or H800 to run and CUDA 12.3 or later to build (12.8 recommended
upstream). The study image has CUDA 12.9 (torch 2.9.1+cu129, Python 3.10.12). A build does
not need a GPU, only `nvcc`, so it can run on a CPU machine.

**The pin.** The archived pre-registration named `main` at `e9515d5d` on 3 Oct 2026 and said
"pinned at lock". There is no lock now. The commit is chosen when the build is
approved, written down before the build, and recorded on every H100 row that used the
wheel (commit and wheel SHA-256). The choice is the researcher's (decision 1).

**ABI.** The wheel must match the H100 image: Python 3.10, torch 2.9.1+cu129. The surest
way is to build on a VM booted from the same plain disk image
(`attnbench-env-v5-20260905` or a later one). A plain disk image carries no accelerator, so
it boots a CPU family (`docs/machine_image_family_lock.md`). The build preflight checks
`nvcc --version`, torch, and Python against the H100 image before compiling, and the wheel's
tags are compared against the H100 instance before it is installed. (A frontier-branch pull
script did the second check; it is not on `main`, so it is rebuilt.)

**Memory.** `attnbench/build_guards.py` refuses a build whose `MAX_JOBS x NVCC_THREADS x 5 GB`
exceeds free RAM. The 5 GB figure was measured on FA2 (OOM kills on 2 Sep 2026). FA3 has not
been measured here, so the guard's figure stays until a first build logs peak RSS. Upstream
also warns that under 96 GB of RAM with many cores, ninja may run too many jobs and exhaust
memory, and says to cap `MAX_JOBS`.

**Build time is not measured.** The only figures on file are the archived pre-registration's
60 to 150 minutes on a CPU VM, and the FA2 build on this project (2 h 12 min, 73 ninja edges,
`sm80;sm90`, `MAX_JOBS=4`, in `docs/retry_session_runbook.md`, on an L4 compile VM; the launcher's default there is 8 vCPUs). FA3 compile cost per file is different, so these are
brackets, not predictions. The first build logs wall time per edge and peak memory, and the
cap below is sized to bound a bad guess.

## 4. Machine, quota and cost options

Quota facts are from `logs/t017_20261019/quotas.json` (saved 19 Oct 2026, read-only export).
Rates are the Cloud Billing catalogue rows saved on 20 Oct 2026 (`logs/t017_20261020/skus_USD.jsonl`),
converted at the plan's own USD-to-INR ratio (Rs 462.42 / $4.8174 = Rs 95.99 per dollar).
Compute only: cores plus RAM. The 200 GB boot disk is added to a session's planning rate as
for the other cards; its line for these machines is not on file and is read from the
catalogue before a launch.

What binds. In `us-central1`, `asia-southeast1` and `asia-northeast1`:

| Quota line | Value | Meaning for a build VM |
|---|---:|---|
| `CPUS_ALL_REGIONS` | 32 | The hard ceiling. A 32-vCPU VM uses all of it, so nothing else runs. |
| `N2_CPUS` | 200 (all three regions) | Not binding. |
| `C3_CPUS` | 24 | Allows up to `c3-standard-22`. |
| `C2_CPUS` | 8 | The archived choice, `c2-standard-30`, is short by 22. No request was filed. |
| `C2D_CPUS` | 100 | Not binding, but the all-regions line still caps at 32. |
| `PREEMPTIBLE_CPUS` | 26 in `us-central1` and `asia-southeast1` | A Spot VM may use at most 26. |
| `GPUS_ALL_REGIONS` | 1 | The H100 uses it. A CPU VM does not. |

The FA3 build and any GPU VM are never concurrent: 32 + 26 (H100) or + 12 (A100) exceeds 32.

| Option | vCPU / RAM | Rs per hour, compute only | 1 to 2.5 h | Quota | Notes |
|---|---|---:|---:|---|---|
| **A. `n2-standard-32`, on-demand, `us-central1`** | 32 / 128 GB | 149 | 149 to 373 | N2 fine; uses the full 32 | Recommended. Most parallelism; `MAX_JOBS` up to about 24 under the memory guard. |
| A2. same, `asia-southeast1` | 32 / 128 GB | 184 | 184 to 460 | same | Dearer. No reason to prefer it. |
| B. `n2-standard-16`, on-demand | 16 / 64 GB | 75 | 75 to 187 at the same wall time | N2 fine | Half the parallelism, so about twice the hours. Cost is roughly flat; wall time doubles. Only useful if 32 vCPUs cannot be had. |
| B2. `n2-standard-16`, Spot, `us-central1` | 16 / 64 GB | 45 | 45 to 112 | 16 of 26 preemptible | Cheapest. A preemption restarts the build. Not recommended for a build the H100 session waits on. |
| C. `c3-standard-22`, on-demand | 22 / 88 GB | 106 | 106 to 266 | C3 24 | Faster cores, fewer of them. Unmeasured whether that wins. |
| D. `c2-standard-30` (archived choice) | 30 / 120 GB | not priced | | C2 8: needs a quota request | Dropped unless the request is made and granted. |
| E. Build on the H100 VM during its session | 26 / 234 GB | 462 | 462 to 1,156 | none extra | Spends the most expensive hour on a compile and lengthens a session capped at 5 h. Fallback only. |
| F. Build on the A100 VM | 12 / 170 GB | 297 | | | Compiles `sm_90a` without a GPU, but with 12 cores and at A100 prices. No. |
| G. On the Mac | | 0 | | | Not possible: arm64 macOS, no CUDA toolchain for Linux x86. |

For comparison, the H100 session itself is Rs 462.42 per hour in `us-central1`: Rs 1,618 for the
planned 3 h 30 min and Rs 2,312 at the 5 h cap (header of `scripts/gcp_launch_h100.sh`).

**Recommendation: option A with a 3 h hard cap.** Expected Rs 149 to 373 for 1 to 2.5 h, ceiling
Rs 448 at 3 h, plus the disk. That is about a fifth of one H100 session and keeps the GPU hour
for measuring. It matches the archived estimate (Rs 130 to 325 at the then rate), re-priced at
the 20 Oct catalogue. If the first build is cut off by the cap, the wheel is not usable and the
fallback is the pre-registered one: the H100 claims are stated as upper bounds, with FA3 excluded
and the reason named. A second build attempt is then a new decision, not an automatic retry.

## 5. What would be built (in order, when approved)

1. `scripts/gcp_launch_fa3_build.sh`: a CPU launcher on the same pattern as the others. It
   sources the launch policy and the project pin, checks the date before the create, requires
   a declared rate, uses on-demand (not Spot), sets `--max-run-duration=3h` with action DELETE,
   and an in-guest halt well short of it. It passes the existing wiring tests, which find any
   script that creates an instance.
2. `scripts/build_fa3_wheel.sh`: the memory guard first, `ninja` on `PATH`, the pinned commit
   checked out and verified, `nvcc` and torch checked against the H100 image, then
   `python setup.py bdist_wheel` in `hopper/`. It logs wall time per edge and peak memory.
3. The wheel and its SHA-256 are written to the results bucket, with the commit, before the
   VM is deleted. The VM is torn down through `scripts/gcp_teardown_session.sh`, and the cost
   goes in `docs/spend_ledger.md`.
4. `scripts/gcp_bootstrap_instance.sh` gains a step that fetches the wheel, checks its hash
   and its tags against the instance, and installs it. Without a wheel the step is skipped
   and the run records `dense_set_incomplete="fa3:<reason>"`.
5. `attnbench/backends/fa3.py` and its tests (Section 2).
6. After the H100 session: the correctness gate result, then K2's selection.

Order against the H100 request: the build comes first and needs its own approved session
(plan Section 6, cloud sessions). It must start by 9 Dec 2026 India time, like any launch,
and its spend counts toward the Rs 14,000 through 30 Nov 2026.

## 6. What is not known

- The FA3 build time and peak memory on this stack (first build measures both).
- The exact import name, return shape and window support of the pinned commit.
- Whether the `n2-standard-32` is available in the pinned zone when asked. Capacity cannot be
  checked in advance.
- Whether the wheel's tags match the H100 instance. Checked at install, not before.
- Whether any prebuilt FA3 wheel exists. None was found in a short search; if one does, it is
  a third-party binary and a separate decision (decision 3).

## 7. Decisions for the researcher

1. **Which commit to pin** (a recent `main` commit at approval time, or the archived
   `e9515d5d`).
2. **Which machine**: option A (recommended), or B or C if 32 vCPUs are not wanted.
3. **Whether a prebuilt third-party FA3 wheel is acceptable** if one is found, in place of a
   build. This note assumes not.

## 8. Decisions taken and what was built (28 Oct 2026)

1. **Pin:** `e9515d5d` only with a documented successful build on this stack, else the
   latest tagged release, never `main`. No build of `e9515d5d` is documented (it is only
   named, as `main` that day, in the archived pre-registration). The latest release is
   `v2.8.3.post2`, commit `49774b31c8d1172a5b28f72e9aed0f322bf4ccba` (GitHub's latest
   release; the `fa4-*` tags are pre-releases of another package). `hopper/` there is
   `flash_attn_3` 3.0.0.b1: it installs `flash_attn_interface`, whose `flash_attn_func`
   returns `out` alone and takes `window_size=(left, right)`; head dims 64, 96, 128, 192, 256.
   Recorded in `scripts/_fa3_pin.sh`.
2. **Machine:** option B, `n2-standard-16`, on-demand, `us-central1-a`. Planning rate
   ₹77.21/h all-in; hard cap 5 h (₹386); in-guest halt at 270 min; the build stops itself
   at 225 min.
3. **No third-party wheel.** The build sets `FLASH_ATTENTION_FORCE_BUILD=TRUE`, without
   which upstream's `bdist_wheel` first tries to download a prebuilt wheel, and fails if
   its log shows the attempt. Upstream's own download of NVIDIA's nvcc/ptxas is kept.

Built: `attnbench/backends/fa3.py` (forward only, bf16/fp16, sm_90; tests with a
stand-in that computes attention, six mutants watched to fail), `attnbench/fa3_wheel.py`,
`scripts/build_fa3_wheel.sh` (drops backward, FP8 and the sm_80 kernels, which the
backend declares unsupported), `scripts/fa3_import_check.sh` (H100),
`scripts/gcp_launch_fa3_build.sh`, all wired against a fake `gcloud`
(`tests/test_fa3_backend.py`, `tests/test_fa3_wheel_build.py`). Launch card:
`docs/t64_fa3_launch_card.md`. Not done: the bootstrap step of Section 5 item 4 (the
import check is a separate script), the README credit line, and the
`docs/limitations.md` FA3 entry.

## 9. First build and the restricted build (28 Oct 2026)

The first build (`attnbench-fa3-build-20261028`, 137 min, ₹176) was killed for memory
12 minutes in, on `flash_fwd_hdimall_*_sm90` units at 8 jobs on 64 GB, and its VM sat idle
for two hours because the Mac-side watcher called `timeout`, which macOS lacks.

Researcher's decisions the same day: a **restricted build** (head dim 128, forward, bf16 and
fp16; also off: paged KV, append KV, split KV, softcap, local attention, FP8, backward, and
the sm_80 kernels), each flag checked against the pinned `setup.py` and recorded in the
manifest and in `code_identity` (`fa3.BUILD_FLAGS`, `fa3.BUILT_HEAD_DIMS`); 15 GB per job,
1 s memory samples, a 3 h cap, halt at 150 minutes, a 120-minute build deadline; the build
halts its VM when it ends; a watcher with no `timeout` (`scripts/gcp_watch_fa3_build.sh`).
The harness uses none of what is removed: every config is head dim 128, no config uses a
sliding window, FP8, paged or appended KV, softcap or `num_splits`, and the backend declined
backward already. Split KV would not engage on the study's causal prefill cells (Section on
FA3 in `docs/limitations.md`). With a restricted wheel a head dim of 64 runs the padded
hdim-128 kernel instead of failing (`flash_api.cpp`), so the backend refuses it itself.
**MAX_JOBS is 3, not 4:** at 15 GB per job, 4 jobs need 60 GB and about 58 are usable on
`n2-standard-16`, so the guard refuses 4. Launch card: `docs/t64_fa3_launch_card.md`.
