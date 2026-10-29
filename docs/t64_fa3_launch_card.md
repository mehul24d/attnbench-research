# Launch card: the FA3 wheel build, second attempt (T6.4)

Prepared 28 Oct 2026. Not launched. The researcher launches it; nothing here
runs on its own.

**The first attempt** (28 Oct 2026, 137 minutes, ₹176, `docs/spend_ledger.md`)
built the full wheel at 8 jobs and was killed for memory 12 minutes in: three
`flash_fwd_hdimall_*_sm90` units, each of which compiles every head dim in one
file. The VM then sat idle for two hours because the Mac-side watcher called
`timeout`, which macOS lacks. This card's changes answer both problems:

| | First attempt | This attempt |
|---|---|---|
| What is built | every head dim, split, paged, softcap, local | **head dim 128, forward, bf16 and fp16** (`fa3.BUILD_FLAGS`; researcher's decision) |
| Compile units | 43 | about 10: 8 forward units (`hdimall`/`hdimdiff` x bf16/fp16 x PackGQA on/off) with only hdim 128 left in them, `flash_api.cpp`, the scheduler |
| Memory guard | 5 GB per job, 8 jobs | **15 GB per job, 3 jobs** (45 of about 58 GB usable; 4 jobs would need 60 and the guard refuses) |
| Memory samples | 15 s (and the sampler died) | **1 s**, sampler fixed |
| When the build ends | the VM kept running | logs stored, then **the VM halts itself**, on success and on failure |
| Watcher | scratchpad loop calling `timeout` | `scripts/gcp_watch_fa3_build.sh`: no `timeout`, every failed check reported, three in a row stop it, verdict from the bucket |

## What runs

`scripts/build_fa3_wheel.sh` on a CPU VM, deployed from this branch's commit
(`gcp_deploy_source.sh` bundles the local HEAD). Before any compile: the
bucket, the memory guard, ninja, nvcc 12.3 or later and the H100 image's stack
(Python 3.10, torch 2.9.1+cu129), a write to the bucket, the pinned source
(`v2.8.3.post2`, commit `49774b31c8d1172a5b28f72e9aed0f322bf4ccba`), and a
check that each of the 13 flags is read by the pinned `hopper/setup.py`. Then
`python3 setup.py bdist_wheel`, stopped at **120 minutes** if not done. The
wheel, `manifest.json` (pin, CUTLASS commit, the flags, head dim 128, stack,
jobs, wall time, memory at 1 s) and `SHA256SUMS` go to
`<results bucket>/fa3/49774b31…/`, and the wheel is read back and hashed. A
failed build stores its logs under `failed-<UTC>/`. Either way the VM then
runs `shutdown -h +1`.

## The instance

| | |
|---|---|
| Machine | `n2-standard-16`, 16 vCPU / 64 GB, no accelerator |
| Zone | `us-central1-a`, on-demand |
| Image | `attnbench-env-v5-20260905` (the H100's; booted N2 on 28 Oct: Python 3.10.12, torch 2.9.1+cu129, nvcc 12.9 at `/usr/local/cuda`, no ninja) |
| Rate | ₹77.21/h, all-in planning rate. Teardown needs `GCP_RATE_INR_HR=77.21` |
| Caps | in-guest halt at **150 min**, `--max-run-duration=3h` DELETE: **₹232 at most** |
| Quota | CPUS_ALL_REGIONS 32: no GPU VM at the same time |
| Launch window | by 9 Dec 2026 |

## Time and cost

Not measured for this build. On 28 Oct, 10 of 43 full-build units finished
in 12 minutes at 8 jobs; the restricted units carry one head dim of five.

| | Hours | ₹ at 77.21/h |
|---|---|---|
| Setup: create, deploy, ninja, checks, source and CUTLASS fetch | 0.35 | 27 |
| Build, bracket: 8 heavy units in 3 waves of 10 to 30 minutes | 0.5 to 1.5 | 39 to 116 |
| Store, read back, self-halt; teardown and cleanup check | 0.15 | 12 |
| **Expected** | **1.0 to 2.0** | **77 to 154** |
| Build deadline (120 min after the build starts): logs stored, VM halts | about 2.5 | 193 |
| In-guest halt (+150 min from boot) | 2.5 | 193 |
| Hard delete (3 h) | 3.0 | **232, the cap** |

If the deadline or memory stops the build, there is no wheel, and a third
attempt is a new decision.

## The commands

On the Mac, from the repo root, with the branch (or `main` after its merge)
checked out and clean:

```bash
export ATTNBENCH_GCP_PROJECT=...          # this study's project (never the gcloud default)
export CLOUDSDK_CORE_PROJECT="$ATTNBENCH_GCP_PROJECT"
export GCP_RATE_INR_HR=77.21
NAME=attnbench-fa3-build-$(date +%Y%m%d-%H%M); ZONE=us-central1-a

bash scripts/gcp_cleanup_check.sh                       # expect no compute running
bash scripts/gcp_launch_fa3_build.sh "$NAME"             # type 'launch'
bash scripts/gcp_deploy_source.sh "$NAME" "$ZONE"
gcloud compute ssh "$NAME" --zone "$ZONE" --project "$ATTNBENCH_GCP_PROJECT" --command '
  cd ~/attnbench_scaffold && pip install --user -q ninja &&
  nohup bash scripts/build_fa3_wheel.sh > ~/fa3_build.out 2>&1 < /dev/null &
  sleep 60; cat ~/fa3_build.out'                          # expect START ... jobs=3
bash scripts/gcp_watch_fa3_build.sh "$NAME" "$ZONE"      # until it exits; 0 is a wheel
bash scripts/gcp_teardown_session.sh "$NAME" "$ZONE"     # the VM has halted; ssh steps may warn
bash scripts/gcp_cleanup_check.sh
```

Then: the cost file goes in `docs/spend_ledger.md`; the sha256 from
`manifest.json` is recorded and later passed as `FA3_WHEEL_SHA256` to
`gcp_bootstrap_instance.sh` on the H100.

## Stop rules

- A refusal before the build (exit 1): the VM does not halt itself. Tear down,
  unless the message names something fixed in a minute on the VM (ninja).
  A stack mismatch or a flag not read by `setup.py`: tear down and ask.
- The watcher exits 3 (cannot see the VM): look by hand at once; tear down if
  in doubt. Exit 4 (still running after 100 polls, about 3.3 h) should not
  happen before the 3 h delete.
- Exit 2 or 3 from the build: the VM halts itself; tear down. Do not retry.

## Known before launch

- **Whether 15 GB per job is enough** is the researcher's assumption; the
  1 s samples will say. Three `cicc` processes killed at 8 jobs on 64 GB give
  only a floor (above about 7 GB each for an all-head-dim unit).
- **Whether a halted VM's serial log can be fetched** by the teardown is not
  observed; the teardown continues to the delete either way.
- **Capacity** for `n2-standard-16` in `us-central1-a` was there on 28 Oct;
  it is not checked in advance.
