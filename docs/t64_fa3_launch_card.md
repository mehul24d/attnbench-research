# Launch card: the FA3 wheel build (T6.4)

Prepared 28 Oct 2026. Not launched. The researcher launches it; nothing here
runs on its own. No FA3 build has run on this project, so every time and cost
below is a bracket, not a measurement.

## What runs

`scripts/build_fa3_wheel.sh` on a CPU VM, from this branch
(`gcp_deploy_source.sh` bundles the local HEAD). In order: the results bucket,
the memory guard (`MAX_JOBS=8 x NVCC_THREADS=1 x 5 GB` against free RAM),
ninja, `nvcc` (12.3 or later) and the stack (Python 3.10, torch
2.9.1+cu129, the H100 image's), a write to the bucket, then the pinned source,
then `python3 setup.py bdist_wheel` in `hopper/`, stopped at 225 minutes if it
has not finished. The wheel, `manifest.json` (pin, CUTLASS commit, stack,
flags, wall time, peak memory) and `SHA256SUMS` go to
`<results bucket>/fa3/49774b31c8d1172a5b28f72e9aed0f322bf4ccba/`, and the
wheel is read back and hashed again. A failed or timed-out build stores its
logs under `failed-<UTC>/` there and exits 2.

| | |
|---|---|
| Source | `Dao-AILab/flash-attention` `v2.8.3.post2`, commit `49774b31c8d1172a5b28f72e9aed0f322bf4ccba` (`hopper/`, package `flash_attn_3` 3.0.0.b1). The latest release; `e9515d5d` has no documented build on this stack. In `scripts/_fa3_pin.sh` |
| Build flags | `FLASH_ATTENTION_FORCE_BUILD=TRUE` (never fetch a prebuilt wheel); `DISABLE_BACKWARD`, `DISABLE_FP8`, `DISABLE_SM80` (the backend is forward-only, bf16/fp16, sm_90a). All head dims kept |
| Toolchain | Upstream's `setup.py` downloads NVIDIA's nvcc 12.6.85 and ptxas 12.8.93 unless the local CUDA is 12.8. That is the upstream build, not a prebuilt wheel; the manifest records which toolchain ran |
| Then, on the H100 | `bash scripts/fa3_import_check.sh <sha256>`: hash against the recorded one, Python and torch against the manifest, the wheel's tag, `pip install --no-deps`, import, sm_90, `fa3` available. Not part of this session |

## The instance

| | |
|---|---|
| Machine | `n2-standard-16`, 16 vCPU / 64 GB, no accelerator |
| Zone | `us-central1-a` |
| Provisioning | STANDARD (on-demand) |
| Image | `attnbench-env-v5-20260905` (the H100's; a plain disk image, so it boots N2) |
| Rate | ₹77.21/h, all-in planning rate (below). Teardown needs `GCP_RATE_INR_HR=77.21` |
| Quota | N2_CPUS 200; CPUS_ALL_REGIONS 32, so no GPU VM may run at the same time (16 + 26 = 42) |
| Launch window | by 9 Dec 2026 (`_launch_policy.sh` enforces it) |

The rate, from the 20 Oct 2026 catalogue (`logs/t017_20261020/skus_USD.jsonl`,
us-central1) at the plan's ₹95.99/$: N2 core `BB77-5FDA-69D9` $0.031611 x 16
plus N2 RAM `5B01-D157-A097` $0.004237 x 64 GiB = ₹74.58/h, plus a 200 GB
pd-balanced boot disk (`6AE1-525F-8B80`, $0.10/GiB-month over 730 h) = ₹2.63/h.

## Time and cost

The only build on file is FA2's: 73 ninja edges in 2 h 12 min at 4 jobs, about
7 minutes per edge per job. This build has about 160 `sm_90a` forward
instantiations plus 3 other units at 8 jobs. At 5 to 10 minutes per unit that
is about 1.7 to 3.3 hours.

| | Hours | ₹ at 77.21/h |
|---|---|---|
| Setup: create, deploy, ninja and toolchain checks, source and CUTLASS fetch | 0.35 | 27 |
| Build, bracket | 1.7 to 3.3 | 131 to 255 |
| Store, read back, teardown, cleanup check | 0.15 | 12 |
| **Expected** | **2.2 to 3.8** | **170 to 293** |
| Build deadline (225 min after the build starts): logs stored, no wheel | about 4.1 | 317 |
| In-guest halt (+270 min from boot): compute billing stops | 4.5 | 347 |
| Hard delete (`--max-run-duration=5h`, DELETE) | 5.0 | **386, the cap** |

**Cost cap: ₹386** (5 h at the planning rate). If the deadline stops the build,
there is no wheel, the H100 claims are stated as upper bounds with FA3
excluded and the reason named (the pre-registered fallback), and a second
build is a new decision, not a retry.

## The commands

On the Mac, from the repo root, with `t64-h100-launch-2026-10-27` checked out:

```bash
export ATTNBENCH_GCP_PROJECT=...          # this study's project (never the gcloud default)
export CLOUDSDK_CORE_PROJECT="$ATTNBENCH_GCP_PROJECT"
export GCP_RATE_INR_HR=77.21
NAME=attnbench-fa3-build-$(date +%Y%m%d-%H%M); ZONE=us-central1-a

bash scripts/gcp_cleanup_check.sh                       # expect "Clean": no GPU VM running
bash scripts/gcp_launch_fa3_build.sh "$NAME"             # type 'launch'
bash scripts/gcp_deploy_source.sh "$NAME" "$ZONE"
gcloud compute ssh "$NAME" --zone "$ZONE" --project "$ATTNBENCH_GCP_PROJECT"
```

On the instance, from `~/attnbench_scaffold` (`nohup`, so a dropped SSH does not
kill the build):

```bash
command -v ninja || pip install --user ninja
python3 -c 'import psutil' || pip install --user psutil
nohup bash scripts/build_fa3_wheel.sh > ~/fa3_build.out 2>&1 &
tail -f ~/fa3_build.out                                  # START ... END rc=0, then the sha256
```

Back on the Mac, after `FA3 wheel stored and read back.` (or exit 2):

```bash
bash scripts/gcp_session_elapsed.sh "$NAME" "$ZONE" 77.21
bash scripts/gcp_teardown_session.sh "$NAME" "$ZONE"
bash scripts/gcp_cleanup_check.sh
```

Record the sha256, the wall time and the cost in `docs/spend_ledger.md`; the
H100 session passes the sha256 to `fa3_import_check.sh`.

## Stop rules

- Any refusal before the source fetch (exit 1): tear down. Nothing was built,
  and the message names the check. A stack mismatch means the image is not
  the H100's; do not override the expectation.
- The bucket write fails: tear down. The scopes or the bucket are wrong.
- `upstream v2.8.3.post2 now points at ...`: tear down and ask. The tag moved.
- Exit 2 (build failed or deadline): the logs are in the bucket; tear down.
- Exit 3 (read-back differs): tear down; the wheel in the bucket is not used.

## Known before launch

- **N2 on this image is unobserved.** The disk image boots any family by
  design (`docs/machine_image_family_lock.md`), but no N2 VM has booted it.
  If the VM does not come up, tear down; the cost is minutes.
- **nvcc's path and version on the image are not on record.** The script
  finds it from `CUDA_HOME`, then `PATH`, then `/usr/local/cuda`, and refuses
  below 12.3.
- **FA3's memory per compile unit is not measured.** The guard uses FA2's
  5 GB; 8 jobs leave room on 64 GB if FA3 needs up to about 7 GB. The build
  logs peak memory and the largest `cicc` every 15 s (`memory.tsv`).
- **Whether the upstream toolchain download works from the VM** is not
  known; it needs outbound HTTPS to `developer.download.nvidia.com`, as the
  source fetch needs GitHub.
- **Capacity for `n2-standard-16` in `us-central1-a`** cannot be checked in
  advance. The launcher tries one zone.
