# Launch card: AWS environment build on `m7a.xlarge` (Sydney)

Prepared 21 Nov 2026 (adviser's rulings dated 21 Nov 2026, third wave, item 4);
revised the same day (fifth wave, items 1 to 5).
**Launch yes given 21 Nov 2026 for one attempt, after the pre-launch pass**
(`docs/permission_log.md`): cap ₹250, halt at 360 min, delete at 375 min.
It times nothing and produces no reported number. It makes the AMI that P3's
rehearsal, P3 and P6 boot from.

## What it is

One CPU-only on-demand instance, `scripts/aws_launch_session.sh ENV`, which
gives it every guarantee of the GPU sessions: launch policy (rule 12: Sydney
`ap-southeast-2` and its availability zones only), pinned account, required
rate, cap and hard-delete limit, worst case under the cap, on-demand only,
in-guest halt (stop), hard delete by EventBridge Scheduler, encrypted 200 GB
gp3 root deleted with the instance, IMDSv2, session tags.

| Setting | Value |
|---|---|
| Region / zone | `ap-southeast-2` / `ap-southeast-2a` |
| Type | `m7a.xlarge` (4 vCPU, 16 GiB, AMD EPYC 4th gen; no GPU, so no G or P quota is drawn) |
| Base image | the AWS Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04), latest in Sydney: `ami-0db4f29a6aa4336c4` (release 20261113, root `/dev/sda1`), read with `describe-images` on 22 Nov 2026. Its description lists no CUDA toolkits and the release-notes page did not load, so the toolkit is checked by step 0 (nvcc 12.x) on the instance. **Plan default, flagged** |
| Rate (`AWS_RATE_INR_HR`) | ₹34.90/h: (instance $0.2898/h, Sydney public price file published 20 Nov 2026, + 200 GB gp3 at an assumed $0.096/GB-month, not checked) × 1.15 × ₹95.99/$ (the method of plan Section 5.2) |
| Planned hours | 4.5 (the build estimate below) |
| Halt (`AWS_HALT_MINUTES`) | 360 min (the ruling), instance stops |
| Hard delete (`AWS_MAX_RUN_MINUTES`) | 375 min (the ruling) |
| Cap (`AWS_COST_CAP_INR`) | ₹250 (the ruling); worst case ₹34.90 × 375 min = ₹219 |
| Expected | about ₹157 (4.5 h) |

## Steps on the instance

1. Deploy the committed HEAD as a git bundle (as on GCP); a clean tree is required. Copy up the FA3 wheel (`flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl`, sha256 `1676ef0b86c377ae…`, local in `attnbench_backups`).
2. to 6. One script, run detached (`nohup`, so a dropped SSH session does not end it): `FA3_WHEEL_SHA256=1676ef0b86c377ae911a7de238c2a1f6b58a7666f1a6eef919d4dc7b0973b64a MAX_JOBS=2 bash scripts/aws_env_build_instance.sh ~/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl` (added 21 Nov 2026, fourth wave, item 3). It stops at the first failure and prints `ENV_BUILD_OK` only when every step passed:
   - **0, guards:** a clean tree, Python 3.10.12, a new record directory (`~/env_build_<UTC>`, never overwritten), the wheel and its sha256, the FA3 manifest at its recorded sha256, nvcc 12.x (`/usr/local/cuda-12.9` when present, v6's), flash-attn's memory guard at `MAX_JOBS`, then `cuobjdump` in the probe's search order (below). 16 GiB admits 2 jobs, not 4, so 4 is refused in seconds.
   - **2, v6's packages** (`docs/image_libraries/attnbench-env-v6-20260917/`, the environment P1, P2 and P4 ran), from an index at exact versions with `--no-deps`. First the system list as root: less the 69 packages apt installed on v6 (they come with the base image), and with `--ignore-installed`, so v6's version shadows an apt copy at another version. Then the session user's site. torch and triton must then read `2.9.1+cu129` and `3.5.1`. **Plan defaults, flagged:**
     - The system Python and the user site, as v6, not a venv.
     - xformers' local build (`0.0.33+5d4b92a5.d20251029`) has no index copy, so PyPI's `0.0.33` stands in. xformers is no arm and is not in the library check.
   - **3, FA3:** the full `fa3_wheel.check_wheel` against the build's manifest (sha256, wheel name, the build's Python 3.10.12 and torch 2.9.1+cu129, tags), then it is installed. It is **not imported** here (no GPU); P6's `fa3_import` step does that. The manifest (1,462 bytes) was read from the GCP bucket on 21 Nov 2026 (fifth wave, item 3), is byte-identical to the build's local copy, and is committed as `configs/t64/fa3_manifest_49774b31.json` (sha256 `3c2f9657…`). The real wheel passes the check on the Mac with v6's stack and cp310 Linux tags; another torch is refused.
   - **4, from source:**
     - flash-attn `2.8.3.post1` (`scripts/build_flash_attn.sh`), compiled with 2 jobs: no official wheel exists for torch 2.9. The `v2.8.3.post1` GitHub release has cp310 wheels for torch 2.4 to 2.8 only (read 21 Nov 2026), and torch's C++ ABI differs between minor versions;
     - block-sparse-attention at `bsa_contract.BSA_COMMIT`;
     - x-attention at `XATTN_COMMIT`, editable and outside the tree.

     **Architectures** (fifth wave, item 2). Two arms import `flash_attn` on the L4: the `fa2` arm (`impls.FlashAttention2`) and XAttention's threshold profiler, whose pinned module imports it at load (`xattention._official_profiler`). So the L4 must be covered. `TORCH_CUDA_ARCH_LIST=8.9;9.0` is set as ruled, for any extension that reads it. flash-attn and block-sparse-attention do not read it. Their pinned `setup.py` read `FLASH_ATTN_CUDA_ARCHS` and `BLOCK_SPARSE_ATTN_CUDA_ARCHS` and know only 80, 90, 100, 110 and 120; an `89` is dropped without a word. The L4 (sm_89) runs their sm_80 code by Ampere minor-version binary compatibility, which is how P1 ran on v6. So both are set to `80;90`. That is v6's setting, and here it is what covers the L4, so it is not a revert. **Flagged against the ruling's wording.** Block-sparse-attention's default would also compile 100, 110 and 120. After the builds, `cuobjdump --list-elf` must show exactly sm_80 and sm_90 in each library (`cuda_archs.txt` in the record). x-attention's import is left to the GPU rehearsal.
   - **5, CPU smoke:**
     - `attnbench` imports, and so do `flash_attn` and `block_sparse_attn` without a GPU or driver (they link the CUDA runtime, not the driver; unchecked until the instance runs it, and the error is logged if not);
     - this environment's libraries equal P2's (`configs/t64/p2_libraries.json`), with FA3 the one allowed difference, as P4's preflight requires;
     - `tests/test_build_guards.py` and `tests/test_env_build_lists.py` pass.
   - **6, the record:** `pip freeze` (user, user site, root), `dpkg -l`, `uname`, each list diffed against v6's, and `SHA256SUMS`. The tree must still be clean.
7. Stop the instance; `ec2 create-image`; wait until `available`; read back the AMI and its snapshot.
8. Teardown (`scripts/aws_teardown_session.sh`, `scripts/aws_cleanup_check.sh`): instance terminated, no volume left. The AMI is kept.

## Attempts

**One attempt on `m7a`** (fifth wave, item 5; replaces the two of the fourth wave; a second ruled 22 Nov 2026 after the probe, below): cap ₹250, halt at 360 min, delete at 375 min. If it does not end in `ENV_BUILD_OK` and an available AMI, stop and report; a second attempt needs a new ruling.

**Attempt rule (adviser's rulings dated 22 Nov 2026, role fix):** an attempt counts once the build's first step has started, that is once `scripts/aws_env_build_instance.sh` is running on the instance. A launch refused, or terminated before that, does not count: neither launch of 21 Nov 2026 (20:12 and 21:15 UTC) counts. Once the first step has started, any failure ends the attempt: stop and report.

**21 Nov 2026, 20:12 UTC (22 Nov, 01:42 IST; from commit `cb7c6fa`):** every check before the create request passed: the Standard-family quota is 32 vCPU, the key pair, security group and hard-delete role were made, and the base image was read. AWS refused the create request with `InvalidParameterCombination`: "The specified instance type is not eligible for Free Tier." The account is on AWS's Free plan, which admits only Free Tier types: in Sydney, at most 2 vCPU and 8 GiB (`m7i-flex.large`, `c7i-flex.large`, `t3`/`t4g`/`t8i`/`t9g` micro and small). It admits no `m7a`, no `g6` and no `p5`. No instance, volume, schedule, image or snapshot exists (read back in Sydney); ₹0. No build ran. Stopped and reported, as the attempt rule requires. Whether this spent the one attempt is the researcher's call.

**21 Nov 2026, 21:15 UTC (22 Nov, 02:45 IST; from commit `472fb86`, the build wave):** the credit gate passed (plan `PAID ACTIVE`, $140 left, ₹13,438.60 > ₹0 planned + ₹250 cap), as did the pre-launch pass (the Mac's IP unchanged, the security group's one rule, the key pair, the role, the 32 vCPU quota, the image available). `run-instances` created `i-040efb08eea1aed0e` at 21:15:09. `create-schedule` was then refused with `ValidationException`: "The execution role you provide must allow AWS EventBridge Scheduler to assume the role." The launcher terminated the instance at 21:15:12, as designed, about 3 s after creation (billed at EC2's 60 s minimum: about ₹1). Read back in Sydney: the instance terminated; no live instance, volume, schedule, snapshot or Elastic IP. No build ran and no AMI exists. The cause is the role's trust policy, made the same day: its `aws:SourceArn` condition names the schedule (`schedule/default/attnbench-hard-delete-*`), but AWS's documented form names the schedule group (`schedule-group/default`). The fake `aws` in the tests accepts any role, so no test could catch it. Stopped and reported. By the attempt rule above it does not count.

**Role fixed, 21 Nov 2026, about 21:35 UTC (adviser's rulings dated 22 Nov 2026, role fix):** before the change, a dry registration against the old role (a throwaway schedule thirty days out, for the non-existent instance `i-00000000000000000`) was refused with the same `ValidationException`, both enabled and disabled; nothing was left. The trust condition is now `StringEquals` on `aws:SourceAccount` (the account) and `aws:SourceArn` = `arn:aws:scheduler:ap-southeast-2:<account>:schedule-group/default`. Read back in that form; a disabled throwaway using the role was then accepted, read back `DISABLED` and deleted (no schedule left). The launcher now does both checks itself on every launch (below).

**21 Nov 2026, 21:54 UTC (22 Nov, 03:24 IST; from commit `5c8cf17`, the role-fix wave): the attempt, failed at step 1.** Credit gate passed ($160 left, ₹15,358.40 > ₹250 planned + ₹250 cap); the trust policy read back in the documented form; the dry registration was accepted and deleted; `run-instances` created `i-0820d2f6a8a6e4572`, and the hard-delete schedule was created and read back `ENABLED` (delete at 04:09:39 UTC). SSH at +1 min; bundle and wheel deployed; the image holds `/usr/local/cuda-12.8`, `-12.9`, `-13.0` and `-13.2` (`/usr/local/cuda` points to 13.2), Python 3.10.12, 4 vCPU, 15 GiB. `scripts/aws_env_build_instance.sh` started at 21:56 UTC, and its guards (step 1 of 6) refused: nvcc 12.x was found, but "no cuobjdump under CUDA_HOME=/usr/local/cuda-12.9. Nothing was installed." Teardown at 22:03:51 UTC: 10 min, about ₹6 (₹34.90/h); read back in Sydney, no instance, volume, snapshot, image, Elastic IP or schedule. No AMI. The first step had started, so by the attempt rule this was the attempt; stopped and reported. Which toolkit, if any, on the image carries `cuobjdump` was not read before teardown.

**The probe, 21 Nov 2026, 22:16 UTC (22 Nov, 03:46 IST; from commit `41aefa6`; adviser's rulings dated 22 Nov 2026, probe).** One `m7a.xlarge` from the same base image, no build: `scripts/aws_launch_session.sh PROBE` (cap ₹15, halt at 6 min, delete at 8 min, worst case ₹5; every other guarantee of the launcher, its new `PROBE` pairing alone allowed limits under 30 and 20 min). Credit gate passed ($160 left, ₹15,358.40 > ₹500 planned + ₹15 cap); trust policy read back; dry registration accepted and deleted; `i-02665417050b16e93` created at 22:16:16, SSH at +34 s, `scripts/aws_env_probe_inventory.sh` piped over SSH, then torn down: about 2 min, about ₹1. Sydney read back with no live instance, volume or schedule. The inventory, verbatim and with no identifiers, is `docs/launch_cards/aws_env_build_probe_20261121.txt`. What it shows:

- **Toolkits:** `/usr/local/cuda-12.8` (nvcc V12.8.93), `-12.9` (V12.9.86), `-13.0` (V13.0.88), `-13.2` (V13.2.51); `/usr/local/cuda` → 13.2. **No toolkit has `cuobjdump`**, and none is anywhere under `/usr` or `/opt`. The toolkits came from runfiles: `dpkg` lists no `cuda-*` package.
- **apt:** the NVIDIA CUDA repository for Ubuntu 22.04 is configured, and its shipped cache offers `cuda-cuobjdump-12-9` 12.9.82 (and 12-0 to 13-5).
- **Python:** `/usr/bin/python3` 3.10.12, pip 26.3, **no torch** (step 2 installs it); no conda or other Python under `/opt`.
- **Machine:** Ubuntu 22.04.5, kernel 6.8.0-1069-aws, 4 vCPU, 15,620 MiB; root 194 GB, 50 GB used, **144 GB free**.

Nothing calls for a design change: nvcc 12.9 is there (12.8 too), Python is image v6's, and the disk is ample. So the build script's step 0 now finds `cuobjdump` in this order: nvcc's own toolkit; any toolkit at or above nvcc's version, lowest first (**plan default, flagged**); then `apt-get install cuda-cuobjdump-12-9` (waiting up to 10 min for the dpkg lock, which a fresh boot's unattended-upgrades can hold), once more after an `apt-get update` if the shipped cache fails; else it refuses. Every refusal or failure now prints the full tool inventory. The search runs last among the guards, so every free check passes before the one apt install. Tested on fixtures built from the committed inventory (`tests/test_aws_env_build_instance.py`: on the image as probed, apt installs 12-9; stale cache; apt failing; a toolkit below nvcc not used; the lowest one at or above it used). One mutation (a toolkit below nvcc admitted) was caught.

**Second attempt (adviser's rulings dated 22 Nov 2026, probe):** one build attempt on the same terms (cap ₹250, halt at 360 min, delete at 375 min); on failure after the first step, stop and report; on success, the AMI and its snapshot cost.

## Results: small files only

Only step 6's lists and logs and step 7's AMI id come back, each under 20 MB, by
`scp` to `~/Desktop/research/attnbench_backups/aws_env_build_<UTC>/`. No S3
bucket, nothing to the GCP bucket.

## Outside the instance cap

- **The AMI's snapshot** is billed until it is deleted: about 60 GB at an assumed $0.055/GB-month (not checked), about ₹320 a month. It is charged to the AWS cap of about ₹10,400 and listed in the AWS budget list (`docs/spend_ledger.md`).
- **Cleanup entry:** deregister the AMI and delete its snapshot after P6, or, if the host-effect rule skips P6, after the skip is read, and in any case before the AWS credits expire (about April 2027, six months from the grant; the researcher's date, 22 Nov 2026); `scripts/aws_cleanup_check.sh` and a read-back must then show neither.
- Transfer: inbound is free; the small files out are under 100 MB (about ₹1).

## Before the yes is asked

- **CPU pre-launch pass (fake AWS CLI), 21 Nov 2026, rerun at the fifth wave's settings** (`tests/test_aws_launch_session.py::test_the_environment_build_cards_settings`): the launcher passes every gate. An abort at the prompt makes no create call. A typed `launch` makes the dry registration (below), then exactly one `run-instances` call (`m7a.xlarge`, `ap-southeast-2a`), then `create-schedule` at launch + 375 min and `get-schedule`, with the halt at +360 min and worst case ₹219 under the ₹250 cap; a ₹218 cap is refused before any AWS call.
- **Pre-launch pass against the account (read-only, at launch time):** `aws login` live; the Standard-family vCPU quota in Sydney at least 4; the key pair, the security group (SSH from this Mac only) and the hard-delete role exist; the base image's id and its CUDA toolkit versions from `describe-images`; no `attnbench-session` instance already running.
- **Credit gate and allowed services (adviser's rulings dated 22 Nov 2026, credits).** The launcher refuses unless the account's plan reads `PAID ACTIVE` and the remaining credits (Free Tier API, USD, at ₹95.99/$) exceed the planned AWS spend so far plus this block's ₹250 cap. The planned spend so far is the sum of the caps in the off-repo launch record (`~/.attnbench/aws_launches.tsv`): the launcher writes one line per create request, at its cap. Read 22 Nov 2026: plan `PAID ACTIVE`, $140 left (about ₹13,439), so ₹250 passes. The base image is Amazon-owned (owner alias `amazon`, the Deep Learning AMI publisher) with no Marketplace product code and no billing product (usage `RunInstances`). The work uses only EC2, EBS volumes, snapshots and AMIs, networking, IAM, EventBridge Scheduler and Service Quotas (the launcher's notes).
- **Hard-delete role, checked by the launcher at every launch (adviser's rulings dated 22 Nov 2026, role fix).** Before the prompt it reads the role's trust policy back and refuses unless it is the documented form: one statement, `scheduler.amazonaws.com` may `sts:AssumeRole`, on `StringEquals` `aws:SourceAccount` = the pinned account and `aws:SourceArn` = the region's `schedule-group/default`. After the typed `launch` and before `run-instances`, it registers a throwaway `DISABLED` schedule thirty days out using the role, for a non-existent instance, reads it back and deletes it; a refusal, a wrong state or a failed delete stops the launch with nothing created. Mock tests in `tests/test_aws_launch_session.py`, among them the trust condition of 21 Nov 2026 naming a schedule (refused); four mutations of the checks were each caught.
- **After the first AWS usage**, check the next day that the credits were applied to the EC2 line of the bill (billing lags by up to a day). If they were not, stop before P3.
- **Steps 2 to 6 scripted, 21 Nov 2026:** `scripts/aws_env_build_instance.sh`, with a fake-command pass on the Mac (`tests/test_aws_env_build_instance.py`: order, refusals before any install, a stop at each step's failure; `tests/test_env_build_lists.py`: the lists, the wheel check, the library check). Step 7 is still by hand.
- **The build time is the risk.** v6's own builds, with 4 jobs on a VM with about 27 GB usable (`attnbench_backups/diag_rerun_20261117/home_logs/`), took 51 min (block-sparse-attention) and 2 h 13 min (flash-attn), both at archs `80;90`. On `m7a.xlarge`, 16 GiB admits 2 jobs. Half the jobs alone would double those times to about 4 h 8 min. `m7a`'s vCPUs are whole Zen 4 cores, not hyperthreads, which should shorten that by an unmeasured amount. With the installs, the estimate is 3.5 to 4.5 h against the 360-minute halt. At the halt the instance stops and the hard delete ends it; the attempt is spent.
- The researcher's launch yes, logged in `docs/permission_log.md` before the launch.

## Risks

- **A changed home IP breaks driver contact** (22 Nov 2026, item 3). The security group `attnbench-ssh` admits SSH only from this Mac's IP at the time it was made (/32). If the home connection's address changes mid-build, the Mac can no longer reach the instance: no record comes back and no AMI is made from the Mac. The build itself runs on, detached; the instance halts itself at 360 min and the hard delete ends it at 375 min, so the worst case stays ₹219. The rule can be re-pointed to the new address by hand.
- A base AMI whose driver does not suit the L4 or H100 shows only on a GPU: the `g6.xlarge` rehearsal (about ₹180) checks it before P3.
- `m7a` is AMD EPYC 4th gen, not P3's and P6's 7R13. The image is portable across x86-64; no host-specific build flag is set.
