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
   - **0, guards:** a clean tree, Python 3.10.12, a new record directory (`~/env_build_<UTC>`, never overwritten), the wheel and its sha256, the FA3 manifest at its recorded sha256, nvcc 12.x (`/usr/local/cuda-12.9` when present, v6's) and `cuobjdump`, and flash-attn's memory guard at `MAX_JOBS`. 16 GiB admits 2 jobs, not 4, so 4 is refused in seconds.
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

**One attempt on `m7a`** (fifth wave, item 5; replaces the two of the fourth wave): cap ₹250, halt at 360 min, delete at 375 min. If it does not end in `ENV_BUILD_OK` and an available AMI, stop and report; a second attempt needs a new ruling.

**22 Nov 2026, 20:12 UTC (from commit `cb7c6fa`):** every check before the create request passed: the Standard-family quota is 32 vCPU, the key pair, security group and hard-delete role were made, and the base image was read. AWS refused the create request with `InvalidParameterCombination`: "The specified instance type is not eligible for Free Tier." The account is on AWS's Free plan, which admits only Free Tier types: in Sydney, at most 2 vCPU and 8 GiB (`m7i-flex.large`, `c7i-flex.large`, `t3`/`t4g`/`t8i`/`t9g` micro and small). It admits no `m7a`, no `g6` and no `p5`. No instance, volume, schedule, image or snapshot exists (read back in Sydney); ₹0. No build ran. Stopped and reported, as the attempt rule requires. Whether this spent the one attempt is the researcher's call.

## Results: small files only

Only step 6's lists and logs and step 7's AMI id come back, each under 20 MB, by
`scp` to `~/Desktop/research/attnbench_backups/aws_env_build_<UTC>/`. No S3
bucket, nothing to the GCP bucket.

## Outside the instance cap

- **The AMI's snapshot** is billed until it is deleted: about 60 GB at an assumed $0.055/GB-month (not checked), about ₹320 a month. It is charged to the AWS cap of about ₹10,400 and listed in the AWS budget list (`docs/spend_ledger.md`).
- **Cleanup entry:** deregister the AMI and delete its snapshot after P6, or, if the host-effect rule skips P6, after the skip is read; `scripts/aws_cleanup_check.sh` and a read-back must then show neither.
- Transfer: inbound is free; the small files out are under 100 MB (about ₹1).

## Before the yes is asked

- **CPU pre-launch pass (fake AWS CLI), 21 Nov 2026, rerun at the fifth wave's settings** (`tests/test_aws_launch_session.py::test_the_environment_build_cards_settings`): the launcher passes every gate. An abort at the prompt makes no create call. A typed `launch` makes exactly one `run-instances` call (`m7a.xlarge`, `ap-southeast-2a`), then `create-schedule` at launch + 375 min and `get-schedule`, with the halt at +360 min and worst case ₹219 under the ₹250 cap; a ₹218 cap is refused before any AWS call.
- **Pre-launch pass against the account (read-only, at launch time):** `aws login` live; the Standard-family vCPU quota in Sydney at least 4; the key pair, the security group (SSH from this Mac only) and the hard-delete role exist; the base image's id and its CUDA toolkit versions from `describe-images`; no `attnbench-session` instance already running.
- **Steps 2 to 6 scripted, 21 Nov 2026:** `scripts/aws_env_build_instance.sh`, with a fake-command pass on the Mac (`tests/test_aws_env_build_instance.py`: order, refusals before any install, a stop at each step's failure; `tests/test_env_build_lists.py`: the lists, the wheel check, the library check). Step 7 is still by hand.
- **The build time is the risk.** v6's own builds, with 4 jobs on a VM with about 27 GB usable (`attnbench_backups/diag_rerun_20261117/home_logs/`), took 51 min (block-sparse-attention) and 2 h 13 min (flash-attn), both at archs `80;90`. On `m7a.xlarge`, 16 GiB admits 2 jobs. Half the jobs alone would double those times to about 4 h 8 min. `m7a`'s vCPUs are whole Zen 4 cores, not hyperthreads, which should shorten that by an unmeasured amount. With the installs, the estimate is 3.5 to 4.5 h against the 360-minute halt. At the halt the instance stops and the hard delete ends it; the attempt is spent.
- The researcher's launch yes, logged in `docs/permission_log.md` before the launch.

## Risks

- **A changed home IP breaks driver contact** (22 Nov 2026, item 3). The security group `attnbench-ssh` admits SSH only from this Mac's IP at the time it was made (/32). If the home connection's address changes mid-build, the Mac can no longer reach the instance: no record comes back and no AMI is made from the Mac. The build itself runs on, detached; the instance halts itself at 360 min and the hard delete ends it at 375 min, so the worst case stays ₹219. The rule can be re-pointed to the new address by hand.
- A base AMI whose driver does not suit the L4 or H100 shows only on a GPU: the `g6.xlarge` rehearsal (about ₹180) checks it before P3.
- `m7a` is AMD EPYC 4th gen, not P3's and P6's 7R13. The image is portable across x86-64; no host-specific build flag is set.
