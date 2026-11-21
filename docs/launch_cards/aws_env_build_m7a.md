# Launch card: AWS environment build on `m7a.xlarge` (Sydney)

Prepared 21 Nov 2026 (adviser's rulings dated 21 Nov 2026, third wave, item 4).
**Not approved: no instance, no AWS spend until the researcher's launch yes.**
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
| Base image | the AWS Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04), latest in Sydney; its id read with a read-only `describe-images` at launch and named on the yes. **Plan default, flagged** |
| Rate (`AWS_RATE_INR_HR`) | ₹34.90/h: (instance $0.2898/h, Sydney public price file published 20 Nov 2026, + 200 GB gp3 at an assumed $0.096/GB-month, not checked) × 1.15 × ₹95.99/$ (the method of plan Section 5.2) |
| Planned hours | 1.5 |
| Halt (`AWS_HALT_MINUTES`) | 117 min (1.3 × planned), instance stops |
| Hard delete (`AWS_MAX_RUN_MINUTES`) | 135 min |
| Cap (`AWS_COST_CAP_INR`) | ₹80; worst case ₹34.90 × 135 min = ₹79 |
| Expected | about ₹52 |

## Steps on the instance

1. Deploy the committed HEAD as a git bundle (as on GCP); a clean tree is required. Copy up the FA3 wheel (`flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl`, sha256 `1676ef0b86c377ae…`, local in `attnbench_backups`).
2. to 6. One script, `FA3_WHEEL_SHA256=1676ef0b86c377ae911a7de238c2a1f6b58a7666f1a6eef919d4dc7b0973b64a MAX_JOBS=2 bash scripts/aws_env_build_instance.sh ~/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl` (added 21 Nov 2026, fourth wave, item 3). It stops at the first failure and prints `ENV_BUILD_OK` only when every step passed:
   - **0, guards:** a clean tree, Python 3.10.12, a new record directory (`~/env_build_<UTC>`, never overwritten), the wheel and its sha256, and flash-attn's memory guard at `MAX_JOBS`. 16 GiB admits 2 jobs, not 4, so 4 is refused in seconds.
   - **2, v6's packages** (`docs/image_libraries/attnbench-env-v6-20260917/`, the environment P1, P2 and P4 ran), from an index at exact versions with `--no-deps`. First the system list as root: less the 69 packages apt installed on v6 (they come with the base image), and with `--ignore-installed`, so v6's version shadows an apt copy at another version. Then the session user's site. torch and triton must then read `2.9.1+cu129` and `3.5.1`. **Plan defaults, flagged:**
     - The system Python and the user site, as v6, not a venv.
     - xformers' local build (`0.0.33+5d4b92a5.d20251029`) has no index copy, so PyPI's `0.0.33` stands in. xformers is no arm and is not in the library check.
   - **3, FA3:** its sha256 and tags are checked, then it is installed. It is **not imported** here (no GPU); P6's `fa3_import` step does that. The build's `manifest.json` (1.4 KB) is only in the GCP bucket; with it, the full `fa3_wheel.check_wheel` runs. Without it, the Python and torch are checked against the stack the wheel was built on.
   - **4, from source:**
     - flash-attn `2.8.3.post1` (`scripts/build_flash_attn.sh`);
     - block-sparse-attention at `bsa_contract.BSA_COMMIT`;
     - x-attention at `XATTN_COMMIT`, editable and outside the tree.

     The architectures are v6's, `80;90` (flash-attn and block-sparse-attention read `FLASH_ATTN_CUDA_ARCHS`, not `TORCH_CUDA_ARCH_LIST`; corrected from this card's `8.9;9.0`). x-attention's import is left to the GPU rehearsal.
   - **5, CPU smoke:**
     - `attnbench` imports;
     - this environment's libraries equal P2's (`configs/t64/p2_libraries.json`), with FA3 the one allowed difference, as P4's preflight requires;
     - `tests/test_build_guards.py` and `tests/test_env_build_lists.py` pass.
   - **6, the record:** `pip freeze` (user, user site, root), `dpkg -l`, `uname`, each list diffed against v6's, and `SHA256SUMS`. The tree must still be clean.
7. Stop the instance; `ec2 create-image`; wait until `available`; read back the AMI and its snapshot.
8. Teardown (`scripts/aws_teardown_session.sh`, `scripts/aws_cleanup_check.sh`): instance terminated, no volume left. The AMI is kept.

## Attempts

At most **two attempts on `m7a`**, each its own launch under this card: cap ₹80, halt at 117 min, delete at 135 min, each with its own record directory. If the second does not end in `ENV_BUILD_OK` and an available AMI, stop and report; there is no third attempt without a new ruling.

## Results: small files only

Only step 6's lists and logs and step 7's AMI id come back, each under 20 MB, by
`scp` to `~/Desktop/research/attnbench_backups/aws_env_build_<UTC>/`. No S3
bucket, nothing to the GCP bucket.

## Outside the instance cap

- **The AMI's snapshot** is billed until it is deleted: about 60 GB at an assumed $0.055/GB-month (not checked), about ₹320 a month. It is charged to the AWS cap of about ₹10,400 and listed in the AWS budget list (`docs/spend_ledger.md`).
- **Cleanup entry:** deregister the AMI and delete its snapshot after P6, or, if the host-effect rule skips P6, after the skip is read; `scripts/aws_cleanup_check.sh` and a read-back must then show neither.
- Transfer: inbound is free; the small files out are under 100 MB (about ₹1).

## Before the yes is asked

- **CPU pre-launch pass (fake AWS CLI), 21 Nov 2026:** with this card's settings the launcher passes every gate. An abort at the prompt makes no create call (calls made: `sts get-caller-identity` ×2, `describe-instance-type-offerings`, `describe-images`). A typed `launch` makes exactly one `run-instances` call, then `create-schedule` and `get-schedule`, with worst case ₹79 under the ₹80 cap.
- **Steps 2 to 6 scripted, 21 Nov 2026:** `scripts/aws_env_build_instance.sh`, with a fake-command pass on the Mac (`tests/test_aws_env_build_instance.py`: order, refusals before any install, a stop at each step's failure; `tests/test_env_build_lists.py`: the lists, the wheel check, the library check). Step 7 is still by hand.
- **The build time is the risk.** v6's own builds, with 4 jobs on a VM with about 27 GB usable (`attnbench_backups/diag_rerun_20261117/home_logs/`), took 51 min (block-sparse-attention) and 2 h 13 min (flash-attn). On `m7a.xlarge`, 16 GiB admits 2 jobs and there are 4 vCPUs, so flash-attn alone is likely to run past the 117-minute halt. That is unmeasured. At the halt the instance stops, nothing is lost but the time and an attempt, and the hard delete ends it.
- The researcher's launch yes, logged in `docs/permission_log.md` before the launch.

## Risks

- A base AMI whose driver does not suit the L4 or H100 shows only on a GPU: the `g6.xlarge` rehearsal (about ₹180) checks it before P3.
- `m7a` is AMD EPYC 4th gen, not P3's and P6's 7R13. The image is portable across x86-64; no host-specific build flag is set.
