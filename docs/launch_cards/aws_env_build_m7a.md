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

1. Deploy the committed HEAD as a git bundle (as on GCP); a clean tree is required.
2. Create the venv from the v6 package list (`docs/image_libraries/attnbench-env-v6-20260917/`), the environment P1, P2 and P4 ran.
3. Copy up the FA3 wheel (`flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl`, sha256 `1676ef0b86c377ae…`, local in `attnbench_backups`), check its hash and wheel tags, install. FA3 is **not imported** here (no GPU); P6's `fa3_import` step does that.
4. XAttention and block-sparse-attention bootstrap at their pinned commits, built for `TORCH_CUDA_ARCH_LIST="8.9;9.0"` (L4 and H100).
5. CPU smoke: the CPU test subset and the imports of `attnbench`.
6. Record `pip freeze`, `dpkg -l` and the build logs.
7. Stop the instance; `ec2 create-image`; wait until `available`; read back the AMI and its snapshot.
8. Teardown (`scripts/aws_teardown_session.sh`, `scripts/aws_cleanup_check.sh`): instance terminated, no volume left. The AMI is kept.

## Results: small files only

Only step 6's lists and logs and step 7's AMI id come back, each under 20 MB, by
`scp` to `~/Desktop/research/attnbench_backups/aws_env_build_<UTC>/`. No S3
bucket, nothing to the GCP bucket.

## Outside the instance cap

- **The AMI's snapshot** is billed until it is deleted: about 60 GB at an assumed $0.055/GB-month (not checked), about ₹320 a month. Plan default: delete the AMI and snapshot after P6, or on 20 Dec 2026, whichever is first. It is charged to the AWS cap of about ₹10,400.
- Transfer: inbound is free; the small files out are under 100 MB (about ₹1).

## Before the yes is asked

- **CPU pre-launch pass (fake AWS CLI), 21 Nov 2026:** with this card's settings the launcher passes every gate. An abort at the prompt makes no create call (calls made: `sts get-caller-identity` ×2, `describe-instance-type-offerings`, `describe-images`). A typed `launch` makes exactly one `run-instances` call, then `create-schedule` and `get-schedule`, with worst case ₹79 under the ₹80 cap.
- **Not built yet:** steps 2 to 7 have no script; until one is written and its own CPU pass runs, they are done by hand over `ssh` under this card. The build time of step 4 on 4 vCPUs is unmeasured. If it runs past the halt, the instance stops and nothing is lost but the time; the hard delete then ends it.
- The researcher's launch yes, logged in `docs/permission_log.md` before the launch.

## Risks

- A base AMI whose driver does not suit the L4 or H100 shows only on a GPU: the `g6.xlarge` rehearsal (about ₹180) checks it before P3.
- `m7a` is AMD EPYC 4th gen, not P3's and P6's 7R13. The image is portable across x86-64; no host-specific build flag is set.
