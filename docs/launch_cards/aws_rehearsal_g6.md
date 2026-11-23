# Launch card: the `g6.xlarge` rehearsal (AWS Sydney, from the built AMI)

Prepared 23 Nov 2026 (adviser's task of 23 Nov 2026, item 4a). **Not launched; for the
adviser's approval.** No step below has run on a GPU, so every time and cost is an
estimate. The researcher launches it; nothing here runs on its own.

## Purpose

Show on one L4, before P3 spends its cap, that the image built on 22 Nov 2026
(`ami-096c5dc30364c9041`, `attnbench-env-20261122T1235Z`, built on an `m7a.xlarge` with
no GPU) works on the GPU P3 uses. The build could not check three things, and only these
three are checked:

| # | Check | Pass | Basis |
|---|---|---|---|
| 1 | GPU and driver | `nvidia-smi` lists exactly one `NVIDIA L4`; the driver's CUDA version is at least 12.9 (torch `2.9.1+cu129`); `torch.cuda.is_available()`; compute capability (8, 9); one bf16 matmul runs | a base-image driver that does not suit the L4 shows only on a GPU (`docs/launch_cards/aws_env_build_m7a.md`) |
| 2 | The `fa2` arm | `flash_attn` 2.8.3.post1 imports from the image; the repository's `fa2` backend (`attnbench/backends/impls.py`, a K2 candidate in `attnbench/dense_selection.py`) passes the dense selection's correctness check against the reference at 2048 and 8192 tokens, batch 1, status `correct` | flash-attn was built for `80;90` (its `setup.py` takes no 89), so the L4 runs the sm_80 cubin; untested on an sm_89 card until now |
| 3 | One dense cell | T3.1's step 3, `dense_select`, on the one cell of `configs/aws/rehearsal_g6.json` (the `dense` arm at 8192 tokens, batch 1, P1's and P4's anchor; P1's session values): the session floor's dense pre-pass, then K2's timed rounds for every L4 candidate, `fa2` among them; `complete`, a candidate chosen | the harness, the model and a RULER prompt on this image and host, once |
| 4 | Host components, one cell | T3.1's step 5, `host_components`, on the same cell (adviser's task of 23 Nov 2026, second): recording, the null-kernel pass, the bare timer, the residual; `bare_prices.json` written; its time | P3's gate reads step 5, its longest block |

Checks 3 and 4 are timed and their times set beside T3.2d's for the same cell (step 5:
51.9 s for `dense_d1_L8192_b1`, 4,729.9 s for all 35 cells) for a sanity read and for
re-estimating P3's hours. They are not a measurement for any hypothesis, are not banked as
one, and do not gate P3: the `g6.xlarge` host (4 vCPU, 16 GiB) is not P3's `g6.2xlarge`
(8 vCPU, 32 GiB), although the processor is the same (AMD EPYC 7R13).

**What runs, and what does not (plan defaults, flagged).** On the instance,
`scripts/aws_rehearsal_instance.sh` runs checks 1 to 4 in order and stops at the first
failure (`scripts/aws_rehearsal_checks.py` for 1 and 2; `scripts/t31_steps.py` for 3 and
4, with `--config configs/aws/rehearsal_g6.json`). T3.1's `preflight` is not run: its T2.20
check expects GCP's `g2-standard-8` and would refuse this host (a P3 build item). Before
the checks the driver bootstraps the image as `gcp_bootstrap_instance.sh` steps 1 to 4 and 8
do (RULER data verified locally, copied, verified there; wonderwords 2.2.0 and nltk 3.9.1;
the tree still clean). The model's weights come from Hugging Face at the first load; the
snapshot is recorded.
**Plan default, flagged:** check 1 also imports `block_sparse_attn` and XAttention, untimed
(under a minute), because a missing library on the GPU host would otherwise show first in P3.

## The instance

| | |
|---|---|
| Type | `g6.xlarge` (1 x L4 24 GB, AMD EPYC 7R13, 4 vCPU, 16 GiB), on-demand |
| Region | `ap-southeast-2` (any of 2a, 2b, 2c; all three offer `g6.xlarge`, read 23 Nov 2026). From the third attempt: 2b, then 2c, then 2a, the next only after an `InsufficientInstanceCapacity` refusal (below) |
| Image | `ami-096c5dc30364c9041` |
| Quota | Running On-Demand G and VT: 8 vCPU (read 23 Nov 2026); this uses 4 |
| Root volume | 200 GB gp3, encrypted, deleted with the instance |
| Rate | **₹118.41/h** all-in: (the instance at $1.0464/h, SKU `N843N2ZTJ23RC5DC`, + 200 GB gp3 at $0.096/GB-month over 730 h) x 1.15 x ₹95.99; both rates read from the AWS Price List API for Sydney on 23 Nov 2026 (terms effective 1 Nov 2026) |

## Time and cost

| | Minutes |
|---|---|
| Launch to SSH | 8 |
| Deploy and bootstrap | 4 |
| Check 1, GPU and driver | 2 |
| Check 2, the `fa2` arm (the model's first load included) | 6 |
| Check 3, one dense cell | 5 |
| Check 4, host components on one cell (P1: 52 s; doubled, then the harness load) | 3 |
| Pull, teardown, cleanup check | 10 |
| Margin | 7 |
| **Planned** | **45** (about ₹89) |

**Cap ₹180** (the adviser's figure); the in-guest halt **stops** the instance at **75 min**
(₹148); the hard-delete schedule terminates it at **90 min** (worst case ₹178, inside the
cap). On any failed check: stop, pull the logs, tear down and report; nothing is retried in
the session.

## Before the launch yes is asked

- **Approved 23 Nov 2026** (adviser's task of 23 Nov 2026, second, item 2): launch once the
  pre-launch pass succeeds; one attempt.
- The launcher's `REH` target (`g6.xlarge`), with the card's settings tested
  (`tests/test_aws_launch_session.py::test_the_rehearsal_cards_settings`: worst case ₹178
  under the ₹180 cap; ₹177 refused before any AWS call); the instance script and the checks'
  verdicts tested on the CPU (`tests/test_aws_rehearsal.py`), and steps 3 and 5 run on the
  CPU behind the stubbed GPU boundary with this config.
- The SSH preflight (the security group admits this Mac's current address) runs before the
  launch; on 22 Nov 2026 the address changed between sessions.
- The credit gate passes (credits x ₹95.99 above the caps in the off-repo launch record plus
  this cap), and the credit check of 23 Nov 2026 is re-read once the bill shows the
  22 Nov 2026 build (it did not yet at 18:18 UTC on 22 Nov).
- A permission-log row is written before the launch.

## Cleanup

Instance terminated, no volume left, the Sydney sweep read back clean. The AMI and its
snapshot stay for P3 (and P6, if it runs); their deletion is on the ledger's cleanup list.

## The run, 23 Nov 2026 (one attempt; failed at check 2)

Launched 10:51 UTC from `d193825` by the researcher with the off-repo driver. The SSH
preflight, the credit gate ($158.39 left, ₹15,204 > ₹1,385 planned + ₹180), the trust
read-back and the dry registration passed; SSH at about +1 min; deploy and bootstrap (RULER
data copied and verified, wonderwords and nltk) passed by +6 min.

- **Check 1 passed** (215 s): one `NVIDIA L4`, driver 595.112.07 with CUDA 13.2, torch
  `2.9.1+cu129` sees capability (8, 9), the bf16 matmul is finite; flash-attn 2.8.3.post1,
  block-sparse-attention and `xattn.src.Xattention` import; flash-attn's library holds
  `sm_80` and `sm_90` only.
- **Check 2 failed in 2.2 s on a defect of this rehearsal's own code**:
  `aws_rehearsal_checks.py fa2` built `Harness(device="cuda")` without the required
  `dry_run`, so no fa2 kernel ran. Checks 3 and 4 did not run. No test ran that path; one
  does now (`test_the_fa2_check_builds_a_real_harness_and_screens_both_lengths`, watched to
  fail on the old call), and the call is fixed.
- The driver's teardown could not reach the EC2 endpoint (`Could not connect to the
  endpoint URL`, then the account pin's credential check failed: a transient network
  failure on this Mac). Re-run at once, the teardown terminated the instance at 11:09 UTC,
  18 min from launch, about ₹36; Sydney read back with no instance, volume, address or
  schedule. The in-guest halt (75 min) and the hard delete (90 min) bounded it meanwhile.

So whether sm_80 code runs on the L4 is **still unconfirmed by a run** (the cubins are there),
and P3's hours are not re-estimated: no step-5 time was measured. A second attempt needs a
launch yes; the driver now also retries the teardown on a failure (below).

## The second attempt, 23 Nov 2026 (no capacity, twice; nothing launched)

The researcher re-pointed the security group to this Mac's current /32 and ran the driver
at 18:03 UTC from `f3d59ac`. The driver ran again, unchanged, at 18:13 UTC from `c8605c4`.
Both times the SSH preflight, the credit gate and the dry registration passed, and then
`RunInstances` was refused with `InsufficientInstanceCapacity` for `g6.xlarge` in
`ap-southeast-2a`. AWS named 2b and 2c as having capacity. No instance id came back, and the
Sydney listing showed no instance, volume, address or schedule. The spend was ₹0. The launch
record holds both requests at ₹180 until they are struck.

## The third attempt: the zone fallback (the researcher's launch yes of 23 Nov 2026)

The launcher takes the ordered list `AWS_SESSION_AZS="ap-southeast-2b ap-southeast-2c
ap-southeast-2a"`. The fallback is `aws_launch_across_zones` in
`scripts/_aws_launch_policy.sh`, shared with P3 and P6. It makes one pass through the zones,
with no loop. The launch policy is checked again before each zone. The next zone is tried only
when three things hold: the refusal is `InsufficientInstanceCapacity`, no instance id came
back, and the session's tag finds no instance and no volume anywhere in the region. Any other
error, a stray instance or volume, or a read that fails stops at once. If all three zones
refuse, the launch stops (`LAUNCH_RESULT=EVERY_ZONE_REFUSED_FOR_CAPACITY`).

- Each zone tried is a create request, so the launch record holds it at ₹180. The worst case
  of a session is unchanged: only one instance can be created.
- The launcher prints `LAUNCH_ZONE=` and `ZONES_REFUSED=`. The teardown finds the instance by
  id in any zone and prints its zone, which is also written to `session_cost.txt`. The cleanup
  check's listings are region-wide; it names the zones they cover and lists the study's
  recently terminated instances with their zone.
- Subnets, read only on 23 Nov 2026: the security group is in the default VPC. Each of 2a, 2b
  and 2c has a default subnet that is available, gives a public address at launch, has free
  addresses, routes 0.0.0.0/0 to an internet gateway, and has a network ACL that allows tcp/22
  in and ephemeral ports out. With `--placement` alone, EC2 uses that subnet. The subnet ids are
  recorded off-repo, and the driver records whether the instance's subnet is the recorded one.
  The hard-delete schedule is regional (the schedule group `default`), so it does not depend on
  the zone.
- Tests on the fake `aws` (`tests/test_aws_launch_session.py`) cover:
  - two refusals, then a launch in 2a;
  - three refusals, after which it stops after one pass;
  - a non-capacity error, which stops at once;
  - a stray instance or volume, or a failed read, which stops;
  - the policy read before each zone.

  The driver ran end to end on the fake remote with the real launcher. The timed-path pin is
  unchanged.

## The third attempt, 23 Nov 2026 (no capacity in any zone; nothing launched)

The researcher ran the driver at 21:14 UTC from `01da24e`. The SSH preflight, the credit gate
and the dry registration passed. `RunInstances` was refused with `InsufficientInstanceCapacity`
for `g6.xlarge` in 2b, then 2c, then 2a, within 17 s. After each refusal the launcher confirmed
that no instance and no volume of the session existed, and it stopped after one pass. Sydney read
back with no instance, volume, address or schedule. The spend was ₹0. The launch record holds the
three requests at ₹180 each until they are struck. The fallback behaved as tested. Checks 1 to 4
are still unrun on a GPU, and P3's hours are not re-estimated. Another attempt needs a ruling.
