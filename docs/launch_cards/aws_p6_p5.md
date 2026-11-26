# Launch card: P6 (AMD EPYC 7R13 + H100, AWS `p5.4xlarge`, Sydney; EXPLORATORY)

Prepared 26 Nov 2026 (P3's post-run task, item 6); built the same day (the researcher's task
after P3's push, item 2). **Nothing here launches anything: the researcher launches, on a yes
that names the commit.** P6 is exploratory (Appendix A, 21 Nov 2026) and scores no hypothesis.

**Time box (item 2).** P6 is dropped if the driver and the prediction are not ready by
30 Nov 2026, or if no H100 launches by **2 Dec 2026, 18:30 UTC** (epoch 1796236200; the driver
refuses after it and shortens a window to end by it). Both were ready on 26 Nov 2026.

## The gate: run

The host-effect rule (Appendix A, 21 Nov 2026; `scripts/p6_host_effect.py`) on P3 against P1,
read after P3's block 2 on 26 Nov 2026 (`docs/p3_run/gate.json`): all 35 cells, **d = 0.2440**
(p95 0.2699, total ratio − 1 = −0.2389). d > 0.10, so **P6 runs**.

## The prediction, committed and pushed before any launch

`docs/p6_prediction/prediction_P6_exploratory_from_P4_20261126T153102766985Z.json`, sha256
`11f2361f…` (sidecar beside it), written by `scripts/p6_prediction_from_p4.py` from a clean
tree at `8b0ae82`. It is the **same-GPU mirror of P4**, built as P3's was from P1: each of the
35 cells carries P4's committed value (T6.4, `prediction_P4_20261116T120732377724Z.json`,
written 16 Nov 2026 at `7a24cba`) for every predictor P4's file holds (recursion, step form,
sum, sum + s0, the fitted sums, learned). P4's record is read against its sidecar; its cells
are `docs/p6_cells.json`'s, key for key. No estimator, threshold or matching rule changed; the
timed-path pin `0e0fb012…` and every earlier prediction are untouched.
`tests/test_score_p6_exploratory.py` checks the file against P4's values as the scorer itself
loads them from the 16 Nov 2026 mirror (35 of 35).

## Dense backends: what the image has, what the prediction priced

- **The image** (`ami-096c5dc30364c9041`, the environment build of 22 Nov 2026; its build
  record, not rebuilt): FA2 `2.8.3.post1` compiled for **sm_80 and sm_90**; block-sparse-attn
  for sm_80 and sm_90; **FA3 present**, `flash_attn_3 3.0.0b1` installed from the restricted
  wheel `1676ef0b…` (built 28 Oct 2026 for sm_90a). The build ran on a CPU instance and the
  rehearsal on the L4: **nothing has run this image on an H100**.
- **Priced:** `sdpa_flash` for every dense cell (`shakedown_arms.ARMS["dense"]`), in P4's
  prediction and so in P6's.
- **P4's ground truth** ran K2's choice: **`fa3` at all six dense cells** (16 Nov 2026; at 2048
  `sdpa_flash`, `sdpa_cudnn`, `fa2` and `fa3` were correct, `sdpa_efficient` incorrect, no
  available kernel). If P6 chooses as P4 did, all six dense cells are mismatched and the
  matched group is empty.
- **Scoring:** `scripts/score_p6_exploratory.py` (the scorer's functions unchanged, against
  P4's committed values) splits every predictor's dense error into matched (chosen = priced)
  and mismatched cells.

## The abort gate, in block 1

`scripts/p6_backend_check.py` runs first in block 1, before any step. It checks:

- the GPU is an H100 at compute capability 9.0;
- the three distributions and the FA3 wheel's recorded sha256;
- the sm_90 code in FA2's and block-sparse-attn's extensions (`cuobjdump`, recorded);
- every K2 candidate supplied;
- K2's own screen at 2048 × 1 on the model's geometry: the four candidates P4 found correct
  there must be correct.

Any failure ends block 1 with exit 3. The driver then pulls what there is, tears down and
reports `DONE FAILED at block b1`, at about 10 min (about ₹170).

## The session

| Item | Value |
|---|---|
| Instance | `p5.4xlarge`: 1x H100 80GB, 16 vCPU, AMD EPYC 7R13, 200 GB gp3 (`scripts/aws_launch_session.sh P6`) |
| Region, zones | `ap-southeast-2` only, on demand. Zones offering it, read 26 Nov 2026: **2b, 2c, 2a**, tried in that order every 5 min on a capacity refusal, for at most 3 h |
| Quota | Sydney's on-demand P quota, read 26 Nov 2026: **16 vCPU**, exactly one `p5.4xlarge`; no instance was running |
| Image | `ami-096c5dc30364c9041` |
| Cells | 35, P1's grid as P3's (`configs/t31/p6_session.json`), equal to P4's (`docs/p6_cells.json`) |
| Steps | P3's runner and its 11 T3.1 steps in three blocks: b1 backend gate, `preflight` .. `probe`; b2 `host_components`; b3 `device_components` .. `postflight`. No delay injection (P4 ran none) |
| Driver | `scripts/aws_p3_driver.sh P6`: P3's, with every fix of 26 Nov 2026 (halts rebuilt, halt pushed back at each pull and every 10 min while copying, final re-pull off, masked logs) |
| Record | `attnbench_backups/aws_p6_p5_<UTC>/` |

**Plan defaults, flagged.**

- **Runner.** P4's own runner (`t61_session.py`, `t64_session.py`) is inside the pin and checks
  for GCP (`cloud_provider == "gcp"`, Flex/Spot provisioning, the GCS bucket). P6 instead runs
  P3's runner, which runs the same T3.1 step code (P4's ground truth is
  `t31_steps.step_ground_truth` too).
- **Steps left out.** P4's untimed extras do not run: `prereqs`, `fa3_import` (the image
  already holds FA3; the gate checks it), `masks`, `launch_queue` (it runs inside `link`), the
  host-speed checks, `late_check` and `trace_checks`.
- **H100 host.** `t227_steps.py`, outside the pin, accepts `aws:p5.4xlarge` with the H100.

## Hours, rate, cap

Planned **4.0 h** (Appendix A, 21 Nov 2026). P3's blocks on the same AMD host took 35, 81 and
99 min (b2 is host-bound), so P6 should take about 3.5 h with boot and teardown. Rate
**₹990.2/h** all-in, the card rate of Appendix A, 21 Nov 2026 ($8.944/h, Sydney's price file
of 20 Nov 2026, + 200 GB gp3, × 1.15 × ₹95.99); not re-read.

| | Minutes | ₹ at the rate |
|---|---|---|
| Planned | 240 | 3,961 |
| In-guest halt | **312** | 5,149 |
| Hard delete | **359** | 5,925 (worst case) |
| **Cap** | | **5,940** (4.0 h × ₹990.2 × 1.5) |

## Budget and the credit gate (read 26 Nov 2026)

- **Plan and credits:** PAID ACTIVE, **$155.85** of credits (₹14,960 at ₹95.99).
- **Launch record:** it counts ₹2,467 planned. With P6's cap that is ₹8,407, below the credits,
  so the launcher's gate passes.
- **AWS spent at the all-in rate:** about ₹963, plus the snapshot at about ₹442 a month. With
  P6's cap that is about ₹6,903, inside the AWS cap of about ₹10,400.

## Before the yes, and the command

1. A fresh `aws login --profile attnbench`, since the session ends at most 12 h after the
   login. The window and the run need 9.0 h, and the driver warns under 9.5 h.
2. The security group must admit this Mac's address. The driver checks it and never re-points
   it; if it refuses, the researcher runs `sg_repoint.sh`.
3. The commit the yes names, checked out clean outside the scratchpad, launched under
   caffeinate:

```bash
cd ~/Desktop/research/attnbench_scaffold && git fetch origin
git worktree add ~/Desktop/research/attnbench_p6 <COMMIT>      # the commit the yes names
cd ~/Desktop/research/attnbench_p6
aws login --profile attnbench
ATTNBENCH_LOGIN_AT=$(date -u +%s) caffeinate -i bash scripts/aws_p3_driver.sh P6
```

## Known gaps, carried from P3

- **TERM does not reach the steps.** A block killed on the instance leaves its step process
  running until it ends.
- **Logins refused 5 min before a halt.** systemd writes `/run/nologin` then. The halt is
  pushed back at each pull, but a login inside a halt's last 5 min is still refused.
- **Manual recovery untested.** Recovery of a stopped instance (`aws_p3_g6.md`) is manual and
  untested.
- **New on the H100.** The backend gate, FA3 and the H100 host path in `t227_steps.py` have
  only run against fakes and CPU stubs.
- **XAttention densities.** P6's measured densities must equal P4's committed ones for
  `measure` to match its cells. They come from the same GPU's masks, as P3's matched P1's on
  the L4; if they do not match, scoring refuses those cells, not the run.
