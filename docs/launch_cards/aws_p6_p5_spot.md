# Launch card: P6 on Spot (AWS `p5.4xlarge`, Sydney; EXPLORATORY)

Prepared 27 Nov 2026 (the researcher's task of 27 Nov 2026: "get a p5.4xlarge in AWS Sydney
through Spot"); second wave 28 Nov 2026 (the researcher's task of 28 Nov 2026, items 1–4). **Nothing here launches anything: the researcher launches, on a yes that names
the commit.** Everything in `docs/launch_cards/aws_p6_p5.md` holds, except what is below:
the gate, the prediction (sha256 `11f2361f…`, unchanged), the cells, the blocks, block 1's
abort gate, the rate ₹990.2/h, the cap **₹5,940 in all**, the halt +312, the delete +359 and
the deadline (**2 Dec 2026, 18:30 UTC**). The timed-path pin `0e0fb012` and every prediction
are untouched.

## Read on 27 Nov 2026 (Sydney only, read-only)

| | 2b | 2c | 2a |
|---|---|---|---|
| p5.4xlarge Spot price, last 3 days, USD/h | 3.75–4.48 | 4.11–4.40 | 5.66–6.07 |
| Spot placement score, 1 instance (1–10) | 1 | 4 | 1 |

A Spot pool exists in all three zones (prices published throughout), against the on-demand
$8.944/h. The region's score is 4. The P Spot quota (`L-7212CCBC`) read **0 vCPU**; a request
for 16 was submitted on 27 Nov 2026 on the researcher's yes (PENDING when submitted). The
quota is read again before every pass: a pass whose read is under 16 makes no Spot request
(it behaves as the on-demand card), and a grant during a window starts Spot at the next pass.

## What Spot mode does (`scripts/aws_p3_driver.sh P6 spot`)

- **Each pass** (every 5 min, windows of 3 h): the quota read, then one-time Spot in **2c, 2b,
  2a** (2c's placement score read 4, the others 1), then on-demand in 2b, 2c, 2a. The bid is
  **$8.944/h, the on-demand price**, so the rate bounds a Spot hour. A Spot refusal for
  capacity, price (`SpotMaxPriceTooLow`) or quota (`MaxSpotInstanceCountExceeded`) is
  confirmed to have left nothing, as a capacity refusal is; after a quota refusal no more Spot
  requests are made in that pass.
- **An expired AWS login** stops the window with "AWS login expired, run aws login" (in
  `driver.log` and `status.txt`), whichever call it refused: the account pin, the quota read,
  the security group read or a create request (each creates nothing; the window ends
  `NOTHING_CREATED`), or the check after a refusal (not confirmed: `UNCLEAR`, run
  `scripts/aws_cleanup_check.sh` after `aws login`). Window 3 of 27 Nov 2026 stopped instead
  as if the security group had changed.
- **A Spot instance terminates** at its halt and on interruption (a one-time Spot request
  cannot stop). It is tagged `SPOT`; block 1's preflight then requires `spot`
  (`ATTNBENCH_EXPECTED_PROVISIONING`, P6's host only).
- **An interruption** (the two-minute notice read over IMDSv2 at each poll, or the instance
  terminated with `Server.SpotInstanceTermination`, or a pull cut by either) discards the
  block it hit **in full**. The driver then tears the instance down and counts its minutes at
  the rate. It relaunches, **at most twice**, in block 1's zone, Spot then on-demand, for a
  window of up to 60 min before the deadline. It requires the new host to match the first
  (type, zone, GPU and driver, CPU, vCPUs), restores the finished blocks (one tar, every
  file's sha256 checked on the instance) and runs the backend check again. It then reruns the
  interrupted block from its start.
- **The money:** a relaunched instance's hard delete is what is left of ₹5,940 at the rate
  (at most +359), and its halt is 47 min earlier (at most +312). The relaunch is refused when
  that halt, less 10 min for the last pull, cannot fit the blocks left at the card's minutes
  (setup 20, b1 35, b2 81, b3 99).
- **The credit gate** counts the session at ₹5,940 in all: after an interruption the driver
  voids the instance's launch-record line (a dated `VOID` line) and counts the instance again
  at its spend at the rate, and the relaunch reserves what is left. With ₹2,467 counted before
  P6, the gate needs ₹8,407 of credits (about ₹14,490 left) however many relaunches.

**Allowed by the plan.** P4's Spot blocks (Appendix A, 15 Nov 2026): "An interrupted block is
still discarded in full and rerun on a new instance"; "later blocks run in b1's zone"; every
later block's instance must match b1's. P4 ran so on 16 Nov 2026 (b4's first VM was preempted
in its bootstrap and was rerun).

**Plan defaults, flagged.**

- A pass tries Spot in all three zones before any on-demand request.
- A quota under 16 vCPU counts as 0.
- Relaunches go to block 1's zone only (Spot and on-demand).
- On-demand keeps 2b, 2c, 2a; only Spot asks 2c first.
- With an instance up, an expired login is said once and the driver goes on over SSH (the
  instance halts and is deleted on its own times); its teardown then needs a new login.
- The relaunch window is 60 min.
- No relaunch after the deadline.
- A rerun block 3 commits its own in-session prediction; the attempt it replaces is kept
  under `attempt_N/`. The scored P6 prediction is the P4 mirror, unchanged.

## Known gaps

- **Before a block starts.** An interruption during boot, deploy, bootstrap or restore stops
  the session (teardown) rather than relaunching.
- **No stopped-instance recovery on Spot.** A Spot instance cannot be started again to pull.
- **The plan still says there is no Spot path.** Section 5.2 (28 Oct 2026) says the launcher
  has no Spot path; an Appendix A line is owed, with the lock's plan hash, before any merge.
- **Tested only on the fakes.** The interruption paths ran on the fake remote only.

## Tests (fake AWS and the fake remote)

- **Launcher, 31 tests:**
  - the pass order, and Spot's own order 2c, 2b, 2a (a list not the session's zones refused);
  - capacity, price-too-low and quota refusals;
  - the quota read (0, 8 or unreadable: no Spot request), again before every pass (0, 0, 16:
    Spot from pass 3; 16, 0: none in pass 2);
  - the bid cap, P6 and Sydney only;
  - an expired login at the start, and at the security group read, a create request, the quota
    read and the check after a refusal;
  - the credit gate on the driver's recount (admitted at ₹3,667 + the relaunch's cap; refused
    with the cap counted too).
- **Driver, 20 tests:**
  - an interruption mid-block 2 (by state) and mid-block 3 (by notice), each relaunched,
    restored and resumed to DONE OK;
  - an interruption in block 1;
  - a third interruption stopping the session;
  - a relaunch refused for capacity;
  - the cap left bounding the hard delete and refusing a relaunch that cannot fit;
  - a host mismatch;
  - a failed backend check;
  - two relaunches, the record at ₹5,940 in all, and the Spot zones handed to the launcher;
  - an expired login at the preflight and in the window (three ways).
- **Break-tests:** 20 mutations of the money, record, minutes, quota, zone order, bid, refusal
  and login code, all killed.

## The command

```bash
cd ~/Desktop/research/attnbench_scaffold
git worktree add ~/Desktop/research/attnbench_p6spot2 <COMMIT>     # the commit the yes names
cd ~/Desktop/research/attnbench_p6spot2
aws login --profile attnbench
ATTNBENCH_LOGIN_AT=$(date -u +%s) caffeinate -i bash scripts/aws_p3_driver.sh P6 spot
```

Start it only after the open on-demand window has ended with its end line: the windows ledger
`~/.attnbench/p6_windows.tsv` is shared, and an open window refuses the next.
