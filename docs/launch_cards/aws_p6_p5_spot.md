# Launch card: P6 on Spot (AWS `p5.4xlarge`, Sydney; EXPLORATORY)

Prepared 27 Nov 2026 (the researcher's task of 27 Nov 2026: "get a p5.4xlarge in AWS Sydney
through Spot"). **Nothing here launches anything: the researcher launches, on a yes that names
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
for 16 was submitted on 27 Nov 2026 on the researcher's yes (PENDING when submitted). With
the quota under 16 the driver makes no Spot request: it behaves as the on-demand card.

## What Spot mode does (`scripts/aws_p3_driver.sh P6 spot`)

- **Each pass** (every 5 min, windows of 3 h): one-time Spot in 2b, 2c, 2a, then on-demand in
  2b, 2c, 2a. The bid is **$8.944/h, the on-demand price**, so the rate bounds a Spot hour.
  A Spot refusal for capacity, price (`SpotMaxPriceTooLow`) or quota
  (`MaxSpotInstanceCountExceeded`) is confirmed to have left nothing, as a capacity refusal is;
  after a quota refusal no more Spot requests are made in the window.
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

**Allowed by the plan.** P4's Spot blocks (Appendix A, 15 Nov 2026): "An interrupted block is
still discarded in full and rerun on a new instance"; "later blocks run in b1's zone"; every
later block's instance must match b1's. P4 ran so on 16 Nov 2026 (b4's first VM was preempted
in its bootstrap and was rerun).

**Plan defaults, flagged.**

- A pass tries Spot in all three zones before any on-demand request.
- A quota under 16 vCPU counts as 0.
- Relaunches go to block 1's zone only.
- The relaunch window is 60 min.
- No relaunch after the deadline.
- A rerun block 3 commits its own in-session prediction; the attempt it replaces is kept
  under `attempt_N/`. The scored P6 prediction is the P4 mirror, unchanged.

## Known gaps

- **Before a block starts.** An interruption during boot, deploy, bootstrap or restore stops
  the session (teardown) rather than relaunching.
- **No stopped-instance recovery on Spot.** A Spot instance cannot be started again to pull.
- **The credit gate allows about one relaunch.** It counts each instance at its full cap:
  ₹2,467 so far + ₹5,940 + the relaunch's cap must stay under the credits (about ₹14,490). A
  second relaunch is refused unless the earlier instances ran long.
- **The plan still says there is no Spot path.** Section 5.2 (28 Oct 2026) says the launcher
  has no Spot path; an Appendix A line is owed, with the lock's plan hash, before any merge.
- **Tested only on the fakes.** The interruption paths ran on the fake remote only.

## Tests (fake AWS and the fake remote)

- **Launcher, 17 tests:**
  - the pass order;
  - capacity, price-too-low and quota refusals;
  - the quota read (0, 8 or unreadable: no Spot request);
  - the bid cap, P6 and Sydney only.
- **Driver, 14 tests:**
  - an interruption mid-block 2 (by state) and mid-block 3 (by notice), each relaunched,
    restored and resumed to DONE OK;
  - an interruption in block 1;
  - a third interruption stopping the session;
  - a relaunch refused for capacity;
  - the cap left bounding the hard delete and refusing a relaunch that cannot fit;
  - a host mismatch;
  - a failed backend check.
- **Break-tests:** 11 mutations of the money, minutes, quota, bid and refusal code, all killed.

## The command

```bash
cd ~/Desktop/research/attnbench_scaffold
git worktree add ~/Desktop/research/attnbench_p6spot <COMMIT>      # the commit the yes names
cd ~/Desktop/research/attnbench_p6spot
aws login --profile attnbench
ATTNBENCH_LOGIN_AT=$(date -u +%s) caffeinate -i bash scripts/aws_p3_driver.sh P6 spot
```

Start it only after the open on-demand window has ended with its end line: the windows ledger
`~/.attnbench/p6_windows.tsv` is shared, and an open window refuses the next.
