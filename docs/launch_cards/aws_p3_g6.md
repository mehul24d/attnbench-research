# Launch card: P3 (AMD EPYC 7R13 + L4, AWS `g6.2xlarge`, Sydney; EXPLORATORY)

Prepared 23 Nov 2026 (adviser's task of 23 Nov 2026, item 4b); **built 26 Nov 2026** (the
researcher's task of 26 Nov 2026: one session for all blocks on one `g6.2xlarge`, the
rehearsal's launch policy, retry window and teardown); **updated 26 Nov 2026** (the researcher's
second task of 26 Nov 2026: the dense backend, the pre-launch CPU pass, the recovery of a
stopped instance, the credentials' end, the Appendix A lines). **Not launched; the launch yes
is owed.** The rehearsal passed on 25 Nov 2026 (`REHEARSAL_OK`, `docs/launch_cards/aws_rehearsal_g6.md`).
No P3 step has run, so every time below is P1's measurement or an estimate.
P3 is exploratory (Appendix A, 21 Nov 2026): it scores no hypothesis.

## What P3 is for

1. **The host-effect rule, which gates P6** (Appendix A, 21 Nov 2026; `scripts/p6_host_effect.py`,
   `tests/test_p6_host_effect.py`). The same L4 on another host: does P3's host move the host
   components by more than session-to-session and GPU-to-GPU noise? The rule was fixed
   before any P3 component exists: **d > 0.10, P6 runs; d ≤ 0.034, P6 does not run;
   in between, P6 does not run unless the researcher overrules in writing before P3's result
   is read.**
2. Reported with it, not part of the gate: device-side H0 for P3 against P1 at the locked 10%,
   and P3's ground truth against the prediction committed before the session (below).

## The measurement the gate reads

Step 5, `host_components` (T2.2, the bare timer): one `bare_prices.json` per cell under
`05_host_components/<cell>/`, each operator's `bare_s` and `calls`. The gate is

    python3 scripts/p6_host_effect.py <P3 session>/05_host_components \
        attnbench_backups/bucket_20261117/attnbench-l4-t32d-20261109-1237/results/t32d_p1_20261109T0726Z/05_host_components

P1's side is T3.2d (`t32d_p1_20261109T0726Z`, the session P1's training file comes from):
all 35 cells' `bare_prices.json` are on this Mac (the 17 Nov 2026 download). The gate needs
only step 5's files, so it can be computed even if a later step fails.

**No go/no-go after block 1 (adviser's task of 23 Nov 2026, second, item 3).** The gate
cannot be computed from block 1: block 1 holds no host components (`preflight`, `link`,
`dense_select` and `probe`; step 5 is block 2 by itself). And the gate needs all the cells:
the rule fixed on 21 Nov 2026 takes d as the median over every cell P3 and P1 share, so a d
over part of the grid would be a new rule, not this one. The first point at which the gate
can be read is the end of block 2, when step 5's 35 cells have been pulled and checked:
`p6_host_effect.py` runs on the Mac then, before block 3 starts. Block 3 (device components,
the in-session prediction, ground truth) does not feed the gate, so stopping P3 there on the
gate's reading would save about 1.5 h (89 min of steps, about ₹212), at the cost of P3's reported device-side
H0 and its ground truth. **Ruled 23 Nov 2026: the go/no-go is after block 2**, once all 35 cells have host components. There the driver pulls step 5, checks every sha256, runs `p6_host_effect.py` and records d and the verdict (`gate.json`, `gate.txt`) before block 3 starts. **Ruled 26 Nov 2026: the gate's verdict gates P6 only; a skip verdict does not stop the remaining P3 blocks**, so block 3 runs whatever d is. What does stop the session there is a block 2 that cannot be read: a failed step, a pull that does not check, or fewer than P1's 35 cells with host components (the rule takes d over every cell the two share, so a d over part of the grid would be a new rule). Then the driver pulls what there is, tears down and reports.

## Cells: P1's 35

The T3.1 grid of `configs/t31/p1_session.json`, as P1 ran it in T3.2d, at the same prompts:

| Arm | Densities | Lengths (batch 1) | Cells |
|---|---|---|---|
| `dense` | 1.0 | 2048, 4096, 8192, 16384, 32768; and 16384 at batch 4 | 6 |
| `oracle` | 0.25; 0.5 | 2048 to 32768; 2048 to 16384 (no 32768: P1's host-memory drop, Appendix A, 2 Nov 2026) | 9 |
| `meanpool` | 0.25; 0.5 | 2048 to 32768 | 10 |
| `xattention` (`dcal`) | the estimator's | 2048 to 32768 | 5 |
| `xattention_noassert` (`dcal`) | the estimator's | 2048 to 32768 | 5 |

**Plan default, flagged:** P3 keeps P1's drop of the oracle at 0.5 and 32768 although the
`g6.2xlarge` has 32 GiB, so the two cell sets match one for one.

## Steps and blocks

P1's runner (`scripts/t31_session.py`, T3.2d's 12 steps) with pairing P3. One on-demand boot
in three blocks; at the end of each block the Mac's driver pulls that block's step
directories over SSH and checks their sha256 before the next block starts, so a later
loss costs only the block in progress.

| Block | Step | P1 minutes (T3.2d, L4, Intel) | P3 planned minutes |
|---|---|---|---|
| b1 | `preflight` | 1.4 | 1.4 |
| b1 | `link` | 3.0 | 3.0 |
| b1 | `dense_select` | 3.4 | 3.4 |
| b1 | `probe` | 13.9 | 28 (host-bound, doubled for the AMD host, as P6's card) |
| b2 | `host_components` (the gate's input) | 80.4 | 161 (host-bound, doubled) |
| b3 | `device_components` | 50.3 | 50.3 |
| b3 | `predict` | 1.5 | 1.5 |
| b3 | `hold` | 1.3 | 1.3 |
| b3 | `ground_truth` | 29.6 | 29.6 |
| b3 | `verify`, `postflight` | 2.6 | 2.6 |
| b3 | `delay_injection` | 3.7 | 3.7 |
| | **Steps** | **191.1** | **285.8** |

## The dense backend: what the prediction priced and what ground truth runs

A read-only check (the researcher's second task of 26 Nov 2026, item 1). **They differ.**

- **Priced: `sdpa_flash`, at every dense cell.** The prediction committed on 22 Nov 2026 is a
  full transfer from P1 (its `basis`): each cell's value is P1's committed value, and its six
  dense cells carry P1's code path `NVIDIA L4|eager|O0|c3b3ddc3bc31`. Steps 5 and 6 record and
  price the harness's dense arm, `ARMS["dense"]` = `sdpa_flash`
  (`attnbench/components/shakedown_arms.py`); P1's banked step-5 records (T3.2d) show
  `aten::_scaled_dot_product_flash_attention` in all six dense cells. P3's own steps 5 and 6
  run the same pinned code, so P3's in-session prediction prices `sdpa_flash` too.
- **Run: the dense selection's choice, per cell.** Ground truth's dense control is
  `registry_candidates(card)[choice.chosen]` from step 3's record (`t31_steps._real_cells`).
  P1 chose `sdpa_flash` at 2048 and 4096, `sdpa_cudnn` at 8192, and `fa2` at 16384 (batch 1
  and 4) and 32768. On P3's host the rehearsal's one selection (8192, batch 1, 25 Nov 2026)
  chose `fa2` (3.10 ms; `sdpa_flash` and `sdpa_cudnn` within the floor). P3's own step 3 decides
  the rest; its record is pulled with block 1.
- **So** at every dense cell where P3's step 3 chooses other than `sdpa_flash`, the predicted
  and the measured dense cells are of different kernels. This is the fault Appendix A, 17 Nov
  2026, disclosed for P4 (FA3 run, `sdpa_flash` priced). It is disclosed for P3 as an
  implementation fault in the committed prediction (`docs/appendix_a_bundled_wave.md`):
  nothing is rescored or re-predicted, and the pin `0e0fb012` and the 22 Nov file are
  untouched. P3's dense cells are reported with the chosen kernel beside each. **The P6 gate is
  not affected:** it compares the host components of P3 and P1, and both price `sdpa_flash`.

## Time and cost

| | Hours |
|---|---|
| Setup: launch, SSH, deploy, bootstrap on the image | 0.50 |
| Suite record (`--suite-record`, as P2 and P4) | 0.33 |
| The steps | 4.76 |
| Teardown, pull, cleanup check | 0.15 |
| Margin | 0.56 |
| **Planned** | **6.3** (the plan's figure for P3, T3.1's high projection for the L4) |

Rate **₹143.21/h** all-in: (`g6.2xlarge` at $1.27107/h, SKU `A3JD2DB89TQHRAA7`, + 200 GB gp3
at $0.096/GB-month over 730 h) x 1.15 x ₹95.99, both read from the AWS Price List API for
Sydney on 23 Nov 2026. **Cap ₹1,353** (₹143.21 x 6.3 h x 1.5, as Appendix A, 21 Nov 2026), for P3 in all: the relaunch (the researcher's yes of 26 Nov 2026) has ₹1,210, the cap less the aborted first session's ₹143; its worst case is ₹1,201.
**Ruled 26 Nov 2026:** the in-guest halt **stops** the instance at the planned hours plus about
25%, **473 min** (7 h 53 min; ₹1,129); the hard-delete schedule terminates it **30 min later,
at 503 min** (8 h 23 min; worst case ₹1,200.58, which the launcher rounds up to ₹1,201, inside
the cap). These replace the 23 Nov figures (halt 491, delete 566). The planned 6.3 h with its
margin is about ₹902. On any step failure: stop, pull, tear down and report.

**Expected about 4.7 h (about ₹673), an observation only.** The rehearsal (25 Nov 2026) timed
step 5 on one cell (dense, 8192, batch 1) at 50.5 s on this host, against T3.2d's 51.9 s on
P1's: a ratio of about 0.97, not the 2x the planned minutes assume. Scaled to the grid, step 5
would take about 77 min rather than 161 and the probe about 14 rather than 28, so the steps about
3.1 h and the session about 4.7 h. It is one cell, so the planned 6.3 h, the cap, the halt and
the delete stay as above. It changes no estimator, price, prediction or matching rule: the
prediction committed on 22 Nov 2026 is the one P3 is read against.

**Launch deadline: 18:30 UTC on 2 Dec 2026** (ruled 26 Nov 2026). The driver refuses to start a
retry window after it and shortens a window to end by it; no L4 by then means P3 is dropped.

## If SSH is lost

The likeliest cause is this Mac's public address changing (it did on 22 Nov 2026, and on
25 Nov 2026 before the rehearsal's night window). The instance keeps running its block and
keeps its results; nothing is lost until the hard delete.

1. **The driver keeps polling** every minute. Each failed poll re-reads whether the security
   group admits this Mac's current address and says which it is: the network, an expired
   `aws login`, or an address the group no longer admits (re-point it with `sg_repoint.sh`;
   the driver never re-points it). When SSH comes back, the block's pull and the next block
   go on as before.
2. **At the pull deadline**, 10 min before the halt (+463 min), the driver pulls what it can
   and tears down. Whatever was not pulled by then is lost; the blocks already pulled were
   checked when they came.
3. **An instance whose driver has gone halts itself.** When a block ends, passed or not, the
   instance schedules its own halt 30 min later (`shutdown -P +30`, `scripts/aws_p3_instance.sh`),
   and only then writes the block's exit file. The next block cancels it by scheduling the
   planned halt (+473 min from boot) in its place (systemd keeps one pending shutdown; `shutdown -c`
   is never used, since it would cancel the planned halt too), reads the pending shutdown back,
   and keeps a watchdog on it while the block runs (the cancel file `~/p3/<block>.running`). So
   the GPU stops billing within 30 min of a block's end, no idle halt powers off a running
   block, the planned halt is never cancelled or delayed, and the disk is deleted with the
   instance at +503 min. *Rebuilt 26 Nov 2026:* the first launch (25 Nov 2026) stopped itself
   in block 2 because the old idle halt, a `sleep` started with TERM ignored, ignored the next
   block's `kill` (`tests/test_p3_halts.py`). When SSH drops, the driver reads the instance's
   state and says at once if it has stopped.
4. **A stopped instance is started once, only to pull, before the delete** (the researcher's
   launch-yes wording of 26 Nov 2026). This is manual, not in the driver; see the next section.
5. **No instance-side upload.** No S3 bucket and no instance role (adviser's task of
   23 Nov 2026, second, item 3): results leave the instance only by the Mac's pulls over SSH.

## If the instance stopped before the last pull: the manual recovery

When: EC2 reads the instance `stopped` (it halted itself, 30 min after a block's end or at
+473), the driver has gone or its teardown failed, some block's directories were never pulled,
and it is at least 20 min before the delete time in the driver's `LAUNCHED` line. Once only, to
pull data only: no step is run. A restart bills only until the hard delete at +503, which the
worst case (₹1,201) already counts. The in-guest halt was scheduled by first-boot user data,
which does not run again, so step 2 sets a halt of its own. If the group no longer admits this
Mac's address, the researcher runs `sg_repoint.sh` first. From the worktree:

```bash
REC=~/Desktop/research/attnbench_backups/aws_p3_g6_<UTC>     # the driver's record
ID=$(cat "$REC/instance_id.txt"); source ~/.attnbench/aws_env.sh
R=(--profile attnbench --region ap-southeast-2)
aws ec2 describe-instances "${R[@]}" --instance-ids "$ID" \
  --query 'Reservations[0].Instances[0].State.Name' --output text        # must say: stopped
# 1. Start it, once.
aws ec2 start-instances "${R[@]}" --instance-ids "$ID" --query 'StartingInstances[0].CurrentState.Name' --output text
aws ec2 wait instance-running "${R[@]}" --instance-ids "$ID"
DNS=$(aws ec2 describe-instances "${R[@]}" --instance-ids "$ID" \
  --query 'Reservations[0].Instances[0].PublicDnsName' --output text)    # a new address; not printed
O=(-i "$AWS_SSH_KEY" -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$REC/known_hosts")
# 2. Its own halt first, so a lost Mac cannot leave it running to the delete.
ssh "${O[@]}" ubuntu@"$DNS" 'sudo -n shutdown -h +45'
# 3. Pack every step directory with each file's sha256, as the driver's last pull does.
ssh "${O[@]}" ubuntu@"$DNS" 'cd ~/p3_session && find [0-9][0-9]_* manifest.json -type f | LC_ALL=C sort > ~/p3/recover.files && tr "\n" "\0" < ~/p3/recover.files | xargs -0 sha256sum > ~/p3/recover.sha256 && tar cf ~/p3/recover.tar -T ~/p3/recover.files && sha256sum ~/p3/recover.tar'
mkdir -p "$REC/pull/recover" "$REC/session_recovered"
scp "${O[@]}" "ubuntu@$DNS:~/p3/recover.tar" "ubuntu@$DNS:~/p3/recover.sha256" "ubuntu@$DNS:~/p3/*.log" "ubuntu@$DNS:~/p3/*.exit" "$REC/pull/recover/"
shasum -a 256 "$REC/pull/recover/recover.tar"                          # must equal the line step 3 printed
tar xf "$REC/pull/recover/recover.tar" -C "$REC/session_recovered"
(cd "$REC/session_recovered" && shasum -a 256 -c ../pull/recover/recover.sha256 > ../pull/recover/check.txt; grep -vc ': OK$' ../pull/recover/check.txt)   # must print 0
# 4. Stop it, and let the hard delete terminate it.
aws ec2 stop-instances "${R[@]}" --instance-ids "$ID" --query 'StoppingInstances[0].CurrentState.Name' --output text
aws ec2 wait instance-stopped "${R[@]}" --instance-ids "$ID"
# 5. After the delete time: the instance terminated and Sydney clean.
bash scripts/aws_cleanup_check.sh
```

A recovered directory is kept beside the driver's `session/`, never merged into it. Each block
is read only if its steps passed in the recovered `manifest.json`. **Not tested (flagged):** these
commands have not been run, on the fake remote or on AWS; the pack, the tar's sha256 and the
per-file check are the driver's `pull` (tested), typed out.

## The credentials

`aws login` credentials rotate every 15 min inside a session that ends at most 12 h after the
login (the AWS CLI user guide, "Login for AWS local development using console credentials").
The window and the run need 3 h + 503 min = 11.4 h. So at the start of each window the driver
logs when the session ends (`credentials.tsv` and the log): the login time the launch command
stamps (`ATTNBENCH_LOGIN_AT`) plus 12 h, or an earlier session expiry if the login cache holds
one more than 1 h ahead (only expiry timestamps are read from it; nothing else is printed). It
**warns** when that is under 11.5 h away (a login more than 30 min old), and does not refuse. An
expiry during the run makes the teardown fail; the instance then halts at +473 and is deleted
at +503, and the recovery above applies. **Plan default, flagged:** a principal whose configured
session is shorter than 12 h is not detected by the login-time rule.

## The prediction committed before any measurement

`docs/p3_prediction/prediction_P3_exploratory_from_P1_20261122T211249520458Z.json` (sha256 `72b907cf4605336bba91ef3f04f8503945e49dabef207d4e3c4fc7b26183e1b2`, written 22 Nov 2026 21:12 UTC from `5b20daa`, 35 cells) with its sha256 sidecar,
written by `scripts/p3_prediction_from_p1.py` from P1's committed T3.2d prediction
(`prediction_P1_20261109T095833754749Z.json`, sha256 `280a3423…`, checked against its
sidecar before it is read), and committed to this repository before any P3 session; its
receipt is the time its commit was pushed to `origin` (adviser's task of 23 Nov 2026,
second, item 3), which must come before any P3 launch. It is
the full transfer from P1: with no P3 component in existence, each P3 cell's prediction is P1's
committed value for that cell (the sum, the step form, the recursion, the sum plus s0).
**EXPLORATORY**: it scores no hypothesis; it tells how far a host change moves the
end-to-end time when nothing about P3 is measured.

## What was built (26 Nov 2026)

- **The session config, `configs/t31/p3_session.json`.** Pairing P3, P1's grid and arms. The
  pinned step code (`t31_steps.load_config`, unchanged) gives a pairing other than P1 its T4.1
  items from the lock and refuses P1's working values for it, so the config leaves those items
  null and they come from `docs/lock_values.json`, as for P2 and P4: the densities, the grid,
  the throttle set, the rerun budget (5 rounds, 10 clean), P1's resolution floors, and the
  chain, residual and pricing tolerances. The lock holds no P3 block, so the four values it
  holds for P2 and P4 alone (the eager group only, 10 reps, 5 dense rounds, a 0.1 s sampler),
  identical for both, are copied into the config. Loaded, it has P1's 35 cells, one for one.
  **Plan default, flagged:** P3 so runs under the rules every later pairing ran under, not
  T3.2d's working values (for example, the dense selection's floor is the lock's P1 table, not
  a floor from the session's own dense pre-pass).
- **Blocks:** `scripts/t31_session.py --until STEP` ends a run after STEP; the next block
  resumes with `--from` on the same session. `t31_session.py` is outside the timed-path pin.
- **The instance's block runner, `scripts/aws_p3_instance.sh b1|b2|b3`:** the runner with no
  bucket sync, the exit marker, and the 30-minute halt after a block's end (above).
- **The prediction store, `scripts/aws_p3_gsutil.py`.** The pinned `predict` step (and step
  12's leg-3 plan) sync through `commit.GsutilStore`, which runs `gsutil`. On the instance a
  `gsutil` first on PATH runs this file, and `ATTNBENCH_BUCKET` is `gs://p3-mac`: each object
  is spooled, and the Mac's driver pulls it, checks its sha256, keeps its copy, sends that copy
  back and writes a witness with the time on the Mac's clock. `gsutil stat` then reports that
  time as the creation time, which `hold` waits from, and `cat` returns the Mac's copy, which
  `verify` reads. With no witness within 15 min (SSH lost at that moment) the sync fails and the
  session stops before any ground truth, as the 23 Nov card planned.
- **The driver, `scripts/aws_p3_driver.sh`:** the launch windows and their ledger
  (`~/.attnbench/p3_windows.tsv`: a new window only after every earlier one created nothing,
  none after a launch), the SSH preflight, the launcher with the settings above, SSH, the
  deploy (a git bundle of HEAD), the bootstrap, then the three blocks. After each block it
  pulls that block's step directories as one tar and checks the tar's sha256 and every file's
  before the next block starts. It runs the gate after block 2, and does the rehearsal's
  teardown (retried until EC2 reads the instance terminated, then the cleanup check). It runs
  from a copy of itself, so the tree can be edited while it runs.
- **The RULER data and the two-package install: already a repository script with tests**
  (`scripts/aws_bootstrap_instance.sh`, `tests/test_aws_bootstrap_instance.py`, 23 Nov 2026). It
  verifies the RULER data against its pins, installs wonderwords and nltk at their pins and
  lightgbm, and checks the word pool, the tokenizer and a clean tree. It passed on the
  rehearsal's instance (`BOOTSTRAP_OK`, 25 Nov 2026).
- **The preflight on AWS: built 23 Nov 2026.** `scripts/t227_steps.py` accepts
  `ATTNBENCH_EXPECTED_HOST=aws:g6.2xlarge` (`tests/test_t227_expected_host.py`), which
  `aws_p3_instance.sh` sets.

**Tests** (each suite run alone; the pin `0e0fb012` unchanged):

- `tests/test_aws_p3_driver.py`: the full P3 flow on a fake remote. It runs the real driver,
  the real `aws_p3_instance.sh`, the real `t31_session.py` on this config, and the real
  `GsutilStore` through `aws_p3_gsutil.py`, with stub step bodies. The flow is checked with
  both gate verdicts, and it covers:
  - a network drop that never ends (the address changed; torn down at the pull deadline; the
    instance halts itself; the group never re-pointed);
  - a drop that ends;
  - a failed step;
  - a block 2 short of 35 cells (no-go);
  - the window ledger and the deadline;
  - the refusals.
- `tests/test_aws_p3_gsutil.py`: the store against the real `GsutilStore` and
  `sync_prediction`.
- `tests/test_p3_session.py`: the config, the blocks, the pin, and the committed prediction's
  sha256.
- `tests/test_aws_launch_session.py::test_the_p3_cards_settings`: the real launcher on P3's
  settings.

Two mutations were each caught: the gate accepting 34 cells, and the halt at 491.

**Built in the second task of 26 Nov 2026:**

- **The pre-launch CPU pass for P3:** `scripts/prelaunch_cpu_pass.sh p3`
  (`tests/test_prelaunch_cpu_pass_p3.py`). It runs `t31_session.main` as
  `aws_p3_instance.sh` runs it: the real config, the armed exit wrapper, the clean-tree
  refusal and the three blocks on one session. Every step's real body runs, with the real
  hold and the real P3 store (`aws_p3_gsutil.py` with a witness thread). The P6 gate script
  then reads block 2's real output (35 cells). Mocked, and only these:
  - probe's and postflight's dry bodies;
  - T2.27's GPU probes;
  - the kernel contract;
  - the dense selection's timing (a fixed `fa2`);
  - the GPU boundary of the components and ground truth;
  - leg 3's timing.

  It checks that a pairing other than P1 takes the lock's values: every T4.1 item from
  `docs/lock_values.json` (its sha256 included), the pairing keys from the lock's P2 and P4
  values, and none of P1's working values. Six refusals of that check are tested. The
  record is cited in the launch's permission-log row.
- **The credentials' end, logged at each window's start** (above), with six tests on the fake
  remote; a mutation of the 11.5 h threshold was caught.
- **The manual recovery of a stopped instance** (above), not tested.

**Still not built (flagged):** the suite record on the instance. It is made after the run,
on the Mac at the deployed commit, and committed before the push
(`docs/appendix_a_bundled_wave.md`); its 0.33 h above becomes margin.

## Budget

Against the AWS cap of about ₹10,400 (Appendix A, 17 Nov 2026): the caps in the ledger's AWS
list, with the rehearsal at ₹180, come to about ₹9,050 with one month of the snapshot,
leaving about ₹1,350; P6's ₹5,940 is in that sum and is spent only if the rule says run.
Credits left: $159.79 (Free Tier API, 23 Nov 2026), about ₹15,338.
