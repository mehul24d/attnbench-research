# Spend ledger

The standing rule is to report elapsed instance time and estimated spend on
every report. Until 2026-09-04 that total was accumulated **in prose**, carried
forward from one report to the next — and it drifted: ₹902, ₹910, ₹918 and
₹942 were all in circulation for the same point in the project, a spread of
about 4%.

Nothing here is load-bearing for the science, and none of the decisions taken
would change at any figure in that range. But a running total maintained by
restating it is the same shape as the failures this project keeps finding: a
number that looks authoritative, is never checked against its source, and has
no mechanism that would notice it going wrong.

So from 2026-09-04 the figure comes from data. `scripts/gcp_teardown_session.sh`
reads the instance's own boot clock and writes `session_cost.txt` into the
session's results directory:

```
instance: attnbench-flex-recheck-20260904-0149
zone: asia-south1-b
boot: 2026-09-04 20:19:32
teardown: 2026-09-04T20:25:10Z
minutes: 6
rate_inr_hr: 80
est_inr: 8
```

Both the boot timestamp and its epoch are computed **on the instance**. This
workstation runs in IST and the instances in UTC, so converting `uptime -s`
locally would be wrong by 5h30m — and wrong in a way that still produces a
plausible-looking duration.

`est_inr` is an estimate at a fixed rate (`GCP_RATE_INR_HR`, default 80 for
g2-standard-8 + 1× L4 on-demand in asia-south1), not billed cost. It ignores
disk, egress, and machine-image storage, which are small but not zero. The
authority is the billing console.

## Sessions

| date | session | minutes | est ₹ | source |
|---|---|---|---|---|
| 2026-09-02 | validation | — | — | prose only, not reconstructable |
| 2026-09-03 | compile session 3 (flash-attn, v3) | — | ~209 | prose only |
| 2026-09-03 | session 4 (32K baseline, BSA, v4) | — | ~113 | prose only |
| 2026-09-03 | Stage 2 segment 1 — aborted attempt | ~12 | ~16 | prose only |
| 2026-09-03 | Stage 2 segment 1 | ~67 | ~90 | prose only |
| 2026-09-04 | flex recheck diagnostic | **6** | **8** | `session_cost.txt` |
| 2026-09-05 | `test-instance` ×4 — **test suite side effect** | **162** | **218** | audit log |
| 2026-09-05 | `attnbench-a100-20260905-1014`, `-1038` — insert **refused** (NVMe disk interface unsupported on a2) | **0** | **0** | audit log |
| 2026-09-05 | `attnbench-a100-20260905-1111` — Ampere confirmation, Stage 0/1 (543 rows @ `d2d8ceb`), **Spot** | **166** | **~888** | audit log; rate see note |
| 2026-09-06 | `attnbench-stage3-s1` — Stage 3 S1, band 2048 | **651** | **868** | instance boot clock |
| 2026-09-07 | `attnbench-stage3-s1b` — bands 4096 + 8192 | **341** | **455** | boot→guestTerminate |
| 2026-09-07 | `attnbench-recover-1628` — e2-medium data recovery | **~25** | **~5** | CPU-only, no GPU |
| 2026-09-08 | `attnbench-decode-unconfound` — L4, asia-south1-c | **59** | **77** | audit log |
| 2026-09-08 | `attnbench-band16384` — L4, asia-south1-c | **65** | **84** | audit log |
| 2026-09-08 | `attnbench-band16384` — **idle, halt→delete** | **162** | **0** | audit log |
| 2026-09-08 | `attnbench-band32768` — L4, asia-south1-b | **83** | **108** | audit log |
| 2026-09-12 | `attnbench-h100-probe-115828` — capacity probe | **<1** | **~3** | audit log |
| 2026-09-15 | `attnbench-h100-20260915-2142` — **total data loss** | **287** | **2,033** | audit log |
| 2026-09-15 | `attnbench-h100-…-0302` — write gate refused (IAM) | **2** | **14** | audit log |
| 2026-09-15 | `attnbench-h100-…-0310` | **21** | **149** | audit log |
| 2026-09-15 | `attnbench-h100-…-0404` — kernel isolation | **25** | **177** | audit log |
| 2026-09-16 | `attnbench-h100-…-0744` | **20** | **142** | audit log |
| 2026-09-16 | `attnbench-h100-…-0822` — Stage 0/1/2 | **32** | **227** | audit log |
| 2026-09-16 | `attnbench-l4-20260916-1319` — Stage 5 replication | **27** | **35** | audit log |
| 2026-09-16 | `attnbench-l4-20260916-1501` — forced-sink + estimator arms | **246** | **320** | boot→delete |
| 2026-09-16 | `attnbench-a100-20260916-1926` — Stage 5, 2nd architecture | **56** | **265** | boot→delete |
| 2026-09-16 | `attnbench-a100-20260916-2307` — A100 block-sparse Stage 2 sweep (107 rows); v6 capture abandoned | **15** | **71** | audit log |
| 2026-09-17 | `attnbench-a100-20260917-1248` — Stage 0/1 at (12,2), matched-geometry kernel, 7B oracle; **90.6 min idle** | **212** | **1,005** | operations log |
| 2026-09-17 | `attnbench-a100-20260917-item4` — vectorised end-to-end; v6 captured | **10** | **49** | `gcp_session_elapsed.sh` |
| 2026-09-17 | `boottest-attnbench-env-v6-20260917` ×2 — e2-medium, CPU-only | **~8** | **~1** | CPU-only, no GPU |
| 2026-09-17 | `attnbench-a100-20260917-cheap7b` — 7B cheap arm | **53** | **249** | `gcp_session_elapsed.sh` |
| 2026-09-19 | `attnbench-l4-s1a-20260919-1939` — audit S1a, band 2048 (forced sink, decode pinned, canary 300/300) | **39** | **52** | `gcp_session_elapsed.sh` |
| 2026-09-20 | `attnbench-l4-s1a2-20260920-0437` — audit S1a, bands 4096 + 8192; asia-south1-c stocked out, ran in `-b` | **108** | **143** | `gcp_session_elapsed.sh` |

**Two sessions were missing from this table until 2026-09-19**, found by
listing every `instances.insert` in the audit log rather than trusting the
table to be complete: the 2026-09-05 A100 session (well documented as
*science* in `limitations.md` and pattern notes, never ledgered as *spend*)
and the 2026-09-16 `-2307` sweep. Together ~₹959. The 2026-09-05 row is priced
at the Spot rate pinned on 2026-09-16 (₹321/h), applied retroactively because
no rate was recorded that day — so it is an estimate to perhaps ±10%, and the
billing console is the authority. **A ledger assembled from sessions someone
remembered to write down is a ledger with the same failure as a total
maintained by restating it**; the audit log is the source.

A100 `a2-ultragpu-1g` DWS Flex Start ₹284/h all-in (asia-southeast1, verified
2026-09-16: GPU $2.277654 + 12 × $0.023340 core + 170 × $0.003128 RAM).
Rates: H100 `a3-highgpu-1g` DWS Flex Start ₹425/h (us-central1, pinned
2026-09-16); L4 `g2-standard-8` ₹78/h. The 2026-09-07 row implies ~₹80/h for
L4 in asia-south1, so the 2026-09-08 rows are within ±3% of that.

Every row from 2026-09-08 on is reconstructed from
`gcloud logging read protoPayload.methodName=…instances.insert/delete/
guestTerminate`, pairing the **successful** insert (a zone walk logs one
insert per zone tried; only the placed one bills) with its terminating event.
Independent confirmation that the method is sound: the 2026-09-15 row comes
out at 287 min × ₹425/h = **₹2,033**, against ₹2,029 derived at the time from
the instance boot clock — a 0.2% agreement between two unrelated sources.

### The 2026-09-05 row

Four `g2-standard-8` + L4 instances in `asia-southeast1-c`, created and deleted
across 05:53–10:22 UTC. **No session ran on any of them.** They were created by
`tests/test_launch_script_zone_retry.py`, which ran the real launcher against
the real project on every full-suite run — see silent-failure instance 12.

| life | window (UTC) | minutes | est ₹ |
|---|---|---|---|
| 1 | 05:53:53 → 06:16:44 | 22.9 | 31 |
| 2 | 06:23:35 → 07:43:05 | 79.5 | 107 |
| 3 | 07:53:53 → 08:48:41 | 54.8 | 74 |
| 4 | 10:17:19 → 10:21:50 | 4.5 | 6 |
| | | **161.7** | **218** |

Rate $0.916/h (8 vCPU + 32 GiB + 1× L4 in Singapore, plus a 200 GB disk) at
₹88/$. Durations are from the Cloud Audit Log's insert/delete timestamps, which
is the same principle as reading the instance's own boot clock: the authority
is the remote record, not anything reconstructed locally.

**This was reported as ₹0 for most of the day**, while being investigated as
someone else's unexplained spend. It is recorded here as project spend because
that is what it is. The ledger exists because a total maintained by restating
it drifts; a total maintained by *misattributing* it is the same failure with a
worse cause.

The pre-09-04 rows are what was reported at the time. They are recorded as
estimates rather than silently promoted to facts, and they are why the
project total is best stated as **≈ ₹1,120–1,170** rather than to the rupee
(₹900–950 of session work plus the ₹218 above). The
instances are deleted, so those durations cannot be recovered from anything
but the billing console; if the exact figure ever matters, that is where to
get it, not from this file's older rows.

Machine images are billed for storage independently of any instance and are
**not** in the table above: `attnbench-l4-image-v3-20260903` (21.5 GB) and
`-v4-20260903` (22.0 GB). v3 is superseded by v4 and is a candidate for
deletion.

### The 2026-09-06 row — ~7 of 11 hours were idle

`g2-standard-8` + L4 in `asia-south1-b`, boot 15:53:40Z, deleted 02:45:04Z.
**651 minutes, ₹868** from `session_cost.txt`, which reads the instance's own
boot clock.

Band 2048 completed and banked 4500 valid rows. But the work inside those 651
minutes was:

| | |
|---|---|
| Phase 0–2 (launcher fix, deploy fix, clock-lock fix, gates) | ~1.6 h |
| band 2048 generation (sum of `latency_ms` over 4500 rows) | **1.75 h** |
| — of which the 3600 rows that survived `INVALID_ROWS.md` | **1.40 h** |
| importance scoring, 900 examples at ~0.9 s | ~0.25 h |
| **idle, after the band finished and before teardown** | **~7.2 h** |

**Roughly ₹575 of the ₹868 bought nothing.**

The cause is not a mis-estimate — the band's compute estimate was 1.71–1.97 h
and it measured 1.75 h, the most accurate estimate this project has made. The
cause is that the session ended a turn with the instance running, relying on a
background watcher to report the band boundary. The watcher fired correctly.
Its notification could only be *delivered* when the session next became
active, which was the next morning.

**A watcher that fires into an inactive session is not a watcher.** The
standing rule — never end a turn with an instance running — exists precisely
because the notification channel is not the billing channel, and only one of
them keeps running overnight.

**The structural fix is on the instance, not in the supervision.** A long run
must arm its own teardown on completion, so the idle window is bounded by the
machine rather than by whether anyone is awake:

```bash
python3 -u scripts/run_accuracy.py ... ; sudo shutdown -h +5
```

That converts an unbounded idle into a 5-minute one. The 660-minute cap was
armed and would have fired at 02:54 — it did its job as a backstop, but a
backstop sized for the whole session cannot bound an idle window inside it.


### The 2026-09-07 row — the self-teardown worked, and then the zone stocked out

Boot 04:14:08Z, halted 09:55:49Z. GCP logged
`compute.instances.guestTerminate — "Instance terminated by guest OS shutdown."`
**341 minutes, ₹455**, against a booked bracket of ₹581–641. Band 4096 ran
105 min against 143 projected; band 8192 ran 182 min against 241.

The chained `; sudo shutdown -h +5` fired on its own with nobody watching.
That is the direct fix for the ₹575 idle burn of 2026-09-06, and it is now
demonstrated rather than argued.

**What it did not cover:** `shutdown -h` halts, it does not delete, and the
results were still on the disk. The restart to copy them off hit a
`ZONE_RESOURCE_POOL_EXHAUSTED` stockout — the same zone that had capacity six
hours earlier would not give the L4 back. Recovery, without retrying the
failed start:

1. `set-disk-auto-delete --no-auto-delete` **first**, so nothing downstream
   could destroy 7200 rows.
2. Delete the instance; the disk survived standalone.
3. `e2-medium` (no GPU, so no stockout risk, ~₹3/h), attach the disk
   `--mode=ro`, mount `norecovery`, copy, delete everything.

**The lesson is about the gap between halt and delete.** A self-halt bounds
the compute bill and leaves the data hostage to whatever capacity exists when
you come back for it. The cheap fix is to sync results off at each band
boundary to somewhere off-instance (GCS), not merely to a second file on the
same disk — the band-boundary copies existed and were on the disk that could
not boot.


### The 2026-09-15 row — ₹2,033 for zero rows, and the three gates that followed

The largest single loss in the project, and none of it was a compute failure.
The instance ran 287 minutes, completed real work, and wrote **nothing**
durable. Three causes compounded:

1. **No OAuth scope for GCS.** The instance was created without
   `--scopes=cloud-platform`, so every `gsutil cp` to the results bucket
   returned 403. Nothing checked this before the long phase started.
2. **Results synced at teardown, not per phase.** The single sync was
   scheduled for the end, so the 403 surfaced once, after everything.
3. **`--max-run-duration` with `action=DELETE`** took the disk with the
   instance, so there was no post-hoc recovery of the kind that saved the
   2026-09-07 session.

A bound on spend is not a bound on loss. The ₹2,033 was capped exactly as
designed; what was uncapped was the *work*, and that is the quantity that
actually matters.

**The three gates now standing, in order:**

1. `scripts/gcp_verify_gcs_writable.sh` — writes a token from the guest,
   reads it back, `cmp`s it, inspects the live scope list, and independently
   `gcloud storage ls` from the client. Non-zero exit means tear down
   immediately. It **earned its place on its first use**: the next instance
   passed the scope check and still failed, because the two permission layers
   are independent and the service account lacked
   `roles/storage.objectAdmin`. Two systems, both returning 403, and only a
   real round trip distinguishes them. That instance was destroyed after 2
   minutes for ₹14 — the gate's entire cost, against ₹2,033 without it.
2. `scripts/run_phase.sh` — syncs from an EXIT trap, so a phase that crashes
   still ships what it produced. Demonstrated 2026-09-16: the cheap-estimator
   arm died on a `NameError` thirty seconds in, and the trap still synced the
   score cache.
3. **A recovery window.** The in-guest halt is set well short of the
   `max-run-duration` DELETE (210 min against 5 h on H100; 300 min against
   7 h on L4), so there is an interval in which billing has stopped but the
   disk still exists.

### The 2026-09-08 idle row — 162 minutes between halt and delete

`attnbench-band16384` self-halted at 08:58:45Z and was not deleted until
11:40:51Z. **Billed ₹0** — a halted instance stops GPU billing, which is what
the `; sudo shutdown -h +5` discipline from 2026-09-06 was for, and it worked.
The row is in the table at zero cost because the gap is worth seeing: the
mechanism that protects the bill leaves the *disk* in a state that depends on
capacity existing when you return for it, which is exactly what stocked out
on 2026-09-07.

### Idle burn is still the dominant recurring waste

Three instances of the same shape now:

| date | idle | cause |
|---|---|---|
| 2026-09-06 | ~7 h of 11 | no self-teardown; nobody awake |
| 2026-09-08 | 162 min | halted (₹0 billed), delete deferred |
| 2026-09-16 | ~54 min (₹70) | **the completion watcher could not fire** |

The 2026-09-16 instance is the one worth reading. The phase finished at
11:32:35Z and synced correctly; it simply was not *noticed* until 12:26Z. The
completion marker was an `echo "rc=$?"` whose output crossed an ssh pipe and a
`grep` with no `--line-buffered`, and never arrived. A separate watcher on the
process table could not help either — `pgrep -c -f <pattern>` run over ssh has
a floor of **1**, because the probe's own command line contains the pattern,
so its terminal condition was unreachable (see
`silent_failure_patterns.md` #38).

**The fix generalises: prefer a signal the watched thing emits over a signal
the watcher derives, and make it a file rather than a stream.** The phase now
writes its exit status to `/tmp/<phase>.rc` on the instance directly. A file
write cannot be buffered away, cannot match itself, and survives the ssh
session that started it.


### The 2026-09-16 A100 row — DWS was cheaper than Spot on a second accelerator

₹265 for 56 minutes, of which the Stage 5 run itself was 16. Five bands, all
`rc=0`, per-band sync, torn down 34 minutes before the in-guest halt.

**DWS Flex Start was attempted first and succeeded**, which settled a question
the quota API cannot answer: DWS draws on `PREEMPTIBLE_NVIDIA_A100_80GB_GPUS`
(the project's only non-zero A100 quota, limit 1). So the Spot fallback never
ran, and the session was **non-preemptible at ₹284/h instead of preemptible at
₹321/h**.

| accelerator | DWS | Spot | on-demand | DWS vs Spot |
|---|---|---|---|---|
| H100 80GB | $4.200761 | — | $9.80–12.74 | **41% cheaper** |
| A100 80GB | $2.277654 | $2.648100 | $4.846072 | **14% cheaper** |
| L4 | $0.560040 | — | $0.560040 | **identical** |

Two accelerators where the bounded-window product is cheaper *and* stronger
than the preemptible one, and one where it is neither. **The rule is "price the
SKU per accelerator", not "DWS is always cheaper"** — the L4 row is what stops
this from becoming a heuristic that quietly costs money.

**Where the 56 minutes went:** 16 in Stage 5, and roughly 25 in boot, repo
sync, the stale-`results/` quarantine and model load. On a 16-minute workload
the fixed setup cost is larger than the measurement. For a session this short
the lever is a warmer image, not a faster card.

---

## Session 10 — 2026-09-17, A100 (`attnbench-a100-20260917-1248`), DWS Flex Start, ₹284/h

**₹1,005 for 3.54 h, of which roughly ₹540 — 54% — was an idle GPU.** That is
the headline and it is a process failure, not a hardware one.

Authoritative timeline from `gcloud compute operations list` (the instance's
own record, not scrollback):

| event | UTC | source |
|---|---|---|
| insert | 07:18:35 | `insert` |
| write gate passed | 07:20:19 | log |
| items 2, 1, 3 complete | 07:32:24 | log |
| item 4 launched (first) | 07:55:52 | log |
| 7B oracle 16384 complete, 800 rows | 08:59:22 | log |
| item 4b guard refused | 09:20:15 | log |
| **in-guest shutdown fired** | **10:50:51** | `guestTerminate` |
| **max-run-duration DELETE** | **11:48:52** | `deferredDelete` |

**The teardown structure worked exactly as designed.** The in-guest halt fired
58 minutes before the API-level DELETE, which is the disk-recovery window the
rule exists to create. Nothing was lost: every phase had already synced to
GCS, and the cleanup check afterwards shows zero instances, zero disks, zero
snapshots.

**What failed was attention to the clock.** Two idle stretches:

| stretch | duration | cost | cause |
|---|---|---|---|
| 07:32 → 07:56 | 23.5 min | ₹111 | items 1–3 were chained; item 4 had not been written yet |
| **09:20 → 10:51** | **90.6 min** | **₹429** | item 4's guard refused, and the diagnosis was done locally while the GPU sat idle |

The second one is the expensive lesson. After the guard fired, the work that
followed — reproducing the failure, measuring fp32 ulp against the jitter,
writing two tests, running the suite three times — was all CPU-local and none
of it needed the A100. **The instance should have been deleted the moment the
guard refused, and re-created when there was something to run on it.** A
diagnosis that runs on a laptop is not a reason to hold a ₹284/h accelerator.

Compounding it: elapsed time was reported from arithmetic on remembered
timestamps rather than read from the clock, so progress updates claimed
"elapsed 2.0 h" while the true figure kept drifting further ahead. The
instance's own operations log was the correct source and was not consulted
until after deletion.

**What the session actually produced** (all banked, all verified readable
from GCS after deletion):

| item | result |
|---|---|
| 1 — Stage 0/1 at `(12,2)` | 149 rows; `block_sparse` 27/27 incl. cross-backend at 8192 and 16384 |
| 2 — instance-CPU construction | instance CPU uniformly ~3.7× slower than laptop |
| 3 — matched-geometry kernel | 53 cells; corrected a published table |
| 4 — vectorised end-to-end | **no data**; guard refused twice, both times correctly |
| 5 — 7B download | rc=0, 85 s |
| 6 — 7B oracle 16384 | 800 rows, `clocks_locked=True`, `git_dirty=False` |

**Two guards fired and both were load-bearing.** The GLA arm gate refused the
7B run in two seconds over an omitted flag whose verdict was already banked,
where the alternative was raising after a full band had been paid for. The
mask-equivalence guard refused to time a builder it could not prove
equivalent — twice, for two different and both-real reasons (fp16 tie-breaking
at 8192, then an fp32 near-tie at 16384 that is inside the reference's own
jitter).

**Clock lock succeeded on the A100 and is stamped `clocks_locked=True`** on
the 7B accuracy rows — the first accuracy band in the project measured with
clocks pinned. Item 3's kernel cells remain `clocks_locked=False` with
`sm_clock_mhz_at_capture=210` (idle), consistent with every Stage 2 row on all
three cards, so the matched-geometry cells inherit that caveat and are no
tighter than the L4 numbers they are compared against.

**The v6 image capture was planned for this session and not attempted.**

---

## Session 11 — 2026-09-17, A100 (`attnbench-a100-20260917-item4`), DWS Flex Start, ₹284/h

**₹49 for 10 minutes.** Boot 12:16:18Z, deleted 12:26:44Z. Launched with
`GCP_MAX_RUN=1h30m` / `GCP_HALT_MINUTES=55` rather than the 4h30m/3h30m
defaults — the cap sized to the work, which is the correction session 10
earned at ₹429.

| step | at | outcome |
|---|---|---|
| write gate | 12:17:28Z | PASS, round trip verified both directions |
| deploy | 12:17:52Z | `ae736b4`, clean tree |
| item 4 (both bands) | 12:18:32 → 12:20:45Z | **rc=0** |
| v6 capture | 12:26Z | `attnbench-env-v6-20260917`, READY |
| delete | 12:26:44Z | immediate |

**Item 4 took 2 minutes 13 seconds of GPU time**, and settled a question three
prior sessions had approached by composition. Elapsed and spend were read from
`scripts/gcp_session_elapsed.sh` throughout rather than recalled.

**v6 boot-tested separately on an `e2-medium`**, CPU-only, ~₹1: boots, sshd at
~40 s, repo at `ae736b4` with a clean tree, all 7 tracked files under
`results/` present, torch 2.9.1+cu129 / transformers 4.46.0 / flash_attn
2.8.3.post1 importable. The first boot-test script reported a pass on **empty
output** and was fixed to refuse unless it counts at least 7 `CHECK` lines —
recorded because it is the same defect class the script was written to catch,
appearing in the script itself.

**Two-session total for 2026-09-17: ₹1,054.**

---

## Session 12 — 2026-09-17, A100 (`attnbench-a100-20260917-cheap7b`), DWS Flex Start, ₹284/h

**₹249 for 53 minutes.** Boot 13:33:59Z, deleted 14:26:29Z. First session to
boot the **v6** image, with `GCP_MAX_RUN=1h45m` / `GCP_HALT_MINUTES=75`.

| step | at | outcome |
|---|---|---|
| write gate | 13:36:04Z | PASS |
| deploy | 13:36:20Z | `0397c60`, clean tree |
| 7B download | 13:36:38Z | rc=0 |
| 7B cheap arm, 16384 | → 14:21:58Z | **rc=0**, 800 rows |
| delete | 14:26:29Z | immediate |

**v6 paid for itself on the first boot: 2 minutes 5 seconds from instance
creation to write-gate-passed**, against the ~25 minutes a cold v5 setup cost.
It was boot-tested on an `e2-medium` beforehand (~₹1) rather than trusted
because `gcloud` called it READY.

**The result inverted the hypothesis that motivated the session.** The cheap
estimator's disadvantage against the oracle *widens* with scale on
`niah_multikey` (+40 → +67 at 0.75, +19 → +76 at 0.9), so the oracle
requirement is not a small-model artifact. That is the more consequential of
the two possible outcomes and the less convenient one.

**Day total: ₹1,303 across four sessions** (10 → 11 → 12 plus the ~₹1
boot-test), against ₹1,005 for session 10 alone. The three sessions after the
idle-burn correction cost ₹299 combined and produced three results.
