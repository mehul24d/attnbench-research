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
