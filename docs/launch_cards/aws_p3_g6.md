# Launch card: P3 (AMD EPYC 7R13 + L4, AWS `g6.2xlarge`, Sydney; EXPLORATORY)

Prepared 23 Nov 2026 (adviser's task of 23 Nov 2026, item 4b). **Not launched; for the
adviser's approval.** It needs the `g6.xlarge` rehearsal (`docs/launch_cards/aws_rehearsal_g6.md`)
to pass first. No P3 step has run, so every time below is P1's measurement or an estimate.
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
Sydney on 23 Nov 2026. **Cap ₹1,353** (₹143.21 x 6.3 h x 1.5, as Appendix A, 21 Nov 2026).
The in-guest halt **stops** the instance at **491 min** (1.3 x 6.3 h; ₹1,172); the hard-delete
schedule terminates it at **566 min** (worst case ₹1,351, inside the cap). Expected, nothing
failing: about ₹822 (5.74 h); the planned 6.3 h with its margin, about ₹902. On any step
failure: stop, pull, tear down and report.

## If SSH is lost

The likeliest cause is this Mac's public address changing (it did on 22 Nov 2026). The
instance keeps running and keeps its results; nothing is lost until the hard delete.

1. **Halted, not terminated, then restart (the default).** The driver checks whether the
   security group still admits the current address. If not: a permission-log row first, then
   the group re-pointed to the new /32 (the one change), and the pulls resume. If the
   instance has already stopped itself at the halt, it is started again after the group is
   updated, the remaining step directories are pulled and checked, and it is terminated.
   The hard delete at 566 min is never moved: whatever is not pulled by then is lost, and
   the worst case stays inside the cap.
2. **Instance-side upload (not proposed without a ruling).** The instance writes each finished
   block to an S3 bucket in Sydney through an instance role. It survives a lost SSH and a
   hard delete, but needs a bucket, a role and a write gate that do not exist (new resources,
   a small storage charge).

## The prediction committed before any measurement

`docs/p3_prediction/prediction_P3_exploratory_from_P1_20261122T211249520458Z.json` (sha256 `72b907cf4605336bba91ef3f04f8503945e49dabef207d4e3c4fc7b26183e1b2`, written 22 Nov 2026 21:12 UTC from `5b20daa`, 35 cells) with its sha256 sidecar,
written by `scripts/p3_prediction_from_p1.py` from P1's committed T3.2d prediction
(`prediction_P1_20261109T095833754749Z.json`, sha256 `280a3423…`, checked against its
sidecar before it is read), and committed to this repository before any P3 session. It is
the full transfer from P1: with no P3 component in existence, each P3 cell's prediction is P1's
committed value for that cell (the sum, the step form, the recursion, the sum plus s0).
**EXPLORATORY**: it scores no hypothesis; it tells how far a host change moves the
end-to-end time when nothing about P3 is measured.

## What must be built before the launch yes

- A P3 session config (`configs/t31/p3_session.json`: P1's grid, pairing P3) and a launcher
  run of P3 from the image.
- **A store for the in-session `predict` step.** P1's runner syncs its prediction to GCS
  before the hold; P3 on AWS has no GCS write path. A decision is needed: the Mac's driver
  pulls the prediction and its sidecar and writes the receipt back (needs SSH at that
  moment), or the S3 path of option 2 above.
- The driver: the SSH preflight, the per-block pulls with sha256 checks, the restart path.
- The pre-launch CPU pass (`scripts/prelaunch_cpu_pass.sh`) with a P3 entry, PASSED at the
  commit the yes names.

## Budget

Against the AWS cap of about ₹10,400 (Appendix A, 17 Nov 2026): the caps in the ledger's AWS
list, with the rehearsal at ₹180, come to about ₹9,050 with one month of the snapshot,
leaving about ₹1,350; P6's ₹5,940 is in that sum and is spent only if the rule says run.
Credits left: $159.79 (Free Tier API, 23 Nov 2026), about ₹15,338.
