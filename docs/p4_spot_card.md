# P4 on Flex-start and Spot in parallel at `lock-2026-11-14-r3` (15 Nov 2026)

**Not launched.** The researcher's decision of 15 Nov 2026 (plan Section 5.2, Budget row; Appendix
A, 15 Nov 2026): "Run P4 on Flex-start and Spot in parallel, full scope, no reduced cell set; the
first RUNNING grant wins, cancel every other request at once, delete any second grant unused and
log it, run P4 once." Spot starts only after every check below passes and only after the
researcher's yes on this card. The step list and minutes are in `docs/launch_cards/t64_p4_h100.md`;
the Spot driver is the scratchpad's `p4_run/spot/spot_blocks.sh`.

## What is asked

| | |
|---|---|
| Machine | one `a3-highgpu-1g` (H100 80GB) at a time; `GPUS_ALL_REGIONS` is 1 |
| Code | a fresh clone at the tag `lock-2026-11-14-r3`, deployed by `gcp_deploy_source.sh` |
| Image | `attnbench-env-v6-20260917`, P2's |
| Flex-start | the running request loop moves from `-r2` to `-r3` (one request at a time, zones in rotation `us-central1-a`, `asia-southeast1-b`, `us-central1-a`, `asia-southeast1-c`, 6,900 s each, the RUNNING check, until a grant or the end of 25 Nov 2026 IST); a grant runs P4 as one session, as the 14 Nov 2026 launch yes says: cap ₹3,245, in-guest halt 419 min, hard delete 429 min |
| Spot | the four blocks of the lock's spec (`blocks.P4`), one VM per block; b1 in any of `us-central1-a`, `asia-southeast1-b`, `asia-southeast1-c`, later blocks in b1's zone (the block identity compares zones); an attempt every 10 min while none is granted, until the end of 25 Nov 2026 IST |
| First grant | the first RUNNING VM runs P4: on a Spot grant for b1 the Flex-start loop is stopped and its waiting request deleted; a Flex-start grant first ends the Spot driver before b1; a second grant is deleted unused and logged |
| Bootstrap | `gcp_bootstrap_instance.sh` without `--suite-record` (the 14 Nov 2026 launch yes); the FA3 wheel `1676ef0b…` on every boot |
| Staged by scp | P1's training file (`de7f0f44…`), P2's (`96c6670c…`), P2's Triton masks (40 files, read on 14 Nov 2026); each Spot block also fetches every earlier block's session from the bucket |
| Interruption | a preempted block (Compute Engine's `preempted` operation, or a discard on a restarted boot) is discarded in full and rerun on a new VM; any other failure stops the driver and nothing reruns without a look |
| Money | before every Spot attempt: the ledger's ₹7,272 + every P4 attempt so far + the caps of the blocks left, at most ₹15,500, else the driver stops and the researcher is asked |

## The Spot blocks (at ₹629.94/h, `us-central1`, the higher rate)

| Block | Steps | Planned h | Expected ₹ | Cap ₹ | Halt min | Hard delete min |
|---|---|---|---|---|---|---|
| b1 | boot (full preflight), `link`, `launch_queue`, `probe`, `masks` | 2.00 | 1,260 | 1,638 | 156 | 166 |
| b2 | boot (short preflight), `host_components` | 2.23 | 1,407 | 1,829 | 174 | 184 |
| b3 | boot (short), `device_components`, `dense_select`, `host_speed_mid`, `predict`, ground truth at 2048 and 4096 with the anchor, `late_check`, `trace_checks` | 2.30 | 1,449 | 1,884 | 179 | 189 |
| b4 | boot (short), ground truth on the other 20 cells with the anchor, `late_check`, `trace_checks` | 1.77 | 1,114 | 1,447 | 137 | 147 |
| | | **8.30** | **5,229** | **6,798** | | |

In `asia-southeast1` (₹603.61/h) the same blocks are ₹5,010 expected. b1's 20-minute suite
record stays in its cap, as in the Flex-start cap, though it does not run.

## Headroom from ₹8,228 (the ceiling ₹15,500 less ₹7,272 spent)

| | ₹ |
|---|---|
| Spot, the caps summed (reserved: one of Flex-start and Spot runs, and Spot is the dearer) | 6,798 |
| Headroom at the caps | **1,430** |
| One interruption at the largest block cap (b3, ₹1,884) | −454 at the caps; the driver stops and asks before that rerun |
| One interruption, at the expected costs (₹5,229 + b3's expected ₹1,449) | 1,550 left |
| Flex-start instead (cap ₹3,245) | 4,983 |

Not counted: image storage, egress, the ₹1,090 of the billing figure not yet attributed.

## Preflight, cheapest first, and the minutes to each failure point

From RUNNING, on b1 (later blocks run the short preflight: provenance, imports, the libraries
and the kernel contract, about 4 min). Estimates; nothing has booted v6 on an `a3-highgpu-1g`.

| Point | Minutes from RUNNING | ₹ at 629.94/h |
|---|---|---|
| Setup: deploy, instance preflight, GCS write gate, bootstrap (v6's first boot on an H100 fails here) | up to 36 | 378 |
| Session start: identity (exit wrapper, clean tree), the timed-path pin `0e0fb012…`, the banking times | 36 | 378 |
| `prereqs`: plan rows, deployed files, staged inputs and their sha256s | 37 | 389 |
| `fa3_import`, first: the libraries against P2's (FA3 alone may differ) | 38 | 399 |
| `fa3_import`: the wheel's hash, tags, install and import | 40 | 420 |
| `preflight`: provenance, imports, libraries again, lightgbm, the GPU variants | 42 | 441 |
| `preflight`: the block-sparse kernel's contract | 43 | 452 |
| `preflight`: the Triton smoke check, every arm at 2048 | 52 | 546 |
| `preflight`: the XAttention arms at 32768 | 55 | 578 |
| `host_components` | b2 | |

On any failure the exit wrapper syncs and halts the VM within minutes, and the driver tears it
down and deletes it. The GPU variants stay before the kernel contract (the order the prompt names
lists neither; moving them is a timed-path change and a re-cut).

## Before the yes (all must pass)

| Check | How | Status |
|---|---|---|
| Money: every billable resource listed | read-only listing, 15 Nov 2026 | done: one waiting Flex-start request, images v5 and v6, the results bucket; nothing else |
| Spot rate re-read, cap and headroom | Cloud Billing catalog, 2026-11-15T14:24:56Z | done (above) |
| Block plan and re-cut | short preflight on later blocks; `lock-2026-11-14-r3`; 17 + 3 mutations killed | done |
| Spot H100 quota in each zone, global GPU quota | Cloud Quotas, 15 Nov 2026 | 1 in `us-central1`, 1 in `asia-southeast1`; `GPUS_ALL_REGIONS` 1, usage 0 |
| Image v6 usable; bucket read and write from the zone's service account; staged inputs in the bucket with their sha256 | rehearsal 1, and the combined read/list | done, 16 Nov 2026: v6 booted and bootstrapped, the bucket written and read back; the staged inputs' sha256s match the ledger |
| CPU pass at the tag with the interruption path | `scripts/prelaunch_cpu_pass.sh p4` with the lock's spec; the driver end to end with gcloud mocked | done at `7a24cba` (one session; four blocks, b4 interrupted and rerun, drift passed); the driver's mock passed |
| Cheap-VM rehearsal 1 (e2 on-demand, cap ₹50, deleted at 30 min) | `p4_run/reh/reh1.sh` | passed, 14 min, about ₹3 |
| Cheap-VM rehearsal 2 (e2 Spot, cap ₹20, deleted at 30 min) | `p4_run/reh/reh2.sh` | passed on the fourth VM, about ₹5 for four: held mid-step, stopped, restarted, discarded, rerun, classified from the bucket |

## What it does not do

- It reads no P2 prediction error and scores nothing.
- It drops no cell: the 35 grid cells are measured, each once, the anchor in b3 and b4.
