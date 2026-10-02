# T4 sparse pilot

Status: pre-registered 2026-10-01, before any sparse row on these tasks
exists. Nothing below changes once the canary has run. The numbers are data
in `attnbench/accuracy/t4_pilot.py`, and `tests/test_t4_sparse_pilot_plan.py`
holds this document to that module.

## Question

On the three tasks the dense pilot selected (`docs/t4_dense_pilot.md`), is
block-sparse prefill at the grid's sparsities non-inferior to dense, within a
fixed margin, when the mask comes from (a) the dense-softmax oracle and
(b) the deployable estimator run inline? The oracle shows what the mask
*could* preserve. The inline estimator shows what a deployed method *does*
preserve.

## Tasks, tiers and n

The 40–90 rule selected these tasks on the n=5 probe. It is not re-applied
at n=50: the probe was a coarse gate against broken tasks, and re-selecting
on the pilot would condition the sparse study on the dense scores it is
compared against.

| task | band | dense pilot (n=50) | tier | n per band |
|---|---:|---:|---|---:|
| `qa_1` | 16384 | 30/50 = 60.0% | primary | 100 |
| `qa_1` | 32768 | 20/50 = 40.0% | primary | 100 |
| `niah_multivalue` | 16384 | 16/50 = 32.0% | secondary | 50 |
| `niah_multivalue` | 32768 | 17/50 = 34.0% | secondary | 50 |
| `niah_multiquery` | 16384 | 13/50 = 26.0% | secondary | 50 |
| `niah_multiquery` | 32768 | 3/50 = 6.0% | observational | 50 |

- **Primary.** `qa_1` sits in the range where both gains and losses are
  observable. Its non-inferiority results are the pilot's claims.
- **Secondary.** `niah_multivalue`, and `niah_multiquery` at 16384, are
  below 40% but well clear of 0. They can expose a catastrophic attention
  failure, and are reported with the same test but never headlined.
  (`niah_multiquery` at 16384 was not named in the tier discussion; it is
  placed with `niah_multivalue` because its dense rate, 26%, is in the same
  regime.)
- **Observational.** `niah_multiquery` at 32768 has 3 of 50 dense
  successes, which is floor-limited. The arm can lose at most 3 examples
  there, so no non-inferiority statement can be made in either direction.
  Counts are reported. No bound is interpreted.

**Why `qa_1` runs at n=100.** At n=50 the exact bound below cannot clear a
10-point margin once dense wins even one discordant pair (the bound is then
−12.1 points). At n=100 the best attainable bound is −4.3, and three
discordant pairs each way give −8.9. 100 is the grid's ceiling at 32768.
Examples are a seeded prefix, so `qa_1`'s first 50 are the dense pilot's 50.

## Arms

Every arm runs at the same commit, in the same session as its dense
reference, and is identified on its rows by `score_source`:

| arm | `score_source` | ranking | cost in `latency_ms` |
|---|---|---|---|
| dense reference | — | none | — |
| oracle | `dense_softmax_fp32` | dense fp32 softmax pass, block-pooled | excluded |
| deployable, inline | `minference_meanpool_inline` | MInference mean-pool, each layer on its own q and k inside the forward; exact device mask builder | included |

Both sparse arms use `block_sparse` (Block-Sparse-Attention), block 128, the
sink-rule importance mask, and sparsities 0.5, 0.75 and 0.9. The dense
reference is re-run in the session, not borrowed from the dense pilot, so
every comparison is within one commit.

Not in this pilot:

- **The two-pass cheap arm** (`minference_meanpool`). It ranks from a dense
  pass, which a deployed method never has.
- **XAttention, to come in a later phase.** Its backend exists
  (`attnbench/backends/xattention.py`), but four things are missing first:
  1. `x-attention` is not installed on the image at `XATTN_COMMIT`, and
     nothing checks it at boot.
  2. Its CUDA bitwise-equality test against `Xattention_prefill` has never
     run.
  3. Its threshold and per-row realised density have no place in the row
     schema.
  4. On an L4 its official code falls back from Triton to torch, so its cost
     there is not its official cost.

  Its thresholds will be pre-registered before its first row.

## Outcome and test

- **Outcome:** the row's `correct` (score = 100), the same binary outcome as
  the dense pilot.
- **Test:** exact paired bound, `attnbench/analysis/exact_noninferiority.py`.
  Let *b* be the sparse-only correct pairs and *c* the dense-only correct
  pairs, out of *n*. The bound on p_sparse − p_dense is
  `CP_lower(b; alpha/2) − CP_upper(c; alpha/2)`, which has coverage of at
  least 1 − alpha by the union bound. It has no asymptotics or resampling,
  and it is valid at the floor and the ceiling.
- **Margin of 10 points; one-sided alpha 0.025.** Non-inferior if the bound
  is above −10 points.
- **Fixed sequence.** Within each (arm, task, band), test 0.5, then 0.75,
  then 0.9. Stop at the first sparsity that is not non-inferior. Sparsities
  after the stop are reported but carry no claim. This controls the
  family-wise error within the family at alpha with no adjustment.
- **No pooling** across tasks, bands or arms, and no post-hoc choice of
  margin, alpha, sparsity or tier.

Attainable bounds, for reading results:

| n | b, c | lower bound (points) |
|---:|---|---:|
| 50 | 0, 0 | −8.4 |
| 50 | 0, 1 | −12.1 |
| 100 | 0, 0 | −4.3 |
| 100 | 0, 1 | −6.2 |
| 100 | 3, 3 | −8.9 |

Analysis:

```bash
python scripts/run_t4_noninferiority.py \
    results/t4_sparse_pilot_oracle_<date>/accuracy.parquet \
    results/t4_sparse_pilot_inline_<date>/accuracy.parquet \
    --out results/t4_sparse_pilot_analysis_<date>
```

The script refuses dirty rows, rows from more than one commit, and tasks or
arms outside this plan.

## What may be claimed

- A claim needs a cell with `claim=True`: tested under the fixed sequence and
  non-inferior. It reads: "at sparsity *s*, `<arm>` is non-inferior to dense
  within 10 points on `<task>` at `<band>` (exact one-sided 97.5% bound
  *L*)". The oracle's claims also carry the qualifier "given a ranking
  computed from the full attention scores" (`analysis/matched.py`).
- **"No accuracy loss" is never claimed.** The difference and its bound are
  reported instead.
- **Failing to certify is not evidence of loss.** The point difference and
  the bound are reported as they are.
- **The observational cell yields no statement**, and the secondary tier
  yields no headline.

## Estimator cost (C1, C2)

`scripts/measure_estimator_cost.py` runs in the same session at the
model's geometry. Per layer, it times:

- the mean-pool estimator;
- the warm device mask builder at each sparsity;
- `sdpa_flash`;
- Block-Sparse-Attention at each sparsity;
- XAttention's estimator, if installed (otherwise recorded as unavailable).

It records whether the warm estimator and builder synchronise the device.
Its stated falsifiers are in its docstring. This gives the estimator's cost
against the saving it buys, kernel for kernel. It is not an end-to-end
speedup.

## Latency at matched accuracy

This is not decided by this pilot. `latency_ms` on accuracy rows is one
end-to-end generate per example, with no warm-up or repeats, and it includes
the inline estimator and excludes the oracle's scoring pass. It is reported
descriptively only. The comparison at matched accuracy is a later, separate
timing run of the inline arm at the sparsities this pilot certifies, on the
existing end-to-end harness.

## Canary, then the pilot

**Canary.** The canary uses 2 examples per task and band, all arms, plus
the CUDA test gate and the estimator-cost script. It must pass all of the
following before the full pilot is proposed:

1. The CUDA tests pass on the instance, including
   `tests/test_device_mask_builder.py` (with the warm no-sync test),
   `tests/test_inline_estimator.py` and `tests/test_inline_arm_wiring.py`.
2. Every row has `git_dirty=False` and one commit, and the inline and
   oracle rows carry their own `score_source`.
3. The oracle scoring pass completes at 32768 on the L4 without OOM.
4. The estimator-cost run reports `sync_free=True` at every sparsity, and
   every non-XAttention component is available.
5. The dense canary predictions equal the banked dense pilot's predictions
   for the same example ids (checked locally against
   `results/t4_selected_dense_pilot_20261001/`).

The canary's rows go to their own directories and are never pooled with the
pilot's. Their per-row times give the pilot's projected cost.

```bash
python scripts/run_accuracy.py --t4-sparse-pilot --n-per-length 2 \
    --out results/t4_sparse_canary_oracle_<date>
python scripts/run_accuracy.py --t4-sparse-pilot --n-per-length 2 \
    --score-source minference_meanpool_inline --only-backends block_sparse \
    --out results/t4_sparse_canary_inline_<date>
python scripts/measure_estimator_cost.py --out results/estimator_cost_l4_<date>
```

**Pilot.** The same two `run_accuracy.py` commands without
`--n-per-length`. The oracle run is 1600 cells: 400 dense and 1200 sparse.
The inline run is 1200 cells. If the canary's projection exceeds one
session's cap, the pilot is split by band with `--seq-lens`, as Stage 3 was.
Each run writes to its own `--out`, and `run_accuracy` refuses to resume one
arm into another's rows.

**Cost, before the canary measures it.** These are estimates, not
measurements:

- The oracle scoring pass is published at 11.8 s per example at 16384
  (`docs/claims.md`). At 32768 it is assumed to be about 4× that, which is
  unmeasured.
- Sparse and dense rows are assumed to take 5–8 s each. That rate is
  inferred from the dense pilot session's 300 rows in 43 minutes.

On those assumptions the pilot is about 2,800 rows plus 400 scoring passes,
or 7–10 L4-hours. At the ledger's L4 rate that is roughly INR 600–800, in
one or two sessions. The canary is about 50 minutes, roughly INR 70. The
cap, hard-delete limit and artifact paths for each session are stated with
the command before it launches.

## Canary result (2026-10-01): passed

The canary ran on an NVIDIA L4 in `asia-northeast1-a` (instance
`attnbench-l4-t4canary-20261002-0214`) at commit `e7eabaa`, with torch
2.9.1+cu129 and transformers 4.46.0. Every phase was synced to GCS, and the
instance was then torn down through `scripts/gcp_teardown_session.sh`. The
rows are under `results/t4_sparse_canary_session_20261002/` locally and under
`gs://attnbench-results-research-507316/attnbench-l4-t4canary-20261002-0214/`.
They are never pooled with the pilot.

| # | criterion | result |
|---|---|---|
| 1 | CUDA tests | 27 passed, 0 skipped, including `test_bitwise_equal_on_cuda` and `test_warm_device_build_does_not_synchronise` |
| 2 | provenance and arm labels | 48 + 36 rows, `git_dirty=False`, one commit; 12 dense rows with no scorer, 36 `dense_softmax_fp32`, 36 `minference_meanpool_inline` |
| 3 | oracle scoring at 32768 | completed, no OOM |
| 4 | estimator cost run | `sync_free=True` at both bands and all three sparsities; every non-XAttention component available (XAttention recorded as not installed) |
| 5 | dense vs banked dense pilot | 12 of 12 predictions identical |

**Estimator cost, per layer** (median of 30, CUDA events, random inputs, the
model's geometry):

| band | dense | meanpool | mask build | block-sparse 0.5 / 0.75 / 0.9 | estimator / saving |
|---:|---:|---:|---:|---:|---:|
| 16384 | 14.39 ms | 1.28 ms | 0.50 ms | 8.67 / 5.19 / 2.96 ms | 0.31 / 0.19 / 0.16 |
| 32768 | 58.13 ms | 2.55 ms | 0.50 ms | 31.94 / 17.66 / 8.98 ms | 0.12 / 0.08 / 0.06 |

At both bands, kernel against kernel, the deployable estimator costs less
than the saving it buys at every sparsity. None of the script's stated
falsifiers fired. These are per-layer kernel numbers on one session's canary,
not an end-to-end speedup, and nothing here enters `docs/claims.md` until the
pilot's own estimator-cost run is banked.

**Timings used for the pilot's cost.** These are the median `latency_ms` per
row at 16384 / 32768:

- dense: 3.35 / 5.70 s;
- oracle block-sparse: 2.90 / 4.91 s;
- inline block-sparse: 2.21 / 4.90 s.

The oracle phase took 581 s for 48 rows and the inline phase 172 s for 36,
so the 12 scoring passes cost 342 s, an average of 28.5 s. Split
quadratically, that is about 11 s at 16384, which matches the published
11.8 s, and about 46 s at 32768. The 32768 figure is now measured on average
rather than assumed.

**Projected pilot:**

- 16384: about 1.7 compute-hours.
- 32768: about 4.5 compute-hours, of which the oracle's scoring passes are
  about 2.5.

The pilot is split by band into two sessions, as the canary's projection
allows. Both sessions deploy the same commit, so the analysis's single-commit
rule holds across them.

## Pilot result (2026-10-02)

Both sessions ran on an NVIDIA L4 at commit `77b48e5`, with torch
2.9.1+cu129 and transformers 4.46.0, on Qwen2.5-1.5B-Instruct. The CUDA gate
passed 27 of 27 in each session before any row was written.

| band | instance | zone | oracle rows | inline rows |
|---:|---|---|---:|---:|
| 16384 | `attnbench-l4-t4pilot-16384-20261002-0312` | `asia-northeast1-a` | 800 | 600 |
| 32768 | `attnbench-l4-t4pilot-32768-20261002-1305` | `asia-northeast1-c` | 800 | 600 |

Every row has `git_dirty=False` and the one commit. The per-task counts are
the plan's: 100/50/50 dense rows per band, and 300/150/150 for each scorer.
The rows are under `results/t4_sparse_pilot_session_16384_20261002/` and
`results/t4_sparse_pilot_session_32768_20261002/` locally, and under each
instance's prefix in `gs://attnbench-results-research-507316/`. The 32768
session's copy also holds `results/s7_7b_16384/accuracy.parquet` at commit
`35ac080`. That is a force-committed file that the quarantine's
`git checkout -- results` restores. It is not pilot data, and the analysis
would refuse it on commit and task.

The analysis is the pre-registered command over the four parquets, written
to `results/t4_sparse_pilot_analysis_20261002/`. The margin is 10 points,
one-sided alpha 0.025, under the fixed sequence. Each cell below is
`correct of n: difference, lower bound`, in points. **Bold** marks a claim.
*Italics* mark a sparsity after the sequence stopped, which carries no claim
whatever its bound.

**Oracle, `dense_softmax_fp32`:**

| tier | task | band | dense | 0.5 | 0.75 | 0.9 |
|---|---|---:|---:|---|---|---|
| primary | `qa_1` | 16384 | 68/100 | **70: +2.0, −9.0** | 66: −2.0, −16.3 | *60: −8.0, −24.8* |
| primary | `qa_1` | 32768 | 46/100 | 44: −2.0, −12.7 | *53: +7.0, −8.3* | *48: +2.0, −16.1* |
| secondary | `niah_multivalue` | 16384 | 16/50 | **23: +14.0, −7.7** | 21: +10.0, −13.3 | *17: +2.0, −21.1* |
| secondary | `niah_multivalue` | 32768 | 17/50 | 17: 0.0, −19.2 | *20: +6.0, −15.9* | *19: +4.0, −19.8* |
| secondary | `niah_multiquery` | 16384 | 13/50 | 16: +6.0, −10.3 | *21: +16.0, −6.3* | *15: +4.0, −19.8* |
| observational | `niah_multiquery` | 32768 | 3/50 | 8 | 27 | 18 |

**Deployable estimator, inline, `minference_meanpool_inline`:**

| tier | task | band | dense | 0.5 | 0.75 | 0.9 |
|---|---|---:|---:|---|---|---|
| primary | `qa_1` | 16384 | 68/100 | 44: −24.0, −39.5 | *35: −33.0, −48.8* | *24: −44.0, −59.6* |
| primary | `qa_1` | 32768 | 46/100 | 39: −7.0, −22.2 | *24: −22.0, −37.4* | *16: −30.0, −44.7* |
| secondary | `niah_multivalue` | 16384 | 16/50 | 5: −22.0, −42.0 | *3: −26.0, −46.3* | *0: −32.0, −48.7* |
| secondary | `niah_multivalue` | 32768 | 17/50 | 8: −18.0, −39.2 | *1: −32.0, −48.7* | *0: −34.0, −50.8* |
| secondary | `niah_multiquery` | 16384 | 13/50 | 6: −14.0, −34.7 | *4: −18.0, −35.7* | *2: −22.0, −37.9* |
| observational | `niah_multiquery` | 32768 | 3/50 | 6 | 10 | 3 |

The dense reference is the oracle run's own. The inline run is compared
against it, pair by pair, on the same example ids.

**What the pilot licenses**, in the plan's wording:

- **Primary.** At sparsity 0.5, the oracle arm is non-inferior to dense
  within 10 points on `qa_1` at 16384 (exact one-sided 97.5% bound −9.0
  points; 70 vs 68 of 100), given a ranking computed from the full attention
  scores. This is the pilot's only primary claim.
- **Secondary, not a headline.** At sparsity 0.5, the oracle arm is
  non-inferior on `niah_multivalue` at 16384 (bound −7.7).
- **The deployable estimator certifies nothing.** No cell of the inline arm
  is non-inferior, at any tier, band or sparsity. Its first test, at 0.5,
  fails in every primary and secondary cell, with bounds from −22.2 to −42.0
  points.

**What it does not license:**

- "No accuracy loss", for any arm. It is never claimed.
- Non-inferiority at 32768 for either arm. Neither arm certifies there; the
  oracle on `qa_1` stops at 0.5 with a bound of −12.7.
- Anything at an italic sparsity. The oracle's bounds at `qa_1`/32768/0.75
  (−8.3) and `niah_multiquery`/16384/0.75 (−6.3) clear the margin, but the
  sequence stopped at 0.5 in both. Reordering the sequence after seeing them
  is exactly what pre-registration rules out.
- Anything from the observational cell. Dense is 3 of 50 there. The oracle's
  27 of 50 at 0.75 is reported as a count; it is not interpreted.

**Read descriptively, not as a test.** The comparisons below were not
pre-registered and carry no error control:

- The inline estimator's failures are not marginal. At 0.5, the discordant
  pairs run toward dense in every primary and secondary cell: 30 dense-only
  vs 6 sparse-only on `qa_1` at 16384, 16 vs 9 at 32768, and 13 vs 2, 12 vs 3
  and 10 vs 3 on the secondary cells.
- The inline arm is below the oracle at 0.5 in every primary and secondary
  cell, by 26, 5, 36, 18 and 20 points.
- The direction matches the 2026-09-16 two-pass comparison on
  `niah_multikey` (`claims.md`, "The oracle requirement DOES survive a change
  of model scale").
- `tests/test_inline_estimator.py` pins the inline arm's layer-0 mask to the
  two-pass estimator's exactly. Its later layers differ by construction,
  because they see sparse inputs.

**Estimator cost.** The pilot session's own run,
`estimator_cost_l4_pilot_20261001`, reproduced the canary's per-layer table
above to within about 2% at both bands; for example, mean-pool was 1.284 ms
at 16384 in both. `sync_free=True` held at every sparsity. Kernel for kernel,
the inline estimator plus its warm mask build costs 0.06–0.31 of the
attention time it saves. So the cost half of the deployability bar is met at
the kernel level. The accuracy half is not met at any sparsity.

**Operations, for the next session:**

- The 32768 teardown was first invoked with the launch zone `-a` while the
  instance was in `-c`. Every step failed and the delete reported "not
  found". Nothing was lost, and the re-run with `-c` deleted the instance.
  `scripts/gcp_teardown_session.sh` now refuses a wrong zone before any step
  and names the right one.
- It also no longer truncates a saved serial log when a re-run's fetch
  fails. Both are covered by `tests/test_teardown_guards.py`.
- Session costs are recorded in each session's `session_cost.txt` and belong
  in `docs/spend_ledger.md`.
