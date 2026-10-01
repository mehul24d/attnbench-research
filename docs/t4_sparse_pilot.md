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
