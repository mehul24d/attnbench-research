# T4 XAttention phase

Status: pre-registered 2026-10-02, before any XAttention row exists. Nothing
below changes once the canary has run. The numbers are data in
`attnbench/accuracy/t4_pilot.py`, and `tests/test_t4_xattention_pilot_plan.py`
holds this document to that module.

## Question

The sparse pilot (`docs/t4_sparse_pilot.md`) found that MInference's
mean-pool estimator, run inline, is non-inferior to dense at no sparsity on
any of its tasks. This phase asks the same question of a second deployable
estimator, XAttention (Xu et al., ICML 2025; arXiv:2503.16428).

Run inline as its authors run it, is block-sparse prefill with XAttention's
own block selection non-inferior to dense, within a fixed margin, on the
same tasks?

## Tasks, tiers and n

These are the sparse pilot's, unchanged: the same three tasks, the same two
bands, the same n per band, and the same tiers. `qa_1` is primary at n=100.
`niah_multivalue`, and `niah_multiquery` at 16384, are secondary at n=50.
`niah_multiquery` at 32768 is observational. Examples are the same seeded
prefix, so every XAttention row pairs with the same example ids the sparse
pilot used.

## The arm

| arm | `score_source` | selection | cost in `latency_ms` |
|---|---|---|---|
| XAttention, inline | `xattention_inline` | antidiagonal scoring per head, blocks kept until their estimated mass reaches tau | included |

- **The official code, not a reimplementation.** The estimator is
  `xattn_estimate` from mit-han-lab/x-attention at `XATTN_COMMIT`.
  `scripts/install_xattention.sh` installs it and refuses to finish unless
  the import resolves to that commit, unedited. The kernel is
  Block-Sparse-Attention, the same kernel the sparse pilot's arms ran.
- **The method's own settings.** These follow its LongBench evaluation at
  the pinned commit: block 128, stride 8, `norm=1`, `select_mode="inverse"`,
  `keep_sink=True`, `keep_recent=True`, the official chunk size, and KV
  heads repeated to the query-head count.
- **One deviation.** That evaluation uses per-layer thresholds profiled for
  Llama-3.1-8B. None exist for Qwen2.5, so this phase uses one scalar
  threshold per run, which `Xattention_prefill` accepts.
- **Sparsity is an outcome, not a setting.** Every row records
  `xattn_threshold` and `realised_density`: the fraction of causally valid
  blocks kept, averaged over the 28 prefill layers. A row without both
  cannot be written.
- **Decode is dense,** as in the sparse pilot. Sparsity applies to prefill
  only.

**Thresholds:** 0.95, 0.9 and 0.8, tested in that order, least aggressive
first.

- 0.9 is `xattn_estimate`'s default at the pinned commit.
- 0.8 is `Xattention_prefill`'s default.
- 0.95 is added first, so the sequence opens at the setting most likely to
  pass if any does.

A lower tau keeps less mass and so fewer blocks. The canary's densities are
not used to change the thresholds.

**The dense reference is re-run.** Its code path is unchanged, but the
XAttention rows are written at a new commit, and the analysis compares
within one commit. So the dense reference runs again in the same session,
and the canary checks that it reproduces the sparse pilot's dense
predictions.

## Outcome and test

- **Outcome:** `correct`, as in both earlier pilots.
- **Test:** the exact paired bound in
  `attnbench/analysis/exact_noninferiority.py`, unchanged.
- **Margin of 10 points; one-sided alpha 0.025.**
- **Fixed sequence.** Within each (task, band), test tau 0.95, then 0.9,
  then 0.8. Stop at the first threshold that is not non-inferior.
  Thresholds after the stop are reported but carry no claim.
- **No pooling** across tasks, bands or arms. The sparse pilot's arms ran at
  another commit, so they are never in the same analysis.

Analysis, over the dense run and the three threshold runs:

```bash
python scripts/run_t4_noninferiority.py \
    results/t4_xattn_dense_<band>_<date>/accuracy.parquet \
    results/t4_xattn_tau0.95_<band>_<date>/accuracy.parquet \
    results/t4_xattn_tau0.9_<band>_<date>/accuracy.parquet \
    results/t4_xattn_tau0.8_<band>_<date>/accuracy.parquet \
    --out results/t4_xattn_analysis_<date>
```

## What may be claimed

These rules are the sparse pilot's, with one qualifier added.

- A claim needs `claim=True`. It reads: "at threshold tau, XAttention is
  non-inferior to dense within 10 points on `<task>` at `<band>` (exact
  one-sided 97.5% bound *L*; mean realised density *d*)". It carries the
  qualifier "with a scalar threshold, not per-layer thresholds profiled for
  this model".
- **"No accuracy loss" is never claimed.**
- **Failing to certify is not evidence of loss.**
- The observational cell yields no statement, and the secondary tier yields
  no headline.
- **Comparison with the earlier arms is descriptive only.** Placing an
  XAttention cell beside the oracle or the mean-pool arm at the nearest
  fixed sparsity crosses commits and densities. It is reported as a
  description, never as a test.

## Estimator cost

`scripts/measure_estimator_cost.py` times `xattn_estimate` at each
threshold, beside the components it already times. On an L4 the official
code's own device test turns Triton off, because the device name lacks
"100". The estimator then runs its torch path, and every timing row records
`xattn_triton`.

So an L4 cost is the torch path's cost, not the method's official one. No
cost claim is made from it without naming both the card and the path.

## Canary, then the pilot

**Canary.** 2 examples per task and band; the dense reference; each
threshold; the CUDA gate; the estimator-cost script. It must pass all of the
following before the pilot is proposed:

1. `scripts/install_xattention.sh` succeeds.
2. The CUDA gate passes. It includes:
   - `tests/test_xattention_backend.py`: the bitwise equality with
     `Xattention_prefill` at three thresholds and two lengths, and the
     pinned-checkout test;
   - `tests/test_xattention_rows.py`;
   - the sparse pilot's gate files.
3. Every row has `git_dirty=False` and one commit. Every XAttention row has
   `backend_role="block_sparse"`, its threshold, and a realised density in
   (0, 1].
4. Every threshold completes at 32768 without OOM.
5. The dense canary predictions equal the sparse pilot's dense predictions
   for the same example ids. Those are under
   `results/t4_sparse_pilot_session_*_20261002/`.
6. The estimator-cost run reports `xattn_estimate` available at every
   threshold.

The canary's rows go to their own directories and are never pooled with the
pilot's.

**Pilot.** The pilot is the canary's four accuracy commands, without
`--n-per-length` and with `--seq-lens <band>`, one band per session if the
canary's timing calls for that. Each threshold writes to its own `--out`:
the resume key carries no threshold, and `run_accuracy` refuses a second
threshold resuming into the first's rows.

The dense run is 400 cells. Each threshold run is 400 cells.

```bash
bash scripts/install_xattention.sh
python scripts/run_accuracy.py --t4-xattn-pilot --only-backends sdpa_flash \
    --n-per-length 2 --out results/t4_xattn_canary_dense_<date>
for TAU in 0.95 0.9 0.8; do
  python scripts/run_accuracy.py --t4-xattn-pilot --xattn-threshold $TAU \
      --only-backends xattention --n-per-length 2 \
      --out results/t4_xattn_canary_tau${TAU}_<date>
done
python scripts/measure_estimator_cost.py --out results/estimator_cost_xattn_<date>
```

**Cost, before the canary measures it.** The dense rows took 3.35 s and
5.70 s at 16384 and 32768 in the sparse canary. Block-sparse rows took 2.2
to 4.9 s, before XAttention's estimator, whose torch-path cost on an L4 is
unmeasured. If an XAttention row costs about what a dense row does, the
pilot is about 1600 rows and 2–3 L4-hours. The canary is about 40 minutes.
The cap, hard-delete limit and artifact paths are stated with the command
before each session launches.

## Canary result (2026-10-02): passed

The canary ran on an NVIDIA L4 in `asia-northeast1-a` (instance
`attnbench-l4-xattn-canary-20261002-2016`) at commit `cc769b3`. That commit
is `2c66de1` plus an edit to `docs/spend_ledger.md` and nothing else. The
environment was torch 2.9.1+cu129 and transformers 4.46.0, with x-attention
installed at `e379887`. Every phase was synced to GCS. The rows are under
`results/t4_xattn_canary_session_20261002/` and are never pooled with the
pilot.

| # | criterion | result |
|---|---|---|
| 1 | install | `x-attention e379887 installed, clean, importable` |
| 2 | CUDA gate | 50 passed, 0 skipped, including all six `test_bitwise_equal_to_xattention_prefill_on_cuda` cases and `test_the_installed_xattention_is_the_pinned_clean_checkout` |
| 3 | rows | 12 dense and 12 per threshold. All have `git_dirty=False` and one commit. Every XAttention row is `block_sparse`, with its threshold (12 per value, none missing) and a realised density in (0, 1] |
| 4 | 32768 at every threshold | completed, no OOM, every phase `rc=0` |
| 5 | dense vs the sparse pilot's dense | 12 of 12 predictions identical |
| 6 | estimator cost | `xattn_estimate` available at every threshold, `xattn_triton=False` on every row |

**Realised density.** This is the fraction of causally valid blocks kept,
averaged over layers, ranging across the canary's examples:

| band | tau 0.95 | tau 0.9 | tau 0.8 |
|---:|---|---|---|
| 16384 | 0.37–0.40 | 0.27–0.29 | 0.18 |
| 32768 | 0.10–0.34 | 0.08–0.23 | 0.06–0.14 |

Even the least aggressive threshold keeps fewer blocks than the sparse
pilot's 0.5 setting. As the plan requires, the thresholds are not changed.

**Estimator cost, per layer, on the L4's torch path.** Medians of 30, at the
model's geometry:

| band | dense | `xattn_estimate` (any tau) | block-sparse 0.5 / 0.75 / 0.9 |
|---:|---:|---:|---:|
| 16384 | 14.54 ms | 19.4–19.6 ms | 8.66 / 5.18 / 2.94 ms |
| 32768 | 58.41 ms | 60.6–60.7 ms | 31.83 / 17.63 / 8.79 ms |

On this card, XAttention's estimator alone costs more than the dense
attention it is meant to prune, at both bands and every threshold. Its
estimator-to-saving ratio therefore exceeds 1 at any density, and the arm
cannot be faster than dense on an L4.

This is the torch fallback's cost. The official code takes that path on any
device whose name lacks "100". It is not the method's official cost, and it
enters no claim without that qualifier. The accuracy question the pilot
pre-registers is unaffected, because the fallback selects the same blocks.

**Timing for the pilot.** Each 12-row phase took about 100 s, including the
model load. At about 5 s per row, the pilot's 1600 rows come to roughly 3
hours in one session covering both bands.
