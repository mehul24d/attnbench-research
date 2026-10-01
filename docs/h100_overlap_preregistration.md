# H100 overlap test — pre-registration

Written and committed **before** the H100 session runs (2026-10-01). Nothing
below may be edited after the data exists; corrections go in a dated section
at the end.

## What is being tested

`limitations.md`, "Measured 2026-10-01", explains the L4/A100 sign difference
as host–device overlap: the CPU builds layer L+1's mask while queued GPU work
runs, and the forward pays only what outlasts it. That section says the
account "predicts the reversal is worse on an H100".

**That prediction assumed the same host CPU, and the H100 host is not the
same.** a3-highgpu-1g carries a Xeon Platinum 8481C (Sapphire Rapids), a
newer generation than the 8273CL (Cascade Lake) on both earlier hosts. The
H100 speeds up both sides of the balance, so the card alone does not decide
the direction. The mechanism decides it through one quantity that can be
measured on any host:

    r = (standalone build, ms per layer) / (vectorised sparse prefill, ms per layer)

## The model, fitted on the existing 18 cells only

Per layer the forward pays the part of the build that outlasts a fraction α of
the layer's GPU work, so the exposed share of the standalone cost is

    share = max(0, 1 − α / r)

Fitted from `results/s12_*` (A100 and L4, nine cells each, derived in
`tests/test_overlap_paired.py`): α ≈ 0.5–0.6 on the A100 and ≈ 0.6–0.8 on the
L4. The pre-registered band is the union, widened: **α ∈ [0.45, 0.85]**.

## Predictions

The run measures, per (band, sparsity) at 8192 / 16384 / 32768 × 0.50 / 0.75
/ 0.90: standalone builder cost on the H100 host, vectorised and reference
end-to-end prefill interleaved per rep (`scripts/run_vectorised_endtoend.py`
at `746abd1` or later), and the dense control.

- **P1 — the mechanism transfers.** For each cell, the observed exposed share
  lies in `[max(0, 1 − 0.85/r_H), max(0, 1 − 0.45/r_H)]` widened by ±0.10, with
  `r_H` the H100's own measured ratio. **Pass:** at least 7 of 9 cells inside.
  **Fail:** 3 or more outside. A fail means one α band does not describe three
  cards and the overlap account is at best card-specific.
- **P2 — the direction is set by r, not by the card.** In every cell where
  `r_H > r_A100`, the H100's exposed share exceeds the A100's; where
  `r_H < r_A100`, it does not. The old sentence "worse on an H100" is a
  special case that holds only where the first condition does.
- **P3 — standalone build is faster on Sapphire Rapids.** The standalone cost
  per cell is below the A100 host's in all nine cells. Not a test of the
  mechanism; it records what changed on the CPU side so P2 can be read.

## What is reported regardless

All nine cells of the paired table for the H100, the reference-builder and
vectorised-builder speedups over flash, `lscpu` and live `cpuPlatform`, and
the clock and steal trace. A failed prediction is reported as failed, in
`limitations.md`, beside the section it tests.

## Run parameters, fixed in advance

a3-highgpu-1g, us-central1, DWS Flex Start, image `attnbench-env-v5-20260905`
(the image earlier H100 block-sparse rows ran on), clocks locked, warmup 3,
reps 10, in-guest halt 40 min, delete at 1 h. Expected cost under Rs 150;
ceiling Rs 425.

## The A100 reference values P2 compares against

| band | sparsity | r (A100) | exposed share (A100) |
|---|---|---|---|
| 8192 | 0.50 | 1.30 | 0.62 |
| 8192 | 0.75 | 0.91 | 0.42 |
| 8192 | 0.90 | 0.63 | 0.13 |
| 16384 | 0.50 | 2.25 | 0.77 |
| 16384 | 0.75 | 1.51 | 0.60 |
| 16384 | 0.90 | 0.97 | 0.34 |
| 32768 | 0.50 | 3.87 | 0.88 |
| 32768 | 0.75 | 2.66 | 0.79 |
| 32768 | 0.90 | 1.65 | 0.60 |
