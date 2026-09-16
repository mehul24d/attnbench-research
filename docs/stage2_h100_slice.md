# Stage 2 slice on H100 (sm90)

122 measured cells, `de0e8edd`, `git_dirty=False`, clocks locked 1695 MHz,
gated on the 492 Stage 1 passes recorded at the same commit on the same
machine. Slice: seq_len {4096, 16384, 32768} x batch {1, 16}, forward only,
dense + random-mask sparse. 2 further cells are recorded `oom` (both
`sdpa_math`, which materialises the full score matrix) rather than dropped.

## Dense causal, batch=1, `latency_ms_p50`

| backend | 4096 | 16384 | 32768 |
|---|---|---|---|
| flex | **0.436** | **5.563** | **21.889** |
| fa2 | 0.478 | 6.103 | 23.425 |
| sdpa_flash | 0.535 | 6.698 | 25.950 |
| sdpa_efficient | 0.965 | 13.421 | 52.318 |
| sdpa_cudnn | 0.261 | — | — |
| naive | 6.247 | 86.413 | OOM |
| sdpa_math | 15.943 | 243.362 | OOM |

`flex` is ahead of `fa2` at every length here, by 7-9%. `sdpa_cudnn` has the
fastest 4096 number in the table and no rows above it, because it faults with
an illegal memory access at 16384 and 32768 (recorded, synchronous -- see the
runbook).

## flex vs block_sparse, and the bias that had to be removed first

`block_sparse` calls `mask.to_block_sparse_attn_mask(...)` inside `forward`,
so the conversion runs on **every timed call**. `flex` hoists its equivalent.
Comparing the two directly therefore measures a real kernel difference plus an
artefact, in a direction that always favours flex.

**Measuring the tax rather than caveating it.** At fixed seq_len and
block_size, the conversion cost is constant in sparsity while kernel work is
proportional to the retained block fraction. So fitting

    latency = T + k * (1 - sparsity)

over the three sparsity points gives an intercept `T` that is the
sparsity-independent per-call cost. The fits are tight -- worst residual 1.3%
across all backends and shapes -- and `naive` is the control that validates
the method: it ignores the mask and computes dense attention, and its fitted
slope is ~0 (|k|/T under 6%), exactly as a backend that cannot benefit from
sparsity should behave.

| shape | flex T | block_sparse T | excess |
|---|---|---|---|
| 4096, b=1 | 0.099 ms | 0.456 ms | **0.357 ms** |
| 16384, b=1 | 0.339 ms | 0.726 ms | **0.388 ms** |
| 4096, b=16 | 1.297 ms | 4.241 ms | **2.944 ms** |

Both backends have a nonzero intercept -- launch overhead, pointer setup and
the like are real and paid by any deployment. What is *artefactual* is
`block_sparse`'s **excess** over flex, so that is what the correction removes.
Removing all of it is deliberately generous to block_sparse: some of that
excess is probably legitimate kernel overhead rather than mask conversion, so
the corrected ratio is a **lower bound on flex's advantage** and the raw ratio
is an upper bound.

| seq_len | batch | sparsity | flex | block_sparse | raw | corrected |
|---|---|---|---|---|---|---|
| 4096 | 1 | 0.50 | 0.294 | 0.729 | 2.480 | **1.266** |
| 4096 | 1 | 0.75 | 0.194 | 0.583 | 3.012 | **1.169** |
| 4096 | 1 | 0.90 | 0.139 | 0.515 | 3.703 | **1.134** |
| 4096 | 16 | 0.50 | 3.902 | 7.359 | 1.886 | **1.131** |
| 4096 | 16 | 0.75 | 2.580 | 5.726 | 2.220 | **1.078** |
| 4096 | 16 | 0.90 | 1.827 | 4.899 | 2.681 | **1.070** |
| 16384 | 1 | 0.50 | 3.102 | 4.302 | 1.387 | **1.262** |
| 16384 | 1 | 0.75 | 1.717 | 2.522 | 1.469 | **1.243** |
| 16384 | 1 | 0.90 | 0.893 | 1.438 | 1.610 | **1.176** |

**The correction is large and the conclusion survives it.** Raw ratios reach
3.70x; corrected they reach at most 1.27x. So an uncorrected reading overstates
flex's advantage by up to ~3x, which is exactly the size of error worth
catching. But **the direction reverses in 0 of 9 cells**: flex is faster than
block_sparse everywhere in this slice, by at least 1.07x, and the artefact
was never load-bearing for that claim.

The artefact matters most where the kernel work is smallest -- the raw ratio
grows monotonically with sparsity (2.48 -> 3.70 at 4096/b1) because a fixed
per-call tax is a larger share of a shorter kernel. Any future claim quoting a
raw block_sparse latency at high sparsity is quoting mostly conversion.

## What is not here

- **block_sparse at 32768**: absent. Its backward kernel faults there
  (`b=16, hkv=32, fwd_bwd`), so Stage 1 licensed nothing at that length and
  Stage 2 correctly ran nothing. `flex`'s 32768 sparse cells consequently have
  no peer.
- **16384, batch=16, flex sparse**: fewer than 3 sparsity points survived, so
  no intercept could be fitted and no corrected ratio is quoted.
- **gla, sage, xformers**: unlicensed on this machine for three unrelated
  non-hardware reasons. See `limitations.md`.
- **fwd_bwd**: Stage 1 validates forward only, so no backward cell is
  licensed and none was measured.
