# Gate A on P1: leg 1 diagnosis, legs 2 and 3 bounds (T3.3 input, 8 Nov 2026)

**What this is.** The adviser's ruling 1 of 8 Nov 2026 (third paste), applied to the two
P1 repeats (T3.2a, T3.2b, both at `3f50aab`, eager). It is not the Gate A decision, which
is the researcher's (T3.3). Scripts, outside git: `scratchpad/tools/diag_leg1.py`,
`diag_residual.py`, `leg2_at_bound.py`.

## 1a. The plan's legs and what failure means

Section 7, Gate A, as written:

> The recursion beats the sum and the KernelSight-style step form on P1 (T3.3); P1's
> replay-versus-trace difference per operation family is within a bound fixed at lock;
> and the delay-injection slope check passes: each synchronisation mechanism is injected
> separately into the dense arm on the L4, where the GPU is busy and where it is idle,
> and the ratio of predicted to measured time per injected synchronisation is within a
> bound of 1.0 fixed at lock, one slope per mechanism (double counting gives about 2; an
> instrument check, Section 10).

If it fails:

> Stop; revise the model, the synchronisation costing, or the device-cost method if the
> replay check failed, before any new pairing

**The step form is not fitted.** It has no parameters. It is computed from the same
priced components as the recursion: per layer max(Σh, Σl + Σd), plus Σs₀
(`predict/baselines.py`, `step_form`; Section 5.4). Both predictors read the same
measured components from the same session.

## 1b. Legs 2 and 3 bounds (committed as `65ca210` before they were applied)

- **Leg 2: 0.25.** This is the smallest two-decimal value at or above every family's
  absolute relative difference in the two passing L4 shakedowns, which are the source
  the plan names (Shakedown 4: largest 0.2434; Shakedown 5: 0.2363).
  - It is applied as `replay.check_passes`.
  - Data had been seen: the T3.2 per-family distribution was reported earlier the same
    day.
- **Leg 3: each mechanism's slope within [1/√2, √2].** That range is centred, on a log
  scale, between the correct 1.0 and the 2 that double counting gives.
  - No slope has been measured.

## 1c. Leg 1 under cell-grouped cross-validation

The step form has nothing to fit, so cell-grouped cross-validation leaves both
predictors unchanged.

| | T3.2a | T3.2b |
|---|---|---|
| Recursion | 13.8% | 14.5% |
| Step form | 3.8% | 3.6% |

Leg 1 fails either way. The ruling's Appendix A line ("correcting a design flaw:
in-sample comparison favours fitted models") is **not written**, because no fitted model
is compared in leg 1.

## 1d. Where the recursion's error is

Median absolute relative error by arm. "Residual out" means each cell's in-layer host
costs are scaled down by that cell's per-layer residual total (step 5).

| Arm | Recursion | Step form | Recursion, no sync wait | Recursion, residual out |
|---|---|---|---|---|
| dense (a / b) | 0.7 / 1.5 | 1.1 / 1.5 | 0.7 / 1.5 | 1.1 / 1.5 |
| mean-pool | 4.0 / 2.2 | 4.2 / 2.7 | 4.2 / 2.7 | 1.9 / 2.0 |
| oracle | 14.4 / 20.2 | 10.7 / 17.3 | 13.6 / 20.0 | 2.4 / 4.6 |
| XAttention | 31.8 / 30.0 | 3.4 / 4.0 | 4.0 / 4.5 | 2.0 / 1.0 |
| XAttention, no asserts | 30.6 / 27.6 | 4.4 / 3.6 | 3.3 / 3.7 | 2.8 / 1.9 |
| **all cells** | **13.8 / 14.5** | **3.8 / 3.6** | | **1.85 / 1.96** (step form 2.42 / 2.44) |

**The error is in a measured component, not an omitted mechanism.** It is the per-layer
residual (T2.2; ruling of 1 Nov 2026). The residual is a layer's uninstrumented span
under the tape, with real device work running, less its bare prices.

- **It labels its own problem.** All 26 residual entries in each repeat's step-5 summary carry
  `DEVICE_BEHIND_LABEL` ("the device fell behind the host in this layer: its span may
  hold waits"). So the span holds device time, and that time is charged again as host
  time.
- **It is larger than the whole prefill.** In the oracle at 32768 the residual adds
  5,021 ms (T3.2b), against a measured end-to-end time of 2,988 ms. That much host time
  cannot be real.
- **Where it lands.** It goes on one position per layer, `_expand_kv`'s `aten::copy_`,
  at 178 ms each.
- **Why the recursion suffers more.** The recursion serialises that host time at every
  synchronisation (XAttention has 168 to 476 of them). The step form's per-layer max
  absorbs it whenever device time dominates.
- **With the residual out, the order flips.** The recursion is below the step form in
  both repeats.

**This is not a fix.** The residual-out figures come from a proportional rescaling
applied after the P1 ground truth was seen, and the per-position residuals are not
banked locally. There are also two caveats.

- **Density is lost in step 5's summary.** The summary (`residual_added_ms`, keyed
  `arm@LxB`) drops density, so for the oracle and mean-pool the 0.25 cell carries the
  0.5 cell's value. The per-cell `cell_index.json` keeps both.
- **Revisions don't count as predictions.** Any revision is a model change under Gate
  A's own failure response. Its predictions for P1 are no longer committed, so the
  revision can only be tested on a new session.

## Leg 2 at 0.25

Applied after `65ca210`, unchanged: **every cell fails in both repeats** (35 of 35).

- **No failure is structural.** Every position has a replay and none is mismatched.
- **Tiny families fail most often.**
  - `aten::cos` and `aten::sin` fail in 22 to 24 cells, with under 1 ms of traced time
    each over all 35 cells and a pooled difference of +3% to +6%.
  - `masked_fill_` fails in 19 to 20 cells: pooled +116% and +120%, 7 ms.
  - `nonzero`, `gather` and `index` each have about 3 to 6 ms in total.
- **Large families fail in single cells.** `mm` reaches 0.49 in one cell,
  `block_sparse_attn_forward` 0.57, `sort` 1.03, and `device_copy` 2.7.

The rule weighs every family of every cell equally, whatever its time. The shakedowns it
was set from had one cell each. **The bound is not moved.** Whether leg 2 should be
weighted by time is a lock decision for the researcher.
