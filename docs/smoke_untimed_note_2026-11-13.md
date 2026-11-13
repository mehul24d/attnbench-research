# The preflight smoke check is not a clocked block (13 Nov 2026)

The adviser's rulings dated 13 Nov 2026, item 5. Written and on origin before P2's launch at
`lock-2026-11-13-r2`.

**Ruling.** The smoke check's calls are not clocked blocks under the stamp rule. The stamp rule
refuses any timed block before `predict` is written, hashed and synced. The smoke check runs in
step 2 (`preflight`), long before `predict` (step 11), and that is allowed because:

1. **No timer runs.** No duration from the smoke reaches a timed path: not the ground truth's
   timer, `measure`, the K2 rounds or a pricing clock (`PricingClocks(enabled=False)`). The
   per-stage seconds in `smoke.json` are wall-clock bookkeeping of the preflight's own length.
2. **Every output is discarded.** The smoke writes into the preflight step's `smoke/` directory.
   Its findings appear in the preflight's result as a report. No later step reads them as an input:
   not the prediction, the floors, the ground truth or the analysis. Its step 7 and
   step 8 cells, its traces and its K2 results are not the session's step 7, step 8, trace or K2
   records. Those steps run again from nothing.

**What the smoke runs, all at 2048 batch 1 except K2's screen:**

- every arm through step 7's and step 8's cells;
- the pricing check;
- the prediction's input;
- one untraced and one traced ground-truth forward per dense control and per arm, under the probe
  and the throttle sampler, with the density twin;
- step 14's trace checks;
- K2's correctness screen at every cell K2 screens (`dense_selection.screen`; no timed rounds).

(`t31_steps.preflight_smoke`)

**What would make a smoke call clocked.** Either of these turns a smoke call into a clocked
block, which the stamp rule then refuses:
- a duration from it used anywhere downstream;
- a smoke output read by a later step.

Neither path exists. A change that adds one needs a new ruling.
