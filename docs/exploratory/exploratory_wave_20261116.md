# EXPLORATORY: the exploratory wave (adviser's rulings, 16 Nov 2026)

**EXPLORATORY. No hypothesis is scored here; H0–H3 and 83b2edf are unchanged.** Numbers: `results_20261116.json` (offline, CPU).

**2. Error by component.** Substituting traced device time barely moves median |rel err|: P2 0.118→0.116, P4 0.250→0.208. Host and launch time dominate. P4's device error is all in dense attention: at 8075 tokens and above, traced FA3 is 35–46% of its prediction, and substituting it takes those errors from +0.21–0.42 to about −0.02. Sparse device time is within 1%. *Inference:* the dense sequence prices a slower kernel than FA3.

**3. Safe dispatch** (5 points per pairing, margin 0.02264). The three BUNGA policies choose identically; the margin changed nothing.
- P2: zero regret, 2/2 wins kept. Always-dense: 0.090 mean regret, 0.333 worst.
- P4: there are no wins. BUNGA is at worst 13.9% slower than dense (meanpool 0.25 at 16370, where the FA3 overprediction misleads it); always-dense regret is 0.004.

**4. Lower bound** (P1 fit: φc 0.325, φm 0.199, τ 33 µs/op). It does not hold: 20/29 P2 cells and 16/29 P4 cells break it. The lowest ratio is 0.43.

**5. Paired sign.** All 58 cells exclude zero. Against ±floor, 4 P2 and 2 P4 cells are undecided. P2 has 3 sparse wins; P4 has none.

**6.** 37 tests; 52/52 mutations killed; full suite 6703 passed, 179 skipped.

**Plan defaults (flagged):**
- Components are grouped by family name; estimator kernels are not separated.
- `fill_` is folded into FA3. The ~7 µs/kernel trace inflation at short lengths is kept.
- P1's zero-false-profitable margin was 0, so the margin is the largest P1 overstatement. P2/P4 cell values had already been shown before it was fixed.
- A win means beating dense by more than the floor.
- Peaks are spec-sheet values; estimator FLOPs are not counted; "close" means within 10%.
- Sample i is round i.

## Decisions needed

1. **Tier 2 (not launched):** one H100 Spot Nsight session on a few P4 cells, with FA3 on/off × CUDA graphs on/off, to test the FA3 inference and the host residual. It needs its own launch yes covering the command, cost, cap and hard-delete limit.
2. **Push/merge** of this branch, asked once.
