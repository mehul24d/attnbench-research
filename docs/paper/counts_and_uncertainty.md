# What the sign counts count, and per-cell uncertainty (28 Nov 2026)

The paper-quality wave of 28 Nov 2026, item 4. The counts in the first section restate scored or exploratory results unchanged. The intervals in the second are **EXPLORATORY / POST HOC**. They are generated in `docs/paper/exploratory/paper_tables.md` by `scripts/paper_tables.py`. P2 and P4 are summarised separately, and no significance test is pooled across pairings.

## The three counts

Each pairing has 35 cells: 6 dense, 29 sparse (mean-pool 10, oracle 9, XAttention 5, XAttention without asserts 5).

| Figure | What it counts | File and line | Code |
|---|---|---|---|
| **25/25** (P2) | H1's sign part, pre-registered. Of P2's 29 sparse cells with a measured dense partner, 4 fall inside the cell's resolution floor and are not scored. The other 25 are scored, and all 25 give the predicted sparse-minus-dense sign (0 disagree). 3 of the 25 are labelled marginal | `docs/scoring/hypotheses_scored_20261116.md` line 8 ("sign 25 agree, 0 disagree"); `.json` line 102 | `attnbench/analysis/hypotheses.py:266` (`_signs`) |
| **25/27** (P4) | The same count on P4: 29 sparse cells, 2 inside the floor, 27 scored, 25 agree and 2 disagree, so H1's sign part fails on P4. 4 of the 27 are marginal | same file, line 9; `.json` line 236 | same |
| **58** | EXPLORATORY paired sign check: the 29 sparse cells of P2 plus the 29 of P4. Per cell it runs a paired bootstrap over that cell's 10 rounds, pairing round i of the sparse row with round i of the same file's dense row. "All 58 exclude zero" means each cell's own interval for median(sparse) − median(dense) excludes 0 | `docs/exploratory/exploratory_wave_20261116.md` line 13; `docs/exploratory/results_20261116.json`, `item5_sign` | `scripts/exploratory_wave.py:184` (`item5`), `attnbench/exploratory/sign.py:49` (`paired`) |

**The 58 are cells, not replications.**
- Each interval resamples rounds inside one ground-truth file in one session: P2's single session, or P4's block b3 (12 cells) or b4 (17 cells).
- So an interval reflects round-to-round noise within that session only. It says nothing about session-to-session, instance-to-instance or provider variance.
- The 58 cells share sessions, dense rows and machines, so they are not 58 independent confirmations. The count must not be read as a pooled test.
- Breakdown:
  - **P2**: 25 dense faster and 4 sparse faster. Against the ±floor band: 22 dense faster, 3 sparse faster, 4 undecided.
  - **P4**: 28 dense faster and 1 sparse faster. Against the ±floor band: 27 dense faster, 2 undecided, 0 sparse faster.

## Per-cell intervals (EXPLORATORY / POST HOC)

For each cell, a percentile bootstrap on the median over its rounds, at the lock's settings (level 0.95, 10,000 replicates, seed 0). From it comes the interval of the committed recursion's signed relative error. Each cell's median over its rounds reproduces the measured value the scorer used, with a largest difference of 0.

| | P2 | P4 |
|---|---|---|
| Cells | 35, each 10 rounds | 35: 34 with 10 rounds, 1 with 20 (dense 8075, pooled over b3 and b4 as scored) |
| Median interval width / median | 0.0051 | 0.0088 |
| Largest | 0.0216 | 0.0467 |
| Recursion error interval vs H1's tolerance (±0.1): within | 17 | 6 |
| straddles | 0 | 1 |
| outside | 18 | 28 |

**Reading.**
- Within-session round noise is under 1% of the median in a typical cell.
- So H1's misses are not a noise artefact at the round level: on P4, 28 of 35 cells sit outside ±0.1 by more than their round interval.
- These intervals are conditional on the session. They are not a test, and they do not change H1 (P2 0.1183, P4 0.2495).
