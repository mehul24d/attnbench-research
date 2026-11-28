# Reproducibility: the clean-clone run (28 Nov 2026)

This is the paper-quality wave of 28 Nov 2026, item 7. `scripts/paper_reproduce.py` regenerates every table the paper draws on from banked data. It then compares each table with its committed copy.

## How it was run

The branch was cloned fresh with `git clone --branch paper-quality-2026-11-28` into a scratch directory, at commit `0057403`. The tree was clean. The run used Python 3.12.1 with torch 2.13.0, on CPU only, with no cloud. The command was:

    python3 scripts/paper_reproduce.py --banked <attnbench_backups> --out <scratch>

It exited 0.

## Results

| # | Step | Result |
|---|---|---|
| 1 | H0–H3 scoring (`score_hypotheses.py`, offline from the scoring wave's mirror) | `scores` identical to `docs/scoring/hypotheses_scored_20261116.json` |
| 2 | Exploratory wave of 16 Nov (`exploratory_wave.py`) | Identical except 29 `block` labels in `item5_sign/P2`, which read `t61_p2_20261114T0022Z` for the committed `p2_session`. The label is the name of the local directory P2's session was read from |
| 3 | Residual split of 22 Nov (`exploratory_residual.py`) | Identical to `docs/exploratory/residual_20261122.json` |
| 4 | Paper tables of 28 Nov (`paper_tables.py`) | Identical to `docs/paper/exploratory/paper_tables.json` |
| 5 | `qa_1`/32768 within-limit sensitivity (`t11_qa1_within_limit.py`, tracked parquets) | Ran. Its built-in check reproduced the certified all-rows bound of −5.97 (reported as −6.0). The within-limit bound is −4.22 with n = 80 |
| 6 | P3's exploratory score | **Not run.** `scripts/score_p3_exploratory.py` is only on the unmerged branch `p3-g6-2026-11-26` (`584516a`) |
| 7 | Figures | **None exist.** `paper/figures/` has no generated figure. Fig. 1 needs T3.1 inputs, and `paper/generated/numbers.tex` (task C10) is not built |

## Known gaps, reported as they are

1. **The banked data are outside the repository.** The run needs the scoring wave's mirror (995 MB), P2's session (pulled 17 Nov) and P2's traces (pulled 21 Nov). It also needs P4's traces and P4's dense-selection step. Without the backup directory nothing in steps 1 to 4 runs.
2. **The scoring wave's inputs file has stale paths.**
   - It names absolute paths on the Mac where it ran.
   - Its P2 prediction copy pointed at a scratch directory, which has since been purged.
   - The script rewrites those paths and reads P2's prediction from the mirror. The scorer checks that prediction's sha256 either way.
   - The original file is left untouched.
3. **The two parts of steps 1 to 4 need different access.**
   - The comparison itself is offline.
   - The mirror was recorded once from the store, on 16 Nov 2026. Rebuilding it needs store access, which this wave did not use.
4. **P3 and the figures** are not covered (rows 6 and 7).
