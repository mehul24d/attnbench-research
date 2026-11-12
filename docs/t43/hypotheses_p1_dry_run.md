# The confirmatory analysis on P1: dry run on P1, the development pairing: not a test (T4.3 wave 2a)

Prediction `280a34238a09`; working values `docs/t43/dry_run_working_values.json`.
Groups left out (short of clean rounds): 0; unmeasured cells: 0.

## H1, read on P1 (not a test)

- Recursion median |relative error| 0.0505 over 35 cells (tolerance 0.1): holds.
- Rivals: sum 0.5121 (recursion below: True), step_form 0.0196 (recursion below: False); not in the file: ['fitted_sum', 'learned'].
- Sign: 26 agree, 0 disagree, 3 inside the floor, 1 marginal.
- XAttention pair: 0 agree, 1 disagree (fails).
- As H1's verdict would read: **fails** (development pairing).

## H3's two halves on P1 (not a test)

| Estimator | Status | Verdict |
|---|---|---|
| meanpool@0.25 | compared | holds |
| meanpool@0.5 | compared | holds |
| oracle@0.25 | compared | fails |
| oracle@0.5 | compared | holds |
| xattention@calibrated | both none in range | not scored |
| xattention_noassert@calibrated | both none in range | not scored |

## Within-session spread (descriptive)

2048x1: 1.10 ms, 4096x1: 7.67 ms, 8192x1: 18.42 ms, 16384x1: 13.77 ms, 16384x4: 27.59 ms, 32768x1: 32.11 ms

## Plumbing

H0 of P1 against itself: {'operations_scored': 112, 'operations_unscorable': 120565, 'verdict': 'holds'}; H2 of P1 on its own host times equals its recursion: True.
