# T4.3 adversarial read A: scoring and gate arithmetic

Commit 23f540e (= origin/main when I started). Clone: `…/scratchpad/repo` (untouched). Mutations and reproductions: `…/scratchpad/mut`; reproductions are `tests/test_read_a_repro.py` there (17 tests, each asserts the plan's behaviour, so FAILED = reproduced). CPU suite on a clean copy: see end. Known open items confirmed in one line: lock values are null in the draft; the timed-path hash is not yet checked at session start.

## Findings, most severe first (all reproduced unless stated)

**1. H6 scores nothing under the study's own keys.** `dispatcher.py:80,359` filters `p.point.context_length in (16384, 32768)`. Appendix A, 9 Nov (item 9) says cells are keyed by realised tokens (T3.2a-d: 1951…32700). Test F2: points at 16370 and 32700 give `n_scored=0` and `unscored` empty, so the points vanish unlisted. F2b: with zero scored points `h6_line(...).no_more_decision_errors` is `True` (0 <= 0). Plan §4 H6: "at 16384 and 32768 tokens". Wrong result: H6 reads "no more decision errors" over an empty score. (Nothing calls `evaluate` outside tests yet.)

**2. Batch 4 is scored.** `score_policy` filters length only. §4 H6: "at batch 4 every rule chooses dense, so batch 4 is not scored (Appendix A, 1 Nov 2026)". F1: a 16384×1 and a 16384×4 point give `n_scored=2`; the batch-4 point adds regret 0 and a CORRECT outcome, so one extra zero halves the mean regret that is compared with the H6 [lock] bound. Makes the bound easier to meet.

**3. H6 compares policies on different points.** `h6_line` ignores `n_scored`/`unscored`. F4: the dispatcher's chosen config was not measured at one point → unscored for it only; the rule is scored there. Dispatcher `n_scored=1`, rule `2`, and the mean-regret comparison is over unequal sets. Exact float match of `Config.density` (XAttention's executed density differs by session, per `gate_a_repeats` docstring) makes this reachable.

**4. One scalar floor for every length.** Appendix A, 8 Nov: "the resolution floor is per (context length, batch)" (9 Nov: 13.77 ms at 16384, 101.16 ms at 32768). `dispatcher()`, `evaluate`, `DispatchSpec.floor_s` (the committed file), `scoring.score_cells`, `crossover_brackets(diffs, floor)`, `hdbi_fit.fit_threshold`, `gate_a.sanity` all take one float (F3, F3b). A 50 ms predicted gain is sparse under 13.77 and dense under 101: no single value gives the plan's answer at both lengths. The dispatcher can recommend sparse where the plan's rule says dense.

**5. Plan statistics with no code (grep, no `def`, no tests).** H0's 10 Nov statistic (median over operations above the cell floor, max and coverage share beside it); H1's per-pairing verdict (10 % median, four comparators, sign outside floor, XAttention-pair difference); H2; H3's "inside the interval widened by the bootstrap"; H4's equivalence test and its m = 18 Holm family (`holm.py` handles only the admissibility family, m = len(declared set); 18 appears nowhere in code); the 16384-only "marginal" label (L796); H5. Bootstrap level, `n_boot`, seed are not T4.1 items and `intervals.bootstrap` has no caller outside tests. Q5 answer: Holm and the bootstrap are internally deterministic and seeded, but their size and order are not fixed by anything the plan requires.

**6. Block drift: right semantics, wired to nothing.** `block_drift.drift` is imported by no script or module (F5d). Inside: flag-not-exclude is correct (mutation B2 killed), `claimable` returns all rows, `sensitivity_sets` drops one flagged block, anchor from the spec. Defects: (a) a block with no anchor row is silently left out and `passed=True` (F5a); (b) a NaN anchor median passes, `abs(nan) > bound` is False (F5b); (c) the reference is the first block with an anchor, so if it is the odd one out it is never flagged and the sensitivity run drops the wrong blocks (F5c: 100/108/108 flags b2 and b3, not b1); (d) `bound` defaults to `WORKING_BOUND`, so a caller that omits it silently uses 6.77 % instead of the lock value.

**7. Gate A.** (a) `gate_a_repeats.as_written` is v1-only: `_leg3` needs `within_bound`, v2 gives `within_range`, so a v2 session reads "not applied: no slope bound" (F7); `scripts/gate_a_repeats.py` has no `--v2`, cell floors or delay injection. No code produces the v2 verdict from both repeats; the 9 Nov reading was by hand. Fails closed, so no easier pass. (b) `check_passes`: NaN relative difference passes (F6a); a cell whose families are all exempt passes vacuously (by design, listed in JSON). (c) `gate_a.replay_report` passes no `Match` and builds `priced` from `trace_s`'s own keys, so `check_passes`' kernel-name and no-replay guards are never fed on Gate A's path (F6b). §5.6 "Replay mismatch": names and durations compared. (d) `short_of_rounds` treats a missing `clean_rounds` as ≥ 10 (F8); `clean_rounds`, `complete` and `prefill_ms_median` are read from the file, never recomputed from `prefill_ms_samples`, which the floors use (code reading; not run). (e) `_leg1` HOLDS from `recursion_below` alone; its verdict carries no cell count or exclusions (F13). (f) `slopes(..., mechanisms=())` returns `within_range == {}`, vacuously true (F7b; the script passes `PRICED_MECHANISMS` under `--v2`).

**8. Degenerate intervals count as determined.** `bootstrap` allows one repeat and `n_boot=1`; the interval is [0.5, 0.5] (F12), and `sign_at` then calls any difference above the floor determined.

## Answers

- **Q1.** Matches: scoring tie rules, `sign_agreement`/`decision_error` floor handling, step form and Sum (no s_0), fitted-sum LP loss, nearest-pairing rule, Holm step-down, exact NI (α/2 per tail, `> -margin`), recursion-below strictness. Mismatches: findings 1-5, 7.
- **Q2.** Drops are counted: inside-floor cells (`inside_floor`), short groups (`excluded_groups`), unmeasured cells (`unmeasured`). Unlisted: finding 1's points, finding 3's dispatcher-only drops, finding 6a. Leg 1's verdict does not carry exclusions. The marginal label is not implemented, so nothing drops by it. Inside-floor residuals stay in the H1 median (`residuals.py:132`); plan says "not interpreted", ambiguous.
- **Q3.** Yes: findings 6a-b, 7b, 7e, 7f, 8. Holm and exact NI fail closed at tiny n.
- **Q4.** See 6.
- **Q5.** See 5.
- **Q6.** `WORKING_BOUND`, `MIN_CLEAN_ROUNDS` are constants; `analysis/decision.py` `SPEEDUP_RESOLUTION_BANDS` (0.006/0.017) is a hard-coded legacy pilot floor used by `run_decision_map.py` only, not by H0-H6; 7d's self-reported fields.
- **Q7.** Rule implemented (`> floor`, ties to dense, then smaller `(arm, density)`); consistent with scoring's strict/`<=` floor edges (S1/S2 killed). Exceptions: findings 1-4. `best_kernel_config` averages over every point including 2048/4096 and batch 4 (unconfirmed: plan silent).

## Mutations (59, on code that produces reported numbers; guard counts as verified only if KILLED)

Killed 51 of 59. **Survived (guard unpinned):** D5 `choice_outcome` missed-profitable `gap > floor`→`>=`; B1 drift `> bound`→`>=` (plan: "above"); R1 `check_passes` `> bound`→`>=`; X2 `sign_at` positive side ignores the interval (negative side killed); X5 `contains` boundary; E2 `non_inferior` `>`→`>=`; E3 Holm `_clears` `>`→`>=` (E2/E3 are near-equivalent: exact equality on a float bound); I3 interval `point` = replicate mean (docstring says full-sample). Not applicable: none.

## Unconfirmed (not reproduced)

`overlaps` uses strict index inequality, so brackets sharing one grid length do not overlap (plan silent). HDBI fit docstring says candidate 1.0 = "every cell host-bound", false if a row has HDBI exactly 1.0 (diagnostic only). `step_form` does not split a step at a synchronisation (plan: "per layer").

## Not read

`ground_truth.py` flagging logic, `training.py`/`learned.py`/`commit.py` beyond the ordering check, `delay_injection.py`, the P2 runner internals, `scripts/cell_floors.py` wiring, model code. Full suite read-through of tests was by file name and mutation response, not line by line.

## Secondary pass (lightly checked)

`session_order.py` matches the plan's 60 s + 30 s numbers; `hold`/`ready_to_clock` use `>=` where the analysis refuses `>=`, so the guard is what keeps them apart (fine while guard > 0 or skew differs). `checklist.py` checks presence only: `SCORED_REQUIRED` is H0 and H2, so H1/H3/H6 pairing lists are not required, and `P1` is not refused in an H1 list (plan: P1 never counted). `_unset` accepts any non-empty source string. The lock check cannot catch a value that contradicts a plan line (e.g. m = 30 vs 18).

## Against docs/t43_first_read.md (read after the findings above were written)

It is dated 8 Nov and tests the P1 repeats, so it predates the 9-10 Nov rulings. **Agree:** one absolute floor across a wide range of cell times was a defect (its item 2); `floors.py` now computes per (length, batch), but my finding 4 shows every consumer still takes one scalar, so that fix stopped at the floors module. Its item 4 (3 % drift bound below the observed gap) is superseded by the 6.77 % flag, which I confirm is a flag (B2 killed); its item 7 (`median_ms` adapter) is in place (`t61_session.t31_ground_truth_result`). Its item 10 (leg 1 per repeat) matches `_leg1`. **Disagree:** none found. **Missed:** everything in my findings 1-8. Its scope was floors, Gate A and blocks on P1 data, so it never reached the P2+ scoring paths (dispatcher/H6, H0, H1 verdict, H4 family), the unwired drift check, or Gate A's dead guards. It lists fixes as "break-tested"; by my mutations B1 (boundary) and R1 still survive.
