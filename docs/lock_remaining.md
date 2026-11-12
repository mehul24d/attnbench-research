# What the lock still waits on (12 Nov 2026, after the lock-wave rulings)

The lock (T4.4) is Sat 14 Nov 2026. `docs/lock_values_prep.json` (built by the session's
`build_lock_prep.py`) holds 39 of T4.1's 40 values, typed as `attnbench/lock/checklist.SCHEMA`
requires, with the scored pairings and P2's and P4's `pairing_values`. Five values are typed
"not applicable" (rows 2 to 6 below), and H4 and H5 are stated "not scored". It is not the
lock: `docs/lock_values.json` does not exist yet, so P2 and P4 still refuse.

**A runner reads nothing from an incomplete lock.** `t31_steps.fill_lock_values` and
`t61_session.mask_gate_bound` refuse a lock the checker does not find complete. There is no
staged lock: the fourth set's deferral form is withdrawn (Appendix A, 12 Nov 2026).

## The one value left

**Row 10, the residual-spread tolerance (T4.1 item 34).** The ruling sets 0.10 only once
T3.2c's and T3.2d's spreads have been extracted and fall inside it. They are in the bucket
(`05_host_components/*/layer_residual.json` of `t32c_p1_20261108T1555Z` and
`t32d_p1_20261109T0726Z`). The only ones read so far, T3.2c's nine oracle cells, hold one layer
at 11.1%, above 0.10 (`docs/t32d_oracle_residual.md`). It needs the researcher's read-only yes,
and then a value.

## The 14-value table, as ruled (lock wave, 12 Nov 2026)

| # | T4.1 item (row) | Lock value |
|---|---|---|
| 1 | H6 regret bound (2) | mean 0, worst 0: the dispatcher no worse than each comparator (`dispatcher.h6_rule`) |
| 2 | Power target and n (6) | Not applicable: H4 not scored (T1.5 never ran) |
| 3 | Primary cell n, admissibility n (7) | Not applicable: H4 not scored |
| 4 | Admissibility Holm family (8) | Not applicable: H4 not scored (T1.8 never ran) |
| 5 | Cross-card dense task-score bound (12) | Not applicable: H4 and H5 not scored (audit B1 never ran) |
| 6 | Head-type criterion (22) | Not applicable: T1.3 not pursued (the researcher's delegation) |
| 7 | Null-kernel floor multiple (14) | 3.302 (15.52 µs at q 0.95 over TaxBreak's 4.7 µs) |
| 8 | Probe bounds (15) | loop CV 0.01, loop median 0.05, null median 0.10; analysis flags only |
| 9 | Chain-check tolerance (32) | 0.10 |
| 10 | Residual-spread tolerance (34) | **Not set** (above) |
| 11 | Pricing-mismatch share (35) | 0.05 |
| 12 | Rerun budget (23) | 5 reruns, 10 clean rounds |
| 13 | Bootstrap settings (38) | 0.95, 10,000, seed 0 |
| 14 | Prompt-set hash (26) | `cee769a2…` under 4.46.0, equal to 5.16.1's |

The researcher's items: 31 = 0; 18 = 0.8867 (re-run at the lock with
`scripts/hdbi_threshold.py`); 28 = P7, then the chunked extension. Scored pairings: H0 [P2],
H1 [P2, P4], H2 [P2], H3 [P4], H6 [P4], H4 and H5 not scored.

## At the lock commit

1. Set row 10. Rebuild the prep, commit it as `docs/lock_values.json` with `plan_sha256` of the
   plan at that commit.
2. `scripts/tag_lock.py --dry-run` must report the lock complete.
3. Re-check the timed-path pin against item 37 (`b5043080…` at `fb6dca8` unless the path changes
   again), and `analysis_commit` (`fb6dca8`'s full sha unless analysis code changes again).
4. Re-check P1's ledger entry's training-file sha256 against the lock's `p1_training`.
5. Run `scripts/prelaunch_cpu_pass.sh p2` and `p4` on the lock commit.
6. Tag the commit; pushing the tag needs its own yes, and so does P4's Flex-start request. Then
   build P2's Flex card from the lock commit, with the headroom table.
