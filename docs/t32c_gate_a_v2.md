# Gate A v2 on T3.2c (9 Nov 2026)

**What this is.** Gate A v2 (Appendix A, 8 Nov 2026) read once, offline on
the Mac, on T3.2c (`results/t32c_p1_20261108T1555Z`, `6c0326a`, eager group),
from its bucket artifacts (read on the researcher's yes, `docs/permission_log.md`,
9 Nov 2026): the committed prediction (sha256 `173fc616…`, matching step 7's
record), step 6's `cells.json`, step 5's recordings of all 35 cells, step 9's
ground truth and step 12's `delay_injection.json`. The bounds are the ones
written before T3.2c: the family bound 0.25 (Appendix A, 27 Oct 2026; not yet
a lock value, so labelled provisional), leg 3's range [1/√2, √2], and the
resolution floor per (length, batch) by the amended rule, recomputed after
T3.2c. **It is not the Gate A decision**, which is the researcher's (T3.3).

## Leg 1 is not read: two ground-truth groups are incomplete

`scripts/gate_a.py --v2` refuses T3.2c's ground truth as written:
`eager/gt_L16384_b1.json is not complete (too few clean rounds)`. Step 9
passed, but two of its six groups have fewer than 10 clean rounds, all lost to
the dense control's drift flag (no rerun round was run):

| Group | Clean rounds | Dense drift per round (ms), against the 52.83 ms floor |
|---|---|---|
| 16384 × 1 | 8 of 10 | −21, −52, −28, −51, **−55**, −45, −42, −45, −49, **−56** |
| 16384 × 4 | 2 of 10 | −37, **−93, −115, −128, −105, −119, −107, −77, −65**, −48 |

The drift is one-sided: the dense control ran faster than its warm-up
reference. The floor is one absolute value for every length (0.8% of the
6.6 s prefill at 16384 × 4). The other four groups (2048, 4096, 8192 and 32768,
21 of the 35 cells) are complete. Leg 1 is left unread until the researcher
rules how an incomplete group enters it. Legs 2 and 3 read no ground truth and
are read below, through the same functions `gate_a.py --v2` calls.

## Leg 2: fails in 3 of 35 cells

| Cell | Family | Relative difference |
|---|---|---|
| oracle@0.25, 16384 × 1 | `aten::mm` | 0.354 |
| oracle@0.5, 16384 × 1 | `block_sparse_attn_forward`; `aten::mm` | 0.378; 0.288 |
| oracle@0.25, 32768 × 1 | `block_sparse_attn_forward` | 0.348 |

Of 1,143 (cell, family) pairs, 994 are exempt (below their cell's floor), and
149 are scored: **the scored share is 13.0%**. This is reported as a finding.
The floor's form was not chosen by it (Appendix A, 8 Nov 2026).

## Leg 3: fails on one mechanism

Pooled slopes (predicted over measured): synchronize 1.052, item 1.067,
bool 1.055, copy to host pinned 1.035, pageable 1.040, copy from host pinned
1.040, pageable 1.059; **event_synchronize 0.386**, outside [0.707, 1.414].
Its busy placement agrees (ratio 0.85). Its idle placement does not: predicted
1.16 µs per injection, measured 8.59 µs (ratio 0.135). All 64 variants were
timed, none refused. The observer effect is +1.8% (2.4 ms on 132 ms).

## The resolution floors (amended rule, quantile 0.95, provisional)

From T3.2a, T3.2b and T3.2c's ground truth (`scripts/cell_floors.py`):

| (length, batch) | Floor ms | Within | Between | Set by |
|---|---|---|---|---|
| 2048 × 1 | 4.25 | 1.70 | 4.25 | between |
| 4096 × 1 | 8.04 | 8.04 | 7.00 | within |
| 8192 × 1 | 69.49 | 23.08 | 69.49 | between |
| 16384 × 1 | 111.05 | 15.04 | 111.05 | between |
| 16384 × 4 | 441.54 | 15.35 | 441.54 | between |
| 32768 × 1 | 321.53 | 25.81 | 321.53 | between |

Two caveats:
- The between-session spread spans two code identities. T3.2a and T3.2b's
  timed path hashes to `a5ade053…`, T3.2c's to `4e78a962…`. The rule names
  all three sessions.
- T3.2c's 16384 series hold 8 and 2 clean repeats.

Reproduce (paths to the bucket copies):

    python3 scripts/cell_floors.py --session T3.2a <t32a>/09_ground_truth --session T3.2b <t32b>/09_ground_truth \
        --session T3.2c <t32c>/09_ground_truth --quantile 0.95 --provisional --out floors
    python3 scripts/gate_a.py --v2 --label "T3.2c" --remote <the committed file in the bucket> --local <its copy> \
        --ground-truth <t32c>/09_ground_truth --cells <t32c>/06_device_components/cells.json \
        --recordings <35 cells' step-5 trace.json and sequence.json> --relative-floor 0.0677 \
        --resolution-floor-ms 19.42 --family-bound 0.25 --family-bound-provisional \
        --cell-floors floors/cell_floors.json --delay-injection <t32c>/12_delay_injection/delay_injection.json --out gate_a

The second command stops at the incomplete group, as above.
