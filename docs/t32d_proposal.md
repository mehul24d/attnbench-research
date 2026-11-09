# T3.2d: proposal (9 Nov 2026; not a launch card, nothing launched)

**Superseded by `docs/t32d_card.md`** (adviser's rulings of 9 Nov 2026, seventh
paste). Options (a) to (c) below were all taken in, with the sustained-clock
soak; the plan, the limits and the cap are unchanged.

The adviser's ruling 8 of 9 Nov 2026 (sixth paste). A launch needs the
researcher's yes, naming the instance, zone, commit and cap. It also needs a
card built from this proposal once the open decisions below are settled.

## Scope: full (recommended)

| | Full | Reduced (the two 16384 groups only) |
|---|---|---|
| Steps | 1 to 12, all 35 cells, eager group (T3.2c's grid) | 1 to 12 on the 16384 cells (14 of 35) |
| Gives | A second repeat on the timed-path hash `4e78a962…`: leg 1 "in each repeat"; the between-session spread for every cell's floor; a second session of targets for the learned predictor | Clean rounds at 16384 only. No between-session floor elsewhere, and no full leg 1 repeat |
| Minutes (measured basis) | 199 | not measured: T3.2c's step times are not banked per cell length |

## Added steps

None are required. Step 12 (`delay_injection`) stays as declared in T3.2c,
with the same set, counts and reps.

Three options, each awaiting a decision:

| Option | What it is | Effect |
|---|---|---|
| (a) Leg 3 | One T2.5 link row timing `ev.record(); ev.synchronize()` per call (`docs/t32c_leg1_filters.md` §3) | Under 1 min. If it is used to price `event_synchronize`: 0 ms on every P1 cell, and step 12's idle prediction changes. If it is only recorded, leg 3 fails again on that mechanism |
| (b) Leg 2 | The session's NVML sampler kept running through step 5's recording and step 6's replay | 0 min. A measurement only; it tests the clock-state reading of §4 |
| (c) Oracle residual | The leg 1 blocker in §1 | Not diagnosed. Step 5's residual records are needed from the bucket first. A fix changes step 5's pricing, not the timed path |

Without (c), T3.2d reproduces T3.2c's oracle overprediction (2.2 to 6.6
times), so its leg 1 repeat would carry the same defect.

## Cost and cap (₹108.64/h; the rule of the T3.2a to T3.2c cards)

| | Minutes from boot | ₹ |
|---|---|---|
| Planned: T3.2c measured, boot to halt | 199 | |
| Option (a), rounded up | +1 | |
| **Planned** | **200** | **362** |
| In-guest halt: 1.3 × 200 (`GCP_HALT_MINUTES=260`) | 260 | 471 |
| Hard delete: the halt + 10 min (`GCP_MAX_RUN=4h30m`, DELETE) | 270 | **489, the cap** |

A fix under (c) that changes step 5's work changes this projection. It
would then be measured on a CPU-boundary rehearsal first and the card
rebuilt.

Spend:
- now ₹4,179 of ₹14,000 (headroom ₹9,821);
- with T3.2d at the cap, at most ₹4,668 (headroom at least ₹9,332).

## Calendar

T3.2 is due to launch by 9 Dec 2026, and the budget window ends 30 Nov 2026.

| Earliest | Step |
|---|---|
| After the push/merge | The decisions on (a) to (c); a card, with the self-halt verified on a CPU dry run |
| 10 Nov 2026 | The launch yes |
| About 3.5 h later | The session ends by itself |
| The next day | The teardown, the ledger row, and Gate A v2 on T3.2c and T3.2d. The floors are then recomputed between sessions |
| After that | T3.3, the researcher's decision |

## The revised identity: exactly what it contains

The base is `6c0326a`, T3.2c's identity. What changes:

1. **Step 9 status** (`scripts/t31_steps.py` `step_ground_truth` and `run_step`; `scripts/t31_session.py`).
   - A ground-truth cell with `complete: false` makes the step `incomplete` (exit 4), not passed.
   - The session records it, runs verify, postflight and step 12, and does not pass.
   - This is orchestration only. The timed ground-truth path is byte-identical: `scripts/timed_path.py 6c0326a 7375474` gives 86 items, `4e78a962…`.
   - T3.2a and T3.2b were not affected: all six groups in each had 10 of 10 clean rounds.
2. **Analysis only:**
   - `attnbench/analysis/drift_filter.py`, the relative drift filter applied after the session;
   - the session rule named in `scripts/cell_floors.py`.
3. **Optional, by decision:** (a), (b) and (c) above.

Nothing else in pricing or measurement changes. The identity commit is the
one that adds the T3.2d card. Self-halt: `scripts/session_exit.sh` (sync,
then halt, on any exit), as in T3.2c, verified again on a CPU dry run for
the card.

## Could the identity change affect P2/P4 code?

**Today, no.** `scripts/t61_session.py` calls T3.1's link, dense-selection,
probe and component steps and `build_prediction`. It does not call
`step_ground_truth` or `run_step`, and has its own statuses. Its
ground-truth harness is not built (`real_harness` raises `NotBuilt`).

**Later, possibly:**
- If P2/P4's ground truth is later wired through `_t31_step("ground_truth")`, an incomplete group arrives as `StepIncomplete`. That is a subclass of `StepFailed`, so P2/P4 would report it as failed, not incomplete. `t61_session.py` has no incomplete status.
- Option (a) would change T2.5's link row, which P2/P4 use.
- Option (c) would change step 5's host components, which P2/P4 also use.
