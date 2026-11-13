# Mutations: the arm's own kernels module, the arm-parity test, the smoke's K2 screen and `tag_lock` (13 Nov 2026)

The adviser's rulings dated 13 Nov 2026, items 1, 2, 3 and 6.

**Method.**
- Each mutation was applied alone to the working tree and run against the named tests with
  `PYTHONDONTWRITEBYTECODE=1`. The tree was restored after each one.
- "Killed" means at least one test failed.
- Results: 12 of 12 killed.

## Items 1 and 2: the Triton null entries and arm parity

Tests: `tests/test_arm_module_parity.py`, `tests/test_xattention_triton_entries.py`.

| # | Mutation | File | Killed |
|---|---|---|---|
| M1 | `null_entries` nulls `xattn.src.kernels` whatever the arm, so the alias module is left unpatched (the `lock-2026-11-13` behaviour) | `scripts/t31_steps.py` | yes |
| M2 | the stripped arm's `kernels_module` names the shipped module | `backends/xattention_noassert.py` | yes |
| M3 | the stripped arm's `estimate` dropped from the probe's score targets | `ground_truth.py` | yes |
| M4 | the Triton entries dropped on `xattention_noassert` | `scripts/t31_steps.py` | yes |
| M5 | the mask capture's kernel module renamed | `components/mask_bank.py` | yes |

## Item 3: K2's screen in the smoke

Test: `tests/test_preflight_smoke.py`.

| # | Mutation | File | Killed |
|---|---|---|---|
| M6 | on a GPU the smoke screens 2048 batch 1 only, not every cell K2 screens | `scripts/t31_steps.py` | yes |

## Item 6: `tag_lock` takes `lock-YYYY-MM-DD-rN`

Test: `tests/test_tag_lock.py`.

| # | Mutation | Killed |
|---|---|---|
| T1 | back to date-only | yes |
| T2 | any suffix accepted | yes |
| T3 | `-r0` and `-r1` accepted | yes |
| T4 | `-R2` accepted | yes |
| T5 | `match` instead of `fullmatch` (a trailing newline passes) | yes |
| T6 | the format check skipped | yes |

**Drivers.** The drivers are in the session scratchpad and are not banked:
- `mut12.py` (M1 to M5);
- `mut6.py` (T1 to T6);
- M6 was applied by hand.
