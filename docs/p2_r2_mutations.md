# Mutations: the arm's own kernels module, the arm-parity test, the smoke's K2 screen, `tag_lock` and the kernel contract in P2's and P4's preflight (13 Nov 2026)

The adviser's rulings dated 13 Nov 2026, items 1, 2, 3 and 6, and item 3 of the second set.

**Method.**
- Each mutation was applied alone to the working tree and run against the named tests with
  `PYTHONDONTWRITEBYTECODE=1`. The tree was restored after each one.
- "Killed" means at least one test failed.
- Results: 20 of 20 killed.

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
| M7 | the harness is smoked with every available candidate, not the screen's correct ones at 2048 (the `-r2` behaviour; added after P2 at `lock-2026-11-13-r2`) | `scripts/t31_steps.py` | yes |
| M8 | `smoke_gt_controls` returns every candidate | `scripts/t31_steps.py` | yes |

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

## Second set, item 3: the kernel contract in P2's and P4's preflight

Test: `tests/test_preflight_kernel_contract.py`.

| # | Mutation | File | Killed |
|---|---|---|---|
| K1 | a refusal or a failed status is not a problem | `scripts/t61_session.py` | yes |
| K2 | a result that is not clean is not a problem | `scripts/t61_session.py` | yes |
| K3 | the preflight drops the contract's problems | `scripts/t61_session.py` | yes |
| K4 | the contract never runs | `scripts/t61_session.py` | yes |
| K5 | the smoke check runs after a refused contract | `scripts/t61_session.py` | yes |
| K6 | the contract skips the real prompt's padded-against-unpadded comparison | `attnbench/bsa_contract.py` | yes |

**Drivers.** The drivers are in the session scratchpad and are not banked:
- `mut12.py` (M1 to M5);
- `mut6.py` (T1 to T6);
- M6 to M8 were applied by hand;
- `mutE.py` (K1 to K6).
