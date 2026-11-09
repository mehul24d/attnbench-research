# The L4 rehearsal session: scope (for its owner)

This is for whoever writes the rehearsal card (adviser's ruling 6 of the
eleventh 9 Nov 2026 paste: own worktree and session; card by Wed 12 Nov,
results by Thu 13 Nov; exploratory only, never training or claims; about
₹400, with (iv) cut first; not launched without the researcher's yes naming
the instance, zone, commit and cap).

**In scope, by priority** (ruling 6, 9 Nov 2026):

1. The real ground-truth harness on a reduced grid, measuring each arm's
   trace file sizes.
2. P2's mask-gate path against P1's saved masks. Agreement on the same
   hardware is the code-correctness test, and it is T2.23's only GPU test
   before P2.
3. The block-aware P2/P4 runner in two small blocks: block spec, anchor,
   drift check, halt, self-halt, and the discard of an interrupted block,
   simulated.
4. Compiled ground truth on a reduced grid, with the recompile-limit
   handling. This is cut first.

**Also in scope** (adviser's ruling 4 of the 10 Nov 2026 paste). Both are
P2 prerequisites that do not read Done.

- **T2.17:** the bitwise mask and output gate of the asserts-removed
  XAttention arm against the shipped arm, on the L4. Shakedown 1's
  `assert_gate` failed on the prompt-length defect (27 Oct 2026) and has not
  run since.
- **T2.26:** the GPU variants of the five GQA tests (`tests/test_*_gpu.py`)
  run on the L4. `scripts/gpu_suite_record.sh` is wired into the launch
  scripts and the session preflight first (CPU work, before the card).

Where these sit against the four priorities, and how they share the ₹400,
is the card's to propose. A pass is recorded in each task's Section 8
status, with the session's ledger row as evidence.
