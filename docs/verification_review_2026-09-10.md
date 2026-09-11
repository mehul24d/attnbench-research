# attnbench Verification Review Record

Date: 2026-09-10

This document records the changes made during the verification-grade review,
the bugs confirmed by executable evidence, and findings deliberately left
unverified or downgraded. It is separate from the review checklist in
`docs/copilot_review_plan.md`.

## Scope and evidence standard

The review began from the repository baseline. There was no pre-existing code
diff; the worktree initially contained only the review documentation added in
the previous step.

Every finding is classified as one of:

- **CONFIRMED BY TEST**: a regression test failed before the fix and passed
  afterward.
- **CONFIRMED BY CALCULATION**: a concrete input and expected arithmetic were
  compared with runtime output.
- **READ-ONLY, UNVERIFIED**: the code path appears risky, but the review did
  not establish a changed result or a reachable bad input.

## Changes made

### Review documentation

- Added `docs/copilot_review_plan.md`, the 17-category numerical and logical
  integrity checklist.
- Added `.github/copilot-instructions.md`, which points repository reviews to
  the canonical plan without duplicating it.
- Added this document, `docs/verification_review_2026-09-10.md`, as the review
  change log and evidence record.

### Provenance subprocess handling

File: `attnbench/provenance.py`

`_sh()` now checks `CompletedProcess.returncode` before accepting stdout.
Failed commands with non-empty stdout are rejected instead of being treated as
successful. This covers the documented `git rev-parse HEAD` echo behavior and
failed GPU command output.

Before the fix, a stubbed command with:

- `returncode = 1`
- `stdout = "HEAD\\n"`
- stderr containing a Git failure

returned the string `"HEAD"`.

After the fix, it returns `None`.

### Decode correction step count

File: `attnbench/analysis/decode_confound.py`

`decode_corrected_ms` now subtracts `(n_generated - 1) * penalty`, matching the
module's documented generation identity: one prefill emits the first token's
logits, so only the remaining `n - 1` tokens incur decode-step cost.

For the regression case:

- measured latency: `130 ms`
- sparse decode: `15 ms`
- dense decode: `10 ms`
- penalty: `5 ms`
- generated tokens: `4`

The old result was `130 - 4*5 = 110 ms`. The corrected result is
`130 - 3*5 = 115 ms`.

### Regression tests

Added `tests/test_verification_review.py` with tests for both confirmed bugs:

- `test_sh_rejects_nonzero_exit_even_with_stdout`
- `test_decode_correction_subtracts_n_minus_one_penalties`

Both tests failed against the old implementation and passed after the fixes.

## Confirmed bugs

### Category 1: subprocess exit status inferred from stdout

**Evidence tier: CONFIRMED BY TEST**

The old `_sh()` accepted non-empty stdout regardless of exit status. This
could stamp invalid commit, clock, or GPU metadata as valid provenance. The
producer was fixed and guarded by the new regression test.

Potential claims impact: provenance-gated rows could have been accepted with
invalid metadata. No currently licensed `claims.md` number was recalculated in
this review.

### Category 14: decode correction off by one

**Evidence tier: CONFIRMED BY TEST and calculation**

The old correction subtracted one penalty for every generated token even
though the same module's normalization identity uses `n - 1` decode steps.
The concrete case above produced `110 ms` before the fix and `115 ms` after it.
The regression test now protects the arithmetic.

Potential claims impact: Stage 6 corrected latency and any downstream
comparison that selects `decode_corrected_ms`. The measured and normalized
columns remain separately named, so the exact affected claim requires a
consumer-specific audit.

## Findings deliberately not promoted to confirmed bugs

### Category 11: Pareto aggregation and unmatched populations

File: `scripts/run_pareto.py`

**Evidence tier: READ-ONLY, UNVERIFIED**

`latency_table()` averages by backend, task, band, and sparsity. The review
found no unequal `example_id` sets within the inspected real Stage 3 comparison
cells, so the stronger claim that unmatched populations currently change the
Pareto verdict was not confirmed.

A real regime mismatch does exist in the source data. At
`niah_multikey`, context `1972`:

- sparse 0.5: `2106.28 ms`, `47` generated tokens, `sdpa_math`
- dense: `1206.35 ms`, `32` generated tokens, `sdpa_flash`

This supports a category-11 warning about unequal generation length and decode
kernel, but the review did not demonstrate a changed Pareto verdict after
matched correction. No fix was applied.

### Category 11: phase-timing observed aggregation

File: `scripts/run_phase_timing.py`

**Evidence tier: READ-ONLY, UNVERIFIED**

The observed path averages `latency_ms` and `n_generated` by backend,
sparsity, and band without explicit example-set or decode-backend checks. Real
input contains token-count variation; for example, one sparse group contained
`2209.55 ms / 50 tokens` and `3166.00 ms / 72 tokens`.

The review did not show that a matched-subset recomputation changes the emitted
reconciliation verdict, so this remains a structural risk rather than a
confirmed claims bug. No fix was applied.

### Category 2: cross-architecture ratio domain

File: `attnbench/analysis/cross_arch.py`

**Evidence tier: CONFIRMED BY TEST**

`Speedup.speedup` initially divided baseline latency by backend latency without
a local finite-positive guard. The real
`results/cross_arch/speedups.parquet` artifact contains 396 rows with no
invalid, zero, or negative latency or ratio values, but a concrete canary probe
with backend latency `inf` returned ratio `0.0`. The shared ratio producer now
rejects such values with `CrossArchError`; the regression test is recorded in
the follow-up section below.

### Category 3: reconciliation sign bias persistence

File: `attnbench/accuracy/phase_timing.py`

**Evidence tier: READ-ONLY, UNVERIFIED**

A toy calculation with 24 residuals of `-1 ms` produced:

- every row: `closes=True`
- `bias_warning()`: `24/24` same-sign residuals, `p=1.2e-7`
- serialized reconciliation rows: no bias field

The warning is printed after the reconciliation parquet is written. The review
did not establish that a downstream consumer treats `closes` as the sole
reportability gate, so this remains unverified. No fix was applied.

## Checks that did not produce findings

- `analysis/composition.py` provides guarded aggregation and matched-subset
  helpers.
- `analysis/matched.py` enforces identical example-ID sets and a minimum paired
  sample size of 30 for bootstrap confidence intervals.
- `analysis/crossover.py` restricts cross-architecture comparisons to shared
  `(seq_len, batch)` cells.
- The existing compile-fallback guard is sticky at process scope and fails
  closed through its callers.
- The real cross-architecture speedup artifact contains only finite positive
  values.

These areas were reviewed but are not listed as bugs.

## Validation

The project environment was repaired using the declared `.[dev,eval]` extras.
The focused regression and analysis suites passed:

- `60 passed`

The complete test suite passed:

- `843 passed, 2 skipped`

`git diff --check` passed. No commit was created.

## Follow-up review pass

The incomplete backend, mask, gate, canary, cache, and subprocess surfaces were
reviewed after the initial record above. This pass added the following fixes
and tests.

### Flex mask cache keyed by mask contents

File: `attnbench/backends/impls.py`

The Flex `BlockMask` cache previously used only `(cfg.key(), device)`. Two
examples can share a config while having different importance-derived active
blocks, so the second example could reuse the first example's mask. The cache
key now includes a digest of `mask.active`.

Regression test: `test_flex_cache_distinguishes_masks_with_one_config`.

### Cross-backend and exact-gate output validation

File: `attnbench/gates.py`

Both correctness paths now refuse wrong-shaped or non-finite outputs before
computing differences. Before the fix, three NaN-producing cross-backend
implementations returned `passed=True` with `max_abs_err=0.0`; the new test
requires a failed result. The exact gate now returns a structured failed result
instead of raising on a malformed output shape.

Regression tests:

- `test_cross_backend_rejects_non_finite_outputs`
- `test_exact_gate_rejects_wrong_shape_without_raising`

### Device-aware memory budgets

File: `attnbench/gates.py`

`device_memory_bytes("cuda:1")` previously queried device 0. It now passes the
requested device through to `torch.cuda.get_device_properties`.

The corrected mocked two-device test failed before the fix (`10` returned for
the requested device whose memory was `20`) and passes afterward.

Regression test: `test_device_memory_uses_requested_cuda_device`.

### OOM classification preserved across adapters

Files: `attnbench/backends/impls.py`, `block_sparse.py`,
`xformers_backend.py`, `sage_attention.py`, and `linear.py`.

Optional backend adapters previously caught `torch.cuda.OutOfMemoryError`
through broad `RuntimeError` handlers and wrapped it as `UnsupportedConfig`.
The adapters now re-raise CUDA OOM before the generic wrapper, allowing
`timing.measure()` to record `oom`.

Regression test: `test_sdpa_does_not_relabel_oom_as_unsupported`.

### BlockSparse mask geometry refusal

File: `attnbench/backends/block_sparse.py`

BlockSparse now validates mask sequence length and block size before importing
or invoking the optional kernel. A malformed mask previously reached the
extension import; on a machine without the extension this surfaced as an
unrelated `ModuleNotFoundError` rather than a geometry refusal.

Regression test: `test_block_sparse_rejects_mismatched_mask_geometry_before_kernel_import`.

### Finite-positive ratio validation

File: `attnbench/analysis/cross_arch.py`

Shared within-host speedup formation now rejects zero, negative, and non-finite
baseline or backend latency. This also protects canary ratios. Before the fix,
an infinite backend latency produced a `0.0` canary ratio and could suppress a
drift report.

Regression test: `test_canary_rejects_non_finite_backend_latency`.

## Follow-up findings not promoted to bugs

- Peak-memory accounting still includes the first warmup/JIT call because the
  peak counter is reset before warmup. The intended meaning of the field needs
  to be settled before changing it; no GPU experiment was available here.
- BlockSparse still constructs `cu_seqlens`, head masks, and block-mask views
  inside each forward call. The implementation risk is visible, but its
  measured effect requires the target CUDA extension and GPU.
- The score-cache key does not encode model revision or scoring-code version.
  Current startup clears the documented derived cache on every boot, but an
  ordinary reused environment could still make this relevant. No stale-cache
  reproduction was established from current data.
- Cross-backend tolerance remains absolute-only by design; no boundary test
  was found proving whether the omission of `rtol` is intentional.
- The exclusion-rule audit found no remaining duplicated GLA exclusion
  definition in the analysis scripts; they import the canonical
  `ACCURACY_EXCLUDED_BACKENDS` tuple from `attnbench/accuracy/grid_configs.py`.
- A claims-ledger spot check found the major qualifiers recorded in
  `docs/claims.md`, including `decode_backend`, `score_source`, `check_kind`,
  and `clocks_locked`. A complete row-by-row claims-to-producing-code trace
  remains outside this pass.

## Follow-up validation

The complete suite after this pass:

- `850 passed, 2 skipped`

`git diff --check` passed.
