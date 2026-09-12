# Code Review — 2026-09-12

Scope: the files and functions listed under **Not yet reviewed**. This is a
documentation-only review. No source, script, test, or configuration file was
modified.

## Coverage map

The prior review record is recovered from commit `7c1fcca` because
`docs/verification_review_2026-09-10.md` is not present in the current
checkout. `docs/copilot_review_plan.md` and the prior record were used only to
exclude explicitly covered functions. A partial review of a file does not
exclude its other functions.

### Already reviewed elsewhere

- `attnbench/provenance.py`: `_sh`, `_git_state`, `capture` command-status
	behavior, and provenance stamping; prior record categories 1 and 9.
- `attnbench/analysis/decode_confound.py`: `correct` decode-step correction;
	prior record category 14.
- `attnbench/analysis/cross_arch.py`: `Speedup.speedup`,
	`speedup_within_host` ratio formation, and finite-positive latency handling;
	prior record category 2.
- `attnbench/analysis/composition.py`: guarded aggregation and
	`matched_subset`; prior record checks with no finding.
- `attnbench/analysis/matched.py`: paired bootstrap minimum-n and example-ID
	alignment; prior record checks with no finding.
- `attnbench/analysis/crossover.py`: shared-cell comparison; prior record
	checks with no finding.
- `attnbench/analysis/pareto.py`: Pareto arithmetic and dense-reference
	handling; prior record checks with no finding.
- `attnbench/analysis/canary.py`: canary ratio path as reached through
	`speedup_within_host`; finite-positive shared ratio fix was covered in the
	follow-up pass.
- `attnbench/analysis/decode_backend_guard.py`: referenced during prior
	Pareto/phase aggregation review.
- `attnbench/analysis/diagnostic_agreement.py`: `_rel` denominator floor was
	reviewed under category 2.
- `attnbench/accuracy/phase_timing.py`: reconciliation identity,
	`bias_warning`, and decode-fit discussion; prior record categories 3, 13,
	and 14. Other functions remain in scope below.
- `attnbench/accuracy/score_cache.py`: cache key and startup-clearing risk
	were reviewed read-only; no confirmed finding. This file is excluded from
	duplicate review, but no new cache conclusion is claimed here.
- `attnbench/backends/impls.py`: Flex mask cache, SDPA OOM handling, and Flex
	geometry/cache path; prior follow-up findings.
- `attnbench/backends/block_sparse.py`: mask geometry and OOM handling;
	prior follow-up findings.
- `attnbench/backends/linear.py`, `sage_attention.py`, and
	`xformers_backend.py`: broad OOM-wrapper behavior; prior follow-up finding.
- `attnbench/gates.py`: exact/cross-backend output validation, device memory
	selection, tolerance path, and compile-fallback dispatch; prior follow-up
	findings.
- `attnbench/compile_guard.py`: sticky fallback detection and callers; prior
	record checks with no finding.
- `scripts/run_pareto.py`: `latency_table`; prior record category 11.
- `scripts/run_phase_timing.py`: observed aggregation; prior record category
	11.
- `tests/test_verification_review.py`: all tests added by the prior review;
	not re-reviewed for new findings.

### Not yet reviewed — actual scope for this pass

#### Core and configuration

- `attnbench/__init__.py`
- `attnbench/build_guards.py`
- `attnbench/checkpoint.py`
- `attnbench/config.py`
- `attnbench/sweep.py`
- `attnbench/timing.py`
- Remaining uncovered functions in `provenance.py`, `gates.py`,
	`compile_guard.py`, `backends/impls.py`, `backends/block_sparse.py`,
	`backends/linear.py`, `backends/sage_attention.py`,
	`backends/xformers_backend.py`, and `accuracy/phase_timing.py`.

#### Accuracy and vendor code

- `attnbench/accuracy/__init__.py`
- `attnbench/accuracy/batch_scaling.py`
- `attnbench/accuracy/config.py`
- `attnbench/accuracy/generation.py`
- `attnbench/accuracy/gla_arm.py`
- `attnbench/accuracy/grid_configs.py`
- `attnbench/accuracy/model.py`
- `attnbench/accuracy/ruler.py`
- `attnbench/accuracy/runner.py`
- `attnbench/accuracy/schema.py`
- `attnbench/accuracy/sizing.py`
- `attnbench/accuracy/stopping.py`
- `attnbench/accuracy/timing_probe.py`
- `attnbench/_vendor/ruler/__init__.py`
- `attnbench/_vendor/ruler/niah.py`
- `attnbench/_vendor/ruler/scoring.py`
- `attnbench/_vendor/ruler/variable_tracking.py`
- `attnbench/_vendor/ruler/VENDORED.md`
- `attnbench/backends/base.py` and `attnbench/backends/__init__.py`

#### Analysis code outside prior function coverage

- `attnbench/analysis/__init__.py`
- `attnbench/analysis/code_identity.py`
- `attnbench/analysis/decision.py`
- Remaining uncovered functions in `canary.py`, `cross_arch.py`,
	`diagnostic_agreement.py`, and `decode_backend_guard.py`.

#### Scripts

- `scripts/build_flash_attn.sh`
- `scripts/decide_gla_arm.py`
- `scripts/gcp_cleanup_check.sh`
- `scripts/gcp_deploy_source.sh`
- `scripts/gcp_launch_compile_session.sh`
- `scripts/gcp_launch_l4.sh`
- `scripts/gcp_status.sh`
- `scripts/gcp_teardown_session.sh`
- `scripts/probe_batch_scaling.py`
- `scripts/procmatch.sh`
- `scripts/reestimate_stage3.py`
- `scripts/repair_reconciliation_identity.py`
- `scripts/repair_stage5_stamp.py`
- `scripts/run_accuracy.py`
- `scripts/run_cross_arch_analysis.py`
- `scripts/run_decision_map.py`
- `scripts/run_decode_confound.py`
- `scripts/run_matched_analysis.py`
- `scripts/run_probe.py`
- `scripts/run_sweep.py`
- `scripts/time_one_accuracy_example.py`
- Remaining uncovered functions in `scripts/run_pareto.py` and
	`scripts/run_phase_timing.py`.

#### Tests

- All tracked tests other than `tests/test_verification_review.py` are in
	scope for a review of vacuity, stale case lists, and whether critical guards
	can be deliberately broken. Existing test coverage is evidence only when
	the test exercises the production path.

## Findings

### Finding 1: Bounded sizing hangs when `max_units=0`

File: `attnbench/accuracy/sizing.py:138-149`

Category: uncategorized, bounded-search termination failure

Severity: Medium

Evidence tier: **CONFIRMED BY TEST**

What the code does: `fit_units_to_budget()` accepts `max_units=0`. It clamps
`hi` to zero, then enters `while count_tokens(render(hi)) <= budget`. When zero
units fit, `lo` and `hi` remain zero and the loop doubles zero forever.

What's wrong: a caller using a valid-looking zero upper bound does not receive
a `ValueError`, a bounded result, or a clear budget error. It hangs instead.
The isolated scratch call was:

```python
fit_units_to_budget(lambda n: n, lambda x: x, 1, max_units=0)
```

The subprocess timed out after one second without returning. This was run from
a scratch Python process and no repository file was changed.

Downstream impact: any future accuracy task or dry-run caller that supplies a
zero maximum can stall the sizing stage before examples are generated. The
current production callers do not appear to pass `max_units=0`, so no existing
result artifact was shown to be affected.

claims.md exposure: none identified in current results; this is a reachable
input-boundary failure that could block or prevent a future accuracy run.

Prose-only remediation direction: reject an upper bound below `min_units`, or
handle the zero-bound case explicitly before the doubling loop.

### Finding 2: NaN GLA scores bypass the decision gates

File: `attnbench/accuracy/gla_arm.py:48-58, 83-118`

Category: 2 and 8, non-finite derived metric and gate-vacuity failure

Severity: High

Evidence tier: **CONFIRMED BY TEST**

What the code does: `_measure()` computes `sum(scores) / n` without checking
that each score is finite. `evaluate()` then tests `score < MIN_MEAN_SCORE`.
For `score = NaN`, that comparison is false, so the score gate is skipped.

What's wrong: a run with ten distinct, answer-shaped predictions and ten NaN
scores returned `ArmVerdict(verdict="keep", mean_score=nan,
failed_gate=None)`. The scratch input used predictions `1000000` through
`1000009`, control scores all `100.0`, and GLA scores all `float("nan")`.
The distinct and format fractions were both `1.0`, but the non-finite score
was accepted as a successful arm.

Downstream impact: `scripts/decide_gla_arm.py` consumes this verdict to decide
whether the GLA arm remains in the accuracy path. A NaN-producing scoring or
serialization failure could therefore produce a `keep` decision rather than a
refusal, and later accuracy rows could be treated as meaningful.

claims.md exposure: the GLA section's supported substitution claim and the
accuracy-arm exclusion decision in `docs/gla_arm_decision.md`; a false `keep`
could license rows that should not support either conclusion.

Prose-only remediation direction: require finite prediction scores before
computing the mean or make non-finite aggregate values an explicit
`no_verdict`/failure state.

### Finding 3: Unknown accuracy backends default to the dense-reference role

File: `attnbench/accuracy/runner.py:28-45`

Category: 10, duplicated/implicit role rule with an unsafe default

Severity: Medium

Evidence tier: **CONFIRMED BY CALCULATION**

What the code does: `backend_role()` recognizes `gla`, `sage`, and
`block_sparse`, then returns `"dense_reference"` for every other backend name
when the mask is not `block_sparse`.

What's wrong: with a causal config and backend name `"typo_backend"`, the
function returned `"dense_reference"`. The unknown name was not refused and
was not marked as an unclassified backend. If a new or misspelled backend
reaches the accuracy runner and its generator accepts the name, its rows carry
the dense-reference role despite not being the configured dense arm.

Downstream impact: `run_accuracy()` writes the returned role into every
`AccuracyResult`; downstream grouping and dense-reference selection can then
include the wrong backend as the reference. The error is especially dangerous
because the row remains schema-valid and no numerical exception is required.

claims.md exposure: potentially every accuracy row that depends on the dense
baseline, including the supported oracle-accuracy rows and Stage 6/7 matched
latency claims. No current artifact with an unknown backend was found.

Prose-only remediation direction: make the known backend-role mapping closed
and refuse unknown non-sparse names, or derive the role from a canonical
backend registry rather than a dense fallback.

### Finding 4: Cross-architecture environment validation is report-only

File: `scripts/run_cross_arch_analysis.py:91-108, main()`

Category: 9 and 15, provenance recorded but not enforced

Severity: High

Evidence tier: **READ-ONLY, UNVERIFIED**

What the code does: `environment_report()` returns a boolean indicating
whether `driver`, `torch`, `torch_cuda`, and `triton` are uniform. Missing
columns are skipped and leave `uniform=True`. In `main()`, a non-uniform result
only prints `"[NOT UNIFORM]"` and a warning; the code then proceeds to form
speedups and write comparison artifacts.

What's wrong: two concrete scratch frames showed the boundary. A frame with
driver values `1` and `2` returned `uniform=False`, but the main control flow
does not refuse it. A frame with all environment columns absent returned
`uniform=True` with an empty report. Thus the environment fields can be absent
or contradictory while the cross-architecture analysis still proceeds.

Downstream impact: `speedup_within_host()`, `compare_across_architectures()`,
`crossover_table()`, and the parquet outputs under `results/cross_arch/` can
be produced across compiled-kernel environments that the module docstring
itself says must be common. The current stored artifact was not recomputed
from a deliberately mismatched environment, so the live result impact is not
established.

claims.md exposure: the supported cross-architecture rows, especially the
claim that backend ranking is stable across sm_80 and sm_89, and the
hardware-conditional crossover claims.

Prose-only remediation direction: treat missing required environment columns
and non-uniform values as a refusal before ratio formation, rather than a
printed caveat after loading.

## Files reviewed with no findings

The following uncovered files were read for this pass and produced no
additional evidence-backed finding. Partial files listed in the coverage map
were reviewed only outside the previously excluded functions.

### Core and configuration

- `attnbench/__init__.py`
- `attnbench/build_guards.py`
- `attnbench/checkpoint.py`
- `attnbench/config.py`
- `attnbench/sweep.py`
- `attnbench/timing.py`
- Remaining uncovered portions of `attnbench/provenance.py`, `gates.py`,
	`compile_guard.py`, and backend implementation files.

### Accuracy and vendor code

- `attnbench/accuracy/__init__.py`
- `attnbench/accuracy/batch_scaling.py`
- `attnbench/accuracy/config.py`
- `attnbench/accuracy/generation.py`
- `attnbench/accuracy/grid_configs.py`
- `attnbench/accuracy/model.py`
- `attnbench/accuracy/ruler.py`
- `attnbench/accuracy/runner.py` outside `backend_role()`
- `attnbench/accuracy/schema.py`
- `attnbench/accuracy/sizing.py` outside `fit_units_to_budget()`
- `attnbench/accuracy/stopping.py`
- `attnbench/accuracy/timing_probe.py`
- `attnbench/_vendor/ruler/__init__.py`
- `attnbench/_vendor/ruler/niah.py`
- `attnbench/_vendor/ruler/scoring.py`
- `attnbench/_vendor/ruler/variable_tracking.py`
- `attnbench/_vendor/ruler/VENDORED.md`
- `attnbench/backends/base.py`
- `attnbench/backends/__init__.py`

### Analysis

- `attnbench/analysis/__init__.py`
- `attnbench/analysis/code_identity.py`
- `attnbench/analysis/decision.py`
- Remaining uncovered portions of `canary.py`, `cross_arch.py`,
	`decode_backend_guard.py`, and `diagnostic_agreement.py`.

### Scripts

- `scripts/build_flash_attn.sh`
- `scripts/decide_gla_arm.py` outside its call into `gla_arm.evaluate`
- `scripts/gcp_cleanup_check.sh`
- `scripts/gcp_deploy_source.sh`
- `scripts/gcp_launch_compile_session.sh`
- `scripts/gcp_launch_l4.sh`
- `scripts/gcp_status.sh`
- `scripts/gcp_teardown_session.sh`
- `scripts/probe_batch_scaling.py`
- `scripts/procmatch.sh`
- `scripts/reestimate_stage3.py`
- `scripts/repair_reconciliation_identity.py`
- `scripts/repair_stage5_stamp.py`
- `scripts/run_accuracy.py`
- `scripts/run_cross_arch_analysis.py` outside `environment_report()` and
	its enforcement path
- `scripts/run_decision_map.py`
- `scripts/run_decode_confound.py`
- `scripts/run_matched_analysis.py`
- `scripts/run_probe.py`
- `scripts/run_sweep.py`
- `scripts/time_one_accuracy_example.py`
- Remaining uncovered portions of `scripts/run_pareto.py` and
	`scripts/run_phase_timing.py`.

### Tests

- `tests/conftest.py`
- `tests/test_accuracy_config.py`
- `tests/test_accuracy_schema.py`
- `tests/test_batch_scaling.py`
- `tests/test_build_guards.py`
- `tests/test_canary.py`
- `tests/test_claims_support_call_sites.py`
- `tests/test_cleanup_check.py`
- `tests/test_clock_state_at_analysis.py`
- `tests/test_code_identity.py`
- `tests/test_compile_guard.py`
- `tests/test_composition_guard.py`
- `tests/test_config_hash.py`
- `tests/test_correctness_families.py`
- `tests/test_cross_arch.py`
- `tests/test_crossover.py`
- `tests/test_decision.py`
- `tests/test_decode_backend_guard.py`
- `tests/test_decode_confound.py`
- `tests/test_decode_state.py`
- `tests/test_deploy_source_guards.py`
- `tests/test_device_aware_budgets.py`
- `tests/test_device_fault_exclusion.py`
- `tests/test_diagnostic_agreement.py`
- `tests/test_first_call_cost.py`
- `tests/test_flex_block_sparse.py`
- `tests/test_flex_compile_options.py`
- `tests/test_flops.py`
- `tests/test_gate_source_on_rows.py`
- `tests/test_generation_wiring.py`
- `tests/test_generation.py`
- `tests/test_gla_arm_decision.py`
- `tests/test_gqa_agreement.py`
- `tests/test_grid_configs.py`
- `tests/test_launch_script_zone_retry.py`
- `tests/test_launch_startup_script.py`
- `tests/test_masks_determinism.py`
- `tests/test_matched.py`
- `tests/test_no_test_reaches_production.py`
- `tests/test_output_depends_on_input.py`
- `tests/test_pareto.py`
- `tests/test_phase_timing_interfaces.py`
- `tests/test_phase_timing.py`
- `tests/test_probe_bands_and_resume.py`
- `tests/test_procmatch.py`
- `tests/test_provenance_consulted.py`
- `tests/test_reference_claims.py`
- `tests/test_reference_device_limits.py`
- `tests/test_resolution_floors.py`
- `tests/test_ruler_integration.py`
- `tests/test_script_call_sites.py`
- `tests/test_shared_project_rules.py`
- `tests/test_sizing.py`
- `tests/test_stage1_gate_provenance.py`
- `tests/test_stage1_memory_lifetime.py`
- `tests/test_stopping.py`
- `tests/test_sweep_planning.py`
- `tests/test_sweep_resume.py`
- `tests/test_teardown_age_manifest.py`
- `tests/test_timed_region_setup.py`
- `tests/test_timing_probe.py`

## Validation

The existing repository test suite was run unchanged: `856 passed, 2 skipped`.
No tracked source, script, test, or configuration file was modified. Only
`docs/code_review.md` was created/edited for this pass.

