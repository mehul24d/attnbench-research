# Code inventory for the new direction (task T0.4)

**Status:** first pass, 4 October 2026. Each decision was proposed from the module's docstring, its imports and the plan in `docs/RESEARCH_PLAN.md`. Nothing is moved or deleted until the researcher confirms a row (T0.4) and the pre-pivot tag exists (T0.3).

**Decisions.**

- **Keep.** Used by the new study as it is.
- **Adapt.** The core is reusable, but it needs changes for a plan task (named).
- **Archive.** No role in the new plan. It moves to `legacy/` with its tests (T0.5), so banked results stay reproducible from the tag *and* from the tree.
- **Check.** Unclear from a first read; the code has to be read before deciding.

**Size today:** about 46,000 lines in total, of which `attnbench/` is about 15,000, `scripts/` about 11,000 and `tests/` about 20,000. Many tests check documents, figures and call sites as well as code. Moving a doc or script therefore breaks tests unless the paths are updated in the same commit.

## Harness core (`attnbench/`)

| Path | Role so far | Decision | Reason / plan task |
|---|---|---|---|
| `provenance.py` | Environment capture, clock control, provenance stamps | Adapt | Add host-speed probe fields, NUMA node, PCIe link, provider and instance type (T2.1, T2.5) |
| `timing.py` | Kernel timing (do_bench / CUDA events) | Keep | Device-side timing primitive |
| `numerics.py` | Enforced fp32 matmul precision for scores | Keep | Recall and oracle depend on it |
| `checkpoint.py` | Parquet append helper | Keep | |
| `compile_guard.py` | Refuses torch.compile silent eager fallback | Keep | Integrity guard |
| `build_guards.py` | Resource guards for CUDA source builds | Keep | FA3 build (K2) |
| `config.py` | `AttnConfig` sweep-point object | Keep | Used everywhere |
| `masks.py` | Mask builders (reference, vectorised, device), selector | Keep | Arms and precomputed-mask ablation (T1.7, T2.12) |
| `gates.py` | Capability matrix and correctness gate | Keep (trim later) | A correctness gate is still needed for every arm; the old-direction stages inside it can be trimmed after T0.5 |
| `sweep.py` | Old kernel-microbenchmark sweep driver | Check | Possibly replaced by isolated per-operation replay (T2.4) |
| `backends/base.py`, `impls.py` | Backend interface and dense references | Keep | Dense arm, reference paths |
| `backends/block_sparse.py` | Block-sparse kernel wrapper | Keep | The one sparse kernel for every arm (K1) |
| `backends/xattention.py` | XAttention as shipped | Keep | Calibrated XAttention arm |
| `backends/block_masked.py` | Exact masked attention at any block size, accuracy only | Keep | H4 at block sizes 32 and 64 (T1.4) |
| `backends/linear.py` | Gated linear attention | Archive | Linear attention is out of scope |
| `backends/sage_attention.py`, `backends/xformers_backend.py` | INT8 and xFormers attention | Archive | Not arms in the new study |
| `_vendor/ruler/` | Vendored RULER generators and scorer | Keep | Accuracy tasks |

## Accuracy pipeline (`attnbench/accuracy/`)

| Path | Role so far | Decision | Reason / plan task |
|---|---|---|---|
| `model.py` | Swappable-attention HF model, inline estimator, scoring pass | Adapt | Own-pass recall (T1.2); the new arms (T1.7) |
| `ruler.py`, `ruler_data.py`, `sizing.py`, `stopping.py` | Task generation, token-exact sizing, stopping caps | Keep | |
| `generation.py`, `runner.py`, `schema.py`, `grid_configs.py`, `config.py` | Per-cell execution, resumable driver, row schema, grid | Keep | Schema gains recall columns (T1.2) |
| `score_cache.py` | Cache of dense-pass scores | Keep | Secondary recall reference |
| `dense_canary.py` | Dense-arm reproduction gate | Keep | |
| `t4_pilot.py` | Pilot design as data | Keep | Banked-evidence record; used by H4 dry analysis (T1.5) |
| `phase_timing.py` | Per-phase timing and reconciliation | Adapt | Basis of the ground-truth harness (T2.13) |
| `timing_probe.py` | Pre-flight timing of one example | Keep | Session sizing |
| `batch_scaling.py` | Batch-size memory and throughput probe | Check | Batch 1 and 4 only in the new plan |
| `gla_arm.py` | GLA arm decision | Archive | GLA dropped |

## Analysis (`attnbench/analysis/`)

| Path | Role so far | Decision | Reason / plan task |
|---|---|---|---|
| `exact_noninferiority.py` | Exact paired non-inferiority | Keep | The accuracy test (Section 5.7) |
| `eras.py`, `code_identity.py`, `composition.py` | Refuse comparisons across mask eras, code versions, incomparable cells | Keep | Integrity guards (rule 3) |
| `canary.py` | Per-session drift canary | Keep | Imports `cross_arch` |
| `cross_arch.py` | Joins timing results across GPUs with clock-state checks | Adapt | Becomes the cross-pairing join for H0 and H2 |
| `crossover.py` | Where one backend overtakes another | Adapt | Crossover per estimator and pairing (H3, H5) |
| `matched.py` | Bootstrap matched-accuracy points | Archive | Superseded by `exact_noninferiority.py`; `scripts/run_matched_analysis.py` and several tests depend on it |
| `pareto.py`, `decision.py` | Old Pareto frontier and decision map | Archive | Old research question |
| `decode_confound.py`, `decode_backend_guard.py` | Decode-kernel confound in end-to-end totals | Check | The new study times prefill only; the guard may still protect accuracy rows |
| `diagnostic_agreement.py` | Agreement with the 4 Sep flex diagnostic | Archive | One-off; `tests/test_resolution_floors.py` imports it, so move it with that test or inline the constant |

## Scripts (`scripts/`)

| Path | Decision | Reason / plan task |
|---|---|---|
| `gcp_*.sh` (launch, preflight, deploy, teardown, cleanup, status, image) | Keep | Session rules 10 to 12; template for AWS (T5.1) |
| `run_phase.sh`, `gpu_suite_record.sh`, `procmatch.sh`, `_image_boot_checks.py`, `build_flash_attn.sh`, `install_xattention.sh` | Keep | Session plumbing |
| `run_accuracy.py`, `check_dense_canary.py`, `check_score_canary.py`, `fetch_ruler_data.py`, `measure_answer_lengths.py`, `time_one_accuracy_example.py` | Keep | Accuracy pipeline |
| `calibrate_xattn_thresholds.py`, `run_t4_noninferiority.py` | Keep | T1.1 |
| `run_scorer_comparison.py` | Keep | Oracle vs estimator on the same rows (G3) |
| `host_cpu_probe.py` | Adapt | Host-speed probe (T2.1) |
| `measure_mask_h2d_tax.py` | Adapt | Link microbenchmarks (T2.5) |
| `measure_estimator_cost.py` | Adapt | Host and device estimator components (T2.2, T2.4) |
| `run_vectorised_endtoend.py`, `run_phase_timing.py`, `_vec_mask_for_measurement.py` | Adapt | Ground-truth harness (T2.13); the vectorised builder stays a measurement variant, never pooled with the reference builder |
| `run_probe.py` | Keep | Correctness gate runner |
| `run_sweep.py` | Check | With `sweep.py` |
| `check_score_dtype_asymmetry.py`, `run_s1a_comparison.py`, `run_vt_stopping_analysis.py`, `s1a_run_band.sh`, `s7_run_jitter_band.sh`, `sink_control_run.sh` | Archive | Single audit items, finished |
| `repair_reconciliation_identity.py`, `repair_stage5_stamp.py`, `reestimate_stage3.py` | Archive | One-off repairs and estimates; keep for provenance of banked files |
| `run_cross_arch_analysis.py` | Adapt | With `cross_arch.py` |
| `run_matched_analysis.py`, `run_pareto.py`, `run_decision_map.py`, `run_decode_confound.py`, `run_scale_comparison.py` | Archive | Old research question |
| `decide_gla_arm.py` | Archive | GLA dropped |
| `probe_batch_scaling.py` | Check | With `batch_scaling.py` |

## Configs, docs, results

| Path | Decision | Reason |
|---|---|---|
| `configs/accuracy/stage3_grid.yaml` | Adapt | Becomes the H4 grid |
| `docs/limitations.md`, `claims.md`, `withdrawn_figures.md`, `silent_failure_patterns.md`, `audit_register.md`, `spend_ledger.md` | Keep in place | Audit history (rule 4); tests read them |
| `docs/t4_*.md` | Keep in place | Pre-registrations of banked and pending runs (T1.1) |
| `docs/stage2_*`, `stage3_*`, `a100_session_plan.md`, `h100_overlap_preregistration.md`, `gla_arm_decision.md`, `copilot_review_plan.md`, `code_review.md`, `writeup_input.md`, `retry_session_runbook.md`, `machine_image_family_lock.md`, `hardware_constraints.md` | Archive to `docs/archive/` unchanged | Old-direction plans; check test references first (T0.6) |
| `results/` | Keep | Banked, commit-stamped evidence |

## Not in the repository yet (new code the plan needs)

Null-kernel host profiler (T2.2), synchronisation map (T2.3), isolated device replay (T2.4), launch-queue probe (T2.6), trace checks (T2.7), the recursion and baselines (T2.8 to T2.10), prediction commit (T2.11), precomputed-mask arm (T2.12), container image (T2.14), AWS tooling (T5.1). Suggested home: a new package `attnbench/predict/` for the model and baselines and `attnbench/components/` for the profilers, so the new study's code is easy to find.
