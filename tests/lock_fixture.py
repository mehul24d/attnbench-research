"""A complete, typed lock for tests (T4.1; `attnbench.lock.checklist.SCHEMA`).

Every value has its item's type, so `checklist.check` reads it complete. The
values are test values, not proposals: nothing here is a lock decision.
"""

from __future__ import annotations

import copy

from attnbench.lock import checklist as CL

SHA = "a" * 64
COMMIT = "b" * 40

VALUES = {
    "h0_to_h3_tolerances": {"H0": 0.1, "H1": 0.1, "H2": 0.1, "H3": "bootstrap widening only"},
    "h6_regret_bound": {"mean": 0.05, "worst": 0.25},
    "ground_truth_density_set": {"dense": [1.0], "oracle": [0.25, 0.5], "meanpool": [0.25, 0.5],
                                 "xattention": ["calibrated"],
                                 "xattention_noassert": ["calibrated"]},
    "h4_equivalence_bound_with_its_justification_from_the_tasks": 10.0,
    "h4_confirmatory_family": {"m": 2, "tests": ["oracle vs meanpool @32",
                                                 "random vs oracle @32"]},
    "power_target_and_n": {"alpha": 0.025, "power": 0.9, "n": 400},
    "primary_cell_n_and_admissibility_n": {"primary_cell_n": 200, "admissibility_n": 200},
    "admissibility_holm_family": [{"arm": "oracle", "density": 0.25},
                                  {"arm": "xattention", "density": "calibrated"}],
    "context_grid_checked_against_each_model_s_position_limit": {
        "lengths": [2048, 4096, 8192, 16384, 32768], "batches": {"16384": [1, 4]},
        "position_limit": 32768},
    "delay_injection_slope_bound": [0.707, 1.414],
    "mask_gate_bound": 0.01,
    "cross_card_dense_task_score_bound": 2.0,
    "replay_versus_trace_bound_for_gate_a": 0.25,
    "null_kernel_floor_multiple": 3.0,
    "probe_bounds": {"structure": 0.05},
    "resolution_and_noise_floors": {
        "pairing": "P1", "noise_floor_relative": 0.05,
        "resolution_floor_ms": {"2048x1": 1.39, "4096x1": 7.69, "8192x1": 17.76,
                                "16384x1": 13.77, "16384x4": 27.59, "32768x1": 101.16}},
    "floors_quantile": 0.95,
    "hdbi_boundedness_threshold": 0.5,
    "learned_predictor_features": ["host_s", "device_s"],
    "learned_predictor_hyperparameters": {"target": "seconds", "params": {}},
    "k2_correctness_tolerance": 0.02,
    "head_type_criterion": "T1.3's rule",
    "rerun_budget": {"max_rerun_rounds": 1, "min_clean_rounds": 3},
    "throttle_set": ["hw_slowdown", "sw_thermal_slowdown", "hw_thermal_slowdown",
                     "hw_power_brake_slowdown"],
    "payload_class_edges": [8, 4096, 1048576],
    "prompt_set_hash_and_tokenizer_version": {"prompt_set_sha256": SHA,
                                              "tokenizer_version": "transformers 4.46.0"},
    "realised_length_key_rule": "the realised token count, the budget a label",
    "budget_cut_order": ["P7", "chunked extension", "compiled group", "rehearsal"],
    "compiled_versus_eager_density_flag_tolerance": 0.01,
    "probe_length_rule": "as ruled 31 Oct 2026",
    "flagged_memset_sensitivity_bound": 0.0,
    "chain_check_tolerance": 0.1,
    "recompile_gate": 0,
    "residual_spread_tolerance": 0.1,
    "pricing_mismatch_share": 0.05,
    "block_drift_bound": 0.0677,
    "timed_path_identity_for_p2_and_p4": SHA,
    "bootstrap_settings": {"level": 0.95, "n_boot": 2000, "seed": 0},
    "analysis_commit": COMMIT,
    "marginal_label_s_sensitivity_floors": {"16384x1": 111.05, "16384x4": 441.54},
}

SCORED = {"H0": ["P1", "P2"], "H1": ["P2", "P4"], "H2": ["P2"], "H3": ["P4"],
          "H4": ["P1"], "H5": ["P4"], "H6": ["P2", "P4"]}


def complete_lock(plan_text: str, **extra) -> dict:
    """A lock `checklist.check` reads as complete against `plan_text`."""
    lock = CL.skeleton(plan_text)
    for k, e in lock["values"].items():
        e["value"], e["source"] = copy.deepcopy(VALUES[k]), "test fixture"
    lock["scored_pairings"] = copy.deepcopy(SCORED)
    lock.update(extra)
    return lock
