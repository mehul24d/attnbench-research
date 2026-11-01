"""T4.1's lock list, read from the plan, against a lock file."""

from __future__ import annotations

import json

import pytest

from attnbench.lock import checklist as L

PLAN_TEXT = L.PLAN.read_text(encoding="utf-8")


def _filled(plan_text=PLAN_TEXT) -> dict:
    lock = L.skeleton(plan_text)
    for k, e in lock["values"].items():
        e["value"] = f"value of {k}"
        e["source"] = "Appendix A, test"
    lock["scored_pairings"] = {"H0": ["P1", "P2"], "H2": ["P2", "P4"]}
    return lock


def test_the_list_is_read_from_the_plans_t41_row():
    items = L.lock_items(PLAN_TEXT)
    assert len(items) == 33
    keys = list(items)
    assert keys[0] == "h0_to_h3_tolerances"
    # three of the researcher's rulings of 31 Oct 2026, the chain check of
    # 1 Nov 2026 and the recompile gate after Shakedown 5 close the list
    assert keys[-6:] == ["budget_cut_order", "compiled_versus_eager_density_flag_tolerance",
                         "probe_length_rule", "flagged_memset_sensitivity_bound",
                         "chain_check_tolerance", "recompile_gate"]
    # Parentheticals are part of the item's text but not of its key, and a
    # comma inside one does not split the item.
    assert items["replay_versus_trace_bound_for_gate_a"].startswith(
        "replay-versus-trace bound for Gate A (set from")
    assert "Appendix A, 27 Oct 2026)" in items["replay_versus_trace_bound_for_gate_a"]
    assert "hdbi_boundedness_threshold" in items
    assert "resolution_and_noise_floors" in items


def test_a_complete_lock_file_passes_and_its_report_says_so():
    result = L.check(_filled(), PLAN_TEXT)
    assert result.complete
    assert result.report().startswith("33 values listed by T4.1; complete")


def test_missing_unset_unsourced_and_unknown_entries_are_each_named():
    lock = _filled()
    del lock["values"]["h6_regret_bound"]
    lock["values"]["mask_gate_bound"]["value"] = None
    lock["values"]["probe_bounds"]["value"] = []
    lock["values"]["budget_cut_order"]["source"] = ""
    lock["values"]["a_value_t41_does_not_list"] = {"value": 1, "source": "x"}
    r = L.check(lock, PLAN_TEXT)
    assert not r.complete
    assert r.missing == ("h6_regret_bound",)
    assert set(r.unset) == {"mask_gate_bound", "probe_bounds"}
    assert r.unsourced == ("budget_cut_order",)
    assert r.unknown == ("a_value_t41_does_not_list",)
    text = r.report()
    for k in ("h6_regret_bound", "mask_gate_bound", "budget_cut_order"):
        assert k in text


def test_a_lock_file_written_against_another_plan_is_not_complete():
    lock = _filled()
    other = PLAN_TEXT + "\n- **A later decision.** Something changed.\n"
    r = L.check(lock, other)
    assert not r.plan_matches and not r.complete
    assert "another version of the plan" in r.report()


def test_a_changed_t41_row_changes_the_checklist():
    row = next(l for l in PLAN_TEXT.splitlines() if l.startswith(L.ROW_PREFIX))
    grown = PLAN_TEXT.replace(row, row.replace("budget cut order",
                                               "budget cut order, new bound"))
    r = L.check(_filled(), grown)
    assert r.missing == ("new_bound",)


@pytest.mark.parametrize("mutate, match", [
    (lambda t: t.replace(L.ROW_PREFIX, "| T4.x |"), "0 T4.1 rows"),
    (lambda t: t.replace(L.LEAD, "Fix the values:"), "no longer starts"),
])
def test_a_plan_whose_t41_row_cannot_be_read_is_refused(mutate, match):
    with pytest.raises(L.LockListError, match=match):
        L.lock_items(mutate(PLAN_TEXT))


def test_a_lock_file_without_values_or_with_a_malformed_entry_is_refused():
    with pytest.raises(L.LockListError, match="values"):
        L.check({}, PLAN_TEXT)
    lock = _filled()
    lock["values"]["h6_regret_bound"] = 3.0
    with pytest.raises(L.LockListError, match="h6_regret_bound"):
        L.check(lock, PLAN_TEXT)


def test_the_plans_other_lock_sentences_are_listed_for_a_reader():
    mentions = L.lock_mentions(PLAN_TEXT)
    text = " ".join(s for _, s in mentions)
    # Values the plan fixes at lock outside T4.1's list are visible here.
    assert "features and hyperparameters fixed at lock" in text
    assert "Both bounds fixed at lock from P1 repeat sessions" in text
    assert all(not s.startswith(L.ROW_PREFIX) for _, s in mentions)


def test_check_file_reads_json(tmp_path):
    p = tmp_path / "lock.json"
    p.write_text(json.dumps(_filled()))
    assert L.check_file(p).complete
    p.write_text("{not json")
    with pytest.raises(L.LockListError, match="not JSON"):
        L.check_file(p)


# --- every "at lock" phrase against T4.1 (researcher's decision, 28 Oct 2026) ---------
#
# Each sentence of the plan that says "at lock" or carries [lock] is mapped to
# the T4.1 item(s) that fix its value, by a phrase from the sentence (matched
# without case). A new sentence the map does not cover fails the first test;
# a value the map needs and T4.1 does not name fails the second.

THE_LIST = "the lock list itself"     # Gate B's reference to all [lock] values

AT_LOCK = [
    ("each has a statistic and a tolerance fixed at lock", ("h0_to_h3_tolerances",)),
    ("per-operation differences within **[lock]**", ("h0_to_h3_tolerances",)),
    ("absolute relative error within **[lock]**", ("h0_to_h3_tolerances",)),
    ("predicted and measured differences agree within **[lock]**", ("h0_to_h3_tolerances",)),
    ("against an equivalence bound **[lock]**",
     ("h4_equivalence_bound_with_its_justification_from_the_tasks",)),
    ("h4 equivalence bound:** justified at lock",
     ("h4_equivalence_bound_with_its_justification_from_the_tasks",)),
    ("h4 bound justified at lock",
     ("h4_equivalence_bound_with_its_justification_from_the_tasks",)),
    ("mean and worst case per pairing against **[lock]**", ("h6_regret_bound",)),
    ("density (a set fixed at lock", ("ground_truth_density_set",)),
    ("each cell is checked at lock against the model's position limit",
     ("context_grid_checked_against_each_model_s_position_limit",)),
    ("features and hyperparameters fixed at lock",
     ("learned_predictor_features", "learned_predictor_hyperparameters")),
    ("its boundedness is one threshold", ("hdbi_boundedness_threshold",)),
    ("hdbi's boundedness thresholds are fitted on the training pairings and recorded at lock",
     ("hdbi_boundedness_threshold",)),
    ("within a tolerance fixed at lock", ("k2_correctness_tolerance",)),
    ("the gate's `tol` and is not yet fixed at lock", ("k2_correctness_tolerance",)),
    ("both bounds fixed at lock from p1 repeat sessions", ("probe_bounds",)),
    ("the session and transfer bounds are set at lock", ("probe_bounds",)),
    ("the resolution floor and the noise floor are fixed at lock",
     ("resolution_and_noise_floors", "floors_quantile")),
    ("the resolution floor is the researcher's at lock", ("resolution_and_noise_floors",)),
    ("admissibility n is fixed at lock", ("primary_cell_n_and_admissibility_n",)),
    ("its holm family is declared at lock", ("admissibility_holm_family",)),
    ("its holm family declared at lock", ("admissibility_holm_family",)),
    ("once its family is declared at lock", ("admissibility_holm_family",)),
    ("criterion fixed at lock", ("head_type_criterion",)),
    ("h4 power rule (fixed at lock)", ("power_target_and_n",)),
    ("the family size, and so n, is fixed at lock", ("h4_confirmatory_family", "power_target_and_n")),
    ("task-score bound fixed at lock", ("cross_card_dense_task_score_bound",)),
    ("replay-versus-trace difference per operation family is within a bound fixed at lock",
     ("replay_versus_trace_bound_for_gate_a",)),
    ("the replay-versus-trace bound (researcher's decision) is set at lock",
     ("replay_versus_trace_bound_for_gate_a",)),
    ("within a bound of 1.0 fixed at lock", ("delay_injection_slope_bound",)),
    ("one pooled slope per mechanism against a bound set at lock", ("delay_injection_slope_bound",)),
    ("mask agreement and density difference against p1's recorded masks are within a bound",
     ("mask_gate_bound",)),
    ("within a fixed multiple (set at lock)", ("null_kernel_floor_multiple",)),
    ("all **[lock]** values fixed", (THE_LIST,)),
    ("cut in an order fixed at lock", ("budget_cut_order",)),
]

# Lock values the researcher named on 28 Oct 2026 that no "at lock" sentence
# carries in those words: the rerun budget for flagged rounds (T2.13) and the
# throttle set (Appendix A, 27 Oct 2026); the others are already mapped above.
NAMED_ALSO = ("replay_versus_trace_bound_for_gate_a", "hdbi_boundedness_threshold",
              "floors_quantile", "admissibility_holm_family", "rerun_budget", "throttle_set")


def _keys_for(sentence: str) -> set:
    s = sentence.lower()
    return {k for phrase, keys in AT_LOCK if phrase in s for k in keys}


def test_every_at_lock_sentence_in_the_plan_maps_to_a_lock_value():
    mentions = L.lock_mentions(PLAN_TEXT)
    unmapped = [(n, s[:160]) for n, s in mentions if not _keys_for(s)]
    assert not unmapped, ("sentences that fix a value at lock with no T4.1 item in the "
                          f"map: {unmapped}")
    # And no map entry is stale.
    texts = [s.lower() for _, s in mentions]
    stale = [p for p, _ in AT_LOCK if not any(p in t for t in texts)]
    assert not stale, f"map phrases no longer in the plan: {stale}"


def test_the_mapping_catches_a_new_at_lock_sentence():
    grown = PLAN_TEXT + "\nThe warm-up count is fixed at lock.\n"
    assert any(not _keys_for(s) for _, s in L.lock_mentions(grown))


def test_t41_names_every_value_the_plan_fixes_at_lock():
    needed = {k for _, keys in AT_LOCK for k in keys} - {THE_LIST} | set(NAMED_ALSO)
    missing = sorted(needed - set(L.lock_items(PLAN_TEXT)))
    assert not missing, f"T4.1 does not name: {missing}"
