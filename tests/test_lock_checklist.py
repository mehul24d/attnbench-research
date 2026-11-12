"""T4.1's lock list, read from the plan, against a lock file."""

from __future__ import annotations

import json

import sys
from pathlib import Path

import pytest

from attnbench.lock import checklist as L

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lock_fixture as LF  # noqa: E402

PLAN_TEXT = L.PLAN.read_text(encoding="utf-8")


def _filled(plan_text=PLAN_TEXT) -> dict:
    """A typed, complete lock (ruling 1b, 11 Nov 2026)."""
    lock = LF.complete_lock(PLAN_TEXT)
    lock["plan_sha256"] = L.plan_sha256(plan_text)
    return lock


def test_the_list_is_read_from_the_plans_t41_row():
    items = L.lock_items(PLAN_TEXT)
    assert len(items) == 40
    keys = list(items)
    assert keys[0] == "h0_to_h3_tolerances"
    # three of the researcher's rulings of 31 Oct 2026, the chain check of
    # 1 Nov 2026, the recompile gate after Shakedown 5 and the residual's
    # spread after Shakedown 6, the pricing gate of 6 Nov 2026 and the block
    # drift bound of 7 Nov 2026, and the timed-path identity of 10 Nov 2026
    # close the list
    # and T4.3's bootstrap settings, analysis commit and the marginal label's
    # sensitivity floors (11 Nov 2026)
    assert keys[-13:] == ["budget_cut_order", "compiled_versus_eager_density_flag_tolerance",
                         "probe_length_rule", "flagged_memset_sensitivity_bound",
                         "chain_check_tolerance", "recompile_gate",
                         "residual_spread_tolerance", "pricing_mismatch_share",
                         "block_drift_bound", "timed_path_identity_for_p2_and_p4",
                         "bootstrap_settings", "analysis_commit",
                         "marginal_label_s_sensitivity_floors"]
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
    assert result.report().startswith("40 values listed by T4.1; complete")


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


@pytest.mark.parametrize("key, bad", [
    ("mask_gate_bound", "TBD"), ("mask_gate_bound", False), ("mask_gate_bound", "0.01"),
    ("mask_gate_bound", float("nan")), ("mask_gate_bound", -0.01),
    ("rerun_budget", {"max_rerun_rounds": 1}),                         # a key missing
    ("rerun_budget", {"max_rerun_rounds": 1, "min_clean_rounds": None}),
    ("rerun_budget", {"max_rerun_rounds": 1.5, "min_clean_rounds": 3}),
    ("resolution_and_noise_floors", {"pairing": "P2", "noise_floor_relative": 0.05,
                                     "resolution_floor_ms": {"2048x1": 1.0}}),
    ("resolution_and_noise_floors", {"pairing": "P1", "noise_floor_relative": 0.05,
                                     "resolution_floor_ms": {"2048": None}}),
    ("throttle_set", [None]), ("throttle_set", ["owed"]), ("delay_injection_slope_bound", [2, 1]),
    ("h4_confirmatory_family", {"m": 18, "tests": ["a", "b"]}),
    ("timed_path_identity_for_p2_and_p4", "8716d7db"),
    ("ground_truth_density_set", {"oracle": [0.25, 0.25]}),
    ("bootstrap_settings", {"level": 0.95, "n_boot": 1, "seed": 0}),
])
def test_a_value_of_the_wrong_type_is_not_fixed(key, bad):
    """Ruling 1b (11 Nov 2026): typed per key; T4.3 read B, finding 4."""
    lock = _filled()
    lock["values"][key]["value"] = bad
    r = L.check(lock, PLAN_TEXT)
    assert not r.complete and {k for k, _ in r.mistyped} == {key}
    assert "mistyped" in r.report()


def test_a_placeholder_source_is_unsourced():
    lock = _filled()
    lock["values"]["mask_gate_bound"]["source"] = "TBD"
    assert L.check(lock, PLAN_TEXT).unsourced == ("mask_gate_bound",)


def test_every_t41_item_has_a_type():
    assert set(L.lock_items(PLAN_TEXT)) <= set(L.SCHEMA)
    assert L.type_problems("an_item_with_no_type", 1) != []


def test_every_hypothesis_states_its_pairings_and_p1_is_never_a_test_of_h1():
    assert L.hypotheses(PLAN_TEXT) == ("H0", "H1", "H2", "H3", "H4", "H5", "H6")
    lock = _filled()
    del lock["scored_pairings"]["H6"]
    assert "H6 is not stated" in L.check(lock, PLAN_TEXT).scored_problems
    for h in ("H1", "H3", "H6"):
        lock = _filled()
        lock["scored_pairings"][h] = ["P1", "P2"]
        assert any("development pairing" in p for p in L.check(lock, PLAN_TEXT).scored_problems)
    lock = _filled()
    lock["scored_pairings"]["H0"] = ["P1", "P2"]           # H0 compares P1 with P2
    assert L.check(lock, PLAN_TEXT).complete


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
    ("p2's and p4's values are fixed at lock (t4.1)", (THE_LIST,)),
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


def test_the_draft_lock_package_lists_every_item_and_fixes_none():
    """docs/lock_package_draft.json (8 Nov 2026) holds proposals only: every
    T4.1 item present, every value null, no pairing value set, so it can
    never be read as a lock."""
    path = L.PLAN.parent / "lock_package_draft.json"
    draft = json.loads(path.read_text(encoding="utf-8"))
    result = L.check(draft, PLAN_TEXT)
    assert not result.missing and not result.unknown
    assert set(result.unset) == set(result.items) and not result.complete
    assert all(e["proposed"] and e["status"] for e in draft["values"].values())
    assert all(v is None for vals in draft["pairing_values"].values() for v in vals.values())


# --- the staged lock (adviser's ruling 4, fourth set of 12 Nov 2026) --------------

DEFER = {"rule": "the 0.95 quantile of the dispatcher's regret over P2's banked cells",
         "input": "predictions/P2/ledger/banked.json", "fixed_by": "the second lock commit, "
         "before P4's prediction"}


def _deferring(key, deferred=DEFER, value=None):
    lock = LF.complete_lock(PLAN_TEXT)
    lock["values"][key].update(value=value,
                               deferred=dict(deferred) if isinstance(deferred, dict) else deferred)
    return lock


def test_a_deferred_value_with_its_rule_written_completes_the_first_lock_only():
    lock = _deferring("h6_regret_bound")
    r = L.check(lock, PLAN_TEXT)
    assert r.complete and r.deferred == ("h6_regret_bound",) and not r.unset
    assert "deferred to the second lock: h6_regret_bound" in r.report()
    second = L.check(lock, PLAN_TEXT, allow_deferred=False)
    assert not second.complete and second.unset == ("h6_regret_bound",)
    assert L.DEFERRABLE == {"h6_regret_bound", "power_target_and_n",
                            "primary_cell_n_and_admissibility_n", "admissibility_holm_family",
                            "cross_card_dense_task_score_bound"}


@pytest.mark.parametrize("key", ["head_type_criterion", "rerun_budget", "throttle_set"])
def test_a_value_a_pairing_reads_or_a_rule_itself_is_never_deferred(key):
    r = L.check(_deferring(key), PLAN_TEXT)
    assert not r.complete and r.deferral_problems[0][0] == key
    assert "not deferrable" in r.deferral_problems[0][1]


@pytest.mark.parametrize("bad,why", [
    ({**DEFER, "rule": "TBD"}, "rule is not written"),
    ({**DEFER, "rule": "  "}, "rule is not written"),
    ({k: v for k, v in DEFER.items() if k != "input"}, "input is not written"),
    ({**DEFER, "fixed_by": "later"}, "names no second lock"),
    ("the rule", "is not {rule, input, fixed_by}")])
def test_a_deferral_without_its_rule_input_and_second_lock_is_refused(bad, why):
    r = L.check(_deferring("power_target_and_n", deferred=bad), PLAN_TEXT)
    assert not r.complete and any(why in w for _, w in r.deferral_problems)
    assert not r.deferred


def test_a_deferral_beside_a_value_is_refused():
    r = L.check(_deferring("h6_regret_bound", value={"mean": 0.1, "worst": 0.2}), PLAN_TEXT)
    assert not r.complete and "both a value and a deferral" in r.deferral_problems[0][1]
