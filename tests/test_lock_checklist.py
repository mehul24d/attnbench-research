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
    return lock


def test_the_list_is_read_from_the_plans_t41_row():
    items = L.lock_items(PLAN_TEXT)
    assert len(items) == 18
    keys = list(items)
    assert keys[0] == "h0_to_h3_tolerances"
    assert keys[-1] == "budget_cut_order"
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
    assert result.report().startswith("18 values listed by T4.1; complete")


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
