"""`generate_examples(per_example_fit=True)`: the sizing ceiling every
Llama-3.1-8B generation needs (estimator-frontier pre-registration, sec. 4.9,
amended 2026-10-03).

Without the flag, a task outside `_PER_EXAMPLE_FIT` is sized once per budget
on example 0, and a later example whose needles tokenize longer lands above
budget. That happened in banked data (up to +82 tokens; 32,769-token prompts
at 32768). The counter below reproduces it on CPU: it charges extra for
digits, so UUID needles with more digits cost more, as they can under a real
BPE tokenizer. The filler has no digits, so the count stays monotone in
filler units, which the binary search needs.
"""
from __future__ import annotations

import pytest

from attnbench.accuracy import ruler, sizing


def _digit_heavy(text: str) -> int:
    return sizing.approximate_token_count(text) + 3 * sum(c.isdigit() for c in text)


def _deltas(task, *, per_example_fit):
    exs = ruler.generate_examples(task, [3000], n_per_length=12, seed=0,
                                  count_tokens=_digit_heavy,
                                  per_example_fit=per_example_fit)
    return [e.context_length - e.token_budget for e in exs]


def test_the_counter_reproduces_the_overshoot_without_the_flag():
    """Non-vacuity: if no example overshoots without the flag, the next test
    proves nothing."""
    assert "niah_multikey" not in ruler._PER_EXAMPLE_FIT
    assert max(_deltas("niah_multikey", per_example_fit=False)) > 0


@pytest.mark.parametrize("task", ["niah_multikey", "niah_single", "vt"])
def test_the_flag_holds_every_example_at_or_under_budget(task):
    assert max(_deltas(task, per_example_fit=True)) <= 0


def test_the_default_is_unchanged():
    kw = dict(token_budgets=[3000], n_per_length=12, seed=0,
              count_tokens=_digit_heavy)
    assert (ruler.generate_examples("niah_multikey", **kw)
            == ruler.generate_examples("niah_multikey", per_example_fit=False, **kw))


def test_every_llama_candidate_task_is_already_fitted_per_example():
    """The Llama task probe draws from T4's candidate set (sec. 4.9). All five
    are in `_PER_EXAMPLE_FIT`, so the flag changes none of their prompts."""
    assert set(ruler.T4_DENSE_PILOT_TASKS) <= ruler._PER_EXAMPLE_FIT


def test_the_flag_refuses_an_example_that_still_lands_over_budget(monkeypatch):
    real_fit = sizing.fit_units_to_budget

    def overfit(render, count_tokens, budget, **kw):
        r = real_fit(render, count_tokens, budget, **kw)
        return sizing.SizingResult(units=r.units * 2, tokens=r.tokens, budget=budget)

    monkeypatch.setattr(sizing, "fit_units_to_budget", overfit)
    with pytest.raises(sizing.BudgetTooSmallError, match="above the 3000-token budget"):
        _deltas("niah_single", per_example_fit=True)
