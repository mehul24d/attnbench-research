"""Token-budget sizing: the replacement for the words-per-unit estimate
that made grid seq_lens mean something other than what they said.

Tests use synthetic renderers with exactly known token counts, so the
search itself is verified rather than the tokenizer.
"""

from __future__ import annotations

import pytest

from attnbench.accuracy.ruler import generate_examples
from attnbench.accuracy.schema import Generated
from attnbench.accuracy.sizing import (
    BudgetTooSmallError, approximate_token_count, fit_units_to_budget)


def _linear_render(fixed: int, per_unit: int):
    """A prompt with `fixed` tokens of template plus `per_unit` per filler
    unit, rendered as that many whitespace-separated words."""
    def render(units: int) -> str:
        return " ".join(["w"] * (fixed + per_unit * units))
    return render


def _exact_word_count(text: str) -> int:
    return len(text.split())


# ---------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------

def test_finds_the_largest_fitting_unit_count():
    # 10 fixed + 5/unit, budget 100 -> 18 units = 100 exactly
    fit = fit_units_to_budget(_linear_render(10, 5), _exact_word_count, 100)
    assert fit.units == 18 and fit.tokens == 100


def test_never_exceeds_the_budget():
    """The search returns the largest FITTING value -- a context longer
    than its budget would recreate the overshoot this module removed."""
    for budget in range(20, 200, 7):
        fit = fit_units_to_budget(_linear_render(13, 7), _exact_word_count, budget)
        assert fit.tokens <= budget


def test_result_is_maximal_one_more_unit_would_overshoot():
    render = _linear_render(10, 5)
    fit = fit_units_to_budget(render, _exact_word_count, 100)
    assert _exact_word_count(render(fit.units + 1)) > 100


def test_budget_too_small_for_the_fixed_template_raises():
    """A cell that cannot be built at its nominal length is a finding, not
    something to paper over by returning an over-budget context."""
    with pytest.raises(BudgetTooSmallError):
        fit_units_to_budget(_linear_render(500, 5), _exact_word_count, 100)


def test_shortfall_reports_unused_budget():
    fit = fit_units_to_budget(_linear_render(10, 7), _exact_word_count, 100)
    assert fit.shortfall == 100 - fit.tokens >= 0


def test_search_is_logarithmic_not_linear_in_units():
    """Doubling-then-bisecting, not incremental fill: a 200k-unit budget
    must not cost 200k tokenizations."""
    calls = []

    def counting(text: str) -> int:
        calls.append(1)
        return _exact_word_count(text)

    fit_units_to_budget(_linear_render(10, 1), counting, 200_000)
    assert len(calls) < 60


def test_min_units_floor_is_respected():
    """Some generators cannot render below a floor (variable_tracking
    raises when there is less filler than chain links)."""
    fit = fit_units_to_budget(_linear_render(10, 5), _exact_word_count,
                               1000, min_units=5)
    assert fit.units >= 5


def test_nonpositive_budget_is_rejected():
    with pytest.raises(ValueError):
        fit_units_to_budget(_linear_render(1, 1), _exact_word_count, 0)


# ---------------------------------------------------------------------------
# end to end through the real generators
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task", ["niah_single", "niah_multikey", "vt"])
def test_generated_contexts_land_close_to_their_budget(task):
    """The property the whole change exists for: a grid seq_len is a token
    count. Previously a 'seq_len' of 4096 produced ~4958 real tokens (the
    measured 1.21x inflation); now it must land at or just under 4096.
    """
    examples = generate_examples(task, [4096], 2, seed=0,
                                 count_tokens=approximate_token_count)
    for ex in examples:
        assert ex.context_length <= ex.token_budget
        assert ex.context_length >= 0.95 * ex.token_budget


@pytest.mark.parametrize("task", ["niah_single", "niah_multikey", "vt"])
def test_budget_and_actual_are_both_recorded(task):
    """Keeping only one of them is how the nominal-vs-real confusion
    started; a row must carry the requested length and the delivered one."""
    ex = generate_examples(task, [2048], 1, seed=0,
                            count_tokens=approximate_token_count)[0]
    assert ex.token_budget == 2048
    assert ex.context_length > 0 and ex.haystack_units > 0


def test_generation_is_still_deterministic_under_the_search():
    a = generate_examples("niah_single", [2048], 3, seed=11,
                           count_tokens=approximate_token_count)
    b = generate_examples("niah_single", [2048], 3, seed=11,
                           count_tokens=approximate_token_count)
    assert [x.context for x in a] == [x.context for x in b]
    assert [x.haystack_units for x in a] == [x.haystack_units for x in b]


def test_larger_budget_yields_a_longer_context():
    short = generate_examples("niah_single", [1024], 1, seed=0,
                               count_tokens=approximate_token_count)[0]
    long = generate_examples("niah_single", [8192], 1, seed=0,
                              count_tokens=approximate_token_count)[0]
    assert long.context_length > short.context_length
    assert long.haystack_units > short.haystack_units


def test_a_budget_below_the_task_floor_raises_rather_than_malforming():
    """variable_tracking needs at least one filler sentence per chain link;
    below that its rng.sample raises an opaque 'Sample larger than
    population'. The floor turns that into a clear budget error."""
    with pytest.raises(BudgetTooSmallError):
        generate_examples("vt", [40], 1, seed=0,
                          count_tokens=approximate_token_count)


# ---------------------------------------------------------------------------
# the exact-sizing guard: approximate counting must never reach a real run
# ---------------------------------------------------------------------------

def test_approximate_counter_is_self_identifying():
    """The marker rides on the function, not a registry, so it cannot drift
    out of sync with what it describes."""
    from attnbench.accuracy.sizing import is_approximate
    assert is_approximate(approximate_token_count)


def test_an_ordinary_tokenizer_wrapper_is_treated_as_exact():
    """Real tokenizer wrappers are plain closures. Requiring them to opt in
    would be ceremony that eventually gets skipped -- approximate marks
    itself instead."""
    from attnbench.accuracy.sizing import is_approximate
    assert not is_approximate(lambda text: len(text.split()))


def test_examples_record_which_kind_of_sizing_produced_them():
    approx = generate_examples("vt", [1024], 1, seed=0,
                                count_tokens=approximate_token_count)[0]
    exact = generate_examples("vt", [1024], 1, seed=0,
                               count_tokens=lambda t: len(t.split()))[0]
    assert approx.sizing == "approximate"
    assert exact.sizing == "exact"


def test_real_run_refuses_approximately_sized_examples():
    """The guard that matters. Choosing the counter correctly at a call site
    is not enough -- this fires where rows would be written, because a
    results file of estimated lengths is indistinguishable from one of real
    lengths after the fact.
    """
    from attnbench.accuracy.runner import AccuracyCell, build_cells, run_accuracy
    from attnbench.config import AttnConfig

    examples = generate_examples("niah_single", [1024], 1, seed=0,
                                  count_tokens=approximate_token_count)
    cfg = AttnConfig(seq_len=1024, batch=1, n_heads_q=1, n_heads_kv=1,
                     head_dim=128, mask="causal")
    cells = build_cells(configs_by_backend={"sdpa_math": [cfg]},
                        examples_by_task_length={("niah_single", 1024): examples})
    examples_by_id = {("niah_single", e.example_id): e for e in examples}

    with pytest.raises(ValueError, match="approximate token counter"):
        run_accuracy(cells, out_dir="/tmp/should-never-be-written",
                     examples_by_id=examples_by_id,
                     generate_fn=lambda *a: Generated("x"), dry_run=False)


def test_dry_run_still_allows_approximate_sizing():
    """Dry runs exist precisely so planning can be exercised without
    downloading a tokenizer."""
    from attnbench.accuracy.runner import build_cells, run_accuracy
    from attnbench.config import AttnConfig

    examples = generate_examples("niah_single", [1024], 1, seed=0,
                                  count_tokens=approximate_token_count)
    cfg = AttnConfig(seq_len=1024, batch=1, n_heads_q=1, n_heads_kv=1,
                     head_dim=128, mask="causal")
    cells = build_cells(configs_by_backend={"sdpa_math": [cfg]},
                        examples_by_task_length={("niah_single", 1024): examples})
    report = run_accuracy(
        cells, out_dir="/tmp/should-never-be-written",
        examples_by_id={("niah_single", e.example_id): e for e in examples},
        generate_fn=lambda *a: Generated("x"), dry_run=True)
    assert report.total == 1


# ---------------------------------------------------------------------------
# the generator floor is derived, not hardcoded
# ---------------------------------------------------------------------------

def test_vt_minimum_units_tracks_num_hops_rather_than_a_fixed_5():
    """The floor exists because variable_tracking's rng.sample raises when
    filler is scarcer than the chain. It is 5 for the current params -- but
    a num_hops change must move it, or the same crash returns at a
    different number."""
    import attnbench.accuracy.ruler as r

    assert r._min_haystack_units("vt") == 5  # current params: 1 chain, 4 hops
    original = dict(r._VT_PARAMS["vt"])
    try:
        r._VT_PARAMS["vt"] = dict(num_chains=1, num_hops=9)
        assert r._min_haystack_units("vt") == 10
        r._VT_PARAMS["vt"] = dict(num_chains=3, num_hops=4)
        assert r._min_haystack_units("vt") == 15
    finally:
        r._VT_PARAMS["vt"] = original


def test_vt_renders_at_its_declared_floor_but_not_below():
    """Pins the floor to the generator's actual behaviour, so the formula
    cannot drift away from what variable_tracking really requires."""
    import attnbench.accuracy.ruler as r

    floor = r._min_haystack_units("vt")
    r._render("vt", 123, floor)  # must not raise
    with pytest.raises(ValueError, match="Sample larger than population"):
        r._render("vt", 123, floor - 1)
