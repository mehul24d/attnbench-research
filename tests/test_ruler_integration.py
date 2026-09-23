"""attnbench.accuracy.ruler: the thin wrapper around vendored RULER task
generators and scorer. Pure text/Python, no torch dependency -- these must
run without a model or GPU.
"""

from __future__ import annotations

import pytest

from attnbench.accuracy.ruler import RulerExample, generate_examples, score
from attnbench.accuracy.sizing import approximate_token_count


def _gen(*args, **kwargs):
    """generate_examples with the explicit test token counter. Real runs
    pass the target model's tokenizer; there is deliberately no default."""
    return generate_examples(*args, count_tokens=approximate_token_count, **kwargs)
from attnbench._vendor.ruler import niah, variable_tracking
from attnbench.accuracy.ruler import (  # noqa: E402
    _NIAH_PARAMS, _min_haystack_units, _render)


def test_determinism_same_seed_reproduces_identical_examples():
    ex1 = _gen("niah_single", [1024], 3, seed=42)
    ex2 = _gen("niah_single", [1024], 3, seed=42)
    assert ex1 == ex2


def test_different_seed_produces_different_examples():
    ex1 = _gen("niah_single", [1024], 1, seed=1)
    ex2 = _gen("niah_single", [1024], 1, seed=2)
    assert ex1[0].context != ex2[0].context


@pytest.mark.parametrize("task", ["niah_single", "niah_multikey"])
def test_niah_needle_values_appear_verbatim_in_context(task):
    examples = _gen(task, [1024], 2, seed=7)
    for ex in examples:
        for answer_value in ex.answer:
            assert answer_value in ex.context


def test_niah_single_uses_noise_haystack_matching_ruler_preset():
    """niah_single is modeled on RULER's niah_single_1, which itself uses
    type_haystack=noise -- so this one has no haystack substitution."""
    examples = _gen("niah_single", [1024], 1, seed=3)
    assert "The grass is green" in examples[0].context


def test_niah_multikey_has_multiple_keys():
    """niah_multikey uses haystack_mode="needle": the filler haystack
    sentences are themselves needle-shaped decoys, so the real signal is 4
    genuine needles (num_needle_k=4) hidden among decoys of the same shape.
    Total matches are therefore (filler units + 4), not 4.

    Asserted against the example's own recorded `haystack_units` rather
    than a hardcoded count: the unit count is now solved against a token
    budget rather than passed in, so hardcoding it would pin this test to
    one tokenizer's arithmetic instead of to the property being tested.
    """
    examples = _gen("niah_multikey", [2048], 1, seed=3)
    ex = examples[0]
    assert ex.context.count("One of the special magic") == ex.haystack_units + 4


def test_vt_answer_is_the_full_variable_chain():
    examples = _gen("vt", [1024], 1, seed=5)
    ex = examples[0]
    assert len(ex.answer) == 5  # num_hops=4 -> 5 variables in the chain
    for var_name in ex.answer:
        assert var_name in ex.context


def test_score_exact_match_is_full_score():
    examples = _gen("niah_single", [1024], 1, seed=9)
    ex = examples[0]
    assert score("niah_single", ex.answer[0], ex.answer) == 100.0


def test_score_wrong_answer_is_zero():
    examples = _gen("niah_single", [1024], 1, seed=9)
    ex = examples[0]
    assert score("niah_single", "definitely not the answer", ex.answer) == 0.0


def test_score_vt_partial_match_is_partial_score():
    """string_match_all averages over every expected answer -- getting half
    the variable chain right should score around half, not all-or-nothing."""
    examples = _gen("vt", [1024], 1, seed=5)
    ex = examples[0]
    half = ex.answer[: len(ex.answer) // 2]
    predicted_text = " ".join(half)
    got = score("vt", predicted_text, ex.answer)
    assert 0.0 < got < 100.0


def test_unknown_task_raises():
    with pytest.raises(ValueError):
        _gen("not_a_real_task", [1024], 1, seed=0)


def test_essay_haystack_raises_not_wired_up():
    """The seam is deliberately left in but not wired -- see VENDORED.md."""
    with pytest.raises(NotImplementedError):
        niah.generate_niah_example(num_haystack=10, seed=0, haystack_mode="essay")


def test_words_needle_type_raises_not_wired_up():
    with pytest.raises(NotImplementedError):
        niah.generate_niah_example(num_haystack=10, seed=0,
                                    type_needle_k="words")


def test_vt_generator_determinism_directly():
    """Same check as the wrapper-level determinism test, but against the
    vendored generator directly -- catches a regression in niah.py/
    variable_tracking.py itself, not just in the seed-derivation wrapper."""
    a = variable_tracking.generate_vt_example(num_noise=20, seed=123)
    b = variable_tracking.generate_vt_example(num_noise=20, seed=123)
    assert a == b


def test_a_smaller_n_per_length_is_the_PREFIX_of_a_larger_one():
    """The whole reduced-n re-run design rests on this.

    Examples are seeded per (seed, task, budget, index) and the filler-unit
    count is solved from index 0's seed, so example k is byte-identical
    whether n is 100 or 300. That is what keeps a 100-example re-run PAIRED
    with the banked 300-example rows -- same example_ids, same contexts, same
    answers -- rather than merely a smaller sample of a similar population.

    If this ever stopped holding, a reduced re-run would silently compare
    different prompts and the paired difference would be meaningless.
    """
    from attnbench.accuracy import sizing
    big = generate_examples("niah_single", [2048], 12, seed=0,
                            count_tokens=sizing.approximate_token_count)
    small = generate_examples("niah_single", [2048], 4, seed=0,
                              count_tokens=sizing.approximate_token_count)
    assert len(small) == 4 and len(big) == 12
    for a, b in zip(big, small):
        assert a.example_id == b.example_id
        assert a.context == b.context
        assert a.answer == b.answer
        assert a.haystack_units == b.haystack_units


def test_the_prefix_property_holds_across_tasks_and_budgets():
    """Not just the one task the re-run happens to use."""
    from attnbench.accuracy import sizing
    for task in ("niah_single", "vt"):
        for budget in (2048, 4096):
            big = generate_examples(task, [budget], 6, seed=0,
                                    count_tokens=sizing.approximate_token_count)
            small = generate_examples(task, [budget], 2, seed=0,
                                      count_tokens=sizing.approximate_token_count)
            assert [e.example_id for e in big[:2]] == \
                   [e.example_id for e in small], (task, budget)
            assert [e.context for e in big[:2]] == \
                   [e.context for e in small], (task, budget)


# --- the filler floor, and the defect it exists to keep out of reach --------
#
# Added 2026-09-23, when the vendored generators were first diffed against
# upstream. `_min_haystack_units` returned a hardcoded 1 for NIAH while its own
# docstring said the floor is "derived from the task's own params, never
# hardcoded" -- honoured in the vt branch, ignored in the NIAH one, where the
# binding parameter is num_needle_k. Below the floor the vendored builder's
# `min(len(needles), num_haystack)` clamp silently drops needles whose answers
# it still reports, where upstream would raise "Sample larger than population".

_UNANSWERABLE_SEEDS = 200


def _unanswerable(task: str, num_haystack: int, seeds: int = _UNANSWERABLE_SEEDS) -> int:
    """How many seeds produce an example whose own answer is absent from its
    prompt -- a question scored as a model failure that no model could pass."""
    params = _NIAH_PARAMS[task]
    bad = 0
    for seed in range(seeds):
        text, answers = niah.generate_niah_example(
            num_haystack=num_haystack, seed=seed, **params)
        if any(a not in text for a in answers):
            bad += 1
    return bad


@pytest.mark.parametrize("task", ["niah_single", "niah_multikey"])
def test_the_niah_filler_floor_is_derived_from_the_tasks_own_params(task):
    """Not the value 4 -- the derivation. Pinning the number would pass again
    the moment a preset changed and the floor did not follow it."""
    params = _NIAH_PARAMS[task]
    expected = max(params["num_needle_k"], params["num_needle_q"])
    assert _min_haystack_units(task) == expected, (
        f"{task}: floor is {_min_haystack_units(task)}, but the builder needs "
        f"{expected} filler units to place all its needles")


@pytest.mark.parametrize("task", ["niah_single", "niah_multikey"])
def test_no_example_at_the_floor_is_unanswerable(task):
    """The property the floor is FOR. Every example the sizing search can
    return must contain the answer it is scored against."""
    floor = _min_haystack_units(task)
    bad = _unanswerable(task, floor)
    assert bad == 0, (
        f"{task}: {bad}/{_UNANSWERABLE_SEEDS} seeds at the floor "
        f"num_haystack={floor} produce an example whose answer is not in its "
        f"own prompt. Those score 0 for every model and are indistinguishable "
        f"from a retrieval failure.")


def test_below_the_floor_examples_ARE_unanswerable():
    """The break-test, permanent rather than performed once.

    Expresses the defect as a fixture so the floor cannot be lowered back to 1
    and stay green. `niah_single` cannot express it -- with one needle the
    clamp is a no-op and its floor is legitimately 1 -- which is exactly why
    the hardcoded 1 looked correct to whoever wrote it.
    """
    floor = _min_haystack_units("niah_multikey")
    assert floor > 1, "niah_multikey's floor is 1 again; the derivation is gone"

    bad = _unanswerable("niah_multikey", 1)
    assert bad > _UNANSWERABLE_SEEDS // 2, (
        f"only {bad}/{_UNANSWERABLE_SEEDS} seeds are unanswerable at "
        f"num_haystack=1, so this fixture no longer demonstrates the defect "
        f"the floor prevents (it was ~75%)")

    # monotone: the closer to the floor, the fewer bad examples, reaching zero
    # AT it. If this ever inverts, the clamp is not what is causing them.
    rates = [_unanswerable("niah_multikey", n) for n in range(1, floor + 1)]
    assert rates == sorted(rates, reverse=True), rates
    assert rates[-1] == 0, rates


def test_the_sizing_search_cannot_return_below_the_floor():
    """Where it actually bites: `search_units` starts at `min_units` and returns
    it, so the floor is the only thing standing between a small budget and an
    unanswerable example. A budget too small must raise, not round down."""
    from attnbench.accuracy.sizing import (BudgetTooSmallError,
                                           fit_units_to_budget)

    floor = _min_haystack_units("niah_multikey")
    render = lambda n: _render("niah_multikey", 7, n)[0]      # noqa: E731

    at_floor = approximate_token_count(render(floor))
    r = fit_units_to_budget(render, approximate_token_count, at_floor,
                            min_units=floor)
    assert r.units >= floor, r

    with pytest.raises(BudgetTooSmallError):
        fit_units_to_budget(render, approximate_token_count, at_floor - 1,
                            min_units=floor)
