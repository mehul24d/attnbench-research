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
