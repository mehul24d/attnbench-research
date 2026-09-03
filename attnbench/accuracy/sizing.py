"""Size a generated context to an exact token budget, using the real
tokenizer, instead of estimating it from a word count.

**Why this replaces an estimate.** The first version of this pipeline sized
haystacks with `_APPROX_TOKENS_PER_HAYSTACK_UNIT = 20` -- a words-per-unit
guess -- so a grid `seq_len` of 16384 was a *target* the generator aimed at,
not a token count. The real tokenized contexts came out ~21% longer
(8192 -> 9984, 16384 -> 19821, 32768 -> 39474). Every Stage 3 FLOPs and
hour estimate built on the nominal number was correspondingly low, including
the 34.08h figure the grid was cut against.

That was patched once by measuring the ratio and multiplying it back in
(`timing_probe.MEASURED_TOKEN_INFLATION = 1.21`). This module removes the
need for that correction: if the generator hits the token budget directly,
a grid length of 16384 means 16384 tokens and no ratio has to be carried,
trusted, or re-measured when the tokenizer or task templates change. The
ratio held to under 1% across a 4x span of lengths, which made it a decent
assumption -- but it was still an assumption about a quantity we can simply
compute.

The approach is RULER's own (and Sparse Frontier's, in
`sparse_frontier/tasks/ruler/`, Apache 2.0): binary-search the amount of
filler so the *rendered prompt* -- template, needles, question and all, not
just the haystack -- lands at or just under the budget.

Search cost is managed by solving once per (task, budget) rather than once
per example: filler-unit count is essentially constant across examples at a
fixed budget, since only needle contents vary. 4500 examples would otherwise
need ~80k tokenizations of 30k-token strings; this needs a few hundred.
Each example's *actual* token count is then measured and recorded, so the
small residual variation is data rather than an assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# text -> number of tokens under the target model's real tokenizer.
TokenCounter = Callable[[str], int]

# Renders the full prompt for a given number of filler units. Must be
# monotonically non-decreasing in its argument for the search to be valid.
Renderer = Callable[[int], str]


def approximate_token_count(text: str) -> int:
    """A deliberately crude words-to-tokens estimate, ~1.3 tokens/word.

    Provided ONLY for dry runs and tests, where no real tokenizer is
    available and the exact length does not matter. It is a named,
    explicitly-chosen function rather than a default, so that no caller can
    silently fall back to estimating -- silently estimating is precisely
    the failure this module exists to end. Anything that produces real
    results must pass a real tokenizer's counter.

    Carries `is_approximate = True` so the marker travels with the function
    itself. Choosing it correctly at a call site is not enough: the guard
    has to hold at the point results would be *written*, which is where the
    harm is (see `is_approximate` and `runner.run_accuracy`).
    """
    return int(len(text.split()) * 1.3)


# Attached to the function object, not tracked in a registry, so it cannot
# drift out of sync with the function it describes and survives being passed
# around, wrapped in functools.partial, or aliased.
approximate_token_count.is_approximate = True


def is_approximate(count_tokens: TokenCounter) -> bool:
    """Whether `count_tokens` is an estimate rather than a real tokenizer.

    Defaults to False for unmarked callables: a real tokenizer wrapper is an
    ordinary closure with no marker, and requiring every honest caller to
    opt in would be the kind of ceremony that gets skipped. Anything
    deliberately approximate marks itself.
    """
    return bool(getattr(count_tokens, "is_approximate", False))


@dataclass(frozen=True)
class SizingResult:
    """The filler-unit count that fits a budget, and what it actually cost."""

    units: int
    tokens: int
    budget: int

    @property
    def shortfall(self) -> int:
        """Tokens left unused under the budget. Small is good; a large
        shortfall means the filler unit is too coarse to size precisely."""
        return self.budget - self.tokens


class BudgetTooSmallError(ValueError):
    """The prompt exceeds the token budget even with the minimum filler.

    Raised rather than silently returning an over-budget context: a cell
    that cannot be built at its nominal length is a finding about the task
    and budget, not something to paper over by returning something longer
    than asked for.
    """


def fit_units_to_budget(render: Renderer, count_tokens: TokenCounter,
                         budget: int, *, min_units: int = 0,
                         max_units: int | None = None) -> SizingResult:
    """Largest `units` whose rendered prompt fits within `budget` tokens.

    Doubles from `min_units` to bracket the budget, then binary-searches.
    O(log n) tokenizations rather than the O(n) an incremental fill would
    need at these lengths.

    Assumes `render` is monotonically non-decreasing in units, which holds
    for every generator here (more filler sentences, longer prompt). The
    result is the largest fitting value, so the context lands at or just
    below budget, never above.
    """
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    if min_units < 0:
        raise ValueError(f"min_units must be >= 0, got {min_units}")

    floor_tokens = count_tokens(render(min_units))
    if floor_tokens > budget:
        raise BudgetTooSmallError(
            f"prompt is {floor_tokens} tokens with the minimum {min_units} "
            f"filler units, which already exceeds the {budget}-token budget "
            f"-- the task's fixed content (template, needles, question) does "
            f"not fit at this length"
        )

    # Bracket: double until we overshoot, so the search range is found in
    # O(log n) rather than assuming an upper bound that may be wrong for a
    # different task or template.
    lo, hi = min_units, max(min_units, 1) * 2
    if max_units is not None:
        hi = min(hi, max_units)
    while count_tokens(render(hi)) <= budget:
        lo = hi
        hi *= 2
        if max_units is not None and hi >= max_units:
            hi = max_units
            if count_tokens(render(hi)) <= budget:
                return SizingResult(units=hi, tokens=count_tokens(render(hi)),
                                    budget=budget)
            break

    # Invariant: lo fits, hi does not. Narrow to the largest fitting value.
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if count_tokens(render(mid)) <= budget:
            lo = mid
        else:
            hi = mid

    return SizingResult(units=lo, tokens=count_tokens(render(lo)), budget=budget)
