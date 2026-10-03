"""Thin wrapper around the vendored RULER task generators and scorer
(attnbench/_vendor/ruler/). See that directory's VENDORED.md for exactly
what's verbatim, what's adapted, and what's deliberately not implemented
(essay haystack, word needles, common-words-extraction, QA).

Every example produced here uses RULER's task-*construction* algorithm with
a noise/needle-haystack substitution in place of real prose -- not RULER's
published benchmark. Results are valid for comparing backends against each
other on identical inputs; they are not comparable to published RULER
numbers. See schema.AccuracyResult.haystack_mode.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from . import ruler_data, sizing
from .._vendor.ruler import niah, qa, scoring, variable_tracking

# Task -> RULER scoring category (attnbench._vendor.ruler.scoring.TASKS key).
_TASK_CATEGORY = {
    "niah_single": "niah",
    "niah_multikey": "niah",
    "vt": "variable_tracking",
    # Audit T4 (2026-10-01): RULER presets under RULER's own names, with
    # RULER's own parameters and no deviation -- candidates for a task the
    # dense model does not ace. Their resources come from ruler_data.py.
    "niah_multikey_1": "niah",
    "niah_multivalue": "niah",
    "niah_multiquery": "niah",
    "qa_1": "qa",
    "qa_2": "qa",
}

# Task -> generator params. Each entry notes exactly which axis (if any)
# deviates from the RULER preset it's modeled on -- see synthetic.yaml in
# the RULER repo for the originals.
_NIAH_PARAMS = {
    # Modeled on RULER's niah_single_1: haystack_mode="noise" matches that
    # preset exactly. type_needle_k="uuids" is the one substitution (RULER
    # uses "words", which needs wonderwords -- not added here).
    "niah_single": dict(haystack_mode="noise", type_needle_k="uuids",
                         type_needle_v="numbers", num_needle_k=1,
                         num_needle_v=1, num_needle_q=1),
    # Modeled on RULER's niah_multikey_3, and DEVIATING from it on one axis:
    # haystack_mode="needle" and uuids/uuids match that preset, but
    # `num_needle_k` is 4 here and **1** upstream (scripts/synthetic.yaml at
    # the vendored commit). Four genuine needles among same-shaped decoys is a
    # harder retrieval task than the preset's single needle, and it is the
    # deviation that matters most for comparability -- it changes the task, not
    # the vocabulary. Recorded in NOTICE's comparability section.
    #
    # This comment read "matches that preset exactly -- no substitution needed
    # for this one" until 2026-09-23, when the vendored tree was first diffed
    # against upstream. Three audit passes had deferred that diff, so the one
    # entry here claiming to need no note was the one carrying the larger of
    # the two deviations. The header above promises each entry names its axis;
    # this one did the opposite.
    "niah_multikey": dict(haystack_mode="needle", type_needle_k="uuids",
                           type_needle_v="uuids", num_needle_k=4,
                           num_needle_v=1, num_needle_q=1),
}
# RULER's synthetic.yaml presets at the vendored commit, verbatim: essay
# haystack, word keys, number values.
_NIAH_PARAMS.update({
    "niah_multikey_1": dict(haystack_mode="essay", type_needle_k="words",
                            type_needle_v="numbers", num_needle_k=4,
                            num_needle_v=1, num_needle_q=1),
    "niah_multivalue": dict(haystack_mode="essay", type_needle_k="words",
                            type_needle_v="numbers", num_needle_k=1,
                            num_needle_v=4, num_needle_q=1),
    "niah_multiquery": dict(haystack_mode="essay", type_needle_k="words",
                            type_needle_v="numbers", num_needle_k=1,
                            num_needle_v=1, num_needle_q=4),
})
_VT_PARAMS = {
    "vt": dict(num_chains=1, num_hops=4),
}
# RULER qa_1 / qa_2. Filler units are DOCUMENTS here, not sentences.
_QA_PARAMS = {"qa_1": "squad", "qa_2": "hotpotqa"}

# Fixed before the T4 dense-only pilot; sparse results cannot change this set.
T4_DENSE_PILOT_TASKS = (
    "niah_multikey_1", "niah_multivalue", "niah_multiquery", "qa_1", "qa_2"
)

# Tasks whose filler differs per example (QA: each question draws its own
# documents) or whose resources make per-example token counts vary (essay,
# word needles). They are sized per example, so every example lands at or
# under its budget by construction rather than by assuming the first
# example's fit holds for the rest -- which is all the existing tasks rely
# on. That is NOT safe for them: their filler is uniform, but their needles
# (UUIDs, numbers) are not, so later examples overshoot by up to +82 tokens
# (see the `RulerExample` docstring; corrected 2026-10-03). Left unchanged
# here because changing the sizing would change every regenerated prompt for
# those tasks and break identity with banked rows. New runs that need the
# ceiling (every Llama-3.1-8B generation) pass `per_example_fit=True`.
_PER_EXAMPLE_FIT = frozenset(_QA_PARAMS) | frozenset(
    t for t, p in _NIAH_PARAMS.items()
    if p["haystack_mode"] == "essay" or "words" in (p["type_needle_k"], p["type_needle_v"]))


@dataclass(frozen=True)
class RulerExample:
    """One generated example. `question` is kept for schema symmetry with
    the plan's original field list, but is always empty here: RULER's own
    templates fold the question into the formatted prompt rather than
    keeping it separate, so there's nothing distinct to put there.

    `context_length` is the REAL tokenized length of `context` under the
    tokenizer that generated it -- not a word-count estimate.

    It is at or below `token_budget` only for tasks in `_PER_EXAMPLE_FIT`,
    which are sized per example, or when `generate_examples` was called with
    `per_example_fit=True`. Every other task is sized ONCE per budget,
    on example 0, and that filler count is reused for the rest; their needles
    (UUIDs, numbers) tokenize to different lengths per example, so an example
    can land ABOVE budget. Measured 2026-10-03 across banked data:
    `niah_multikey` up to +82 tokens at 16384, `niah_single` up to +20,
    `vt` up to +5, and `niah_single` +1 at 32768 -- a 32,769-token prompt,
    past Qwen2.5's 32,768-position limit
    (`results/positions_over_limit/sizing_delta_by_band_task.csv`). Callers
    that need a hard ceiling must check `context_length` themselves.
    (Until 2026-10-03 this said "never above".)

    Both are recorded because they are not identical: the budget is the
    grid's requested length, and the actual is what the filler granularity
    allowed. Keeping only one of them is how the earlier nominal-vs-real
    confusion started.
    """

    task: str
    example_id: str
    context: str
    question: str
    answer: list[str]
    context_length: int
    token_budget: int = 0
    haystack_units: int = 0
    # "exact" (a real tokenizer sized this) or "approximate" (a dry-run /
    # test estimate). Travels on the example so the guard can be enforced
    # where results are written, not only where the counter was chosen.
    sizing: str = "exact"


def _min_haystack_units(task: str) -> int:
    """Smallest filler count a task's generator can actually render.

    Not cosmetic: variable_tracking inserts one assignment statement per
    chain link into the noise sentences via `rng.sample(range(len(
    sentences)), len(chain))`, which raises "Sample larger than population"
    whenever there is less filler than chain. With num_chains=1/num_hops=4
    that floor is 5.

    NIAH needs the same floor for the opposite reason, and this returned a
    hardcoded 1 until 2026-09-23. Its vendored builder clamps insertion with
    `min(len(needles), num_haystack)` where upstream samples `len(needles)`, so
    it does not raise -- it silently inserts fewer needles than it has answers
    for and emits an example whose answer is absent from its own prompt.
    "Tolerates 0" meant "does not crash", not "produces a valid example".
    Measured at num_needle_k=4: unanswerable for 74.8% of seeds at
    num_haystack=1, 53.8% at 2, 24.8% at 3, 0% at 4. So the binding constraint
    is `num_haystack >= num_needle_k` (which the builder itself raises to
    `max(num_needle_k, num_needle_q)`), and that is what is derived below.
    No banked measurement was in that range -- every band is >= 2048 tokens,
    i.e. hundreds of filler sentences -- but the sizing search starts at this
    floor and RETURNS it, so a small enough budget reached it and
    BudgetTooSmallError did not fire.

    The sizing search starts here, and a budget too small to fit even this
    raises BudgetTooSmallError rather than silently producing a malformed
    example.

    Derived from the task's own params, never hardcoded: the current value
    happens to be 5, but a future `num_hops` or `num_chains` change must
    move this floor with it rather than reintroducing the same crash at a
    different number. `_generate_chains` builds `num_hops + 1` statements
    per chain (one initial assignment plus one per hop), and the first
    chain's `rng.sample` runs against the un-grown noise list, so
    `num_noise >= num_hops + 1` is the binding constraint. Multiplying by
    `num_chains` is deliberately conservative -- later chains sample from a
    list already grown by earlier insertions, so they need less -- and
    costs only a slightly larger minimum haystack.
    """
    if task in _VT_PARAMS:
        params = _VT_PARAMS[task]
        statements_per_chain = params["num_hops"] + 1
        return params["num_chains"] * statements_per_chain
    if task in _QA_PARAMS:
        qas, _ = ruler_data.qa_dataset(_QA_PARAMS[task])
        return max(qa.min_docs(qas, i) for i in range(min(len(qas), 1000)))
    if task in _NIAH_PARAMS:
        params = _NIAH_PARAMS[task]
        # The builder's own `num_needle_k = max(num_needle_k, num_needle_q)`,
        # mirrored here so a future preset change moves this floor with it
        # rather than reintroducing silent unanswerable examples at a different
        # number. Derived from the task's params, never hardcoded -- a rule this
        # function's docstring already stated and the NIAH branch ignored.
        return max(params["num_needle_k"], params["num_needle_q"])
    raise ValueError(f"unknown task {task!r}")


def _niah_resources(params: dict) -> dict:
    """The injected resources a NIAH preset needs, and only those: the
    existing noise/needle presets get none, so their examples are built
    exactly as before (pinned by tests/test_ruler_t4_tasks.py)."""
    out = {}
    if params["haystack_mode"] == "essay":
        out["essay_words"] = ruler_data.essay_words()
        out["sent_tokenize"] = ruler_data.sent_tokenizer()
    if "words" in (params["type_needle_k"], params["type_needle_v"]):
        out["word_pool"] = ruler_data.word_pool()
    return out


def _render(task: str, example_seed: int, num_haystack: int,
            index: int = 0) -> tuple[str, list[str]]:
    """One example's (prompt, answers) at a given filler-unit count.
    `index` selects the question for QA (the same questions at every band,
    as upstream); the generated tasks ignore it."""
    if task in _QA_PARAMS:
        qas, docs = ruler_data.qa_dataset(_QA_PARAMS[task])
        return qa.generate_qa_example(qas, docs, index=index,
                                      num_docs=num_haystack, seed=example_seed)
    if task in _NIAH_PARAMS:
        params = _NIAH_PARAMS[task]
        return niah.generate_niah_example(
            num_haystack=num_haystack, seed=example_seed, **params,
            **_niah_resources(params))
    if task in _VT_PARAMS:
        return variable_tracking.generate_vt_example(
            num_noise=num_haystack, seed=example_seed, **_VT_PARAMS[task])
    raise AssertionError(f"task {task!r} in _TASK_CATEGORY but not wired "
                          f"to a generator")


def _example_seed(seed: int, task: str, num_haystack: int, index: int) -> int:
    """Deterministic per-example seed: same (seed, task, num_haystack,
    index) always reproduces the identical example, matching the
    determinism discipline masks.py uses for mask generation.
    """
    blob = f"{seed}|{task}|{num_haystack}|{index}".encode()
    return int(hashlib.sha1(blob).hexdigest()[:8], 16)


def generate_examples(task: str, token_budgets: list[int], n_per_length: int,
                       seed: int, *, count_tokens: sizing.TokenCounter,
                       per_example_fit: bool = False,
                       ) -> list[RulerExample]:
    """Generate `n_per_length` examples at each of `token_budgets` for
    `task`.

    `token_budgets` are EXACT TOKEN COUNTS, not haystack-unit counts. The
    filler is binary-searched so the rendered prompt lands at or just under
    its budget under `count_tokens` -- per example for `_PER_EXAMPLE_FIT`
    tasks or with `per_example_fit=True`, otherwise once per budget on
    example 0, which later examples can overshoot (see `RulerExample`; this
    said "Each example's filler" until 2026-10-03) -- so a grid `seq_len` of
    16384 produces ~16384 real tokens, not the ~19821 the old word-count
    heuristic produced. See accuracy/sizing.py for why this replaced the
    estimate-then-correct approach.

    `count_tokens` is required and has no default: it must be the target
    model's real tokenizer for anything that produces results. Dry runs and
    tests pass `sizing.approximate_token_count` explicitly, so estimating
    is always a visible choice at the call site rather than a fallback.

    Deterministic: identical arguments always reproduce identical examples.
    The filler-unit count is solved once per budget (from the first
    example's seed) and reused across that budget's examples -- filler size
    is essentially constant at a fixed budget since only needle contents
    vary -- but each example's own token count is measured and recorded.

    `per_example_fit=True` sizes EVERY example on its own, whatever the
    task, and raises if any example still lands above its budget. The
    estimator-frontier pre-registration (sec. 4.9) requires it for every
    Llama-3.1-8B generation. It defaults to False because turning it on for
    a task outside `_PER_EXAMPLE_FIT` changes that task's prompts, and the
    banked rows were generated without it (added 2026-10-03).
    """
    if task not in _TASK_CATEGORY:
        raise ValueError(f"unknown task {task!r}; known tasks: "
                          f"{sorted(_TASK_CATEGORY)}")

    examples = []
    for budget in token_budgets:
        # Solve the filler count once per budget, against the first
        # example's seed, rather than per example: ~log2(n) tokenizations
        # of a 30k-token string per cell instead of per row.
        probe_seed = _example_seed(seed, task, budget, 0)
        fit = sizing.fit_units_to_budget(
            lambda units: _render(task, probe_seed, units)[0],
            count_tokens, budget, min_units=_min_haystack_units(task))

        for i in range(n_per_length):
            example_seed = _example_seed(seed, task, budget, i)
            example_id = f"{task}_{budget}_{i}"
            units = fit.units
            if per_example_fit or task in _PER_EXAMPLE_FIT:
                units = sizing.fit_units_to_budget(
                    lambda u, s=example_seed, i=i: _render(task, s, u, index=i)[0],
                    count_tokens, budget, min_units=_min_haystack_units(task)).units
            text, answer = _render(task, example_seed, units, index=i)
            if per_example_fit and count_tokens(text) > budget:
                raise sizing.BudgetTooSmallError(
                    f"{example_id}: {count_tokens(text)} tokens after a "
                    f"per-example fit, above the {budget}-token budget")
            examples.append(RulerExample(
                task=task, example_id=example_id, context=text, question="",
                answer=answer, context_length=count_tokens(text),
                token_budget=budget, haystack_units=units,
                sizing=("approximate" if sizing.is_approximate(count_tokens)
                        else "exact")))
    return examples


def haystack_mode_for(task: str) -> str:
    """The haystack_mode a task's examples were generated with -- for
    populating AccuracyResult.haystack_mode. vt only implements the noise
    haystack (see variable_tracking.py), so it's hardcoded there; niah
    tasks read it from their own params so this can't drift from what
    generate_examples actually did.
    """
    if task in _NIAH_PARAMS:
        return _NIAH_PARAMS[task]["haystack_mode"]
    if task in _VT_PARAMS:
        return "noise"
    if task in _QA_PARAMS:
        return "documents"
    raise ValueError(f"unknown task {task!r}")


def score(task: str, predicted: str, expected: list[str]) -> float:
    """Score one prediction against its expected answer(s), via RULER's own
    (unmodified, vendored verbatim) per-task-category metric function."""
    category = _TASK_CATEGORY[task]
    metric_fn = scoring.TASKS[category]["metric_fn"]
    return metric_fn([predicted], [expected])
