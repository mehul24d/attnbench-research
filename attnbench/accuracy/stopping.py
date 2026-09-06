"""When a Stage 3 generation stops, and what it is allowed to cost.

The decision this implements is written up in
docs/stage3_generation_decision.md and confirmed there; this module is the
executable form of it, kept out of the runner script so the caps and the
token-set derivation are testable on CPU without a model.

The load-bearing finding: **EOS essentially never fires on these prompts.**
RULER's templates are completion-style and end mid-sentence with a trailing
space ("... mentioned in the provided text are", "... they are: "). There is
no chat template and no assistant turn, so an instruct model's EOS -- Qwen's
`<|im_end|>`, which closes an assistant turn that never opened -- has nothing
to close. An "EOS or cap" rule would therefore be a "cap" rule in practice:
every example would exit on its cap, and truncation rate would read ~100% for
every backend and carry no information about any of them. Hence the newline
stop, which is what actually terminates an answer here.
"""

from __future__ import annotations

from typing import Iterable, Optional

# 2x the longest answer observed over 200 examples per task, tokenised with
# the real Qwen tokenizer (7 / 36 / 20). Per task, not a flat budget: a flat
# cap generous enough for niah_multikey would be ~10x niah_single's answer
# and would let a rambling backend look identical to a terse one.
#
# The risk is one-sided, which is why generous caps are cheap. Over-capping
# costs decode steps at well under 1% of the measured phase. Under-capping
# scores a correct answer wrong -- a false negative that is invisible in the
# data, because a truncated answer and a wrong answer both just score 0.
TASK_TOKEN_CAPS: dict[str, int] = {
    "niah_single": 14,
    "niah_multikey": 72,
    "vt": 40,
}


def token_cap(task: str) -> int:
    """The per-task cap, or a raise. Never a default.

    A default here would silently apply one task's answer-length distribution
    to another's, and the failure would be a quietly truncated answer scored
    as wrong -- indistinguishable in the parquet from a backend that got it
    wrong. Adding a task to the grid must mean measuring its answer lengths.
    """
    try:
        return TASK_TOKEN_CAPS[task]
    except KeyError:
        raise KeyError(
            f"no measured token cap for task {task!r} (have "
            f"{sorted(TASK_TOKEN_CAPS)}). Caps come from measured answer "
            f"lengths -- see docs/stage3_generation_decision.md -- so a new "
            f"task needs its distribution measured, not a borrowed number."
        ) from None


def newline_token_ids(tokenizer) -> frozenset[int]:
    """Every token id whose text contains a newline.

    Derived from the vocabulary, not hardcoded. A byte-level BPE vocabulary
    has many of these -- bare "\\n", "\\n\\n", and any token that merges a
    newline onto neighbouring text -- and which ids they are is a property of
    the tokenizer, not something to look up once and paste in. Getting the
    set wrong in the direction of missing entries is the dangerous one: the
    stop silently never fires and every example runs to its cap.

    `skip_special_tokens=False` deliberately: a special token whose surface
    form contains a newline still ends the line.
    """
    ids = list(range(len(tokenizer)))
    texts = tokenizer.batch_decode([[i] for i in ids], skip_special_tokens=False)
    return frozenset(i for i, text in zip(ids, texts) if "\n" in text)


def whitespace_token_ids(tokenizer) -> frozenset[int]:
    """Token ids whose text is entirely whitespace (and non-empty).

    Only used to decide when the newline stop arms -- see
    `first_stop_index`. Kept separate from `newline_token_ids` because a
    token can be both, and the two questions are different: "does this end
    the line" versus "has the model said anything yet".
    """
    ids = list(range(len(tokenizer)))
    texts = tokenizer.batch_decode([[i] for i in ids], skip_special_tokens=False)
    return frozenset(i for i, text in zip(ids, texts)
                     if text != "" and text.strip() == "")


def eos_token_ids(tokenizer, generation_config=None) -> frozenset[int]:
    """Every id the model or tokenizer considers a stop token.

    Collected from both places because they disagree: Qwen2.5-Instruct's
    tokenizer reports `<|im_end|>` while its generation_config lists both
    `<|im_end|>` and `<|endoftext|>`. Taking the union is right -- an id in
    either list is a stop token by somebody's definition, and this rule is
    expected to fire ~never anyway (see the module docstring), so a false
    positive here costs a truncated answer and a false negative costs
    nothing at all.
    """
    found: set[int] = set()

    def _add(value) -> None:
        if value is None:
            return
        if isinstance(value, int):
            found.add(value)
        elif isinstance(value, Iterable):
            for v in value:
                _add(v)

    _add(getattr(tokenizer, "eos_token_id", None))
    if generation_config is not None:
        _add(getattr(generation_config, "eos_token_id", None))
    return frozenset(found)


def first_stop_index(token_ids: list[int], *, newline_ids: frozenset[int],
                     eos_ids: frozenset[int],
                     whitespace_ids: Optional[frozenset[int]] = None,
                     ) -> Optional[tuple[int, str]]:
    """Pure form of the stopping rule, for testing it without a model.

    Returns (index of the stopping token, reason) or None if nothing stops.

    `whitespace_ids` is a refinement of the confirmed rule rather than a
    change to it: the newline stop does not arm until at least one
    non-whitespace token has been produced. Both templates end with a
    trailing space mid-sentence so a leading newline is unlikely, but if one
    ever came first the rule as literally written would stop with an empty
    answer -- scoring 0 on every row of every backend for a reason that has
    nothing to do with attention. It cannot turn a correct answer into a
    wrong one; it can only prevent an empty one.
    """
    whitespace_ids = whitespace_ids or frozenset()
    seen_content = False
    for i, tid in enumerate(token_ids):
        if tid in eos_ids:
            return i, "eos"
        if tid in newline_ids and seen_content:
            return i, "newline"
        if tid not in newline_ids and tid not in whitespace_ids:
            seen_content = True
    return None
