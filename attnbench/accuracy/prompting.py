"""One way to turn a prompt into the model's input ids, used both to size it
and to run it (estimator-frontier pre-registration, sec. 4.9, 2026-10-03).

Sizing counts tokens in `ruler.generate_examples`, and generation tokenizes
again in `generation.generate_one`. Both called `tokenizer(text)` with the
default `add_special_tokens=True`, so they agreed, but nothing held them to
it. That matters on Llama-3.1, whose tokenizer prepends `<|begin_of_text|>`
(128000) by default: a sizer that counted with `add_special_tokens=False`
would undercount every prompt by one and could put a prompt over its budget
by exactly that token. Both sides now call `encode_prompt`, and
`generate_one` checks the fed length against the sized one.

The double-BOS guard exists because of the XAttention authors' code at
`e379887`. Their RULER template and every `text.json` calibration prompt
begin with a literal `<|begin_of_text|>` and are tokenized with
`add_special_tokens` left at its default (`profile_threshold.py:212`,
`eval/RULER/scripts/pred/model_wrappers.py:64`). Under those default
settings the pinned tokenizer gives `[128000, 128000, ...]` (checked
2026-10-03 on transformers 4.46.0 and 5.18.0, 156 of 156 texts). Their
library versions are not pinned, and their pipeline was not run here. The
file-and-line trace is in the estimator-frontier pre-registration, sec. 4.9.
This study feeds one BOS, and refuses two.
"""

from __future__ import annotations

# The run's setting, stated rather than defaulted. Sizing and generation both
# read it through `encode_prompt`.
ADD_SPECIAL_TOKENS = True


class DoubleBOSError(ValueError):
    """The encoded prompt starts with two BOS ids."""


def prompt_ids(tokenizer, text: str) -> list[int]:
    """The ids the model is fed for `text`, as a flat list."""
    ids = tokenizer(text, add_special_tokens=ADD_SPECIAL_TOKENS).input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        (ids,) = ids
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None and len(ids) >= 2 and ids[0] == bos and ids[1] == bos:
        raise DoubleBOSError(
            f"prompt encodes to two leading BOS ids ({bos}, {bos}); the text "
            f"already starts with the BOS token and the tokenizer added "
            f"another. Strip the literal BOS from the text.")
    return list(ids)


def encode_prompt(tokenizer, text: str, device=None):
    """`prompt_ids` as a (1, n) long tensor, optionally on `device`."""
    import torch
    t = torch.tensor([prompt_ids(tokenizer, text)], dtype=torch.long)
    return t if device is None else t.to(device)


def prompt_token_counter(tokenizer):
    """The sizer for `ruler.generate_examples`: exactly `len(prompt_ids)`."""
    def count_tokens(text: str) -> int:
        return len(prompt_ids(tokenizer, text))
    return count_tokens
