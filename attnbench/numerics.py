"""Float32 matmul precision, enforced rather than inherited.

`score_source="dense_softmax_fp32"` is stamped on every accuracy row in this
study and is load-bearing: the oracle ranking is the thing the whole
deployability argument is about, and "fp32" is the claim that distinguishes it
from a cheaper estimator. Until 2026-09-20 nothing in this codebase enforced
it.

**The gap, precisely.** `model._scoring_forward_chunked` does
`q32, k32 = q.float(), k.float()` and then `torch.matmul(...)`. On Ampere and
later that is a **TF32-eligible** matmul: torch may run it with a 10-bit
mantissa instead of 24. Whether it does is decided by
`torch.backends.cuda.matmul.allow_tf32`, a global whose default has *moved*
across torch versions (True in 1.7-1.11, False from 1.12), and
`pyproject.toml` pins `torch>=2.6` with no upper bound. So the banked runs
were almost certainly true fp32 -- by the default's doing, not the code's --
and a future `pip install -U` could silently change what the oracle ranks on
while every row went on claiming `dense_softmax_fp32`.

That is the exact shape this project keeps meeting: a field that is correct
because of something outside the code, and a label that would go on being
written after it stopped being true.

**Why this is a raise and not a provenance field.** The obvious alternative is
to stamp the flag state onto every row. `tests/test_provenance_consulted.py`
requires every `Provenance` field to be GATED (read by `sweep`, `cross_arch`
or `canary`) or RECORDED (consulted by nobody, deliberately). TF32 belongs to
the accuracy path, which none of those gate, so it could only be RECORDED --
a field written correctly and read by nothing, which is instance 10 in
`docs/silent_failure_patterns.md` and the reason that test exists.

Refusing at the point the label is applied is strictly stronger: a row
claiming `dense_softmax_fp32` **cannot be produced** under TF32, so the
existing `score_source` column becomes the record and no new column has to be
trusted. Enforcement where the claim is made, not a note beside it.
"""

from __future__ import annotations

import torch

# Score sources whose name asserts a float32 matmul. Adding a scorer that
# claims fp32 means adding it here; a scorer that does not claim it is not
# constrained, which is why this is a set rather than a blanket rule.
FP32_SCORE_SOURCES = frozenset({"dense_softmax_fp32"})


class PrecisionNotEnforced(RuntimeError):
    """A float32 claim the current torch settings do not deliver."""


def fp32_matmul_state() -> dict:
    """What torch will actually do with an fp32 matmul right now."""
    return {
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def enforce_fp32_matmul() -> dict:
    """Pin fp32 matmuls to real fp32, and return the resulting state.

    Idempotent, and safe on CPU-only machines -- these are plain globals and
    setting them without CUDA present is a no-op that still makes the
    intended state explicit rather than inherited.

    `cudnn.benchmark` is pinned False alongside the precision flags. It is a
    different property -- autotuning, not mantissa width -- but it is the
    other global whose default could change a measurement between torch
    versions without any code changing, and Stage 2/5 time what it selects.
    Its default is already False; pinning it costs nothing and removes it
    from the list of things a version bump could move.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    return fp32_matmul_state()


def assert_fp32_matmul(score_source: str) -> None:
    """Raise unless `score_source`'s float32 claim is actually deliverable.

    Called where the claim is applied -- once per example in
    `SwappableAttentionModel.compute_importance_scores` -- so the cost is
    nothing against an 11.8 s scoring pass and no row can be written under a
    setting that contradicts its own `score_source`.

    A scorer not in `FP32_SCORE_SOURCES` makes no float32 claim and is not
    constrained: `minference_meanpool` pools in fp32 too, but its name does
    not assert the precision, and widening this to every scorer would turn a
    guard into a global policy that a future cheap-and-deliberately-lower-
    precision estimator would have to fight.
    """
    if score_source not in FP32_SCORE_SOURCES:
        return
    state = fp32_matmul_state()
    bad = []
    if state["allow_tf32_matmul"]:
        bad.append("torch.backends.cuda.matmul.allow_tf32=True")
    if state["float32_matmul_precision"] != "highest":
        bad.append(f"torch.get_float32_matmul_precision()="
                   f"{state['float32_matmul_precision']!r}")
    if not bad:
        return
    raise PrecisionNotEnforced(
        f"score_source={score_source!r} claims a float32 matmul, but "
        f"{', '.join(bad)} -- torch may run the oracle's QK^T with a 10-bit "
        f"mantissa. Every row this run writes would carry a precision label "
        f"the computation did not honour.\n\n"
        f"Call attnbench.numerics.enforce_fp32_matmul() before scoring. It is "
        f"already called by the result-producing scripts; if you are reaching "
        f"this from a notebook or a new entry point, that is the missing line."
    )
