"""Refuse to pool latency across rows that decoded through different kernels.

`DENSE_DECODE_BACKEND` changed from `sdpa_math` to `sdpa_flash` on
2026-09-08 (see `grid_configs`). The change is correct and deliberate: the
old value handicapped every sparse arm by 23-64% per decode token on the
phase that dominates the bill, while the dense arm decoded through
`sdpa_flash` all along.

Deliberate is not the same as harmless. Rows from either side of the change
are stamped -- `decode_backend` has been on every row since the schema was
written -- so nothing is lost. What is easy is *averaging across them*: a
`groupby(["backend", "task", "band", "sparsity"]).latency_ms.mean()` over a
mixed set silently returns a number describing neither regime, and it looks
exactly like a number describing both.

That is the failure this module exists to make impossible rather than
remembered. The information was already on every row before the change, and
nothing compared it -- which is how the confound survived to Stage 5 in the
first place (docs/silent_failure_patterns.md #23, #24). Recording a fact and
checking it are different things, and only the second one survives an author
who was not there for the decision.
"""

from __future__ import annotations

import pandas as pd

# The grouping every latency consumer in this project uses. Kept here so the
# guard and the consumers cannot disagree about what "the same cell" means.
OPERATING_POINT_KEYS = ("backend", "task", "_band", "sparsity")


class MixedDecodeBackend(ValueError):
    """Raised when one cell pools rows decoded through different kernels."""


def offending_cells(df: pd.DataFrame,
                    keys: tuple[str, ...] = OPERATING_POINT_KEYS
                    ) -> list[tuple]:
    """Cells whose rows do not agree on `decode_backend`.

    Returns [(key..., sorted_backends)] so a caller can report which cells
    are mixed rather than only that some are.
    """
    if "decode_backend" not in df.columns:
        raise MixedDecodeBackend(
            "rows carry no `decode_backend` column, so it cannot be checked "
            "that they decoded through the same kernel. Rows this old "
            "predate the field and cannot be pooled with rows that have it.")
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise KeyError(f"grouping keys absent from the frame: {missing}")

    bad = []
    for key, rows in df.groupby(list(keys), dropna=False):
        seen = sorted({str(b) for b in rows["decode_backend"].dropna().unique()})
        if len(seen) > 1:
            bad.append((key if isinstance(key, tuple) else (key,), seen))
    return bad


def assert_uniform(df: pd.DataFrame,
                   keys: tuple[str, ...] = OPERATING_POINT_KEYS) -> None:
    """Raise unless every cell decoded through one kernel.

    Call this before any operation that averages `latency_ms` across rows.
    """
    bad = offending_cells(df, keys)
    if not bad:
        return
    lines = [f"  {k}: {seen}" for k, seen in bad]
    raise MixedDecodeBackend(
        f"{len(bad)} cell(s) pool rows decoded through different kernels:\n"
        + "\n".join(lines)
        + "\n\nDENSE_DECODE_BACKEND changed sdpa_math -> sdpa_flash on "
          "2026-09-08. A mean over both regimes describes neither: the gap "
          "is 23-64% per decode token. Filter to one era "
          "(`df[df.decode_backend == ...]`) and say which one the result "
          "is about.")


def era_of(decode_backend: str) -> str:
    """Which side of the change a row belongs to, by its own stamp."""
    from ..accuracy.grid_configs import DENSE_DECODE_BACKEND_HISTORY
    for value, since, until in DENSE_DECODE_BACKEND_HISTORY:
        if value == decode_backend:
            return f"{value} ({since or 'project start'} .. {until or 'current'})"
    return f"{decode_backend} (not a recorded DENSE_DECODE_BACKEND value)"
