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

Two different failures live here, and the first one does not imply the
second:

- `assert_uniform` -- *within* one arm, are all the rows from one era?
  Groups by `OPERATING_POINT_KEYS`, which includes `backend`.
- `assert_comparable` -- *across* the arms of one comparison, did they all
  decode through the same kernel? Groups by `COMPARISON_KEYS`, which
  deliberately does not.

Only the second one can see the cross-arm confound (#24), because
`decode_backend` is a function of `backend` and so every group of the first
is uniform by construction. That was a real gap in this module from
2026-09-08 to 2026-09-12: it reported zero offending cells on
`results/stage3_s1b/`, where the sparse arms decoded through `sdpa_math`
and the dense arm through `sdpa_flash` -- the exact confound the module was
written for. A guard whose grouping makes the checked property constant is
not a weak guard, it is not a guard.
"""

from __future__ import annotations

import pandas as pd

# The grouping every latency consumer in this project uses. Kept here so the
# guard and the consumers cannot disagree about what "the same cell" means.
OPERATING_POINT_KEYS = ("backend", "task", "_band", "sparsity")

# The grouping one *comparison* has: every arm whose latency is divided by
# another's at this operating point. `backend` and `sparsity` are what
# distinguish the arms, so neither may be in this key -- see
# `cross_arm_cells`.
COMPARISON_KEYS = ("task", "_band")


class MixedDecodeBackend(ValueError):
    """Raised when one cell pools rows decoded through different kernels."""


def _require_columns(df: pd.DataFrame, keys: tuple[str, ...]) -> None:
    if "decode_backend" not in df.columns:
        raise MixedDecodeBackend(
            "rows carry no `decode_backend` column, so it cannot be checked "
            "that they decoded through the same kernel. Rows this old "
            "predate the field and cannot be pooled with rows that have it.")
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise KeyError(f"grouping keys absent from the frame: {missing}")


def offending_cells(df: pd.DataFrame,
                    keys: tuple[str, ...] = OPERATING_POINT_KEYS
                    ) -> list[tuple]:
    """Cells whose rows do not agree on `decode_backend`.

    Returns [(key..., sorted_backends)] so a caller can report which cells
    are mixed rather than only that some are.
    """
    _require_columns(df, keys)
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


def cross_arm_cells(df: pd.DataFrame,
                    keys: tuple[str, ...] = COMPARISON_KEYS
                    ) -> list[tuple]:
    """Comparison cells whose *arms* did not decode through the same kernel.

    `offending_cells` groups by `OPERATING_POINT_KEYS`, which includes
    `backend`. `decode_backend` is a function of `backend`, so every such
    group is uniform by construction and the check can only ever catch era
    mixing *within one arm*. It is structurally incapable of firing on the
    confound that motivated this module: a sparse arm decoding through
    `sdpa_math` compared against a dense arm decoding through `sdpa_flash`,
    which is exactly the shape of `results/stage3_s1b/`.

    This function drops `backend` (and `sparsity`, which distinguishes the
    arms) from the grouping, so a cell is one *comparison* -- every arm that
    a speedup, a Pareto frontier, or a dominance verdict at that
    (task, band) is computed from. If those arms disagree, the ratio between
    them is partly a kernel difference and not a sparsity effect.

    Returns [(key..., {backend: sorted_decode_backends})].
    """
    _require_columns(df, tuple(keys) + ("backend",))
    bad = []
    for key, rows in df.groupby(list(keys), dropna=False):
        seen = sorted({str(b) for b in rows["decode_backend"].dropna().unique()})
        if len(seen) > 1:
            by_arm = {
                str(arm): sorted({str(b) for b in r["decode_backend"].dropna().unique()})
                for arm, r in rows.groupby("backend", dropna=False)
            }
            bad.append((key if isinstance(key, tuple) else (key,), by_arm))
    return bad


def assert_comparable(df: pd.DataFrame,
                      keys: tuple[str, ...] = COMPARISON_KEYS) -> None:
    """Raise unless every arm of every comparison decoded through one kernel.

    Call this before forming any *ratio* across arms -- a speedup, a Pareto
    frontier, a dominance verdict. `assert_uniform` is the weaker within-arm
    check and does not imply this one.
    """
    bad = cross_arm_cells(df, keys)
    if not bad:
        return
    lines = [f"  {k}: {by_arm}" for k, by_arm in bad]
    raise MixedDecodeBackend(
        f"{len(bad)} comparison cell(s) compare arms that decoded through "
        f"different kernels:\n" + "\n".join(lines)
        + "\n\nA ratio between these arms is part sparsity effect and part "
          "kernel difference, and the two are not separable from the "
          "end-to-end number alone. Either re-measure the arms on one decode "
          "kernel, or route the rows through "
          "`analysis.decode_confound` and report the corrected quantity "
          "with the correction named.")


def era_of(decode_backend: str) -> str:
    """Which side of the change a row belongs to, by its own stamp."""
    from ..accuracy.grid_configs import DENSE_DECODE_BACKEND_HISTORY
    for value, since, until in DENSE_DECODE_BACKEND_HISTORY:
        if value == decode_backend:
            return f"{value} ({since or 'project start'} .. {until or 'current'})"
    return f"{decode_backend} (not a recorded DENSE_DECODE_BACKEND value)"
