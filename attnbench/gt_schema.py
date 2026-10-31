"""The fields every ground-truth row carries (T2.13, T2.20; audit C3,
finding 2), and the check the harness and every reader run on a row.

Plan Section 5.3 step 7: "Mask density and executed density are recorded on
every row." A row also names the context length and batch it was measured
at, so a caller's statement of the cell can be checked against the row, and
its builder kind, so CPU-builder and GPU-builder results are never pooled
(rule 2). The builder kinds and the field's name are task C10's
(`figures/paper_inputs.py`), which reads them from here.

  - context_length, batch: positive integers.
  - mask_density, executed_density: as `densities.py` defines them. A dense
    row has no mask (None) and executes every block (1.0); any other row has
    both in (0, 1].
  - builder_kind: "cpu" or "gpu", where the arm's mask is built. A dense row
    has none (None). A deployable row (inline or in-backend estimator) must
    name one. An oracle or precomputed row may name one or None.

torch-free, so the readers in `figures` can use it.
"""

from __future__ import annotations

import math
from typing import Mapping

BUILDER_KINDS = ("cpu", "gpu")
ROW_FIELDS = ("context_length", "batch", "mask_density", "executed_density", "builder_kind")
DENSE_KIND = "dense"
DEPLOYABLE_ARM_KINDS = ("inline", "in_backend")


class RowFieldError(ValueError):
    """A ground-truth row without the fields every row must carry."""


def _positive_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v > 0


def _fraction(v) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            and 0.0 < float(v) <= 1.0)


def row_problems(row: Mapping) -> list[str]:
    """Why `row` lacks the fields; empty when it has them all."""
    if not isinstance(row, Mapping):
        return [f"a row is a mapping, got {type(row).__name__}"]
    problems = [f"{f} is missing" for f in ("arm_kind",) + ROW_FIELDS if f not in row]
    if problems:
        return problems
    kind = row["arm_kind"]
    for f in ("context_length", "batch"):
        if not _positive_int(row[f]):
            problems.append(f"{f}={row[f]!r} is not a positive integer")
    md, ed, b = row["mask_density"], row["executed_density"], row["builder_kind"]
    if kind == DENSE_KIND:
        if md is not None:
            problems.append(f"a dense row has no mask, but mask_density={md!r}")
        if ed != 1.0 or isinstance(ed, bool):
            problems.append(f"a dense row executes every block, but executed_density={ed!r}")
        if b is not None:
            problems.append(f"a dense row builds no mask, but builder_kind={b!r}")
        return problems
    for f, v in (("mask_density", md), ("executed_density", ed)):
        if not _fraction(v):
            problems.append(f"{f}={v!r} is not in (0, 1]")
    if b is None and kind in DEPLOYABLE_ARM_KINDS:
        problems.append(f"builder_kind is None on a deployable ({kind}) row; name "
                        f"{BUILDER_KINDS[0]!r} or {BUILDER_KINDS[1]!r}")
    elif b is not None and b not in BUILDER_KINDS:
        problems.append(f"builder_kind={b!r} is not one of {BUILDER_KINDS}")
    return problems


def check_row(row: Mapping, name: str = "the row") -> Mapping:
    problems = row_problems(row)
    if problems:
        arm = row.get("arm") if isinstance(row, Mapping) else None
        raise RowFieldError(f"{name} (arm {arm!r}): " + "; ".join(problems))
    return row
