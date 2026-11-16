"""EXPLORATORY: where a cell's recursion error comes from (item 2 of the
16 Nov 2026 rulings).

For each cell, the committed recursion prediction is recomputed from its
priced sequence, then recomputed again with the traced device time put in
place of the priced device time, one model component at a time and then for
all of them. The change in error when a component's device time is replaced
is that component's device error; what remains with every device time
replaced is the host and launch path (host costs, synchronisation costs and
the queue), which the traces cannot measure untraced and so is reported as
one residual.

Traced device time is inflated at short lengths (about 7 us per kernel at
2048 tokens on P2), so a component's device error there is an upper bound;
the cells it matters for are host-bound and the recursion barely moves.
"""

from __future__ import annotations

import dataclasses
import statistics
from typing import Mapping, Optional, Sequence

from attnbench.predict.recursion import PricedSequence, recurse

COMPONENTS = ("attention", "gemm", "other_device", "copy")


def component_of(family: str) -> Optional[str]:
    """The model component an operation family belongs to; None for a
    synchronisation (it has no device time to replace)."""
    kind, _, name = family.partition(":")
    if kind == "sync":
        return None
    if kind == "copy":
        return "copy"
    if "attn" in name or "attention" in name:
        return "attention"
    if name in ("aten::mm", "aten::addmm", "aten::matmul", "aten::linear"):
        return "gemm"     # memset:aten::mm (cuBLAS workspace) included
    return "other_device"


def substitute(priced: PricedSequence, traced: Sequence[Optional[float]],
               families: Sequence[str], only: Optional[str] = None) -> PricedSequence:
    """`priced` with its device (or, for a copy, link) time replaced by the
    traced time at every measured position of component `only` (None: every
    component)."""
    if not (len(traced) == len(families) == len(priced)):
        raise ValueError(f"lengths differ: {len(traced)} traced, {len(families)} "
                         f"families, {len(priced)} priced")
    dev, link = list(priced.device), list(priced.link)
    for i, (t, f) in enumerate(zip(traced, families)):
        comp = component_of(f)
        if t is None or comp is None or (only is not None and comp != only):
            continue
        if comp == "copy":
            link[i] = t
        else:
            dev[i] = t
    return dataclasses.replace(priced, device=tuple(dev), link=tuple(link))


def device_sums(priced: PricedSequence, traced: Sequence[Optional[float]],
                families: Sequence[str]) -> dict[str, tuple[float, float]]:
    """component -> (priced, traced) device seconds over measured positions."""
    out = {c: [0.0, 0.0] for c in COMPONENTS}
    for i, (t, f) in enumerate(zip(traced, families)):
        comp = component_of(f)
        if t is None or comp is None:
            continue
        out[comp][0] += priced.link[i] if comp == "copy" else priced.device[i]
        out[comp][1] += t
    return {c: (a, b) for c, (a, b) in out.items()}


@dataclasses.dataclass(frozen=True)
class CellSplit:
    measured: float
    predicted: float                  # the recursion on the priced sequence
    replaced: Mapping[str, float]     # component -> recursion with it replaced
    all_replaced: float               # recursion with every device time replaced
    device: Mapping[str, tuple[float, float]]

    def rel(self, t: float) -> float:
        return t / self.measured - 1.0


def split(priced: PricedSequence, queue_depth: int, measured: float,
          traced: Sequence[Optional[float]], families: Sequence[str]) -> CellSplit:
    if measured <= 0:
        raise ValueError("measured time must be positive")
    return CellSplit(
        measured=measured,
        predicted=recurse(priced, queue_depth).end_to_end,
        replaced={c: recurse(substitute(priced, traced, families, c), queue_depth).end_to_end
                  for c in COMPONENTS},
        all_replaced=recurse(substitute(priced, traced, families), queue_depth).end_to_end,
        device=device_sums(priced, traced, families))


def median_abs_rel(pairs: Sequence[tuple[float, float]]) -> float:
    """Median |predicted / measured - 1| over (predicted, measured) pairs."""
    if not pairs:
        raise ValueError("no cells")
    return statistics.median(abs(p / m - 1.0) for p, m in pairs)


def summary(splits: Sequence[CellSplit]) -> dict:
    """The median absolute relative error as committed, with each component's
    device time replaced, and with all of them replaced (what is left is the
    host and launch path)."""
    base = median_abs_rel([(s.predicted, s.measured) for s in splits])
    out = {"n": len(splits), "median_abs_rel": base, "replaced": {}}
    for c in COMPONENTS:
        out["replaced"][c] = median_abs_rel([(s.replaced[c], s.measured) for s in splits])
    out["all_device_replaced"] = median_abs_rel([(s.all_replaced, s.measured) for s in splits])
    return out
