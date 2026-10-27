"""The two naive baselines of plan Section 5.4 (T2.19).

  - **Constant host cost**: "the recursion with every host cost replaced by
    one per-host constant (the median null-kernel launch cost)". Every
    kernel's and copy's h_i becomes the constant, host-side compute such as
    CPU mask building included, since that is part of c_host (Section 5.1).
    A synchronisation keeps h_i = 0 (its host cost is s_fixed, Appendix A,
    22 Oct 2026) and its measured s_0. Link and device costs are unchanged,
    so the gap to the recursion is what per-operation host costs add.
  - **Nearest measured pairing**: "the measured end-to-end time of the most
    similar pairing already run (same GPU if one exists, otherwise same host
    platform), unscaled". "Already run" is the pairings banked before the
    prediction, passed in as for the fitted baselines (training.py). A tie
    between two equally near pairings is refused: the plan gives no rule
    for it, and none arises in the planned run order.

`PAIRINGS` is Section 5.2's table; tests/test_predict_naive.py reads it
back from the plan. "Same platform is not the same SKU" (Section 5.2), so
P6's "AMD EPYC 3rd gen (7R13)" is the platform "AMD EPYC 3rd gen".
"""

from __future__ import annotations

import dataclasses
import math
import statistics
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

from .recursion import CostModelError, PricedSequence, Timeline, recurse
from .training import CellKey, GroundTruthRow, statistic_of


@dataclass(frozen=True)
class Pairing:
    name: str
    host_platform: str
    gpu: str


PAIRINGS = {
    "P1": Pairing("P1", "Intel Cascade Lake", "L4"),
    "P2": Pairing("P2", "Intel Cascade Lake", "A100"),
    "P4": Pairing("P4", "Intel Sapphire Rapids", "H100"),
    "P3": Pairing("P3", "AMD EPYC 3rd gen", "L4"),
    "P5": Pairing("P5", "AMD EPYC 3rd gen", "L40S"),
    "P6": Pairing("P6", "AMD EPYC 3rd gen", "H100"),
    "P7": Pairing("P7", "AMD EPYC 2nd gen", "A10G"),
}


# --------------------------------------------------------------------------
# Constant host cost
# --------------------------------------------------------------------------

def median_launch_cost(samples: Iterable[float]) -> float:
    """The constant: the median of the null-kernel launch-cost samples
    (T2.2), in seconds."""
    xs = [float(x) for x in samples]
    if not xs:
        raise CostModelError("no null-kernel launch samples")
    if any(not math.isfinite(x) or x <= 0 for x in xs):
        raise CostModelError("launch-cost samples must be positive and finite")
    return float(statistics.median(xs))


def with_constant_host_cost(priced: PricedSequence,
                            constant: float) -> PricedSequence:
    """The same components with every non-synchronising h_i = `constant`."""
    if isinstance(constant, bool) or not isinstance(constant, (int, float)) \
            or not math.isfinite(constant) or constant <= 0:
        raise CostModelError(
            f"the constant host cost must be a positive finite number of "
            f"seconds, got {constant!r}")
    host = tuple(0.0 if s else float(constant) for s in priced.is_sync)
    return dataclasses.replace(priced, host=host)


def constant_host_cost(priced: PricedSequence, constant: float,
                       queue_depth: Optional[int]) -> Timeline:
    """The recursion over `with_constant_host_cost`."""
    return recurse(with_constant_host_cost(priced, constant), queue_depth)


# --------------------------------------------------------------------------
# Nearest measured pairing
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Nearest:
    target: str
    source: Optional[str]     # None: no banked pairing shares GPU or host
    rule: str                 # "same GPU", "same host platform" or "none"


def nearest_pairing(target: str, banked: Iterable[str],
                    pairings: Mapping[str, Pairing] = PAIRINGS) -> Nearest:
    """The most similar pairing banked before `target`."""
    banked = sorted(set(banked))
    if target in banked:
        raise CostModelError(
            f"{target} is listed as banked before its own prediction")
    unknown = [p for p in [target, *banked] if p not in pairings]
    if unknown:
        raise CostModelError(f"unknown pairing(s) {unknown}")
    t = pairings[target]
    for rule, same in (("same GPU", lambda p: p.gpu == t.gpu),
                       ("same host platform",
                        lambda p: p.host_platform == t.host_platform)):
        hits = [b for b in banked if same(pairings[b])]
        if len(hits) > 1:
            raise CostModelError(
                f"{hits} are equally near {target} ({rule}); the plan names no "
                f"tie-break, so the caller must choose")
        if hits:
            return Nearest(target=target, source=hits[0], rule=rule)
    return Nearest(target=target, source=None, rule="none")


def _without_code_path(k: CellKey) -> tuple:
    return (k.model, k.arm, k.density, k.context_length, k.batch)


def nearest_pairing_predictions(target: str, keys: Sequence[CellKey],
                                rows: Iterable[GroundTruthRow],
                                banked: Iterable[str],
                                pairings: Mapping[str, Pairing] = PAIRINGS, *,
                                source_code_path: Optional[str] = None
                                ) -> tuple[Nearest, dict]:
    """For each cell key, the nearest pairing's measured time at the same
    model, arm, density, length and batch, unscaled; None where that pairing
    has no such cell (or no pairing is near).

    **The code path is not matched.** A code path names a GPU architecture,
    so the nearest pairing's code path is by construction not the target's
    whenever the two differ in GPU (the "same host platform" rule), and an
    exact match would find nothing. The cost is that a source with two code
    paths for one cell (XAttention's Triton and PyTorch paths on one card)
    has two rows for it; that is refused, not averaged, and
    `source_code_path` names the one to read."""
    near = nearest_pairing(target, banked, pairings)
    out: dict = {tuple(k): None for k in keys}
    if near.source is None:
        return near, out
    source_rows = [r for r in rows if r.pairing == near.source]
    if not source_rows:
        raise CostModelError(
            f"{near.source} is banked but has no ground-truth rows here")
    statistic_of(source_rows)
    if source_code_path is not None:
        source_rows = [r for r in source_rows
                       if r.cell.code_path == source_code_path]
        if not source_rows:
            raise CostModelError(
                f"{near.source} has no ground-truth rows on code path "
                f"{source_code_path!r}")
    by_key: dict = {}
    for r in source_rows:
        k = _without_code_path(r.cell.key)
        if k in by_key:
            raise CostModelError(
                f"{near.source} has two rows for cell {k} (on different code "
                f"paths, or twice); name source_code_path")
        by_key[k] = float(r.measured)
    for k in out:
        out[k] = by_key.get(_without_code_path(CellKey(*k)))
    return near, out
