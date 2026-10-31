"""The Holm-corrected admissible set (Section 5.7; T4.1, T2.22).

Section 5.7: "Admissibility is a decision input, not a confirmatory claim.
Its Holm family is declared at lock as every admissibility test that feeds
the dispatcher. Correcting shrinks the admissible set, which can move regret
either way, so dispatcher regret is reported with and without the
correction." The uncorrected set is `predict.dispatcher.admissible_at_margin`:
a configuration is admissible when the exact paired lower bound
(`analysis.exact_noninferiority.paired_lower_bound`) at one-sided alpha is
above -margin.

**Holm on an interval test.** The test is a confidence bound, not a p-value.
Its p-value here is the smallest alpha at which the same bound clears
-margin: p = inf{a : paired_lower_bound(b, c, n, a) > -margin/100}, found by
bisection (the bound rises with alpha). Holm's step-down then orders the
family by p and, at step k of m, admits the configuration only if its bound
at alpha / (m - k + 1) clears -margin, stopping at the first that does not.
The decision at each step is the bound itself at the adjusted level, not the
bisected p, so with a family of one this is exactly the uncorrected rule.
p is used only to order the family.

**The family** is the admissible set declared at lock (researcher's
decision, 28 Oct 2026): every configuration that feeds the dispatcher. It is
passed twice, as `declared` (the lock value) and as the keys of `family`
(the counts measured for it), and the two must be the same set: a declared
configuration with no counts, or counts for one outside the declaration, is
refused, so the family cannot grow or shrink after the lock. Dense is
always admissible and is not a member.

**Monotonicity.** The bisection, and so the p-value, rests on the bound
rising with alpha. `tests/test_lock_holm.py` checks it over a grid of counts
and levels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from ..analysis.exact_noninferiority import paired_lower_bound
from ..predict.dispatcher import Config

PRIMARY_ALPHA = 0.025


class HolmError(ValueError):
    """A family or counts the correction cannot be applied to."""


@dataclass(frozen=True)
class Counts:
    """Paired outcomes of one admissibility test: b sparse-right and
    dense-wrong, c dense-right and sparse-wrong, out of n pairs."""

    b: int
    c: int
    n: int

    def __post_init__(self) -> None:
        for name in ("b", "c", "n"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise HolmError(f"{name} must be an int >= 0, got {v!r}")
        if self.n == 0 or self.b + self.c > self.n:
            raise HolmError(f"impossible counts b={self.b}, c={self.c} of n={self.n}")


def _clears(cnt: Counts, alpha: float, margin_pts: float) -> bool:
    return 100 * paired_lower_bound(cnt.b, cnt.c, cnt.n, alpha) > -margin_pts


def ni_p_value(cnt: Counts, margin_pts: float, *, tol: float = 1e-12) -> float:
    """The smallest one-sided alpha at which the exact bound clears -margin
    (1.0 when no alpha below 1 does)."""
    hi = 1.0 - 1e-12
    if not _clears(cnt, hi, margin_pts):
        return 1.0
    lo = 0.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if mid > 0 and _clears(cnt, mid, margin_pts):
            hi = mid
        else:
            lo = mid
    return hi


@dataclass(frozen=True)
class HolmResult:
    alpha: float
    margin_pts: float
    family_size: int
    uncorrected: frozenset         # Config
    holm: frozenset                # Config
    order: tuple                   # (Config, p, adjusted alpha, admitted), Holm's order


def holm_admissible(family: Mapping[Config, Counts], *, declared, margin_pts: float,
                    alpha: float = PRIMARY_ALPHA) -> HolmResult:
    """`declared`: the admissible set declared at lock (a collection of
    Configs); `family`: {Config: Counts} for exactly those configurations."""
    if not family:
        raise HolmError("the Holm family is empty; it is declared at lock")
    if isinstance(declared, (str, bytes, Mapping)) or not hasattr(declared, "__iter__"):
        raise HolmError(f"`declared` is the set of Configs declared at lock, got {declared!r}")
    declared = list(declared)
    if len(set(declared)) != len(declared):
        raise HolmError("a configuration is declared twice")
    unmeasured = sorted(map(str, set(declared) - set(family)))
    undeclared = sorted(map(str, set(family) - set(declared)))
    if unmeasured or undeclared:
        raise HolmError("the family must be the set declared at lock: "
                        f"declared with no counts {unmeasured}, counts not declared {undeclared}")
    if isinstance(margin_pts, bool) or not isinstance(margin_pts, (int, float)) \
            or not math.isfinite(margin_pts) or margin_pts <= 0:
        raise HolmError(f"a margin is a positive number of points, got {margin_pts!r}")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise HolmError(f"alpha must be in (0, 1), got {alpha!r}")
    for k, v in family.items():
        if not isinstance(k, Config):
            raise HolmError(f"family members are dispatcher Configs, got {k!r}")
        if not isinstance(v, Counts):
            raise HolmError(f"{k}: counts are a `Counts`, got {v!r}")
    uncorrected = frozenset(k for k, v in family.items() if _clears(v, alpha, margin_pts))
    m = len(family)
    ranked = sorted(((ni_p_value(v, margin_pts), (k.arm, k.density), k) for k, v in family.items()))
    admitted, order, stopped = set(), [], False
    for i, (p, _, k) in enumerate(ranked):
        level = alpha / (m - i)
        ok = (not stopped) and _clears(family[k], level, margin_pts)
        if ok:
            admitted.add(k)
        else:
            stopped = True
        order.append((k, p, level, ok))
    return HolmResult(alpha=alpha, margin_pts=margin_pts, family_size=m,
                      uncorrected=uncorrected, holm=frozenset(admitted), order=tuple(order))
