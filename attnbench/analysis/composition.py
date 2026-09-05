"""Refusing to average across cells that aren't comparable.

**The incident this exists for.** On 2026-09-05, two findings came out of the
A100 session that were entirely artefacts of aggregation:

  - "GLA's cost FALLS from 8192 to 16384" -- physically impossible for an
    attention kernel, and it happened because the 16384 band had lost its
    largest batches to OOM. The surviving cells were cheaper, so the median
    over the band dropped even though every individual cell got more
    expensive.
  - "fa2 is 15x faster" -- same mechanism, opposite sign.

Both were Simpson's paradox: a median taken over a set of cells whose
*composition* differs between the groups being compared. Neither was a bug in
any measurement. Every latency in the dataset was correct. The arithmetic that
combined them was the whole error, and the result looked like a finding rather
than like a mistake, which is why it nearly shipped.

**Why this is structurally dangerous in this study specifically.** The
attrition that changes composition is OOM, OOM correlates with sequence length
and batch size, and sequence length is the independent variable in most of the
study's claims. So the composition is not merely *possibly* unbalanced -- it is
unbalanced *as a function of the thing being measured*, which is the exact
condition under which marginal aggregates invert. Any long-context table in
this project is in scope by default, not by exception.

**What this module does about it.** `aggregate()` is the only aggregation
helper in the analysis layer, and it cannot return a bare number:

  1. It refuses, by default, to aggregate across groups whose facet
     composition differs at all.
  2. When a caller explicitly accepts a mismatch, the composition is still
     attached to every output row -- the "at minimum, report it" fallback --
     so a reader of the resulting table can see what the number is an average
     over without going back to the raw data.
  3. `matched_subset()` provides the actual repair rather than only the
     complaint: restrict to the facet levels present in every group, then
     aggregate that. A guard offering no alternative gets bypassed under time
     pressure; this one hands you the correct call.

The refusal names groups and levels concretely, in the style of
`cross_arch.CrossArchError`, because a guard that says only "composition
differs" gets disabled rather than investigated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence

import pandas as pd

# `batch` is the default facet because it is what actually varied under the
# 2026-09-05 artefacts, but nothing here is batch-specific -- OOM attrition
# also removes head counts and sequence lengths, and the same inversion
# follows. Pass `facet=` for those.
DEFAULT_FACET = "batch"

HOW: dict[str, Callable[[pd.Series], float]] = {
    "median": lambda s: float(s.median()),
    "mean": lambda s: float(s.mean()),
    "min": lambda s: float(s.min()),
    "max": lambda s: float(s.max()),
}


class _Missing:
    """Canonical stand-in for a NaN/None facet level.

    NaN is not equal to itself, so a NaN dict key cannot be looked up, a NaN
    set member does not intersect with another group's NaN, and `isin([nan])`
    is False for NaN. Every one of those failures is SILENT and each pushes
    the same direction: it makes two groups look less alike than they are, so
    the guard over-refuses and, worse, `matched_subset` drops rows that are
    genuinely shared. Concretely: `sparsity` is NaN on every dense row, so a
    dense-vs-sparse comparison faceted on sparsity hits all three at once.

    Normalising to one sentinel at the point of counting makes ordinary
    equality correct everywhere downstream.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "<missing>"

    def __lt__(self, other) -> bool:      # sorts last among mixed levels
        return False


MISSING = _Missing()


def _norm_level(v):
    """NaN, None and pd.NA all collapse to MISSING; everything else passes."""
    if v is None or v is MISSING:
        return MISSING
    try:
        if v != v:                         # NaN, and pd.NA raises instead
            return MISSING
    except (TypeError, ValueError):
        return MISSING
    return v


class CompositionMismatch(RuntimeError):
    """An aggregate that would compare differently-composed groups."""


@dataclass(frozen=True)
class Composition:
    """How many rows each facet level contributed to one group.

    Counts, not just the level set: two groups can both cover batches
    {1, 2, 4, 8} and still be incomparable if one has ten rows at batch 1 and
    one row at batch 8 while the other is the reverse. The level set catches
    attrition; the weights catch imbalance. Both invert a median.
    """

    facet: str
    counts: dict[Any, int]

    @property
    def n(self) -> int:
        return sum(self.counts.values())

    @property
    def levels(self) -> list:
        """Facet values present, sorted, MISSING last.

        A missing facet value is itself a composition fact worth reporting,
        not a reason to abort the report -- hence a sort that tolerates it
        rather than raising on the mixed comparison."""
        return sorted(self.counts, key=lambda v: (v is MISSING, 0 if v is MISSING else v))

    @property
    def weights(self) -> dict[Any, float]:
        n = self.n
        return {k: v / n for k, v in self.counts.items()} if n else {}

    def distance(self, other: "Composition") -> float:
        """Total-variation distance between the two facet distributions.

        0.0 means identical proportions; 1.0 means disjoint level sets. TVD
        rather than a set comparison because it degrades smoothly: dropping
        one row out of a hundred should not read the same as losing every
        large batch, and a hard set test cannot tell those apart.
        """
        a, b = self.weights, other.weights
        keys = set(a) | set(b)
        return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)

    def missing_versus(self, other: "Composition") -> list:
        """Levels `other` has that this group does not.

        Reported separately from `distance` because it is the attrition
        signature specifically -- a level present in one group and absent in
        another is what OOM does, and it is strictly worse than reweighting:
        no amount of the surviving data reconstructs the missing cell.
        """
        return sorted((set(other.counts) - set(self.counts)),
                      key=lambda v: (v is MISSING, 0 if v is MISSING else v))

    def describe(self) -> str:
        return "{" + ", ".join(f"{lvl}:{self.counts[lvl]}" for lvl in self.levels) + "}"


def composition(df: pd.DataFrame, facet: str = DEFAULT_FACET) -> Composition:
    """Facet-level row counts for one group of rows."""
    if facet not in df.columns:
        raise CompositionMismatch(
            f"no {facet!r} column, so this frame's composition cannot be "
            f"checked. Aggregating it would be exactly the unguarded median "
            f"this module exists to prevent -- pass the correct facet= or add "
            f"the column."
        )
    counts: dict = {}
    for k, v in df[facet].value_counts(dropna=False).to_dict().items():
        key = _norm_level(k)
        counts[key] = counts.get(key, 0) + int(v)
    return Composition(facet=facet, counts=counts)


def _group_keys(df: pd.DataFrame, group_by: Sequence[str]) -> list[str]:
    missing = [c for c in group_by if c not in df.columns]
    if missing:
        raise CompositionMismatch(f"group_by column(s) {missing} not in the frame")
    return list(group_by)


def compositions_by_group(df: pd.DataFrame, *, group_by: str | Sequence[str],
                          facet: str = DEFAULT_FACET,
                          ) -> dict[Any, Composition]:
    """One `Composition` per group, keyed by the group's value(s)."""
    keys = _group_keys(df, [group_by] if isinstance(group_by, str) else group_by)
    out: dict[Any, Composition] = {}
    for value, group in df.groupby(keys[0] if len(keys) == 1 else keys, dropna=False):
        out[value] = composition(group, facet)
    return out


def composition_report(df: pd.DataFrame, *, group_by: str | Sequence[str],
                       facet: str = DEFAULT_FACET, max_lines: int = 12) -> str:
    """Human-readable per-group composition plus the worst pairwise gap.

    Printed inside the refusal and available on its own, because the question
    a reader has on seeing "composition differs" is always *how*, and an
    exception that does not answer it gets suppressed rather than fixed.
    """
    comps = compositions_by_group(df, group_by=group_by, facet=facet)
    lines = [f"  {k!r}: n={c.n} {facet}={c.describe()}"
             for k, c in list(comps.items())[:max_lines]]
    if len(comps) > max_lines:
        lines.append(f"  ... and {len(comps) - max_lines} more group(s)")
    worst = max_pairwise_distance(comps)
    if worst is not None:
        (ka, kb), dist = worst
        lines.append(f"  worst pair: {ka!r} vs {kb!r}, TVD={dist:.3f}")
        gone = comps[ka].missing_versus(comps[kb]) + comps[kb].missing_versus(comps[ka])
        if gone:
            lines.append(f"  {facet} level(s) present in one group and not the "
                         f"other: {gone}")
    return "\n".join(lines)


def max_pairwise_distance(comps: dict[Any, Composition],
                          ) -> Optional[tuple[tuple[Any, Any], float]]:
    """Worst ((group_a, group_b), TVD) over all pairs, or None for <2 groups.

    The worst pair, not the average: an aggregate is invalidated by its single
    most mismatched comparison, and averaging the mismatch would let one bad
    pair hide behind many good ones -- the same averaging error the module is
    about, applied to its own diagnostic.
    """
    keys = list(comps)
    if len(keys) < 2:
        return None
    worst: Optional[tuple[tuple[Any, Any], float]] = None
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            d = comps[ka].distance(comps[kb])
            if worst is None or d > worst[1]:
                worst = ((ka, kb), d)
    return worst


def check_composition(df: pd.DataFrame, *, group_by: str | Sequence[str],
                      facet: str = DEFAULT_FACET,
                      max_distance: float = 0.0) -> None:
    """Raise unless every pair of groups is composed within `max_distance`.

    Default 0.0 -- identical composition -- because that is the only setting
    under which a cross-group aggregate needs no caveat, and a guard whose
    default is "some slack" ends up documenting the slack rather than the
    result. Loosen it deliberately at the call site, where a reader can see it.
    """
    comps = compositions_by_group(df, group_by=group_by, facet=facet)
    worst = max_pairwise_distance(comps)
    if worst is None or worst[1] <= max_distance:
        return
    (ka, kb), dist = worst
    raise CompositionMismatch(
        f"cannot aggregate across {group_by!r}: groups {ka!r} and {kb!r} have "
        f"{facet} compositions differing by TVD {dist:.3f} (limit "
        f"{max_distance:.3f}).\n{composition_report(df, group_by=group_by, facet=facet)}"
        f"\n\nA median over differently-composed groups can move in the "
        f"opposite direction to every cell inside it -- on 2026-09-05 this "
        f"produced 'GLA's cost falls with sequence length'. Either restrict to "
        f"the shared cells with matched_subset(), or pass "
        f"on_mismatch='annotate' to accept a marginal number with its "
        f"composition attached."
    )


def matched_subset(df: pd.DataFrame, *, group_by: str | Sequence[str],
                   facet: str = DEFAULT_FACET) -> pd.DataFrame:
    """Rows whose facet level appears in EVERY group -- the repair.

    This is what makes a cross-group aggregate legitimate: after it, each
    group covers the same facet levels, so a difference between groups is a
    difference in the measurement rather than in which cells survived.

    It does not equalise *weights* within a level (a group with two rows at
    batch 8 still counts it twice), so `check_composition` can still refuse
    afterwards. That is deliberate: unequal replication is a real and
    different problem, and silently de-duplicating it here would hide it.
    """
    keys = _group_keys(df, [group_by] if isinstance(group_by, str) else group_by)
    comps = compositions_by_group(df, group_by=keys, facet=facet)
    if not comps:
        return df.iloc[0:0].copy()
    shared = set.intersection(*(set(c.counts) for c in comps.values()))
    if not shared:
        raise CompositionMismatch(
            f"no {facet} level is present in every group of {group_by!r}, so "
            f"there is no matched subset to aggregate. The groups have no cell "
            f"in common:\n{composition_report(df, group_by=keys, facet=facet)}"
        )
    # isin() cannot express "keep the NaN rows" -- NaN never equals itself --
    # so the sentinel is translated back into an explicit isna() term here.
    keep = df[facet].isin([v for v in shared if v is not MISSING])
    if MISSING in shared:
        keep = keep | df[facet].isna()
    return df[keep].copy()


def aggregate(df: pd.DataFrame, *, group_by: str | Sequence[str], value: str,
              facet: str = DEFAULT_FACET, how: str = "median",
              max_distance: float = 0.0,
              on_mismatch: str = "raise") -> pd.DataFrame:
    """Aggregate `value` per group, with composition attached to every row.

    `on_mismatch`:
      - "raise" (default): refuse a differently-composed aggregate outright.
      - "annotate": compute it anyway, and carry the composition and the
        pairwise distance in the output columns so the caveat travels with the
        number instead of living in whoever-ran-it's memory.
      - "match": silently-but-visibly repair by restricting to
        `matched_subset` first, then re-check. Named as an option rather than
        done automatically, because dropping cells changes what the number
        answers and that belongs in the caller's code.

    There is deliberately no mode that returns a bare Series. Every path out
    of this function carries `n`, `facet_levels`, `facet_counts` and
    `composition_tvd`, so a table built from it cannot lose the provenance of
    its own averages -- which is what happened on 2026-09-05.
    """
    if how not in HOW:
        raise CompositionMismatch(f"unknown how={how!r}; expected one of {sorted(HOW)}")
    if value not in df.columns:
        raise CompositionMismatch(f"no {value!r} column to aggregate")
    if on_mismatch not in ("raise", "annotate", "match"):
        raise CompositionMismatch(
            f"unknown on_mismatch={on_mismatch!r}; expected 'raise', "
            f"'annotate' or 'match'")

    keys = _group_keys(df, [group_by] if isinstance(group_by, str) else group_by)

    if on_mismatch == "match":
        df = matched_subset(df, group_by=keys, facet=facet)
    if on_mismatch in ("raise", "match"):
        check_composition(df, group_by=keys, facet=facet, max_distance=max_distance)

    comps = compositions_by_group(df, group_by=keys, facet=facet)
    worst = max_pairwise_distance(comps)
    tvd = worst[1] if worst is not None else 0.0

    rows = []
    for key, group in df.groupby(keys[0] if len(keys) == 1 else keys, dropna=False):
        c = comps[key]
        key_values = (key,) if len(keys) == 1 else key
        rows.append({
            **dict(zip(keys, key_values)),
            value: HOW[how](group[value]),
            "how": how,
            "n": c.n,
            "facet": facet,
            "facet_levels": c.levels,
            "facet_counts": c.describe(),
            # Constant across rows by construction: it is a property of the
            # whole aggregate, and repeating it per row is what keeps it
            # attached when a single row is quoted out of the table.
            "composition_tvd": tvd,
            "composition_matched": tvd <= max_distance,
        })
    return pd.DataFrame(rows)
