"""Split firewall and matched row budgets shared by live code.

These definitions were in `attnbench/analysis/frontier_prereg.py`, the
estimator-frontier pre-registration as code. That study was superseded, and
on 21 Oct 2026 (T0.16) its pre-registration, the rest of that module and its
budget gate moved to `docs/archive/`. The names below are used by code that
stays, so they moved here unchanged:

- the T1.1 calibration script (`scripts/calibrate_xattn_thresholds.py`,
  amendment A1 and gate G12): `SplitLeak`, `check_splits_disjoint`;
- the study's split generator (`accuracy/frontier_splits.py`) and the recall
  rows (`analysis/frontier_recall.py`): `SPLIT_INDEX`, `SplitLeak`,
  `check_splits_disjoint`;
- the recall pass (`accuracy/recall.py`) and the per-head selector's tests:
  the matched row budgets.

Section numbers in the docstrings ("sec. 5", "sec. 3, H7") are those of
`docs/archive/estimator_frontier_preregistration.md`.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from ..masks import era_budget

# --------------------------------------------------------------------------
# Matched row budgets (the per-head era-4 rule at b = 128, and rows at
# smaller block sizes matched to their parent row's realised density)
# --------------------------------------------------------------------------

H7_DENSITY_TOL = 0.01
H7_MIN_PARENT_ROW = 7           # b = 128 rows scored by H7: p >= 7 (tokens >= 896)


def kept_128(p: int, d_nom: float) -> int:
    """Blocks kept in row p at b = 128 under the era-4 rule (sec. 2.1): the
    sink and the diagonal free, plus the budget over the p - 1 candidates,
    computed as the selector computes it (`masks.era_budget`)."""
    if p < 2:
        return p + 1
    return 2 + era_budget(d_nom, p - 1)


def matched_budget(i: int, b: int, d_nom: float) -> int:
    """The budget of row i at block size b < 128 that matches the realised
    density of its parent row p = floor(i * b / 128) at b = 128 (sec. 3, H7).
    The two free blocks are still granted, outside this budget."""
    if i < 2:
        return 0
    p = i * b // 128
    target = kept_128(p, d_nom) / (p + 1)
    return min(max(round(target * (i + 1)) - 2, 0), i - 1)


def matched_row_budgets(n: int, b: int, d_nom: float) -> list:
    """`matched_budget` for rows 0..n-1, as the selector's `row_budgets`."""
    return [matched_budget(i, b, d_nom) for i in range(n)]


def row_density_matches(i: int, b: int, d_nom: float) -> bool:
    """The per-row check: row i's realised density at b is within 0.01 of its
    parent row's at b = 128."""
    p = i * b // 128
    target = kept_128(p, d_nom) / (p + 1)
    got = (min(i + 1, 2) + matched_budget(i, b, d_nom)) / (i + 1)
    return abs(got - target) <= H7_DENSITY_TOL


def h7_rows(n_tokens: int, b: int) -> range:
    """The rows H7 scores at block size b: those whose parent row at b = 128
    is p >= 7. Below that, a row at b = 16 has too few blocks to match its
    parent's density within 0.01."""
    return range(H7_MIN_PARENT_ROW * 128 // b, n_tokens // b)


# --------------------------------------------------------------------------
# Splits (sec. 5)
# --------------------------------------------------------------------------

class SplitLeak(ValueError):
    """Two splits share an example by any of the four identities."""


SPLIT_KEYS = ("example_id", "context_sha256", "qa_question", "niah_needles")


def check_splits_disjoint(splits: Mapping[str, Sequence[Mapping]]) -> None:
    """Every pair of splits disjoint by example_id, context sha256, QA
    question index AND gold-document set (as `qa_question`, a hashable
    (index, docs) or just the index), and NIAH (key, value) pairs. A shared
    `qa_1` question under another seed and context is a leak."""
    names = sorted(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for key in SPLIT_KEYS:
                sa = _identities(splits[a], key)
                sb = _identities(splits[b], key)
                both = sa & sb
                if both:
                    raise SplitLeak(f"{a} and {b} share {key}: {sorted(map(str, both))[:3]}")


def _identities(rows, key):
    out = set()
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        if key == "qa_question":
            # The index alone decides: the same SQuAD question is a leak
            # whatever its distractors (ruler.py: `index=i`).
            out.add(v[0] if isinstance(v, tuple) else v)
        elif key == "niah_needles":
            out.update(tuple(p) for p in v)
        else:
            out.add(v)
    return out


# The evaluation split uses indices no banked file has used (sec. 5). The
# registry is every (task, band) -> (first, last) example index found in a
# banked parquet on 2026-10-03; `tests/test_frontier_eval_ids.py` re-derives
# it from the parquets and fails if they disagree.
BANKED_INDEX_RANGES = {
    ("niah_multikey", 2048): (0, 299), ("niah_multikey", 4096): (0, 299),
    ("niah_multikey", 8192): (0, 299), ("niah_multikey", 16384): (0, 99),
    ("niah_multikey_1", 16384): (0, 4), ("niah_multikey_1", 32768): (0, 4),
    ("niah_multiquery", 16384): (0, 49), ("niah_multiquery", 32768): (0, 49),
    ("niah_multivalue", 16384): (0, 49), ("niah_multivalue", 32768): (0, 49),
    ("niah_single", 2048): (0, 299), ("niah_single", 4096): (0, 299),
    ("niah_single", 8192): (0, 299), ("niah_single", 16384): (0, 99),
    ("niah_single", 32768): (0, 49),
    ("qa_1", 16384): (0, 99), ("qa_1", 32768): (0, 99),
    ("qa_2", 16384): (0, 4), ("qa_2", 32768): (0, 4),
    ("vt", 2048): (0, 299), ("vt", 4096): (0, 299), ("vt", 8192): (0, 299),
    ("vt", 16384): (0, 99),
}


# split -> (first index, how many), for every task. Seeds: sec. 5.
SPLIT_INDEX = {"evaluation": (3000, 300), "calibration": (2000, 48),
               "selection": (1000, 32), "t4_replication": (0, 100)}


def evaluation_indices(n: int) -> range:
    first, most = SPLIT_INDEX["evaluation"]
    if not 0 < n <= most:
        raise ValueError(f"evaluation n must be 1..{most}, got {n}")
    return range(first, first + n)
