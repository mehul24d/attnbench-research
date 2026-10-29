"""The lock-list checker (T4.1, Gate B).

Gate B: "All **[lock]** values fixed, two adversarial reads done, lock commit
tagged." T4.1 names the values: "Fix every **[lock]** value: H0 to H3
tolerances, H6 regret bound, ...". This module reads that list from the plan
itself, so the checklist cannot drift from the plan, and checks a lock file
against it.

**The lock file** is JSON: `{"plan_sha256": ..., "values": {key: entry}}`,
one entry per T4.1 item, keyed by `item_key` of the item's text. An entry is
`{"value": ..., "source": ...}`: the value, and where it came from (a fit, a
banked file, a decision in Appendix A). A value that is null, an empty
string or an empty list is not fixed. `plan_sha256` is the sha256 of the
plan the file was written against; a lock file checked against another plan
is reported, because the list it answers may not be the list the plan now
holds.

**Every "at lock" phrase.** Other sentences of the plan also fix a value
"at lock". `lock_mentions` lists every sentence outside T4.1 that says "at
lock" or carries **[lock]**, with its line. `tests/test_lock_checklist.py`
maps each one to the T4.1 item that covers it, and fails on a sentence the
map does not cover, so a value added to the plan cannot escape the list.
The researcher's decision (28 Oct 2026) is that T4.1 names every such value;
the plan edit that adds the ones it lacks is the integration session's.

**The pairings each hypothesis is scored on.** H0 and H2 compare components
only within the same sequence under the strict definition
(`op_sequence.same_sequence`), so which pairings they are scored on follows
from that classification. The lock file states them, `{"scored_pairings":
{"H0": ["P1", ...], "H2": [...], ...}}`, the report prints them, and a lock
without them for H0 and H2 is not complete (audit C4, finding 3).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

PLAN = Path(__file__).resolve().parents[2] / "docs" / "RESEARCH_PLAN.md"
# Hypotheses whose scored pairings the lock must state (module docstring).
SCORED_REQUIRED = ("H0", "H2")
_HYPOTHESIS = re.compile(r"H\d+")
_PAIRING = re.compile(r"P\d+")
ROW_PREFIX = "| T4.1 |"
LEAD = "Fix every **[lock]** value:"


class LockListError(ValueError):
    """A plan or lock file the checker cannot read as it requires."""


def item_key(text: str) -> str:
    """A stable key for a T4.1 item: its text without parentheticals or a
    leading article, lower case, words joined by underscores."""
    t = re.sub(r"\([^)]*\)", " ", text)
    t = re.sub(r"^\s*the\s+", "", t, flags=re.I)
    words = re.findall(r"[a-z0-9]+", t.lower())
    if not words:
        raise LockListError(f"no key in item {text!r}")
    return "_".join(words)


def _split_top_level(body: str) -> list[str]:
    items, depth, cur = [], 0, ""
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            items.append(cur.strip())
            cur = ""
        else:
            cur += ch
    items.append(cur.strip())
    return [i for i in items if i]


def lock_items(plan_text: str) -> dict[str, str]:
    """{key: item text} for every value T4.1 lists, in the plan's order."""
    rows = [l for l in plan_text.splitlines() if l.startswith(ROW_PREFIX)]
    if len(rows) != 1:
        raise LockListError(f"the plan has {len(rows)} T4.1 rows, expected one")
    cells = rows[0].split("|")
    if len(cells) < 3:
        raise LockListError("the T4.1 row has no task column")
    task = cells[2].strip()
    if not task.startswith(LEAD):
        raise LockListError(f"T4.1's task no longer starts {LEAD!r}: {task[:60]!r}")
    out: dict[str, str] = {}
    for text in _split_top_level(task[len(LEAD):]):
        k = item_key(text)
        if k in out:
            raise LockListError(f"two T4.1 items share the key {k!r}")
        out[k] = text
    return out


def lock_mentions(plan_text: str) -> list[tuple[int, str]]:
    """(line, sentence) for every sentence outside the T4.1 row that says
    "at lock" or carries **[lock]**."""
    pat = re.compile(r"\bat lock\b|\[lock\]", re.I)
    out = []
    for n, line in enumerate(plan_text.splitlines(), 1):
        if line.startswith(ROW_PREFIX):
            continue
        for sentence in re.split(r"(?<=[.;])\s+", line):
            if pat.search(sentence):
                out.append((n, sentence.strip()))
    return out


def _unset(v) -> bool:
    return v is None or (isinstance(v, (str, list, dict)) and len(v) == 0)


@dataclass(frozen=True)
class LockCheck:
    items: dict            # key -> T4.1 text
    missing: tuple         # keys with no entry
    unset: tuple           # keys whose value is not fixed
    unsourced: tuple       # keys whose entry names no source
    unknown: tuple         # entries for keys T4.1 does not list
    plan_matches: bool     # the file's plan_sha256 is this plan's
    scored_pairings: tuple = ()    # (hypothesis, (pairing, ...)), as stated
    scored_problems: tuple = ()    # why the statement is unusable

    @property
    def complete(self) -> bool:
        return not (self.missing or self.unset or self.unsourced or self.unknown
                    or self.scored_problems) and self.plan_matches

    def report(self) -> str:
        lines = [f"{len(self.items)} values listed by T4.1; "
                 f"{'complete' if self.complete else 'NOT complete'}"]
        for name in ("missing", "unset", "unsourced", "unknown"):
            for k in getattr(self, name):
                lines.append(f"  {name}: {k} ({self.items.get(k, 'not in T4.1')})")
        if not self.plan_matches:
            lines.append("  the lock file was written against another version of the plan")
        for h, ps in self.scored_pairings:
            lines.append(f"  scored pairings, {h}: {', '.join(ps)}")
        for p in self.scored_problems:
            lines.append(f"  scored pairings: {p}")
        return "\n".join(lines)


def _scored(stated) -> tuple[tuple, tuple]:
    """(the statement as sorted pairs, its problems)."""
    if not isinstance(stated, Mapping) or not stated:
        return (), (f"none stated; {list(SCORED_REQUIRED)} at least",)
    problems, pairs = [], []
    for h in SCORED_REQUIRED:
        if h not in stated:
            problems.append(f"{h} is not stated")
    for h, ps in sorted(stated.items()):
        if not (isinstance(h, str) and _HYPOTHESIS.fullmatch(h)):
            problems.append(f"{h!r} is not a hypothesis (H0, H1, ...)")
            continue
        if not isinstance(ps, list) or not ps:
            problems.append(f"{h} names no pairing")
            continue
        bad = [p for p in ps if not (isinstance(p, str) and _PAIRING.fullmatch(p))]
        if bad or len(set(ps)) != len(ps):
            problems.append(f"{h}: {ps!r} is not a list of distinct pairings (P1, P2, ...)")
            continue
        pairs.append((h, tuple(ps)))
    return tuple(pairs), tuple(problems)


def plan_sha256(plan_text: str) -> str:
    return hashlib.sha256(plan_text.encode("utf-8")).hexdigest()


def check(lock: Mapping, plan_text: str) -> LockCheck:
    """`lock` is the parsed lock file (module docstring)."""
    items = lock_items(plan_text)
    values = lock.get("values")
    if not isinstance(values, Mapping):
        raise LockListError("the lock file has no `values` mapping")
    missing, unset, unsourced = [], [], []
    for k in items:
        e = values.get(k)
        if e is None:
            missing.append(k)
            continue
        if not isinstance(e, Mapping) or "value" not in e:
            raise LockListError(f"entry {k!r} is not {{'value': ..., 'source': ...}}")
        if _unset(e["value"]):
            unset.append(k)
        if _unset(e.get("source")):
            unsourced.append(k)
    unknown = sorted(set(values) - set(items))
    scored, scored_problems = _scored(lock.get("scored_pairings"))
    return LockCheck(items=items, missing=tuple(missing), unset=tuple(unset),
                     unsourced=tuple(unsourced), unknown=tuple(unknown),
                     plan_matches=lock.get("plan_sha256") == plan_sha256(plan_text),
                     scored_pairings=scored, scored_problems=scored_problems)


def skeleton(plan_text: str) -> dict:
    """A lock file with every T4.1 item present and nothing fixed."""
    return {"plan_sha256": plan_sha256(plan_text),
            "values": {k: {"item": text, "value": None, "source": None}
                       for k, text in lock_items(plan_text).items()},
            "scored_pairings": {h: [] for h in SCORED_REQUIRED}}


def check_file(path, plan: Optional[Path] = None) -> LockCheck:
    plan_text = Path(plan or PLAN).read_text(encoding="utf-8")
    try:
        lock = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise LockListError(f"{path} is not JSON: {e}") from e
    return check(lock, plan_text)
