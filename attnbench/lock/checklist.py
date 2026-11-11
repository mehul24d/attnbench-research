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
string or an empty list is not fixed. **Typed** (adviser's ruling 1b of 11
Nov 2026; T4.3 read B, finding 4): every item has a type in `SCHEMA`, and a
value is fixed only when it has that type all the way down. A placeholder
string ("TBD", "owed", ...), `False` or `True` where a number is wanted, a
null or an empty container at any depth, and a missing or unexpected key of
a typed mapping are each refused, with the path. A T4.1 item with no type
is refused too, so the plan cannot add an item the checker would read as
any value. `plan_sha256` is the sha256 of the
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
without them for every hypothesis of Section 4 (read from the plan's table)
is not complete (audit C4, finding 3; ruling 1b, 11 Nov 2026). P1 is the
development pairing and is never counted as a test (H1), so P1 in H1's list
is refused; **plan default, flagged:** also in H3's and H6's, which Section
4 scores on held-out pairings only.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

PLAN = Path(__file__).resolve().parents[2] / "docs" / "RESEARCH_PLAN.md"
# Hypotheses whose scored pairings the lock must state (module docstring):
# every row of Section 4's table. `hypotheses(plan_text)` reads them; this is
# the plan's list as of 11 Nov 2026, kept for the skeleton and the report.
SCORED_REQUIRED = ("H0", "H1", "H2", "H3", "H4", "H5", "H6")
# Hypotheses scored on held-out pairings only (module docstring).
NO_P1 = ("H1", "H3", "H6")
PLACEHOLDERS = frozenset({"tbd", "tba", "todo", "owed", "pending", "unset", "unknown",
                          "none", "null", "n/a", "na", "?", "-", "...", "xxx", "fixme"})
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


# --------------------------------------------------------------------------
# Types (module docstring). Each returns the problems with a value at a path.
# --------------------------------------------------------------------------

def _is_placeholder(s: str) -> bool:
    return s.strip().lower().rstrip(".") in PLACEHOLDERS


def num(lo=None, hi=None, *, lo_open=False, hi_open=False, integer=False):
    def check(v, at):
        if isinstance(v, bool) or not isinstance(v, (int, float)) \
                or (integer and not isinstance(v, int)):
            return [f"{at}: {v!r} is not {'an integer' if integer else 'a number'}"]
        if v != v or v in (float("inf"), float("-inf")):
            return [f"{at}: {v!r} is not finite"]
        if lo is not None and (v <= lo if lo_open else v < lo):
            return [f"{at}: {v!r} is {'at or ' if lo_open else ''}below {lo}"]
        if hi is not None and (v >= hi if hi_open else v > hi):
            return [f"{at}: {v!r} is {'at or ' if hi_open else ''}above {hi}"]
        return []
    return check


def text(pattern: Optional[str] = None):
    def check(v, at):
        if not isinstance(v, str) or not v.strip():
            return [f"{at}: {v!r} is not a non-empty string"]
        if _is_placeholder(v):
            return [f"{at}: {v!r} is a placeholder, not a value"]
        if pattern and not re.fullmatch(pattern, v):
            return [f"{at}: {v!r} does not match {pattern}"]
        return []
    return check


def const(value):
    def check(v, at):
        return [] if v == value and type(v) is type(value) else [f"{at}: {v!r} is not {value!r}"]
    return check


def one_of(*types):
    def check(v, at):
        results = [t(v, at) for t in types]
        return [] if any(not r for r in results) else results[0]
    return check


def seq(item, *, min_len=1, increasing=False, distinct=False, length=None):
    def check(v, at):
        if not isinstance(v, list):
            return [f"{at}: {v!r} is not a list"]
        if len(v) < min_len or (length is not None and len(v) != length):
            return [f"{at}: a list of {length or f'at least {min_len}'} items, got {len(v)}"]
        out = [p for i, x in enumerate(v) for p in item(x, f"{at}[{i}]")]
        if not out and increasing and any(b <= a for a, b in zip(v, v[1:])):
            out.append(f"{at}: {v!r} does not increase")
        if not out and distinct and len({json.dumps(x, sort_keys=True) for x in v}) != len(v):
            out.append(f"{at}: {v!r} repeats an item")
        return out
    return check


def record(fields: Mapping, *, optional: Mapping = None):
    """A mapping with exactly `fields` (and any of `optional`), each typed."""
    optional = dict(optional or {})

    def check(v, at):
        if not isinstance(v, Mapping):
            return [f"{at}: {v!r} is not a mapping"]
        out = [f"{at}: missing key {k!r}" for k in fields if k not in v]
        out += [f"{at}: unexpected key {k!r}" for k in v if k not in fields and k not in optional]
        for k, ty in {**fields, **optional}.items():
            if k in v:
                out += ty(v[k], f"{at}.{k}")
        return out
    return check


def table(key: str, value, *, min_len=1):
    """A mapping whose keys match the pattern `key`, every value typed."""
    def check(v, at):
        if not isinstance(v, Mapping):
            return [f"{at}: {v!r} is not a mapping"]
        if len(v) < min_len:
            return [f"{at}: at least {min_len} entries, got {len(v)}"]
        out = []
        for k, x in v.items():
            if not (isinstance(k, str) and re.fullmatch(key, k)):
                out.append(f"{at}: key {k!r} does not match {key}")
            out += value(x, f"{at}.{k}")
        return out
    return check


def scalar():
    """A non-null scalar: a number, a bool, or a non-placeholder string (a
    hyperparameter may be a flag)."""
    def check(v, at):
        if isinstance(v, bool):
            return []
        return one_of(num(), text())(v, at)
    return check


def _family_matches_m(check_):
    def check(v, at):
        out = check_(v, at)
        if not out and v["m"] != len(v["tests"]):
            out.append(f"{at}: m = {v['m']} but {len(v['tests'])} tests are listed")
        return out
    return check


def _ordered_pair(check_):
    def check(v, at):
        out = check_(v, at)
        if not out and not v[0] < v[1]:
            out.append(f"{at}: {v!r} is not an interval lo < hi")
        return out
    return check


FRACTION = num(0, 1, lo_open=True)
POSITIVE = num(0, lo_open=True)
NONNEG = num(0)
SHA256 = text(r"[0-9a-f]{64}")
GIT_COMMIT = text(r"[0-9a-f]{40}")
LABEL = r"[A-Za-z0-9_.:@+-]+"
DENSITY = one_of(num(0, 1, lo_open=True), const("calibrated"))
CELL = r"\d+x\d+"

# One type per T4.1 item (module docstring). The runner reads the items it
# uses from here and nowhere else (`scripts/t31_steps.fill_lock_values`).
SCHEMA: dict = {
    "h0_to_h3_tolerances": record({"H0": FRACTION, "H1": FRACTION, "H2": FRACTION,
                                   "H3": text()}),
    "h6_regret_bound": record({"mean": NONNEG, "worst": NONNEG}),
    "ground_truth_density_set": table(LABEL, seq(DENSITY, distinct=True)),
    "h4_equivalence_bound_with_its_justification_from_the_tasks": POSITIVE,
    "h4_confirmatory_family": _family_matches_m(record(
        {"m": num(1, integer=True), "tests": seq(text(), distinct=True)})),
    "power_target_and_n": record({"alpha": FRACTION, "power": FRACTION,
                                  "n": num(1, integer=True)}),
    "primary_cell_n_and_admissibility_n": record({"primary_cell_n": num(200, integer=True),
                                                  "admissibility_n": num(1, integer=True)}),
    "admissibility_holm_family": seq(record({"arm": text(LABEL), "density": DENSITY}),
                                     distinct=True),
    "context_grid_checked_against_each_model_s_position_limit": record({
        "lengths": seq(num(1, integer=True), increasing=True),
        "batches": table(r"\d+", seq(num(1, integer=True), increasing=True), min_len=0),
        "position_limit": num(1, integer=True)}),
    "delay_injection_slope_bound": _ordered_pair(seq(POSITIVE, length=2)),
    "mask_gate_bound": POSITIVE,
    "cross_card_dense_task_score_bound": POSITIVE,
    "replay_versus_trace_bound_for_gate_a": POSITIVE,
    "null_kernel_floor_multiple": POSITIVE,
    "probe_bounds": table(LABEL, POSITIVE),
    "resolution_and_noise_floors": record({"pairing": const("P1"),
                                           "resolution_floor_ms": table(CELL, NONNEG),
                                           "noise_floor_relative": NONNEG}),
    "floors_quantile": num(0, 1, lo_open=True),
    "hdbi_boundedness_threshold": num(0, 1),
    "learned_predictor_features": seq(text(), distinct=True),
    "learned_predictor_hyperparameters": record({"target": text(),
                                                 "params": table(LABEL, scalar(), min_len=0)}),
    "k2_correctness_tolerance": POSITIVE,
    "head_type_criterion": text(),
    "rerun_budget": record({"max_rerun_rounds": num(0, integer=True),
                            "min_clean_rounds": num(1, integer=True)}),
    "throttle_set": seq(text(LABEL), distinct=True),
    "payload_class_edges": seq(num(0, lo_open=True, integer=True), increasing=True),
    "prompt_set_hash_and_tokenizer_version": record({"prompt_set_sha256": SHA256,
                                                     "tokenizer_version": text()}),
    "realised_length_key_rule": text(),
    "budget_cut_order": seq(text(), distinct=True),
    "compiled_versus_eager_density_flag_tolerance": POSITIVE,
    "probe_length_rule": text(),
    "flagged_memset_sensitivity_bound": NONNEG,
    "chain_check_tolerance": FRACTION,
    "recompile_gate": num(0, integer=True),
    "residual_spread_tolerance": FRACTION,
    "pricing_mismatch_share": FRACTION,
    "block_drift_bound": FRACTION,
    "timed_path_identity_for_p2_and_p4": SHA256,
    "bootstrap_settings": record({"level": FRACTION, "n_boot": num(1000, integer=True),
                                  "seed": num(0, integer=True)}),
    "analysis_commit": GIT_COMMIT,
}


def type_problems(key: str, value) -> list[str]:
    """Why `value` is not a fixed value of T4.1 item `key` ([] when it is)."""
    if key not in SCHEMA:
        return [f"{key}: T4.1 lists it but the checker has no type for it"]
    return SCHEMA[key](value, key)


def hypotheses(plan_text: str) -> tuple:
    """Every hypothesis row of Section 4's table, in order."""
    return tuple(dict.fromkeys(re.findall(r"^\| (H\d+) \|", plan_text, flags=re.M)))


@dataclass(frozen=True)
class LockCheck:
    items: dict            # key -> T4.1 text
    missing: tuple         # keys with no entry
    unset: tuple           # keys whose value is not fixed
    unsourced: tuple       # keys whose entry names no source
    unknown: tuple         # entries for keys T4.1 does not list
    plan_matches: bool     # the file's plan_sha256 is this plan's
    mistyped: tuple = ()           # (key, problem): set, but not of the item's type
    scored_pairings: tuple = ()    # (hypothesis, (pairing, ...)), as stated
    scored_problems: tuple = ()    # why the statement is unusable

    @property
    def complete(self) -> bool:
        return not (self.missing or self.unset or self.unsourced or self.unknown
                    or self.scored_problems or self.mistyped) and self.plan_matches

    def report(self) -> str:
        lines = [f"{len(self.items)} values listed by T4.1; "
                 f"{'complete' if self.complete else 'NOT complete'}"]
        for name in ("missing", "unset", "unsourced", "unknown"):
            for k in getattr(self, name):
                lines.append(f"  {name}: {k} ({self.items.get(k, 'not in T4.1')})")
        for k, why in self.mistyped:
            lines.append(f"  mistyped: {why}")
        if not self.plan_matches:
            lines.append("  the lock file was written against another version of the plan")
        for h, ps in self.scored_pairings:
            lines.append(f"  scored pairings, {h}: {', '.join(ps)}")
        for p in self.scored_problems:
            lines.append(f"  scored pairings: {p}")
        return "\n".join(lines)


def _scored(stated, required=SCORED_REQUIRED) -> tuple[tuple, tuple]:
    """(the statement as sorted pairs, its problems)."""
    if not isinstance(stated, Mapping) or not stated:
        return (), (f"none stated; {list(required)} at least",)
    problems, pairs = [], []
    for h in required:
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
        if h in NO_P1 and "P1" in ps:
            problems.append(f"{h} lists P1, the development pairing, which is never counted "
                            f"as a test of {h}")
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
    missing, unset, unsourced, mistyped = [], [], [], []
    for k in items:
        e = values.get(k)
        if e is None:
            missing.append(k)
            continue
        if not isinstance(e, Mapping) or "value" not in e:
            raise LockListError(f"entry {k!r} is not {{'value': ..., 'source': ...}}")
        if _unset(e["value"]):
            unset.append(k)
        else:
            mistyped += [(k, why) for why in type_problems(k, e["value"])]
        src = e.get("source")
        if _unset(src) or not isinstance(src, str) or _is_placeholder(src):
            unsourced.append(k)
    unknown = sorted(set(values) - set(items))
    scored, scored_problems = _scored(lock.get("scored_pairings"),
                                      hypotheses(plan_text) or SCORED_REQUIRED)
    return LockCheck(items=items, missing=tuple(missing), unset=tuple(unset),
                     unsourced=tuple(unsourced), unknown=tuple(unknown),
                     plan_matches=lock.get("plan_sha256") == plan_sha256(plan_text),
                     scored_pairings=scored, scored_problems=scored_problems,
                     mistyped=tuple(mistyped))


def skeleton(plan_text: str) -> dict:
    """A lock file with every T4.1 item present and nothing fixed."""
    return {"plan_sha256": plan_sha256(plan_text),
            "values": {k: {"item": text, "value": None, "source": None}
                       for k, text in lock_items(plan_text).items()},
            "scored_pairings": {h: [] for h in (hypotheses(plan_text) or SCORED_REQUIRED)}}


def check_file(path, plan: Optional[Path] = None) -> LockCheck:
    plan_text = Path(plan or PLAN).read_text(encoding="utf-8")
    try:
        lock = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise LockListError(f"{path} is not JSON: {e}") from e
    return check(lock, plan_text)
