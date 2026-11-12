"""A pairing's ground truth from its blocks, and the between-block drift
check (researcher's rulings, 7 Nov 2026; `attnbench.lock.blocks`).

A session run in blocks has a session directory per block, each with its own
boot. `collect` reads them: a block whose instance restarted carries
`INTERRUPTED_DISCARD.md` and is discarded alone (its rows are listed as
discarded, never read); a block whose steps did not all pass, or whose
ground truth's late check did not pass, is pooled from nothing and listed
as unpassed (ruling 1d, 11 Nov 2026: the runner's refusal of a ground truth
is final, so its rows are never evidence); the others must be blocks of one
spec, run on one commit, each once. Every ground-truth row carries its block's id and whether
it is the anchor cell, which every block with ground truth measures.

`drift` compares the anchor between blocks: each block's anchor statistic
against the median of the other blocks', as a relative difference (adviser's
ruling 2c of 11 Nov 2026: the reference is the median of the blocks'
anchors, not one block, which could be the odd one out). **Plan default,
flagged:** the median is of the *other* blocks, so that with P4's two anchor
blocks the difference is the gap between them, the quantity the bound was
set from; a median over all blocks would be their midpoint, half the gap,
which widens the bound. With two blocks either may be flagged. Every block it is given must have one anchor row with a finite,
positive statistic; a block without one, or with NaN, is refused, never
passed (T4.3 read A, finding 6). The bound is T4.1's block drift bound (lock
table row 36, `block_drift_bound`), a required argument the caller reads from
the lock (`lock_bound`); the working value until lock is 6.77%
(`WORKING_BOUND`: the largest systematic gap observed between two L4
sessions, T3.2a and T3.2b; adviser's ruling of 8 Nov 2026, replacing the 3%
of 7 Nov 2026). It is a flag, not an exclusion: a block above it is named and
its rows are labelled `DRIFT_LABEL` (`labelled`), every non-interrupted
block stays in primary scoring (`claimable`), and each flagged block gets a
sensitivity run without it (`sensitivity_sets`). The bound is never widened
to pass. The check reads the untraced rows, the ones the hypotheses are
scored on. `pairing_drift` is the post-session step: the blocks collected,
the check run over those that ran ground truth, in the spec's order.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Mapping, Optional, Sequence

from ..lock.blocks import row_key

DISCARD = "INTERRUPTED_DISCARD.md"
STATISTIC = "median_ms"
# T4.1's block drift bound (lock table row 36): the working value until lock.
WORKING_BOUND = 0.0677
DRIFT_LABEL = "drift above bound"


class BlockDriftError(ValueError):
    pass


def _read(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _unpassed(m: Mapping, b: Mapping) -> list:
    """The block's steps whose last attempt did not pass (a dry run skips
    its prerequisites), and the late check if it ran ground truth without one."""
    last = {}
    for a in m.get("attempts") or ():
        last[a["step"]] = a.get("status")
    ok = ("passed", "skipped") if m.get("dry_run") else ("passed",)
    steps = list(b.get("steps") or ())
    out = [s for s in steps if last.get(s) not in ok]
    if "ground_truth" in steps and "late_check" not in steps:
        out.append("late_check (not run)")
    return out


def collect(dirs: Sequence, *, ground_truth_dir: str) -> dict:
    """{"rows", "blocks", "discarded", "unpassed", "spec_sha256", "code_commit"}
    from the blocks' session directories. `ground_truth_dir` is the
    ground-truth step's directory name within each (`t61_session.step_dir`)."""
    rows, blocks, discarded, unpassed, specs, commits = [], [], [], [], set(), set()
    for d in map(Path, dirs):
        m = _read(d / "manifest.json")
        b = (m or {}).get("block")
        if not b:
            raise BlockDriftError(f"{d} is not a block's session")
        if (d / DISCARD).exists():
            discarded.append({"block": b["id"], "dir": str(d), "why": DISCARD})
            continue
        if b["id"] in blocks or b["id"] in [u["block"] for u in unpassed]:
            raise BlockDriftError(f"block {b['id']} is given twice")
        bad = _unpassed(m, b)
        if bad:
            unpassed.append({"block": b["id"], "dir": str(d), "steps": bad})
            continue
        specs.add(b["spec_sha256"])
        commits.add(m.get("code_commit"))
        blocks.append(b["id"])
        gt = _read(d / ground_truth_dir / "ground_truth.json")
        if gt is None:
            continue
        for r in (gt.get("untraced") or {}).get("rows") or ():
            if r.get("block") != b["id"]:
                raise BlockDriftError(f"{d}: a row says block {r.get('block')!r}")
            rows.append(dict(r))
    if len(specs) > 1:
        raise BlockDriftError(f"the blocks are of {len(specs)} specs")
    if len(commits) > 1:
        raise BlockDriftError(f"the blocks ran {len(commits)} commits: {sorted(map(str, commits))}")
    return {"rows": rows, "blocks": blocks, "discarded": discarded, "unpassed": unpassed,
            "spec_sha256": next(iter(specs), None), "code_commit": next(iter(commits), None)}


def lock_bound(lock: Mapping) -> float:
    """T4.1's `block_drift_bound` from a lock file's values; refused if unset."""
    v = ((lock.get("values") or {}).get("block_drift_bound") or {}).get("value")
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v < 1:
        raise BlockDriftError(f"the lock's block_drift_bound is {v!r}, not a fraction in (0, 1)")
    return float(v)


def drift(rows: Sequence[Mapping], *, order: Sequence[str], bound: float,
          statistic: str = STATISTIC) -> dict:
    """The anchor's statistic per block, in `order` (the blocks that ran
    ground truth, in the spec's order), each block's relative difference from
    the median of the other blocks', and whether all are within `bound` (the
    lock's: `lock_bound`)."""
    if isinstance(bound, bool) or not isinstance(bound, (int, float)) \
            or not math.isfinite(bound) or not bound > 0:
        raise BlockDriftError("the bound is a positive relative difference, fixed at lock")
    anchors = [r for r in rows if r.get("anchor")]
    keys = {row_key(r) for r in anchors}
    if len(keys) > 1:
        raise BlockDriftError(f"the anchor rows are of {len(keys)} cells")
    by_block: dict = {}
    for r in anchors:
        if r["block"] in by_block:
            raise BlockDriftError(f"block {r['block']} has two anchor rows")
        by_block[r["block"]] = float(r[statistic])
    missing = [b for b in order if b not in by_block]
    if missing:
        raise BlockDriftError(f"blocks {missing} have no anchor row: every block that ran "
                              f"ground truth measures the anchor")
    bad = {b: by_block[b] for b in order if not (math.isfinite(by_block[b]) and by_block[b] > 0)}
    if bad:
        raise BlockDriftError(f"the anchor's {statistic} is not a finite positive time in {bad}")
    present = list(order)
    if len(present) < 2:
        return {"anchor": next(iter(keys), None), "blocks": present, "per_block": {},
                "passed": None, "why": f"{len(present)} block(s) with the anchor: nothing to compare"}
    per = {}
    for b in present:
        ref = statistics.median(by_block[o] for o in present if o != b)
        per[b] = {statistic: by_block[b], "reference": ref,
                  "relative_difference": (by_block[b] - ref) / ref}
    above = [b for b in present if abs(per[b]["relative_difference"]) > bound]
    return {"anchor": next(iter(keys)), "reference": "the median of the other blocks' anchors",
            "bound": bound, "per_block": per, "above": above, "passed": not above}


def pairing_drift(dirs: Sequence, spec, *, ground_truth_dir: str, bound: float) -> dict:
    """The post-session step: `collect` the blocks' directories and `drift`
    over those that ran ground truth, in `spec`'s order (a `blocks.BlockSpec`)."""
    got = collect(dirs, ground_truth_dir=ground_truth_dir)
    if got["spec_sha256"] not in (None, spec.sha256):
        raise BlockDriftError(f"the blocks ran spec {got['spec_sha256']}, not {spec.sha256}")
    order = [b.id for b in spec.blocks if "ground_truth" in b.steps and b.id in got["blocks"]]
    res = drift(got["rows"], order=order, bound=bound)
    return {"blocks": got["blocks"], "discarded": got["discarded"], "unpassed": got["unpassed"],
            "spec_sha256": got["spec_sha256"], "code_commit": got["code_commit"],
            "drift": res, "sensitivity_without": sorted(res.get("above") or ())}


def labelled(rows: Sequence[Mapping], result: Mapping) -> list[dict]:
    """The rows, those of a block above the bound labelled `DRIFT_LABEL`
    (`drift_label`); the input is not changed."""
    above = set(result.get("above") or ())
    return [dict(r, drift_label=DRIFT_LABEL) if r.get("block") in above else dict(r)
            for r in rows]


def claimable(rows: Sequence[Mapping], result: Mapping) -> list[Mapping]:
    """The rows primary scoring reads: all of them, a block above the bound
    included (the bound flags, it does not exclude; module docstring)."""
    return list(rows)


def sensitivity_sets(rows: Sequence[Mapping], result: Mapping) -> dict:
    """{flagged block: the rows without it}, one sensitivity run per block
    above the bound; empty when none is."""
    return {b: [r for r in rows if r.get("block") != b] for b in result.get("above") or ()}
