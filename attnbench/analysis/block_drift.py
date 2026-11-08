"""A pairing's ground truth from its blocks, and the between-block drift
check (researcher's rulings, 7 Nov 2026; `attnbench.lock.blocks`).

A session run in blocks has a session directory per block, each with its own
boot. `collect` reads them: a block whose instance restarted carries
`INTERRUPTED_DISCARD.md` and is discarded alone (its rows are listed as
discarded, never read); the others must be blocks of one spec, run on one
commit, each once. Every ground-truth row carries its block's id and whether
it is the anchor cell, which every block with ground truth measures.

`drift` compares the anchor between blocks: each block's anchor statistic
against the first block's that has one, as a relative difference. The bound
is T4.1's block drift bound (lock table row 36), working 6.77%
(`WORKING_BOUND`: the largest systematic gap observed between two L4
sessions, T3.2a and T3.2b; adviser's ruling of 8 Nov 2026, replacing the 3%
of 7 Nov 2026). It is a flag, not an exclusion: a block above it is named and
its rows are labelled `DRIFT_LABEL` (`labelled`), every non-interrupted
block stays in primary scoring (`claimable`), and each flagged block gets a
sensitivity run without it (`sensitivity_sets`). The bound is never widened
to pass. The check reads the untraced rows, the ones the hypotheses are
scored on.
"""

from __future__ import annotations

import json
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


def collect(dirs: Sequence, *, ground_truth_dir: str) -> dict:
    """{"rows", "blocks", "discarded", "spec_sha256", "code_commit"} from the
    blocks' session directories. `ground_truth_dir` is the ground-truth
    step's directory name within each (`t61_session.step_dir`)."""
    rows, blocks, discarded, specs, commits = [], [], [], set(), set()
    for d in map(Path, dirs):
        m = _read(d / "manifest.json")
        b = (m or {}).get("block")
        if not b:
            raise BlockDriftError(f"{d} is not a block's session")
        if (d / DISCARD).exists():
            discarded.append({"block": b["id"], "dir": str(d), "why": DISCARD})
            continue
        if b["id"] in blocks:
            raise BlockDriftError(f"block {b['id']} is given twice")
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
    return {"rows": rows, "blocks": blocks, "discarded": discarded,
            "spec_sha256": next(iter(specs), None), "code_commit": next(iter(commits), None)}


def drift(rows: Sequence[Mapping], *, order: Sequence[str], bound: float = WORKING_BOUND,
          statistic: str = STATISTIC) -> dict:
    """The anchor's statistic per block, in `order` (the spec's), each block's
    relative difference from the first that has the anchor, and whether all
    are within `bound` (the lock's, `WORKING_BOUND` until then)."""
    if not bound > 0:
        raise ValueError("the bound is a positive relative difference, fixed at lock")
    anchors = [r for r in rows if r.get("anchor")]
    keys = {row_key(r) for r in anchors}
    if len(keys) > 1:
        raise BlockDriftError(f"the anchor rows are of {len(keys)} cells")
    by_block: dict = {}
    for r in anchors:
        if r["block"] in by_block:
            raise BlockDriftError(f"block {r['block']} has two anchor rows")
        by_block[r["block"]] = float(r[statistic])
    present = [b for b in order if b in by_block]
    if len(present) < 2:
        return {"anchor": next(iter(keys), None), "blocks": present, "per_block": {},
                "passed": None, "why": f"{len(present)} block(s) with the anchor: nothing to compare"}
    ref = by_block[present[0]]
    per = {b: {statistic: by_block[b], "relative_difference": (by_block[b] - ref) / ref}
           for b in present}
    above = [b for b in present if abs(per[b]["relative_difference"]) > bound]
    return {"anchor": next(iter(keys)), "reference_block": present[0], "bound": bound,
            "per_block": per, "above": above, "passed": not above}


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
