"""A pairing's timed session in blocks, each on its own boot (researcher's
rulings of 7 Nov 2026, after the H100 FA3 gate).

On Spot an instance can be preempted, no cell resumes across instances, and
an interrupted session is discarded in full, so a long session risks all of
it. The session is therefore split into short blocks. **The block spec is in
the lock file**, `{"blocks": {pairing: spec}}`:

    {"max_hours": 2.2,
     "anchor": {"arm": "dense", "density": 1.0, "context_length": 8192, "batch": 1},
     "blocks": [{"id": "b1", "steps": ["link", ...], "planned_hours": 1.88},
                {"id": "b2", "steps": [..., "predict", "ground_truth", "late_check",
                                        "trace_checks"],
                 "cells": [{"arm": ..., "density": ..., "context_length": ..., "batch": ...}],
                 "planned_hours": 2.2},
                ...]}

**Each block is self-contained on its own boot.** It runs the pairing's boot
steps (`prereqs`, `preflight` with its provenance, the end host-speed probe;
P4 adds `fa3_import`, its wheel installed on this boot) and then its own
steps, which name no boot step. What it needs from an earlier block (the
components a later prediction reads, the committed prediction a later ground
truth reuses) it imports from that block's own session directory, verified
(`scripts/t61_session.py`). Each block is its own session directory, so an
interrupted block writes `INTERRUPTED_DISCARD.md` and is discarded alone.

**The order rules hold across blocks.** Every step that is neither a boot
step nor a split step (`SPLIT_STEPS`: ground truth and the two checks that
follow it) is in exactly one block, and the blocks in order run them in the
session's order. `predict` is in one block, with `ground_truth` after it in
that block (nothing clocks between the two); every later block with ground
truth reuses that prediction and is never before it. Every block with ground
truth runs the checks after it and names its cells; no cell is in two
blocks. **One anchor cell** is measured in every block with ground truth and
is in no block's own cells: the analysis compares it between blocks
(`analysis.block_drift`).

`check` returns the problems; a spec whose cells are not yet fixed (the grid
is a lock value) is checked with `complete=False`, for the planner's
skeleton and a dry run.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

BOOT_STEPS = ("prereqs", "preflight", "host_speed_end")
SPLIT_STEPS = ("ground_truth", "late_check", "trace_checks")
CELL_FIELDS = ("arm", "density", "context_length", "batch")
_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,15}$")


class BlockSpecError(ValueError):
    pass


def cell_key(c: Mapping) -> tuple:
    """(arm, density, context length, batch) of a cell or a ground-truth row."""
    return (c["arm"], None if c.get("density") is None else float(c["density"]),
            int(c["context_length"]), int(c["batch"]))


@dataclass(frozen=True)
class Block:
    id: str
    steps: tuple
    cells: Optional[tuple] = None          # cell keys, None until the grid is locked
    planned_hours: Optional[float] = None


@dataclass(frozen=True)
class BlockSpec:
    pairing: str
    max_hours: float
    anchor: Optional[tuple]
    blocks: tuple
    sha256: str

    @property
    def ids(self) -> tuple:
        return tuple(b.id for b in self.blocks)

    def block(self, block_id: str) -> Block:
        if block_id not in self.ids:
            raise BlockSpecError(f"{self.pairing} has no block {block_id!r}; it has {list(self.ids)}")
        return self.blocks[self.ids.index(block_id)]

    def before(self, block_id: str) -> tuple:
        """The blocks that run before `block_id`, in order."""
        return self.blocks[:self.ids.index(self.block(block_id).id)]

    def run_steps(self, block_id: str, names: Sequence[str], boot: Sequence[str]) -> tuple:
        """The steps one block's session runs, in the session's order: the boot
        steps and the block's own."""
        own = set(self.block(block_id).steps)
        return tuple(n for n in names if n in boot or n in own)

    def predict_block(self) -> Optional[str]:
        held = [b.id for b in self.blocks if "predict" in b.steps]
        return held[0] if len(held) == 1 else None


def parse(lock: Mapping, pairing: str) -> BlockSpec:
    """The pairing's spec from a parsed lock file."""
    raw = (lock.get("blocks") or {}).get(pairing)
    if not isinstance(raw, Mapping):
        raise BlockSpecError(f"the lock file has no block spec for {pairing}")
    try:
        blocks = tuple(Block(str(b["id"]), tuple(b["steps"]),
                             None if b.get("cells") is None else
                             tuple(cell_key(c) for c in b["cells"]),
                             None if b.get("planned_hours") is None else float(b["planned_hours"]))
                       for b in raw["blocks"])
        anchor = None if raw.get("anchor") is None else cell_key(raw["anchor"])
        max_hours = float(raw["max_hours"])
    except (KeyError, TypeError, ValueError) as e:
        raise BlockSpecError(f"{pairing}'s block spec is malformed: {e!r}") from e
    text = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    return BlockSpec(pairing, max_hours, anchor, blocks, hashlib.sha256(text.encode()).hexdigest())


def check(spec: BlockSpec, names: Sequence[str], boot: Sequence[str] = BOOT_STEPS, *,
          complete: bool = True) -> list:
    """Why `spec` cannot run the session `names` (with boot steps `boot`);
    empty if it can. `complete`: every block with ground truth names its
    cells and an anchor is fixed (a real run's spec)."""
    names, boot = tuple(names), tuple(boot)
    out = []
    if not spec.blocks:
        return ["no blocks"]
    if len(set(spec.ids)) != len(spec.ids):
        out.append(f"block ids repeat: {list(spec.ids)}")
    out += [f"block id {i!r} is not lower-case letters, digits, - or _ (16 at most)"
            for i in spec.ids if not _ID.match(i)]
    if not 0 < spec.max_hours:
        out.append("max_hours must be positive")
    for b in spec.blocks:
        unknown = [s for s in b.steps if s not in names]
        if unknown:
            out.append(f"{b.id}: steps {unknown} are not this session's")
        listed = [s for s in b.steps if s in boot]
        if listed:
            out.append(f"{b.id}: names boot steps {listed}; every block runs them on its own boot")
        known = [s for s in b.steps if s in names]
        if known != sorted(known, key=names.index) or len(set(known)) != len(known):
            out.append(f"{b.id}: its steps are not in the session's order, once each")
        if b.planned_hours is not None and b.planned_hours > spec.max_hours + 1e-9:
            out.append(f"{b.id}: planned {b.planned_hours} h, above the block limit {spec.max_hours} h")
    single = [n for n in names if n not in boot and n not in SPLIT_STEPS]
    ran = [s for b in spec.blocks for s in b.steps if s in single]
    if ran != single:
        missing = [s for s in single if s not in ran]
        twice = sorted({s for s in ran if ran.count(s) > 1})
        out.append(f"the blocks do not run every other step once, in the session's order "
                   f"(missing {missing}, twice {twice})")
    pb = [b for b in spec.blocks if "predict" in b.steps]
    if "predict" in names and len(pb) != 1:
        out.append(f"predict is in {len(pb)} blocks; it is in exactly one")
    gt = [b for b in spec.blocks if "ground_truth" in b.steps]
    if pb and len(pb) == 1:
        p = spec.ids.index(pb[0].id)
        if "ground_truth" not in pb[0].steps:
            out.append(f"{pb[0].id}: ground truth follows predict in its block")
        early = [b.id for b in gt if spec.ids.index(b.id) < p]
        if early:
            out.append(f"blocks {early} run ground truth before the prediction is committed")
    for b in gt:
        lacking = [s for s in SPLIT_STEPS if s in names and s not in b.steps]
        if lacking:
            out.append(f"{b.id}: runs ground truth without {lacking} after it")
    for b in spec.blocks:
        stray = [s for s in SPLIT_STEPS[1:] if s in b.steps and "ground_truth" not in b.steps]
        if stray:
            out.append(f"{b.id}: runs {stray} with no ground truth")
        if b.cells is not None and "ground_truth" not in b.steps:
            out.append(f"{b.id}: names cells but runs no ground truth")
    seen: dict = {}
    for b in gt:
        if b.cells is None:
            if complete:
                out.append(f"{b.id}: runs ground truth and names no cells")
            continue
        if not b.cells:
            out.append(f"{b.id}: runs ground truth on no cell")
        for c in b.cells:
            if c in seen:
                out.append(f"cell {c} is in blocks {seen[c]} and {b.id}")
            seen.setdefault(c, b.id)
    if spec.anchor is not None and spec.anchor in seen:
        out.append(f"the anchor {spec.anchor} is also {seen[spec.anchor]}'s own cell; it is "
                   f"measured in every block, as no block's own")
    if len(gt) > 1 and spec.anchor is None and complete:
        out.append("ground truth in more than one block needs an anchor cell")
    return out
