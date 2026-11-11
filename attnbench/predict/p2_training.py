"""P2's training rows for P4's committed fits (adviser's ruling 1h of 11 Nov
2026; Section 5.4: each later pairing trains only on the pairings banked
before it, so P4's fits read P1's rows and these).

  - **One row per cell**, its target the median of the cell's clean samples
    from P2's passed blocks (`block_drift.collect`: a discarded or unpassed
    block is never read, ruling 1d). A cell timed in more than one block
    (the anchor, which every block measures) pools its samples across them,
    as P1 pools its two repeats; each block's median and count are kept as
    metadata, never folded in. A cell timed twice within one block is
    refused. (A plan default, flagged in Appendix A.)
  - **A group short of clean rounds** (`gate_a.short_of_rounds`) gives no
    samples; a cell with none left is left out of the rows and named in
    `left_out`, never filled from elsewhere (P1's fallback to the other
    repeat has no counterpart on one pairing; a plan default, flagged).
  - **Features** are P2's committed cell (its priced sequence, read through
    `load_committed`), with the session's link figures and the model's
    shape, exactly as P1's rows carry repeat a's.
  - **Only P2's own committed prediction** trains P4: its store path is under
    `predictions/P2/` and its pairing is P2, so a rehearsal's (`P2-R`) or an
    exploratory session's is refused.

The banked file is `p1_training`'s form under its own schema: each row's
cell, components, target and metadata, and the file's own sha256.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from ..analysis.gate_a import MIN_CLEAN_ROUNDS, place_row, short_of_rounds
from ..analysis.gate_a_repeats import cell_key
from .commit import CommittedPrediction, require_loaded
from .intensity import ModelShape
from .p1_training import CELL_FIELDS, MS_PER_S, TrainingRowsError, _digest, dump, load
from .training import Cell, GroundTruthRow, HardwareFeatures

__all__ = ["PAIRING", "PairingSession", "training_rows", "to_record", "rows_from_record",
           "load", "dump", "TrainingRowsError"]

PAIRING = "P2"


@dataclass(frozen=True)
class PairingSession:
    remote: str                                  # the committed prediction's store path
    cp: CommittedPrediction                      # read through load_committed
    blocks: Sequence[tuple[str, Sequence[tuple[str, Mapping]]]]   # passed blocks' (id, results)
    hardware: HardwareFeatures                   # the session's link figures
    model: Optional[ModelShape] = None
    unpassed: Sequence[Mapping] = ()             # collect's discarded and unpassed, for the record


def _check(s: PairingSession) -> None:
    prefix = f"predictions/{PAIRING}/"
    if prefix not in s.remote.split("://", 1)[-1]:
        raise TrainingRowsError(
            f"{s.remote} is not under {prefix}: only P2's own committed prediction trains "
            f"P4, never a rehearsal's or an exploratory session's")
    cp = require_loaded(s.cp)
    if cp.pairing != PAIRING:
        raise TrainingRowsError(f"the prediction is {cp.pairing}'s, not {PAIRING}'s")
    if not s.blocks:
        raise TrainingRowsError("no passed block: P2 has no ground truth to train on")


def training_rows(s: PairingSession, *, min_clean_rounds: int = MIN_CLEAN_ROUNDS) -> dict:
    """The rows, and per cell the metadata, from P2's passed blocks."""
    _check(s)
    cells: dict = {}             # cell key -> committed cell
    per: dict = {}               # cell key -> [(block, [seconds])]
    short: dict = {}             # cell key -> [why]
    for bid, results in s.blocks:
        seen = set()
        for name, result in results:
            if result.get("traced"):
                raise TrainingRowsError(f"{bid} {name}: a traced result is not ground truth")
            why = (f"{bid} {name}: {result.get('clean_rounds')} clean of "
                   f"{result.get('rounds_run')} rounds, complete {result.get('complete')}"
                   if short_of_rounds(result, min_clean_rounds) else None)
            for row in result.get("rows", ()):
                k = cell_key(row)
                if k in seen:
                    raise TrainingRowsError(f"block {bid}: cell {k} is timed twice")
                seen.add(k)
                c = place_row(s.cp, row, f"{bid} {name}")
                if cells.setdefault(k, c) is not c:
                    raise TrainingRowsError(f"cell {k} is placed on two committed cells")
                if why is not None:
                    short.setdefault(k, []).append(why)
                    continue
                xs = [float(x) / MS_PER_S for x in row["prefill_ms_samples"]]
                if not xs:
                    raise TrainingRowsError(f"block {bid}: cell {k} has no clean samples")
                per.setdefault(k, []).append((bid, xs))
    if not cells:
        raise TrainingRowsError("no ground-truth rows in P2's passed blocks")
    rows, meta, left_out = [], [], []
    for k in sorted(cells):
        if k not in per:
            left_out.append({"cell": list(k), "why": short[k]})
            continue
        c = cells[k]
        target = statistics.median([x for _, xs in per[k] for x in xs])
        cell = Cell(pairing=PAIRING, model_name=c.model, code_path=c.code_path, arm=c.arm,
                    context_length=c.context_length, density=c.density, batch=c.batch,
                    priced=c.priced, hardware=s.hardware, model=s.model)
        rows.append(GroundTruthRow(cell=cell, measured=target))
        meta.append({"cell": list(k), "target_s": target,
                     "blocks": [{"block": b, "median_s": statistics.median(xs), "n": len(xs)}
                                for b, xs in per[k]],
                     "left_out": short.get(k, [])})
    if not rows:
        raise TrainingRowsError(f"every cell is short of clean rounds: {left_out}")
    return {"rows": rows, "metadata": meta, "left_out": left_out,
            "blocks": [b for b, _ in s.blocks], "unpassed": [dict(u) for u in s.unpassed],
            "min_clean_rounds": min_clean_rounds, "prediction": s.cp.sha256}


SCHEMA = "attnbench.p2_training/1"
DECISION = ("adviser's ruling 1h, 11 Nov 2026: P4 trains on P1's and P2's rows; one row per "
            "cell, the target the median of its clean samples pooled over P2's passed blocks; "
            "a cell with no clean group is left out and named")


def to_record(out: Mapping, hardware: HardwareFeatures) -> dict:
    """`training_rows`' output as the banked file, with its own sha256."""
    from .commit import _components
    rows = []
    for r, m in zip(out["rows"], out["metadata"]):
        c = r.cell
        rows.append({"cell": {f: getattr(c, f) for f in CELL_FIELDS},
                     "components": _components(c.priced), "target_s": r.measured,
                     "metadata": dict(m)})
    rec = {"schema": SCHEMA, "decision": DECISION, "pairing": PAIRING,
           "prediction": out["prediction"], "blocks": list(out["blocks"]),
           "unpassed": list(out["unpassed"]), "left_out": list(out["left_out"]),
           "min_clean_rounds": out["min_clean_rounds"],
           "hardware": {"pinned_bandwidth": hardware.pinned_bandwidth,
                        "pageable_bandwidth": hardware.pageable_bandwidth},
           "rows": rows}
    rec["sha256"] = _digest(rec)
    return rec


def rows_from_record(rec: Mapping, *, model: Optional[ModelShape] = None) -> list:
    """The rows from a banked file; refuses one whose content is not its
    sha256's or not this schema."""
    from .commit import priced_from_components
    if rec.get("schema") != SCHEMA:
        raise TrainingRowsError(f"not a P2 training file: schema {rec.get('schema')!r}")
    if rec.get("sha256") != _digest(rec):
        raise TrainingRowsError("the P2 training file does not match its own sha256")
    hw = HardwareFeatures(**rec["hardware"])
    out = []
    for r in rec["rows"]:
        c = r["cell"]
        cell = Cell(pairing=PAIRING, model_name=c["model_name"], code_path=c["code_path"],
                    arm=c["arm"], context_length=int(c["context_length"]),
                    density=float(c["density"]), batch=int(c["batch"]),
                    priced=priced_from_components(r["components"]), hardware=hw, model=model)
        out.append(GroundTruthRow(cell=cell, measured=float(r["target_s"])))
    return out
