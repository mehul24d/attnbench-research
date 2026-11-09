"""P1's training rows for the fitted baselines of every later pairing
(researcher's decision, 8 Nov 2026; plan Appendix A; Section 5.4).

  - **One row per cell**, its target the median of both T3.2 repeats'
    clean samples pooled (`PRIMARY`). Each repeat's own median, its count and
    the between-session spread are kept beside the row as metadata, so the
    spread is reported, never folded in.
  - **Pre-declared sensitivity analyses** are separate row sets, never the
    primary and never used to choose it: `SEPARATE_ROWS`, two rows per cell,
    each repeat's median its own row's target, and `REPEAT_A_ONLY`, every
    cell's target repeat a's median alone. Every set's rows carry repeat a's
    features (below), so a set differs from the primary in its targets only.
  - **Cross-validation groups by cell** (`cell_groups`): rows of one cell,
    in any row set, are one group, so a fold never trains on a cell it
    scores.
  - **Exploratory sessions never enter training**: a session is accepted only
    as one of a pool's two repeats (`POOLS`), by its prediction's store prefix
    (`predictions/P1-T3.2d/`, say); a debug, rehearsal or exploratory
    session's prefix is refused.
  - **The pools** (Appendix A, 8 and 9 Nov 2026): targets pool only sessions
    whose timed ground-truth path is byte-identical (`scripts/timed_path.py`).
    The primary is T3.2d and T3.2c (`4e78a962…`), features from T3.2d, the
    final identity (adviser's ruling 4 of 9 Nov 2026, eighth paste); T3.2a
    and T3.2b (`a5ade053…`) are a sensitivity. The first of a pool is its
    repeat a.
  - **A group short of clean rounds** (`gate_a.short_of_rounds`, the 10 of
    leg 1's rule) gives no target: its cells' targets pool the other repeat
    alone, and the cell's metadata names the repeat left out. A cell short in
    both repeats is refused.
  - **If repeat b fails**, the pooled row is repeat a alone (`repeat_b=None`
    with `b_failed` naming why); the rows say so.

`REPEAT_A_ONLY` keeps repeat a's median as recorded, short or not; the
metadata's `left_out` says when it is.

Each row's features are repeat a's committed cell (its priced sequence, as
the committed file holds it, read through `load_committed`), with the
session's link figures and the model's shape. Repeat a is the repeat the
fallback keeps, so the features are the same whichever target is used
(a plan default, flagged in Appendix A). A cell is matched across the
repeats by its ground-truth row label, token budget and batch
(`gate_a_repeats.cell_key`); an arm whose density its estimator sets
(XAttention) may differ in executed density between repeats, any other
difference in the committed cell is refused.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from ..analysis.gate_a import MIN_CLEAN_ROUNDS, GateAError, place_row, short_of_rounds
from ..analysis.gate_a_repeats import cell_key
from .commit import CommittedPrediction, require_loaded
from .intensity import ModelShape
from .training import Cell, GroundTruthRow, HardwareFeatures

PAIRING = "P1"
POOLS = (("T3.2d", "T3.2c"), ("T3.2a", "T3.2b"))   # (repeat a, the features; repeat b)
PRIMARY = "pooled"
SEPARATE_ROWS = "separate_rows"
REPEAT_A_ONLY = "repeat_a_only"
SENSITIVITY = (SEPARATE_ROWS, REPEAT_A_ONLY)
MS_PER_S = 1000.0


class TrainingRowsError(GateAError):
    pass


@dataclass(frozen=True)
class RepeatSession:
    label: str                                   # a session of `POOLS`, "T3.2d" say
    remote: str                                  # the committed prediction's store path
    cp: CommittedPrediction                      # read through load_committed
    results: Sequence[tuple[str, Mapping]]       # (name, untraced ground-truth result)
    hardware: HardwareFeatures                   # the session's link figures
    model: Optional[ModelShape] = None


def _check(s: RepeatSession, label: str) -> None:
    if s.label != label:
        raise TrainingRowsError(f"expected repeat {label}, got {s.label!r}")
    prefix = f"predictions/{PAIRING}-{label}/"
    path = s.remote.split("://", 1)[-1]
    if prefix not in path:
        raise TrainingRowsError(
            f"{s.remote} is not under {prefix}: only the T3.2 repeats train the baselines, "
            f"never an exploratory, debug or rehearsal session")
    cp = require_loaded(s.cp)
    if cp.pairing != PAIRING:
        raise TrainingRowsError(f"{s.label}'s prediction is {cp.pairing}'s, not {PAIRING}'s")


def _samples(s: RepeatSession, min_clean_rounds: int) -> dict:
    """{cell key: (committed cell, [seconds], nominal density, short)} from one
    repeat; `short` names a group short of clean rounds, else None."""
    out = {}
    for name, result in s.results:
        if result.get("traced"):
            raise TrainingRowsError(f"{s.label} {name}: a traced result is not ground truth")
        short = (f"{s.label} {name}: {result.get('clean_rounds')} clean of "
                 f"{result.get('rounds_run')} rounds, complete {result.get('complete')}"
                 if short_of_rounds(result, min_clean_rounds) else None)
        for row in result.get("rows", ()):
            k = cell_key(row)
            if k in out:
                raise TrainingRowsError(f"{s.label}: cell {k} is timed twice")
            xs = [float(x) / MS_PER_S for x in row["prefill_ms_samples"]]
            if not xs:
                raise TrainingRowsError(f"{s.label}: cell {k} has no clean samples")
            out[k] = (place_row(s.cp, row, f"{s.label} {name}"), xs, row.get("nominal_density"),
                      short)
    if not out:
        raise TrainingRowsError(f"{s.label}: no ground-truth rows")
    return out


def _same_cell(k, ca, cb, estimator_density: bool) -> None:
    fields = ("model", "code_path", "arm", "context_length", "batch")
    differ = [f for f in fields if getattr(ca, f) != getattr(cb, f)]
    if not estimator_density and ca.density != cb.density:
        differ.append("density")
    if differ:
        raise TrainingRowsError(f"cell {k} differs between the repeats in {differ}")


def _cell(s: RepeatSession, c) -> Cell:
    return Cell(pairing=PAIRING, model_name=c.model, code_path=c.code_path, arm=c.arm,
                context_length=c.context_length, density=c.density, batch=c.batch,
                priced=c.priced, hardware=s.hardware, model=s.model)


def training_rows(a: RepeatSession, b: Optional[RepeatSession], *,
                  b_failed: Optional[str] = None,
                  min_clean_rounds: int = MIN_CLEAN_ROUNDS) -> dict:
    """The primary rows, the sensitivity row sets, and per cell the metadata."""
    pool = next((p for p in POOLS if p[0] == a.label), None)
    if pool is None:
        raise TrainingRowsError(f"expected repeat a of a pool {POOLS}, got {a.label!r}")
    _check(a, pool[0])
    if (b is None) == (b_failed is None):
        raise TrainingRowsError("give repeat b, or say why it failed (b_failed), not both")
    if b is not None:
        _check(b, pool[1])
    sa = _samples(a, min_clean_rounds)
    sb = _samples(b, min_clean_rounds) if b is not None else {}
    if b is not None and set(sa) != set(sb):
        raise TrainingRowsError(f"the repeats timed different cells: only in a "
                                f"{sorted(set(sa) - set(sb))}, only in b {sorted(set(sb) - set(sa))}")
    primary, meta = [], []
    targets = {name: [] for name in SENSITIVITY}     # per cell, a list of targets
    for k in sorted(sa):
        ca, xa, nominal, short_a = sa[k]
        cell = _cell(a, ca)
        ma = statistics.median(xa)
        m = {"cell": list(k), "median_a_s": ma, "n_a": len(xa), "left_out": []}
        if b is not None:
            cb, xb, _, short_b = sb[k]
            _same_cell(k, ca, cb, estimator_density=nominal is None)
            mb = statistics.median(xb)
            kept = [(x, med) for x, med, short in ((xa, ma, short_a), (xb, mb, short_b))
                    if short is None]
            if not kept:
                raise TrainingRowsError(f"cell {k} is short of clean rounds in both repeats: "
                                        f"{short_a}; {short_b}")
            pooled = statistics.median([x for xs, _ in kept for x in xs])
            targets[SEPARATE_ROWS].append([med for _, med in kept])
            m.update(median_b_s=mb, n_b=len(xb),
                     between_session_spread=(max(ma, mb) - min(ma, mb)) / statistics.median([ma, mb]))
            m["left_out"] = [s for s in (short_a, short_b) if s is not None]
        else:
            if short_a is not None:
                raise TrainingRowsError(f"cell {k} is short of clean rounds: {short_a}")
            pooled = ma
            targets[SEPARATE_ROWS].append([ma])
            m.update(median_b_s=None, n_b=0, between_session_spread=None)
        m["target_s"] = pooled
        meta.append(m)
        primary.append(GroundTruthRow(cell=cell, measured=pooled))
        targets[REPEAT_A_ONLY].append([ma])
    sensitivity = {name: [GroundTruthRow(cell=r.cell, measured=y)
                          for r, ys in zip(primary, ts) for y in ys]
                   for name, ts in targets.items()}
    return {"primary": PRIMARY, "rows": primary, "sensitivity": sensitivity,
            "sensitivity_targets": targets,
            "metadata": meta, "repeats": [a.label] + ([b.label] if b is not None else []),
            "features_from": a.label, "min_clean_rounds": min_clean_rounds,
            "b_failed": b_failed,
            "predictions": {a.label: a.cp.sha256, **({b.label: b.cp.sha256} if b else {})}}


def cell_groups(rows: Sequence[GroundTruthRow]) -> list:
    """A cross-validation group per row: rows of one cell share it, in any
    row set, so a fold never trains on the cell it scores."""
    ids: dict = {}
    out = []
    for r in rows:
        c = r.cell
        k = (c.model_name, c.code_path, c.arm, c.context_length, c.batch,
             round(c.density, 6))
        out.append(ids.setdefault(k, len(ids)))
    return out


# --- the banked file a later pairing's committed prediction trains on -------------------------

SCHEMA = "attnbench.p1_training/2"     # /2: a sensitivity set holds targets per cell
DECISION = ("researcher's decision, 8 Nov 2026: one row per cell, the target the median of "
            "both T3.2 repeats' samples pooled; repeat a alone if repeat b fails. Appendix A, "
            "8 and 9 Nov 2026: the repeats share a timed ground-truth path, the primary T3.2d "
            "and T3.2c with T3.2d's features; a group short of clean rounds gives no target")
CELL_FIELDS = ("model_name", "code_path", "arm", "context_length", "density", "batch")


def _digest(rec: Mapping) -> str:
    import hashlib
    import json
    body = {k: v for k, v in rec.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def to_record(out: Mapping, hardware: HardwareFeatures) -> dict:
    """`training_rows`' output as the banked file: each row's cell, its
    committed components, its target and its metadata; each sensitivity set
    as a list of targets per row (two per cell for `SEPARATE_ROWS`); the repeats' prediction sha256s; and its own sha256."""
    from .commit import _components
    rows = []
    for r, m in zip(out["rows"], out["metadata"]):
        c = r.cell
        rows.append({"cell": {f: getattr(c, f) for f in CELL_FIELDS},
                     "components": _components(c.priced), "target_s": r.measured,
                     "metadata": dict(m)})
    rec = {"schema": SCHEMA, "decision": DECISION, "pairing": PAIRING,
           "primary": out["primary"], "repeats": list(out["repeats"]),
           "b_failed": out["b_failed"], "predictions": dict(out["predictions"]),
           "features_from": out["features_from"], "min_clean_rounds": out["min_clean_rounds"],
           "hardware": {"pinned_bandwidth": hardware.pinned_bandwidth,
                        "pageable_bandwidth": hardware.pageable_bandwidth},
           "rows": rows,
           "sensitivity": {k: [list(ys) for ys in v] for k, v in out["sensitivity_targets"].items()}}
    rec["sha256"] = _digest(rec)
    return rec


def rows_from_record(rec: Mapping, *, sensitivity: Optional[str] = None,
                     model: Optional[ModelShape] = None) -> list:
    """The primary rows from a banked file, or, named, one sensitivity set's.
    Refuses a file whose content is not its sha256's or not this schema."""
    from .commit import priced_from_components
    if rec.get("schema") != SCHEMA:
        raise TrainingRowsError(f"not a P1 training file: schema {rec.get('schema')!r}")
    if rec.get("sha256") != _digest(rec):
        raise TrainingRowsError("the P1 training file does not match its own sha256")
    if sensitivity is not None and sensitivity not in rec["sensitivity"]:
        raise TrainingRowsError(f"no sensitivity set {sensitivity!r}: {sorted(rec['sensitivity'])}")
    hw = HardwareFeatures(**rec["hardware"])
    targets = ([[r["target_s"]] for r in rec["rows"]] if sensitivity is None
               else rec["sensitivity"][sensitivity])
    if len(targets) != len(rec["rows"]):
        raise TrainingRowsError(f"{sensitivity!r} holds {len(targets)} cells' targets, "
                                f"not {len(rec['rows'])}")
    out = []
    for r, ys in zip(rec["rows"], targets):
        c = r["cell"]
        cell = Cell(pairing=PAIRING, model_name=c["model_name"], code_path=c["code_path"],
                    arm=c["arm"], context_length=int(c["context_length"]),
                    density=float(c["density"]), batch=int(c["batch"]),
                    priced=priced_from_components(r["components"]), hardware=hw, model=model)
        out += [GroundTruthRow(cell=cell, measured=float(y)) for y in ys]
    return out


def load(path) -> dict:
    """A banked training file (`.json`, or `.json.gz`), checked by `rows_from_record`."""
    import gzip
    import json
    from pathlib import Path
    path = Path(path)
    data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return json.loads(data)


def dump(rec: Mapping, path) -> None:
    """Write a banked training file, gzipped when the name ends in `.gz` (mtime 0,
    so the same content is the same bytes)."""
    import gzip
    import json
    from pathlib import Path
    path = Path(path)
    data = (json.dumps(rec, separators=(",", ":"), sort_keys=True) + "\n").encode()
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)
