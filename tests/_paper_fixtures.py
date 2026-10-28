"""Fixtures for the paper-artifact tests (C10): a committed P1 prediction at
three lengths, loaded the way the analysis loads it, and its ground truth,
every number chosen by hand.

Committed cells (recursion, from test_predict_commit's worked examples):
  xattn  (deployable, code path CP)  OPS at every length:            16 s
  oracle (oracle, code path CP2)     OPS_HOST at every length:        8 s
  dense  (dense, code path CP)       K(h1, d) then the closing sync:  d + 2 s,
                                     d = 4, 12, 30 at L = 4, 8, 16:   6, 14, 32 s
Measured medians (s), with three repeats at 0.99x, 1x and 1.01x:
  xattn 16.8, 15.2, 16.8;  oracle 5.0, 8.4, 8.4;  dense 6.2, 15.5, 31.0
"""

from __future__ import annotations

import json
from pathlib import Path

from attnbench.figures.paper_inputs import INTERVALS_SCHEMA, SPEC_SCHEMA
from attnbench.figures.results import sha256_file
from attnbench.predict import commit as C
from attnbench.predict.commit import load_committed
from attnbench.predict.recursion import close_prefill, kernel_op

from _figure_fixtures import STARTED_AT, SYNC_AT
from test_predict_commit import (CLEAN, CODE_PATHS, CP, MODEL, NO_SKEW, OPS, OPS_HOST, T0,
                                 Clock, MemoryStore, build_record, cell, committed)

CP2_DEF = {"gpu": "L4", "implementation": "oracle:torch",
           "libraries": [["torch", "2.8.0"]], "python_optimize": 0}
CP2 = C.code_path_label(CP2_DEF)
LENGTHS = (4, 8, 16)
DENSE_D = {4: 4, 8: 12, 16: 30}
MEASURED = {"xattn": {4: 16.8, 8: 15.2, 16: 16.8},
            "oracle": {4: 5.0, 8: 8.4, 16: 8.4},
            "dense": {4: 6.2, 8: 15.5, 16: 31.0}}
KINDS = {"xattn": ("in_backend", True), "oracle": ("oracle", False),
         "dense": ("dense", False)}
CLASSES = {"xattn": "deployable_estimator", "oracle": "oracle", "dense": "dense"}
ADMISSIBLE = {("xattn", 4): True, ("xattn", 8): True, ("xattn", 16): True,
              ("oracle", 4): True, ("oracle", 8): True, ("oracle", 16): False}


def key(arm: str, L: int) -> list:
    return {"xattn": [MODEL, CP, "xattn", 0.5, L, 1],
            "oracle": [MODEL, CP2, "oracle", 0.5, L, 1],
            "dense": [MODEL, CP, "dense", 1.0, L, 1]}[arm]


def record(**extra):
    """The committed record; `extra` adds top-level sections."""
    cells = []
    for L in LENGTHS:
        cells.append(cell(ops=OPS, arm="xattn", L=L))
        cells.append(cell(ops=OPS_HOST, arm="oracle", L=L, code_path=CP2))
        cells.append(cell(ops=close_prefill([kernel_op(1, DENSE_D[L], layer=0)]),
                          arm="dense", density=1.0, L=L))
    rec = build_record(pairing="P1", cells=cells, queue_depth=8, banked=[],
                        fitted_sum=None, learned=None, code=CLEAN, written_at=T0,
                        code_paths={**CODE_PATHS, **({CP2: CP2_DEF} if CP2 != CP else {})})
    rec.update(extra)
    return rec


def loaded(tmp_path: Path, **extra):
    """(CommittedPrediction from load_committed, local file, store, remote)."""
    store = MemoryStore(Clock(T0))
    f, receipt = committed(tmp_path, record(**extra), store, SYNC_AT)
    cp = load_committed(store, receipt.remote, ground_truth_started_at=STARTED_AT,
                        clock_skew=NO_SKEW, local=f.path)
    return cp, f.path, store, receipt.remote


def result(L: int, measured=None, kinds=None, started=STARTED_AT) -> dict:
    measured = measured or MEASURED
    kinds = kinds or KINDS
    rows = []
    for arm in ("xattn", "oracle", "dense"):
        v = measured[arm][L] * 1000.0
        kind, dep = kinds[arm]
        rows.append({"arm": arm, "arm_kind": kind, "deployable": dep,
                     "prefill_ms_samples": [0.99 * v, v, 1.01 * v],
                     "prefill_ms_median": v, "prefill_ms_p90": 1.008 * v,
                     "prefill_ms_p99": 1.0098 * v})
    return {"traced": False, "complete": True,
            "first_timed_block_started_at": started.timestamp(), "rows": rows}


def write_json(path: Path, doc) -> Path:
    path.write_text(json.dumps(doc, indent=1, sort_keys=True))
    return path


def intervals_doc(cp, half_width=0.5, sha=None) -> dict:
    entries = []
    for arm in ("xattn", "oracle"):
        for L in LENGTHS:
            s, d = cp.cell(*key(arm, L)), cp.cell(*key("dense", L))
            p = s.predictions["recursion"] - d.predictions["recursion"]
            entries.append({"predictor": "recursion", "sparse_cell": key(arm, L),
                            "dense_cell": key("dense", L), "point": p,
                            "lo": p - half_width, "hi": p + half_width, "level": 0.9,
                            "n_boot": 200, "seed": 7})
    return {"schema": INTERVALS_SCHEMA, "prediction_sha256": sha or cp.sha256,
            "entries": entries}


def write_spec(d: Path, cp, *, crossover=True, results=None, **over) -> Path:
    """The ground truth, the intervals and the spec, written into `d`."""
    d.mkdir(parents=True, exist_ok=True)
    entries = []
    for L in LENGTHS:
        r = (results or {}).get(L, result(L))
        p = write_json(d / f"gt_L{L}.json", r)
        entries.append({"file": p.name, "sha256": sha256_file(p),
                        "arms": {a: key(a, L) for a in ("xattn", "oracle", "dense")}})
    spec = {"schema": SPEC_SCHEMA, "prediction_sha256": cp.sha256,
            "results": entries, "result_classes": dict(CLASSES), "dense_arm": "dense",
            "relative_floor": 0.05, "floor_provisional": True, "resolution_floor_s": 0.5,
            "admissible": [{"cell": key(a, L), "admissible": v}
                           for (a, L), v in ADMISSIBLE.items()]}
    if crossover:
        iv = write_json(d / "intervals.json", intervals_doc(cp))
        spec["crossover"] = {"level": 0.9, "n_boot": 200, "seed": 3,
                             "predicted_intervals": {"file": iv.name,
                                                     "sha256": sha256_file(iv)}}
    spec.update(over)
    return write_json(d / "spec.json", spec)
