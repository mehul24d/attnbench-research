"""Figure 1's events table: one decoder layer of a recorded prefill, as rows
a plotting script can draw without reading a Chrome trace.

**Source.** The pair T2.27's step 11 records (`scripts/t227_steps.py`,
`step_figure1`): a trace written under T2.3's recording rules, with
`host_compute()` around the mask build, and the operation sequence the
repo's parser made of it (`op_sequence.parse_trace`), banked beside it as a
sequence record. The converter classifies nothing itself. Each row's kind
and mechanism are the parsed operation's, tied to the trace's events by
correlation id; only the times come from the trace.

A banked sequence record carries no correlation ids (it holds no
durations, Section 5.1), so `events_from_files` parses the trace again
with the record's own code path, length and batch, and refuses unless the
result equals the banked sequence. The table is then the banked sequence's,
with the times of the trace it came from.

**Rows.** For every operation of the layer, in sequence order:

  - a host row for each runtime or driver call the operation was parsed
    from (two for a copy folded with its wait, `api` "copy+wait"), and for
    each `host_compute` span;
  - a device row for each device activity launched by those calls (kernel,
    copy or memset), which may end after the layer's annotation does.

A runtime call inside the layer that no parsed operation accounts for, or
an operation with no call in the layer, is refused: the table and the
sequence would disagree. Cross-stream waits and calls the parser ignores
make no operation and no row.

**The CSV is the plotting script's schema** (`COLUMNS`): gpu, lane, op,
start_ms, end_ms, kind (researcher's decision, 28 Oct 2026).

  - gpu: the panel's label, given by the caller (`L4`, `A100`); it must
    name the trace's own device, so a panel cannot be mislabelled.
  - lane: host or device.
  - op: on a host row, the operation's aten op where the parser found one,
    else the call's name; on a device row, the kernel or copy's name; on a
    mask-build row, the annotation's name.
  - start_ms, end_ms: from the start of the layer's annotation.
    `us_to_ms` is the one microseconds-to-milliseconds conversion.
  - kind: one of `KINDS`. A host row's kind is its parsed operation's
    (`HOST_KIND`: host compute is the mask build, which T2.27's step 11
    wraps in `host_compute()`); a device row's is its activity's
    (`DEVICE_KIND`), so the copy an `.item()` waits on is drawn as a copy
    on the device lane and as a sync on the host lane.

Rows also carry, outside the CSV, the operation's position, parsed kind,
mechanism and correlation id, so a test can tie each row to the parse.
Rows are ordered by sequence position, then host before device, then start.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

from ..components import op_sequence as ops

EVENTS_SCHEMA = "attnbench.fig1_events/2"
COLUMNS = ("gpu", "lane", "op", "start_ms", "end_ms", "kind")
HOST, DEVICE = "host", "device"
MASK_BUILD, SYNC, COPY, KERNEL, OTHER = "mask_build", "sync", "copy", "kernel", "other"
KINDS = (MASK_BUILD, SYNC, COPY, KERNEL, OTHER)
HOST_KIND = {ops.SeqKind.HOST_COMPUTE: MASK_BUILD, ops.SeqKind.SYNC: SYNC,
             ops.SeqKind.COPY: COPY, ops.SeqKind.DEVICE_COPY: COPY,
             ops.SeqKind.KERNEL: KERNEL, ops.SeqKind.GRAPH: KERNEL,
             ops.SeqKind.MEMSET: OTHER}
DEVICE_KIND = {"kernel": KERNEL, "gpu_memcpy": COPY, "gpu_memset": OTHER}
US_PER_MS = 1000.0
_HOST_CATS = ("cuda_runtime", "cuda_driver")
_DEVICE_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
_EPS = 1e-3          # microseconds, as op_sequence


class Fig1Error(ValueError):
    """A trace and sequence the events table cannot be made from."""


def _corr(e: Mapping) -> Optional[int]:
    c = (e.get("args") or {}).get("correlation")
    return int(c) if isinstance(c, (int, float)) and not isinstance(c, bool) else None


def us_to_ms(us: float) -> float:
    """The one microseconds-to-milliseconds conversion: Kineto writes
    microseconds, the figure reads milliseconds. Rounded to the nanosecond,
    the trace's resolution."""
    return round(float(us) / US_PER_MS, 6)


def _check_gpu(gpu, doc) -> str:
    if not isinstance(gpu, str) or not gpu.strip() or "," in gpu:
        raise Fig1Error(f"a gpu label is a short name without commas, got {gpu!r}")
    names = list(ops.trace_gpus(doc).values())
    if not any(gpu.lower() in n.lower() for n in names):
        raise Fig1Error(f"the label {gpu!r} does not name this trace's device {names}")
    return gpu


def layer_events(trace, seq: ops.OpSequence, layer: int, *, gpu: str) -> list[dict]:
    """The rows for `layer` of `trace`, whose parse is `seq` (with its
    correlation ids, as `op_sequence.parse_trace` returns it)."""
    doc = ops.load_trace(trace)
    gpu = _check_gpu(gpu, doc)
    events = [e for e in doc.get("traceEvents", ()) if e.get("ph") == "X"]
    name = ops.layer_annotation(layer)
    anns = [e for e in events if e.get("cat") == "user_annotation" and e.get("name") == name]
    if len(anns) != 1:
        raise Fig1Error(f"{len(anns)} annotations {name!r}; the table is drawn from one")
    a = anns[0]
    thread = (a.get("pid"), a.get("tid"))
    t0, t1 = float(a["ts"]), float(a["ts"]) + float(a["dur"])

    def inside(e):
        return (e.get("pid"), e.get("tid")) == thread and float(e["ts"]) >= t0 - _EPS \
            and float(e["ts"]) + float(e.get("dur", 0)) <= t1 + _EPS

    calls = sorted((e for e in events if e.get("cat") in _HOST_CATS and inside(e)),
                   key=lambda e: float(e["ts"]))
    counted = [e for e in calls if ops.api_class(e["name"]) not in (None, "cross_stream")]
    by_corr = {}
    for e in counted:
        c = _corr(e)
        if c is None:
            raise Fig1Error(f"{e['name']} at {e['ts']} has no correlation id")
        by_corr[c] = e
    marks = sorted((e for e in events if e.get("cat") == "user_annotation"
                    and e.get("name") == ops.HOST_COMPUTE and inside(e)),
                   key=lambda e: float(e["ts"]))
    device: dict[int, list] = {}
    for e in events:
        if e.get("cat") in _DEVICE_CATS and _corr(e) is not None:
            device.setdefault(_corr(e), []).append(e)

    layer_ops = [(i, op) for i, op in enumerate(seq.ops) if op.layer == layer]
    if not layer_ops:
        raise Fig1Error(f"the sequence holds no operation in layer {layer}")

    rows, used, mark_i = [], set(), 0

    def row(i, op, lane, e):
        start = float(e["ts"]) - t0
        end = start + float(e.get("dur", 0))
        if lane == HOST:
            kind = HOST_KIND[op.kind]
            name = e.get("name") if op.kind is ops.SeqKind.HOST_COMPUTE \
                else (op.aten_op or e.get("name"))
        else:
            kind = DEVICE_KIND[e.get("cat")]
            name = e.get("name")
        return {"gpu": gpu, "lane": lane, "op": name, "start_ms": us_to_ms(start),
                "end_ms": us_to_ms(end), "kind": kind,
                # outside the CSV: what ties the row to the parse
                "seq_index": i, "parsed_kind": op.kind.value,
                "mechanism": op.mechanism or "", "correlation": _corr(e)}

    for i, op in layer_ops:
        if op.kind is ops.SeqKind.HOST_COMPUTE:
            if mark_i >= len(marks):
                raise Fig1Error(f"operation {i} is host compute, and the layer has no "
                                f"host_compute span left for it")
            rows.append(row(i, op, HOST, marks[mark_i]))
            mark_i += 1
            continue
        if op.correlation is None:
            raise Fig1Error(f"operation {i} has no correlation id: pass the sequence "
                            f"parse_trace returns, not one read from a banked record")
        first = by_corr.get(op.correlation)
        if first is None or id(first) in used:
            raise Fig1Error(f"operation {i} ({op.api}, correlation {op.correlation}) has no "
                            f"call in layer {layer} of this trace")
        host = [first]
        apis = op.api.split("+")
        if len(apis) == 2:          # a copy folded with its wait
            nxt = [e for e in counted if float(e["ts"]) >= float(first["ts"])
                   and e is not first and id(e) not in used and e["name"] == apis[1]]
            if not nxt:
                raise Fig1Error(f"operation {i} ({op.api}): the wait is not in the layer")
            host.append(nxt[0])
        for e in host:
            used.add(id(e))
            rows.append(row(i, op, HOST, e))
        for e in sorted(device.get(op.correlation, ()), key=lambda e: float(e["ts"])):
            rows.append(row(i, op, DEVICE, e))

    left = [f"{e['name']} at {e['ts']}" for e in counted if id(e) not in used]
    if left:
        raise Fig1Error(f"calls in layer {layer} that no parsed operation accounts for: "
                        f"{left[:5]}; the sequence was not parsed from this trace")
    if mark_i != len(marks):
        raise Fig1Error(f"{len(marks) - mark_i} host_compute spans in layer {layer} with no "
                        f"parsed operation")
    rows.sort(key=lambda r: (r["seq_index"], r["lane"] != HOST, r["start_ms"]))
    return rows


def _load_record(path: Union[str, Path]) -> tuple[dict, ops.OpSequence]:
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    stated = rec.pop("sha256", None)
    if stated is None or stated != ops.record_sha256(rec):
        raise Fig1Error(f"{path}: the sequence record's sha256 is missing or wrong")
    return rec, ops.from_record(rec)


def events_from_files(trace_path, sequence_path, layer: int, *, gpu: str) -> list[dict]:
    """Rows for one panel from a trace and the sequence record banked beside
    it (`t227_steps._save_sequence`). The trace is parsed again with the
    record's code path, length and batch, and must give the banked sequence."""
    rec, banked = _load_record(sequence_path)
    parsed = ops.parse_trace(trace_path, banked.code_path,
                             context_length=banked.context_length, batch=banked.batch)
    if parsed != banked:
        raise Fig1Error(f"{trace_path} does not parse to the sequence banked in "
                        f"{sequence_path}")
    return layer_events(trace_path, parsed, layer, gpu=gpu)


def to_csv(rows: Iterable[Mapping]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        if not set(COLUMNS) <= set(r) or r["kind"] not in KINDS:
            raise Fig1Error(f"a row without the schema's fields or kinds: {r}")
        w.writerow({k: (f"{r[k]:.6f}" if k.endswith("_ms") else r[k]) for k in COLUMNS})
    return buf.getvalue()


def write_events(panels: Sequence[Sequence[Mapping]], path: Union[str, Path]) -> Path:
    """One events.csv from one or more GPUs' rows. Never overwrites."""
    names = [p[0]["gpu"] for p in panels if p]
    if len(names) != len(panels) or len(set(names)) != len(names):
        raise Fig1Error(f"each panel needs rows and a distinct gpu label, got {names}")
    path = Path(path)
    with open(path, "x", encoding="utf-8", newline="") as f:
        f.write(to_csv([r for p in panels for r in p]))
    return path
