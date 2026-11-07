"""A Chrome trace read one event at a time, and the compact event stream
step 5 keeps of its isolated replays (researcher's ruling, 2 Nov 2026, after
T3.1's third session).

That session's step 5 was killed by the kernel's out-of-memory killer at
31.6 GB on the 31 GB host, in the isolated replay of `oracle_d0.5_L32768`.
The oracle's isolated traces grow with the length (3.1 GB at
`oracle_d0.25_L32768`), and `json.load` with `op_sequence._events` holds
about 2.7 times a trace's size while it reads one (measured on the banked
oracle cells, 2048 to 16384).

`iter_trace_events` reads `traceEvents` element by element and never holds
the file. `write_compact` keeps, of an isolated replay's trace, the events
`host_profile.read_isolated` reads: complete events (`ph` "X") of the host
categories, each as its operation class (`cat`), its payload class (`name`),
its thread, its timing and its index in the original trace, so the calls it
groups are keyed as before. Device activity, flows and arguments are not
kept. `load_compact` reads the stream back as `op_sequence._Ev` events.

The isolated replay of the largest cells does not fit the host's memory as
one profiler session either (the profiler holds about five times the trace:
15.7 GB at `oracle_d0.25_L32768`, killed at 26 GB at density 0.5), so
`replay.isolated_replay_compact` runs it in blocks of layers, a profiler
session each, and `stitch_compact` joins the blocks' compact streams.

Step 6 prices the positions its CUDA graph does not hold from an isolated
replay of the original operators (`replay.price_positions`), and that replay
of `oracle_d0.25_L32768` is a 3 GB trace, killed at 26 GiB in its full read
by the debug session (2 Nov 2026). It is written the same way, segmented and
compacted (researcher's ruling 3 of 6 Nov 2026), with the fields
`replay.replay_nodes` reads (`REPLAY_CATS`, `DEVICE_FIELDS`): the region and
the flushes, the runtime and driver calls with their correlation, and the
device activities with their correlation, stream and bytes. No operator
(`cpu_op`) is kept. `replay.replay_nodes` reads such a stream in three passes,
holding the flushes' bounds, a set of correlations and the nodes it returns,
never the events.

The recording's own trace (`recorder.record_arm_prefill`) is not compacted:
the sequence parser reads its Python frames and the launch check its shapes,
and `recording.json` pins it by sha256.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Union

from .op_sequence import _Ev

SCHEMA = "attnbench.compact_trace/1"
FIELDS = ("i", "cat", "name", "pid", "tid", "ts", "dur")
# What `host_profile.read_isolated` reads: the replay region and the flushes
# (user annotations), the operators (cpu_op) and the launches (runtime, driver).
ISOLATED_CATS = frozenset({"user_annotation", "cpu_op", "cuda_runtime", "cuda_driver"})
# What `replay.replay_nodes` reads (step 6's isolated original, 6 Nov 2026 ruling 3):
# the region and the flushes, the calls that launch, and the device activities, each
# with the three arguments the reader takes (absent: null).
REPLAY_CATS = frozenset({"user_annotation", "cuda_runtime", "cuda_driver",
                         "kernel", "gpu_memcpy", "gpu_memset"})
ARG_FIELDS = ("correlation", "stream", "bytes")
DEVICE_FIELDS = FIELDS + ARG_FIELDS
SUFFIX = ".events.jsonl"

_CHUNK = 1 << 22                      # characters read at a time
_KEY = '"traceEvents"'
_SKIP = " \t\r\n,"
_NO_ARGS = MappingProxyType({})       # a compact event has no arguments


class TraceStreamError(ValueError):
    pass


def iter_trace_events(path: Union[str, Path], *, chunk: int = _CHUNK) -> Iterator[tuple[int, dict]]:
    """(index, event) for each element of the trace's `traceEvents`, in file
    order; the index is the element's position, as `json.load` would give it."""
    path = Path(path)
    dec = json.JSONDecoder()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        buf, pos, eof = "", 0, False

        def more() -> None:
            nonlocal buf, pos, eof
            s = f.read(chunk)
            eof = not s
            buf, pos = buf[pos:] + s, 0

        while True:                                   # find the array
            k = buf.find(_KEY, pos)
            if k >= 0:
                b = buf.find("[", k + len(_KEY))
                if b >= 0:
                    pos = b + 1
                    break
                pos = k
            else:
                pos = max(pos, len(buf) - len(_KEY))
            if eof:
                raise TraceStreamError(f"{path}: no traceEvents array")
            more()
        idx = 0
        while True:
            while True:
                while pos < len(buf) and buf[pos] in _SKIP:
                    pos += 1
                if pos < len(buf) or eof:
                    break
                more()
            if pos >= len(buf):
                raise TraceStreamError(f"{path}: traceEvents is not terminated")
            if buf[pos] == "]":
                return
            try:
                obj, end = dec.raw_decode(buf, pos)
            except json.JSONDecodeError:
                if eof:
                    raise TraceStreamError(f"{path}: event {idx} is not valid JSON") from None
                more()
                continue
            yield idx, obj
            idx += 1
            pos = end


def compact_path(path: Union[str, Path]) -> Path:
    path = Path(path)
    return path.with_name(path.name.removesuffix(".gz").removesuffix(".json") + SUFFIX)


def is_compact(source) -> bool:
    return isinstance(source, (str, Path)) and str(source).endswith(SUFFIX)


def write_compact(src: Union[str, Path], dst: Union[str, Path, None] = None, *,
                  cats=ISOLATED_CATS, fields=FIELDS) -> dict:
    """Write `src`'s complete events of `cats` to `dst` (default
    `compact_path(src)`): a header line, then one array per event in
    `fields` order (`FIELDS`, or `DEVICE_FIELDS` with the arguments after
    them). The same events `op_sequence._events` takes: `ph` "X" with a
    numeric `ts` and `dur`. Returns the counts."""
    fields = tuple(fields)
    if fields not in (FIELDS, DEVICE_FIELDS):
        raise TraceStreamError(f"fields {fields}: FIELDS or DEVICE_FIELDS only")
    extra = fields[len(FIELDS):]
    src = Path(src)
    dst = compact_path(src) if dst is None else Path(dst)
    tmp = dst.with_name(dst.name + ".part")
    total = kept = 0
    try:
        with open(tmp, "w", encoding="utf-8") as out:
            out.write(json.dumps({"schema": SCHEMA, "source": src.name, "cats": sorted(cats),
                                  "fields": list(fields)}) + "\n")
            for idx, e in iter_trace_events(src):
                total += 1
                if not isinstance(e, dict) or e.get("ph") != "X" or e.get("cat") not in cats:
                    continue
                ts, dur = e.get("ts"), e.get("dur", 0)
                if not isinstance(ts, (int, float)) or not isinstance(dur, (int, float)):
                    continue
                row = [idx, e["cat"], e.get("name", ""), e.get("pid"), e.get("tid"), ts, dur]
                if extra:
                    args = e.get("args") or {}
                    row += [args.get(k) for k in extra]
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
                kept += 1
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, dst)
    return {"path": str(dst), "events": total, "kept": kept,
            "source_bytes": src.stat().st_size, "bytes": dst.stat().st_size}


def read_header(f, path) -> dict:
    head = json.loads(f.readline() or "null")
    if not isinstance(head, dict) or head.get("schema") != SCHEMA:
        raise TraceStreamError(f"{path}: not a {SCHEMA} stream")
    return head


def iter_rows(path: Union[str, Path]) -> Iterator[list]:
    """The stream's rows, one list per event in its header's field order."""
    with open(path, encoding="utf-8") as f:
        read_header(f, path)
        for line in f:
            yield json.loads(line)


def fields_of(path: Union[str, Path]) -> tuple:
    with open(path, encoding="utf-8") as f:
        return tuple(read_header(f, path).get("fields") or FIELDS)


def load_compact(path: Union[str, Path]) -> list:
    """The stream's events, as `op_sequence._Ev`, with no arguments."""
    names: dict = {}
    out = []
    with open(path, encoding="utf-8") as f:
        read_header(f, path)
        for line in f:
            i, cat, name, pid, tid, ts, dur = json.loads(line)[:7]
            cat, name = names.setdefault(cat, cat), names.setdefault(name, name)
            ts = float(ts)
            out.append(_Ev(i, cat, name, pid, tid, ts, ts + float(dur), _NO_ARGS))
    return out


def _bounds(path: Path, region: str) -> dict:
    """One pass over a part: its header, event count, highest index and
    correlation, and the replay region it holds (exactly one)."""
    n = hi = corr = 0
    regions = []
    with open(path, encoding="utf-8") as f:
        head = read_header(f, path)
        for line in f:
            row = json.loads(line)
            i, cat, name, pid, tid, ts, dur = row[:7]
            n += 1
            hi = max(hi, i)
            if len(row) > 7 and isinstance(row[7], int):
                corr = max(corr, row[7])
            if cat == "user_annotation" and name == region:
                regions.append((i, pid, tid, float(ts), float(ts) + float(dur)))
    if len(regions) != 1:
        raise TraceStreamError(f"{path}: {len(regions)} {region!r} regions, expected one per segment")
    return {"head": head, "events": n, "max_index": hi, "max_correlation": corr,
            "region": regions[0]}


def stitch_compact(parts, dst: Union[str, Path], *, region: str) -> dict:
    """One compact stream from the streams of a replay's consecutive segments
    (researcher's ruling, 2 Nov 2026: the isolated replay in blocks of
    operations, so the profiler holds only a slice).

    Each segment is a profiler session of its own, with its own `region`
    event, exported and compacted by itself. The replay prices every
    operation independently (each alone on a cold cache), so the stream of
    the whole is the segments' events in order: concatenated, their
    indices offset to stay unique, and the segments' regions replaced by one
    region from the first's start to the last's end, on the first's thread.
    A stream with `DEVICE_FIELDS` has its correlations offset the same way,
    so a launch and its activity stay paired and no two segments share one.
    Refused if the segments' fields differ, a segment is not a `SCHEMA`
    stream, does not hold exactly one region, is on another thread than the
    first, or starts before the one before it ended (the segments are not on
    one timeline). Returns the counts."""
    parts, dst = [Path(p) for p in parts], Path(dst)
    if not parts:
        raise TraceStreamError("no segments to stitch")
    info = [_bounds(p, region) for p in parts]
    fields = tuple(info[0]["head"].get("fields") or FIELDS)
    for p, b in zip(parts, info):
        if tuple(b["head"].get("fields") or FIELDS) != fields:
            raise TraceStreamError(f"{p}: its fields differ from the first segment's")
    thread = info[0]["region"][1:3]
    prev_end = float("-inf")
    for p, b in zip(parts, info):
        i, pid, tid, r0, r1 = b["region"]
        if (pid, tid) != tuple(thread):
            raise TraceStreamError(f"{p}: its region is on thread {(pid, tid)}, not {tuple(thread)}")
        if r0 < prev_end - 1e-3:
            raise TraceStreamError(f"{p}: starts at {r0}, before the segment before it ended "
                                   f"({prev_end}); not one timeline")
        prev_end = r1
    first, last = info[0]["region"], info[-1]["region"]
    head = dict(info[0]["head"], source=dst.name, segments=len(parts))
    tmp = dst.with_name(dst.name + ".part")
    offset = corr_offset = 0
    kept = 0
    pad = [None] * (len(fields) - len(FIELDS))
    try:
        with open(tmp, "w", encoding="utf-8") as out:
            out.write(json.dumps(head) + "\n")
            out.write(json.dumps([first[0], "user_annotation", region, first[1], first[2],
                                  first[3], last[4] - first[3]] + pad,
                                 separators=(",", ":")) + "\n")
            kept += 1
            for p, b in zip(parts, info):
                with open(p, encoding="utf-8") as f:
                    f.readline()
                    for line in f:
                        row = json.loads(line)
                        if row[1] == "user_annotation" and row[2] == region:
                            continue
                        row[0] += offset
                        if pad and isinstance(row[7], int):
                            row[7] += corr_offset
                        out.write(json.dumps(row, separators=(",", ":")) + "\n")
                        kept += 1
                offset += b["max_index"] + 1
                corr_offset += b["max_correlation"] + 1
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, dst)
    return {"path": str(dst), "segments": len(parts), "kept": kept, "bytes": dst.stat().st_size}
