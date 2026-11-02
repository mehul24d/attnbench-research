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
                  cats=ISOLATED_CATS) -> dict:
    """Write `src`'s complete events of `cats` to `dst` (default
    `compact_path(src)`): a header line, then one array per event in
    `FIELDS` order. The same events `op_sequence._events` takes: `ph` "X"
    with a numeric `ts` and `dur`. Returns the counts."""
    src = Path(src)
    dst = compact_path(src) if dst is None else Path(dst)
    tmp = dst.with_name(dst.name + ".part")
    total = kept = 0
    try:
        with open(tmp, "w", encoding="utf-8") as out:
            out.write(json.dumps({"schema": SCHEMA, "source": src.name, "cats": sorted(cats),
                                  "fields": list(FIELDS)}) + "\n")
            for idx, e in iter_trace_events(src):
                total += 1
                if not isinstance(e, dict) or e.get("ph") != "X" or e.get("cat") not in cats:
                    continue
                ts, dur = e.get("ts"), e.get("dur", 0)
                if not isinstance(ts, (int, float)) or not isinstance(dur, (int, float)):
                    continue
                out.write(json.dumps([idx, e["cat"], e.get("name", ""), e.get("pid"),
                                      e.get("tid"), ts, dur], separators=(",", ":")) + "\n")
                kept += 1
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, dst)
    return {"path": str(dst), "events": total, "kept": kept,
            "source_bytes": src.stat().st_size, "bytes": dst.stat().st_size}


def load_compact(path: Union[str, Path]) -> list:
    """The stream's events, as `op_sequence._Ev`, with no arguments."""
    names: dict = {}
    out = []
    with open(path, encoding="utf-8") as f:
        head = json.loads(f.readline() or "null")
        if not isinstance(head, dict) or head.get("schema") != SCHEMA:
            raise TraceStreamError(f"{path}: not a {SCHEMA} stream")
        for line in f:
            i, cat, name, pid, tid, ts, dur = json.loads(line)
            cat, name = names.setdefault(cat, cat), names.setdefault(name, name)
            ts = float(ts)
            out.append(_Ev(i, cat, name, pid, tid, ts, ts + float(dur), _NO_ARGS))
    return out
