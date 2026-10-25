"""Trace checks on a pairing's ground-truth trace: copy-kernel concurrency
and unified memory (T2.7; plan Sections 5.4 and 5.6).

A check on the model, never an input to it. Two sources, each with the claim
it licenses (researcher's decision, 25 Oct 2026; Appendix A):

  - **The Kineto trace** (`torch.profiler` chrome-trace JSON) gives
    copy-kernel concurrency and **managed-memory use**: a copy or memset on
    managed memory, or a call that allocates, prefetches or advises managed
    memory. A run checked only this way may claim "no managed-memory use",
    never "no page faults": under HMM a GPU can fault on ordinary pageable
    memory, and the PyTorch profiler does not record faults.
  - **An Nsight Systems export** (`nsys export --type sqlite`) of a separate
    traced pass, never a timing run, gives **page faults**. The claim "no
    page faults" is made only when a positive control, captured in the same
    session with the same nsys options, did record faults: that is what shows
    the collection was on. It covers only the arms that pass traced.

**Copy-kernel concurrency.** The recursion serialises copies and kernels on
one stream (Section 5.4). A copy whose device interval overlaps a kernel or
memset on another stream of the same device breaks that, and is flagged.
Overlap on the same stream cannot happen on the device; when the trace shows
it, it is a timestamp artifact and is counted separately, not flagged.
"""

from __future__ import annotations

import bisect
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

COPY_CATS = {"gpu_memcpy"}
COMPUTE_CATS = {"kernel", "gpu_memset"}
API_CATS = {"cuda_runtime", "cuda_driver"}
UM_APIS = {"cudaMallocManaged", "cudaMemPrefetchAsync", "cudaMemAdvise",
           "cuMemAllocManaged", "cuMemPrefetchAsync", "cuMemAdvise",
           "cudaMemPrefetchAsync_v2", "cudaMemAdvise_v2"}


def load_kineto(path) -> list[dict]:
    data = json.loads(Path(path).read_text())
    events = data["traceEvents"] if isinstance(data, dict) else data
    return [e for e in events if e.get("ph") == "X"]


def _device_intervals(events: Iterable[dict], cats: set) -> list[tuple]:
    """(device, stream, start, end, name) in microseconds."""
    out = []
    for e in events:
        if e.get("cat") in cats and "dur" in e:
            a = e.get("args", {})
            start = float(e["ts"])
            out.append((a.get("device"), a.get("stream"), start,
                        start + float(e["dur"]), e.get("name", "")))
    return out


def _merge(intervals: list[tuple[float, float]]) -> tuple[list[float], list[float]]:
    intervals.sort()
    starts, ends = [], []
    for s, e in intervals:
        if starts and s <= ends[-1]:
            ends[-1] = max(ends[-1], e)
        else:
            starts.append(s)
            ends.append(e)
    return starts, ends


def _overlap(starts, ends, s, e) -> float:
    """Length of [s, e) inside a sorted set of disjoint intervals."""
    total = 0.0
    i = max(0, bisect.bisect_right(starts, s) - 1)
    while i < len(starts) and starts[i] < e:
        total += max(0.0, min(e, ends[i]) - max(s, starts[i]))
        i += 1
    return total


def copy_kernel_concurrency(events: Sequence[dict]) -> dict:
    copies = _device_intervals(events, COPY_CATS)
    compute = _device_intervals(events, COMPUTE_CATS)
    by_stream: dict = defaultdict(list)
    for dev, stream, s, e, _ in compute:
        by_stream[dev, stream].append((s, e))
    merged = {k: _merge(v) for k, v in by_stream.items()}

    cross, same, pairs = [], [], set()
    for dev, stream, s, e, name in copies:
        for (kdev, kstream), (starts, ends) in merged.items():
            if kdev != dev:
                continue
            ov = _overlap(starts, ends, s, e)
            if ov <= 0:
                continue
            if kstream == stream:
                same.append(ov)
            else:
                cross.append(ov)
                pairs.add((stream, kstream))
    return {
        "copies": len(copies), "kernels": len(compute),
        "streams": sorted({str(k[1]) for k in by_stream} |
                          {str(c[1]) for c in copies}),
        "copy_kernel_concurrency": bool(cross),
        "cross_stream_overlaps": len(cross),
        "cross_stream_overlap_us_total": sum(cross),
        "cross_stream_overlap_us_max": max(cross, default=0.0),
        "cross_stream_pairs": sorted(f"{a}->{b}" for a, b in pairs),
        "same_stream_overlaps": len(same),
        "same_stream_overlap_us_max": max(same, default=0.0),
    }


def managed_memory_use(events: Sequence[dict]) -> dict:
    """Copies or memsets that name managed memory, and managed-memory API
    calls. Kineto names a copy `Memcpy <kind> (<src> -> <dst>)` with the
    memory kinds in the name; the name is matched, case-sensitively, on
    "Managed"."""
    managed = sorted({e.get("name", "") for e in events
                      if e.get("cat") in COPY_CATS | {"gpu_memset"}
                      and "Managed" in e.get("name", "")})
    apis = sorted({e.get("name", "") for e in events
                   if e.get("cat") in API_CATS and e.get("name") in UM_APIS})
    return {"managed_memory_use": bool(managed or apis),
            "managed_copy_names": managed, "managed_api_calls": apis,
            "claim": ("managed memory used" if managed or apis
                      else "no managed-memory use")}


def check_kineto(path) -> dict:
    events = load_kineto(path)
    if not any(e.get("cat") in COPY_CATS | COMPUTE_CATS for e in events):
        raise ValueError(f"{path}: no device activity; was CUDA activity recorded?")
    return {"source": "kineto", "trace": str(path),
            **copy_kernel_concurrency(events), **managed_memory_use(events)}


# --- Nsight Systems ----------------------------------------------------------------

def page_fault_counts(sqlite_path) -> dict[str, int]:
    """Rows in every table of the export whose name contains PAGE_FAULT."""
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        names = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name LIKE '%PAGE_FAULT%'")]
        return {n: con.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0]
                for n in sorted(names)}
    finally:
        con.close()


def check_page_faults(export, control, *, arms: Sequence[str]) -> dict:
    """The page-fault verdict for one nsys pass (module docstring)."""
    if not arms:
        raise ValueError("name the arms the traced pass ran; the claim covers only them")
    target = page_fault_counts(export)
    ctrl = page_fault_counts(control)
    collected = sum(ctrl.values()) > 0
    faults = sum(target.values())
    if not collected:
        verdict = ("unknown: the positive control recorded no faults, so fault "
                   "collection is not shown to have been on")
    elif faults:
        verdict = f"page faults: {faults}"
    else:
        verdict = "no page faults"
    return {"source": "nsys", "export": str(export), "control": str(control),
            "arms": list(arms), "fault_tables": target, "control_fault_tables": ctrl,
            "collection_shown": collected, "page_faults": faults if collected else None,
            "claim": verdict + (f" (arms traced: {', '.join(arms)})" if collected else "")}


NSYS_PROFILE_FLAGS = ("--trace=cuda,nvtx", "--cuda-um-cpu-page-faults=true",
                      "--cuda-um-gpu-page-faults=true")


def nsys_available(which=None) -> Optional[str]:
    """Path of `nsys`, or None. Looks on PATH, then in the CUDA toolkit."""
    import shutil
    found = (which or shutil.which)("nsys")
    if found:
        return found
    for p in sorted(Path("/usr/local").glob("cuda*/bin/nsys")) + \
             sorted(Path("/opt/nvidia").glob("nsight-systems*/*/bin/nsys")):
        if p.exists():
            return str(p)
    return None
