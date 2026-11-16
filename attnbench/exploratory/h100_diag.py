"""EXPLORATORY: the H100 diagnostic session (adviser's rulings of 17 Nov 2026,
item 2). It explains P4's miss and scores nothing; the H0 to H3 verdicts, the
analysis commit 83b2edf and the timed-path pin are untouched.

Pure functions only (the GPU runner is `scripts/h100_diag.py`):

  - the cell list, checked (at most 8 cells, batch 1, one CUDA-graph cell
    with its eager twin);
  - device seconds per model component, from a sequence's families and a
    trace's per-position durations;
  - host timeline statistics of one prefill (wall, device busy, idle, host
    submit time, time blocked in synchronising calls), from Kineto events or
    from an Nsight Systems SQLite export;
  - the four outcome rules, declared before launch with their thresholds
    (`docs/exploratory/h100_diag_card.md`).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from attnbench.exploratory.breakdown import COMPONENTS, component_of

SCHEMA = "h100-diag-cells-1"
MAX_CELLS = 8
LENGTHS = (8192, 16384, 32768)
DENSE_BACKENDS = ("fa3", "sdpa_flash")
MODES = ("eager", "graph")
NVTX_PREFIX = "h100diag:prefill:"

# The outcome rules' thresholds (card, "What each outcome means").
SAME = 0.10            # Q1, Q2, Q3: "the same" is within 10%
GRAPH_EXPLAINS = 2.0   # Q2: the graph alone explains most of P4's 2.52x at 16370
MOST = 0.5             # Q4: at least half of the residual explained
LITTLE = 0.2           # Q4: at most a fifth explained
TRACED_WALL_MAX = 1.0 + SAME   # Q4: a traced wall above 1.10x P4's untraced one is not comparable

LAUNCHES = ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC", "cudaMemcpyAsync",
            "cudaMemsetAsync", "cudaGraphLaunch")
BLOCKING = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize",
            "cudaMemcpy", "cudaMemcpy2D", "cuStreamSynchronize", "cuCtxSynchronize")
DEVICE_CATS = ("kernel", "gpu_memcpy", "gpu_memset")


class DiagError(ValueError):
    pass


@dataclass(frozen=True)
class Cell:
    id: str
    arm: str                     # dense | xattention
    backend: Optional[str]       # dense: fa3 | sdpa_flash; xattention: None (Triton)
    length: int                  # the budget; P4's prompts are 8075, 16370, 32700 tokens
    batch: int
    mode: str                    # eager | graph


def load_cells(path) -> tuple[tuple[Cell, ...], str]:
    """The cell list and its sha256; refused unless every rule holds."""
    raw = Path(path).read_bytes()
    d = json.loads(raw)
    if d.get("schema") != SCHEMA:
        raise DiagError(f"schema {d.get('schema')!r}, not {SCHEMA}")
    cells = tuple(Cell(**c) for c in d["cells"])
    check_cells(cells)
    return cells, hashlib.sha256(raw).hexdigest()


def check_cells(cells: Sequence[Cell]) -> None:
    if not 0 < len(cells) <= MAX_CELLS:
        raise DiagError(f"{len(cells)} cells; 1 to {MAX_CELLS} allowed")
    if len({c.id for c in cells}) != len(cells):
        raise DiagError("cell ids repeat")
    for c in cells:
        if c.batch != 1:
            raise DiagError(f"{c.id}: batch {c.batch}; batch 1 only")
        if c.length not in LENGTHS:
            raise DiagError(f"{c.id}: length {c.length} not in {LENGTHS}")
        if c.mode not in MODES:
            raise DiagError(f"{c.id}: mode {c.mode!r}")
        if c.arm == "dense":
            if c.backend not in DENSE_BACKENDS:
                raise DiagError(f"{c.id}: dense backend {c.backend!r} not in {DENSE_BACKENDS}")
        elif c.arm == "xattention":
            if c.backend is not None or c.mode != "eager":
                raise DiagError(f"{c.id}: XAttention runs eager on its own backend")
        else:
            raise DiagError(f"{c.id}: arm {c.arm!r}")
    graphs = [c for c in cells if c.mode == "graph"]
    if len(graphs) > 1:
        raise DiagError(f"{len(graphs)} CUDA-graph cells; one comparison allowed")
    for g in graphs:
        if not any(c.mode == "eager" and (c.arm, c.backend, c.length) == (g.arm, g.backend, g.length)
                   for c in cells):
            raise DiagError(f"{g.id}: no eager twin to compare the graph with")


def component_seconds(families: Sequence[str], durations: Mapping[int, float]) -> dict:
    """component -> device seconds over the positions that have a duration,
    plus how many device positions had none."""
    out = {c: 0.0 for c in COMPONENTS}
    missing = 0
    for i, f in enumerate(families):
        comp = component_of(f)
        if comp is None:
            continue
        d = durations.get(i)
        if d is None:
            missing += 1
            continue
        out[comp] += float(d)
    return {**out, "total": sum(out[c] for c in COMPONENTS), "positions_without_duration": missing}


def _union(intervals: Iterable[tuple[float, float]], lo: float, hi: float) -> float:
    """Length of the union of intervals clipped to [lo, hi]."""
    xs = sorted((max(a, lo), min(b, hi)) for a, b in intervals if b > lo and a < hi)
    total, cur_a, cur_b = 0.0, None, None
    for a, b in xs:
        if cur_b is None or a > cur_b:
            if cur_b is not None:
                total += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    if cur_b is not None:
        total += cur_b - cur_a
    return total


def host_stats(start: float, end: float, launches: Sequence[tuple[float, float]],
               blocking: Sequence[tuple[float, float]],
               device: Sequence[tuple[float, float]]) -> dict:
    """One prefill's timeline, in seconds. `end` is when the prefill's last
    device work finished (or its closing synchronise returned). Host time is
    the submit time (start to the last launch call's return) less the time
    spent inside synchronising calls before it."""
    if end <= start:
        raise DiagError("the prefill ends before it starts")
    wall = end - start
    busy = _union(device, start, end)
    last = max((b for a, b in launches if a >= start), default=start)
    blocked = _union(blocking, start, last)
    submit = last - start
    return {"wall_s": wall, "device_busy_s": busy, "idle_s": wall - busy,
            "idle_share": (wall - busy) / wall, "submit_s": submit,
            "blocked_s": blocked, "host_s": submit - blocked,
            "launches": sum(1 for a, _ in launches if start <= a <= end)}


def kineto_host(trace: Mapping, region: str) -> dict:
    """`host_stats` of the one `region` annotation of a Kineto trace (the
    torch.profiler fallback; its host times carry the profiler's overhead)."""
    ev = trace["traceEvents"]
    regs = [e for e in ev if e.get("name") == region and e.get("cat") == "user_annotation"]
    if len(regs) != 1:
        raise DiagError(f"{len(regs)} '{region}' annotations; one expected")
    r = regs[0]
    lo, hi = r["ts"], r["ts"] + r["dur"]
    rt = [e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver") and lo <= e["ts"] <= hi]
    launches = [(e["ts"], e["ts"] + e["dur"]) for e in rt if _is_launch(e["name"])]
    blocking = [(e["ts"], e["ts"] + e["dur"]) for e in rt if e["name"] in BLOCKING]
    device = [(e["ts"], e["ts"] + e["dur"]) for e in ev if e.get("cat") in DEVICE_CATS
              and e["ts"] + e["dur"] > lo]
    end = max([hi] + [b for _, b in device])
    s = host_stats(lo, end, launches, blocking, device)
    return {k: (v * 1e-6 if k.endswith("_s") else v) for k, v in s.items()}


def _is_launch(name: str) -> bool:
    return any(name.startswith(p) for p in LAUNCHES)


def nsys_prefills(db_path, prefix: str = NVTX_PREFIX) -> list[dict]:
    """Per NVTX range named `prefix<i>` in an `nsys export --type=sqlite`
    file: `host_stats` (the range closes after the prefill's synchronise, so
    its end is the prefill's end) and the device seconds by kernel name."""
    con = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for t in ("NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_RUNTIME",
                  "StringIds"):
            if t not in tables:
                raise DiagError(f"the export has no {t} table")
        s = "StringIds"
        ranges = con.execute(
            f"SELECT n.start, n.end, COALESCE(n.text, t.value) FROM NVTX_EVENTS n "
            f"LEFT JOIN {s} t ON n.textId = t.id WHERE n.end IS NOT NULL ORDER BY n.start").fetchall()
        ranges = [(a, b, x) for a, b, x in ranges if x and x.startswith(prefix)]
        if not ranges:
            raise DiagError(f"no NVTX range named {prefix}<i>")
        kern = con.execute(f"SELECT k.start, k.end, t.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                           f"JOIN {s} t ON k.shortName = t.id").fetchall()
        other = []
        for t in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
            if t in tables:
                other += con.execute(f"SELECT start, end FROM {t}").fetchall()
        rt = con.execute(f"SELECT r.start, r.end, t.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                         f"JOIN {s} t ON r.nameId = t.id").fetchall()
    finally:
        con.close()
    out = []
    for a, b, text in ranges:
        launches = [(x, y) for x, y, n in rt if a <= x <= b and _is_launch(_base(n))]
        blocking = [(x, y) for x, y, n in rt if a <= x <= b and _base(n) in BLOCKING]
        device = [(x, y) for x, y, _ in kern if y > a and x < b] + \
                 [(x, y) for x, y in other if y > a and x < b]
        st = host_stats(a, b, launches, blocking, device)
        names: dict[str, float] = {}
        for x, y, n in kern:
            if a <= x < b:
                names[n] = names.get(n, 0.0) + (y - x) * 1e-9
        out.append({"range": text, **{k: (v * 1e-9 if k.endswith("_s") else v)
                                      for k, v in st.items()}, "kernels_s": names})
    return out


def _base(name: str) -> str:
    """A runtime API name without nsys's version suffix (`cudaLaunchKernel_v7000`)."""
    return name.split("_v", 1)[0]


def median_of(rows: Sequence[Mapping], key: str) -> float:
    if not rows:
        raise DiagError("no rows")
    return statistics.median(float(r[key]) for r in rows)


# --- the outcome rules (declared before launch) ---------------------------------------

def q1_priced_kernel(off_attention_s: float, priced_attention_s: float) -> dict:
    """Q1. Is the dense attention P4 priced at 16370 tokens the FA3-off
    kernel (`sdpa_flash`, the harness's dense backend)? The ratio of this
    session's eager `sdpa_flash` attention seconds to P4's priced ones."""
    if off_attention_s <= 0 or priced_attention_s <= 0:
        raise DiagError("Q1: no attention seconds to compare")
    r = off_attention_s / priced_attention_s
    ok = abs(r - 1.0) <= SAME
    verdict = ("confirmed: P4 priced the sdpa_flash kernel; the ground truth ran FA3" if ok else
               "not confirmed: the priced attention is not sdpa_flash's eager time")
    return {"ratio": r, "verdict": verdict, "confirmed": ok}


def q2_graph(graph_attention_s: float, eager_attention_s: float) -> dict:
    """Q2. Does step 6's CUDA-graph replay price FA3 as it runs eager? The
    ratio of FA3's attention seconds in the replay to the eager twin's."""
    if graph_attention_s <= 0 or eager_attention_s <= 0:
        raise DiagError("Q2: no attention seconds to compare")
    g = graph_attention_s / eager_attention_s
    if abs(g - 1.0) <= SAME:
        v = "faithful: the graph replay prices FA3 as it runs eager"
    elif g >= GRAPH_EXPLAINS:
        v = "the graph replay alone inflates FA3 enough to explain most of P4's 2.52x"
    elif g > 1.0 + SAME:
        v = "the graph replay inflates FA3, but less than P4's misprice"
    else:
        v = "the graph replay deflates FA3"
    return {"ratio": g, "verdict": v, "faithful": abs(g - 1.0) <= SAME}


def q3_comparable(pairs: Mapping[str, tuple[float, float]]) -> dict:
    """Q3 at one length. Is this session P4's device? name -> (this session,
    P4). Each within 10%; otherwise Q1 (16370 tokens) or Q4 at this length,
    which use P4's numbers, are read as not comparable. Q2 (within this
    session) is not gated. No pairs (the cells did not pass) is not comparable."""
    rows = {k: {"this": a, "p4": b, "ratio": a / b, "within": abs(a / b - 1.0) <= SAME}
            for k, (a, b) in pairs.items()}
    return {"rows": rows, "comparable": bool(rows) and all(r["within"] for r in rows.values())}


def q4_host(traced_wall_s: float, p4_wall_s: float, traced_host_s: float) -> dict:
    """Q4's measured host at one length. The tracer's overhead is the traced
    wall minus P4's untraced wall (at least zero), taken off the traced host
    before the thresholds; a traced wall above 1.10x P4's is not comparable."""
    if traced_wall_s <= 0 or p4_wall_s <= 0:
        raise DiagError("Q4: no wall to take the tracing overhead from")
    over = max(0.0, traced_wall_s - p4_wall_s)
    return {"traced_wall_s": traced_wall_s, "p4_wall_s": p4_wall_s,
            "wall_ratio": traced_wall_s / p4_wall_s, "overhead_s": over,
            "host_traced_s": traced_host_s, "host_s": traced_host_s - over,
            "comparable": traced_wall_s <= TRACED_WALL_MAX * p4_wall_s}


def q4_fraction(predicted_all_replaced: float, predicted_host_scaled: float,
                measured: float) -> dict:
    """Q4. How much of an XAttention cell's residual the measured host time
    explains: phi = (R_all - R_host) / (R_all - M), where R_all is the
    recursion with every priced device time replaced by P4's traced one,
    R_host the same with each priced host cost scaled by h / H (h the measured
    host per prefill after `q4_host`, H the cell's summed priced host) and M
    P4's measured wall. The card's host thresholds are this phi solved for
    0.5 and 0.2 on P4's data."""
    resid = predicted_all_replaced - measured
    if resid == 0:
        raise DiagError("no residual to explain")
    phi = (predicted_all_replaced - predicted_host_scaled) / resid
    if phi >= MOST:
        v = "the host costs are overstated: they explain at least half of the residual"
    elif phi <= LITTLE:
        v = "not the host costs: the residual is in the overlap, queue or sync costs"
    else:
        v = "partly the host costs"
    return {"residual_rel": resid / measured, "fraction": phi, "verdict": v}
