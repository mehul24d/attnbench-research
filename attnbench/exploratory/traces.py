"""EXPLORATORY: a ground-truth Kineto trace -> its cell, its prefill region
and per-position device time.

The ground-truth traces hold one prefill each and carry no
`attnbench::prefill` annotation, so the region is taken from the prefill's
`aten::embedding` call to the timer's first `cudaDeviceSynchronize` after it.
Kineto drops some device records (the first activity after the profiler
starts, and a few more at long lengths); each launch in the region whose
device record is missing gets a zero-length placeholder, so the operation
sequence still parses, and its position is reported as unmeasured (None).
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from attnbench.components import op_sequence as ops

BUDGET = {2048: 1951, 4096: 4068, 8192: 8075, 16384: 16370, 32768: 32700}
NAME = re.compile(r"trace_L(\d+)_b(\d+)_(\d+)_(.+?)(?:_d([0-9.]+))?\.json$")
DEVICE_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
CAT_OF = {"cudaLaunchKernel": "kernel", "cuLaunchKernel": "kernel",
          "cudaMemcpyAsync": "gpu_memcpy", "cudaMemsetAsync": "gpu_memset"}
# The parsed family prefix -> the priced OpKind (memsets and device-to-device
# copies are priced as kernels).
OPKIND = {"kernel": "kernel", "memset": "kernel", "device_copy": "kernel",
          "graph": "kernel", "copy": "copy", "sync": "sync"}
# Operations a pairing's ground truth ran that its priced sequence lacks
# (P4's dense ground truth ran FlashAttention-3, which issues one fill_ per
# layer before its forward kernel).
UNPRICED = ("kernel:aten::fill_",)


class TraceError(ValueError):
    pass


def trace_key(name: str) -> tuple[int, int, str, Optional[float]]:
    """(realised length, batch, arm, nominal density or None) from a trace's
    file name."""
    m = NAME.search(name)
    if not m:
        raise TraceError(f"{name}: not a ground-truth trace name")
    L, b, _, arm, d = m.groups()
    return BUDGET[int(L)], int(b), arm, (float(d) if d else None)


def cell_for(cells, name: str):
    """The one cell a trace belongs to: arm, length, batch, and density
    where the name carries one (dense and XAttention names do not)."""
    L, b, arm, d = trace_key(name)
    hits = [c for c in cells if c.arm == arm and c.context_length == L and c.batch == b
            and ((d is None and (arm == "dense" or arm.startswith("xattention")))
                 or (d is not None and abs(c.density - d) < 1e-9))]
    if len(hits) != 1:
        raise TraceError(f"{name}: {len(hits)} cells match")
    return hits[0]


def window(tj: dict) -> tuple[dict, dict]:
    """(the prefill's embedding call, the timer's synchronise after it)."""
    ev = tj["traceEvents"]
    emb = [e for e in ev if e.get("name") == "aten::embedding" and e.get("cat") == "cpu_op"]
    if len(emb) != 1:
        raise TraceError(f"{len(emb)} embedding calls; one prefill per trace expected")
    syncs = sorted((e for e in ev if e.get("cat") == "cuda_runtime"
                    and e.get("name") == "cudaDeviceSynchronize" and e["ts"] > emb[0]["ts"]),
                   key=lambda e: e["ts"])
    if not syncs:
        raise TraceError("no synchronise after the embedding call")
    return emb[0], syncs[0]


def placeholders(tj: dict, lo: float, hi: float) -> tuple[list[dict], list]:
    """A zero-length device activity, on the busiest stream, for each launch
    in [lo, hi] whose device record the trace lacks; (events, correlations)."""
    ev = tj["traceEvents"]
    have, streams = set(), {}
    for e in ev:
        if e.get("cat") in DEVICE_CATS:
            a = e.get("args", {})
            have.add(a.get("correlation"))
            k = (a.get("device"), a.get("stream"))
            streams[k] = streams.get(k, 0) + 1
    if not streams:
        raise TraceError("no device activity in the trace")
    (dev, stream), _ = max(streams.items(), key=lambda kv: kv[1])
    add, corrs = [], []
    for e in ev:
        name = e.get("name", "")
        cat = next((v for k, v in CAT_OF.items() if name.startswith(k)), None)
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and cat and lo <= e["ts"] <= hi:
            c = e.get("args", {}).get("correlation")
            if c not in have:
                add.append({"ph": "X", "cat": cat, "name": "placeholder", "pid": dev,
                            "tid": stream, "ts": e["ts"], "dur": 0.0,
                            "args": {"correlation": c, "device": dev, "stream": stream}})
                corrs.append(c)
    return add, corrs


def with_region(tj: dict) -> tuple[dict, set]:
    """The trace with a synthetic prefill region and placeholders added;
    (trace, the placeholders' correlations)."""
    e0, s = window(tj)
    reg = {"ph": "X", "cat": "user_annotation", "name": ops.REGION, "pid": e0["pid"],
           "tid": e0["tid"], "ts": e0["ts"] - 0.01, "dur": s["ts"] - e0["ts"] + 0.005,
           "args": {}}
    add, corrs = placeholders(tj, e0["ts"], s["ts"])
    return {**tj, "traceEvents": tj["traceEvents"] + [reg] + add}, set(corrs)


def align(family: Sequence[str], device: Sequence[Optional[float]],
          priced_kinds: Sequence[str], unpriced: Sequence[str] = UNPRICED
          ) -> tuple[list[str], list[Optional[float]], int]:
    """Traced positions laid onto the priced sequence.

    When the trace is longer than the priced sequence, the operations named
    in `unpriced` are removed and their device time is folded into the next
    operation (FA3's fill_ precedes its forward kernel). The result must then
    match the priced kinds one for one; anything else raises. Returns
    (families, device times, operations folded)."""
    fam, dev, folded = list(family), list(device), 0
    if len(fam) != len(dev):
        raise TraceError(f"{len(fam)} families vs {len(dev)} device times")
    if len(fam) > len(priced_kinds):
        f2, d2, carry = [], [], 0.0
        for f, d in zip(fam, dev):
            if f in unpriced:
                carry += d or 0.0
                folded += 1
                continue
            f2.append(f)
            d2.append(None if d is None else d + carry)
            carry = 0.0
        if carry:
            raise TraceError("an unpriced operation at the end of the sequence")
        fam, dev = f2, d2
    if len(fam) != len(priced_kinds):
        raise TraceError(f"{len(fam)} traced vs {len(priced_kinds)} priced operations")
    bad = [i for i, (f, k) in enumerate(zip(fam, priced_kinds))
           if OPKIND[f.split(":", 1)[0]] != k]
    if bad:
        raise TraceError(f"kind differs at {len(bad)} positions, first {bad[0]}")
    return fam, dev, folded
