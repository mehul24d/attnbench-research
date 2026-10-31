"""Memsets with no device record: count, and an upper-bound sensitivity
(researcher's ruling, 31 Oct 2026).

Shakedown 2's XAttention traces held 84 `cudaMemsetAsync` calls, three per
layer under `addmm`, with no device activity. `op_sequence.parse_trace`
keeps each one as an operation flagged `NO_DEVICE_RECORD`: its host cost is
counted and its device cost is zero. This module gives the two numbers the
paper reports beside that choice:

  - `count`: the flagged memsets in a sequence;
  - the shift in the prediction when each flagged memset is priced instead
    at the memset zoo's measured duration for its size (`upper_bound_s`,
    `prediction_shift`). That is an upper bound: a memset with no device
    record did no more work than the zoo's memset of the same size.

**Sizes.** The host call carries no size; only a device activity has
`bytes`. A flagged memset takes the largest size recorded by a memset under
the same aten operator in the same trace (`flagged_sizes`); where there is
none, it is priced at the zoo's largest memset. Both choices can only raise
the bound.

**The memset zoo.** `run_memset_zoo` issues `cudaMemsetAsync` itself (the
call cuBLAS makes), at `ZOO_SIZES`, under the profiler, and `zoo_durations`
reads each size's longest device duration from that trace. UNRUN on a GPU.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

from . import op_sequence as ops
from .op_sequence import OpSequence, SeqKind

ZOO_SIZES = (64, 256, 1024, 4096, 16384, 65536, 262144, 1 << 20, 1 << 22)
ZOO_REPEATS = 5
LARGEST = "largest zoo memset"


def flagged(seq: OpSequence) -> tuple[int, ...]:
    """Positions of the memsets with no device record."""
    return tuple(i for i, op in enumerate(seq.ops) if op.no_device_record)


def count(seq: OpSequence) -> int:
    return len(flagged(seq))


def _memset_events(trace) -> tuple[list, dict]:
    """The trace's memset host calls in issue order, and device bytes by
    correlation."""
    events = ops._events(ops.load_trace(trace))
    reg = ops._region(events, ops.REGION, None)
    calls = sorted((e for e in events if e.cat in ops._HOST_CATS
                    and ops.api_class(e.name) == "memset"
                    and reg.ts <= e.ts <= reg.end), key=lambda e: (e.ts, e.idx))
    nbytes = {}
    for e in events:
        if e.cat == "gpu_memset":
            c, b = ops._int_arg(e.args, "correlation"), ops._int_arg(e.args, "bytes")
            if c is not None and b is not None:
                nbytes[c] = b
    return calls, nbytes


def flagged_sizes(trace, seq: OpSequence) -> dict[int, Optional[int]]:
    """Per flagged position, the bytes to price it at (module docstring),
    or None. The k-th memset of the sequence is the k-th memset call of the
    trace, in issue order: a recording read back keeps no correlation ids."""
    calls, nbytes = _memset_events(trace)
    positions = [i for i, op in enumerate(seq.ops) if op.kind is SeqKind.MEMSET]
    if len(calls) != len(positions):
        raise ops.SequenceError(f"the trace's region has {len(calls)} memset calls, the "
                                f"sequence {len(positions)} memsets; not the same recording")
    known: dict = {}
    for pos, call in zip(positions, calls):
        b = nbytes.get(ops._int_arg(call.args, "correlation"))
        if b is not None:
            key = seq.ops[pos].aten_op
            known[key] = max(known.get(key, 0), b)
    return {i: known.get(seq.ops[i].aten_op) for i in flagged(seq)}


def zoo_durations(trace) -> dict[int, float]:
    """Bytes -> the longest device duration (seconds) of the zoo's memsets
    of that size."""
    out: dict = {}
    for e in ops._events(ops.load_trace(trace)):
        if e.cat == "gpu_memset":
            b = ops._int_arg(e.args, "bytes")
            if b is not None:
                out[b] = max(out.get(b, 0.0), (e.end - e.ts) * 1e-6)
    if not out:
        raise ops.SequenceError("the memset zoo's trace holds no memset with a size")
    return dict(sorted(out.items()))


def upper_bound_s(sizes: Mapping[int, Optional[int]],
                  zoo: Mapping[int, float]) -> tuple[dict, dict]:
    """(position -> seconds, position -> source): the zoo's duration at the
    smallest zoo size not below the memset's, else the zoo's largest."""
    if not zoo:
        raise ValueError("an empty memset zoo prices nothing")
    biggest = max(zoo.values())
    secs, source = {}, {}
    for i, b in sizes.items():
        at = [z for z in sorted(zoo) if b is not None and z >= b]
        if at:
            secs[i], source[i] = zoo[at[0]], f"zoo at {at[0]} bytes"
        else:
            secs[i], source[i] = biggest, LARGEST
    return secs, source


def prediction_shift(seq: OpSequence, *, host, device, link,
                     predict: Callable[[list], float],
                     upper: Mapping[int, float]) -> dict:
    """The prediction with flagged memsets at zero device cost (the model's)
    and at `upper` (the bound), and the shift between them. `predict` takes
    the recursion operations and returns D_N in seconds."""
    base = predict(ops.to_operations(seq, host=host, device=device, link=link))
    if not flagged(seq):
        return {"flagged_memsets": 0, "prediction_s": base, "upper_bound_s": base,
                "shift_s": 0.0}
    bound = predict(ops.to_operations(seq, host=host, device=device, link=link,
                                      flagged_memset_s=upper))
    return {"flagged_memsets": len(upper), "prediction_s": base, "upper_bound_s": bound,
            "shift_s": bound - base}


# --- the memset zoo (GPU) ----------------------------------------------------

def _cudart():
    import ctypes
    import ctypes.util
    import torch
    names = [os.path.join(os.path.dirname(torch.__file__), "lib", "libcudart.so*")]
    try:
        import nvidia.cuda_runtime as nv
        names.append(os.path.join(list(nv.__path__)[0], "lib", "libcudart.so*"))
    except ImportError:
        pass
    for pattern in names:
        for path in sorted(glob.glob(pattern)):
            return ctypes.CDLL(path)
    found = ctypes.util.find_library("cudart")
    if found:
        return ctypes.CDLL(found)
    raise RuntimeError("libcudart was not found; the memset zoo needs the CUDA runtime")


def run_memset_zoo(device="cuda", sizes: Sequence[int] = ZOO_SIZES,
                   repeats: int = ZOO_REPEATS) -> None:
    """`cudaMemsetAsync` at each size, `repeats` times, on the current
    stream: the prefill `record_prefill_trace` profiles. UNRUN on a GPU."""
    import ctypes
    import torch
    lib = _cudart()
    stream = torch.cuda.current_stream(device).cuda_stream
    buf = torch.empty(max(sizes), dtype=torch.uint8, device=device)
    for n in sizes:
        for _ in range(repeats):
            rc = lib.cudaMemsetAsync(ctypes.c_void_p(buf.data_ptr()), ctypes.c_int(0),
                                     ctypes.c_size_t(n), ctypes.c_void_p(stream))
            if rc != 0:
                raise RuntimeError(f"cudaMemsetAsync returned {rc} at {n} bytes")
    torch.cuda.synchronize(device)


def measure_zoo(path: Union[str, Path], device="cuda") -> dict[int, float]:
    """Record the memset zoo under the profiler and read its durations."""
    from . import trace_recording as tr
    trace = tr.record_prefill_trace(lambda: run_memset_zoo(device), path, warmup=1, cuda=True)
    return zoo_durations(trace)
