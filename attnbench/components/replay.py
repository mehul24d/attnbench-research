"""Device components by replay (T2.4; plan Sections 5.1, 5.3 step 5, 5.6).

c_dev(i) is "GPU execution time of operation i, from replaying the arm's
whole kernel sequence in order on the target GPU, with the masks recorded
for its code path, as a CUDA graph (primary; keeps cache state between
kernels and has no host gaps). Isolated per-operation replay is the
cross-check, and the difference between the two is reported as the cache
effect."

**In-order replay.** The arm's own prefill code is captured as one CUDA
graph, on the recorded prompt, with two things held fixed from the
recording (`recorder.record_arm_prefill`):

  - every host-device interaction, served from the tape (`sync_tape`), so
    the capture holds no synchronisation and no host-to-device copy;
  - the mask the block-sparse kernel receives at each call, the banked one
    (`mask_bank.MaskCapture(substitute=...)`). The estimator's own kernels
    still run, so their device cost is in the replay; the mask they produce
    is kept and compared with the bank (`mask_agreement`), which must be
    bitwise on the code path that recorded it.

The graph is replayed under the profiler once to read every node's device
time (`replay_nodes`), and without it `reps` times with CUDA events for the
total (`time_replay`), so the profiler's effect on the node times is
visible as the ratio of the two.

**Positions.** `match` assigns replayed nodes to the positions of the
recorded sequence by kernel name, in order (`op_sequence.OpSequence.
device_ops`). A host-device copy and a synchronisation have no device cost
in the recursion (l_i and s_0 are T2.5's), and the tape keeps them out of
the graph. Whatever does not match is reported: positions with no node
(the tape's substitutions, `nonzero_static` for `nonzero`, are the expected
case) and nodes with no position. `device_costs` refuses an incomplete
match, so a position never gets a device cost from a different kernel.

**Isolated replay.** The same prefill, eagerly, from the tape and the bank,
with the device synchronised and the L2 cache flushed before every aten
operator (`IsolationMode`), under the profiler. Each operation then runs
alone on a cold cache. Kernels that no aten operator issues (a Triton kernel
called from Python) are not separated from their neighbours. The flush's own
kernels are excluded by their annotation.

**The replay-versus-trace check** (Gate A, Section 5.6 "Replay mismatch")
compares replayed device time with the device time in a recorded trace, per
operation family (`op_sequence.operation_family`), and reports kernel-name
mismatches. The bound is fixed at lock, so `replay_vs_trace` reports and
`check_passes` takes the bound as an argument.

The graph is behind `GraphBackend` so the bookkeeping is tested on CPU with
`EagerBackend`; `CudaGraphBackend` is the real one.
"""

from __future__ import annotations

import difflib
import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

import torch
from torch.autograd.profiler import record_function
from torch.utils._python_dispatch import TorchDispatchMode

from . import mask_bank, op_sequence
from .op_sequence import OpSequence, SeqKind, operation_family
from .sync_tape import SyncTape, TapeReplayer

REPLAY_REGION = "attnbench::replay"
FLUSH = "attnbench::isolation_flush"
SCHEMA = "attnbench.replay/1"


class ReplayError(RuntimeError):
    pass


# --- backends ------------------------------------------------------------------

class CudaGraphBackend:
    """Capture with `torch.cuda.graph`; time with CUDA events."""

    def capture(self, body: Callable[[], None]):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            body()
        return g

    def warmup(self, body: Callable[[], None]) -> None:
        """PyTorch's recipe: warm up on a side stream before capture, so lazy
        initialisation is not captured."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            body()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

    def replay(self, g) -> None:
        g.replay()

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def time_once(self, g) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        g.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1e-3


class EagerBackend:
    """Stands in for the graph on CPU: "capture" runs the body once and keeps
    it; "replay" runs it again, under a fresh tape replayer each time."""

    def __init__(self):
        self.replays = 0

    def capture(self, body):
        body()
        return body

    def warmup(self, body) -> None:
        body()

    def replay(self, body) -> None:
        self.replays += 1
        body()

    def synchronize(self) -> None:
        pass

    def time_once(self, body) -> float:
        t0 = time.perf_counter()
        self.replay(body)
        return time.perf_counter() - t0


# --- capture -----------------------------------------------------------------------

@dataclass
class Captured:
    handle: object
    backend: object
    tape: SyncTape
    banked: Optional[list[torch.Tensor]]
    live_masks: list[torch.Tensor] = field(default_factory=list)
    staged: dict = field(default_factory=dict)

    def replay(self) -> None:
        self.backend.replay(self.handle)


def _bank_capture(banked, kernel_module) -> mask_bank.MaskCapture:
    return mask_bank.MaskCapture(module=kernel_module, substitute=banked,
                                 optional=banked is None)


def _check_calls(cap: mask_bank.MaskCapture, banked) -> None:
    n = len(banked) if banked is not None else 0
    if len(cap.calls) != n:
        raise ReplayError(f"the replay made {len(cap.calls)} kernel calls; "
                          f"{n} masks are banked")


def _body(run_prefill, tape, staged, banked, kernel_module, sink: list):
    def body():
        rep = TapeReplayer(tape, staged)
        with rep, _bank_capture(banked, kernel_module) as cap:
            run_prefill()
        rep.finish()
        _check_calls(cap, banked)
        sink[:] = cap.live_masks()
    return body


def capture(run_prefill: Callable[[], None], tape: SyncTape, *, device,
            masks: Optional[mask_bank.MaskRecord] = None, backend=None,
            warmup: int = 1, kernel_module=None) -> Captured:
    """The arm's prefill as one graph, from the tape and the bank. The modes
    are entered inside the capture, so the backend's own synchronisation
    around it is not taken for the arm's."""
    backend = backend or CudaGraphBackend()
    if tape.unhandled:
        raise ReplayError(f"the tape has unhandled interactions at {tape.unhandled}: "
                          f"{tape.entries[tape.unhandled[0]].note}")
    staged = tape.stage(device)
    banked = masks.tensors(device) if masks is not None else None
    live: list = []
    body = _body(run_prefill, tape, staged, banked, kernel_module, live)
    for _ in range(warmup):
        backend.warmup(body)
    handle = backend.capture(body)
    return Captured(handle, backend, tape, banked, live, staged)


def mask_agreement(cap: Captured) -> Optional[dict]:
    """The estimator's live masks after a replay against the bank."""
    if cap.banked is None:
        return None
    cap.backend.synchronize()
    return mask_bank.agreement(cap.live_masks, cap.banked)


def time_replay(cap: Captured, reps: int) -> dict:
    """Total device seconds per replay, unprofiled, over `reps` replays."""
    if reps < 1:
        raise ValueError("reps must be >= 1")
    xs = sorted(cap.backend.time_once(cap.handle) for _ in range(reps))
    return {"reps": reps, "median_s": statistics.median(xs), "min_s": xs[0],
            "p90_s": xs[min(len(xs) - 1, int(0.9 * len(xs)))]}


def profile_replay(cap: Captured, path) -> Path:
    """One replay under the profiler, inside REPLAY_REGION."""
    from torch.profiler import ProfilerActivity, profile
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
    cap.backend.synchronize()
    with profile(activities=acts) as prof:
        with record_function(REPLAY_REGION):
            cap.replay()
        cap.backend.synchronize()
    prof.export_chrome_trace(str(path))
    return Path(path)


# --- reading replayed nodes ----------------------------------------------------------

@dataclass(frozen=True)
class Node:
    cat: str
    name: str
    duration_s: float
    correlation: Optional[int]


def replay_nodes(trace: Union[str, Path, Mapping], *, region: str = REPLAY_REGION,
                 exclude: Optional[str] = None) -> list[Node]:
    """Device activities of the host calls inside `region`, in device order;
    those whose host call sits inside an `exclude` annotation are dropped."""
    trace = op_sequence.load_trace(trace)
    events = op_sequence._events(trace)
    regions = [e for e in events if e.cat == "user_annotation" and e.name == region]
    if len(regions) != 1:
        raise ReplayError(f"{len(regions)} {region!r} regions in the trace, expected one")
    reg = regions[0]
    inside = lambda e: e.ts >= reg.ts - op_sequence._EPS and e.end <= reg.end + op_sequence._EPS
    skips = [e for e in events if exclude and e.cat == "user_annotation" and e.name == exclude]
    keep: set = set()
    for e in events:
        if e.cat in op_sequence._HOST_CATS and inside(e) and (e.pid, e.tid) == (reg.pid, reg.tid):
            if any(s.ts <= e.ts and e.end <= s.end for s in skips):
                continue
            c = op_sequence._int_arg(e.args, "correlation")
            if c is not None:
                keep.add(c)
    acts = [e for e in events if e.cat in op_sequence._DEVICE_CATS
            and op_sequence._int_arg(e.args, "correlation") in keep]
    acts.sort(key=lambda e: e.ts)
    return [Node(e.cat, e.name, (e.end - e.ts) * 1e-6, op_sequence._int_arg(e.args, "correlation"))
            for e in acts]


# --- matching nodes to positions -------------------------------------------------------

@dataclass(frozen=True)
class Match:
    device_s: dict                 # position -> device seconds (matched nodes only)
    unmatched_positions: tuple     # positions with at least one kernel not replayed
    extra_nodes: tuple             # replayed node names with no position
    families_unmatched: dict       # family -> count of unmatched positions
    n_expected: int
    n_nodes: int

    @property
    def complete(self) -> bool:
        return not self.unmatched_positions and not self.extra_nodes


def expected_nodes(seq: OpSequence) -> list[tuple[int, str]]:
    """(position, activity name) for every device activity of the sequence."""
    return [(i, name) for i, op in seq.device_ops() for name in op.kernels]


def match(seq: OpSequence, nodes: Sequence[Node]) -> Match:
    exp = expected_nodes(seq)
    a = [name for _, name in exp]
    b = [n.name for n in nodes]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    got = [False] * len(a)
    used = [False] * len(b)
    device: dict = defaultdict(float)
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            pos = exp[blk.a + k][0]
            device[pos] += nodes[blk.b + k].duration_s
            got[blk.a + k] = used[blk.b + k] = True
    missing = sorted({exp[k][0] for k in range(len(a)) if not got[k]})
    fam: dict = defaultdict(int)
    for p in missing:
        fam[operation_family(seq.ops[p])] += 1
    for p in missing:
        device.pop(p, None)
    return Match(dict(device), tuple(missing), tuple(b[k] for k in range(len(b)) if not used[k]),
                 dict(fam), len(a), len(b))


def device_costs(m: Match) -> dict[int, float]:
    """d_i by sequence position, for `op_sequence.to_operations(device=...)`."""
    if not m.complete:
        raise ReplayError(
            f"the replay does not match the sequence: {len(m.unmatched_positions)} "
            f"positions unmatched ({m.families_unmatched}), {len(m.extra_nodes)} extra "
            f"nodes; no device cost is taken from a different kernel")
    return dict(m.device_s)


# --- isolated replay ---------------------------------------------------------------------

class IsolationMode(TorchDispatchMode):
    """Synchronise and flush the L2 cache before every aten operator."""

    def __init__(self, flush: Callable[[], None], sync: Callable[[], None]):
        super().__init__()
        self.flush, self.sync = flush, sync
        self.operators = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.operators += 1
        with record_function(FLUSH):
            self.sync()
            self.flush()
            self.sync()
        return func(*args, **(kwargs or {}))


def l2_flush(device, nbytes: Optional[int] = None) -> Callable[[], None]:
    """A write over twice the L2 cache (from the device's properties)."""
    if nbytes is None:
        props = torch.cuda.get_device_properties(device)
        nbytes = 2 * int(getattr(props, "L2_cache_size", 0) or (64 << 20))
    buf = torch.empty(nbytes, dtype=torch.uint8, device=device)
    return lambda: buf.zero_()


def isolated_replay(run_prefill: Callable[[], None], tape: SyncTape, path, *, device,
                    masks: Optional[mask_bank.MaskRecord] = None,
                    flush: Optional[Callable[[], None]] = None,
                    sync: Optional[Callable[[], None]] = None,
                    kernel_module=None, profile=True) -> Path:
    """The prefill once, eagerly, each aten operator alone on a cold cache,
    under the profiler; returns the trace. Read it with `replay_nodes(...,
    exclude=FLUSH)`."""
    from torch.profiler import ProfilerActivity, profile as _profile
    sync = sync or torch.cuda.synchronize          # taken before the tape patches it
    flush = flush or l2_flush(device)
    staged = tape.stage(device)
    banked = masks.tensors(device) if masks is not None else None
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
    sync()
    with _profile(activities=acts) as prof:
        with record_function(REPLAY_REGION):
            rep = TapeReplayer(tape, staged)
            with rep, _bank_capture(banked, kernel_module) as cap, IsolationMode(flush, sync):
                run_prefill()
            rep.finish()
            _check_calls(cap, banked)
        sync()
    prof.export_chrome_trace(str(path))
    return Path(path)


# --- reports -------------------------------------------------------------------------------

def by_family(seq: OpSequence, per_position: Mapping[int, float]) -> dict[str, dict]:
    out: dict = defaultdict(lambda: {"n": 0, "s": 0.0})
    for i, v in per_position.items():
        f = operation_family(seq.ops[i])
        out[f]["n"] += 1
        out[f]["s"] += v
    return dict(out)


def cache_effect(seq: OpSequence, in_order: Mapping[int, float],
                 isolated: Mapping[int, float]) -> dict:
    """Isolated minus in-order device time, per position, family and total."""
    common = sorted(set(in_order) & set(isolated))
    fam_in, fam_iso = by_family(seq, {i: in_order[i] for i in common}), \
        by_family(seq, {i: isolated[i] for i in common})
    tot_in = sum(in_order[i] for i in common)
    tot_iso = sum(isolated[i] for i in common)
    return {
        "positions": len(common),
        "missing_in_order": sorted(set(isolated) - set(in_order)),
        "missing_isolated": sorted(set(in_order) - set(isolated)),
        "in_order_s": tot_in, "isolated_s": tot_iso,
        "effect_s": tot_iso - tot_in,
        "isolated_over_in_order": tot_iso / tot_in if tot_in > 0 else None,
        "families": {f: {"n": fam_in[f]["n"], "in_order_s": fam_in[f]["s"],
                         "isolated_s": fam_iso[f]["s"],
                         "effect_s": fam_iso[f]["s"] - fam_in[f]["s"]} for f in fam_in},
    }


def replay_vs_trace(seq: OpSequence, trace_s: Mapping[int, float],
                    replay_s: Mapping[int, float], m: Optional[Match] = None) -> dict:
    """Per operation family: positions, trace device time, replayed device
    time, and their relative difference (replay - trace) / trace."""
    common = sorted(set(trace_s) & set(replay_s))
    ft = by_family(seq, {i: trace_s[i] for i in common})
    fr = by_family(seq, {i: replay_s[i] for i in common})
    fams = {f: {"n": ft[f]["n"], "trace_s": ft[f]["s"], "replay_s": fr[f]["s"],
                "relative_difference": ((fr[f]["s"] - ft[f]["s"]) / ft[f]["s"]
                                        if ft[f]["s"] > 0 else None)} for f in ft}
    return {"families": fams,
            "positions_compared": len(common),
            "positions_without_replay": sorted(set(trace_s) - set(replay_s)),
            "kernel_name_mismatch": None if m is None else {
                "unmatched_positions": len(m.unmatched_positions),
                "families": m.families_unmatched, "extra_nodes": len(m.extra_nodes)}}


def check_passes(report: Mapping, bound: float) -> tuple[bool, list[str]]:
    """Gate A's family check at an explicit bound (a lock value)."""
    if not bound > 0:
        raise ValueError("the bound is a positive relative difference, fixed at lock")
    why = []
    km = report.get("kernel_name_mismatch")
    if km and (km["unmatched_positions"] or km["extra_nodes"]):
        why.append(f"kernel names do not match: {km}")
    if report["positions_without_replay"]:
        why.append(f"{len(report['positions_without_replay'])} positions have no replay")
    for f, r in report["families"].items():
        d = r["relative_difference"]
        if d is None or abs(d) > bound:
            why.append(f"{f}: relative difference {d}")
    return not why, why


def to_record(*, seq: OpSequence, prompt_id: str, m: Match, timing: Optional[dict],
              agreement: Optional[dict], tape: SyncTape,
              masks: Optional[mask_bank.MaskRecord],
              isolated: Optional[Mapping[int, float]] = None) -> dict:
    """The banked device components of one arm, prompt and cell."""
    cp = seq.code_path
    node_sum = sum(m.device_s.values())
    return {
        "schema": SCHEMA, "code_path": mask_bank._code_path_dict(cp),
        "context_length": seq.context_length, "batch": seq.batch, "prompt_id": prompt_id,
        "sequence_sha256": op_sequence.record_sha256(op_sequence.to_record(seq)),
        "masks_sha256": masks.sha256 if masks else None,
        "tape_entries": len(tape), "tape_substitutions": tape.substitutions,
        "complete": m.complete, "unmatched_positions": list(m.unmatched_positions),
        "families_unmatched": m.families_unmatched, "extra_nodes": list(m.extra_nodes),
        "device_s": {str(k): v for k, v in sorted(m.device_s.items())},
        "isolated_s": None if isolated is None else {str(k): v for k, v in sorted(isolated.items())},
        "node_sum_s": node_sum, "graph_timing": timing,
        "profiled_over_unprofiled": (node_sum / timing["median_s"]
                                     if timing and timing["median_s"] > 0 else None),
        "mask_agreement": agreement,
    }


def device_costs_from_record(record: Mapping) -> dict[int, float]:
    if record.get("schema") != SCHEMA:
        raise ReplayError(f"schema {record.get('schema')!r}")
    if not record["complete"]:
        raise ReplayError("this replay did not match its sequence; it has no device costs")
    return {int(k): float(v) for k, v in record["device_s"].items()}
