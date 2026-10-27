"""The structural probe (T2.18; plan Section 5.1).

"A new code path is recorded on the target by a structural probe: each arm
runs once at short contexts (2048 and 4096 tokens) under the profiler, the
order of operations and synchronisations is kept and every duration
discarded, and the sequence is extended to the target length by the arm's
chunking rule, checked by agreement between the two short contexts."

**The chunking rule.** An arm's operation sequence can depend on the
context length in one way the probe can extend: a run of operations that
repeats once per query chunk. `ChunkingRule.chunks(L)` gives the number of
chunks at length L. Per layer, the sequences at the two short lengths must
then decompose as

    prefix + unit * chunks(L) + suffix

with one prefix, unit and suffix (compared by `op_sequence.op_key` without
payload bytes), and the extension at the target is prefix + unit *
chunks(target) + suffix. That decomposition existing at both lengths is the
agreement check. Where it does not exist, the arm's sequence is
**different** (plan Section 9: a disagreement means the arm is taken as
"different" for H0 and H2), and the probe says why.

**What two short contexts cannot check, and is refused:**

  - a target whose chunk count differs from the probe's when both short
    contexts have the same count: the unit was never seen to repeat;
  - XAttention at its official chunk size runs one chunk at every length
    up to 16384 and four at 32768 (`backends.xattention.official_chunk_size`),
    so its probe at 2048 and 4096 cannot be extended to 32768; see
    `xattention_rule`.

**Payloads.** A copy's or synchronisation's bytes can grow with length.
Each payload's growth between the two short lengths must be a power 0, 1 or
2 of the length ratio, and the extension scales it by that power. This is
inferred from two points, not checked by them: a third length would check
it. Every repetition of the unit must carry the same payloads at each short
length, since the extension repeats one unit: a payload that varies by chunk
(a key length that grows chunk by chunk) is not extended. Anything else is
"different".

`probe` records each length through `recorder.record_arm_prefill`, keeps the
sequences and deletes the traces (durations discarded), and `check_against_
full` is the L4 test of record: the extension against a sequence recorded at
the target length itself.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from . import op_sequence
from .op_sequence import (DIFFERENT, SAME, Comparison, OpSequence, SeqOp, _segments,
                          compare, op_key)

PROBE_LENGTHS = (2048, 4096)


@dataclass(frozen=True)
class ChunkingRule:
    name: str
    chunks: Callable[[int], int]


NO_CHUNKING = ChunkingRule("no chunking", lambda L: 1)


def xattention_rule() -> ChunkingRule:
    """ceil(L / chunk) query chunks at XAttention's official chunk size.

    Assumed from `Xattention_prefill`'s chunk_size default, not read from
    the estimator's loop: confirm on the L4 against a full trace at 32768."""
    from ..backends.xattention import official_chunk_size
    return ChunkingRule("xattention official chunk size",
                        lambda L: math.ceil(L / official_chunk_size(L)))


@dataclass(frozen=True)
class Extension:
    verdict: str                          # SAME (extended) or DIFFERENT
    reason: str
    sequence: Optional[OpSequence]        # at the target, when SAME
    chunks: dict                          # length -> chunks
    payload_exponents: tuple              # (layer, offset in segment, exponent), inferred

    @property
    def extended(self) -> bool:
        return self.verdict == SAME


def _keys(ops: Sequence[SeqOp]) -> list[tuple]:
    return [op_key(o, payload=False) for o in ops]


def _decompose(a: list, b: list, c1: int, c2: int) -> Optional[tuple[int, int, int]]:
    """(prefix, unit, suffix) lengths with a = P U^c1 S and b = P U^c2 S."""
    if c1 == c2:
        return (len(a), 0, 0) if a == b else None
    d = len(b) - len(a)
    if d <= 0 or d % (c2 - c1):
        return None
    u = d // (c2 - c1)
    for p in range(len(a) - c1 * u + 1):
        s = len(a) - p - c1 * u
        unit = b[p:p + u]
        if a[:p] != b[:p] or (s and a[len(a) - s:] != b[len(b) - s:]):
            continue
        if a[p:p + c1 * u] == unit * c1 and b[p:p + c2 * u] == unit * c2:
            return p, u, s
    return None


def _exponent(pa: int, pb: int, ratio: float) -> Optional[int]:
    if pa == pb:
        return 0
    if pa <= 0:
        return None
    for e in (1, 2):
        if pb == round(pa * ratio ** e):
            return e
    return None


def extend(seqs: Mapping[int, OpSequence], rule: ChunkingRule, target: int) -> Extension:
    """The probe's sequence extended to `target` (module docstring)."""
    if len(seqs) != 2:
        raise op_sequence.SequenceError("the probe extends from exactly two lengths")
    (l1, s1), (l2, s2) = sorted(seqs.items())
    if s1.code_path != s2.code_path or s1.batch != s2.batch:
        raise op_sequence.SequenceError("the two probe sequences are not one arm, code path and batch")
    c1, c2, ct = rule.chunks(l1), rule.chunks(l2), rule.chunks(target)
    chunks = {l1: c1, l2: c2, target: ct}

    def different(reason):
        return Extension(DIFFERENT, reason, None, chunks, ())

    if c1 == c2 and ct != c1:
        return different(
            f"{rule.name}: {c1} chunk(s) at both {l1} and {l2} but {ct} at {target}; "
            f"two short contexts with one chunk count cannot check how the unit repeats")
    if c2 < c1:
        return different(f"{rule.name}: fewer chunks at {l2} than at {l1}")
    if min(c1, c2, ct) < 1:
        raise op_sequence.SequenceError(f"{rule.name}: a chunk count below one ({chunks})")
    sa, sb = _segments(s1.ops), _segments(s2.ops)
    if [l for l, _ in sa] != [l for l, _ in sb]:
        return different("the two lengths differ in their layer structure")
    ratio_t, ratio = target / l2, l2 / l1
    out: list[SeqOp] = []
    exps = []
    for (layer, ia), (_, ib) in zip(sa, sb):
        a_ops = [s1.ops[i] for i in ia]
        b_ops = [s2.ops[i] for i in ib]
        dec = _decompose(_keys(a_ops), _keys(b_ops), c1, c2)
        if dec is None:
            return different(f"layer {layer}: no prefix + unit x chunks + suffix fits "
                             f"both lengths ({len(a_ops)} and {len(b_ops)} operations, "
                             f"{c1} and {c2} chunks)")
        p, u, s = dec
        for ops, c, L in ((a_ops, c1, l1), (b_ops, c2, l2)):
            for r in range(1, c):
                for j in range(u):
                    if ops[p + r * u + j].payload_bytes != ops[p + j].payload_bytes:
                        return different(
                            f"layer {layer}: at {L}, chunk {r} carries "
                            f"{ops[p + r * u + j].payload_bytes} bytes where chunk 0 "
                            f"carries {ops[p + j].payload_bytes}; per-chunk payloads "
                            f"that vary cannot be extended by repeating one unit")
        # Positions in b: prefix, the first unit, suffix. Their partners in a
        # (for payload growth) sit at the same offsets.
        def partner(k_b: int) -> int:
            if k_b < p:
                return k_b
            if k_b < p + c2 * u:
                return p + (k_b - p) % u if u else k_b
            return k_b - (c2 - c1) * u

        layer_ops = []
        target_idx = list(range(p)) + [p + j for _ in range(ct) for j in range(u)] + \
            list(range(len(b_ops) - s, len(b_ops)))
        for kb in target_idx:
            ob, oa = b_ops[kb], a_ops[partner(kb)]
            payload = ob.payload_bytes
            if oa.payload_bytes or ob.payload_bytes:
                e = _exponent(oa.payload_bytes, ob.payload_bytes, ratio)
                if e is None:
                    return different(
                        f"layer {layer}: a payload grows from {oa.payload_bytes} to "
                        f"{ob.payload_bytes} bytes between {l1} and {l2}, not as "
                        f"L^0, L^1 or L^2")
                if (layer, kb, e) not in exps:
                    exps.append((layer, kb, e))
                payload = int(round(ob.payload_bytes * ratio_t ** e))
            layer_ops.append(replace(ob, payload_bytes=payload, correlation=None))
        out.extend(layer_ops)
    seq = OpSequence(code_path=s2.code_path, context_length=int(target), batch=s2.batch,
                     ops=tuple(out), streams=tuple(sorted(set(s1.streams) | set(s2.streams))),
                     cross_stream_waits=max(s1.cross_stream_waits, s2.cross_stream_waits))
    return Extension(SAME, f"{rule.name}: extended from {l1} and {l2} ({c1}, {c2} chunks) "
                     f"to {target} ({ct})", seq, chunks, tuple(exps))


def check_against_full(ext: Extension, full: OpSequence) -> Comparison:
    """The L4 test of record: the extension against a sequence recorded at
    the target length (T2.18's "Done when")."""
    if not ext.extended:
        raise op_sequence.SequenceError(f"nothing to check: {ext.reason}")
    return compare(ext.sequence, full, payload=True)


def probe(record: Callable[[int, Path], OpSequence], out_dir, *,
          lengths: Sequence[int] = PROBE_LENGTHS, keep_traces: bool = False) -> dict[int, OpSequence]:
    """Record the arm at each probe length and keep only the sequences.

    `record(length, directory)` records one prefill there (for an arm,
    `recorder.record_arm_prefill` at that length) and returns its sequence.
    The trace is deleted afterwards unless `keep_traces`: the probe's
    durations are discarded (Section 5.1), and only the duration-free
    `sequence.json` and the banked masks remain.
    """
    out = Path(out_dir)
    seqs = {}
    for L in lengths:
        d = out / f"probe_{L}"
        seqs[int(L)] = record(int(L), d)
        trace = d / "trace.json"
        if trace.exists() and not keep_traces:
            trace.unlink()
            manifest = d / "recording.json"
            if manifest.exists():
                meta = json.loads(manifest.read_text())
                meta["files"].pop("trace.json", None)
                meta["trace_discarded"] = True
                manifest.write_text(json.dumps(meta, indent=1))
    return seqs
