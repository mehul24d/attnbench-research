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

**Chunked arms: three probe lengths** (plan T2.18, Appendix A, 27 Oct
2026). An arm whose rule chunks is probed at three lengths, each with at
least two chunks and no two with the same count (`chunked_probe_lengths`).
`extend_checked` extends from the two shorter to the longest, which must be
the same sequence as the one recorded there (payloads included, so the
growth law inferred from two points is checked by the third), and only then
extends to the target. XAttention at its official chunk size first reaches
two chunks above 16384 tokens, so its probe lengths are 20480, 28672 and
34816 (3, 4 and 9 chunks), not short contexts. The lengths sit mid-band
because a real prompt lands at or below its budget: the probe keys each
sequence by its recorded context length and checks the counts there.

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
CHUNKED_PROBES = 3                 # probe lengths for a chunked arm
MIN_PROBE_CHUNKS = 2               # chunks at each of them
LENGTH_STEP = 128                  # probe lengths are whole kernel blocks
MAX_PROBE_LENGTH = 131072


@dataclass(frozen=True)
class ChunkingRule:
    name: str
    chunks: Callable[[int], int]
    chunked: bool = True


NO_CHUNKING = ChunkingRule("no chunking", lambda L: 1, chunked=False)


def xattention_rule() -> ChunkingRule:
    """ceil(L / chunk) query chunks at XAttention's official chunk size.

    Assumed from `Xattention_prefill`'s chunk_size default, not read from
    the estimator's loop: confirm on the L4 against a full trace at 32768."""
    from ..backends.xattention import official_chunk_size
    return ChunkingRule("xattention official chunk size",
                        lambda L: math.ceil(L / official_chunk_size(L)))


def chunked_probe_lengths(rule: ChunkingRule, *, start: int = PROBE_LENGTHS[0],
                          n: int = CHUNKED_PROBES, min_chunks: int = MIN_PROBE_CHUNKS,
                          step: int = LENGTH_STEP, limit: int = MAX_PROBE_LENGTH
                          ) -> tuple[int, ...]:
    """The probe lengths of a chunked arm: one per chunk count, for the first
    `n` counts of at least `min_chunks` from `start`, each the multiple of
    `step` nearest below the middle of the lengths with that count, so a
    prompt that lands below its budget keeps the count."""
    if not rule.chunked:
        raise op_sequence.SequenceError(f"{rule.name}: not a chunked rule; probe at {PROBE_LENGTHS}")
    bands: dict[int, list[int]] = {}
    L = -(-start // step) * step
    while L <= limit and len(bands) <= n:
        c = rule.chunks(L)
        if c >= min_chunks:
            bands.setdefault(c, []).append(L)
        L += step
    counts = sorted(bands)[:n]
    if len(counts) < n:
        raise op_sequence.SequenceError(
            f"{rule.name}: fewer than {n} chunk counts of at least {min_chunks} up to {limit}")
    out = []
    for c in counts:
        lo, hi = bands[c][0], bands[c][-1]
        out.append(max(lo, (lo + hi) // 2 // step * step))
    return tuple(out)


@dataclass(frozen=True)
class Extension:
    verdict: str                          # SAME (extended) or DIFFERENT
    reason: str
    sequence: Optional[OpSequence]        # at the target, when SAME
    chunks: dict                          # length -> chunks
    payload_exponents: tuple              # (layer, offset in segment, exponent), inferred
    checked_at: Optional[int] = None      # the third probe length, for a chunked arm

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


def extend_checked(seqs: Mapping[int, OpSequence], rule: ChunkingRule,
                   target: int) -> Extension:
    """A chunked arm's extension from three probe lengths (module docstring)."""
    if len(seqs) != CHUNKED_PROBES:
        raise op_sequence.SequenceError(
            f"a chunked arm is probed at {CHUNKED_PROBES} lengths, not {len(seqs)}")
    (l1, s1), (l2, s2), (l3, s3) = sorted(seqs.items())
    counts = {L: rule.chunks(L) for L in (l1, l2, l3)}
    counts[target] = rule.chunks(target)
    if min(counts[L] for L in (l1, l2, l3)) < MIN_PROBE_CHUNKS or \
            len({counts[L] for L in (l1, l2, l3)}) != CHUNKED_PROBES:
        return Extension(DIFFERENT, f"{rule.name}: the probe lengths {l1}, {l2}, {l3} have "
                         f"{counts[l1]}, {counts[l2]}, {counts[l3]} chunks; each needs at "
                         f"least {MIN_PROBE_CHUNKS}, and no two the same", None, counts, ())
    pair = {l1: s1, l2: s2}
    third = extend(pair, rule, l3)
    if not third.extended:
        return replace(third, chunks=counts)
    c = compare(third.sequence, s3, payload=True)
    if c.verdict != SAME:
        return Extension(DIFFERENT, f"{rule.name}: extended from {l1} and {l2}, the sequence "
                         f"at {l3} differs from the one recorded there ({c.reason})",
                         None, counts, third.payload_exponents)
    ext = extend(pair, rule, target)
    return replace(ext, chunks=counts, checked_at=l3,
                   reason=f"{ext.reason}; checked at {l3} ({counts[l3]})" if ext.extended
                   else ext.reason)


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
    `recorder.record_arm_prefill` at that length) and returns its sequence;
    the result is keyed by each sequence's `context_length`.
    The trace is deleted afterwards unless `keep_traces`: the probe's
    durations are discarded (Section 5.1), and only the duration-free
    `sequence.json` and the banked masks remain.
    """
    out = Path(out_dir)
    seqs = {}
    for L in lengths:
        d = out / f"probe_{L}"
        seq = record(int(L), d)
        # Keyed by the recorded length: a real prompt lands at or below its
        # budget, and the chunking rule is read where the arm ran.
        if seq.context_length in seqs:
            raise op_sequence.SequenceError(
                f"two probe lengths recorded at {seq.context_length} tokens")
        seqs[int(seq.context_length)] = seq
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
