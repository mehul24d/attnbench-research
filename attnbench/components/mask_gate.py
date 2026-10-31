"""The mask gates of plan Section 7 (T2.23 on P2, T2.24 after P4).

"On P2, XAttention's Triton and PyTorch paths run on the same recorded
prompts through `use_triton`; per-layer mask agreement and density difference
against P1's recorded masks are within a bound fixed at lock. P4's Triton
masks are then compared with P2's." If the gate fails, the XAttention arms'
admissibility does not transfer to the new code path (the per-arm rule,
Section 5.1); if P4 and P2 differ, P6 records its own masks and the
full-transfer claim (H2) is restated.

**What is compared.** Two banks of `mask_bank.MaskRecord`s (T2.4), one per
code path. A record is one prefill of one arm on one prompt at one context
length and batch, and holds one mask per block-sparse kernel call, which is
one per layer. Records are paired by (arm, context length, batch, prompt);
a record on one side with no partner on the other is refused, never skipped,
since the gate is over the same recorded prompts. Two records of one code
path are refused: the gate compares across code paths.

**Per layer**, from the masks themselves:

  - `agreement`: the share of causally valid blocks, over every head, on
    which the two masks agree. Blocks above the causal diagonal are never
    kept by either side, so counting them would only push agreement up.
  - `identical`: the two masks equal bitwise (valid blocks and the rest).
  - each side's density, kept blocks over valid blocks times heads (Section
    5.5's convention, as `mask_bank.density` has it), and the difference,
    candidate minus reference.

The densities are recomputed from the masks, never read from a bank's
manifest, since nothing hashes the manifest's density.

**The bound** is a lock value (T4.1) with no default here: a minimum
agreement per layer and a maximum absolute density difference per layer,
both required. A comparison passes when every layer of every paired record
is within both. Which arms the gate covers is the per-arm rule's (the
XAttention arms, Section 5.1); the gate reports the arms its records belong
to and decides nothing else.

**For T2.28.** `density_deltas` turns a comparison into the cross-path
difference `predict.density_sensitivity` reads: an absolute change in mask
density by arm or by (arm, context length). A record's mask density is the
mean over its layers, so the shift for one prompt is the candidate's mean
minus the reference's; `statistic` says how the prompts are summarised
("mean": the absolute mean shift; "max": the largest absolute shift).

Researcher's decision (28 Oct 2026, to be dated in Appendix A by the
integration session): the pre-registered sensitivity uses "max", the worst
case, and the mean is reported alongside it (`density_shift_report`). The
argument stays required, with no default and no module constant, so the
choice is written at each call.

Everything here is numpy on the host; nothing imports torch, so the gates
run wherever the banks can be read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


class MaskGateError(ValueError):
    """Banks that cannot be compared as the gate requires."""


# --------------------------------------------------------------------------
# One layer
# --------------------------------------------------------------------------

def _valid(nq: int, nk: int, causal: bool) -> np.ndarray:
    v = np.ones((nq, nk), dtype=bool)
    return np.tril(v, k=nk - nq) if causal else v


def mask_density(mask: np.ndarray, causal: bool) -> float:
    """Kept blocks over valid blocks times leading dims (`mask_bank.density`)."""
    nq, nk = mask.shape[-2:]
    valid = _valid(nq, nk, causal)
    lead = mask.size // (nq * nk)
    return float((mask & valid).sum()) / float(valid.sum() * lead)


def valid_agreement(a: np.ndarray, b: np.ndarray, causal: bool) -> float:
    """The share of causally valid blocks, over every leading dim, on which
    `a` and `b` agree."""
    nq, nk = a.shape[-2:]
    valid = np.broadcast_to(_valid(nq, nk, causal), a.shape)
    return float((a == b)[valid].mean())


@dataclass(frozen=True)
class LayerComparison:
    layer: int
    agreement: float
    identical: bool
    reference_density: float
    candidate_density: float

    @property
    def density_difference(self) -> float:
        """Candidate minus reference."""
        return self.candidate_density - self.reference_density


# --------------------------------------------------------------------------
# One pair of records
# --------------------------------------------------------------------------

def code_path_name(cp) -> str:
    libs = ",".join(f"{k}={v}" for k, v in cp.libraries)
    return f"{cp.gpu}|{cp.implementation}|O{cp.python_optimize}|{libs}"


def record_key(rec) -> tuple:
    return (rec.code_path.arm, int(rec.context_length), int(rec.batch), str(rec.prompt_id))


@dataclass(frozen=True)
class RecordComparison:
    arm: str
    context_length: int
    batch: int
    prompt_id: str
    reference: str        # code path, `code_path_name`
    candidate: str
    layers: tuple[LayerComparison, ...]

    @property
    def mean_density_shift(self) -> float:
        """The candidate's mean density over layers minus the reference's."""
        n = len(self.layers)
        return (sum(l.candidate_density for l in self.layers)
                - sum(l.reference_density for l in self.layers)) / n


def _array(m) -> np.ndarray:
    a = np.asarray(m.array())
    if a.dtype != bool:
        raise MaskGateError(f"masks are bool; got {a.dtype}")
    if a.ndim < 2:
        raise MaskGateError(f"a mask has at least two dims, got {a.shape}")
    return a


def compare_records(reference, candidate) -> RecordComparison:
    """Layer by layer, two records of one arm on one prompt, from two code
    paths."""
    rk, ck = record_key(reference), record_key(candidate)
    if rk != ck:
        raise MaskGateError(
            f"the records are not the same arm, length, batch and prompt: {rk} "
            f"and {ck}")
    rname, cname = code_path_name(reference.code_path), code_path_name(candidate.code_path)
    if rname == cname:
        raise MaskGateError(f"both records are of the code path {rname}; the gate "
                            f"compares two code paths")
    if len(reference.masks) != len(candidate.masks):
        raise MaskGateError(
            f"{rk}: {len(reference.masks)} kernel calls against "
            f"{len(candidate.masks)}; the two paths ran different layers")
    if not reference.masks:
        raise MaskGateError(f"{rk}: no masks")
    layers = []
    for i, (rm, cm) in enumerate(zip(reference.masks, candidate.masks)):
        if bool(rm.causal) != bool(cm.causal):
            raise MaskGateError(f"{rk} layer {i}: one mask is causal and the other not")
        a, b = _array(rm), _array(cm)
        if a.shape != b.shape:
            raise MaskGateError(f"{rk} layer {i}: shapes {a.shape} and {b.shape}")
        causal = bool(rm.causal)
        layers.append(LayerComparison(
            layer=i, agreement=valid_agreement(a, b, causal),
            identical=bool(np.array_equal(a, b)),
            reference_density=mask_density(a, causal),
            candidate_density=mask_density(b, causal)))
    arm, length, batch, prompt = rk
    return RecordComparison(arm, length, batch, prompt, rname, cname, tuple(layers))


def compare_banks(reference: Iterable, candidate: Iterable) -> tuple[RecordComparison, ...]:
    """Every record of `reference` against its partner in `candidate`, by
    (arm, context length, batch, prompt). Both banks must hold the same
    keys, each once, and each bank one code path."""
    sides = {}
    for name, bank in (("reference", reference), ("candidate", candidate)):
        recs = list(bank)
        if not recs:
            raise MaskGateError(f"the {name} bank is empty")
        paths = {code_path_name(r.code_path) for r in recs}
        if len(paths) != 1:
            raise MaskGateError(f"the {name} bank mixes code paths {sorted(paths)}")
        by_key = {}
        for r in recs:
            k = record_key(r)
            if k in by_key:
                raise MaskGateError(f"the {name} bank holds {k} twice")
            by_key[k] = r
        sides[name] = by_key
    ref, cand = sides["reference"], sides["candidate"]
    if set(ref) != set(cand):
        raise MaskGateError(
            f"the banks are not on the same prompts: only in the reference "
            f"{sorted(set(ref) - set(cand))}, only in the candidate "
            f"{sorted(set(cand) - set(ref))}")
    return tuple(compare_records(ref[k], cand[k]) for k in sorted(ref))


# --------------------------------------------------------------------------
# The bound and the verdict
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Bounds:
    """The gate's bound, fixed at lock (T4.1). No defaults."""

    min_agreement: float             # per layer, over valid blocks, in (0, 1]
    max_density_difference: float    # per layer, absolute, >= 0

    def __post_init__(self) -> None:
        for name, v in (("min_agreement", self.min_agreement),
                        ("max_density_difference", self.max_density_difference)):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise MaskGateError(f"{name} must be a finite number, got {v!r}")
        if not 0 < self.min_agreement <= 1:
            raise MaskGateError(f"min_agreement must be in (0, 1], got {self.min_agreement}")
        if self.max_density_difference < 0:
            raise MaskGateError(
                f"max_density_difference must be >= 0, got {self.max_density_difference}")


@dataclass(frozen=True)
class GateResult:
    reference: str
    candidate: str
    passed: bool
    arms: tuple[str, ...]
    n_records: int
    n_layers: int
    min_agreement: float
    max_abs_density_difference: float
    identical_layers: int
    # (arm, length, batch, prompt, layer, reasons) for every layer outside the bound
    failures: tuple[tuple, ...]
    bounds: Bounds


def gate(comparisons: Sequence[RecordComparison], bounds: Bounds) -> GateResult:
    """Whether every layer of every comparison is within `bounds`."""
    comparisons = tuple(comparisons)
    if not comparisons:
        raise MaskGateError("nothing to gate")
    if not isinstance(bounds, Bounds):
        raise MaskGateError("the bound is a `Bounds`, fixed at lock")
    pairs = {(c.reference, c.candidate) for c in comparisons}
    if len(pairs) != 1:
        raise MaskGateError(f"one gate compares one pair of code paths, got {sorted(pairs)}")
    (ref, cand), = pairs
    failures = []
    for c in comparisons:
        for l in c.layers:
            why = []
            if l.agreement < bounds.min_agreement:
                why.append(f"agreement {l.agreement:.6f} < {bounds.min_agreement}")
            if abs(l.density_difference) > bounds.max_density_difference:
                why.append(f"|density difference| {abs(l.density_difference):.6f} > "
                           f"{bounds.max_density_difference}")
            if why:
                failures.append((c.arm, c.context_length, c.batch, c.prompt_id,
                                 l.layer, tuple(why)))
    layers = [l for c in comparisons for l in c.layers]
    return GateResult(
        reference=ref, candidate=cand, passed=not failures,
        arms=tuple(sorted({c.arm for c in comparisons})),
        n_records=len(comparisons), n_layers=len(layers),
        min_agreement=min(l.agreement for l in layers),
        max_abs_density_difference=max(abs(l.density_difference) for l in layers),
        identical_layers=sum(l.identical for l in layers),
        failures=tuple(failures), bounds=bounds)


STATISTICS = ("mean", "max")


def density_deltas(comparisons: Sequence[RecordComparison], *, statistic: str,
                   by_length: bool) -> dict:
    """The cross-path density difference for `predict.density_sensitivity`:
    {arm: d} or {(arm, context_length): d}, d >= 0 (module docstring)."""
    if statistic not in STATISTICS:
        raise MaskGateError(f"statistic must be one of {STATISTICS}, got {statistic!r}")
    groups: dict = {}
    for c in comparisons:
        k = (c.arm, c.context_length) if by_length else c.arm
        groups.setdefault(k, []).append(c.mean_density_shift)
    if not groups:
        raise MaskGateError("no comparisons")
    if statistic == "mean":
        return {k: abs(sum(v) / len(v)) for k, v in groups.items()}
    return {k: max(abs(x) for x in v) for k, v in groups.items()}


@dataclass(frozen=True)
class DensityShiftReport:
    """The deltas T2.28 reads (`deltas`, by `statistic`), with both
    summaries reported beside them."""

    statistic: str
    deltas: dict        # what predict.density_sensitivity reads
    max_shift: dict     # largest absolute shift, the pre-registered worst case
    mean_shift: dict    # absolute mean shift, reported alongside


def density_shift_report(comparisons: Sequence[RecordComparison], *, statistic: str,
                         by_length: bool) -> DensityShiftReport:
    """`density_deltas` by `statistic`, with the max and the mean shift
    beside it (module docstring)."""
    deltas = density_deltas(comparisons, statistic=statistic, by_length=by_length)
    return DensityShiftReport(
        statistic=statistic, deltas=deltas,
        max_shift=density_deltas(comparisons, statistic="max", by_length=by_length),
        mean_shift=density_deltas(comparisons, statistic="mean", by_length=by_length))


# --------------------------------------------------------------------------
# T2.23 and T2.24
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class P2MaskGate:
    """T2.23. Both P2 paths against P1's masks; the gate passes only if both
    do. Triton against PyTorch on P2 is reported beside them: with the card
    held fixed it isolates the implementation."""

    pytorch_vs_p1: GateResult
    triton_vs_p1: GateResult
    triton_vs_pytorch_on_p2: GateResult

    @property
    def passed(self) -> bool:
        return self.pytorch_vs_p1.passed and self.triton_vs_p1.passed


def _implementation(bank) -> str:
    impls = {r.code_path.implementation for r in bank}
    if len(impls) != 1:
        raise MaskGateError(f"a bank of one code path has one implementation, got {sorted(impls)}")
    return impls.pop()


def _gpu(bank) -> str:
    gpus = {r.code_path.gpu for r in bank}
    if len(gpus) != 1:
        raise MaskGateError(f"a bank of one code path has one GPU, got {sorted(gpus)}")
    return gpus.pop()


def p2_mask_gate(p1, p2_pytorch, p2_triton, bounds: Bounds) -> P2MaskGate:
    """T2.23 on the three banks: P1's (the L4 PyTorch path), and P2's two
    paths, run on P2's card through `use_triton` on the same prompts."""
    p1, p2_pytorch, p2_triton = list(p1), list(p2_pytorch), list(p2_triton)
    if _gpu(p2_pytorch) != _gpu(p2_triton):
        raise MaskGateError("P2's two paths must be recorded on the same card")
    if _implementation(p2_pytorch) == _implementation(p2_triton):
        raise MaskGateError("P2's two banks must be the two implementations")
    if _gpu(p1) == _gpu(p2_pytorch):
        raise MaskGateError("P1 and P2 are recorded on different cards")
    return P2MaskGate(
        pytorch_vs_p1=gate(compare_banks(p1, p2_pytorch), bounds),
        triton_vs_p1=gate(compare_banks(p1, p2_triton), bounds),
        triton_vs_pytorch_on_p2=gate(compare_banks(p2_pytorch, p2_triton), bounds))


P6_INHERITS = "P6 inherits P4's masks"
P6_RECORDS = ("P6 records its own masks, and the full-transfer claim (H2) is "
              "restated")


@dataclass(frozen=True)
class P4MaskComparison:
    """T2.24. P4's recorded Triton masks against P2's."""

    result: GateResult

    @property
    def decision(self) -> str:
        return P6_INHERITS if self.result.passed else P6_RECORDS


def p4_mask_comparison(p2_triton, p4_triton, bounds: Bounds) -> P4MaskComparison:
    p2_triton, p4_triton = list(p2_triton), list(p4_triton)
    if _implementation(p2_triton) != _implementation(p4_triton):
        raise MaskGateError("T2.24 compares one implementation (Triton) on two cards")
    if _gpu(p2_triton) == _gpu(p4_triton):
        raise MaskGateError("P2 and P4 are recorded on different cards")
    return P4MaskComparison(gate(compare_banks(p2_triton, p4_triton), bounds))
