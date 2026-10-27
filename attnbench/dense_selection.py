"""Dense-baseline selection (control K2; plan Section 5.5; T2.16).

"Candidates: SDPA flash, SDPA cuDNN, SDPA memory-efficient, FA2, and FA3 on
the H100. A candidate is *correct* if it matches the float64 naive reference
of `attnbench/gates.py` on fixed inputs within a tolerance fixed at lock, or,
where that reference cannot be allocated, passes the gate's cross-backend
comparison (weaker evidence), and runs N repeats without a device fault.
Which of the two each cell used is named per cell. The *fastest* correct
candidate has the lowest median untraced time per card, context length and
batch, measured in a warm-up block separate from ground truth. The choice is
recorded before the timed runs, and every speedup is stated against it."

This module does that, for one cell (card, context length, batch) at a time,
and records the result in a file that can be checked against its own inputs.

**What a candidate's status can be, and which ones are results.**

- `correct`: passed the correctness gate and ran every warm-up round.
- `incorrect`: failed it. Fast and wrong is the case this protocol exists
  for; such a candidate is never timed into the choice, and the record keeps
  its error.
- `declined`: the backend's own capability declined the cell. A *known
  device fault* (cuDNN above 8192 tokens on the L4 and the A100) is declined
  here and never launched, because an illegal access corrupts the CUDA
  context and cannot be caught; the record names it as a fault.
- `failed_repeats`: raised during the warm-up rounds.
- `unavailable`: planned for this card and not supplied, because it is not
  installed or no backend of that name exists (FA3 has none in this
  repository yet). **This is a gap, not a result.** A record with any
  unavailable candidate says `complete: False`, and `check_control` refuses
  it: the fastest of the candidates that could run is not "the fastest
  correct dense kernel".

**The correctness check is `gates.check_for_family`.** It compares with the
float64 reference where that fits in memory and falls back to agreement
among independent implementations where it does not, and it says which in
`check_kind`. The other candidates are offered as the references, so on a
long cell each is checked against its peers. The tolerance is the gate's
(`gates.TOL`), recorded on the row. The plan says it is fixed at lock, and it
is not fixed yet (the table's own comment says it is loosely calibrated), so
a record carries the tolerance it used rather than a claim about it.

**A tie is recorded, not broken by a rule nobody chose.** The choice is the
strictly lowest median, the earlier candidate in the planned order if two are
equal, and `within_floor_of_chosen` lists every candidate whose median is
within the resolution floor of it: for those the choice is not resolved by
this measurement.

**The record follows from its own contents.** `load_choice` recomputes every
median from the raw warm-up samples and the choice, the completeness and the
within-floor list from the medians. It refuses a file whose bytes do not match
its sidecar, and one whose derived fields disagree with its samples. A record
re-hashed after an edit still fails the second check.

**Binding the ground truth to it.** `check_control` is what the session runner
calls with the dense arm's backend name and the first timed block's timestamp
(`ground_truth.time_arms`): the dense control must be the recorded choice, the
record must be complete, and it must have been recorded before the first timed
block started.

**Restating banked figures.** `measured_restatement` gives a speedup from rows
of one session in which the dense arm is the chosen kernel. The banked A100
end-to-end restatements are arithmetic on per-layer kernel times, not
measurements (plan Section 2, 18 Oct 2026); `arithmetic_restatement` computes
that kind and labels it so, and the two kinds are not interchangeable.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from . import gates
from .backends.base import FAULT_REASON_PREFIX
from .config import AttnConfig

SCHEMA = "attnbench.dense_choice/1"

SDPA_FLASH, SDPA_CUDNN, SDPA_EFFICIENT = "sdpa_flash", "sdpa_cudnn", "sdpa_efficient"
FA2, FA3 = "fa2", "fa3"
BASE_CANDIDATES = (SDPA_FLASH, SDPA_CUDNN, SDPA_EFFICIENT, FA2)

CORRECT, INCORRECT, DECLINED = "correct", "incorrect", "declined"
FAILED_REPEATS, UNAVAILABLE = "failed_repeats", "unavailable"
STATUSES = (CORRECT, INCORRECT, DECLINED, FAILED_REPEATS, UNAVAILABLE)


class SelectionError(RuntimeError):
    """A selection that cannot be made as the protocol requires."""


class SelectionRecordError(ValueError):
    """A recorded choice that does not follow from its own contents, or whose
    bytes are not the ones that were hashed."""


def planned_candidates(card: str) -> tuple:
    """The protocol's candidates for a card: FA3 only on the H100."""
    return BASE_CANDIDATES + ((FA3,) if "H100" in card.upper() else ())


def registry_candidates(card: str) -> dict:
    """{planned name: backend or None} from the repository's own backends.
    None where it is not installed or no backend of that name exists."""
    from .backends import base, impls
    out = {}
    for name in planned_candidates(card):
        backend = None
        if name.startswith("sdpa_"):
            backend = impls.SDPABackend(kernel=name.split("_", 1)[1])
        elif name in base._REGISTRY and base.get(name).is_available():
            backend = base.get(name)()
        out[name] = backend
    return out


@dataclass(frozen=True)
class Cell:
    card: str
    context_length: int
    batch: int

    @property
    def key(self) -> str:
        return f"{self.card}_{self.context_length}_{self.batch}"


@dataclass
class CandidateResult:
    name: str
    status: str
    reason: str = ""
    known_device_fault: bool = False
    check_kind: Optional[str] = None          # "exact", "cross_backend", ...
    max_abs_err: Optional[float] = None
    tolerance: Optional[dict] = None
    correctness_detail: str = ""
    warmup_ms: list = field(default_factory=list)     # raw, in round order
    median_ms: Optional[float] = None


@dataclass
class Choice:
    cell: Cell
    cfg_key: str
    planned: list
    candidates: list              # CandidateResult, in planned order
    rounds: int
    resolution_floor_ms: float
    recorded_at: float            # unix seconds, wall clock, after the last round
    chosen: Optional[str] = None
    chosen_median_ms: Optional[float] = None
    runner_up: Optional[str] = None
    margin_ms: Optional[float] = None
    within_floor_of_chosen: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    complete: bool = False
    schema: str = SCHEMA

    def to_dict(self) -> dict:
        return asdict(self)


def derive(candidates: Sequence[CandidateResult], floor_ms: float) -> dict:
    """Everything in a `Choice` that follows from the candidates' medians."""
    ok = [c for c in candidates if c.status == CORRECT]
    ranked = sorted(ok, key=lambda c: c.median_ms)       # stable: planned order breaks ties
    out = {"missing": [c.name for c in candidates if c.status == UNAVAILABLE],
           "chosen": None, "chosen_median_ms": None, "runner_up": None,
           "margin_ms": None, "within_floor_of_chosen": []}
    out["complete"] = not out["missing"]
    if ranked:
        best = ranked[0]
        out.update(chosen=best.name, chosen_median_ms=best.median_ms,
                   within_floor_of_chosen=[c.name for c in ranked[1:]
                                           if c.median_ms - best.median_ms <= floor_ms])
        if len(ranked) > 1:
            out.update(runner_up=ranked[1].name,
                       margin_ms=ranked[1].median_ms - best.median_ms)
    return out


def _default_sync(device: str) -> Callable[[], None]:
    if device.startswith("cuda"):
        import torch
        return torch.cuda.synchronize
    return lambda: None


def select_dense_baseline(cell: Cell, cfg: AttnConfig, candidates: dict, *,
                          rounds: int, resolution_floor_ms: float,
                          device: str = "cuda",
                          synchronize: Optional[Callable[[], None]] = None,
                          clock: Callable[[], float] = time.perf_counter,
                          wall_clock: Callable[[], float] = time.time,
                          check: Callable = gates.check_for_family) -> Choice:
    """Run the protocol for one cell. `candidates` maps each planned name to a
    backend, or to None where it is unavailable; a name that is not planned
    for the card is refused, and a planned name left out is unavailable."""
    if rounds < 2:
        raise ValueError("rounds must be >= 2: a median of one is not a median")
    if not (resolution_floor_ms > 0):
        raise ValueError("resolution_floor_ms must be positive")
    if cfg.seq_len != cell.context_length or cfg.batch != cell.batch:
        raise SelectionError(f"cfg ({cfg.seq_len}, batch {cfg.batch}) is not the cell "
                             f"{cell.key}")
    planned = list(planned_candidates(cell.card))
    extra = sorted(set(candidates) - set(planned))
    if extra:
        raise SelectionError(f"{extra} are not planned candidates for {cell.card} "
                             f"({planned}); the candidate set is the protocol's")
    sync = synchronize or _default_sync(device)

    results = {n: CandidateResult(n, UNAVAILABLE,
                                  "not installed, or no backend of this name exists"
                                  if candidates.get(n) is None else "")
               for n in planned}
    live = {}
    for name in planned:
        backend = candidates.get(name)
        if backend is None:
            continue
        ok, why = backend.claims_support(cfg)
        if not ok:
            results[name] = CandidateResult(
                name, DECLINED, why, known_device_fault=why.startswith(FAULT_REASON_PREFIX))
        else:
            live[name] = backend

    # Correctness, with the other supported candidates as references.
    # The references are every candidate the capability check let through, the
    # same set for each, so that the verdicts do not depend on the order they
    # are reached in.
    tol_key = cfg.quant_scheme if cfg.quant_scheme is not None else cfg.dtype
    supported = dict(live)
    for name, backend in supported.items():
        refs = [b for n, b in supported.items() if n != name]
        verdict = check(backend, cfg, device=device, references=refs)
        results[name] = CandidateResult(
            name, CORRECT if verdict.passed else INCORRECT,
            reason="" if verdict.passed else verdict.detail,
            check_kind=verdict.check_kind, max_abs_err=verdict.max_abs_err,
            tolerance=dict(gates.TOL[tol_key]), correctness_detail=verdict.detail)
        if not verdict.passed:
            del live[name]

    # The warm-up block: interleaved rounds, the order rotating each round, on
    # one set of inputs every candidate sees.
    if live:
        order = list(live)
        q, k, v = live[order[0]].make_inputs(cfg, device=device)
        for name in order:                       # one untimed call each: allocator, JIT
            try:
                live[name].timed_call(q, k, v, cfg)
                sync()
            except Exception as e:
                results[name] = _failed(results[name], e)
                live.pop(name)
        for r in range(rounds):
            names = list(live)
            if not names:
                break
            for name in names[r % len(names):] + names[:r % len(names)]:
                try:
                    sync()
                    t0 = clock()
                    live[name].timed_call(q, k, v, cfg)
                    sync()
                    results[name].warmup_ms.append((clock() - t0) * 1e3)
                except Exception as e:
                    results[name] = _failed(results[name], e)
                    live.pop(name)
        for name in live:
            results[name].median_ms = statistics.median(results[name].warmup_ms)

    cands = [results[n] for n in planned]
    return Choice(cell=cell, cfg_key=cfg.key(), planned=planned, candidates=cands,
                  rounds=rounds, resolution_floor_ms=resolution_floor_ms,
                  recorded_at=wall_clock(), **derive(cands, resolution_floor_ms))


def _failed(prev: CandidateResult, exc: BaseException) -> CandidateResult:
    prev.status = FAILED_REPEATS
    prev.reason = f"{type(exc).__name__}: {exc}"[:300]
    prev.median_ms = None
    return prev


# --- the record -----------------------------------------------------------------

def _canonical(d: dict) -> bytes:
    return (json.dumps(d, sort_keys=True, indent=1, allow_nan=False) + "\n").encode()


def _filename(choice: Choice) -> str:
    return f"dense_choice_{choice.cell.key}.json"


def write_choice(choice: Choice, directory) -> Path:
    """Write the record and a `sha256sum`-format sidecar. Refuses to overwrite:
    a choice is recorded once, before the timed runs."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _filename(choice)
    side = path.with_name(path.name + ".sha256")
    if path.exists() or side.exists():
        raise SelectionError(f"{path} already exists; a recorded choice is not overwritten")
    data = _canonical(choice.to_dict())
    path.write_bytes(data)
    side.write_text(f"{hashlib.sha256(data).hexdigest()}  {path.name}\n")
    return path


def load_choice(path) -> Choice:
    """Read a record, refusing one whose bytes are not the hashed ones or whose
    derived fields do not follow from its samples."""
    path = Path(path)
    data = path.read_bytes()
    side = path.with_name(path.name + ".sha256")
    if not side.exists():
        raise SelectionRecordError(f"{path.name} has no sha256 sidecar")
    stated = side.read_text().split()
    if len(stated) != 2 or stated[1] != path.name:
        raise SelectionRecordError(f"{side.name} is not in sha256sum format for {path.name}")
    if hashlib.sha256(data).hexdigest() != stated[0]:
        raise SelectionRecordError(f"{path.name} does not match its sidecar's sha256")
    d = json.loads(data)
    if d.get("schema") != SCHEMA:
        raise SelectionRecordError(f"schema {d.get('schema')!r}, expected {SCHEMA!r}")
    cands = [CandidateResult(**c) for c in d["candidates"]]
    choice = Choice(**{**d, "cell": Cell(**d["cell"]), "candidates": cands})
    _check_follows(choice)
    return choice


def _check_follows(choice: Choice) -> None:
    names = [c.name for c in choice.candidates]
    if names != list(choice.planned) or names != list(planned_candidates(choice.cell.card)):
        raise SelectionRecordError(
            f"candidates {names} are not the planned set for {choice.cell.card}, in order")
    for c in choice.candidates:
        if c.status not in STATUSES:
            raise SelectionRecordError(f"{c.name}: unknown status {c.status!r}")
        if c.status == CORRECT:
            if len(c.warmup_ms) != choice.rounds:
                raise SelectionRecordError(
                    f"{c.name} has {len(c.warmup_ms)} warm-up samples for {choice.rounds} rounds")
            if c.median_ms != statistics.median(c.warmup_ms):
                raise SelectionRecordError(
                    f"{c.name}: median {c.median_ms} is not the median of its samples")
            if c.check_kind is None:
                raise SelectionRecordError(
                    f"{c.name} is correct but names no check kind")
        elif c.median_ms is not None:
            raise SelectionRecordError(f"{c.name} is {c.status} but carries a median")
    want = derive(choice.candidates, choice.resolution_floor_ms)
    got = {k: getattr(choice, k) for k in want}
    if got != want:
        diff = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
        raise SelectionRecordError(f"derived fields do not follow from the samples: {diff}")


def check_control(choice: Choice, control_backend_name: str,
                  first_clocked_block_started_at: float) -> None:
    """What the session runner asks before ground truth is accepted: the dense
    control is the recorded choice, the record is complete, and it was made
    before the first clocked block of any kind started (the strict rule,
    Appendix A, 27 Oct 2026: warm-ups count)."""
    if not choice.complete:
        raise SelectionError(
            f"the choice for {choice.cell.key} is incomplete (unavailable: "
            f"{choice.missing}); the fastest of the candidates that could run is "
            f"not the fastest correct dense kernel")
    if choice.chosen is None:
        raise SelectionError(f"no candidate was correct for {choice.cell.key}")
    if control_backend_name != choice.chosen:
        raise SelectionError(
            f"the dense control is {control_backend_name!r} but the recorded choice "
            f"for {choice.cell.key} is {choice.chosen!r}")
    if not choice.recorded_at < first_clocked_block_started_at:
        raise SelectionError(
            f"the choice was recorded at {choice.recorded_at}, not before the first "
            f"clocked block at {first_clocked_block_started_at}")


# --- restating banked figures ---------------------------------------------------

MEASURED, ARITHMETIC = "measured_same_session", "arithmetic"


@dataclass(frozen=True)
class Restated:
    kind: str
    ratio: float
    basis: dict


def measured_restatement(choice: Choice, *, session: str, dense_arm: str,
                         dense_ms: float, dense_session: str,
                         sparse_ms: float, sparse_session: str) -> Restated:
    """Dense time over sparse time, both measured in one session with the dense
    arm being the chosen kernel. Anything else is not this kind of number."""
    if choice.chosen is None or not choice.complete:
        raise SelectionError("no complete recorded choice to state a speedup against")
    if dense_arm != choice.chosen:
        raise SelectionError(f"the dense arm is {dense_arm!r}, not the chosen "
                             f"{choice.chosen!r}")
    if not (dense_session == sparse_session == session):
        raise SelectionError(
            f"dense ({dense_session!r}) and sparse ({sparse_session!r}) are not both from "
            f"session {session!r}: that is not a same-session measurement")
    return Restated(MEASURED, dense_ms / sparse_ms,
                    {"dense_arm": dense_arm, "dense_ms": dense_ms,
                     "sparse_ms": sparse_ms, "session": session})


def arithmetic_restatement(*, dense_e2e_ms: float, sparse_e2e_ms: float, layers: int,
                           baseline_kernel_ms: float, chosen_kernel_ms: float) -> Restated:
    """The banked method: the dense end-to-end time less what the faster kernel
    would save over `layers` layers, over the sparse end-to-end time. Not a
    measurement; labelled so."""
    adjusted = dense_e2e_ms - layers * (baseline_kernel_ms - chosen_kernel_ms)
    return Restated(ARITHMETIC, adjusted / sparse_e2e_ms,
                    {"adjusted_dense_e2e_ms": adjusted, "layers": layers,
                     "baseline_kernel_ms": baseline_kernel_ms,
                     "chosen_kernel_ms": chosen_kernel_ms})
