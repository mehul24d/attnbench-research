"""The ground-truth harness's timed region (plan Section 5.3 step 7, T2.13).

**The requirement** (researcher's decision, plan Appendix A, 26 Oct 2026).
For every deployable arm, the timed region includes its estimator, its mask
build, its copies and its synchronisations. The oracle arm may use cached
scores; its rows are labelled a non-deployable upper bound.

**Why this is a check and not a convention.** Both scripts T2.13 adapts
compute importance scores before the clock starts: `run_vectorised_endtoend.py`
calls `compute_importance_scores` and then times `run_measured` with the
scores passed in, and `accuracy/phase_timing.py` says scoring is "EXCLUDED
from every latency number in this study by deliberate decision". That was
right for an oracle study. For a deployable arm it times a method nobody can
run. So the harness records where every scoring call and every mask build
happened, and `check_timed_region` refuses a measurement in which a deployable
arm's scores were computed outside the timed block.

**How it is observed.** `ScoringProbe` wraps the functions that compute
scores and build masks (`SCORE_TARGETS`, `MASK_TARGETS`) and logs each call
with whether a timed block was open at the time. The targets are the
functions the forward actually calls. A probe that saw nothing would make
every check pass, so a deployable arm must log at least one score event
inside every timed rep: a probe pointing at the wrong function fails loudly
instead of passing.

**Copies and synchronisations.** Everything a forward does runs inside the
block, including mask copies and the waits an estimator makes. The block
synchronises the device on entry and again before the clock stops, so queued
GPU work is charged to the arm that queued it. The closing synchronise is
logged and checked.

**Scope.** This is the timed region, arm interleaving, the timestamp of the
first timed block (Appendix A, 25 Oct 2026) and the median, p90 and p99. It
is not yet the dense control's drift check, traced-against-untraced overhead,
or GPU clock and throttle sampling. Those are still owed under T2.13.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

DENSE = "dense"
INLINE = "inline"          # estimator called per layer inside run_measured
IN_BACKEND = "in_backend"  # estimator inside the backend's own forward (XAttention)
ORACLE = "oracle"          # cached scores from a separate dense pass

KINDS = (DENSE, INLINE, IN_BACKEND, ORACLE)
DEPLOYABLE_KINDS = (INLINE, IN_BACKEND)
ORACLE_LABEL = "non-deployable upper bound"
DEPLOYABLE_LABEL = "deployable"

# (module, attribute path) of every function that computes importance scores.
# The forward looks each one up at call time, so patching the attribute is
# what the forward sees.
SCORE_TARGETS = (
    ("attnbench.accuracy.model", "minference_meanpool_scores_on_device"),
    ("attnbench.accuracy.model", "minference_meanpool_scores"),
    ("attnbench.accuracy.model", "_scoring_forward_chunked"),
    ("attnbench.accuracy.model", "SwappableAttentionModel.compute_importance_scores"),
    ("attnbench.backends.xattention", "XAttentionBackend.estimate"),
)
MASK_TARGETS = (
    ("attnbench.masks", "importance_block_mask_device"),
    ("attnbench.masks", "mask_for"),
)


class TimedRegionError(RuntimeError):
    """A measurement whose timed region does not contain what it must."""


@dataclass(frozen=True)
class Arm:
    name: str
    kind: str
    backend: object
    cfg: object
    inline_estimator: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"arm kind must be one of {KINDS}, got {self.kind!r}")
        if (self.kind == INLINE) != (self.inline_estimator is not None):
            raise ValueError("an inline arm names its estimator, and only an inline arm does")

    @property
    def deployable(self) -> bool:
        return self.kind in DEPLOYABLE_KINDS

    @property
    def label(self) -> str:
        return ORACLE_LABEL if self.kind == ORACLE else DEPLOYABLE_LABEL if self.deployable else DENSE


SETUP = "setup"      # before any forward: where an oracle's cached scores come from
WARMUP = "warmup"    # untimed forwards, not part of the measurement
TIMED = "timed"      # inside a timed block


@dataclass(frozen=True)
class Event:
    kind: str            # "score" | "mask_build" | "sync"
    what: str
    phase: str           # SETUP, WARMUP or TIMED
    rep: Optional[int]   # which timed block, or None outside one

    @property
    def inside(self) -> bool:
        return self.phase == TIMED


@dataclass
class EventLog:
    events: list = field(default_factory=list)
    phase: str = SETUP
    open_rep: Optional[int] = None

    def record(self, kind: str, what: str) -> None:
        phase = TIMED if self.open_rep is not None else self.phase
        self.events.append(Event(kind, what, phase, self.open_rep))


def _resolve(module: str, path: str):
    owner = importlib.import_module(module)
    *parents, attr = path.split(".")
    for p in parents:
        owner = getattr(owner, p)
    return owner, attr


class ScoringProbe:
    """Logs every call to a scoring or mask-building target while active."""

    def __init__(self, log: EventLog, *, score_targets=None, mask_targets=None):
        self.log = log
        score_targets = SCORE_TARGETS if score_targets is None else score_targets
        mask_targets = MASK_TARGETS if mask_targets is None else mask_targets
        self._targets = ([("score", t) for t in score_targets]
                         + [("mask_build", t) for t in mask_targets])
        self._saved = []

    def __enter__(self) -> "ScoringProbe":
        for kind, (module, path) in self._targets:
            try:
                owner, attr = _resolve(module, path)
            except ImportError:
                continue   # an optional backend that is not installed
            original = owner.__dict__[attr] if isinstance(owner, type) else getattr(owner, attr)
            self._saved.append((owner, attr, original))
            setattr(owner, attr, self._wrap(kind, f"{module}.{path}", original))
        return self

    def _wrap(self, kind: str, what: str, fn):
        log = self.log

        @functools.wraps(fn)
        def recorded(*args, **kwargs):
            log.record(kind, what)
            return fn(*args, **kwargs)
        return recorded

    def __exit__(self, *exc) -> None:
        for owner, attr, original in reversed(self._saved):
            setattr(owner, attr, original)
        self._saved.clear()


@contextlib.contextmanager
def timed_block(log: EventLog, rep: int, *, synchronize: Callable[[], None],
                clock: Callable[[], float], out: list):
    """Synchronise, start the clock, run the body, synchronise, stop the
    clock. Appends the elapsed seconds to `out`. The closing synchronise is
    inside the block and is logged as such."""
    synchronize()
    log.open_rep = rep
    t0 = clock()
    try:
        yield
        synchronize()
        log.record("sync", "closing synchronise")
        out.append(clock() - t0)
    finally:
        log.open_rep = None


def check_timed_region(log: EventLog, arm: Arm, reps: list) -> None:
    """Refuse a measurement that breaks the T2.13 requirement for `arm`."""
    for rep in reps:
        last = [e for e in log.events if e.rep == rep]
        if not last or last[-1].kind != "sync":
            raise TimedRegionError(
                f"{arm.name}: timed block {rep} does not end with its closing "
                f"synchronise, so queued GPU work is not charged to it")
    if arm.deployable:
        outside = [e for e in log.events
                   if e.kind in ("score", "mask_build") and e.phase == SETUP]
        if outside:
            raise TimedRegionError(
                f"{arm.name} is deployable, but {len(outside)} scoring or mask "
                f"build call(s) ran outside the timed block "
                f"({sorted({e.what for e in outside})}). Its estimator must "
                f"be timed (plan Section 5.3, T2.13)")
        for rep in reps:
            if not any(e.kind == "score" and e.rep == rep for e in log.events):
                raise TimedRegionError(
                    f"{arm.name} is deployable, but timed block {rep} computed "
                    f"no scores: they were computed earlier and reused, or the "
                    f"probe does not see the scorer the forward calls")
    if arm.kind == ORACLE:
        outside = [e for e in log.events if e.kind == "mask_build" and e.phase == SETUP]
        if outside:
            raise TimedRegionError(
                f"{arm.name}: the oracle may cache its scores, not its masks")


def untimed_setup(wrapped, input_ids, arm: Arm, *, score_kwargs: Optional[dict]) -> dict:
    """The keyword arguments `run_measured` gets for `arm`. Only the oracle's
    scores are computed here, before the clock starts."""
    if arm.kind == ORACLE:
        if score_kwargs is None:
            raise ValueError("the oracle arm needs score_kwargs (task, example_id, cache_dir)")
        return {"layer_scores": wrapped.compute_importance_scores(input_ids, **score_kwargs)}
    if arm.kind == INLINE:
        return {"inline_estimator": arm.inline_estimator}
    return {}


def quantile(samples: list, q: float) -> float:
    """Linear-interpolation quantile (numpy's default), q in [0, 1]."""
    xs = sorted(samples)
    pos = (len(xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def time_arms(wrapped, input_ids, arms, *, reps: int, warmup: int = 1,
              synchronize: Callable[[], None] = lambda: None,
              clock: Callable[[], float] = time.perf_counter,
              wall_clock: Callable[[], float] = time.time,
              score_kwargs: Optional[dict] = None) -> dict:
    """Time `arms` interleaved per rep, the order rotating each rep. Returns
    {"first_timed_block_started_at": unix seconds, "rows": [...]}, one row
    per arm, after checking every arm's timed region."""
    if reps < 1:
        raise ValueError("reps must be >= 1")
    names = [a.name for a in arms]
    if len(set(names)) != len(names):
        raise ValueError(f"arm names must be unique: {names}")
    logs = {a.name: EventLog() for a in arms}
    samples = {a.name: [] for a in arms}
    kwargs = {}
    first_started = None
    for a in arms:
        log = logs[a.name]
        with ScoringProbe(log):
            log.phase = SETUP
            kwargs[a.name] = untimed_setup(wrapped, input_ids, a, score_kwargs=score_kwargs)
            log.phase = WARMUP
            for _ in range(warmup):
                wrapped.run_measured(input_ids, a.backend, cfg=a.cfg, logits_to_keep=1,
                                     **kwargs[a.name])
    for r in range(reps):
        k = r % len(arms)
        for a in arms[k:] + arms[:k]:
            if first_started is None:
                first_started = wall_clock()
            with ScoringProbe(logs[a.name]):
                with timed_block(logs[a.name], r, synchronize=synchronize,
                                 clock=clock, out=samples[a.name]):
                    wrapped.run_measured(input_ids, a.backend, cfg=a.cfg,
                                         logits_to_keep=1, **kwargs[a.name])
    rows = []
    for a in arms:
        check_timed_region(logs[a.name], a, list(range(reps)))
        ms = [s * 1e3 for s in samples[a.name]]
        ev = [e for e in logs[a.name].events if e.inside]
        rows.append({
            "arm": a.name, "arm_kind": a.kind, "deployable": a.deployable,
            "arm_label": a.label, "reps": reps,
            "prefill_ms_samples": ms,
            "prefill_ms_median": statistics.median(ms),
            "prefill_ms_p90": quantile(ms, 0.90),
            "prefill_ms_p99": quantile(ms, 0.99),
            "score_calls_timed": sum(e.kind == "score" for e in ev),
            "mask_builds_timed": sum(e.kind == "mask_build" for e in ev),
        })
    return {"first_timed_block_started_at": first_started, "rows": rows}
