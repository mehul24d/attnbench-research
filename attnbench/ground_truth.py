"""The ground-truth harness's timed region (plan Section 5.3 step 7, T2.13).

**The requirement** (researcher's decision, plan Appendix A, 26 Oct 2026).
For every deployable arm, the timed region includes its estimator, its mask
build, its copies and its synchronisations. The oracle arm may use cached
scores; its rows are labelled a non-deployable reference (scoring cost
outside the timed path). It is an attention-mass reference, not a quality
or speed ceiling (plan Appendix A, 4 Oct 2026).

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

**The dense control and the rerun rule** (plan Section 5.6). Every session
has exactly one dense arm, the control. Its forwards are clocked in the
warm-up block (the first is dropped as the cold call) and their median is its
warm-up time. In every round its time is compared with that: a difference
beyond the resolution floor, in either direction, flags the round. A round is
also flagged when the clock sampler (`clock_sampling`) flags any arm's block
as throttled. A flagged round is excluded from every statistic and rerun, up
to `Controls.max_rerun_rounds` extra rounds; if fewer than `reps` clean rounds
result, the result says `complete: False` and nothing hides it. The floor is
fixed at lock from the measured run-to-run spread (Appendix A, 25 Oct 2026),
so it has no default here.

**The first clocked block's timestamp** (Appendix A, 25 and 27 Oct 2026) is
the wall clock read immediately before the opening synchronise of the first
block of any kind that the harness clocks, warm-ups included, so it is no
later than that block's own clock start. Under the strict rule a prediction
synced while the dense control's warm-up was already being clocked is late.
The earlier rule (25 Oct 2026) stamped the first ground-truth block after the
warm-ups; it is not what this does. A rerun round never moves the stamp. The
result key is `first_clocked_block_started_at`.

**Traced against untraced** (Section 5.3 step 8). Ground truth is untraced.
`time_with_profiler_overhead` runs the whole untraced measurement first and
only then the traced one, in that order and in one process, because a
profiler that has been started once can leave its hooks and its flush work
behind, and an untraced block measured after it would no longer be untraced.
The tracer is entered outside each timed block, so its start and stop are not
charged: what the traced time includes is the forwards running while the
trace is being recorded. Traced rows are labelled and are never ground truth.

**One compiled variant per process** (Section 5.5, Compiled variants).
Dynamo caches compiled code per Python code object and does not guard on its
own settings, so a second setting compiled in the same process silently
reuses the first (measured on PyTorch 2.14.1, 25 Oct 2026; see
`backends/compiled.py` on the T2.21 branch). An arm names its
`compile_setting`, the harness checks that against the backend, and
`time_arms` refuses a second distinct setting in the call or in the process.
This is the harness's own check, so it holds whether or not the backend
enforces the same rule.

**Scope.** The timed region, arm interleaving, the dense control and the
rerun rule, clock and throttle sampling, the traced-against-untraced
overhead, the one-compiled-variant rule, the first clocked block's timestamp, and the
median, p90 and p99. Not here: running it on a GPU (the L4 dry run, T2.27);
a clock sampler's own cost on a host-bound arm, which that dry run measures.
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

from .clock_sampling import (RESEARCHER_FLAG_BITS, BlockClock, BlockSampler, ProcessSampler,
                             counters_only_misses, reason_names)
from .precomputed_masks import prompt_digest

DENSE = "dense"
INLINE = "inline"          # estimator called per layer inside run_measured
IN_BACKEND = "in_backend"  # estimator inside the backend's own forward (XAttention)
ORACLE = "oracle"          # cached scores from a separate dense pass
PRECOMPUTED = "precomputed"  # masks recorded offline and preloaded (T2.12)

KINDS = (DENSE, INLINE, IN_BACKEND, ORACLE, PRECOMPUTED)
DEPLOYABLE_KINDS = (INLINE, IN_BACKEND)
ORACLE_LABEL = "non-deployable reference (scoring cost outside the timed path)"
DEPLOYABLE_LABEL = "deployable"
PRECOMPUTED_LABEL = "ablation (masks precomputed offline; estimator cost outside every block)"

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


class CompileSettingError(TimedRegionError):
    """Two compiled settings in one process, or a declaration the backend
    does not bear out."""


@dataclass(frozen=True)
class Arm:
    name: str
    kind: str
    backend: object
    cfg: object
    inline_estimator: Optional[str] = None
    compile_setting: Optional[str] = None   # None: eager
    prompt_digest: Optional[str] = None     # the prompt a precomputed arm's masks belong to

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"arm kind must be one of {KINDS}, got {self.kind!r}")
        if (self.kind == INLINE) != (self.inline_estimator is not None):
            raise ValueError("an inline arm names its estimator, and only an inline arm does")
        if (self.kind == PRECOMPUTED) != (self.prompt_digest is not None):
            raise ValueError("a precomputed arm names the prompt its masks were computed on, "
                             "and only a precomputed arm does")

    @property
    def deployable(self) -> bool:
        return self.kind in DEPLOYABLE_KINDS

    @property
    def label(self) -> str:
        if self.kind == ORACLE:
            return ORACLE_LABEL
        if self.kind == PRECOMPUTED:
            return PRECOMPUTED_LABEL
        return DEPLOYABLE_LABEL if self.deployable else DENSE


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
                clock: Callable[[], float], out: list,
                sampler: "Optional[BlockSampler | ProcessSampler]" = None,
                clock_out: Optional[list] = None):
    """Synchronise, start the clock, run the body, synchronise, stop the
    clock. Appends the elapsed seconds to `out`. The closing synchronise is
    inside the block and is logged as such. With a `sampler`, its opening
    read comes after the opening synchronise and before the clock starts, and
    its closing read after the clock has stopped, so neither is timed; the
    block's `BlockClock` is appended to `clock_out`."""
    if sampler is not None and clock_out is None:
        raise ValueError("a sampler needs clock_out to hand its record to")
    synchronize()
    if sampler is not None:
        sampler.begin()
    log.open_rep = rep
    t0 = clock()
    try:
        yield
        synchronize()
        log.record("sync", "closing synchronise")
        out.append(clock() - t0)
    except BaseException:
        if sampler is not None:
            sampler.abort()
        raise
    else:
        if sampler is not None:
            clock_out.append(sampler.end())
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
    if arm.kind == PRECOMPUTED:
        ran = [e for e in log.events if e.kind in ("score", "mask_build")]
        if ran:
            raise TimedRegionError(
                f"{arm.name} is a precomputed-mask arm, but {len(ran)} scoring or mask "
                f"build call(s) ran ({sorted({e.what for e in ran})}); its masks are "
                f"computed offline and nothing here may compute one (plan Section 5.5)")
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


@dataclass(frozen=True)
class Controls:
    """What the session decided before timing anything. No defaults: each is
    a lock-time decision (plan Section 5.6) and is recorded on the result."""
    resolution_floor_ms: float   # dense drift beyond this flags a round
    max_rerun_rounds: int        # extra rounds allowed to replace flagged ones
    sampler: "BlockSampler | ProcessSampler"   # clocks and throttle reasons, every block

    def __post_init__(self) -> None:
        if not (math.isfinite(self.resolution_floor_ms) and self.resolution_floor_ms > 0):
            raise ValueError("resolution_floor_ms must be a positive number")
        if self.max_rerun_rounds < 0:
            raise ValueError("max_rerun_rounds must be >= 0")
        s = self.sampler
        # Audit C3, finding 1: a counters-only sampler reads the bits only at
        # the block's edges, where the GPU is idle, and no counter can show
        # hardware slowdown or tell the brake from the software cap.
        if isinstance(s, BlockSampler) and not s.sees_in_block:
            raise ValueError(
                f"a counters-only sampler cannot see "
                f"{reason_names(counters_only_misses(RESEARCHER_FLAG_BITS))} inside a "
                f"block: use the session's ProcessSampler (or an in-block interval)")
        missing = RESEARCHER_FLAG_BITS & ~getattr(s, "flag_bits", 0)
        if missing:
            raise ValueError(
                f"the sampler's flag set leaves out {reason_names(missing)}, which the "
                f"researcher's set flags (Appendix A, 27 Oct 2026)")


_PROCESS_COMPILE_SETTING: Optional[str] = None


def reset_process_compile_setting() -> None:
    """Forget the process's compiled setting. Only valid after Dynamo's caches
    were cleared (`torch._dynamo.reset()`); in tests otherwise."""
    global _PROCESS_COMPILE_SETTING
    _PROCESS_COMPILE_SETTING = None


def _claim_compile_setting(arms) -> None:
    for a in arms:
        actual = getattr(a.backend, "setting", None)
        if actual != a.compile_setting:
            raise CompileSettingError(
                f"{a.name} declares compile_setting={a.compile_setting!r} but its "
                f"backend carries setting={actual!r}")
    settings = sorted({a.compile_setting for a in arms if a.compile_setting is not None})
    if len(settings) > 1:
        raise CompileSettingError(
            f"arms carry the compiled settings {settings} in one process; Dynamo "
            f"would reuse the first setting's compiled code for the rest. Run "
            f"each setting in its own process")
    global _PROCESS_COMPILE_SETTING
    if settings:
        if _PROCESS_COMPILE_SETTING not in (None, settings[0]):
            raise CompileSettingError(
                f"this process already timed the {_PROCESS_COMPILE_SETTING!r} "
                f"compiled setting; {settings[0]!r} would reuse its compiled code")
        _PROCESS_COMPILE_SETTING = settings[0]


def _tracing(tracer):
    return contextlib.nullcontext() if tracer is None else tracer()


def _clocked(fn, synchronize, clock) -> float:
    synchronize()
    t0 = clock()
    fn()
    synchronize()
    return clock() - t0


def time_arms(wrapped, input_ids, arms, *, reps: int, controls: Controls,
              warmup: int = 3,
              synchronize: Callable[[], None] = lambda: None,
              clock: Callable[[], float] = time.perf_counter,
              wall_clock: Callable[[], float] = time.time,
              score_kwargs: Optional[dict] = None,
              tracer: Optional[Callable] = None) -> dict:
    """Time `arms` interleaved per round, the order rotating each round, until
    `reps` clean rounds exist or the rerun budget is spent. Every arm's timed
    region is checked, the dense control's drift and the clock sampler decide
    which rounds are flagged, and statistics are over unflagged rounds only.

    Returns {"first_clocked_block_started_at": unix seconds, "complete": bool,
    "clean_rounds": int, "dense_control": {...}, "rounds": [...],
    "rows": [one per arm]}. With a `tracer` (a context-manager factory) every
    forward, warm-up included, runs inside it and the result is labelled
    traced; use `time_with_profiler_overhead` for that.
    """
    if reps < 1:
        raise ValueError("reps must be >= 1")
    if warmup < 2:
        raise ValueError("warmup must be >= 2: the first clocked forward is the "
                         "cold call and is dropped from the dense control's "
                         "warm-up time")
    names = [a.name for a in arms]
    if len(set(names)) != len(names):
        raise ValueError(f"arm names must be unique: {names}")
    dense = [a for a in arms if a.kind == DENSE]
    if len(dense) != 1:
        raise TimedRegionError(
            f"a ground-truth run has exactly one dense control, got {len(dense)} "
            f"({[a.name for a in dense]}); without it a drifting machine looks "
            f"like a result (plan Section 5.6)")
    control = dense[0]
    _claim_compile_setting(arms)
    digest = prompt_digest(input_ids)
    for a in arms:
        if a.kind == PRECOMPUTED and a.prompt_digest != digest:
            raise TimedRegionError(
                f"{a.name}'s masks were computed on another prompt than the one being "
                f"timed; a mask is valid only for the prompt it was computed from")
    sampler, floor = controls.sampler, controls.resolution_floor_ms

    logs = {a.name: EventLog() for a in arms}
    kwargs = {}
    dense_warmup_ms = []
    first_started = None
    for a in arms:
        log = logs[a.name]
        with ScoringProbe(log):
            log.phase = SETUP
            kwargs[a.name] = untimed_setup(wrapped, input_ids, a, score_kwargs=score_kwargs)
            log.phase = WARMUP
            for _ in range(warmup):
                with _tracing(tracer):
                    if first_started is None:      # strict rule: the first clocked block of any kind
                        first_started = wall_clock()
                    took = _clocked(
                        lambda: wrapped.run_measured(input_ids, a.backend, cfg=a.cfg,
                                                     logits_to_keep=1, **kwargs[a.name]),
                        synchronize, clock)
                if a is control:
                    dense_warmup_ms.append(took * 1e3)
    reference_ms = statistics.median(dense_warmup_ms[1:])

    samples = {a.name: [] for a in arms}       # (round, ms), every round
    block_clock = {a.name: {} for a in arms}   # round -> BlockClock
    rounds = []
    clean = r = 0
    while clean < reps and r < reps + controls.max_rerun_rounds:
        k = r % len(arms)
        order = arms[k:] + arms[:k]
        round_ms, clocks = {}, {}
        for a in order:
            took, got = [], []
            with ScoringProbe(logs[a.name]), _tracing(tracer):
                with timed_block(logs[a.name], r, synchronize=synchronize,
                                 clock=clock, out=took, sampler=sampler,
                                 clock_out=got):
                    wrapped.run_measured(input_ids, a.backend, cfg=a.cfg,
                                         logits_to_keep=1, **kwargs[a.name])
            round_ms[a.name] = took[0] * 1e3
            clocks[a.name] = got[0]
            samples[a.name].append((r, round_ms[a.name]))
            block_clock[a.name][r] = got[0]
        drift = round_ms[control.name] - reference_ms
        reasons = []
        if abs(drift) > floor:
            reasons.append(f"dense_drift_{drift:+.3f}ms_vs_{floor}ms_floor")
        for a in arms:
            if clocks[a.name].flagged:
                reasons.append(f"{a.name}:throttled:" + "+".join(clocks[a.name].flag_reasons))
        rounds.append({"round": r, "order": [a.name for a in order],
                       "dense_ms": round_ms[control.name], "dense_drift_ms": drift,
                       "flagged": bool(reasons), "flag_reasons": reasons,
                       "blocks": {n: c.to_dict() for n, c in clocks.items()}})
        clean += not reasons
        r += 1

    for a in arms:
        check_timed_region(logs[a.name], a, list(range(r)))
    flagged_rounds = {x["round"] for x in rounds if x["flagged"]}
    rows = []
    for a in arms:
        ms = [v for rd, v in samples[a.name] if rd not in flagged_rounds]
        ev = [e for e in logs[a.name].events if e.inside]
        seen = 0
        for c in block_clock[a.name].values():
            seen |= c.throttle_bits_seen
        rows.append({
            "arm": a.name, "arm_kind": a.kind, "deployable": a.deployable,
            "arm_label": a.label, "compile_setting": a.compile_setting,
            "traced": tracer is not None,
            "reps": reps, "rounds_run": r, "clean_rounds": len(ms),
            "prefill_ms_samples": ms,
            "prefill_ms_median": statistics.median(ms) if ms else None,
            "prefill_ms_p90": quantile(ms, 0.90) if ms else None,
            "prefill_ms_p99": quantile(ms, 0.99) if ms else None,
            "flagged_ms_samples": [v for rd, v in samples[a.name] if rd in flagged_rounds],
            "throttle_reasons_seen": reason_names(seen),
            "sm_mhz_min": min(c.sm_mhz_min for c in block_clock[a.name].values()),
            "sm_mhz_max": max(c.sm_mhz_max for c in block_clock[a.name].values()),
            "score_calls_timed": sum(e.kind == "score" for e in ev),
            "mask_builds_timed": sum(e.kind == "mask_build" for e in ev),
        })
    return {
        "first_clocked_block_started_at": first_started,
        "complete": clean == reps, "clean_rounds": clean, "rounds_run": r,
        "traced": tracer is not None,
        "dense_control": {"arm": control.name, "warmup_ms": dense_warmup_ms,
                          "reference_ms": reference_ms,
                          "resolution_floor_ms": floor},
        "controls": {"max_rerun_rounds": controls.max_rerun_rounds,
                     "flag_bits": reason_names(sampler.flag_bits),
                     "sampler_interval_s": sampler.interval_s,
                     "sampler_kind": type(sampler).__name__},
        "rounds": rounds, "rows": rows,
    }


def profiler_overhead(untraced: dict, traced: dict, *, resolution_floor_ms: float) -> list:
    """Per arm, the traced median minus the untraced median, from two results
    of `time_arms`. Refuses a pair that is not one untraced and one traced
    result over the same arms. An overhead inside the resolution floor is
    reported as such and is not to be interpreted; an arm with no clean round
    on either side has none."""
    if untraced.get("traced") or not traced.get("traced"):
        raise TimedRegionError("profiler_overhead takes the untraced result first "
                               "and the traced result second")
    u = {r["arm"]: r for r in untraced["rows"]}
    t = {r["arm"]: r for r in traced["rows"]}
    if set(u) != set(t):
        raise TimedRegionError(f"untraced arms {sorted(u)} differ from traced arms {sorted(t)}")
    out = []
    for name in u:
        mu, mt = u[name]["prefill_ms_median"], t[name]["prefill_ms_median"]
        row = {"arm": name, "untraced_median_ms": mu, "traced_median_ms": mt,
               "untraced_clean_rounds": u[name]["clean_rounds"],
               "traced_clean_rounds": t[name]["clean_rounds"]}
        if mu is None or mt is None:
            row.update(overhead_ms=None, overhead_ratio=None, inside_resolution_floor=None)
        else:
            row.update(overhead_ms=mt - mu, overhead_ratio=mt / mu,
                       inside_resolution_floor=abs(mt - mu) <= resolution_floor_ms)
        out.append(row)
    return out


def time_with_profiler_overhead(wrapped, input_ids, arms, *, reps: int,
                                controls: Controls, tracer: Callable, **kw) -> dict:
    """The untraced measurement, then the traced one, then the per-arm
    overhead. The order is fixed here so no caller can trace first (module
    docstring). The timestamp to hand to the late-prediction check is the
    untraced result's."""
    untraced = time_arms(wrapped, input_ids, arms, reps=reps, controls=controls, **kw)
    traced = time_arms(wrapped, input_ids, arms, reps=reps, controls=controls,
                       tracer=tracer, **kw)
    return {"untraced": untraced, "traced": traced,
            "first_clocked_block_started_at": untraced["first_clocked_block_started_at"],
            "overhead": profiler_overhead(
                untraced, traced, resolution_floor_ms=controls.resolution_floor_ms)}
