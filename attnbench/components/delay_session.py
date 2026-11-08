"""Gate A's leg 3 in a P1 session: T2.25 on the L4, the session's last step
(adviser's ruling 1 of 8 Nov 2026, fifth paste; Appendix A, 8 Nov 2026).

The set is declared before any run, in the session config of record
(`configs/t31/p1_session.json`, `delay_injection`), and `declaration`
refuses anything else: the dense arm at one cell, every mechanism the
recursion prices, a count grid per placement that includes 0, the timed
repeats and the untimed warm-ups per variant, and payload 0 (a copy moves
`link.SMALLEST_PAYLOAD_BYTES`, the size its s_fixed is measured at).

`run` does the step's work in this order:

  1. calibrate: one untimed observed pass, checked against the recording
     by the null pass's own once-only alignment (`recorded_alignment`): the
     observer meets a real prefill's operations through their call chains
     (an SDPA kernel under its `_scaled_dot_product_*` call), and the
     positions step 5 prices by the isolated replay (the embedding's gather
     outside the layers) are seen by neither, so they are never a site;
  2. plan: every (mechanism, placement) through `delay_injection.
     plan_variants` on the session's own priced dense cell (step 6) and
     mechanism table (step 2), on sites the injector accepts. A pair the
     planner or the injector refuses is recorded as refused, with why, and
     is not timed; nothing is re-planned at other counts;
  3. the plan, every variant's prediction and schedule, is written and
     hashed (`write_plan`) before the first timed pass;
  4. time: every variant and the unobserved c = 0 prefill, `warmup`
     untimed passes each, then `reps` rounds, each round every pass once in
     an order rotated by one per round; each pass is synchronise, clock,
     run, synchronise, clock, as the ground-truth timer's block is;
  5. score: per mechanism, `delay_injection.slope_for` (the median of each
     variant's repeats); a mechanism whose measured time does not rise with
     the count is recorded as unresolved, with why. The observer's own
     effect is reported from the two c = 0 passes (`observer_effect`).

Nothing here compares a slope with a bound: Gate A reads the record
(`analysis.gate_a`, leg 3), against the bound committed before the run.
There is no clock sampler in this step: the slopes are differences over
counts within interleaved rounds, and the record says so.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

from ..predict import delay_injection as DI
from ..predict import recursion
from . import delay_injector as di

SCHEMA = "attnbench.delay_injection_session/1"
PLAN = "plan.json"
RESULT = "delay_injection.json"
UNOBSERVED = "unobserved/0"
NO_SAMPLER = ("no clock sampler: the slopes are differences over counts within "
              "interleaved rounds")


class LegThreeError(ValueError):
    pass


@dataclass(frozen=True)
class Declaration:
    arm: str
    length: int
    batch: int
    mechanisms: tuple[str, ...]
    counts: Mapping[str, tuple[int, ...]]     # placement -> counts
    reps: int
    warmup: int
    payload_bytes: int


def declaration(d: Mapping) -> Declaration:
    """The declared set, checked: Gate A's leg 3 injects into the dense arm,
    one mechanism at a time, busy and idle (plan Section 7)."""
    want = {"arm", "length", "batch", "mechanisms", "counts", "reps", "warmup",
            "payload_bytes"}
    if set(d) - {"_comment"} != want:
        raise LegThreeError(f"the leg-3 declaration has keys {sorted(set(d) - {'_comment'})}; "
                            f"it needs exactly {sorted(want)}")
    if d["arm"] != "dense":
        raise LegThreeError(f"leg 3 injects into the dense arm, not {d['arm']!r}")
    mech = tuple(d["mechanisms"])
    unknown = [m for m in mech if m not in recursion.PRICED_MECHANISMS]
    if unknown or len(set(mech)) != len(mech) or not mech:
        raise LegThreeError(f"mechanisms {mech}: each once, each one the recursion prices "
                            f"({', '.join(recursion.PRICED_MECHANISMS)})")
    if set(d["counts"]) != set(DI.PLACEMENTS):
        raise LegThreeError(f"counts are per placement, {DI.PLACEMENTS}; got {sorted(d['counts'])}")
    counts = {p: DI._check_counts(tuple(d["counts"][p])) for p in DI.PLACEMENTS}
    for k in ("length", "batch", "reps", "warmup"):
        if isinstance(d[k], bool) or not isinstance(d[k], int) or d[k] < 1:
            raise LegThreeError(f"{k} is an int >= 1, got {d[k]!r}")
    if d["payload_bytes"] != 0:
        raise LegThreeError("leg 3 injects at payload 0 (s_0 is s_fixed there)")
    return Declaration(arm="dense", length=d["length"], batch=d["batch"], mechanisms=mech,
                       counts=counts, reps=d["reps"], warmup=d["warmup"], payload_bytes=0)


def key(mechanism: str, placement: str, count: int) -> str:
    return f"{mechanism}/{placement}/{count}"


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def recorded_alignment(skip=()) -> "di.Align":
    """The injector's `align` from step 5's rule (`host_profile.align_calls`):
    per observed call, the positions it prices, `skip` (the isolated
    replay's positions, `host_profile.isolated_positions`) passed over."""
    from . import host_profile as hp
    skip = tuple(skip)

    def align(seq, calls):
        met = hp.met_at(seq, (c.name for c in calls), skip=skip)
        try:
            rows = hp.align_calls(seq, met, [(c.name, c.layer) for c in calls], skip=skip,
                                  what="the observed pass")
        except hp.ProfileMismatch as e:
            raise di.InjectionError(str(e)) from e
        return tuple(share for _, _, share in rows)
    return align


def recorded_observer(seq, *, skip=(), device_type: str = "cuda", layers=()):
    """An observer (`delay_injector.Observer`) that reports a call by step 5's
    rule (`host_profile.NullDeviceMode`): an operator in the recording's
    chains (`launching_names` without `skip`) that touches the device or
    names it (`device_call`: a factory call such as `aten::arange` counts),
    whose outputs are not all empty, and that is not the isolated replay's
    own call (`skip`'s issuing operator and layer). The call runs for real;
    `on_call` runs with the mode off, after the call has returned."""
    from torch.utils._python_dispatch import TorchDispatchMode
    from . import host_profile as hp
    launching = hp.launching_names(seq, skip=skip)
    isolated = hp._isolated_origins(seq, tuple(skip)) if skip else frozenset()

    class _Mode(TorchDispatchMode):
        def __init__(self, on_call, state):
            super().__init__()
            self.on_call, self.state = on_call, state

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            name = hp.aten_name(func)
            if name not in launching:
                return func(*args, **kwargs)
            how = hp.device_call(func, args, kwargs, device_type)
            out = func(*args, **kwargs)
            layer = self.state["layer"]
            if how is not None and not hp.all_empty(out) and (name, layer) not in isolated:
                self.on_call(name, layer)
            return out

    def observe(run_prefill, on_call) -> None:
        state = {"layer": None}
        handles = []
        try:
            for i, m in enumerate(layers):
                handles.append(m.register_forward_pre_hook(
                    lambda mod, a, _i=i: state.__setitem__("layer", _i)))
                handles.append(m.register_forward_hook(
                    lambda mod, a, o: state.__setitem__("layer", None)))
            with _Mode(on_call, state):
                run_prefill()
        finally:
            for h in handles:
                h.remove()
    return observe


def plan(base, seq, calls, sync_costs, queue_depth, decl: Declaration, *, align=None):
    """({(mechanism, placement): (variants, schedules)}, {pair: why refused})."""
    allowed = di.injectable_sites(base, seq, calls, align=align)
    planned, refused = {}, {}
    for m in decl.mechanisms:
        for p in DI.PLACEMENTS:
            try:
                vs = DI.plan_variants(base, m, p, decl.counts[p], sync_costs, queue_depth,
                                      payload_bytes=decl.payload_bytes, sites_allowed=allowed)
                planned[(m, p)] = (vs, di.schedules(vs, base, seq, calls, align=align,
                                                    payload_bytes=decl.payload_bytes))
            except (recursion.CostModelError, di.InjectionError) as e:
                refused[(m, p)] = f"{type(e).__name__}: {e}"
    return planned, refused


def plan_record(planned, refused, decl: Declaration, *, n_sites: int, context: Mapping) -> dict:
    variants = []
    for (m, p), (vs, ss) in planned.items():
        for v, s in zip(vs, ss):
            variants.append({"key": key(m, p, v.count), "mechanism": m, "placement": p,
                             "count": v.count, "predicted_s": v.predicted,
                             "predicted_waits_s": list(v.predicted_waits),
                             "schedule": di.to_record(s)})
    rec = {"schema": SCHEMA, "declaration": {**decl.__dict__, "counts": dict(decl.counts)},
           "injectable_sites": n_sites, "context": dict(context), "variants": variants,
           "refused": {f"{m}/{p}": why for (m, p), why in refused.items()}}
    return {**rec, "sha256": _sha(rec)}


def rotated(keys: Sequence[str], r: int) -> list[str]:
    k = r % len(keys)
    return list(keys[k:]) + list(keys[:k])


def time_passes(runs: Mapping[str, Callable[[], object]], *, reps: int, warmup: int,
                synchronize: Callable[[], None], clock: Callable[[], float]) -> dict:
    """{key: [seconds per timed repeat]}: `warmup` untimed passes of each,
    then `reps` rounds of every pass once, the order rotated each round."""
    from ..densities import timed_region
    keys = list(runs)
    for k in keys:
        for _ in range(warmup):
            runs[k]()
    synchronize()
    out = {k: [] for k in keys}
    for r in range(reps):
        for k in rotated(keys, r):
            synchronize()
            with timed_region():
                t0 = clock()
                runs[k]()
                synchronize()
                out[k].append(clock() - t0)
    return out


def score(planned, samples: Mapping[str, Sequence[float]]) -> dict:
    """Per mechanism, its `slope_for` record or why it is unresolved, and
    the observer's effect from the two c = 0 passes."""
    mechs: dict = {}
    for m in dict.fromkeys(m for m, _ in planned):
        vs = [v for (mm, _), (pv, _) in planned.items() if mm == m for v in pv]
        measured = {(v.placement, v.count): samples[key(m, v.placement, v.count)] for v in vs}
        try:
            s = DI.slope_for(m, vs, measured)
        except recursion.CostModelError as e:
            mechs[m] = {"status": "unresolved", "why": str(e)}
            continue
        mechs[m] = {"status": "measured", "slope": s.slope, "placements": {
            p: {"counts": list(ps.counts), "predicted_s": list(ps.predicted),
                "measured_median_s": list(ps.measured), "predicted_slope": ps.predicted_slope,
                "measured_slope": ps.measured_slope, "ratio": ps.ratio}
            for p, ps in s.placements.items()}}
    zero = next((key(m, p, 0) for (m, p) in planned), None)
    eff = None
    if zero is not None and UNOBSERVED in samples:
        e = di.observer_effect(samples[zero], samples[UNOBSERVED])
        eff = {"observed_c0": zero, **e.__dict__, "repeats": list(e.repeats)}
    return {"mechanisms": mechs, "observer_effect": eff}


def slopes_from_record(rec: Mapping) -> dict:
    """{mechanism: delay_injection.MechanismSlope}, recomputed from the
    record's predictions and raw repeats (never from its stored slopes);
    a mechanism with no timed variants is absent."""
    if rec.get("schema") != SCHEMA:
        raise LegThreeError(f"not a leg-3 record: schema {rec.get('schema')!r}")
    plan_rec = rec["plan"]
    body = {k: v for k, v in plan_rec.items() if k != "sha256"}
    if _sha(body) != plan_rec["sha256"]:
        raise LegThreeError("the leg-3 plan does not match its own sha256")
    by_m: dict = {}
    for v in plan_rec["variants"]:
        by_m.setdefault(v["mechanism"], []).append(v)
    out = {}
    for m, vs in by_m.items():
        variants = [DI.Variant(mechanism=m, placement=v["placement"], count=v["count"],
                               sites=tuple(v["schedule"]["sites"]), positions=(),
                               predicted=v["predicted_s"],
                               predicted_waits=tuple(v["predicted_waits_s"])) for v in vs]
        measured = {(v["placement"], v["count"]): rec["samples_s"][v["key"]] for v in vs}
        try:
            out[m] = DI.slope_for(m, variants, measured)
        except recursion.CostModelError:
            continue
    return out


def run(run_prefill: Callable[[], None], seq, base, sync_costs, queue_depth,
        decl: Declaration, *, observer, fire_of: Callable[[str], Callable[[], object]],
        synchronize: Callable[[], None], clock: Callable[[], float],
        write_plan: Callable[[dict], None], context: Optional[Mapping] = None,
        align=None) -> dict:
    """The step's work (module docstring). `write_plan` must have stored the
    plan when it returns; no pass is timed before it does. `align` is the
    injector's (`recorded_alignment` on a real prefill)."""
    calls = di.calibrate(run_prefill, seq, observer, align=align)
    planned, refused = plan(base, seq, calls, sync_costs, queue_depth, decl, align=align)
    n_sites = len(di.injectable_sites(base, seq, calls, align=align))
    prec = plan_record(planned, refused, decl, n_sites=n_sites, context=context or {})
    write_plan(prec)
    fires = {m: fire_of(m) for m in dict.fromkeys(m for m, _ in planned)}
    runs = {}
    for (m, p), (vs, ss) in planned.items():
        for v, s in zip(vs, ss):
            runs[key(m, p, v.count)] = di.injected_prefill(run_prefill, s, fires[m], observer)
    if not runs:
        raise LegThreeError(f"every (mechanism, placement) was refused: {prec['refused']}")
    runs[UNOBSERVED] = run_prefill
    samples = time_passes(runs, reps=decl.reps, warmup=decl.warmup,
                          synchronize=synchronize, clock=clock)
    scored = score(planned, samples)
    return {"schema": SCHEMA, "plan": prec, "samples_s": samples, **scored,
            "statistic": "median", "sampler": NO_SAMPLER,
            "medians_s": {k: statistics.median(v) for k, v in samples.items()}}
