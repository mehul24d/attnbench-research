#!/usr/bin/env python3
"""The steps of Shakedown 2 (T2.13, T2.16, T2.12 and T2.2 on the target
host, and the real NVML reader), one subcommand each.

Shakedown 2 is run by the same runner as T2.27 (`scripts/t227_shakedown.py`,
configured here as `CONFIG`): each step in a process of its own, a log and a
`result.json` per step, the session stopping at the first failure. The step
helpers, the exit codes and `result.json` are `t227_steps.py`'s.

Steps, in the runner's order:

  nvml    the real NVML reader and the sampler process on a card, idle and
          under load: that the throttle bits decode, that the process keeps
          about 10 Hz, and which bits an L4 shows (the software power cap
          included, which is recorded and never flags)
  t216    T2.16: dense-baseline selection (K2) at `--length`, recorded before
          any clocked block of the next step
  t213    T2.13 dry run: the ground-truth harness on the shakedown arms with
          the sampler process, the strict first-clocked-block stamp, the
          traced-against-untraced overhead, the sampler's own cost on a
          host-bound arm, and `dense_selection.check_control` against t216's
          record
  density each arm's backend through the density probe at the working,
          pre-lock densities, one process per compile setting (rehearsal)
  mask_ab compiled XAttention timed with and without its mask as an output
          of the compiled function, at `--length` (ruling, 1 Nov 2026): the
          timing effect of the extra outputs, a finding
  t212    T2.12: each source arm's masks recorded, replayed through
          `PrecomputedMaskBackend` with the real block-sparse kernel, compared
          with the source arm's logits, and timed against it
  t22     T2.2: the null-kernel host profile of one arm's prefill from its
          tape, and the reference mask builder profiled under both instruments
          on this host
  queue   T2.6's Q for the null kernel and the one-element add, cold and warm,
          with the blocker and the waiting threshold moved: the add's Q, its
          lower bound and their sensitivity to the probe

Usage, from the repo root:
    python3 scripts/t228_steps.py run [--dry-run] [--session DIR] [options]
    python3 scripts/t228_steps.py STEP --out DIR [--dry-run] [options]
`run` is `t227_shakedown.py`'s runner with this step list. A CPU dry run
passes `t213` on the dense arm with a scripted reader in place of NVML, and
skips the steps that need CUDA; their wiring is tested with fakes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from t227_steps import (EXIT, PASSED, Skip, StepFailed, _cuda, _harness,  # noqa: E402,F401
                        _jsonable, _need_cuda, _py, _run, _write_json, run_step)

STEPS = ("nvml", "t216", "t213", "density", "mask_ab", "t212", "t22", "queue")

CHECKLIST = {
    "nvml": "the real NVML reader and the sampler process on a card",
    "t216": "T2.16 dense-baseline selection (K2), recorded first",
    "t213": "T2.13 dry run: harness, strict stamp, sampler cost, check_control",
    "density": "each arm's backend through the density probe, every compile setting",
    "mask_ab": "compiled XAttention with and without the mask output, one length",
    "t212": "T2.12 precomputed replay with the real kernel",
    "t22": "T2.2 null-kernel and reference-builder profile on the target host",
    "queue": "T2.6 Q for both kinds: the add's Q, its lower bound and their sensitivity",
}


# --- shared ------------------------------------------------------------------

def _sampler(args):
    """The session's detector: the sampler process at `--sampler-hz`, the
    researcher's flag set. A dry run reads a scripted reader (no card)."""
    from attnbench import clock_sampling as cs
    from t227_steps import _boundary
    factory, fa = ((cs.scripted_reader_factory, ()) if args.dry_run or _boundary()
                   else (cs.nvml_reader_factory, (0,)))
    return cs.ProcessSampler(factory, fa, flag_bits=cs.RESEARCHER_FLAG_BITS,
                             interval_s=1.0 / args.sampler_hz)


class NullSampler:
    """Reads nothing and flags nothing. It exists to measure what the real
    sampler costs a host-bound arm, and is never a session's detector."""
    interval_s = None

    def __init__(self):
        from attnbench import clock_sampling as cs
        self.flag_bits = cs.RESEARCHER_FLAG_BITS

    def begin(self) -> None:
        pass

    def abort(self) -> None:
        pass

    def end(self):
        from attnbench.clock_sampling import BlockClock
        return BlockClock(samples=0, sm_mhz_min=0, sm_mhz_max=0, mem_mhz_min=0,
                          mem_mhz_max=0, throttle_bits_seen=0, violation_delta_ns=None,
                          flagged=False)


def _controls(args, sampler):
    from attnbench import ground_truth as gt
    return gt.Controls(resolution_floor_ms=args.floor_ms,
                       max_rerun_rounds=args.max_rerun, sampler=sampler)


def _tracer():
    import torch
    acts = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(torch.profiler.ProfilerActivity.CUDA)
    return lambda: torch.profiler.profile(activities=acts)


def _medians(result: dict) -> dict:
    return {r["arm"]: r["prefill_ms_median"] for r in result["rows"]}


# --- the real NVML reader --------------------------------------------------------

def _gpu_load(seconds: float) -> None:
    """Keep the card busy with half-precision matrix multiplies."""
    import torch
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        for _ in range(8):
            a = (a @ a).clamp_(-1, 1)
        torch.cuda.synchronize()


def step_nvml(args) -> dict:
    """`NvmlReader` against a card, then the sampler process on it: idle for
    `--sampler-seconds`, then one block under matrix-multiply load. Fails if
    NVML or its throttle function is missing, if the process cannot keep half
    the asked rate, or if the loaded block has fewer than three samples. The
    bits and the violation counters the card reports are findings, not
    failures: they are what the researcher's flag set was chosen without
    having seen."""
    if args.dry_run:
        raise Skip("needs NVML and a card; the sampler process itself is tested on fakes")
    _need_cuda(args, "the load for the sampler's loaded block")
    import torch
    from attnbench import clock_sampling as cs

    reader = cs.NvmlReader(0)
    first = reader.read()
    findings: dict = {
        "gpu": torch.cuda.get_device_name(0),
        "throttle_function": reader._throttle.__name__,
        "violation_counters": reader.observes_transients,
        "first_read": {"sm_mhz": first.sm_mhz, "mem_mhz": first.mem_mhz,
                       "bits": cs.reason_names(first.throttle_bits),
                       "violation_ns": first.violation_ns},
        "flag_set": cs.reason_names(cs.RESEARCHER_FLAG_BITS)}
    with _sampler(args) as ps:
        time.sleep(args.sampler_seconds)
        idle = ps.timing_stats()
        ps.begin()
        _gpu_load(args.sampler_seconds)
        loaded = ps.end()
        findings["idle"] = idle
        findings["loaded_block"] = loaded.to_dict()
        findings["loaded_stats"] = ps.timing_stats()
    seen = loaded.throttle_bits_seen
    findings["gpu_idle_bit_during_load"] = bool(seen & cs.GPU_IDLE)
    findings["software_power_cap_seen_under_load"] = bool(seen & cs.SW_POWER_CAP)
    _write_json(Path(args.out) / "nvml_reader.json", findings)
    hz = idle["achieved_hz"]
    if hz is None or hz < args.sampler_hz / 2:
        raise StepFailed(f"the sampler process kept {hz} Hz, under half of the asked "
                         f"{args.sampler_hz}", findings)
    if loaded.samples < 3 or loaded.sm_mhz_max <= 0:
        raise StepFailed("the loaded block has too few samples, or no SM clock", findings)
    return findings


# --- T2.16 ---------------------------------------------------------------------------

def step_t216(args) -> dict:
    """K2 at one cell (this card, `--length`, batch 1): every planned candidate
    declined where its capability says it faults the device, checked, timed in
    interleaved warm-up rounds, the choice written once with its sha256
    sidecar and read back (which recomputes every median and the choice from
    the samples). Fails if the record does not read back or no candidate was
    correct. An incomplete record (a planned candidate not installed) is a
    finding: `check_control` refuses it in the next step."""
    _need_cuda(args, "the candidate kernels")
    import torch
    from attnbench import dense_selection as ds
    from attnbench.components.shakedown_arms import ARMS

    out = Path(args.out)
    h = _harness(args)
    card = torch.cuda.get_device_name(0)
    cfg = h._cfg(ARMS["dense"], args.length)
    cell = ds.Cell(card=card, context_length=args.length, batch=1)
    choice = ds.select_dense_baseline(cell, cfg, ds.registry_candidates(card),
                                      rounds=args.k2_rounds, resolution_floor_ms=args.floor_ms,
                                      device="cuda")
    path = ds.write_choice(choice, out)
    back = ds.load_choice(path)
    findings = {"cell": cell.key, "record": path.name, "chosen": back.chosen,
                "complete": back.complete, "missing": back.missing,
                "within_floor_of_chosen": back.within_floor_of_chosen,
                "margin_ms": back.margin_ms, "recorded_at": back.recorded_at,
                "resolution_floor_ms": args.floor_ms,
                "floor_note": "provisional; the lock sets the floor",
                "candidates": [{"name": c.name, "status": c.status, "check_kind": c.check_kind,
                                "median_ms": c.median_ms, "reason": c.reason}
                               for c in back.candidates]}
    if back.chosen is None:
        raise StepFailed(f"no candidate was correct for {cell.key}", findings)
    return findings


# --- T2.13 dry run -----------------------------------------------------------------------

def _find_choice(args):
    from attnbench import dense_selection as ds
    if args.k2_dir:
        d = Path(args.k2_dir)
    else:
        found = sorted(Path(args.out).parent.glob("*_t216"))
        d = found[-1] if found else None
    if d is None or not d.is_dir():
        return None
    files = sorted(d.glob("*.json"))
    files = [f for f in files if f.name not in ("result.json", "nvml_reader.json")]
    return ds.load_choice(files[-1]) if files else None


def step_t213(args) -> dict:
    """The ground-truth harness once, on the shakedown arms (`--arms`), with
    the sampler process as the detector: (A) a pass with no sampler at all,
    (B) the untraced pass with the sampler and then the traced one, (A) again.
    From these: the strict stamp lies inside the run; every block was seen by
    at least two samples and the process kept its rate; the sampler's cost is
    B minus the mean of the two A passes, per arm, with their disagreement as
    the noise; the traced-against-untraced overhead; and
    `check_control` against t216's record. Fails on a refused timed region, a
    block no sampler saw, a stamp outside the run, or a sampler under half
    its rate. Throttled rounds, an incomplete result and a `check_control`
    refusal are findings (a dry run's floor is provisional)."""
    from attnbench import dense_selection as ds
    from attnbench import ground_truth as gt
    from attnbench.components.shakedown_arms import ARMS, runnable

    cuda = _cuda()
    if not cuda and not args.dry_run:
        raise StepFailed("no CUDA device")
    names = [a for a in args.arms.split(",") if a]
    unknown = [a for a in names if a not in ARMS]
    if unknown:
        raise StepFailed(f"unknown arms {unknown}; known: {list(ARMS)}")
    if names[:1] != ["dense"]:
        raise StepFailed(f"the dense control must come first in --arms, got {names}")
    names = [a for a in names if runnable(a, cuda)]

    out = Path(args.out)
    h = _harness(args)
    wrapped, ids, prompt_id, arms, score_kwargs = h.timing_arms(names, args.length)
    sync = (lambda: __import__("torch").cuda.synchronize()) if cuda else (lambda: None)
    kw = dict(reps=args.reps, warmup=args.warmup, synchronize=sync, score_kwargs=score_kwargs,
              density_of=gt.kernel_densities)

    gt.reset_process_compile_setting()
    t_call = time.time()
    a1 = gt.time_arms(wrapped, ids, arms, controls=_controls(args, NullSampler()), **kw)
    gt.reset_process_compile_setting()
    with _sampler(args) as ps:
        t_b = time.time()
        both = gt.time_with_profiler_overhead(wrapped, ids, arms, controls=_controls(args, ps),
                                              tracer=_tracer(), **kw)
        t_b_end = time.time()
        stats = ps.timing_stats()
    gt.reset_process_compile_setting()
    a2 = gt.time_arms(wrapped, ids, arms, controls=_controls(args, NullSampler()), **kw)
    h.unwrap()
    b = both["untraced"]
    stamp = both["first_clocked_block_started_at"]

    m_a1, m_a2, m_b = _medians(a1), _medians(a2), _medians(b)
    cost = []
    for n in names:
        if None in (m_a1[n], m_a2[n], m_b[n]):
            cost.append({"arm": n, "cost_ms": None})
            continue
        base = (m_a1[n] + m_a2[n]) / 2
        cost.append({"arm": n, "a1_ms": m_a1[n], "a2_ms": m_a2[n], "with_sampler_ms": m_b[n],
                     "cost_ms": m_b[n] - base, "noise_ms": abs(m_a1[n] - m_a2[n]),
                     "inside_floor": abs(m_b[n] - base) <= args.floor_ms})

    verdict = {"passed": None, "why": "no t216 record (a CPU dry run has none)"}
    choice = _find_choice(args)
    if choice is not None:
        try:
            card = __import__("torch").cuda.get_device_name(0) if cuda else "cpu"
            ds.check_control(choice, getattr(arms[0].backend, "name", None), stamp,
                             key=ds.Cell(card, int(ids.shape[-1]), int(ids.shape[0])))
            verdict = {"passed": True, "chosen": choice.chosen}
        except ds.SelectionError as e:
            verdict = {"passed": False, "why": str(e), "chosen": choice.chosen,
                       "recorded_at": choice.recorded_at, "stamp": stamp}

    blocks = [(r["round"], n, c) for r in b["rounds"] for n, c in r["blocks"].items()]
    unseen = [(rd, n) for rd, n, c in blocks if c["samples"] < 2]
    findings = {
        "arms": names, "length": args.length, "prompt_id": prompt_id,
        "resolution_floor_ms": args.floor_ms,
        "floor_note": "provisional; the lock sets the floor from the dense spread seen here",
        "stamp": {"first_clocked_block_started_at": stamp,
                  "seconds_after_call": stamp - t_call, "inside_untraced_run": t_b <= stamp <= t_b_end,
                  "before_the_traced_run": stamp < both["traced"]["first_clocked_block_started_at"]},
        "complete": b["complete"], "clean_rounds": b["clean_rounds"], "rounds_run": b["rounds_run"],
        "flagged_rounds": [r["round"] for r in b["rounds"] if r["flagged"]],
        "throttle_reasons_seen": {r["arm"]: r["throttle_reasons_seen"] for r in b["rows"]},
        "sampler": stats, "blocks_seen_by_under_two_samples": unseen,
        "sampler_cost": cost, "profiler_overhead": both["overhead"],
        "dense_control": b["dense_control"], "check_control": verdict,
        "medians_ms": m_b}
    _write_json(out / "ground_truth.json", {"untraced": b, "traced": both["traced"],
                                            "no_sampler_before": a1, "no_sampler_after": a2})
    _write_json(out / "sampler_cost.json", cost)
    in_run = t_call <= stamp <= t_b_end
    if not in_run:
        raise StepFailed("the first-clocked-block stamp is outside the run", findings)
    if unseen:
        raise StepFailed(f"{len(unseen)} blocks were seen by fewer than two samples", findings)
    hz = stats["achieved_hz"]
    if hz is None or hz < args.sampler_hz / 2:
        raise StepFailed(f"the sampler process kept {hz} Hz of the asked {args.sampler_hz}",
                         findings)
    return findings


# --- T2.12 -------------------------------------------------------------------------------

def step_t212(args) -> dict:
    """For each source arm (`--replay-arms`): its masks recorded in one
    untimed run, saved with their sha256, replayed through
    `PrecomputedMaskBackend` with the real `block_sparse_attn_func`, and the
    logits compared with the source arm's own. Then the dense control, the
    source arm and the precomputed arm are timed together (`--reps`), so the
    difference is the estimator's compute, synchronisations and mask build.
    The plan expects a bitwise match for XAttention and does not know it for
    mean-pool, so a mean-pool mismatch is a finding and an XAttention one
    fails the step. Executed density must equal the source's: both are the
    kernel-input recorder's (`densities.KernelInputRecorder`, ruling of
    1 Nov 2026), which sees the mask the kernel received for any arm, and a
    comparison over no layers on either side is refused."""
    _need_cuda(args, "the block-sparse kernel")
    import torch
    from attnbench import clock_sampling as cs  # noqa: F401
    from attnbench import densities as D
    from attnbench import ground_truth as gt
    from attnbench import precomputed_masks as pm
    from attnbench.components.shakedown_arms import ARMS

    out = Path(args.out)
    names = ["dense"] + [a for a in args.replay_arms.split(",") if a]
    bad = [n for n in names[1:] if n not in ARMS or n == "dense"]
    if bad:
        raise StepFailed(f"not source arms for a mask replay: {bad}")
    h = _harness(args)
    wrapped, ids, prompt_id, arms, score_kwargs = h.timing_arms(names, args.length)
    dense = arms[0]
    findings: dict = {"length": args.length, "prompt_id": prompt_id, "arms": {}}
    timed = [dense]
    failures = []
    for a in arms[1:]:
        spec = ARMS[a.name]
        kw = gt.untimed_setup(wrapped, ids, a, score_kwargs=score_kwargs)
        rec = pm.MaskRecorder(a.backend, estimator=a.name, code_path=a.name,
                              mask_selector="xattn_native" if spec.xattention else "head_uniform")
        if hasattr(a.backend, "reset_density"):
            a.backend.reset_density()
        wrapped.run_measured(ids, rec, cfg=a.cfg, logits_to_keep=1, **kw)
        mset = rec.mask_set(ids, n_heads_q=a.cfg.n_heads_q)
        mset.save(out / f"masks_{a.name}")
        if hasattr(a.backend, "reset_density"):
            a.backend.reset_density()
        with D.KernelInputRecorder() as src_rec:
            ref = wrapped.run_measured(ids, a.backend, cfg=a.cfg, logits_to_keep=1, **kw)
        src_density = src_rec.per_call_density()
        pre = pm.PrecomputedMaskBackend(mset, ids, device="cuda")
        pre.reset_density()
        with D.KernelInputRecorder() as pre_rec:
            got = wrapped.run_measured(ids, pre, cfg=dense.cfg, logits_to_keep=1)
        pre_density = pre_rec.per_call_density()
        torch.cuda.synchronize()
        bitwise = bool(torch.equal(got.logits, ref.logits))
        diff = float((got.logits.float() - ref.logits.float()).abs().max())
        try:
            same_density = D.same_per_layer(f"{a.name}'s executed density", src_density,
                                            pre_density, names=("source", "precomputed"))
        except D.DensityMismatch as e:
            raise StepFailed(str(e), findings)
        findings["arms"][a.name] = {
            "layers_recorded": len(mset.masks), "mask_density": mset.mask_density(),
            "bitwise_equal_logits": bitwise, "max_abs_logit_diff": diff,
            "executed_density_equal": same_density,
            "density_record_layers": {"source": len(src_density),           # what was compared
                                      "precomputed": len(pre_density)},
            "density_source": "kernel-input recorder",
            "expected_bitwise": bool(spec.xattention)}
        if not same_density:
            failures.append(f"{a.name}: the replay's executed density differs from its source's")
        if spec.xattention and not bitwise:
            failures.append(f"{a.name}: the replay is not bitwise equal to its source "
                            f"(max abs {diff}); the plan expects it to be")
        # Findings only (31 Oct 2026), after every check above has read its
        # inputs: these runs append to the density records.
        from attnbench.components import mask_diagnosis as md
        diag = md.guarded(
            md.t212_diagnosis,
            lambda: wrapped.run_measured(ids, a.backend, cfg=a.cfg, logits_to_keep=1, **kw),
            lambda: wrapped.run_measured(ids, pre, cfg=dense.cfg, logits_to_keep=1),
            [mset.masks[i] for i in sorted(mset.masks)], source_backend=a.backend,
            precomputed_backend=pre)
        _write_json(out / f"mask_diagnosis_{a.name}.json", diag)
        findings["arms"][a.name]["mask_diagnosis"] = md.summaries(diag)
        timed += [a, gt.Arm(f"precomputed_{a.name}", gt.PRECOMPUTED, pre, dense.cfg,
                            prompt_digest=mset.prompt_digest)]
    if len(timed) > 1:
        sync = torch.cuda.synchronize
        with _sampler(args) as ps:
            res = gt.time_arms(wrapped, ids, timed, reps=args.reps, warmup=args.warmup,
                               controls=_controls(args, ps), density_of=gt.kernel_densities,
                               synchronize=sync, score_kwargs=score_kwargs)
        med = _medians(res)
        findings["timing"] = {"medians_ms": med, "complete": res["complete"],
                              "clean_rounds": res["clean_rounds"],
                              "labels": {r["arm"]: r["arm_label"] for r in res["rows"]},
                              "estimator_cost_ms": {
                                  a.name: (med[a.name] - med[f"precomputed_{a.name}"])
                                  if None not in (med[a.name], med[f"precomputed_{a.name}"]) else None
                                  for a in arms[1:]}}
        _write_json(out / "timing.json", res)
    h.unwrap()
    _write_json(out / "replay.json", findings)
    if failures:
        raise StepFailed("; ".join(failures), findings)
    return findings


# --- T2.2 --------------------------------------------------------------------------------

def step_t22(args) -> dict:
    """The null-kernel host profile of one arm (`--profile-arm`) from its
    recorded tape: `--profile-reps` passes, the per-position median, every
    repetition kept. Then `profile_reference_builder.py` on this host, both
    instruments side by side, at `--builder-lengths`, and the 32768 row read
    against the plan's unbanked 94%. Fails if a pass is refused
    (`ProfileMismatch`: the tape and the sequence disagree) or the builder
    profile is not written. The values are findings."""
    _need_cuda(args, "the null kernel and the recorded tape")
    import pandas as pd
    from attnbench.components import host_profile as hp
    from attnbench.components import recorder

    out = Path(args.out)
    h = _harness(args)
    arm = h.ready(args.profile_arm, args.length)
    rec = recorder.record_arm_prefill(arm.run_prefill, out / "recording",
                                      code_path=arm.code_path, context_length=args.length,
                                      batch=1, prompt_id=arm.prompt_id, layers=arm.layers)
    entries = () if arm.spec.name == "dense" else (hp.block_sparse_entry(),)
    from t31_steps import host_profile_cell, working_values
    prof, check = host_profile_cell(arm, rec, out, reps=args.profile_reps, entries=entries,
                                    tolerance=working_values()[0]["chain_check_tolerance"])
    (out / "host_profile.json").write_text(hp.dumps(prof))
    findings: dict = {
        "arm": arm.name, "length": args.length, "positions": len(prof.host_s),
        "cost_sources": prof.sources(), "chain_check": check,
        "host_total_ms": prof.total_s * 1e3, "launch_floor_us": prof.launch_floor_s * 1e6,
        "reps": prof.reps, "sequence_sha256": prof.sequence_sha256}
    h.unwrap()

    bdir = out / "builder"
    rc = _run(_py("profile_reference_builder.py", "--out", str(bdir),
                  "--seq-lens", args.builder_lengths, "--reps", str(args.builder_reps)))
    path = bdir / "builder_profile.parquet"
    if rc or not path.exists():
        raise StepFailed("profile_reference_builder.py did not write its profile", findings)
    df = pd.read_parquet(path)
    cols = ["seq_len", "unprofiled_s_per_call_median", "aten_calls_per_call", "python_share",
            "python_share_vs_unprofiled", "cprofile_python_share"]
    findings["builder"] = df[cols].to_dict("records")
    top = df[df.seq_len == df.seq_len.max()].iloc[0]
    findings["against_the_plans_94pct"] = {
        "seq_len": int(top.seq_len), "plan_figure": 0.94,
        "cprofile": float(top.cprofile_python_share), "torch_profiler": float(top.python_share),
        "lower_bound": float(top.python_share_vs_unprofiled),
        "note": "the plan's figure was an unbanked laptop cProfile; this is the target host"}
    return findings


# --- the density probe ----------------------------------------------------------------------

def density_arms(vals: dict, group: str):
    """(names, densities) for one ground-truth group of the working values:
    the dense control first, then each of the group's arms at each density.
    The same expansion as T3.1's `group_arms` at batch 1."""
    from t31_steps import nominal_density
    names, dens = ["dense"], [None]
    for arm in vals["gt_groups"][group]:
        if arm == "dense":
            continue
        for d in vals["densities"][arm]:
            names.append(arm)
            dens.append(nominal_density(d))
    return names, dens


def step_density(args) -> dict:
    """Each arm's backend through the session's density probe
    (`ground_truth.kernel_densities`), at the working, pre-lock densities of
    `docs/lock_value_table.md` and under each compile setting T3.1's
    ground-truth groups give it: one child process per group, since Dynamo
    keeps one setting per process. This is the probe T3.1's ground truth runs
    after every timed block; here it runs untimed, before any. Fails on a
    child that fails: a backend that keeps no per-layer mask density, or a
    row `densities.density_problems` refuses. Each arm's mask density against
    its nominal one is a finding (the mask gate, T2.23, is not built). The
    values are a rehearsal's; nothing here is a result."""
    _need_cuda(args, "the block-sparse kernel's density probe")
    from t31_steps import working_values
    vals, sha = working_values()
    out = Path(args.out)
    found: dict = {"values": "rehearsal (working, pre-lock)", "values_sha256": sha,
                   "length": args.length, "groups": {}}
    bad = []
    for group in vals["gt_groups"]:
        d = out / group
        rc = _run(_py("t228_steps.py", "density-one", "--group", group, "--out", str(d),
                      "--length", str(args.length), *(["--dry-run"] if args.dry_run else []),
                      *(["--cpu-boundary"] if getattr(args, "cpu_boundary", False) else [])))
        path = d / "densities.json"
        found["groups"][group] = {"rc": rc,
                                  "arms": json.loads(path.read_text()) if path.exists() else None}
        if rc:
            bad.append(group)
    if bad:
        raise StepFailed(f"the density probe failed for groups {bad}; see their logs", found)
    return found


def density_one(args) -> int:
    """One group of `step_density`, in its own process: prepare the group's
    arms as `time_arms` would (the same untimed setup), run each once through
    `kernel_densities`, check each row, write `densities.json`."""
    import torch
    from attnbench import densities as D
    from attnbench import ground_truth as gt
    from attnbench.components.shakedown_arms import ARMS
    from t31_steps import group_setting, working_values

    vals, _ = working_values()
    setting = group_setting(args.group)
    names, dens = density_arms(vals, args.group)
    h = _harness(args)
    backends, settings = [None], [None]
    for n in names[1:]:
        be = h._backend(ARMS[n])
        if setting is not None:
            from attnbench import cpu_boundary
            from attnbench.backends.compiled import CompiledBackend
            be = CompiledBackend(be, setting, **cpu_boundary.compile_kwargs())
        backends.append(be)
        settings.append(setting)
    wrapped, ids, prompt_id, arms, score_kwargs = h.timing_arms(
        names, args.length, densities=dens, backends=backends, compile_settings=settings)
    gt._claim_compile_setting(arms)
    rows, failed = [], []
    for a, nominal in zip(arms, dens):
        kw = gt.untimed_setup(wrapped, ids, a, score_kwargs=score_kwargs)

        def run(backend=None, a=a, kw=kw):
            wrapped.run_measured(ids, backend if backend is not None else a.backend,
                                 cfg=a.cfg, logits_to_keep=1, **kw)
        run()                                    # warm: a compiled arm compiles here
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        row = {"arm": a.name, "compile_setting": a.compile_setting, "nominal_density": nominal,
               "prompt_id": prompt_id, "length": int(ids.shape[-1])}
        try:
            row.update(gt.kernel_densities(a, run))
        except RuntimeError as e:
            row["problems"] = [str(e)]
        else:
            row["problems"] = D.density_problems(row, dense=a.kind == gt.DENSE)
            if nominal is not None and row["mask_density"] is not None:
                row["mask_minus_nominal"] = row["mask_density"] - nominal
        if row["problems"]:
            failed.append(a.name)
        rows.append(row)
    h.unwrap()
    _write_json(Path(args.out) / "densities.json", rows)
    for r in rows:
        print(f"  {r['arm']:40s} mask {r.get('mask_density')}  executed "
              f"{r.get('executed_density')}  {r['problems'] or 'ok'}", flush=True)
    return 1 if failed else 0


# --- the mask-output A/B -------------------------------------------------------------------

# The A/B's processes, one side each: two compiled wrappers of one arm in one
# process share `block_sparse_attn`'s resume frames, whose cache entries the
# other side's calls invalidate, and Shakedown 5's interleaved run reached
# Dynamo's recompile limit (256); each side alone compiles in two prefills.
AB_ORDER = ("without_mask_output", "with_mask_output", "with_mask_output", "without_mask_output")


def step_mask_ab(args) -> dict:
    """Compiled XAttention (`compiled_default`) timed with its mask as an
    output of the compiled function (B, how T3.1 times it, so the timed graph
    is its density twin's) and without (A, the output alone), at `--length`
    (researcher's ruling, 1 Nov 2026). One side per process, in the order
    `AB_ORDER`, so drift falls on both sides alike. A finding: if the
    difference is material, compiled rows take their density from the eager
    twin, labelled `compiled.EAGER_TWIN_LABEL`; the researcher rules on it."""
    _need_cuda(args, "the compiled arm's CUDA graphs")
    out = Path(args.out)
    times: dict = {k: [] for k in AB_ORDER}
    runs = []
    for i, side in enumerate(AB_ORDER):
        part = out / f"mask_ab_{i}_{side}.json"
        rc = _run(_py("t228_steps.py", "mask-ab-one", "--out", str(out), "--side", side,
                      "--part", str(i), "--length", str(args.length), "--reps", str(args.reps),
                      "--warmup", str(args.warmup),
                      *(["--dry-run"] if args.dry_run else []),
                      *(["--cpu-boundary"] if getattr(args, "cpu_boundary", False) else [])))
        if rc or not part.exists():
            raise StepFailed(f"the mask-output A/B failed in process {i} ({side}); see its log",
                             {"rc": rc, "process": i, "side": side, "runs": runs})
        run = json.loads(part.read_text())
        runs.append({k: v for k, v in run.items() if k != "rounds_s"})
        times[side] += run["rounds_s"]
    first = runs[0]
    found = {"arm": "xattention", "compile_setting": "default", "length": first["length"],
             "prompt_id": first["prompt_id"], "reps": args.reps, "warmup": args.warmup,
             "order": list(AB_ORDER), "runs": runs, **ab_summary(times), "rounds_s": times,
             "if_material": first["if_material"]}
    _write_json(out / "mask_ab.json", found)
    print(f"  mask output: B - A = {found['b_minus_a_ms']:+.3f} ms "
          f"({found['b_minus_a_pct']:+.2f}%), spread {found['spread_ms']:.3f} ms", flush=True)
    return found


def ab_summary(times: dict) -> dict:
    """Medians, spreads and the difference of the A/B's rounds (seconds)."""
    def q(xs, f):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(f * (len(xs) - 1) + 0.5))]
    rows = {k: {"rounds": len(v), "median_ms": statistics.median(v) * 1e3,
                "iqr_ms": (q(v, 0.75) - q(v, 0.25)) * 1e3} for k, v in times.items()}
    a, b = rows["without_mask_output"]["median_ms"], rows["with_mask_output"]["median_ms"]
    return {"arms": rows, "b_minus_a_ms": b - a, "b_minus_a_pct": 100.0 * (b - a) / a,
            "spread_ms": max(r["iqr_ms"] for r in rows.values())}


def mask_ab_one(args) -> int:
    """One side of the A/B (`--side`) in its own process: `--warmup` prefills
    to compile, then `--reps` timed ones, none of which may compile."""
    import torch
    from attnbench import cpu_boundary
    from attnbench import ground_truth as gt
    from attnbench.backends import compiled
    from attnbench.components.shakedown_arms import ARMS
    if args.side not in AB_ORDER:
        raise SystemExit(f"--side must be one of {sorted(set(AB_ORDER))}")
    h = _harness(args)
    eager = h._backend(ARMS["xattention"])
    be = compiled.CompiledBackend(eager, "default",
                                  mask_output=args.side == "with_mask_output",
                                  **cpu_boundary.compile_kwargs())
    wrapped, ids, prompt_id, arms, score_kwargs = h.timing_arms(
        ["dense", "xattention"], args.length, densities=[None, None],
        backends=[None, be], compile_settings=[None, "default"])
    a = arms[1]
    setup = gt.untimed_setup(wrapped, ids, a, score_kwargs=score_kwargs)
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)

    def once() -> float:
        sync()
        t0 = time.perf_counter()
        wrapped.run_measured(ids, be, cfg=a.cfg, logits_to_keep=1, **setup)
        sync()
        return time.perf_counter() - t0
    for _ in range(args.warmup):                        # compiles here
        once()
    before = compiled.graphs_compiled()
    times = [once() for _ in range(args.reps)]
    h.unwrap()
    if compiled.graphs_compiled() != before:
        raise SystemExit(f"{compiled.graphs_compiled() - before} graphs compiled inside the "
                         f"timed prefills; raise --warmup")
    _write_json(Path(args.out) / f"mask_ab_{args.part}_{args.side}.json", {
        "side": args.side, "length": int(ids.shape[-1]), "prompt_id": prompt_id,
        "compiles": len(be.events), "rounds_s": times,
        "if_material": f"compiled rows' density from the eager twin, labelled "
                       f"{compiled.EAGER_TWIN_LABEL!r}"})
    return 0


# --- the launch queue --------------------------------------------------------------------

# Shakedown 1's row (cold, blocker 2 s, fraction 0.25) first, then the same
# warm, then the blocker and the waiting threshold moved one at a time.
QUEUE_SETTINGS = (
    {"name": "cold_2s_0.25", "warm": False, "blocker_s": 2.0, "blocked_fraction": 0.25},
    {"name": "warm_2s_0.25", "warm": True, "blocker_s": 2.0, "blocked_fraction": 0.25},
    {"name": "warm_1s_0.25", "warm": True, "blocker_s": 1.0, "blocked_fraction": 0.25},
    {"name": "warm_4s_0.25", "warm": True, "blocker_s": 4.0, "blocked_fraction": 0.25},
    {"name": "warm_2s_0.1", "warm": True, "blocker_s": 2.0, "blocked_fraction": 0.1},
    {"name": "warm_2s_0.5", "warm": True, "blocker_s": 2.0, "blocked_fraction": 0.5},
)


def step_queue(args) -> dict:
    """T2.6's Q for both kinds under each of `QUEUE_SETTINGS`, one
    `measure_launch_queue.py` row each: the one-element add's Q and its lower
    bound, and how both move with the probe's settings. Shakedown 1 left the
    add unresolved (`1,1022,1022,1022,1022`, lower bound 1); the warm rows test
    whether that 1 was the kernel's first launch. Fails only if the null
    kernel's Q is not agreed in Shakedown 1's own setting, as T2.27's step
    does; the add's Q, agreed or not, is a finding. The model's Q stays the
    null kernel's (Appendix A, 25 Oct 2026)."""
    import pandas as pd
    out = Path(args.out)
    found: dict = {"settings": {}}
    for s in QUEUE_SETTINGS:
        d = out / s["name"]
        cmd = _py("measure_launch_queue.py", "--out", str(d),
                  "--blocker-s", str(s["blocker_s"]), "--blocked-fraction", str(s["blocked_fraction"]))
        if not s["warm"]:
            cmd.append("--cold")
        if args.dry_run:          # a small simulated queue, so the dry run is quick
            cmd += ["--dry-run", "--dry-run-depth", "64", "--launches", "512", "--trials", "2"]
        if _run(cmd):
            raise StepFailed(f"measure_launch_queue.py failed in setting {s['name']}", found)
        row = pd.read_parquet(d / "launch_queue.parquet").iloc[0].to_dict()
        found["settings"][s["name"]] = {
            **s, "warm_launch": row.get("warm_launch"),
            **{k: row[k] for k in row if k.endswith(("_q", "_q_trials", "_q_lower_bound"))}}
    got = found["settings"]
    adds = [v["add_q"] for v in got.values()]
    agreed = sorted({q for q in adds if q is not None})
    found["add"] = {
        "q_by_setting": {n: v["add_q"] for n, v in got.items()},
        "lower_bound_by_setting": {n: v["add_q_lower_bound"] for n, v in got.items()},
        "lower_bound": min(v["add_q_lower_bound"] for v in got.values()),
        "agreed_values": agreed,
        "same_as_null_kernel": {n: v["add_q"] == v["sleep0_q"] for n, v in got.items()
                                if v["add_q"] is not None}}
    if got[QUEUE_SETTINGS[0]["name"]]["sleep0_q"] is None:
        raise StepFailed("the null kernel's Q was not agreed in Shakedown 1's setting", found)
    return found


# --- entry points ----------------------------------------------------------------------------

STEP_FUNCS = {"nvml": step_nvml, "t216": step_t216, "t213": step_t213, "density": step_density,
              "mask_ab": step_mask_ab, "t212": step_t212, "t22": step_t22, "queue": step_queue}
if tuple(STEP_FUNCS) != STEPS or tuple(CHECKLIST) != STEPS:   # not `assert`: T2.17
    raise RuntimeError("STEP_FUNCS, CHECKLIST and STEPS disagree")


def parser() -> argparse.ArgumentParser:
    from attnbench.components.shakedown_arms import ARMS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=STEPS + ("run", "density-one", "mask-ab-one"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cpu-boundary", action="store_true",
                    help="the real path on a CPU behind the stubbed GPU boundary")
    ap.add_argument("--length", type=int, default=8192, help="the cell, for every step")
    ap.add_argument("--arms", default="dense,oracle,meanpool,xattention",
                    help="t213's arms; the dense control first")
    ap.add_argument("--reps", type=int, default=10, help="clean rounds per timing pass")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--max-rerun", type=int, default=3)
    ap.add_argument("--floor-ms", type=float, default=1.0,
                    help="PROVISIONAL resolution floor for the dry run; the lock sets the real one")
    ap.add_argument("--sampler-hz", type=float, default=10.0)
    ap.add_argument("--sampler-seconds", type=float, default=5.0, help="nvml: idle, then loaded")
    ap.add_argument("--k2-rounds", type=int, default=7)
    ap.add_argument("--k2-dir", default=None, help="t213: the directory holding t216's record")
    ap.add_argument("--replay-arms", default="meanpool,xattention",
                    help="t212: comma-separated source arms")
    ap.add_argument("--profile-arm", default="meanpool", choices=sorted(ARMS))
    ap.add_argument("--profile-reps", type=int, default=5)
    ap.add_argument("--builder-lengths", default="2048,4096,8192,16384,32768")
    ap.add_argument("--builder-reps", type=int, default=10)
    ap.add_argument("--group", default=None, help="density-one: the ground-truth group")
    ap.add_argument("--side", default=None, help="mask-ab-one: the A/B side")
    ap.add_argument("--part", type=int, default=0, help="mask-ab-one: the process's place in AB_ORDER")
    return ap


def _config():
    from t227_shakedown import Shakedown
    return Shakedown(STEPS, CHECKLIST, "t228_steps.py", "t228", "t228_shakedown", parser,
                     "Shakedown 2 (T2.13, T2.16, T2.12, T2.2, NVML, density, Q)")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "run":
        from t227_shakedown import main as run_main
        return run_main(argv[1:], config=_config())
    args = parser().parse_args(argv)
    children = {"density-one": density_one, "mask-ab-one": mask_ab_one}
    if args.cpu_boundary:
        from t227_steps import boundary_main
        return boundary_main(args, lambda: children[args.step](args) if args.step in children
                             else run_step(args.step, args, STEP_FUNCS))
    if args.step in children:
        return children[args.step](args)
    return run_step(args.step, args, STEP_FUNCS)


if __name__ == "__main__":
    raise SystemExit(main())
