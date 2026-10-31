#!/usr/bin/env python3
"""The steps of the L4 shakedown (T2.27), one subcommand each.

`scripts/t227_shakedown.py` runs these in order, each in its own process
with its own log, and stops at the first failure. Each step writes
`result.json` in its directory:

    {"step", "status": "passed" | "failed" | "skipped", "reason",
     "dry_run", "findings": {...}}

and exits 0 (passed), 1 (failed) or 3 (skipped: needs a CUDA device and this
is a CPU dry run). The runner treats a skip as a failure unless it is a dry
run. Every row a step writes carries a provenance stamp; the first step
refuses a dirty tree outside a dry run, so the rows that follow are banked
with `git_dirty = False`.

Steps, in the runner's order (what each checks is in its docstring):

  t21_t220      T2.1 host-speed probe and T2.20 provenance fields
  zoo           T2.3 mechanism zoo, one stream and two
  link          T2.5 link microbenchmarks
  launch_queue  T2.6 launch-queue depth
  nsys          T2.7 Kineto checks, then the separate nsys-traced pass
  replay        T2.4 on one arm and cell
  probe         T2.18 structural probe, every arm
  assert_gate   T2.17 per-session checks and bitwise gate on real prompts
  graph_breaks  T2.21 graph-break report, each arm and setting in its own process
  null_kernel   the null kernel under the profiler
  figure1       Figure 1's trace pair

Two helper subcommands are run by steps, not by the runner: `arm-pass` (one
untimed prefill per arm, the process nsys traces) and `graph-breaks-one`
(one arm and setting, since Dynamo allows one setting per process).

Usage, from the repo root:
    python3 scripts/t227_steps.py STEP --out DIR [--dry-run] [options]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASSED, FAILED, SKIPPED = "passed", "failed", "skipped"
EXIT = {PASSED: 0, FAILED: 1, SKIPPED: 3}

STEPS = ("t21_t220", "zoo", "link", "launch_queue", "nsys", "replay", "probe",
         "assert_gate", "graph_breaks", "null_kernel", "figure1")

EXPECTED_INSTANCE_TYPE = "g2-standard-8"
# A row name (`attnbench.provisioning`), which is what a row carries. The
# launch name "STANDARD" failed every real t21_t220 (Shakedown 2, 31 Oct 2026).
from attnbench.provisioning import ON_DEMAND as EXPECTED_PROVISIONING  # noqa: E402
EXPECTED_GPU = "L4"


class Skip(Exception):
    """This step needs what a CPU dry run does not have."""


class StepFailed(Exception):
    def __init__(self, reason: str, findings: dict | None = None):
        super().__init__(reason)
        self.findings = findings or {}


# --- shared ------------------------------------------------------------------

def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if hasattr(x, "item") and callable(x.item) and getattr(x, "ndim", 1) == 0:
        return x.item()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


def _write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=1))
    return path


def _cuda() -> bool:
    import torch
    return torch.cuda.is_available()


def _need_cuda(args, what: str) -> None:
    if not _cuda():
        if args.dry_run:
            raise Skip(f"needs a CUDA device: {what}")
        raise StepFailed(f"no CUDA device, which this needs: {what}")


def _harness(args):
    from attnbench.components.shakedown_arms import Harness
    return Harness(device="cuda" if _cuda() else "cpu", dry_run=args.dry_run,
                   score_cache_dir=str(Path(args.out) / "score_cache"))


def _arms(args) -> list[str]:
    from attnbench.components.shakedown_arms import ARMS, runnable
    names = [a for a in args.arms.split(",") if a]
    unknown = [a for a in names if a not in ARMS]
    if unknown:
        raise StepFailed(f"unknown arms {unknown}; known: {list(ARMS)}")
    cuda = _cuda()
    if not cuda and not args.dry_run:
        raise StepFailed("no CUDA device")
    return [a for a in names if runnable(a, cuda)]


def _py(script: str, *argv: str) -> list[str]:
    return [sys.executable, str(ROOT / "scripts" / script), *argv]


def _run(cmd: list[str], **kw) -> int:
    """A child process, its output into this step's log."""
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, **kw).returncode


def _sequence_summary(seq) -> dict:
    from attnbench.components import op_sequence as ops
    sc = ops.stream_check(seq)
    return {"ops": len(seq), "layers": len(seq.layers), "streams": list(seq.streams),
            "single_stream": sc.single_stream, "sync_column": ops.sync_column(seq),
            "sha256": ops.record_sha256(ops.to_record(seq))}


def _save_sequence(seq, path: Path) -> None:
    from attnbench.components import op_sequence as ops
    rec = ops.to_record(seq)
    _write_json(path, {**rec, "sha256": ops.record_sha256(rec)})


# --- T2.1 and T2.20 ----------------------------------------------------------

T220_FIELDS = ("cpu_microcode", "cloud_provider", "instance_type", "instance_generation",
               "provisioning_model", "provisioning_model_problem", "python_optimize")


def step_t21_t220(args) -> dict:
    """T2.1: CPU snapshot and the host-speed probe, tag `start` (the null-kernel
    burst is required on the L4). T2.20: the provenance stamp on the probe's
    row carries every T2.20 field; on the L4 it names the g2-standard-8, the
    STANDARD provisioning model with no problem, the L4, sys.flags.optimize 0,
    a CPU microcode, and a clean tree. Executed density is read from the
    tensors handed to the block-sparse kernel in one meanpool prefill."""
    import pandas as pd
    from attnbench import provenance

    out = Path(args.out)
    if _run(_py("host_cpu_probe.py", "snapshot", "--out", str(out), "--tag", "start")):
        raise StepFailed("host_cpu_probe.py snapshot failed")
    speed = _py("host_cpu_probe.py", "speed", "--out", str(out), "--tag", "start")
    if not args.dry_run:
        speed.append("--require-gpu")
    if _run(speed):
        raise StepFailed("host_cpu_probe.py speed failed (T2.1)")
    row = pd.read_parquet(out / "host_speed_start.parquet").iloc[0].to_dict()

    problems, notes = [], []
    missing = [f for f in T220_FIELDS if f not in row]
    if missing:
        problems.append(f"T2.20 fields missing from the row: {missing}")
    if row.get("python_optimize") != 0:
        problems.append(f"python_optimize is {row.get('python_optimize')!r}, not 0")
    integrity = provenance.stamp_integrity_problems(row)
    if not args.dry_run:
        problems += integrity
        expect = {"cloud_provider": "gcp", "instance_type": EXPECTED_INSTANCE_TYPE,
                  "provisioning_model": EXPECTED_PROVISIONING,
                  "provisioning_model_problem": None}
        for k, v in expect.items():
            if row.get(k) != v:
                problems.append(f"{k} is {row.get(k)!r}, expected {v!r}")
        for k in ("cpu_microcode", "instance_generation"):
            if not row.get(k):
                problems.append(f"{k} is empty")
        if EXPECTED_GPU not in str(row.get("gpu_name")):
            problems.append(f"gpu_name {row.get('gpu_name')!r} is not an {EXPECTED_GPU}")
        if row.get("null_launch_s_median") is None:
            problems.append("no null-kernel burst on a GPU instance")
    else:
        notes.append(f"dry run: integrity problems recorded, not enforced: {integrity}")

    densities = None
    if _cuda():
        from attnbench import densities as dens
        h = _harness(args)
        arm = h.prepare("meanpool", args.probe_short)
        arm.run_prefill()                         # warm, outside the recorder
        with dens.KernelInputRecorder() as rec:
            arm.run_prefill()
        import torch
        torch.cuda.synchronize()
        ed = rec.executed_density()
        dp = dens.density_problems({"mask_density": ed, "executed_density": ed}, dense=False)
        densities = {"arm": "meanpool", "length": arm.length, "budget": arm.budget,
                     "kernel_calls": len(rec.per_call), "executed_density": ed,
                     "dense_fields_ok": not dens.density_problems(dens.DENSE_FIELDS, dense=True)}
        if dp or len(rec.per_call) != len(arm.layers):
            problems.append(f"executed density: {dp}, {len(rec.per_call)} kernel calls "
                            f"for {len(arm.layers)} layers")
    else:
        notes.append("dry run: executed density needs the block-sparse kernel (CUDA)")

    findings = {"t220_fields": {f: row.get(f) for f in T220_FIELDS},
                "gpu_name": row.get("gpu_name"), "git_commit": row.get("git_commit"),
                "git_dirty": row.get("git_dirty"),
                "null_launch_s_median": row.get("null_launch_s_median"),
                "densities": densities, "notes": notes}
    if problems:
        raise StepFailed("; ".join(problems), findings)
    return findings


# --- T2.3: the mechanism zoo -------------------------------------------------

def step_zoo(args) -> dict:
    """Both zoos recorded under T2.3's recording rules
    (`trace_recording.record_prefill_trace`: one warm-up outside the trace,
    the device synchronised before the region opens and after it closes,
    one `attnbench::prefill` region, one layer annotation per step, CPU and
    CUDA activity, one host thread), parsed by `op_sequence.parse_trace`
    against a code path whose GPU is the trace's own. Single stream: every
    step's observed waits must be one of its listed answers, and where more
    than one is listed the answer that held is recorded. Two streams:
    `stream_check` must flag the arm."""
    _need_cuda(args, "the mechanism zoo")
    import torch
    from attnbench.components import op_sequence as ops
    from attnbench.components import trace_recording as tr
    from attnbench.components.shakedown_arms import libraries

    out = Path(args.out)
    gpu = torch.cuda.get_device_name(0)

    def code_path(name):
        return ops.CodePath(arm=name, gpu=gpu, implementation="zoo",
                            libraries=libraries(), python_optimize=sys.flags.optimize)

    path = tr.record_prefill_trace(lambda: tr.run_mechanism_zoo(), out / "zoo_trace.json",
                                   warmup=1, cuda=True)
    seq = ops.parse_trace(path, code_path("mechanism_zoo"), context_length=4096, batch=1)
    _save_sequence(seq, out / "zoo_sequence.json")
    results = tr.check_zoo(seq)
    steps = []
    for r in results:
        alt = (r.step.expected.index(r.observed) if r.observed in r.step.expected else None)
        steps.append({"layer": r.step.layer, "name": r.step.name, "ok": r.ok,
                      "observed": r.observed, "composites": r.composites,
                      "alternatives": len(r.step.expected), "alternative_held": alt})

    path2 = tr.record_prefill_trace(lambda: tr.run_two_stream_zoo(), out / "zoo2_trace.json",
                                    warmup=1, cuda=True)
    seq2 = ops.parse_trace(path2, code_path("two_stream_zoo"), context_length=4096, batch=1)
    _save_sequence(seq2, out / "zoo2_sequence.json")
    sc = ops.stream_check(seq2)

    findings = {"single_stream": {"steps": steps, **_sequence_summary(seq)},
                "two_stream": {"flagged": not sc.single_stream, "message": sc.message,
                               "streams": list(sc.streams),
                               "cross_stream_waits": sc.cross_stream_waits},
                "ignored_calls": list(seq.ignored_calls)}
    bad = [s["name"] for s in steps if not s["ok"]]
    if bad or sc.single_stream:
        why = []
        if bad:
            why.append(f"zoo steps not as expected: {bad}")
        if sc.single_stream:
            why.append("the two-stream zoo was not flagged")
        raise StepFailed("; ".join(why), findings)
    return findings


# --- T2.5 and T2.6 -------------------------------------------------------------

def step_link(args) -> dict:
    """`measure_link.py`: every field of the link row, one provenance-stamped
    row (T2.5's Done when). Dry run: its fake device."""
    out = Path(args.out)
    cmd = _py("measure_link.py", "--out", str(out))
    if args.dry_run:
        cmd += ["--dry-run", "--sync-reps", "20", "--copy-reps", "3"]
    if _run(cmd):
        raise StepFailed("measure_link.py failed")
    import pandas as pd
    row = pd.read_parquet(out / "link.parquet").iloc[0].to_dict()
    empty = sorted(k for k, v in row.items()
                   if (k.startswith("s_fixed_") or k.startswith("bw_")) and v is None)
    findings = {"dry_run_row": bool(row.get("dry_run")), "empty_fields": empty,
                "pcie_gen": row.get("pcie_gen_current_under_copy"),
                "gpu_numa_node": row.get("gpu_numa_node"),
                "affinity_on_gpu_node": row.get("affinity_on_gpu_node")}
    if empty and not args.dry_run:
        raise StepFailed(f"link row has empty fields: {empty}", findings)
    return findings


def step_launch_queue(args) -> dict:
    """`measure_launch_queue.py`: Q for the null kernel and the one-element
    add; the model's Q is the null kernel's (Appendix A, 25 Oct 2026), so a
    null-kernel Q the trials do not agree on fails the step."""
    out = Path(args.out)
    cmd = _py("measure_launch_queue.py", "--out", str(out))
    if args.dry_run:
        cmd += ["--dry-run", "--blocker-s", "0.05"]
    if _run(cmd):
        raise StepFailed("measure_launch_queue.py failed")
    import pandas as pd
    row = pd.read_parquet(out / "launch_queue.parquet").iloc[0].to_dict()
    findings = {k: row[k] for k in row if k.endswith("_q") or k.endswith("_q_lower_bound")}
    if row.get("sleep0_q") is None:         # the null kernel, torch.cuda._sleep(0)
        raise StepFailed("the null kernel's Q was not agreed across trials", findings)
    return findings


# --- T2.7 ------------------------------------------------------------------------

def step_nsys(args) -> dict:
    """T2.7. First each arm's Kineto trace through `check_trace.py kineto`
    (copy-kernel concurrency, managed-memory use). Then the separate traced
    pass, never a timing run: is nsys on the image (`check_trace.py
    nsys-available`); if not, the step fails here, saying so, and the fix
    (installing it in this session) is recorded by rerunning from this step.
    The positive control `um_fault_control.py` and each arm (`arm-pass`, its
    own process and export) run under the same nsys options, and
    `check_trace.py nsys-faults` reads each export against the control. "No
    page faults" holds only for an arm whose export shows none, and only if
    the control faulted."""
    _need_cuda(args, "the T2.7 checks")
    from attnbench.analysis import trace_checks as tc
    from attnbench.components import trace_recording as tr

    out = Path(args.out)
    arms = _arms(args)
    findings: dict = {"kineto": {}, "nsys": {}}

    h = _harness(args)
    for name in arms:
        arm = h.ready(name, args.probe_short)
        trace = tr.record_prefill_trace(arm.run_prefill, out / f"kineto_{name}.json",
                                        layers=arm.layers, warmup=1, cuda=True)
        rc = _run(_py("check_trace.py", "kineto", str(trace), "--out", str(out / f"kineto_{name}")))
        findings["kineto"][name] = {"rc": rc, **{k: v for k, v in tc.check_kineto(trace).items()
                                               if k in ("copy_kernel_concurrency", "managed_memory_use",
                                                        "cross_stream_overlaps", "claim")}}
    h.unwrap()
    del h

    nsys = tc.nsys_available()
    findings["nsys_binary"] = nsys
    if nsys is None:
        raise StepFailed("nsys is not on the image. Install it in this session, record "
                         "the install, and rerun from this step (--from nsys)", findings)
    findings["nsys_version"] = subprocess.run([nsys, "--version"], capture_output=True,
                                              text=True).stdout.strip()
    flags = list(tc.NSYS_PROFILE_FLAGS)

    def traced(base: Path, cmd: list[str]) -> Path:
        if _run([nsys, "profile", *flags, "--force-overwrite=true", "-o", str(base), *cmd]):
            raise StepFailed(f"nsys profile of {cmd[1:3]} failed", findings)
        sqlite = base.with_suffix(".sqlite")
        if _run([nsys, "export", "--type=sqlite", "--force-overwrite=true",
                 f"--output={sqlite}", str(base.with_suffix(".nsys-rep"))]):
            raise StepFailed(f"nsys export of {base} failed", findings)
        return sqlite

    control = traced(out / "um_control", _py("um_fault_control.py"))
    ctrl_counts = tc.page_fault_counts(control)
    findings["control_fault_tables"] = ctrl_counts
    if not sum(ctrl_counts.values()):
        raise StepFailed("the positive control recorded no page faults: fault collection "
                         "is not shown to be on, so no arm can claim none", findings)
    bad = []
    for name in arms:
        export = traced(out / f"nsys_{name}",
                        _py("t227_steps.py", "arm-pass", "--out", str(out / f"arm_pass_{name}"),
                            "--arms", name, "--length", str(args.probe_short)))
        rc = _run(_py("check_trace.py", "nsys-faults", str(export), "--control", str(control),
                      "--arms", name, "--out", str(out / f"nsys_{name}_check")))
        r = tc.check_page_faults(export, control, arms=[name])
        findings["nsys"][name] = {"rc": rc, "page_faults": r["page_faults"], "claim": r["claim"]}
        if r["page_faults"] != 0:
            bad.append(name)
    kineto_bad = [a for a, r in findings["kineto"].items() if r["rc"]]
    if bad or kineto_bad:
        raise StepFailed(f"page faults in {bad}; Kineto flags in {kineto_bad}", findings)
    return findings


def cmd_arm_pass(args) -> int:
    """One untimed prefill of one arm: the process nsys traces."""
    h = _harness(args)
    (name,) = [a for a in args.arms.split(",") if a]
    arm = h.prepare(name, args.length)
    arm.run_prefill()
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"arm-pass {name} at {arm.length} (budget {args.length}): one prefill of {arm.prompt_id}")
    return 0


# --- T2.4 ------------------------------------------------------------------------

def step_replay(args) -> dict:
    """T2.4 end to end on one arm and cell (`--replay-arm`, `--length`): record
    the prefill (trace, sequence, masks of the traced run, tape); capture it
    as a CUDA graph from the tape and the bank; check the live masks against
    the bank (bitwise, or the step fails); time the graph; read its nodes
    under the profiler and match them to the sequence; the replay-versus-
    trace report; the isolated replay and the cache effect. A dry run: the
    bound is a lock value and the check of record is P1's in T3.1, so an
    incomplete match is recorded, not failed."""
    _need_cuda(args, "the CUDA-graph replay")
    import torch
    from attnbench.components import op_sequence as ops
    from attnbench.components import recorder, replay

    out = Path(args.out)
    h = _harness(args)
    arm = h.ready(args.replay_arm, args.length)
    rec = recorder.record_arm_prefill(arm.run_prefill, out / "recording",
                                      code_path=arm.code_path, context_length=arm.length,
                                      batch=1, prompt_id=arm.prompt_id, layers=arm.layers)
    seq = rec.sequence
    findings: dict = {"arm": arm.name, "length": arm.length, "budget": args.length,
                      "prompt_id": arm.prompt_id,
                      "sequence": _sequence_summary(seq),
                      "tape_entries": len(rec.tape), "tape_substitutions": rec.tape.substitutions,
                      "mask_calls": len(rec.masks.masks) if rec.masks else 0}
    cap = replay.capture(arm.run_prefill, rec.tape, device="cuda", masks=rec.masks)
    agreement = replay.mask_agreement(cap)
    findings["mask_agreement"] = agreement
    timing = replay.time_replay(cap, args.replay_reps)
    nodes = replay.replay_nodes(replay.profile_replay(cap, out / "replay_trace.json"))
    m = replay.match(seq, nodes)
    trace_s = ops.device_durations(rec.directory / "trace.json", seq)
    rvt = replay.replay_vs_trace(seq, trace_s, m.device_s, m)
    torch.cuda.synchronize()
    iso_path = replay.isolated_replay(arm.run_prefill, rec.tape, out / "isolated_trace.json",
                                      device="cuda", masks=rec.masks)
    iso = replay.match(seq, replay.replay_nodes(iso_path, exclude=replay.FLUSH))
    ce = replay.cache_effect(seq, m.device_s, iso.device_s)
    original, pricing, pricing_error = None, None, None
    if not m.complete:
        # Substituted positions, priced from the original operators alone.
        torch.cuda.synchronize()
        orig_path = replay.isolated_replay(arm.run_prefill, rec.tape,
                                           out / "isolated_original_trace.json",
                                           device="cuda", masks=rec.masks, substitute=False)
        original = replay.match(seq, replay.replay_nodes(orig_path, exclude=replay.FLUSH))
    try:
        pricing = replay.price_positions(seq, m, original)
    except replay.ReplayError as e:
        pricing_error = str(e)
    record = replay.to_record(seq=seq, prompt_id=arm.prompt_id, m=m, timing=timing,
                              agreement=agreement, tape=rec.tape, masks=rec.masks,
                              isolated=iso.device_s, pricing=pricing)
    _write_json(out / "replay_record.json", record)
    _write_json(out / "replay_vs_trace.json", rvt)
    _write_json(out / "cache_effect.json", ce)
    findings.update({"graph_timing": timing, "complete": m.complete,
                     "unmatched_positions": len(m.unmatched_positions),
                     "families_unmatched": m.families_unmatched,
                     "extra_nodes": len(m.extra_nodes),
                     "profiled_over_unprofiled": record["profiled_over_unprofiled"],
                     "replay_vs_trace_families": rvt["families"],
                     "isolated_complete": iso.complete,
                     "isolated_over_in_order": ce["isolated_over_in_order"],
                     "original_complete": None if original is None else original.complete,
                     "priced": pricing is not None,
                     "cost_sources": None if pricing is None else pricing.counts(),
                     "pricing_error": pricing_error})
    if agreement is not None and not agreement.get("identical"):
        raise StepFailed("the replay's live masks differ from the bank on the code "
                         "path that recorded them", findings)
    return findings


# --- T2.18 -------------------------------------------------------------------------

def step_probe(args) -> dict:
    """T2.18 for each arm: recorded at its probe points (traces deleted,
    durations discarded; `structural_probe.probe_plan`, a chunked arm
    checked at the third), extended by the arm's chunking rule to the length
    the full recording ran at (`--length` is its budget), and checked against
    that recording (the test of record). "Different" is a verdict, not a
    failure (researcher's ruling, 31 Oct 2026): library kernel selection
    really does change the ordered operations, so where the extension or its
    check says different, the arm's structure is its full recording at that
    length (T2.18's fallback; durations discarded, nothing extrapolated). An
    arm fails only if the probe did not complete with a verdict and its
    evidence."""
    _need_cuda(args, "the structural probe (the parser needs CUDA activity)")
    from attnbench.components import op_sequence as ops, recorder, structural_probe as sp

    out = Path(args.out)
    h = _harness(args)
    findings: dict = {}
    failed = []
    for name in _arms(args):
        spec_dir = out / name

        def record(L, d, name=name):
            a = h.ready(name, L)
            return recorder.record_arm_prefill(
                a.run_prefill, d, code_path=a.code_path, context_length=a.length, batch=1,
                prompt_id=a.prompt_id, layers=a.layers).sequence

        try:
            from attnbench.components.shakedown_arms import ARMS
            rule = sp.xattention_rule() if ARMS[name].xattention else sp.NO_CHUNKING
            plan = (sp.probe_plan(rule) if rule.chunked else
                    sp.ProbePlan((args.probe_short, args.probe_long), False, rule.name))
            lengths = plan.lengths
            seqs = sp.probe(record, spec_dir, lengths=lengths)
            full = record(args.length, spec_dir / f"full_{args.length}")
            target = full.context_length
            ext = (sp.extend_checked(seqs, rule, target, full_third=plan.full_third)
                   if rule.chunked else sp.extend(seqs, rule, target))
            r = {"rule": rule.name, "probe_budgets": list(lengths),
                 "probe_plan": plan.reason, "full_third": plan.full_third,
                 "probe_lengths": sorted(seqs), "chunks": ext.chunks,
                 "checked_at": ext.checked_at, "extension": ext.verdict,
                 "reason": ext.reason, "payload_exponents": ext.payload_exponents,
                 "full": _sequence_summary(full)}
            if ext.extended:
                c = sp.check_against_full(ext, full)
                r.update({"check": c.verdict, "first_difference": c.first_difference,
                          "check_reason": c.reason})
                _save_sequence(ext.sequence, spec_dir / f"extended_{args.length}.json")
            same = ext.extended and r.get("check") == "same sequence"
            r["structure"] = "extended" if same else f"recorded at full length ({target})"
            r["evidence"] = r.get("check_reason") if ext.extended else ext.reason
            verdicts = (ops.SAME, ops.DIFFERENT)
            ok = ext.verdict in verdicts and (not ext.extended or r.get("check") in verdicts)
        except Exception as e:                       # one arm's failure, recorded
            traceback.print_exc()
            r, ok = {"error": f"{type(e).__name__}: {e}"}, False
        r["passed"] = ok
        findings[name] = r
        if not ok:
            failed.append(name)
        print(f"probe {name}: {'passed' if ok else 'FAILED'} "
              f"{r.get('structure') or ''} {r.get('evidence') or r.get('error')}",
              flush=True)
    if failed:
        raise StepFailed(f"structural probe failed for {failed}", findings)
    return findings


# --- T2.17 -------------------------------------------------------------------------

def step_assert_gate(args) -> dict:
    """T2.17 on the real package. The installed x-attention checkout is the
    pinned commit with a clean tree; both arms are imported; the per-session
    check (`assert_strip.session_check`: the process unoptimised, the hook's
    log matching the `xattn` package, only asserts and `__debug__` blocks
    removed, the shipped arm still carrying its asserts); then the bitwise
    mask and output gate (`AssertGate`) over `--gate-prompts` real RULER
    prompts at `--length`."""
    _need_cuda(args, "the T2.17 gate (XAttention's estimator moves its inputs to cuda)")
    from attnbench import assert_strip
    from attnbench.backends.xattention import XATTN_COMMIT, XAttentionBackend, installed_checkout
    from attnbench.backends.xattention_noassert import AssertGate, XAttentionNoAssertBackend

    findings: dict = {}
    head, dirty = installed_checkout()
    findings["xattn_checkout"] = {"head": head, "dirty": dirty, "pinned": XATTN_COMMIT}
    if head != XATTN_COMMIT or dirty:
        raise StepFailed(f"the installed x-attention checkout is {head} (dirty {dirty}), "
                         f"not the pinned {XATTN_COMMIT}", findings)
    import xattn.src.Xattention  # noqa: F401  (the shipped arm)
    XAttentionNoAssertBackend._import_check()
    sc = assert_strip.session_check()
    findings["session_check"] = {"optimize_flag": sc.optimize_flag,
                                 "stripped_modules": sc.stripped_modules,
                                 "asserts_removed": sc.asserts_removed,
                                 "shipped_modules": sc.shipped_modules,
                                 "shipped_asserts": sc.shipped_asserts,
                                 "functions_checked": sc.functions_checked}

    h = _harness(args)
    from attnbench.backends.xattention import ThresholdTable
    from attnbench.accuracy.t4_pilot import XATTN_CALIBRATION_TABLES, XATTN_STRIDE
    from attnbench.components.shakedown_arms import REPO, XATTN_TABLE
    table = ThresholdTable.load(REPO / XATTN_CALIBRATION_TABLES[XATTN_TABLE])
    gate = AssertGate(XAttentionBackend(threshold=table, stride=XATTN_STRIDE),
                      XAttentionNoAssertBackend(threshold=table, stride=XATTN_STRIDE))
    prompts = []
    for i in range(args.gate_prompts):
        arm = h.prepare("xattention", args.length, index=i, backend=gate)
        before = len(gate.records)
        arm.run_prefill()
        prompts.append({"prompt_id": arm.prompt_id, "length": arm.length,
                        "layers": len(gate.records) - before})
    fails = gate.failures()
    findings.update({"table": table.label, "prompts": prompts, "layer_records": len(gate.records),
                     "failures": [{"layer": r.layer, "mask_equal": r.mask_equal,
                                   "output_equal": r.output_equal} for r in fails],
                     "triton_effective": gate.shipped.triton_effective,
                     "passed": gate.passed})
    if not gate.passed:
        raise StepFailed(f"the bitwise gate failed on {len(fails)} layer records", findings)
    return findings


# --- T2.21 ---------------------------------------------------------------------------

class _InputCapture:
    """A backend that records each layer's (q, k, v, mask, cfg) and runs the
    arm's own backend, so the report sees the inputs the real prefill gives."""

    def __init__(self, inner):
        self.inner = inner
        self.layer_idx = None
        self.layers: list = []

    def set_layer(self, i):
        self.layer_idx = i
        if hasattr(self.inner, "set_layer"):
            self.inner.set_layer(i)

    def reset_density(self):
        if hasattr(self.inner, "reset_density"):
            self.inner.reset_density()

    def forward(self, q, k, v, cfg, mask=None):
        inputs = (q, k, v) if mask is None else (q, k, v, mask)
        self.layers.append((self.layer_idx if self.layer_idx is not None else len(self.layers),
                            inputs, cfg))
        return self.inner.forward(q, k, v, cfg, mask=mask)


def step_graph_breaks(args) -> dict:
    """T2.21's graph-break report: each arm under each setting, each pair in a
    process of its own (Dynamo does not guard on its own settings), on the
    inputs a real prefill hands each layer. The mean-pool estimator is built
    in `accuracy/model.py`, outside any backend, so the meanpool and oracle
    reports cover the kernel call, not the estimator. Records whether the two
    settings split identically."""
    from attnbench.backends.compiled import SETTINGS
    out = Path(args.out)
    findings: dict = {}
    failed = []
    for name in _arms(args):
        reports = {}
        for setting in SETTINGS:
            d = out / f"{name}@{setting}"
            rc = _run(_py("t227_steps.py", "graph-breaks-one", "--out", str(d), "--arms", name,
                          "--setting", setting, "--length", str(args.gb_length),
                          "--gb-layers", args.gb_layers, *(["--dry-run"] if args.dry_run else [])))
            p = d / "graph_breaks.json"
            if rc or not p.exists():
                failed.append(f"{name}@{setting}")
                continue
            reports[setting] = json.loads(p.read_text())["layers"]
        r = {s: {"graphs": sum(l["graphs"] for l in rep),
                 "breaks": sum(len(l["breaks"]) for l in rep),
                 "reasons": sorted({b["reason"] for l in rep for b in l["breaks"]})}
             for s, rep in reports.items()}
        if len(reports) == len(SETTINGS):
            a, b = (reports[s] for s in SETTINGS)
            r["settings_identical"] = a == b
        findings[name] = r
    if failed:
        raise StepFailed(f"graph-break report failed for {failed}", findings)
    return findings


def cmd_graph_breaks_one(args) -> int:
    from attnbench.backends import compiled
    out = Path(args.out)
    h = _harness(args)
    (name,) = [a for a in args.arms.split(",") if a]
    from attnbench.components.shakedown_arms import ARMS
    inner = h._backend(ARMS[name])
    cap = _InputCapture(inner)
    arm = h.prepare(name, args.gb_length, backend=cap)
    arm.run_prefill()
    layers = cap.layers
    if args.gb_layers != "all":
        keep = {int(x) for x in args.gb_layers.split(",")}
        layers = [l for l in layers if l[0] in keep]
    t0 = time.time()
    report = compiled.graph_break_report(inner, layers, args.setting)
    _write_json(out / "graph_breaks.json", {
        "arm": name, "setting": args.setting, "length": arm.length, "budget": args.gb_length,
        "variant": compiled.variant_name(name, args.setting), "seconds": time.time() - t0,
        "layers": [{"layer": l.layer, "graphs": l.graphs,
                    "breaks": [{"reason": b.reason, "where": list(b.where)} for b in l.breaks]}
                   for l in report]})
    print(f"{name}@{args.setting}: {sum(l.graphs for l in report)} graphs, "
          f"{sum(len(l.breaks) for l in report)} breaks over {len(report)} layers")
    return 0


# --- the null kernel -------------------------------------------------------------------

def step_null_kernel(args) -> dict:
    """The null kernel (`torch.cuda._sleep(0)`, host_speed's) under the
    profiler: one burst of `--null-burst` launches (below Q) inside a region,
    with the device synchronised outside it, after an untraced burst to warm
    the path. The trace must hold exactly that many launches and kernels;
    the host time of each launch call is read from the trace, beside the
    unprofiled burst timed by `host_speed.null_kernel_burst`."""
    _need_cuda(args, "the null kernel")
    import statistics
    import torch
    from torch.autograd.profiler import record_function
    from torch.profiler import profile
    from attnbench.components import host_speed, op_sequence as ops
    from attnbench.components.trace_recording import profiler_kwargs

    out = Path(args.out)
    launch, sync = host_speed.cuda_null_kernel()
    unprofiled = host_speed.null_kernel_burst(launch, sync, args.null_burst)
    sync()
    with profile(**profiler_kwargs(True)) as prof:
        with record_function(ops.REGION):
            for _ in range(args.null_burst):
                launch()
        sync()
    path = out / "null_kernel_trace.json"
    out.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(path))
    events = ops.load_trace(path)["traceEvents"]
    region = [e for e in events if e.get("name") == ops.REGION and e.get("cat") == "user_annotation"]
    if len(region) != 1:
        raise StepFailed(f"{len(region)} regions in the null-kernel trace")
    r0, r1 = region[0]["ts"], region[0]["ts"] + region[0]["dur"]
    launches = [e for e in events if e.get("cat") == "cuda_runtime"
                and "LaunchKernel" in e.get("name", "") and r0 <= e["ts"] <= r1]
    corr = {e.get("args", {}).get("correlation") for e in launches}
    kernels = [e for e in events if e.get("cat") == "kernel"
               and e.get("args", {}).get("correlation") in corr]
    host_us = [e["dur"] for e in launches]
    findings = {"burst": args.null_burst, "launches_in_trace": len(launches),
                "kernels_in_trace": len(kernels),
                "profiled_launch_us_median": statistics.median(host_us) if host_us else None,
                "unprofiled_launch_us_median": statistics.median(unprofiled) * 1e6,
                "gpu": torch.cuda.get_device_name(0)}
    _write_json(out / "null_kernel.json", findings)
    if len(launches) != args.null_burst or len(kernels) != args.null_burst:
        raise StepFailed("the trace does not hold one launch and one kernel per null "
                         "kernel issued", findings)
    return findings


# --- Figure 1 ---------------------------------------------------------------------------

MASK_BUILD = {"oracle": ("attnbench.masks", "mask_for"),
              "meanpool": ("attnbench.masks", "importance_block_mask_device")}


@contextlib.contextmanager
def _host_compute_around_mask_build(arm):
    """Wrap the arm's mask build in `host_compute()` for the recording only:
    the module attribute the forward looks up (oracle, meanpool), or the
    backend's own `estimate` (XAttention). Nothing is edited (rule 8)."""
    import importlib
    from attnbench.components.trace_recording import host_compute
    if arm.spec.xattention:
        orig = arm.backend.estimate

        def wrapped(*a, **k):
            with host_compute():
                return orig(*a, **k)
        arm.backend.estimate = wrapped
        try:
            yield
        finally:
            del arm.backend.estimate
        return
    mod_name, attr = MASK_BUILD[arm.name]
    mod = importlib.import_module(mod_name)
    orig = getattr(mod, attr)

    def wrapped(*a, **k):
        with host_compute():
            return orig(*a, **k)
    setattr(mod, attr, wrapped)
    try:
        yield
    finally:
        setattr(mod, attr, orig)


def _crop_layer(trace_path: Path, layer: int, dest: Path) -> dict:
    """One layer of a trace: host events inside its annotation, the device
    activity those calls launched, and the trace's metadata."""
    from attnbench.components import op_sequence as ops
    doc = ops.load_trace(trace_path)
    events = doc["traceEvents"]
    name = ops.layer_annotation(layer)
    ann = [e for e in events if e.get("name") == name and e.get("cat") == "user_annotation"]
    if len(ann) != 1:
        raise StepFailed(f"{len(ann)} annotations {name!r} in {trace_path.name}")
    a = ann[0]
    t0, t1 = a["ts"], a["ts"] + a["dur"]
    host = [e for e in events if e.get("ph") == "X" and e.get("pid") == a["pid"]
            and t0 <= e.get("ts", -1) and e.get("ts", 0) + e.get("dur", 0) <= t1]
    corr = {e.get("args", {}).get("correlation") for e in host} - {None}
    dev = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
           and e.get("args", {}).get("correlation") in corr]
    meta = [e for e in events if e.get("ph") == "M"]
    _write_json(dest, {**{k: v for k, v in doc.items() if k != "traceEvents"},
                       "traceEvents": meta + host + dev})
    return {"host_events": len(host), "device_events": len(dev),
            "layer_us": a["dur"],
            "host_compute_us": sum(e["dur"] for e in host if e.get("name") == ops.HOST_COMPUTE)}


def step_figure1(args) -> dict:
    """Figure 1's trace pair: the same arm (`--fig1-arm`) and length
    (`--fig1-length`, 16384 tokens), the same prompt, recorded twice under T2.3's recording rules with `host_compute()`
    around the mask build, and one layer (`--fig1-layer`) cut out of each.
    The pair must be the same sequence, and the layer must hold a
    `host_compute` marker in both. This is the L4 half only: the A100
    counterpart is not part of this session."""
    _need_cuda(args, "the Figure 1 traces")
    from attnbench.components import op_sequence as ops
    from attnbench.components import trace_recording as tr

    out = Path(args.out)
    h = _harness(args)
    arm = h.ready(args.fig1_arm, args.fig1_length)
    seqs, findings = {}, {"arm": arm.name, "length": arm.length, "budget": args.fig1_length,
                          "prompt_id": arm.prompt_id, "layer": args.fig1_layer}
    with _host_compute_around_mask_build(arm):
        for k in ("a", "b"):
            p = tr.record_prefill_trace(arm.run_prefill, out / f"fig1_{k}_trace.json",
                                        layers=arm.layers, warmup=1, cuda=True)
            seqs[k] = ops.parse_trace(p, arm.code_path, context_length=arm.length,
                                      batch=1)
            _save_sequence(seqs[k], out / f"fig1_{k}_sequence.json")
            markers = [o for o in seqs[k].ops
                       if o.layer == args.fig1_layer and o.kind is ops.SeqKind.HOST_COMPUTE]
            findings[k] = {"host_compute_markers_in_layer": len(markers),
                           **_crop_layer(p, args.fig1_layer,
                                         out / f"fig1_{k}_layer{args.fig1_layer}.json")}
    c = ops.compare(seqs["a"], seqs["b"])
    findings.update({"pair": c.verdict, "first_difference": c.first_difference,
                     "reason": c.reason})
    if c.verdict != ops.SAME or not all(findings[k]["host_compute_markers_in_layer"]
                                        for k in ("a", "b")):
        raise StepFailed("the Figure 1 pair is not the same sequence, or its layer has "
                         "no host_compute marker", findings)
    return findings


# --- entry point ----------------------------------------------------------------------------

STEP_FUNCS = {
    "t21_t220": step_t21_t220, "zoo": step_zoo, "link": step_link,
    "launch_queue": step_launch_queue, "nsys": step_nsys, "replay": step_replay,
    "probe": step_probe, "assert_gate": step_assert_gate,
    "graph_breaks": step_graph_breaks, "null_kernel": step_null_kernel,
    "figure1": step_figure1,
}
if tuple(STEP_FUNCS) != STEPS:     # not `assert`: T2.17 keeps the harness free of them
    raise RuntimeError("STEP_FUNCS and STEPS disagree")


def run_step(name: str, args, funcs: dict | None = None) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        status, reason, findings = PASSED, "", (funcs or STEP_FUNCS)[name](args)
    except Skip as e:
        status, reason, findings = SKIPPED, str(e), {}
    except StepFailed as e:
        status, reason, findings = FAILED, str(e), e.findings
    except Exception as e:
        traceback.print_exc()
        status, reason, findings = FAILED, f"{type(e).__name__}: {e}", {}
    _write_json(out / "result.json", {
        "step": name, "status": status, "reason": reason, "dry_run": args.dry_run,
        "seconds": round(time.time() - started, 1), "findings": findings})
    print(f"[{name}] {status.upper()}{': ' + reason if reason else ''}", flush=True)
    return EXIT[status]


def parser() -> argparse.ArgumentParser:
    from attnbench.components.shakedown_arms import ARMS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=STEPS + ("arm-pass", "graph-breaks-one"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--length", type=int, default=8192,
                    help="the cell: T2.4's, T2.17's and T2.18's target")
    ap.add_argument("--probe-short", type=int, default=2048)
    ap.add_argument("--probe-long", type=int, default=4096)
    ap.add_argument("--replay-arm", default="meanpool", choices=sorted(ARMS))
    ap.add_argument("--replay-reps", type=int, default=20)
    ap.add_argument("--gate-prompts", type=int, default=5)
    ap.add_argument("--gb-length", type=int, default=8192)
    ap.add_argument("--gb-layers", default="all", help="'all' or comma-separated indices")
    ap.add_argument("--setting", default="default")
    ap.add_argument("--null-burst", type=int, default=64)
    ap.add_argument("--fig1-arm", default="oracle", choices=("oracle", "meanpool", "xattention"))
    ap.add_argument("--fig1-layer", type=int, default=14)
    ap.add_argument("--fig1-length", type=int, default=16384,
                    help="Figure 1's pair, on the L4 (the A100 counterpart is not this session's)")
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.step == "arm-pass":
        return cmd_arm_pass(args)
    if args.step == "graph-breaks-one":
        return cmd_graph_breaks_one(args)
    return run_step(args.step, args)


if __name__ == "__main__":
    raise SystemExit(main())
