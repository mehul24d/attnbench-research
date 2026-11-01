#!/usr/bin/env python3
"""The XAttention crash diagnostic (researcher's ruling, 2 Nov 2026): five
cases, each in a process of its own, then compute-sanitizer on the smallest
case that reproduces.

T3.1's first P1 session (1 Nov 2026, harness at `c38e196`, x-attention at
`e379887`; `results/t31_p1/`) stopped in step 4. Dense, the oracle and
mean-pool were recorded at every grid length, XAttention's three probes
(20480, 28672, 32768) were recorded, and its full recording at 2048 raised
"CUDA error: an illegal memory access" at x-attention `utils.py:145`
(`find_blocks_chunked`). Observed under our harness at `c38e196`; the cause
is under investigation. The error is asynchronous, so the kernel that
faulted may be an earlier one.

What the traceback and the code already say (static review, 2 Nov 2026):
  - The failing prefill was the recorder's untraced warm-up
    (`trace_recording.record_prefill_trace`, the loop before the profiler
    opens), not the profiled one. `Harness.ready` had run one 2048 prefill
    of the same arm and synchronised without error just before it.
  - The frame is the shipped `xattn` package (`backends/xattention.py`
    `estimate`), asserts intact. The assert-strip hook (T2.17) serves only
    its alias, and only the assert-free arm, which runs later, installs it.
  - Every recording builds a new backend (`Harness.prepare`), so no adapter
    state crosses recordings; on the L4, x-attention's Triton path is off
    (`triton_for_device`), so the estimator is PyTorch operators and the one
    non-PyTorch kernel in the layer is `block_sparse_attn_func`.

The cases (each a fresh process; the order is the ruling's):
  C  T3.1's step-4 order: dense, the oracle and mean-pool (probes, then the
     grid lengths), XAttention (its probes, then the grid lengths), then the
     assert-free arm; with CUDA_LAUNCH_BLOCKING=1, which reports the faulting
     kernel at its own launch.
  B  XAttention's three probes, then its grid lengths from 2048.
  A  XAttention alone at 2048, recorded three times.
  D  C's order with no profiler: each recording runs the same prefills, with
     the same synchronises and the same mask capture, untraced.
  E  C's order without the assert-free arm, the assert-strip hook held off:
     any call to `assert_strip.install` fails the case.
Every case writes `events.jsonl` (each recording's start and end, with the
XAttention layer and length last entered) and `case.json`. A case stops at
its first exception: after an illegal address the CUDA context is lost.

compute-sanitizer (memcheck, with PyTorch's caching allocator off so a stray
access is checked against the tensor it belongs to) runs once, on the
smallest of A and B that reproduced, as soon as that is known, under a
10-minute limit. When neither reproduced, it runs on A only if
`--sanitize-a-anyway` was given (an out-of-bounds access may land in mapped
memory and not fault). C, D and E are too long for the limit.

On the L4, from the repo root, once (synced to GCS per case):
    python3 scripts/t31_xa_diag.py session --deadline-min 48
One case on its own:
    python3 scripts/t31_xa_diag.py case A --out results/t31_xa_diag_x/A
The CPU dry run, behind the stubbed GPU boundary (`attnbench.cpu_boundary`),
which must pass before launch:
    python3 scripts/t31_xa_diag.py session --cpu-boundary --session /tmp/xa_dry
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

CASES = ("C", "B", "A", "D", "E")
SMALL = ("A", "B")                      # sanitizer candidates, smallest first
SANITIZER_LIMIT_S = 600
A_REPEATS = 3
XA_ORDER = ("xattention",)
C_ARMS = ("dense", "oracle", "meanpool", "xattention", "xattention_noassert")
E_ARMS = ("dense", "oracle", "meanpool", "xattention")
CASE_ENV = {"C": {"CUDA_LAUNCH_BLOCKING": "1"}}
SANITIZER_ENV = {"PYTORCH_NO_CUDA_MEMORY_CACHING": "1"}
REPRODUCED, CLEAN, ERROR, CUT = "reproduced", "clean", "other error", "cut by the deadline"
ILLEGAL = "illegal memory access"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))


# --- one case, in this process ------------------------------------------------------

class Events:
    """An append-only log, flushed and fsynced per line, so a process that
    dies still leaves its last recording named."""

    def __init__(self, path: Path):
        self.path = path
        self.last_layer = None          # (arm, layer, length) last entered
        self.arm = None

    def write(self, **row) -> None:
        row = {"t": _now(), **row}
        with self.path.open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def _watch_layers(ev: Events) -> None:
    """Note the layer and length each XAttention forward enters: a Python
    attribute write, no synchronise, no device work."""
    from attnbench.backends.xattention import XAttentionBackend
    inner = XAttentionBackend.forward_and_mask

    def forward_and_mask(self, q, k, v, cfg, mask=None):
        ev.last_layer = (ev.arm, self.layer_idx, int(q.shape[2]))
        return inner(self, q, k, v, cfg, mask=mask)

    XAttentionBackend.forward_and_mask = forward_and_mask


def _hold_hook_off() -> None:
    """E: the assert-strip hook is off, and stays off. Behind the CPU
    boundary, which installs it to import both packages under the stubs, it
    is taken off first; on the L4 nothing has installed it, or E refuses."""
    from attnbench import assert_strip, cpu_boundary
    if hook_installed():
        if not cpu_boundary.active():
            raise RuntimeError("the assert-strip hook is already installed")
        assert_strip.uninstall()

    def refuse(*_a, **_kw):
        raise RuntimeError("case E holds the assert-strip hook off; something installed it")

    assert_strip.install = refuse


def hook_installed() -> bool:
    from attnbench import assert_strip
    return assert_strip.installed() is not None or assert_strip.ALIAS in sys.modules


def _sync(cuda: bool) -> None:
    if cuda:
        import torch
        torch.cuda.synchronize()


def record_unprofiled(a, d: Path, *, cuda: bool, warmup: int = 1) -> None:
    """D: `recorder.record_arm_prefill`'s prefills without the profiler. The
    same count (warm-up, the profiler's warm-up step, the recorded prefill),
    the same synchronises, the traced capture keeping references and cleared
    at the recorded call, then the same untraced pass under the tape."""
    from attnbench.components import mask_bank, recorder
    from attnbench.components.trace_recording import record_prefill_trace
    d.mkdir(parents=True, exist_ok=False)
    profiled = warmup + getattr(record_prefill_trace, "profiler_warmup", 0) + 1
    traced = mask_bank.MaskCapture(optional=True)
    with traced:
        for i in range(1, profiled + 1):
            if i == profiled:
                traced.calls.clear()
            a.run_prefill()
            if i >= warmup:             # after the warm-up loop, and after each step
                _sync(cuda)
    recorder._untraced_pass(a.run_prefill, "cuda" if cuda else "cpu")


def case_plan(case: str, cfg: dict) -> list[tuple[str, str, int]]:
    """(arm, kind, length) in the order the case records them; kind is
    "probe" or "full". C's is step 4's (`t31_steps.step_probe`)."""
    from attnbench.components import structural_probe as sp
    from attnbench.components.shakedown_arms import ARMS

    def arm_plan(name, lengths):
        rule = sp.xattention_rule() if ARMS[name].xattention else sp.NO_CHUNKING
        return ([(name, "probe", int(L)) for L in sp.probe_plan(rule).lengths]
                + [(name, "full", int(L)) for L in lengths])

    grid = [int(L) for L in cfg["lengths"]]
    if case == "A":
        return [("xattention", "full", min(grid))] * A_REPEATS
    if case == "B":
        return arm_plan("xattention", grid)
    arms = E_ARMS if case == "E" else C_ARMS
    return [row for name in arms for row in arm_plan(name, grid)]


def run_case(case: str, out: Path, cfg: dict, *, cuda: bool, harness) -> dict:
    from attnbench.components import recorder
    out.mkdir(parents=True, exist_ok=True)
    ev = Events(out / "events.jsonl")
    _watch_layers(ev)
    if case == "E":
        _hold_hook_off()
    plan = case_plan(case, cfg)
    env = {k: os.environ.get(k) for k in ("CUDA_LAUNCH_BLOCKING",
                                          "PYTORCH_NO_CUDA_MEMORY_CACHING")}
    ev.write(event="start", case=case, n_recordings=len(plan), env=env,
             hook_installed=hook_installed())
    status, failure, done = CLEAN, None, 0
    for i, (name, kind, L) in enumerate(plan):
        d = out / name / f"{kind}_{L}" if case != "A" else out / name / f"full_{L}_{i}"
        ev.arm = name
        ev.write(event="record", i=i, arm=name, kind=kind, length=L,
                 hook_installed=hook_installed())
        try:
            a = harness.ready(name, L)
            if case == "D":
                record_unprofiled(a, d, cuda=cuda)
            else:
                recorder.record_arm_prefill(
                    a.run_prefill, d, code_path=a.code_path, context_length=a.length,
                    batch=1, prompt_id=a.prompt_id, layers=a.layers)
            _sync(cuda)
        except Exception as e:
            tb = traceback.format_exc()
            print(tb, flush=True)
            status = REPRODUCED if ILLEGAL in str(e) else ERROR
            failure = {"i": i, "arm": name, "kind": kind, "length": L,
                       "exception": f"{type(e).__name__}: {e}".splitlines()[0],
                       "last_layer_entered": ev.last_layer, "traceback": tb,
                       "hook_installed": hook_installed()}
            ev.write(event="failed", **{k: v for k, v in failure.items() if k != "traceback"})
            break
        trace = d / "trace.json"
        if trace.exists():               # durations are not this diagnostic's; keep the disk
            trace.unlink()
        done += 1
        ev.write(event="recorded", i=i, arm=name, kind=kind, length=L)
    result = {"case": case, "status": status, "recordings_done": done,
              "recordings_planned": len(plan), "failure": failure, "env": env,
              "cuda": cuda, "ended": _now()}
    _write_json(out / "case.json", result)
    return result


def cmd_case(args) -> int:
    import t31_steps
    cfg = t31_steps.load_config(args.config, dry_run=False, cpu_boundary=args.cpu_boundary)
    from attnbench.components.shakedown_arms import Harness
    out, case, boundary = Path(args.out), args.case, args.cpu_boundary

    def go():
        h = Harness(device="cpu" if boundary else "cuda", dry_run=boundary,
                    score_cache_dir=str(out / "score_cache"))
        return run_case(case, out, cfg, cuda=not boundary, harness=h)

    if boundary:
        from attnbench import cpu_boundary
        with cpu_boundary.installed():
            r = go()
    else:
        r = go()
    print(f"[{args.case}] {r['status'].upper()} after {r['recordings_done']} of "
          f"{r['recordings_planned']} recordings", flush=True)
    return 0 if r["status"] in (CLEAN, REPRODUCED) else 1


# --- the session: one process per case ----------------------------------------------

def case_command(case: str, session: Path, *, cpu_boundary: bool, sync: bool,
                 config=None) -> list[str]:
    cmd = [sys.executable, str(Path(__file__).resolve()), "case", case,
           "--out", str(session / case), *(["--cpu-boundary"] if cpu_boundary else []),
           *(["--config", str(config)] if config else [])]
    if sync:
        cmd = ["bash", str(ROOT / "scripts" / "run_phase.sh"), f"t31xa_{case}", "--", *cmd]
    return cmd


def run_logged(cmd: list[str], log: Path, *, env: dict, timeout: float) -> tuple[int, bool]:
    """(exit code, cut by the timeout), output to `log` and to this terminal."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as fh:
        fh.write(f"# {_now()}\n# env {json.dumps(env)}\n# $ {' '.join(cmd)}\n")
        fh.flush()
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, start_new_session=True,
                             env={**os.environ, "PYTHONUNBUFFERED": "1", **env})
        cut = threading.Event()

        def stop():                      # a silent process is cut too: not read-driven
            cut.set()
            try:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        timer = threading.Timer(max(0.0, timeout), stop)
        timer.start()
        try:
            for line in p.stdout:
                sys.stdout.write(line)
                fh.write(line)
            rc = p.wait()
        finally:
            timer.cancel()
        fh.write(f"# exit {rc}{' (cut by the time limit)' if cut.is_set() else ''} at {_now()}\n")
    return rc, cut.is_set()


def sanitizer_target(results: dict, *, anyway_a: bool):
    """The smallest of A and B that reproduced; A when neither did and
    `anyway_a`; else None."""
    for c in SMALL:
        if results.get(c, {}).get("status") == REPRODUCED:
            return c
    if anyway_a and all(c in results for c in SMALL):
        return "A"
    return None


def sanitizer_command(case: str, session: Path, *, cpu_boundary: bool, tool: str,
                      config=None) -> list[str]:
    inner = case_command(case, session / "sanitizer", cpu_boundary=cpu_boundary, sync=False,
                         config=config)
    return [tool, "--tool", "memcheck", "--print-limit", "50", "--error-exitcode", "99",
            "--log-file", str(session / "sanitizer" / "memcheck.log"), *inner]


def find_sanitizer():
    return shutil.which("compute-sanitizer") or next(
        (str(p) for p in (Path("/usr/local/cuda/bin/compute-sanitizer"),) if p.exists()), None)


def run_session(session: Path, *, deadline_s: float, cpu_boundary: bool, sync: bool,
                anyway_a: bool, runner=run_logged, sanitizer=None, config=None,
                min_case_s: float = 120) -> dict:
    """C, B, A, D, E, each a fresh process, until the deadline; the sanitizer
    as soon as A and B have both run (before D and E, which are C again and
    least likely to be cut short by it)."""
    session.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    summary = {"started": _now(), "deadline_s": deadline_s, "cases": {}, "sanitizer": None,
               "cpu_boundary": cpu_boundary}

    def left():
        return deadline_s - (time.monotonic() - start)

    def save():
        _write_json(session / "summary.json", summary)

    def sanitize():
        target = sanitizer_target(summary["cases"], anyway_a=anyway_a)
        tool = sanitizer if sanitizer is not None else find_sanitizer()
        s = {"target": target, "tool": tool}
        if target is None:
            s["status"] = "not run: neither A nor B reproduced"
        elif tool is None:
            s["status"] = "not run: compute-sanitizer not found"
        elif left() < min_case_s:
            s["status"] = CUT
        else:
            (session / "sanitizer").mkdir(parents=True, exist_ok=True)
            limit = min(SANITIZER_LIMIT_S, left())
            rc, cut = runner(sanitizer_command(target, session, cpu_boundary=cpu_boundary,
                                               tool=tool, config=config),
                             session / "sanitizer" / "run.log", env=dict(SANITIZER_ENV),
                             timeout=limit)
            s.update(exit=rc, limit_s=limit,
                     status=("did not finish in the limit" if cut else
                             "memcheck reported errors" if rc == 99 else f"exit {rc}"))
        summary["sanitizer"] = s
        save()

    for case in CASES:
        if left() < min_case_s:
            summary["cases"][case] = {"status": CUT}
            save()
            continue
        rc, cut = runner(case_command(case, session, cpu_boundary=cpu_boundary, sync=sync,
                                      config=config),
                         session / case / "case.log", env=dict(CASE_ENV.get(case, {})),
                         timeout=left())
        res_path = session / case / "case.json"
        res = json.loads(res_path.read_text()) if res_path.exists() else {
            "status": CUT if cut else ERROR, "note": f"no case.json; exit {rc}"}
        res["exit"] = rc
        summary["cases"][case] = res
        save()
        if case == "A":
            sanitize()
    if summary["sanitizer"] is None:
        sanitize()
    summary["ended"] = _now()
    save()
    return summary


def cmd_session(args) -> int:
    session = Path(args.session or ROOT / "results" /
                   time.strftime("t31_xa_diag_%Y%m%dT%H%MZ", time.gmtime()))
    if session.exists() and any(session.iterdir()):
        print(f"REFUSING: {session} is not empty", file=sys.stderr)
        return 2
    s = run_session(session, deadline_s=args.deadline_min * 60, cpu_boundary=args.cpu_boundary,
                    sync=not (args.cpu_boundary or args.no_sync), anyway_a=args.sanitize_a_anyway,
                    config=args.config)
    for c, r in s["cases"].items():
        print(f"{c}: {r['status']}", flush=True)
    print(f"sanitizer: {s['sanitizer']}", flush=True)
    print(f"summary: {session / 'summary.json'}", flush=True)
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("case")
    c.add_argument("case", choices=CASES)
    c.add_argument("--out", required=True)
    c.add_argument("--cpu-boundary", action="store_true")
    c.add_argument("--config", default=None)
    s = sub.add_parser("session")
    s.add_argument("--session", default=None)
    s.add_argument("--deadline-min", type=float, default=48.0,
                   help="minutes from now after which no case or sanitizer starts")
    s.add_argument("--cpu-boundary", action="store_true")
    s.add_argument("--no-sync", action="store_true")
    s.add_argument("--sanitize-a-anyway", action="store_true")
    s.add_argument("--config", default=None)
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.cmd == "case":
        try:
            return cmd_case(args)
        except Exception:
            traceback.print_exc()
            return 1
    return cmd_session(args)


if __name__ == "__main__":
    raise SystemExit(main())
