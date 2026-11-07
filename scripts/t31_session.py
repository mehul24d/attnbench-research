#!/usr/bin/env python3
"""The P1 session (T3.1): components, the committed prediction, then ground
truth, in order, stopping at the first failure, with a log per step.

Each step is `scripts/t31_steps.py STEP` in a process of its own, written to
`<session>/<NN>_<step>/`: its `step.log`, its `result.json` and its rows.
`<session>/manifest.json` lists every attempt of every step. On the instance
each step runs through `scripts/run_phase.sh`, so the session is synced to
GCS as each step ends, failed or not. The process, log, manifest and resume
rules are the T2.27 runner's (`t227_shakedown.py`), whose helpers this uses.

   1 preflight          T2.1 probe `start`; T2.20 fields; the config of record
   2 link               T2.5 link row; T2.6 Q; the recursion's mechanism table
   3 dense_select       T2.16 record per (length, batch), before any timed block
   4 probe              T2.18 structural probe (durations discarded)
   5 host_components    T2.2 null-kernel profile
   6 device_components  T2.4 replay; the priced cells
   7 predict            T2.11 write, sync, read back: the receipt
   8 hold               until sync + PLANNED_SYNC_WAIT_S on the instance clock
   9 ground_truth       T2.13, one process per compile-setting group
  10 verify             the analysis's late-prediction check, in session
  11 postflight         T2.1 probe `end`

**The order of the prediction and the clock** (plan Section 5.1; Appendix
A, 25 and 27 Oct 2026). Nothing before `predict` times the composed
workload. `hold` waits until the instance clock reads the prediction's
sync time plus `PLANNED_SYNC_WAIT_S` (`commit`), and each ground-truth process checks that
again before its first forward of any kind, and records the time it did so.
`verify` then runs the analysis's own refusal (`commit.load_committed`)
against the earliest of those stamps.

**Resuming.** After a failure and its fix, rerun from that step with
`--from STEP --session DIR`; the steps before it must have passed in that
session, and the failed attempt's directory is kept beside the new one.
Steps 5 and 6 resume per cell (`t31_steps.guarded_cells`): their directory
stays in place, so a cell that passed under the same code identity and
instance is not run again, and the earlier attempt is kept beside it as a
hard-linked copy (`keep_attempt`).
Once any ground-truth process has started, a resume from `predict` or
earlier is refused: a new prediction written after timed data exists would
not be a prediction. Such a session ends there, and a new one starts.

Usage, on the L4 from the repo root (after deploy, preflight, write gate and
`gcp_bootstrap_instance.sh`):
    python3 scripts/t31_session.py
    python3 scripts/t31_session.py --from link --session results/t31_p1_...
    python3 scripts/t31_session.py --list      # the steps, and what is not built
Dry run on a CPU (fakes for steps 4 to 6 and a dense-only ground truth on the
random two-layer model; no sync, no clean-tree refusal):
    python3 scripts/t31_session.py --dry-run --session /tmp/t31_dry
A rehearsal (Shakedown 2) runs steps 1 to 8 on the card and stops after
`hold`: no ground truth, so nothing is clocked. Its config is the session
config with `"rehearsal": true` and its own prediction prefix, written into
the session as `rehearsal_config.json`; its results are discarded:
    python3 scripts/t31_session.py --rehearsal
The same rehearsal on a CPU, with the real arms behind the stubbed GPU
boundary (`attnbench.cpu_boundary`; x-attention from `ATTNBENCH_XATTN_DIR`),
which must pass before a shakedown is launched (31 Oct 2026):
    python3 scripts/t31_session.py --cpu-boundary --session /tmp/t31_cpu
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from t227_shakedown import (_now, last_status, load_manifest, run_logged,  # noqa: E402
                            save_manifest, tree_clean)
from t31_steps import NOT_BUILT, STEPS, load_config, step_dir  # noqa: E402

# The steps that resume per cell, in place (ruling, 2 Nov 2026).
CELL_STEPS = ("host_components", "device_components")

CHECKLIST = {
    "preflight": "T2.1 probe start; T2.20 fields; session config of record",
    "link": "T2.5 link row; T2.6 Q; mechanism table per length",
    "dense_select": "T2.16 dense-baseline record per (length, batch)",
    "probe": "T2.18 structural probe, every arm",
    "host_components": "T2.2 null-kernel host profile",
    "device_components": "T2.4 replay; priced cells",
    "predict": "T2.11 prediction written, synced, read back",
    "hold": "sync + PLANNED_SYNC_WAIT_S (commit) on the instance clock",
    "ground_truth": "T2.13 ground truth, one process per compile group",
    "verify": "late-prediction check (load_committed) in session",
    "postflight": "T2.1 probe end",
}
if tuple(CHECKLIST) != STEPS:      # not `assert`: T2.17 keeps the harness free of them
    raise RuntimeError("CHECKLIST and t31_steps.STEPS disagree")

FIRST_CLOCKED = "ground_truth"

# A rehearsal ends before the first clocked step, and its prediction is
# written where P1's never is.
REHEARSAL_LAST = "hold"
REHEARSAL_PREFIX = "predictions/P1-rehearsal"
# Any values will do for a rehearsal (researcher, 31 Oct 2026); a short grid
# that still has a batch above 1 keeps a shakedown inside its halt. The batch
# above 1 is at 2048 since Shakedown 6 (1 Nov 2026): the bare timer runs each
# call's device work at least 1000 times, which at batch 4 and 16384 tokens
# is about 11 minutes of the L4 for one dense cell.
REHEARSAL_GRID = {"lengths": [2048, 16384], "batches": {"2048": [1, 4]}}
if STEPS.index(REHEARSAL_LAST) >= STEPS.index(FIRST_CLOCKED):
    raise RuntimeError("a rehearsal must stop before ground truth")


def step_command(step: str, session: Path, extra: list[str], *, dry_run: bool,
                 sync: bool) -> list[str]:
    out = step_dir(session, step)
    cmd = [sys.executable, str(ROOT / "scripts" / "t31_steps.py"), step, "--out", str(out),
           "--session", str(session), *extra, *(["--dry-run"] if dry_run else [])]
    if sync:
        n = STEPS.index(step) + 1
        cmd = ["bash", str(ROOT / "scripts" / "run_phase.sh"), f"t31_{n:02d}_{step}", "--", *cmd]
    return cmd


NEVER_STARTED = ("blocked", "not_run")      # a debug session's records of a step that did not run


def clocked_started(manifest: dict) -> bool:
    """Whether a ground-truth process has started: an attempt a debug session
    recorded as blocked or not run did not (and refusing a resume for it
    refused every resume after a failure before ground truth)."""
    return any(a["step"] == FIRST_CLOCKED and a["status"] not in NEVER_STARTED
               for a in manifest["attempts"])


def run_session(session: Path, *, start: str, extra: list[str], dry_run: bool, sync: bool,
                runner=run_logged, rehearsal: bool = False, debug: bool = False,
                keep_going: bool = False, rehearsal_end: bool = False,
                cap_s: float | None = None, clock=time.time) -> int:
    """`debug`: a debug session (`t31_debug`), whose manifest says so and which
    no other session may share a directory with. `keep_going`: a step whose
    inputs did not fail runs after a failure (`DEPENDS`); the others are
    recorded as blocked. `cap_s`: no step starts after this many seconds since
    the session began. `rehearsal_end`: end at `hold`, as a rehearsal does."""
    session.mkdir(parents=True, exist_ok=True)
    session_t0 = clock()
    manifest = load_manifest(session)
    if bool(manifest.get("debug")) != debug:
        print(f"REFUSING: {session} is {'a' if manifest.get('debug') else 'not a'} debug session",
              file=sys.stderr)
        return 2
    manifest.setdefault("dry_run", dry_run)
    if manifest["dry_run"] != dry_run:
        print(f"REFUSING: {session} is a {'dry' if manifest['dry_run'] else 'real'} session",
              file=sys.stderr)
        return 2
    manifest.setdefault("rehearsal", rehearsal)
    if manifest["rehearsal"] != rehearsal:
        print(f"REFUSING: {session} is {'a' if manifest['rehearsal'] else 'not a'} rehearsal",
              file=sys.stderr)
        return 2
    last = STEPS.index(REHEARSAL_LAST) if rehearsal or rehearsal_end else len(STEPS) - 1
    first = STEPS.index(start)
    if first > last:
        print(f"REFUSING --from {start}: a rehearsal ends at {REHEARSAL_LAST}", file=sys.stderr)
        return 2
    if clocked_started(manifest) and first <= STEPS.index("predict"):
        print(f"REFUSING --from {start}: ground truth has started in {session}, so a "
              f"prediction written now would follow timed data. Start a new session.",
              file=sys.stderr)
        return 2
    done = last_status(manifest)
    unpassed = [s for s in STEPS[:first] if done.get(s) != "passed"]
    if unpassed:
        print(f"REFUSING --from {start}: earlier steps have not passed in {session}: "
              f"{unpassed}", file=sys.stderr)
        return 2
    failed_steps = []
    for step in STEPS[first:last + 1]:
        d = step_dir(session, step)
        if cap_s is not None and clock() - session_t0 > cap_s:
            print(f"\nSTOPPED before {step}: the session's cap of {cap_s / 60:.0f} min "
                  f"has passed", flush=True)
            manifest["attempts"].append({
                "step": step, "n": STEPS.index(step) + 1, "status": "not_run",
                "reason": f"debug cap of {cap_s / 60:.0f} min", "exit": None,
                "started": _now(), "seconds": 0.0, "dir": d.name, "log": ""})
            save_manifest(session, manifest)
            return 1
        if keep_going:
            from t31_debug import blocked
            why = blocked(step, last_status(manifest))
            if why:
                print(f"\n----- {step}: BLOCKED (not run; failed or unrun inputs: {why})",
                      flush=True)
                manifest["attempts"].append({
                    "step": step, "n": STEPS.index(step) + 1, "status": "blocked",
                    "reason": f"inputs did not pass: {why}", "exit": None, "started": _now(),
                    "seconds": 0.0, "dir": d.name, "log": ""})
                save_manifest(session, manifest)
                failed_steps.append(step)
                continue
        if d.exists():
            kept = d.with_name(f"{d.name}.{done.get(step, 'previous')}."
                               f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
            keep_attempt(d, kept, in_place=step in CELL_STEPS)
            print(f"kept the earlier attempt at {kept}")
        n = STEPS.index(step) + 1
        print(f"\n===== {n:02d}/{len(STEPS)} {step}: {CHECKLIST[step]} =====", flush=True)
        t0, started = time.time(), _now()
        if step == FIRST_CLOCKED:     # recorded before it runs, so a crash still counts
            manifest["attempts"].append({"step": step, "n": n, "status": "started",
                                         "reason": "", "exit": None, "started": started,
                                         "seconds": 0.0, "dir": d.name,
                                         "log": f"{d.name}/step.log"})
            save_manifest(session, manifest)
        rc = runner(step_command(step, session, extra, dry_run=dry_run, sync=sync),
                    d / "step.log")
        res_path = d / "result.json"
        if res_path.exists():
            import json
            res = json.loads(res_path.read_text())
            status, reason = res["status"], res["reason"]
        else:
            status, reason = "failed", f"no result.json (exit {rc})"
        if status == "passed" and rc != 0:
            status, reason = "failed", f"result says passed but the step exited {rc} (sync?)"
        if status == "skipped":
            status, reason = "failed", f"skipped: {reason}"
        manifest["attempts"].append({
            "step": step, "n": n, "status": status, "reason": reason, "exit": rc,
            "started": started, "seconds": round(time.time() - t0, 1),
            "dir": d.name, "log": f"{d.name}/step.log"})
        save_manifest(session, manifest)
        print(f"----- {step}: {status.upper()}{': ' + reason if reason else ''}", flush=True)
        if status == "failed" and keep_going:
            print(f"\n{step} FAILED; going on with the steps that do not need it. "
                  f"Log: {d / 'step.log'}", flush=True)
            failed_steps.append(step)
            continue
        if status == "failed":
            print(f"\nSTOPPED at step {n} ({step}). Log: {d / 'step.log'}", flush=True)
            if clocked_started(manifest) and STEPS.index(step) >= STEPS.index(FIRST_CLOCKED):
                print("Ground truth has started: resume from ground_truth or later only.",
                      flush=True)
            print(f"After the fix: python3 scripts/t31_session.py --from {step} "
                  f"--session {session}", flush=True)
            return 1
    statuses = last_status(manifest)
    label = ("T3.1 rehearsal (results discarded)" if rehearsal else
             f"T3.1 DEBUG session ({manifest['debug']})" if debug else "T3.1 session")
    print(f"\n{label}: " + ", ".join(f"{s} {statuses.get(s)}" for s in STEPS[:last + 1]))
    return 1 if failed_steps else 0


def rehearsal_config(session: Path, source) -> Path:
    """The session config as a rehearsal's: `"rehearsal": true`, its own
    prediction prefix and `REHEARSAL_GRID`, written once into the session (a
    resume reuses it)."""
    import json
    from t31_steps import DEFAULT_CONFIG
    path = Path(session) / "rehearsal_config.json"
    if not path.exists():
        cfg = json.loads(Path(source or DEFAULT_CONFIG).read_text())
        cfg.update(rehearsal=True, prediction_prefix=REHEARSAL_PREFIX, **REHEARSAL_GRID)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg, indent=1))
    return path


def keep_attempt(d: Path, kept: Path, *, in_place: bool) -> None:
    """Keep a step's earlier attempt at `kept`. A step that resumes per cell
    (`in_place`) keeps its directory where it is, for its cells to resume
    from, and `kept` is a copy: the step's own files (its log, result and
    index, which the new attempt rewrites in place) copied, the cells'
    files hard-linked, which the new attempt never rewrites (a cell it runs
    again is removed and written afresh). Any other step's directory is
    moved."""
    if not in_place:
        shutil.move(str(d), str(kept))
        return
    kept.mkdir()
    for p in d.iterdir():
        if p.is_dir():
            shutil.copytree(p, kept / p.name, copy_function=os.link)
        else:
            shutil.copy2(p, kept / p.name)


def debug_main(args) -> int:
    """`--debug`: the session staged by `t31_debug.py stage`, from where it was
    staged, on its debug config; the cap is 1.5 times the projected duration
    of the steps it runs (not applied on a CPU, where nothing is timed). An
    exploratory session (`t31_debug.py explore`) starts at step 1 and its cap
    is the card's."""
    import json
    from t31_debug import DEBUG_CONFIG, DEBUG_STARTS
    from t31_steps import DEBUG_LABEL, STAGED_INPUTS
    if args.rehearsal or (args.dry_run and args.cpu_boundary):
        print("REFUSING: a debug session is not a rehearsal; --cpu-boundary or --dry-run "
              "alone is its CPU check", file=sys.stderr)
        return 2
    if args.session is None or args.config is not None:
        print("REFUSING: --debug takes the --session `t31_debug.py stage` made, and no "
              "--config (its own is in the session)", file=sys.stderr)
        return 2
    session = Path(args.session)
    staged = session / STAGED_INPUTS
    if not staged.exists() or not (session / DEBUG_CONFIG).exists():
        print(f"REFUSING: {session} was not staged by `t31_debug.py stage`", file=sys.stderr)
        return 2
    rec = json.loads(staged.read_text())
    if not rec.get("exploratory") and rec["start"] not in DEBUG_STARTS:
        print(f"REFUSING: a debug session starts at one of {DEBUG_STARTS}", file=sys.stderr)
        return 2
    start = rec["start"] if args.start == STEPS[0] else args.start
    if STEPS.index(start) < STEPS.index(rec["start"]):
        print(f"REFUSING: this debug session was staged to start at {rec['start']}; it "
              f"never reruns the steps before it (a resume after a fix may start later)",
              file=sys.stderr)
        return 2
    args.config = str(session / DEBUG_CONFIG)
    try:
        load_config(args.config, dry_run=args.dry_run)
    except Exception as e:
        print(f"REFUSING: {e}", file=sys.stderr)
        return 2
    cpu = args.dry_run or args.cpu_boundary
    if not cpu and not tree_clean():
        print("REFUSING: the tree is dirty, so no row could be banked with git_dirty = False",
              file=sys.stderr)
        return 2
    cap_s = None
    if not cpu and rec.get("exploratory"):
        cap_s = 60.0 * float(rec["cap_min"])
        print(f"{rec['debug'].upper()}: from {start}, cap {cap_s / 60:.0f} min (the card's)",
              flush=True)
    elif not cpu:
        from t31_projection import debug_cap_min
        cap_s = 60.0 * debug_cap_min(start, rec["not_measured"], rec["run_in_this_session"])
        print(f"{DEBUG_LABEL.upper()}: from {start}, cap {cap_s / 60:.0f} min "
              f"(1.5 x the projected steps)", flush=True)
    extra = ["--config", args.config] + (["--cpu-boundary"] if args.cpu_boundary else [])
    return run_session(session, start=start, extra=extra, dry_run=args.dry_run,
                       sync=not (cpu or args.no_sync), debug=True, keep_going=args.keep_going,
                       rehearsal_end=args.cpu_boundary, cap_s=cap_s)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--session", default=None,
                    help="session directory (default results/t31_p1_<UTC time>)")
    ap.add_argument("--from", dest="start", default=STEPS[0], choices=STEPS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-sync", action="store_true",
                    help="do not wrap steps in run_phase.sh (implied by --dry-run)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--rehearsal", action="store_true",
                    help="steps 1 to 8 only, no ground truth; results discarded (Shakedown 2)")
    ap.add_argument("--cpu-boundary", action="store_true",
                    help="the rehearsal (steps 1 to 8) on a CPU, real arms behind the stubbed "
                         "GPU boundary")
    ap.add_argument("--debug", action="store_true",
                    help="a debug session staged by `t31_debug.py stage` (data discarded, "
                         "'debug, not T3.1 data'); starts where it was staged")
    ap.add_argument("--keep-going", action="store_true",
                    help="with --debug: run every step whose inputs did not fail")
    ap.add_argument("--list", action="store_true",
                    help="print the steps, and what is not built yet, and exit")
    args = ap.parse_args(argv)
    if args.list:
        for i, s in enumerate(STEPS, 1):
            print(f"{i:2d} {s:18s} {CHECKLIST[s]}")
            if s in NOT_BUILT:   # empty since 31 Oct 2026; kept for a step that regresses
                print(f"   {'':18s} NOT BUILT: {NOT_BUILT[s]}")
        return 0
    if args.debug:
        return debug_main(args)
    if args.keep_going:
        print("REFUSING: --keep-going is for a debug session only", file=sys.stderr)
        return 2
    if args.cpu_boundary and (args.dry_run or args.rehearsal):
        print("REFUSING: --cpu-boundary is its own rehearsal; not with --dry-run or --rehearsal",
              file=sys.stderr)
        return 2
    if args.cpu_boundary:
        if args.session is None:
            print("REFUSING: --cpu-boundary needs a --session outside results/", file=sys.stderr)
            return 2
        extra = ["--cpu-boundary"] + (["--config", args.config] if args.config else [])
        return run_session(Path(args.session), start=args.start, extra=extra, dry_run=False,
                           sync=False, rehearsal=True)
    if args.rehearsal and args.dry_run:
        print("REFUSING: a rehearsal runs on the card; use --dry-run alone on a CPU",
              file=sys.stderr)
        return 2
    if args.rehearsal and args.session is None and args.start == STEPS[0]:
        args.session = str(ROOT / "results" /
                           f"t31_rehearsal_{time.strftime('%Y%m%dT%H%MZ', time.gmtime())}")
    if args.rehearsal and args.session is not None:
        args.config = str(rehearsal_config(Path(args.session), args.config))
    try:
        load_config(args.config, dry_run=args.dry_run)   # refuse undecided values up front
    except Exception as e:
        print(f"REFUSING: {e}", file=sys.stderr)
        return 2
    if args.session is None:
        if args.start != STEPS[0]:
            print("REFUSING: --from needs the --session it continues", file=sys.stderr)
            return 2
        args.session = str(ROOT / "results" / f"t31_p1_{time.strftime('%Y%m%dT%H%MZ', time.gmtime())}")
    if not args.dry_run and not tree_clean():
        print("REFUSING: the tree is dirty, so no row could be banked with git_dirty = False",
              file=sys.stderr)
        return 2
    extra = ["--config", args.config] if args.config else []
    return run_session(Path(args.session), start=args.start, extra=extra,
                       dry_run=args.dry_run, sync=not (args.dry_run or args.no_sync),
                       rehearsal=args.rehearsal)


if __name__ == "__main__":
    raise SystemExit(main())
