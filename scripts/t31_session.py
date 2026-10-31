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
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from t227_shakedown import (_now, last_status, load_manifest, run_logged,  # noqa: E402
                            save_manifest, tree_clean)
from t31_steps import NOT_BUILT, STEPS, load_config, step_dir  # noqa: E402

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


def step_command(step: str, session: Path, extra: list[str], *, dry_run: bool,
                 sync: bool) -> list[str]:
    out = step_dir(session, step)
    cmd = [sys.executable, str(ROOT / "scripts" / "t31_steps.py"), step, "--out", str(out),
           "--session", str(session), *extra, *(["--dry-run"] if dry_run else [])]
    if sync:
        n = STEPS.index(step) + 1
        cmd = ["bash", str(ROOT / "scripts" / "run_phase.sh"), f"t31_{n:02d}_{step}", "--", *cmd]
    return cmd


def clocked_started(manifest: dict) -> bool:
    return any(a["step"] == FIRST_CLOCKED for a in manifest["attempts"])


def run_session(session: Path, *, start: str, extra: list[str], dry_run: bool, sync: bool,
                runner=run_logged) -> int:
    session.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(session)
    manifest.setdefault("dry_run", dry_run)
    if manifest["dry_run"] != dry_run:
        print(f"REFUSING: {session} is a {'dry' if manifest['dry_run'] else 'real'} session",
              file=sys.stderr)
        return 2
    first = STEPS.index(start)
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
    for step in STEPS[first:]:
        d = step_dir(session, step)
        if d.exists():
            kept = d.with_name(f"{d.name}.{done.get(step, 'previous')}."
                               f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
            shutil.move(str(d), str(kept))
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
        if status == "failed":
            print(f"\nSTOPPED at step {n} ({step}). Log: {d / 'step.log'}", flush=True)
            if clocked_started(manifest) and STEPS.index(step) >= STEPS.index(FIRST_CLOCKED):
                print("Ground truth has started: resume from ground_truth or later only.",
                      flush=True)
            print(f"After the fix: python3 scripts/t31_session.py --from {step} "
                  f"--session {session}", flush=True)
            return 1
    statuses = last_status(manifest)
    print("\nT3.1 session: " + ", ".join(f"{s} {statuses.get(s)}" for s in STEPS))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--session", default=None,
                    help="session directory (default results/t31_p1_<UTC time>)")
    ap.add_argument("--from", dest="start", default=STEPS[0], choices=STEPS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-sync", action="store_true",
                    help="do not wrap steps in run_phase.sh (implied by --dry-run)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--list", action="store_true",
                    help="print the steps, and what is not built yet, and exit")
    args = ap.parse_args(argv)
    if args.list:
        for i, s in enumerate(STEPS, 1):
            print(f"{i:2d} {s:18s} {CHECKLIST[s]}")
            if s in NOT_BUILT:
                print(f"   {'':18s} NOT BUILT: {NOT_BUILT[s]}")
        return 0
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
                       dry_run=args.dry_run, sync=not (args.dry_run or args.no_sync))


if __name__ == "__main__":
    raise SystemExit(main())
