#!/usr/bin/env python3
"""The L4 shakedown (T2.27): the checklist's steps in order, stopping at the
first failure, with a log per step.

Each step is `scripts/t227_steps.py STEP` in a process of its own, written to
`<session>/<NN>_<step>/`: its `step.log` (everything the step printed), its
`result.json` (passed, failed or skipped, the reason, the findings) and its
rows and traces. `<session>/manifest.json` lists every attempt of every step.
On the instance each step runs through `scripts/run_phase.sh`, so the session
is synced to GCS as each step ends, failed or not.

A failure stops the session. After the fix, rerun from that step with
`--from STEP --session DIR`: the steps before it must have passed in that
session, and the failed attempt's directory is kept beside the new one
(`<NN>_<step>.failed.<time>`), so the failure and its fix are both on record,
as T2.27 asks.

`--keep-going` records a failure and goes on to the next step, and lets
`--from` skip earlier steps that have not passed, so one session collects
every step that does not depend on the failed one (Shakedown 1, 28 Oct
2026: step 1 failed and `--from` refused the four independent steps after
it). The session still exits 1 if any step failed.

The checklist (T2.27, as run on the L4; plan Section T2.27 has the rows):

   1 t21_t220      T2.1 host-speed probe; T2.20 provenance fields
   2 zoo           mechanism zoo, single and two-stream, under T2.3's recording rules
   3 link          T2.5 link microbenchmarks
   4 launch_queue  T2.6 launch-queue depth Q
   5 nsys          T2.7 Kineto checks; nsys-traced pass with the positive control
   6 replay        T2.4 on one arm and cell
   7 probe         T2.18 for each arm
   8 assert_gate   T2.17 gate on real prompts, real-package checks
   9 graph_breaks  T2.21 graph-break report
  10 null_kernel   null-kernel profile
  11 figure1       Figure 1 trace pair

Usage, on the L4 from the repo root (after deploy, preflight, write gate and
`gcp_bootstrap_instance.sh`):
    python3 scripts/t227_shakedown.py
    python3 scripts/t227_shakedown.py --from nsys --session results/t227_shakedown_...
Dry run on a CPU (no sync, no clean-tree refusal; steps that need CUDA are
recorded as skipped):
    python3 scripts/t227_shakedown.py --dry-run --session /tmp/t227_dry
Step options (`--length`, `--replay-arm`, `--arms`, ...) pass through to every
step; `python3 scripts/t227_steps.py --help` lists them.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from t227_steps import STEPS  # noqa: E402

CHECKLIST = {
    "t21_t220": "T2.1 host-speed probe; T2.20 provenance fields",
    "zoo": "T2.3 mechanism zoo, single and two-stream, with its recording rules",
    "link": "T2.5 link microbenchmarks",
    "launch_queue": "T2.6 launch-queue depth Q",
    "nsys": "T2.7 Kineto checks and the nsys-traced pass",
    "replay": "T2.4 on one arm and cell",
    "probe": "T2.18 structural probe, each arm",
    "assert_gate": "T2.17 gate on real prompts, real-package checks",
    "graph_breaks": "T2.21 graph-break report",
    "null_kernel": "null-kernel profile",
    "figure1": "Figure 1 trace pair",
}
if tuple(CHECKLIST) != STEPS:      # not `assert`: T2.17 keeps the harness free of them
    raise RuntimeError("CHECKLIST and t227_steps.STEPS disagree")


@dataclass(frozen=True)
class Shakedown:
    """What distinguishes one shakedown from another: its steps, the script
    that runs one, the phase prefix its GCS sync uses and its session name.
    `scripts/t228_steps.py` is the second (Shakedown 2); the runner below is
    shared."""
    steps: tuple
    checklist: dict
    script: str
    phase_prefix: str
    session_prefix: str
    parser: Callable
    title: str


T227 = Shakedown(STEPS, CHECKLIST, "t227_steps.py", "t227", "t227_shakedown",
                 lambda: __import__("t227_steps").parser(),
                 "The L4 shakedown (T2.27)")


def step_dir(session: Path, step: str, config: Shakedown = T227) -> Path:
    return session / f"{config.steps.index(step) + 1:02d}_{step}"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_manifest(session: Path) -> dict:
    p = session / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {"attempts": []}


def save_manifest(session: Path, manifest: dict) -> None:
    (session / "manifest.json").write_text(json.dumps(manifest, indent=1))


def last_status(manifest: dict) -> dict[str, str]:
    out = {}
    for a in manifest["attempts"]:
        out[a["step"]] = a["status"]
    return out


def step_command(step: str, out: Path, extra: list[str], *, dry_run: bool,
                 sync: bool, config: Shakedown = T227) -> list[str]:
    cmd = [sys.executable, str(ROOT / "scripts" / config.script), step, "--out", str(out),
           *extra, *(["--dry-run"] if dry_run else [])]
    if sync:
        n = config.steps.index(step) + 1
        cmd = ["bash", str(ROOT / "scripts" / "run_phase.sh"),
               f"{config.phase_prefix}_{n:02d}_{step}", "--", *cmd]
    return cmd


def run_logged(cmd: list[str], log: Path) -> int:
    """Run `cmd`, writing its output to `log` and to this terminal."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as fh:
        fh.write(f"# {_now()}\n# $ {' '.join(cmd)}\n")
        fh.flush()
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        for line in p.stdout:
            sys.stdout.write(line)
            fh.write(line)
        rc = p.wait()
        fh.write(f"# exit {rc} at {_now()}\n")
    return rc


def tree_clean() -> bool:
    r = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0 and not r.stdout.strip()


def run_session(session: Path, *, start: str, extra: list[str], dry_run: bool, sync: bool,
                runner=run_logged, keep_going: bool = False,
                config: Shakedown = T227, only: tuple | None = None) -> int:
    """`only` runs just those steps, in checklist order, in a new session: a
    rerun of the steps an earlier session failed (Shakedown 3). The others
    are not run and not reported."""
    STEPS, CHECKLIST = config.steps, config.checklist     # this session's, not T2.27's
    session.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(session)
    manifest.setdefault("dry_run", dry_run)
    if manifest["dry_run"] != dry_run:
        print(f"REFUSING: {session} is a {'dry' if manifest['dry_run'] else 'real'} session",
              file=sys.stderr)
        return 2
    if only is not None:
        if manifest["attempts"] or start != STEPS[0]:
            print("REFUSING: --only starts a new session; it does not continue one",
                  file=sys.stderr)
            return 2
        manifest["only"] = [s for s in STEPS if s in only]
        save_manifest(session, manifest)
    selected = tuple(manifest.get("only") or STEPS)
    done = last_status(manifest)
    first = STEPS.index(start)
    unpassed = [s for s in STEPS[:first] if s in selected and done.get(s) != "passed"]
    if unpassed and not keep_going and \
            not (dry_run and all(done.get(s) == "skipped" for s in unpassed)):
        print(f"REFUSING --from {start}: earlier steps have not passed in {session}: "
              f"{unpassed}", file=sys.stderr)
        return 2
    for step in (s for s in STEPS[first:] if s in selected):
        d = step_dir(session, step, config)
        if d.exists():
            kept = d.with_name(f"{d.name}.{done.get(step, 'previous')}.{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
            shutil.move(str(d), str(kept))
            print(f"kept the earlier attempt at {kept}")
        n = STEPS.index(step) + 1
        print(f"\n===== {n:02d}/{len(STEPS)} {step}: {CHECKLIST[step]} =====", flush=True)
        t0, started = time.time(), _now()
        rc = runner(step_command(step, d, extra, dry_run=dry_run, sync=sync, config=config),
                    d / "step.log")
        res_path = d / "result.json"
        if res_path.exists():
            res = json.loads(res_path.read_text())
            status, reason = res["status"], res["reason"]
        else:
            status, reason = "failed", f"no result.json (exit {rc})"
        if status == "passed" and rc != 0:
            status, reason = "failed", f"result says passed but the step exited {rc} (sync?)"
        if status == "skipped" and not dry_run:
            status, reason = "failed", f"skipped on a real run: {reason}"
        manifest["attempts"].append({
            "step": step, "n": n, "status": status, "reason": reason, "exit": rc,
            "started": started, "seconds": round(time.time() - t0, 1),
            "dir": d.name, "log": f"{d.name}/step.log"})
        save_manifest(session, manifest)
        print(f"----- {step}: {status.upper()}{': ' + reason if reason else ''}", flush=True)
        if status == "failed" and keep_going:
            print(f"(--keep-going: step {n} failed; going on to the next step)", flush=True)
        elif status == "failed":
            print(f"\nSTOPPED at step {n} ({step}). Log: {d / 'step.log'}\n"
                  f"After the fix: python3 scripts/{config.script.replace('_steps', '_shakedown')}"
                  f" --from {step} --session {session}", flush=True)
            return 1
    statuses = last_status(manifest)
    print(f"\n{config.title}: " + ", ".join(f"{s} {statuses.get(s)}" for s in selected))
    return 1 if any(statuses.get(s) == "failed" for s in selected) else 0


def main(argv=None, config: Shakedown = T227) -> int:
    STEPS, CHECKLIST = config.steps, config.checklist
    ap = argparse.ArgumentParser(description=config.title)
    ap.add_argument("--session", default=None,
                    help=f"session directory (default results/{config.session_prefix}_<UTC time>)")
    ap.add_argument("--from", dest="start", default=STEPS[0], choices=STEPS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-sync", action="store_true",
                    help="do not wrap steps in run_phase.sh (implied by --dry-run)")
    ap.add_argument("--list", action="store_true", help="print the checklist and exit")
    ap.add_argument("--keep-going", action="store_true",
                    help="record a failed step and go on; --from may then skip unpassed steps")
    ap.add_argument("--only", default=None, metavar="STEP[,STEP...]",
                    help="a new session of just these steps, in checklist order")
    args, extra = ap.parse_known_args(argv)
    only = None
    if args.only is not None:
        only = tuple(s.strip() for s in args.only.split(",") if s.strip())
        unknown = [s for s in only if s not in STEPS]
        if not only or unknown:
            print(f"REFUSING --only: unknown steps {unknown}; the steps are {list(STEPS)}",
                  file=sys.stderr)
            return 2
    if args.list:
        for i, s in enumerate(STEPS, 1):
            print(f"{i:2d} {s:13s} {CHECKLIST[s]}")
        return 0
    # Step options are validated here, before anything runs.
    config.parser().parse_args([STEPS[0], "--out", "x", *extra])

    if args.session is None:
        if args.start != STEPS[0]:
            print("REFUSING: --from needs the --session it continues", file=sys.stderr)
            return 2
        args.session = str(ROOT / "results" / f"{config.session_prefix}_{time.strftime('%Y%m%dT%H%MZ', time.gmtime())}")
    if not args.dry_run and not tree_clean():
        print("REFUSING: the tree is dirty, so no row could be banked with git_dirty = False",
              file=sys.stderr)
        return 2
    return run_session(Path(args.session), start=args.start, extra=extra,
                       dry_run=args.dry_run, sync=not (args.dry_run or args.no_sync),
                       keep_going=args.keep_going, config=config, only=only)


if __name__ == "__main__":
    raise SystemExit(main())
