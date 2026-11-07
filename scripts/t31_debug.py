#!/usr/bin/env python3
"""A T3.1 debug session (researcher's ruling, 2 Nov 2026): steps 5 or 6 to 11
on staged step-5 outputs, to find every failure in steps 6 to 11 without
paying for step 5 again. Its data is discarded.

Two sessions stopped at `oracle_d0.5_L32768` in step 5, and each relaunch
spent about 90 minutes there before reaching steps 6 to 11, which have never
run on a GPU. A debug session stages the fourth session's passed cells and
runs the rest:

  * the staged cells are the donor's, byte for byte, each checked against the
    hashes its `recording.json` pins; `staged_inputs.json` names the donor
    (session, code identity, instance, config and manifest hashes) and every
    cell's entry hash, and a cell resumes only if its entry is that one
    (`t31_steps.staged_accept`);
  * the cell the ruling names (`oracle_d0.5_L32768`) is `not_measured` in the
    session's config, so no step prices, times or predicts it, and so is any
    other cell the donor never ran unless the session starts at step 5, which
    runs just those;
  * every output says "debug, not T3.1 data" (`result.json`, `manifest.json`,
    `DEBUG_NOT_T31_DATA.txt`), the prediction goes under `predictions/P1-debug`,
    and nothing of it is banked as T3.1;
  * `--keep-going` runs every step whose inputs did not fail, so one run finds
    all the failures (`DEPENDS`);
  * the session stops when its elapsed time passes the cap, 1.5 times the
    projected duration of the steps it runs (`t31_projection.debug_cap_min`).

    python3 scripts/t31_debug.py stage --donor DIR --session DIR [--start STEP]
    python3 scripts/t31_debug.py fetch --instance NAME --donor-session NAME --dest DIR
    python3 scripts/t31_debug.py explore --session DIR --cap-min MIN [--include-stuck-cell]

`explore` (researcher's ruling 4 of 6 Nov 2026) sets up the session after the
debug session: T3.1 live from step 1, nothing staged, every output labelled
"T3.1 exploratory, code identity may change" (`EXPLORATORY_LABEL`), the
prediction under `predictions/P1-exploratory`, `--keep-going`, and a fix
committed and deployed on the instance resumed as a debug session's is (each
step's passed cells resume under the session's own earlier commits). Its
cap is the card's (`--cap-min`, from the ₹500 cap), not 1.5 times a
projection. The T3.2 repeats are the real data, on the final code identity.

`scripts/t31_session.py --debug --session DIR --keep-going` then runs it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from t31_steps import (CELL_INDEX, CELL_MANIFEST, DEBUG_LABEL, PRICE_CACHE,  # noqa: E402
                       STAGED_INPUTS, STEPS, cell_dir, cells, nominal_density, step_dir)

DEBUG_PREFIX = "predictions/P1-debug"
EXPLORATORY_LABEL = "T3.1 exploratory, code identity may change"
EXPLORATORY_PREFIX = "predictions/P1-exploratory"
EXPLORATORY_NOTE = "T31_EXPLORATORY.txt"
DEBUG_NOTE = "DEBUG_NOT_T31_DATA.txt"
DEBUG_CONFIG = "debug_config.json"
DEBUG_STARTS = ("host_components", "device_components")
CAP_FACTOR = 1.5                          # ruling: projected duration of the steps x 1.5
# The cell the ruling marks, with the reason the permanent drop would give.
STUCK_CELL = {"arm": "oracle", "density": 0.5, "length": 32768, "batch": 1,
              "reason": "not measured: host memory limit of P1's pinned machine"}
# What a step needs from the others to mean anything: a step whose inputs
# failed is skipped, not run, under --keep-going. Steps 1 to 5 are staged or
# run in order, and a failure there stops what follows (an exploratory session
# runs them live: each needs every step before it).
DEPENDS = {
    "link": ("preflight",),
    "dense_select": ("preflight", "link"),
    "probe": ("preflight", "link", "dense_select"),
    "host_components": ("preflight", "link", "dense_select", "probe"),
    "device_components": ("host_components", "link"),
    "predict": ("device_components", "link"),
    "hold": ("predict",),
    "ground_truth": ("hold", "dense_select"),
    "verify": ("ground_truth", "predict"),
    "postflight": (),
}
NOT_STAGED = ("isolated_host.events.jsonl",)     # the donor's large event log; no step reads it


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def blocked(step: str, status: dict) -> list[str]:
    """The steps `step` depends on that did not pass (`status`: step -> status)."""
    return [d for d in DEPENDS.get(step, ()) if status.get(d) != "passed"]


def _donor_cells(donor: Path) -> dict:
    """Directory name -> the cell's manifest, for every cell of the donor's step 5
    that passed."""
    out = {}
    base = step_dir(donor, "host_components")
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        m = _read(d / CELL_MANIFEST)
        if m and m.get("status") == "passed" and (d / CELL_INDEX).exists():
            out[d.name] = m
    return out


def _read(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def check_recording(cd: Path) -> list[str]:
    """The files `recording.json` pins that are missing or differ."""
    rec = _read(cd / "recording" / "recording.json")
    if not rec:
        return ["recording.json"]
    return [f"recording/{n}" for n, want in rec["files"].items()
            if not (cd / "recording" / n).exists() or sha256(cd / "recording" / n) != want]


def debug_config(donor_cfg: dict, *, extra_missing=()) -> dict:
    """The donor's config of record as a debug session's: `debug`, its own
    prediction prefix, and `not_measured` the ruling's cell and any other
    named (`extra_missing`: entries with `reason`)."""
    cfg = json.loads(json.dumps(donor_cfg))
    cfg.update(debug=True, rehearsal=False, prediction_prefix=DEBUG_PREFIX)
    named = {(e["arm"], nominal_density(e["density"]), int(e["length"]), int(e["batch"]))
             for e in cfg.get("not_measured") or ()}
    nm = list(cfg.get("not_measured") or ())
    for e in (STUCK_CELL, *extra_missing):
        k = (e["arm"], nominal_density(e["density"]), int(e["length"]), int(e["batch"]))
        if k not in named:
            nm.append(dict(e))
            named.add(k)
    cfg["not_measured"] = nm
    return cfg


def stage(donor: Path, session: Path, *, start: str = "host_components",
          donor_source: str = "", cpu_donor: bool = False, dry_run: bool = False) -> dict:
    """Stage the donor session's steps 1 to 4 and its passed step-5 cells into
    `session`, verified, and write the debug config, the staged-inputs record,
    the note and the manifest. Refuses a donor that did not pass steps 1 to 4,
    a cell whose recording does not match its pinned hashes, or a session
    that already exists. `cpu_donor` accepts a CPU-boundary rehearsal as the donor
    and `dry_run` a dry-run session, with no cells, for the runner's two CPU
    checks; neither is a session whose data could be staged for a launch.
    Returns the staged-inputs record."""
    if start not in DEBUG_STARTS:
        raise SystemExit(f"a debug session starts at one of {DEBUG_STARTS}, not {start!r}")
    donor, session = Path(donor), Path(session)
    if session.exists() and any(session.iterdir()):
        raise SystemExit(f"{session} is not empty; a debug session starts in a new directory")
    dm = _read(donor / "manifest.json")
    if not dm or dm.get("debug") or bool(dm.get("dry_run")) != dry_run \
            or (dm.get("rehearsal") and not cpu_donor):
        raise SystemExit("the donor must be a real, non-rehearsal, non-debug session")
    last = {}
    for a in dm["attempts"]:
        last[a["step"]] = a["status"]
    early = STEPS[:4]
    if any(last.get(s) != "passed" for s in early):
        raise SystemExit(f"the donor's steps 1 to 4 did not all pass: "
                         f"{ {s: last.get(s) for s in early} }")
    cfg_bytes = (step_dir(donor, "preflight") / "session_config.json").read_bytes()
    donor_cfg = json.loads(cfg_bytes)
    passed = {} if dry_run else _donor_cells(donor)
    if not passed and not dry_run:
        raise SystemExit("the donor has no passed step-5 cell")
    ident = {json.dumps(m["identity"], sort_keys=True) for m in passed.values()}
    inst = {json.dumps(m["instance"], sort_keys=True) for m in passed.values()}
    if len(ident) > 1 or len(inst) > 1:
        raise SystemExit("the donor's cells are of more than one code identity or instance")
    bad = {n: b for n in passed if (b := check_recording(step_dir(donor, "host_components") / n))}
    if bad:
        raise SystemExit(f"cells whose recording differs from its pinned hashes: {bad}")

    # what the donor never ran (or did not pass), unless this session runs it
    probe = debug_config(donor_cfg)
    wanted = cells(probe)
    missing = [c for c in wanted if cell_dir(Path("."), *c).name not in passed]
    extra = []
    if start == "device_components" and not dry_run:
        for arm, d, L, b in missing:
            extra.append({"arm": arm, "density": d, "length": L, "batch": b,
                          "reason": "not run in the donor session; this debug session starts "
                                    "at step 6"})
    cfg = debug_config(donor_cfg, extra_missing=extra)
    session.mkdir(parents=True, exist_ok=True)
    for s in STEPS[:4]:
        shutil.copytree(step_dir(donor, s), step_dir(session, s))
    if (donor / "score_cache").exists():
        shutil.copytree(donor / "score_cache", session / "score_cache")
    hc = step_dir(session, "host_components")
    hc.mkdir()
    keep = {cell_dir(Path("."), *c).name for c in cells(cfg)} & set(passed)
    staged = []
    for n in sorted(keep):
        src = step_dir(donor, "host_components") / n
        shutil.copytree(src, hc / n, ignore=shutil.ignore_patterns(*NOT_STAGED))
        staged.append({"dir": n, "cell_index_sha256": sha256(hc / n / CELL_INDEX),
                       "trace_sha256": json.loads(
                           (hc / n / "recording" / "recording.json").read_text())["files"]["trace.json"]})
    pc = step_dir(donor, "host_components") / PRICE_CACHE
    if pc.exists():
        shutil.copy2(pc, hc / PRICE_CACHE)
    if start == "device_components":      # what step 5 would have written
        index = [_read(hc / cell_dir(Path("."), *c).name / CELL_INDEX) for c in cells(cfg)]
        (hc / "cells_index.json").write_text(json.dumps({"cells": index}))
        (hc / "result.json").write_text(json.dumps({
            "step": "host_components", "status": "passed", "reason": "", "staged": True,
            "debug": DEBUG_LABEL, "findings": {"cells": len(index)}}))

    record = {
        "debug": DEBUG_LABEL, "start": start,
        "donor": {"session": donor.name, "source": donor_source,
                  "identity": json.loads(next(iter(ident), "null")),
                  "instance": json.loads(next(iter(inst), "null")),
                  "config_sha256": hashlib.sha256(cfg_bytes).hexdigest(),
                  "manifest_sha256": sha256(donor / "manifest.json"),
                  "steps_passed": list(early)},
        "cells": staged,
        "not_measured": cfg["not_measured"],
        "run_in_this_session": [cell_dir(Path("."), *c).name for c in missing] if start == "host_components" else [],
        "debug_config_sha256": hashlib.sha256(
            json.dumps(cfg, indent=1).encode()).hexdigest(),
    }
    (session / DEBUG_CONFIG).write_text(json.dumps(cfg, indent=1))
    (session / STAGED_INPUTS).write_text(json.dumps(record, indent=1))
    (session / DEBUG_NOTE).write_text(
        f"{DEBUG_LABEL.upper()}\n\nThis directory is a T3.1 debug session (researcher's ruling, 2 Nov "
        f"2026). Its step-5 cells were staged from session {donor.name}; the cell(s) named in "
        f"{STAGED_INPUTS} as not measured were not run. Nothing here is T3.1 data: do not bank it "
        f"as such, and do not cite it.\n")
    manifest = {"attempts": [], "dry_run": dry_run, "rehearsal": False, "debug": DEBUG_LABEL}
    for a in dm["attempts"]:
        if a["status"] == "passed" and a["step"] in early:
            manifest["attempts"].append(dict(a, staged=True, staged_from=donor.name))
    if start == "device_components":
        manifest["attempts"].append({
            "step": "host_components", "n": 5, "status": "passed", "reason": "", "exit": 0,
            "started": "", "seconds": 0.0, "dir": "05_host_components",
            "log": "05_host_components/step.log", "staged": True, "staged_from": donor.name})
    (session / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return record


def explore(session: Path, *, cap_min: float, config=None,
            include_stuck_cell: bool = False) -> dict:
    """Set up an exploratory session in `session` (new): its config (the
    session config as a debug session's, labelled `EXPLORATORY_LABEL`, its
    own prefix, `oracle_d0.5_L32768` not measured unless included: the 6 Nov
    2026 ruling keeps the drop fallback in force), the staged-inputs record
    with no donor and no cells, the note and the manifest."""
    from t31_steps import DEFAULT_CONFIG
    session = Path(session)
    if session.exists() and any(session.iterdir()):
        raise SystemExit(f"{session} is not empty; an exploratory session starts in a new directory")
    if not cap_min > 0:
        raise SystemExit("--cap-min is the card's, in minutes, > 0")
    cfg = json.loads(Path(config or DEFAULT_CONFIG).read_text())
    cfg.update(debug=True, debug_label=EXPLORATORY_LABEL, rehearsal=False,
               prediction_prefix=EXPLORATORY_PREFIX)
    if not include_stuck_cell:
        cfg = debug_config(cfg)
        cfg.update(prediction_prefix=EXPLORATORY_PREFIX)
        for e in cfg["not_measured"]:
            if e["reason"] == STUCK_CELL["reason"]:
                e["reason"] += " (the drop fallback in force, researcher's ruling 3 of 6 Nov 2026)"
    session.mkdir(parents=True, exist_ok=True)
    text = json.dumps(cfg, indent=1)
    record = {"debug": EXPLORATORY_LABEL, "exploratory": True, "start": STEPS[0],
              "donor": None, "cells": [], "not_measured": cfg.get("not_measured") or [],
              "run_in_this_session": [], "cap_min": float(cap_min),
              "debug_config_sha256": hashlib.sha256(text.encode()).hexdigest()}
    (session / DEBUG_CONFIG).write_text(text)
    (session / STAGED_INPUTS).write_text(json.dumps(record, indent=1))
    (session / EXPLORATORY_NOTE).write_text(
        f"{EXPLORATORY_LABEL.upper()}\n\nThis directory is the T3.1 exploratory session "
        f"(researcher's ruling 4 of 6 Nov 2026): T3.1 live from step 1, fixed and resumed on the "
        f"instance, so its code identity may change between steps (every cell manifest names "
        f"its commit). Repeatability, the Gate A inputs and the floors come from the T3.2 "
        f"repeats on the final code identity, not from here.\n")
    (session / "manifest.json").write_text(json.dumps(
        {"attempts": [], "dry_run": False, "rehearsal": False, "debug": EXPLORATORY_LABEL,
         "exploratory": True}, indent=1))
    return record


def fetch_command(bucket: str, instance: str, donor_session: str, dest: Path) -> list[str]:
    """gsutil command for the donor's results in the bucket (layout of
    `run_phase.sh`: <hostname>/results/<session>), leaving out the event log.
    `bucket` is `t31_steps.results_bucket()`'s, which already starts `gs://`
    (the first debug session's fetch doubled the prefix)."""
    bucket = bucket.rstrip("/")
    if not bucket.startswith("gs://"):
        bucket = f"gs://{bucket}"
    return ["gsutil", "-m", "rsync", "-r", "-x", r".*isolated_host\.events\.jsonl$",
            f"{bucket}/{instance}/results/{donor_session}", str(dest)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage", help="stage a donor session into a new debug session")
    s.add_argument("--donor", required=True)
    s.add_argument("--session", required=True)
    s.add_argument("--start", default="host_components", choices=DEBUG_STARTS)
    s.add_argument("--cpu-donor", action="store_true",
                   help="the donor is a CPU-boundary rehearsal (the CPU check only)")
    s.add_argument("--dry-run", action="store_true",
                   help="the donor is a --dry-run session, no cells (the CPU check only)")
    e = sub.add_parser("explore", help="set up the exploratory session (6 Nov 2026 ruling 4)")
    e.add_argument("--session", required=True)
    e.add_argument("--cap-min", type=float, required=True,
                   help="no step starts after this many minutes (the card's)")
    e.add_argument("--include-stuck-cell", action="store_true",
                   help="measure oracle_d0.5_L32768 too (the researcher's word only)")
    f = sub.add_parser("fetch", help="pull the donor session from the results bucket")
    f.add_argument("--instance", required=True)
    f.add_argument("--donor-session", required=True)
    f.add_argument("--dest", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "explore":
        rec = explore(Path(args.session), cap_min=args.cap_min,
                      include_stuck_cell=args.include_stuck_cell)
        print(f"{rec['debug']}: from {rec['start']}, cap {rec['cap_min']:.0f} min; not measured: "
              f"{[(e['arm'], e['density'], e['length']) for e in rec['not_measured']]}")
        return 0
    if args.cmd == "fetch":
        from t31_steps import results_bucket
        Path(args.dest).mkdir(parents=True, exist_ok=True)
        return subprocess.run(fetch_command(results_bucket(), args.instance, args.donor_session,
                                            Path(args.dest))).returncode
    rec = stage(Path(args.donor), Path(args.session), start=args.start,
                donor_source=f"<instance>/results/{Path(args.donor).name}",
                cpu_donor=args.cpu_donor, dry_run=args.dry_run)
    print(f"staged {len(rec['cells'])} cells from {rec['donor']['session']}; not measured: "
          f"{[(e['arm'], e['density'], e['length']) for e in rec['not_measured']]}; "
          f"run here: {rec['run_in_this_session']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
