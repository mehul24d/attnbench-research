#!/usr/bin/env python3
"""The P2 session (T6.1: Cascade Lake + A100, XAttention's Triton path), and
the session runner that the P4 session (`scripts/t64_session.py`) shares.

Not launched by anything. The researcher launches the instance from the Mac
(`scripts/gcp_launch_a100.sh`: Flex-start first, Spot on refusal); this runs
on the instance after deploy, preflight, the GCS write gate and
`scripts/gcp_bootstrap_instance.sh`. The launch card is
`docs/launch_cards/t61_p2_a100.md`.

**How it runs.** The steps of plan Section 5.3, in order, each in a process
of its own (`t61_session.py step NAME`), each written to
`<session>/<NN>_<step>/`: `step.log` (everything the step printed) and
`result.json` (passed, failed or skipped, the reason, the findings).
`<session>/manifest.json` lists every attempt. On the instance each step
runs through `scripts/run_phase.sh`, so the session is synced to GCS as each
step ends. The first failure stops the session. After the fix,
`--from STEP --session DIR` resumes: every earlier step must have passed in
that session, and the failed attempt's directory is kept beside the new one.

**How it ends** (researcher's item 4, 8 Nov 2026, after T3.2b idled about
80 minutes past its end and lost ~/t32b.log with its disk): a real session
runs under `scripts/session_exit.sh` and refuses to start without it
(`EXIT_WRAPPER_ENV`). On any exit of the session (passed, failed, refused,
killed, or hung up on) the wrapper's last act is to sync the logs,
manifests and every `~/*.log` to the bucket and then halt the VM, with no
watcher on the Mac; a block ends the same way. The teardown pulls the logs
and manifests from the bucket when the instance is halted or gone
(`gcp_teardown_session.sh`, step 2c). `tests/test_session_exit.py` is the
CPU dry run of it, failure exits included.

**The order the runner enforces (plan Section 5.1; Appendix A, 25 and 27 Oct
2026).**

  1. `predict` writes the prediction file and its sha256 sidecar and syncs
     both to the results bucket (`predict.commit.sync_prediction`, which
     reads them back). Its receipt holds the store's creation time.
  2. `ground_truth` comes immediately after `predict`; no step between them
     can clock anything. Before it starts, the runner holds until the
     instance clock is the planned wait (`commit.PLANNED_SYNC_WAIT_S`, the
     required 60 s gap plus 30 s) past the store's sync time, then records
     when it opened the step. The step refuses to start inside the gap.
  3. The strict first-clocked-block rule: the harness must report
     `first_clocked_block_started_at` (the first block of any kind it
     clocks, warm-ups included). A result carrying only the 25 Oct key
     (`first_timed_block_started_at`) is refused. The session's start is the
     earliest of every ground-truth attempt's open time and stamp, so a
     rerun never moves it later, and it must be more than 60 s after the
     sync.
  4. `late_check` runs `predict.commit.load_committed`, the analysis's own
     refusal, against the store with 60 s of allowed clock skew.
  5. Once `predict` has passed in a session, `--from` any step up to it is
     refused: a second prediction for the pairing, made in the same session,
     would let one be chosen after the other. Start a new session instead;
     the synced file stays on record.

**What is built and what is not.** Steps whose code exists call it: the
host-speed probe and the provenance checks (T2.1, T2.20), the link
microbenchmarks (T2.5), the launch-queue probe (T2.6), the prediction
commit and its sync (T2.11) and the late check, and the trace checks (T2.7).
The rest wait for their Phase 2 tasks, and their real bodies refuse, naming
the task. `prereqs`, the first step, reads the deployed plan's Section 8 and
refuses a real session while any task the session depends on is not Done,
so a missing piece stops the session before anything is measured.

**Blocks** (researcher's rulings, 7 Nov 2026; `attnbench.lock.blocks`): with
`--lock FILE --block ID` the session is one block of the pairing's spec in
the lock file, on its own boot and in its own directory. It runs the boot
steps and the block's own; an earlier block's results come in with
`--import-block DIR` (that block's session directory, from the bucket), each
checked before anything runs: the same block spec and code commit, every
step passed, not discarded. After this block's preflight, its instance type,
GPU and CPU model must be the earlier blocks'. Ground truth runs the block's
cells and the anchor cell, and every row carries the block's id. A block whose
instance restarted is discarded alone (`INTERRUPTED_DISCARD.md`).

**Dry run** (`--dry-run`, on a CPU): every step's GPU work is replaced by a
fake that writes the same files; the order, the hold, the stamp checks, the
commit, the sync (to a directory store) and the late check are the real
code. The dry clock is virtual: the hold advances it instead of sleeping.
No sync to GCS, no clean-tree refusal.

Usage, on the A100 from the repo root:
    python3 scripts/t61_session.py run --banked-at P1=<ISO time P1 was banked>
    python3 scripts/t61_session.py run --from STEP --session results/t61_p2_<UTC> ...
    python3 scripts/t61_session.py run --dry-run --session /tmp/t61_dry
    python3 scripts/t61_session.py run --list
    python3 scripts/t61_session.py run --lock LOCK.json --block b2 --import-block DIR ...
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PASSED, FAILED, SKIPPED = "passed", "failed", "skipped"
EXIT = {PASSED: 0, FAILED: 1, SKIPPED: 3}

from attnbench.predict.commit import (CLOCK_SKEW_ALLOWED_S,  # noqa: E402
                                      PLANNED_SYNC_WAIT_S, REQUIRED_SYNC_GAP_S)

# Plan Section 5.1: the workload starts at least the required gap after the
# sync, and the analysis allows skew between the store's clock and the
# instance's. Both come from predict/commit.py, the one source; the runner
# holds for the planned wait, the gap plus the skew plus 30 s (audit C2,
# finding 1; ruling 1g, 11 Nov 2026).
MIN_GAP_S = REQUIRED_SYNC_GAP_S
CLOCK_SKEW_S = CLOCK_SKEW_ALLOWED_S
STRICT_KEY = "first_clocked_block_started_at"   # Appendix A, 27 Oct 2026
OLD_KEY = "first_timed_block_started_at"        # the 25 Oct 2026 rule; refused

RECEIPT = "prediction_receipt.json"
T31_RECEIPT = "receipt.json"
OPENED = "ground_truth_opened.json"
STAMPS = "ground_truth_stamps.json"
DRY_CLOCK = "dry_clock.json"
DISCARD = "INTERRUPTED_DISCARD.md"
EXPLORATORY = "EXPLORATORY.md"
# Set by scripts/session_exit.sh, whose exit path syncs and halts; a real
# session refuses to start without it (researcher's item 4, 8 Nov 2026).
EXIT_WRAPPER_ENV = "ATTNBENCH_SESSION_EXIT"
# What a later block's preflight must share with every block it imports;
# a field missing on either side refuses (ruling 1d, 11 Nov 2026).
BLOCK_IDENTITY = ("instance_type", "gpu_name", "cpu_model", "zone", "boot_image", "driver",
                  "bootstrap_sha256")


class Skip(Exception):
    """This step needs what a CPU dry run does not have."""


class StepFailed(Exception):
    def __init__(self, reason: str, findings: Optional[dict] = None):
        super().__init__(reason)
        self.findings = findings or {}


class NotBuilt(StepFailed):
    """The step's real body waits for a Phase 2 task."""

    def __init__(self, task: str, what: str):
        super().__init__(f"not built: {what} ({task}); prereqs should have refused this session")


# --- the session's description ---------------------------------------------------------

@dataclass(frozen=True)
class Requirement:
    """A plan task whose Section 8 status must start with "Done", or, for P2
    and a task that is not a gate, "Waived for P2" with a dated Appendix A
    citation (`unmet`). `read`: a P2 or P4 runner reads the task's output, so
    it cannot be "Waived for P2/P4" (adviser's ruling 5, fourth set of 12 Nov
    2026)."""
    task: str
    why: str
    gate: bool = False
    read: bool = True


@dataclass(frozen=True)
class Step:
    name: str
    task: str            # the plan task(s) the step carries out
    what: str


@dataclass(frozen=True)
class SessionConfig:
    task: str                     # T6.1
    pairing: str                  # P2
    title: str
    script: str                   # scripts/<script>
    phase_prefix: str             # run_phase.sh phase names
    session_prefix: str           # results/<session_prefix>_<UTC>
    gpu: str                      # substring of gpu_name
    instance_type: str
    provisioning: tuple           # provisioning models the launcher can create
    code_path: dict               # the code path definition the cells carry
    banked: tuple                 # pairings banked before this prediction (Section 5.4)
    steps: tuple                  # Step, in order
    requirements: tuple           # Requirement
    files: tuple = ()             # (path, substring or None) the deployed tree must hold
    fakes: dict = field(default_factory=dict)    # step -> dry-run body
    reals: dict = field(default_factory=dict)    # step -> real body
    preflight_findings: Optional[Callable] = None   # row -> extra findings (P4's floor)
    step_args: tuple = ()         # passed to every step's process (P4's --gate-only)
    discard_on_restart: bool = False   # a resume on another boot discards the session (P4, Spot)
    boot_steps: tuple = ("prereqs", "preflight", "host_speed_end")   # every block's own (blocks.py)
    components_config: str = ""   # the T3.1 steps' config for this pairing, from the repo root
    exploratory: str = ""         # why no row of the session is evidence (a rehearsal), or ""
    staged: Optional[Callable] = None   # (cfg, args, steps) -> (findings, problems), at step 1 (P2)
    pin_timed_path: bool = False  # refuse to start unless the timed path is the lock's (P2, P4)

    @property
    def names(self) -> tuple:
        return tuple(s.name for s in self.steps)

    def step(self, name: str) -> Step:
        return self.steps[self.names.index(name)]


def validate(cfg: SessionConfig) -> None:
    """The order rules, checked when the config is built (not `assert`: T2.17
    keeps the harness free of them)."""
    n = cfg.names
    problems = []
    if len(set(n)) != len(n):
        problems.append("step names repeat")
    if n[:1] != ("prereqs",):
        problems.append("prereqs must be the first step")
    for need in ("predict", "ground_truth", "late_check", "dense_select"):
        if need not in n:
            problems.append(f"no {need} step")
    if not problems:
        p = n.index("predict")
        if n[p + 1:p + 2] != ("ground_truth",):
            problems.append("ground_truth must come immediately after predict")
        if n.index("late_check") < n.index("ground_truth"):
            problems.append("late_check must come after ground_truth")
        if n.index("dense_select") > p:
            problems.append("dense_select must come before predict: the prediction uses its choice")
        if "fa3_import" in n and n.index("fa3_import") > n.index("dense_select"):
            problems.append("fa3_import must come before dense_select")
    missing = [s for s in n if s not in cfg.fakes or s not in cfg.reals]
    if missing:
        problems.append(f"steps without a body: {missing}")
    if problems:
        raise RuntimeError(f"{cfg.task} session config: " + "; ".join(problems))


# --- clocks ------------------------------------------------------------------------------

def now(session: Path, dry_run: bool) -> float:
    """Unix seconds. A dry run's clock is time.time() plus a virtual offset
    that `hold` advances, so the 60 s rule is exercised without waiting. A
    real run refuses a session that has a virtual clock."""
    p = Path(session) / DRY_CLOCK
    if not dry_run:
        if p.exists():
            raise StepFailed(f"{p} exists: this is a dry session, and a real run never "
                             f"reads a virtual clock")
        return time.time()
    off = json.loads(p.read_text())["offset_s"] if p.exists() else 0.0
    return time.time() + off


def advance(session: Path, dry_run: bool, seconds: float) -> None:
    if not dry_run:
        time.sleep(seconds)
        return
    p = Path(session) / DRY_CLOCK
    off = json.loads(p.read_text())["offset_s"] if p.exists() else 0.0
    p.write_text(json.dumps({"offset_s": off + seconds}))


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat(timespec="microseconds")


def epoch(s: str) -> float:
    t = datetime.fromisoformat(s)
    if t.tzinfo is None:
        raise StepFailed(f"naive time {s!r}; every stamp here is UTC-aware")
    return t.timestamp()


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --- shared files ------------------------------------------------------------------------

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


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=1))
    return path


def read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def append_json(path: Path, item) -> list:
    items = read_json(path, [])
    items.append(item)
    write_json(path, items)
    return items


# --- stores --------------------------------------------------------------------------------

class DirStore:
    """The dry run's store: a directory, never clobbering, whose creation
    times come from the session's (virtual) clock. It has the same
    put/get/created as `predict.commit.Store`."""

    PREFIX = "drystore://"

    def __init__(self, root: Path, clock: Callable[[], float]):
        self.root, self.clock = Path(root), clock
        self.index = self.root / "_created.json"

    def _path(self, remote: str) -> Path:
        if not remote.startswith(self.PREFIX):
            raise StepFailed(f"not a {self.PREFIX} URI: {remote!r}")
        rel = remote[len(self.PREFIX):]
        if ".." in Path(rel).parts:
            raise StepFailed(f"bad remote {remote!r}")
        return self.root / rel

    def put(self, local, remote: str) -> None:
        p = self._path(remote)
        if p.exists():
            return                      # like `gsutil cp -n`; the read-back then fails
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, p)
        idx = read_json(self.index, {})
        idx[remote] = self.clock()
        write_json(self.index, idx)

    def get(self, remote: str) -> bytes:
        return self._path(remote).read_bytes()

    def exists(self, remote: str) -> bool:
        return self._path(remote).exists()

    def created(self, remote: str) -> datetime:
        return datetime.fromtimestamp(read_json(self.index, {})[remote], tz=timezone.utc)


METADATA_URL = "http://metadata.google.internal/computeMetadata/v1/project/project-id"


def results_bucket() -> str:
    """The results bucket the way `scripts/run_phase.sh` names it:
    ATTNBENCH_BUCKET, else attnbench-results-<the instance's project>. Never
    printed or written: the receipt keeps the path inside the bucket only."""
    b = os.environ.get("ATTNBENCH_BUCKET")
    if b:
        return b.rstrip("/")
    try:
        req = urllib.request.Request(os.environ.get("ATTNBENCH_METADATA_URL", METADATA_URL),
                                     headers={"Metadata-Flavor": "Google"})
        project = urllib.request.urlopen(req, timeout=5).read().decode().strip()
    except OSError as e:
        raise StepFailed(f"no results bucket: ATTNBENCH_BUCKET is unset and the metadata "
                         f"server did not answer ({type(e).__name__})")
    if not re.match(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$", project):
        raise StepFailed("no results bucket: the metadata server gave no project")
    return f"gs://attnbench-results-{project}"


def store_for(args):
    """(store, prefix) where prefix + receipt path is the remote URI."""
    if args.dry_run:
        session = Path(args.session)
        # a receipt names the dry store its prediction is in (a redone block's
        # is its discarded attempt's)
        held = (read_json(session / RECEIPT) or {}).get("dry_store")
        if held:
            return DirStore(Path(held), lambda: now(session, True)), DirStore.PREFIX
        # a block reads the prediction from the dry store of the block that committed it,
        # as every real block reads the one bucket
        b = load_manifest(session).get("block") or {}
        home = Path((b.get("imported") or {}).get(b.get("predict_block"), session))
        return DirStore(home / "dry_store", lambda: now(session, True)), DirStore.PREFIX
    from attnbench.predict.commit import GsutilStore
    return GsutilStore(), results_bucket() + "/"


def ledger_for(cfg: SessionConfig, args, pairing: Optional[str] = None):
    """The pairing's prediction ledger (`attnbench.predict.ledger`; ruling 1c,
    11 Nov 2026): in the bucket beside the predictions, or for a dry run in
    a store beside the session directory, shared by every dry session of the
    pairing run side by side."""
    from attnbench.predict.ledger import Ledger
    if args.dry_run:
        session = Path(args.session)
        return Ledger(DirStore(session.parent / "dry_pairing_store",
                               lambda: now(session, True)), DirStore.PREFIX,
                      pairing or cfg.pairing)
    from attnbench.predict.commit import GsutilStore
    return Ledger(GsutilStore(), results_bucket() + "/", pairing or cfg.pairing)


def banked_times(cfg: SessionConfig, args) -> tuple:
    """({pairing: banked at}, {pairing: training file sha256}) for every
    pairing this one is trained on. A real run reads both from each pairing's
    ledger entry (ruling 1e, 11 Nov 2026: the store's creation time of
    `banked.json`, written by `scripts/bank_pairing.py`) and refuses a typed
    --banked-at; a dry run takes --banked-at as given."""
    if args.dry_run:
        return parse_banked_at(args.banked_at), {}
    if args.banked_at:
        raise StepFailed("a real run reads each banking time from the pairing ledger "
                         "(ruling 1e, 11 Nov 2026); --banked-at is for a dry run only")
    times, files = {}, {}
    for p in cfg.banked:
        got = ledger_for(cfg, args, p).banked()
        if got is None:
            raise StepFailed(f"{p} has no banked entry in the pairing ledger; bank it with "
                             f"scripts/bank_pairing.py before {cfg.pairing} predicts")
        times[p], files[p] = got[0], got[1]["training_file_sha256"]
    return times, files


def ground_truth_cells(session: Path, block: Optional[dict]) -> list:
    """The ledger cells this ground-truth process clocks: a block's own and
    its anchor, else every cell the session's prediction covers."""
    from attnbench.predict.ledger import cell_name
    if block:
        return sorted({cell_name(c[0], c[2], c[3]) for c in block["run_cells"]})
    return list((read_json(session / RECEIPT) or {}).get("cells") or ())


# --- plan statuses ------------------------------------------------------------------------

_ROW = re.compile(r"^\|\s*(T\d+\.\d+)\s*\|(.*)\|\s*$")


def plan_statuses(text: str) -> dict:
    """Task ID -> its Section 8 status (the row's last cell)."""
    out = {}
    for line in text.splitlines():
        m = _ROW.match(line)
        if m:
            out[m.group(1)] = m.group(2).split("|")[-1].strip()
    return out


WAIVER = "Waived for P2"
WAIVER_BOTH = "Waived for P2/P4"     # ruling 5, fourth set of 12 Nov 2026
WAIVER_PAIRINGS = ("P2", "P4")
_CITATION = re.compile(r"Appendix A, (\d{1,2} [A-Z][a-z]{2} \d{4})")
_A_LINE = re.compile(r"^- \*\*(\d{1,2} [A-Z][a-z]{2} \d{4})\.\*\*", re.M)


class AppendixDates(frozenset):
    """The dates Appendix A has a line for, each with its lines (`lines`)."""

    def __new__(cls, lines: dict):
        obj = super().__new__(cls, lines)
        obj.lines = lines
        return obj


def appendix_a_dates(text: str) -> AppendixDates:
    """The dates Appendix A has a line for ("- **10 Nov 2026.** ...")."""
    a = text.find("## Appendix A")
    b = text.find("## Appendix B", a)
    lines: dict = {}
    if a >= 0:
        for line in text[a:b if b > a else None].splitlines():
            m = _A_LINE.match(line)
            if m:
                lines.setdefault(m.group(1), []).append(line)
    return AppendixDates(lines)


def waived(r: Requirement, status: str, appendix: frozenset, pairing: str | None) -> bool:
    """A waiver meets a requirement only for P2, never for a gate, and only
    when the status cites an Appendix A line that exists (adviser's ruling 1b
    dated 10 Nov 2026). "Waived for P2/P4" meets it for either pairing, for a
    task no runner reads, when a line of the cited date names the task and
    states what is lost (adviser's ruling 5, fourth set of 12 Nov 2026)."""
    if r.gate:
        return False
    m = _CITATION.search(status)
    if status.startswith(WAIVER_BOTH):
        if pairing not in WAIVER_PAIRINGS or r.read or m is None:
            return False
        lines = getattr(appendix, "lines", {}).get(m.group(1), ())
        return any(r.task in x and "lost" in x for x in lines)
    if pairing != "P2" or not status.startswith(WAIVER):
        return False
    return m is not None and m.group(1) in appendix


def unmet(requirements, statuses: dict, appendix: frozenset = frozenset(),
          pairing: str | None = None) -> list:
    rows = []
    for r in requirements:
        s = statuses.get(r.task)
        if s is None or not (s.startswith("Done") or waived(r, s, appendix, pairing)):
            rows.append({"task": r.task, "why": r.why,
                         "status": (s or "no row in Section 8")[:160]})
    return rows


def missing_files(files, root: Path = ROOT) -> list:
    out = []
    for path, needle in files:
        p = root / path
        if not p.exists():
            out.append(f"{path}: absent")
        elif needle is not None and needle not in p.read_text(errors="replace"):
            out.append(f"{path}: does not contain {needle!r}")
    return out


# --- shared step bodies ---------------------------------------------------------------------

def py(script: str, *argv: str) -> list:
    return [sys.executable, str(ROOT / "scripts" / script), *argv]


def run(cmd: list, **kw) -> int:
    """A child process, its output into this step's log."""
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, **kw).returncode


def body_prereqs(cfg: SessionConfig, args) -> dict:
    """Every plan task the session depends on is Done in the deployed plan,
    and the deployed tree holds the files the session calls. A real run
    refuses otherwise; a dry run records what is missing and skips."""
    text = (ROOT / "docs" / "RESEARCH_PLAN.md").read_text()
    statuses, appendix = plan_statuses(text), appendix_a_dates(text)
    rows = unmet(cfg.requirements, statuses, appendix, cfg.pairing)
    files = missing_files(cfg.files)
    findings = {"unmet": rows, "missing_files": files,
                "required": [r.task for r in cfg.requirements],
                "waived": [r.task for r in cfg.requirements
                           if waived(r, statuses.get(r.task) or "", appendix, cfg.pairing)]}
    bad = []
    if cfg.staged is not None:
        runs = (load_manifest(Path(args.session)).get("block") or {}).get("steps") or cfg.names
        findings["staged"], bad = cfg.staged(cfg, args, tuple(runs))
        findings["staged_problems"] = bad
    if rows or files or bad:
        msg = (f"{len(rows)} task(s) not Done ({', '.join(r['task'] for r in rows)})"
               + (f"; {len(files)} file check(s) failed" if files else "")
               + (f"; {len(bad)} staged input(s) refused: {'; '.join(bad)}" if bad else ""))
        if args.dry_run:
            print(json.dumps(findings, indent=1))
            raise Skip(f"a real session would refuse: {msg}")
        raise StepFailed(f"the session cannot run yet: {msg}", findings)
    return findings


T220_FIELDS = ("cpu_microcode", "cloud_provider", "instance_type", "instance_generation",
               "provisioning_model", "provisioning_model_problem", "python_optimize")


def check_provenance_row(cfg: SessionConfig, row: dict, integrity: list) -> list:
    """What the host-speed row must say on this pairing's instance."""
    problems = [f"T2.20 field missing: {f}" for f in T220_FIELDS if f not in row]
    problems += list(integrity)
    expect = {"cloud_provider": "gcp", "instance_type": cfg.instance_type,
              "provisioning_model_problem": None, "python_optimize": 0,
              "git_dirty": False}
    for k, v in expect.items():
        if row.get(k) != v:
            problems.append(f"{k} is {row.get(k)!r}, expected {v!r}")
    # `cfg.provisioning` holds launch names; a row carries row names
    # (`attnbench.provisioning`). Comparing the two directly refused every real row.
    from attnbench.provisioning import row_names
    allowed = list(row_names(cfg.provisioning))
    if row.get("provisioning_model") not in allowed:
        problems.append(f"provisioning_model {row.get('provisioning_model')!r} is not one of "
                        f"{allowed}")
    for k in ("cpu_microcode", "instance_generation"):
        if not row.get(k):
            problems.append(f"{k} is empty")
    if cfg.gpu not in str(row.get("gpu_name")):
        problems.append(f"gpu_name {row.get('gpu_name')!r} is not an {cfg.gpu}")
    if row.get("null_launch_s_median") is None:
        problems.append("no null-kernel burst on a GPU instance")
    return problems


def _host_speed_row(out: Path, tag: str) -> dict:
    if run(py("host_cpu_probe.py", "speed", "--out", str(out), "--tag", tag, "--require-gpu")):
        raise StepFailed(f"host_cpu_probe.py speed --tag {tag} failed (T2.1)")
    import pandas as pd
    return pd.read_parquet(out / f"host_speed_{tag}.parquet").iloc[0].to_dict()


GPU_VARIANTS = ("tests/test_swappable_attention_model_gpu.py",
                "tests/test_generation_wiring_gpu.py", "tests/test_inline_estimator_gpu.py")
GPU_VARIANT_COUNT = 5     # the five GQA tests of tests/_sdpa_env.py (T2.26)


def gpu_variant_problems(junit: Path, rc: int) -> tuple[dict, list]:
    """T2.26's five GPU variants as their JUnit record says they ran. On a
    GPU instance a skipped variant checked nothing, so it is a problem like
    a failure; so is a count other than five."""
    import xml.etree.ElementTree as ET
    try:
        cases = list(ET.parse(junit).getroot().iter("testcase"))
    except (OSError, ET.ParseError) as e:
        return {"rc": rc}, [f"no JUnit record of the GPU variants ({e})"]
    kinds = {"failure", "error", "skipped"}
    out = {k: [c.get("name") for c in cases if c.find(k) is not None] for k in kinds}
    passed = [c.get("name") for c in cases if not any(c.find(k) is not None for k in kinds)]
    counts = {"rc": rc, "passed": len(passed), **{k: len(v) for k, v in out.items()}}
    problems = [f"GPU variant {k}: {n}" for k in sorted(kinds) for n in out[k]]
    if len(cases) != GPU_VARIANT_COUNT:
        problems.append(f"{len(cases)} GPU variants ran, not {GPU_VARIANT_COUNT}")
    if rc != 0 and not problems:
        problems.append(f"pytest exited {rc} on the GPU variants")
    return counts, problems


def run_gpu_variants(out: Path) -> tuple[dict, list]:
    """T2.26: the GPU variants on this instance's GPU, from the preflight."""
    junit = out / "gpu_variants.xml"
    rc = run([sys.executable, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider",
              f"--junitxml={junit}", *GPU_VARIANTS],
             env={**os.environ, "ATTNBENCH_TEST_DEVICE": "cuda"})
    return gpu_variant_problems(junit, rc)


def lightgbm_problem() -> str | None:
    """The learned predictor's step (11) needs lightgbm at the pinned version;
    a session without it would spend steps 3 to 10 before stopping there."""
    from attnbench.predict.learned import LIGHTGBM_VERSION
    got = subprocess.run([sys.executable, "-c", "import lightgbm; print(lightgbm.__version__)"],
                         capture_output=True, text=True)
    if got.returncode != 0:
        return f"lightgbm does not import, so the learned predictor cannot fit (pinned {LIGHTGBM_VERSION})"
    if got.stdout.strip() != LIGHTGBM_VERSION:
        return f"lightgbm {got.stdout.strip()} is installed; the learned baseline is pinned to {LIGHTGBM_VERSION}"
    return None


def import_check(cfg: SessionConfig) -> tuple[list, dict]:
    """Every third-party module that the session's scripts can import, found by
    a scan of their imports, imported here; a missing one is a preflight problem
    that names the package. Attempt 1 of the L4 rehearsal ran ten steps before
    the eleventh found lightgbm missing."""
    from attnbench.session_imports import session_import_problems
    chains = sorted(p for p in (ROOT / "scripts").glob("*.sh") if cfg.script in p.read_text())
    return session_import_problems([cfg.script], shell_files=chains)


def real_preflight(cfg: SessionConfig, args) -> dict:
    """T2.20 provenance and the T2.1 host-speed probe at the start, read from
    the probe's own row; Triton imports (the XAttention code path here); the
    GPU variants of T2.26 pass on this GPU; on a Triton pairing, the smoke
    check (`run_preflight_smoke`)."""
    from attnbench import provenance
    out = Path(args.out)
    if run(py("host_cpu_probe.py", "snapshot", "--out", str(out), "--tag", "start")):
        raise StepFailed("host_cpu_probe.py snapshot failed")
    row = _host_speed_row(out, "start")
    problems = check_provenance_row(cfg, row, provenance.stamp_integrity_problems(row))
    tri = subprocess.run([sys.executable, "-c", "import triton; print(triton.__version__)"],
                         capture_output=True, text=True)
    if tri.returncode != 0:
        problems.append("triton does not import, so XAttention cannot take its Triton path")
    imports, found = import_check(cfg)
    problems += imports
    lgbm = lightgbm_problem()
    if lgbm:
        problems.append(lgbm)
    variants, vproblems = run_gpu_variants(out)
    problems += vproblems
    smoke, sproblems = run_preflight_smoke(cfg, args, out)
    problems += sproblems
    findings = {"provisioning_model": row.get("provisioning_model"),
                "instance_type": row.get("instance_type"), "gpu_name": row.get("gpu_name"),
                "cpu_model": row.get("cpu_model"), "zone": row.get("zone"),
                "boot_image": row.get("boot_image"), "driver": row.get("driver"),
                "bootstrap_sha256": row.get("bootstrap_sha256"),
                "null_launch_us_median": _us(row.get("null_launch_s_median")),
                "triton": tri.stdout.strip() or None, "gpu_variants": variants,
                "smoke": smoke,
                "imports_checked": sorted(found["required"]),
                "kernel_modules_absent": found["kernel_problems"],
                "problems": problems}
    if cfg.preflight_findings is not None:
        findings.update(cfg.preflight_findings(row))
    if problems:
        raise StepFailed(f"{len(problems)} preflight problem(s): {problems[:3]}", findings)
    return findings


def triton_pairing(cfg: SessionConfig) -> bool:
    """Whether the pairing's XAttention arms take the Triton path (an A100 or
    H100) and it runs T3.1's component steps."""
    return bool(cfg.components_config) and \
        str(cfg.code_path.get("implementation", "")).endswith(":triton")


def run_preflight_smoke(cfg: SessionConfig, args, out: Path) -> tuple:
    """T3.1's preflight smoke check (`t31_steps.preflight_smoke`) in its own
    process, before anything long (the adviser's rulings dated 13 Nov 2026,
    item 3): every arm at 2048 through steps 7 and 8, the prediction's input
    and the ground truth's harness, untimed. (findings, problems); none for a
    pairing off the Triton path."""
    if not triton_pairing(cfg):
        return None, []
    d = Path(out) / "smoke"
    d.mkdir(parents=True, exist_ok=True)
    rc = run(py("t31_steps.py", "preflight-smoke", "--out", str(d), "--session",
                str(args.session), "--config", str(ROOT / cfg.components_config)))
    found = read_json(d / "smoke.json") or {}
    if rc != 0 or not found or "error" in found:
        return found, [f"the preflight smoke check failed (rc {rc}): "
                       f"{found.get('error', 'no smoke.json')}"]
    return found, []


def _us(s):
    return None if s is None else float(s) * 1e6


def fake_preflight(cfg: SessionConfig, args) -> dict:
    from attnbench.provisioning import row_name
    # The name a real row carries: the fake used the GCP name, which the
    # check also compared against, so every dry run passed a real row's failure.
    row = {"cloud_provider": "gcp", "instance_type": cfg.instance_type,
           "instance_generation": cfg.instance_type.split("-")[0], "cpu_microcode": "0xdry",
           "provisioning_model": row_name(cfg.provisioning[0]),
           "provisioning_model_problem": None,
           "python_optimize": 0, "git_dirty": False, "gpu_name": f"fake {cfg.gpu}",
           "null_launch_s_median": 5e-6}
    problems = check_provenance_row(cfg, row, [])
    if problems:
        raise StepFailed(f"the fake row fails its own checks: {problems}")
    write_json(Path(args.out) / "host_speed_start.json", row)
    findings = {"provisioning_model": row["provisioning_model"], "fake": True,
                "instance_type": row["instance_type"], "gpu_name": row["gpu_name"],
                "cpu_model": os.environ.get("ATTNBENCH_FAKE_CPU_MODEL", "fake cpu"),
                "zone": "dry-zone", "boot_image": "dry-image", "driver": "dry",
                "bootstrap_sha256": "0" * 64}
    if cfg.preflight_findings is not None:
        findings.update(cfg.preflight_findings(row))
    return findings


def _real_host_speed(tag: str, cfg, args) -> dict:
    row = _host_speed_row(Path(args.out), tag)
    return {"tag": tag, "null_launch_us_median": _us(row.get("null_launch_s_median")),
            "git_dirty": row.get("git_dirty")}


def _fake_host_speed(tag: str, cfg, args) -> dict:
    write_json(Path(args.out) / f"host_speed_{tag}.json", {"tag": tag, "fake": True})
    return {"tag": tag, "fake": True}


def real_host_speed(tag: str):
    return functools.partial(_real_host_speed, tag)


def fake_host_speed(tag: str):
    return functools.partial(_fake_host_speed, tag)


# --- the T3.1 steps, run for this pairing ------------------------------------------------
#
# The component steps are T3.1's (`scripts/t31_steps.py`), the code P1 ran,
# called in this step's process with the pairing's config
# (`SessionConfig.components_config`). Their cross-step reads find this
# session's directories through `t31_steps.STEP_LAYOUT`, which `run_session`
# writes, since this runner numbers its steps in its own order.

T31_STEPS = ("link", "dense_select", "probe", "host_components", "device_components",
             "predict", "ground_truth")


def write_step_layout(cfg: SessionConfig, session: Path) -> Optional[Path]:
    """`t31_steps.STEP_LAYOUT`: each T3.1 step this session runs, by its
    directory here; none for a pairing with no components config."""
    if not cfg.components_config:
        return None
    sys.path.insert(0, str(ROOT / "scripts"))
    import t31_steps as st
    return write_json(Path(session) / st.STEP_LAYOUT,
                      {s: step_dir(session, s, cfg).name for s in T31_STEPS if s in cfg.names})


def _t31_step(name: str, cfg: SessionConfig, args) -> dict:
    from types import SimpleNamespace
    if not cfg.components_config:
        raise StepFailed(f"{cfg.pairing} names no components config for T3.1's {name}")
    sys.path.insert(0, str(ROOT / "scripts"))
    import t31_steps as st
    config = str(ROOT / cfg.components_config)
    ns = SimpleNamespace(out=args.out, session=args.session, dry_run=False,
                         cpu_boundary=False, config=config, step=name)
    try:
        return st.STEP_FUNCS[name](ns, st.load_config(config, dry_run=False))
    except st.StepFailed as e:
        raise StepFailed(str(e), e.findings) from e


def t31_step(name: str):
    """A real body that runs T3.1's step `name` on this pairing's config; a
    T3.1 refusal is this step's, with its findings."""
    return functools.partial(_t31_step, name)


def _check_link_row(row: dict) -> dict:
    empty = sorted(k for k, v in row.items()
                   if (k.startswith("s_fixed_") or k.startswith("bw_")) and v is None)
    findings = {"empty_fields": empty, "gpu_numa_node": row.get("gpu_numa_node"),
                "affinity_on_gpu_node": row.get("affinity_on_gpu_node")}
    if empty:
        raise StepFailed(f"link row has empty fields: {empty}", findings)
    return findings


def real_link(cfg, args) -> dict:
    """T2.5 and T2.6 as T3.1's `link` measures them (the link row, Q, and the
    recursion's mechanism table at each length), every field of the link row
    filled."""
    import pandas as pd
    findings = t31_step("link")(cfg, args)
    row = pd.read_parquet(Path(args.out) / "link" / "link.parquet").iloc[0].to_dict()
    return {**findings, **_check_link_row(row)}


def real_launch_queue(cfg, args) -> dict:
    """T2.6: Q as `link` measured it (T3.1's `link` measures the launch queue
    with the link), read and reported here, not measured twice."""
    d = step_dir(Path(args.session), "link", cfg)
    rin = read_json(d / "recursion_inputs.json")
    if rin is None or rin.get("fake") is not False:
        raise StepFailed(f"no measured recursion inputs in {d}: `link` has not passed")
    return {"q": int(rin["queue_depth"]), "from": "link"}


def _fake_files(names: tuple, findings: dict, cfg, args) -> dict:
    for n in names:
        write_json(Path(args.out) / n, {"fake": True, "pairing": cfg.pairing})
    return {"fake": True, **findings}


def fake_files(*names: str, **findings):
    """A dry-run body that writes placeholder files and returns findings."""
    return functools.partial(_fake_files, names, findings)


def not_built(task: str, what: str):
    def body(cfg, args):
        raise NotBuilt(task, what)
    return body


# --- the prediction --------------------------------------------------------------------------

def parse_banked_at(items) -> dict:
    out = {}
    for it in items or ():
        k, sep, v = it.partition("=")
        if not sep or not k:
            raise StepFailed(f"--banked-at takes PAIRING=ISO-time, got {it!r}")
        out[k] = datetime.fromisoformat(v)
        if out[k].tzinfo is None:
            raise StepFailed(f"--banked-at {k}: the time needs a UTC offset")
    return out


def fake_record(cfg: SessionConfig, written_at: datetime) -> dict:
    """A small, real prediction record from synthetic components: one dense
    and one sparse cell on this pairing's code path. No baselines are fitted,
    so it names no banked pairing; a real P2 or P4 record does."""
    from attnbench.predict import commit as C
    from attnbench.predict.intensity import ModelShape
    from attnbench.predict.recursion import MechanismCost, close_prefill, kernel_op, price, sync_op
    from attnbench.predict.training import Cell, HardwareFeatures
    costs = {"synchronize": MechanismCost(s_fixed=1e-5),
             "item": MechanismCost(s_fixed=2e-5, bandwidth=1e9, loaded_bandwidth=5e8)}
    shape = ModelShape(hidden=64, n_heads_q=4, n_heads_kv=2, head_dim=16, intermediate=128,
                       vocab=100)
    label = C.code_path_label(cfg.code_path)

    def cell(arm, density, ops):
        return Cell(pairing=cfg.pairing, model_name="dry-model", code_path=label, arm=arm,
                    context_length=2048, density=density, batch=1,
                    priced=price(close_prefill(ops), costs),
                    hardware=HardwareFeatures(pinned_bandwidth=2e10, pageable_bandwidth=1e10),
                    model=shape)
    cells = [cell("dense", 1.0, [kernel_op(1e-5, 4e-3, layer=0)]),
             cell("xattention", 0.5, [kernel_op(2e-5, 1e-3, layer=0),
                                      sync_op("item", payload_bytes=4, layer=0),
                                      kernel_op(1e-5, 1e-3, layer=0)])]
    return C.build_record(cfg.pairing, cells, 64, code_paths={label: cfg.code_path},
                          banked=[], fitted_sum=None, learned=None,
                          code=C.CodeStamp(git_commit="d" * 40, git_dirty=False),
                          written_at=written_at)


P1_TRAINING = ROOT / "results" / "p1_training" / "p1_training.json.gz"
P2_TRAINING = ROOT / "results" / "p2_training" / "p2_training.json.gz"


def training_rows(cfg: SessionConfig, path: Optional[Path] = None,
                  p2_path: Optional[Path] = None) -> list:
    """The banked pairings' ground-truth rows this pairing's committed fits
    train on. P1's are its training file (`attnbench.predict.p1_training`,
    researcher's decision, 8 Nov 2026: one row per cell, both repeats pooled);
    P2's, for P4, are P2's training file (`attnbench.predict.p2_training`,
    ruling 1h, 11 Nov 2026), whose sha256 is the pairing ledger's P2 entry
    (`training_files`, checked against the ledger by `real_record`)."""
    from attnbench.predict import p1_training as PT
    from attnbench.predict import p2_training as P2T
    later = [p for p in cfg.banked if p not in (PT.PAIRING, P2T.PAIRING)]
    if later:
        raise NotBuilt("T6.1", f"{later}'s training rows, which {cfg.pairing} is also trained on")
    path = Path(path or P1_TRAINING)
    if not path.exists():
        raise StepFailed(f"no P1 training file at {path}: fetch it (its sha256 is the lock's)")
    if not skips_lock(cfg):            # a rehearsal reads the file under its own lock
        _, why = p1_training_check(path)
        if why:
            raise StepFailed(why)
    # The rows carry the model shape the learned predictor's features
    # need (attempt 2 of the L4 rehearsal stopped at `predict` without it).
    from attnbench.predict.intensity import shape_for
    try:
        rows = PT.rows_from_record(PT.load(path), model=shape_for("primary"))
    except (PT.TrainingRowsError, OSError, ValueError, KeyError, TypeError) as e:
        raise StepFailed(str(e)) from e
    if P2T.PAIRING not in cfg.banked:
        return rows
    p2 = Path(p2_path or P2_TRAINING)
    if not p2.exists():
        raise StepFailed(f"no P2 training file at {p2}: build it with scripts/p2_training.py "
                         f"and fetch it (its sha256 is the pairing ledger's P2 entry)")
    try:
        return rows + P2T.rows_from_record(P2T.load(p2), model=shape_for("primary"))
    except (P2T.TrainingRowsError, OSError, ValueError, KeyError, TypeError) as e:
        raise StepFailed(str(e)) from e


def training_files(cfg: SessionConfig) -> dict:
    """{pairing: sha256} of the training files `training_rows` reads."""
    import hashlib
    out = {}
    for p, path, how in (("P1", P1_TRAINING, "its sha256 is the lock's"),
                         ("P2", P2_TRAINING, "its sha256 is the pairing ledger's P2 entry")):
        if p not in cfg.banked:
            continue
        path = Path(path)
        if not path.exists():
            raise StepFailed(f"no {p} training file at {path}: fetch it ({how})")
        out[p] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def real_record(cfg: SessionConfig, args, written_at: datetime) -> dict:
    """Every cell from the session's measured components (T3.1's assembly,
    `t31_steps.build_prediction`, on this pairing's components config), with
    the fitted sums and the learned predictor trained on the banked pairings
    (`training_rows`). The learned predictor's configuration is a lock value
    (Gate B): until `learned.LOCKED_CONFIG` is set this refuses, naming it."""
    from attnbench.predict import baselines as BL
    from attnbench.predict import learned as LM
    from attnbench.predict.training import committed_training_set
    from attnbench.predict import commit as C
    banked, files = banked_times(cfg, args)
    missing = [p for p in cfg.banked if p not in banked]
    if missing:
        raise StepFailed(f"{cfg.pairing} is trained on {list(cfg.banked)}; give each one's "
                         f"banking time with --banked-at (missing: {missing})")
    rows = training_rows(cfg)
    read = training_files(cfg)
    if not args.dry_run and files != read:
        raise StepFailed(f"the pairing ledger names training files {files}; this session "
                         f"reads {read}")
    ts = committed_training_set(rows, cfg.pairing, cfg.banked)
    # Ruling 1e: the record names the files and the rows it was fitted on.
    fitted = {"banked": list(cfg.banked), "fitted_sum": BL.fit_fitted_sum(ts),
              "fitted_sum_plus_s0": BL.fit_fitted_sum(ts, add_s0=True),
              "training_files": read, "rows_sha256": C.training_rows_sha256(ts.rows)}
    if LM.LOCKED_CONFIG is None:
        raise StepFailed("the learned predictor's configuration is a lock value (Gate B): "
                         "learned.LOCKED_CONFIG is not set, so no committed prediction can "
                         "carry the learned baseline")
    fitted["learned"] = LM.fit_learned(ts, LM.LOCKED_CONFIG)
    sys.path.insert(0, str(ROOT / "scripts"))
    import t31_steps as st
    try:
        comp = st.load_config(str(ROOT / cfg.components_config), dry_run=False)
        return st.build_prediction(comp, Path(args.session), dry_run=False, now=written_at,
                                   baselines=fitted, pairing=cfg.pairing)
    except st.StepFailed as e:
        raise StepFailed(str(e), e.findings) from e


def body_predict(cfg: SessionConfig, args, *, record_fn=None) -> dict:
    """Section 5.3 step 6: write the prediction and its sidecar, sync both,
    read them back, and leave the receipt the hold and the late check read."""
    from attnbench.predict import commit as C
    session = Path(args.session)
    if (session / RECEIPT).exists():
        raise StepFailed(f"{session / RECEIPT} exists: this session already committed a "
                         f"prediction, and a second one is never made in the same session")
    written_at = datetime.fromtimestamp(now(session, args.dry_run), tz=timezone.utc)
    if record_fn is None:
        record_fn = (lambda: fake_record(cfg, written_at)) if args.dry_run else \
            (lambda: real_record(cfg, args, written_at))
    record = record_fn()
    # The pairing's ledger first (ruling 1c, 11 Nov 2026): one prediction per
    # cell, none after any ground truth of the cell has started, and a redone
    # block's re-prediction bit for bit its discarded attempt's.
    from attnbench.predict import ledger as LG
    ledger = ledger_for(cfg, args)
    cells = LG.record_cells(record)
    digest = LG.predictions_digest(record)
    bid = (load_manifest(session).get("block") or {}).get("id")
    try:
        original = ledger.check_prediction(cells, digest, block=bid)
    except LG.LedgerError as e:
        raise StepFailed(str(e))
    if original is not None:
        # Nothing new is committed: the session uses the original file, synced
        # before any ground truth; the re-prediction is kept beside it.
        write_json(Path(args.out) / "re_prediction.json", record)
        local = Path(args.out) / Path(original["remote_path"]).name
        out = {**original["receipt"], "local": str(local), "re_predicted": True}
        write_json(session / RECEIPT, out)
        store, prefix = store_for(args)          # the original's store, from the receipt
        local.write_bytes(store.get(prefix + original["remote_path"]))
        write_json(Path(args.out) / RECEIPT, out)
        write_json(Path(args.out) / T31_RECEIPT, {**original["t31_receipt"], "local": local.name})
        return {k: out[k] for k in ("sha256", "remote_path", "synced_at", "written_at")}
    committed = C.write_prediction(record, Path(args.out))
    store, prefix = store_for(args)
    rel = f"predictions/{cfg.pairing}/{session.name}"
    receipt = C.sync_prediction(committed, store, prefix + rel)
    out = {"pairing": cfg.pairing, "task": cfg.task, "sha256": committed.sha256,
           "local": str(committed.path), "remote_path": f"{rel}/{committed.path.name}",
           "synced_at": receipt.synced_at.isoformat(timespec="microseconds"),
           "written_at": record["written_at"], "store": "dir" if args.dry_run else "gcs",
           "dry_run": args.dry_run, "cells": cells}
    if args.dry_run:
        out["dry_store"] = str(store.root)
    t31 = {**receipt.to_dict(), "local": committed.path.name}
    try:
        ledger.record_prediction(cells, {
            "session": session.name, "block": bid, "sha256": committed.sha256,
            "predictions_digest": digest, "remote_path": out["remote_path"],
            "receipt": out, "t31_receipt": t31})
    except LG.LedgerError as e:
        raise StepFailed(str(e))
    write_json(session / RECEIPT, out)
    write_json(Path(args.out) / RECEIPT, out)
    # T3.1's steps read the receipt at <predict>/receipt.json, in its own form
    # (`session_order.receipt_from_dict`, with the file's name in `local`).
    write_json(Path(args.out) / T31_RECEIPT, t31)
    return {k: out[k] for k in ("sha256", "remote_path", "synced_at", "written_at")}


# --- the hold, the ground truth and the strict stamp ------------------------------------------

def hold_for_sync(session: Path, dry_run: bool) -> float:
    """Wait until the instance clock is PLANNED_SYNC_WAIT_S past the store's
    sync time; record and return when ground truth was opened."""
    receipt = read_json(session / RECEIPT)
    if receipt is None:
        raise StepFailed(f"no {RECEIPT} in {session}: no prediction was synced")
    target = epoch(receipt["synced_at"]) + PLANNED_SYNC_WAIT_S
    t = now(session, dry_run)
    if t < target:
        print(f"holding {target - t:.1f} s: ground truth starts at least {MIN_GAP_S:.0f} s "
              f"after the prediction was synced (plan Section 5.1)", flush=True)
    while (t := now(session, dry_run)) < target:
        advance(session, dry_run, min(target - t, 5.0))
    opened = now(session, dry_run)
    append_json(session / OPENED, {"opened_at": iso(opened)})
    return opened


def session_start(session: Path) -> Optional[float]:
    """The earliest of every ground-truth open time and harness stamp."""
    ts = [epoch(o["opened_at"]) for o in read_json(session / OPENED, [])]
    ts += [epoch(s["stamp"]) for s in read_json(session / STAMPS, [])]
    return min(ts) if ts else None


def check_stamp(result: dict, *, synced_at: float, opened_at: float, ended_at: float,
                earlier_start: Optional[float]) -> dict:
    """The strict first-clocked-block rule on one harness result. Returns the
    numbers reported; raises StepFailed on any breach."""
    if STRICT_KEY not in result:
        if OLD_KEY in result:
            raise StepFailed(f"the harness stamped {OLD_KEY!r}, the first ground-truth block "
                             f"after warm-ups (25 Oct 2026 rule); the strict rule needs "
                             f"{STRICT_KEY!r}, the first block of any kind it clocks")
        raise StepFailed(f"the harness result has no {STRICT_KEY!r}")
    stamp = result[STRICT_KEY]
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
        raise StepFailed(f"{STRICT_KEY} is {stamp!r}, not unix seconds")
    stamp = float(stamp)
    if stamp < opened_at:
        raise StepFailed(f"{STRICT_KEY} ({iso(stamp)}) is before this attempt opened "
                         f"({iso(opened_at)}): not this run's stamp, or the clock moved back")
    if stamp > ended_at:
        raise StepFailed(f"{STRICT_KEY} ({iso(stamp)}) is after the harness returned")
    start = min(x for x in (stamp, opened_at, earlier_start) if x is not None)
    gap = start - synced_at
    if not gap > MIN_GAP_S:
        raise StepFailed(f"the first clocked block began {gap:.3f} s after the prediction "
                         f"was synced; the plan needs more than {MIN_GAP_S:.0f} s (K9)",
                         {"gap_s": gap})
    return {"stamp": iso(stamp), "session_start": iso(start), "gap_s": gap,
            "stamp_after_open_s": stamp - opened_at}


class FakeHarness:
    """Stands in for the T2.13 grid driver on a CPU: warm-ups, then rounds,
    with the strict stamp taken before the first warm-up's clock."""

    def __init__(self, session: Path, *, warmup_s: float = 0.5, rounds_s: float = 2.0):
        self.session, self.warmup_s, self.rounds_s = session, warmup_s, rounds_s

    def __call__(self, cfg, args, block=None) -> dict:
        first = now(self.session, True)
        advance(self.session, True, self.warmup_s + self.rounds_s)
        if block is None:
            rows = [{"arm": a, "context_length": 2048, "batch": 1, "median_ms": m, "fake": True}
                    for a, m in (("dense", 4.0), ("xattention", 2.5))]
        else:
            drift = float(os.environ.get("ATTNBENCH_FAKE_DRIFT", "0") or 0)
            rows = [dict(zip(("arm", "density", "context_length", "batch"), c),
                         median_ms=(4.0 if c[0] == "dense" else 2.5) * (1 + drift), fake=True)
                    for c in block["run_cells"]]
        return {STRICT_KEY: first, "complete": True, "untraced": {"rows": rows},
                "traced": {"rows": rows}, "overhead": {"fake": True}, "traces": []}


# The ground truth is T3.1's (`t31_steps.step_ground_truth`, T2.13): untraced,
# one process per compile-setting group, each with its dense control, on this
# pairing's components config, then each cell's traced pass (Section 5.3 step
# 8, in T3.1's step since the revised identity of 8 Nov 2026), written to
# `traced_L*_b*.json` and never to the untraced rows.

def _cell_of(c) -> tuple:
    from attnbench.lock.blocks import CELL_FIELDS, cell_key
    return cell_key(dict(zip(CELL_FIELDS, c)))


def block_config(comp: dict, block: dict) -> dict:
    """The components config for one block: every grid cell outside the
    block's run cells (its own and the anchor) named not measured here, and
    the lengths and batches cut to those the run cells name, so T3.1's grid
    is the block's. T3.1 times the dense control at every length and batch
    it visits (`t31_steps.group_arms`), so a block whose cells at a length
    and batch leave out that length's dense cell is refused here: its rows
    could never be exactly its cells (`tag_rows`), and the refusal comes
    before anything is clocked, not after."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import t31_steps as st
    want = {_cell_of(c) for c in block["run_cells"]}
    out = json.loads(json.dumps(comp))
    grid = st.cells(out)
    missing = want - {_cell_of(c) for c in grid}
    if missing:
        raise StepFailed(f"block {block['id']} names cells not in the grid: "
                         f"{sorted(missing, key=str)}")
    visited = sorted({(L, b) for _, _, L, b in want})
    no_control = [(L, b) for L, b in visited if ("dense", 1.0, L, b) not in want]
    if no_control:
        raise StepFailed(f"block {block['id']} has cells at {no_control} (length, batch) "
                         f"without the dense cell there: T3.1 times the dense control at "
                         f"every length it visits, so put it in the block or make it the anchor")
    out["lengths"] = sorted({L for L, _ in visited})
    out["batches"] = {str(L): sorted({b for L2, b in visited if L2 == L}) for L in out["lengths"]}
    out["not_measured"] = list(out.get("not_measured") or ()) + [
        {"arm": a, "density": st.CALIBRATED if d is None else d, "length": L, "batch": b,
         "reason": f"another block's cell (this is block {block['id']})"}
        for a, d, L, b in grid if _cell_of((a, d, L, b)) not in want and (L, b) in visited]
    return out


def t31_row_cell(row: dict) -> list:
    """The grid cell a T3.1 ground-truth row times: its cell arm (not its
    label), the density setting (dense's 1.0, which `density_setting` drops),
    the token budget (not the prompt's tokens) and the batch."""
    d = row["nominal_density"]
    if row["cell_arm"] == "dense" and d is None:
        d = 1.0
    return [row["cell_arm"], d, int(row["budget"]), int(row["batch"])]


def t31_ground_truth_result(d: Path) -> dict:
    """T3.1's ground-truth step output (`<group>/gt_*.json` and
    `<group>/stamps.json`) as this runner's result: every untraced row with
    its grid cell (`cell`) and its median as `median_ms` (the statistic the
    block drift check reads), the earliest stamp T3.1's session start reads
    (each group's pre-forward stamp and first clocked blocks), complete only
    if every result is."""
    from attnbench.analysis.drift_filter import flag_counts
    rows, stamps, complete, short = [], [], True, []
    for st_path in sorted(Path(d).glob("*/stamps.json")):
        s = json.loads(st_path.read_text())
        stamps += [s["clock_not_before"], *s["first_clocked_block_started_at"]]
        for g in sorted(st_path.parent.glob("gt_*.json")):
            r = json.loads(g.read_text())
            if r.get("traced"):
                raise StepFailed(f"{g} is traced: ground truth is untraced")
            complete = complete and r.get("complete") is True
            if r.get("complete") is not True:
                short.append({"group": st_path.parent.name, "cell": g.stem[len("gt_"):],
                              "clean_rounds": r.get("clean_rounds"),
                              "rounds_run": r.get("rounds_run"),
                              "reps": ((r.get("controls") or {}).get("min_clean_rounds")
                                       or (r.get("rows") or [{}])[0].get("reps")),
                              "max_rerun_rounds": (r.get("controls") or {}).get("max_rerun_rounds"),
                              "flags": flag_counts(r) if "rounds" in r else {}})
            rows += [dict(row, group=st_path.parent.name, cell=t31_row_cell(row),
                          median_ms=row["prefill_ms_median"])
                     for row in r["rows"]]
    if not rows:
        raise StepFailed(f"no ground-truth rows under {d}")
    return {STRICT_KEY: min(stamps), "complete": complete, "incomplete": short,
            "untraced": {"rows": rows}}


def incomplete_message(result: dict) -> str:
    """Why a ground truth is incomplete, by cell and flag type (the adviser's
    ruling 2, third set of 12 Nov 2026: launch 4 A's named neither)."""
    short = result.get("incomplete")
    if not short:
        return "the ground truth is incomplete, and the harness named no short cell"
    parts = []
    for c in short:
        flags = ", ".join(f"{k} {n}" for k, n in c["flags"].items()) or "no flagged round"
        parts.append(f"{c['group']} {c['cell']}: {c['clean_rounds']} clean of {c['reps']} in "
                     f"{c['rounds_run']} rounds, rerun budget {c['max_rerun_rounds']} "
                     f"(flagged rounds: {flags})")
    return "the ground truth is incomplete: " + "; ".join(parts)


def real_harness(cfg, args, block=None, *, step_fn=None) -> dict:
    """T3.1's ground-truth step on this pairing's components config, or on a
    block's grid (`block_config`, written beside the step's output), read back
    by `t31_ground_truth_result`. A cell short of clean rounds makes the result
    incomplete, which `body_ground_truth` refuses after writing it. The traced
    pass's files are listed as `traced_files`, and the Kineto traces they
    name (one per arm per cell, `t31_steps.TraceExport`) as `traces`, both
    relative to the step's output, for T2.7's step."""
    from types import SimpleNamespace
    if not cfg.components_config:
        raise StepFailed(f"{cfg.pairing} names no components config for the ground truth")
    sys.path.insert(0, str(ROOT / "scripts"))
    import t31_steps as st
    out = Path(args.out)
    config = ROOT / cfg.components_config
    if block is not None:
        comp = st.load_config(str(config), dry_run=False)
        config = write_json(out / "block_config.json", block_config(comp, block))
    ns = SimpleNamespace(out=str(out), session=args.session, dry_run=False,
                         cpu_boundary=False, config=str(config), step="ground_truth")
    step_fn = step_fn or st.step_ground_truth
    try:
        step_fn(ns, st.load_config(str(config), dry_run=False))
    except st.StepIncomplete:
        pass
    except st.StepFailed as e:
        raise StepFailed(str(e), e.findings) from e
    result = t31_ground_truth_result(out)
    traced = sorted(out.glob("*/traced_L*_b*.json"))
    result["traced_files"] = [str(p.relative_to(out)) for p in traced]
    result["traces"] = [str((p.parent / t).relative_to(out))
                        for p in traced for t in read_json(p, {}).get("traces", [])]
    return result


def body_ground_truth(cfg: SessionConfig, args, *, harness=None) -> dict:
    """Section 5.3 steps 7 and 8, after the hold. Refuses to start inside the
    60 s; checks the strict stamp after."""
    session = Path(args.session)
    receipt = read_json(session / RECEIPT)
    if receipt is None:
        raise StepFailed("no prediction receipt: predict has not passed in this session")
    synced = epoch(receipt["synced_at"])
    opened = read_json(session / OPENED, [])
    if not opened:
        raise StepFailed("the runner did not record opening ground truth (run it through "
                         "`run`, which holds for the sync)")
    opened_at = epoch(opened[-1]["opened_at"])
    t = now(session, args.dry_run)
    if t - synced < MIN_GAP_S:
        raise StepFailed(f"only {t - synced:.3f} s since the prediction was synced; ground "
                         f"truth starts at least {MIN_GAP_S:.0f} s after it")
    earlier = session_start(session)
    if harness is None:
        harness = FakeHarness(session) if args.dry_run else real_harness
    block = block_plan(session)
    # The pairing's ledger records the start before anything is clocked (ruling 1c).
    gt_cells = ground_truth_cells(session, block)
    if not gt_cells:
        raise StepFailed("no ledger cells for this ground truth: the receipt names none")
    ledger_for(cfg, args).record_ground_truth_start(
        gt_cells, {"session": session.name, "block": (block or {}).get("id"),
                   "opened_at": iso(opened_at)})
    result = harness(cfg, args, block) if block else harness(cfg, args)
    ended = now(session, args.dry_run)
    if block:
        tag_rows(result, block)
    stamp = result.get(STRICT_KEY)
    if isinstance(stamp, (int, float)) and not isinstance(stamp, bool):
        append_json(session / STAMPS, {"stamp": iso(float(stamp)), "opened_at": iso(opened_at)})
    write_json(Path(args.out) / "ground_truth.json", result)
    findings = check_stamp(result, synced_at=synced, opened_at=opened_at, ended_at=ended,
                           earlier_start=earlier)
    if result.get("complete") is not True:
        raise StepFailed(incomplete_message(result), findings)
    write_json(Path(args.out) / "traces.json", result.get("traces", []))
    return findings


def block_plan(session: Path) -> Optional[dict]:
    """This block's ground-truth plan from its manifest: its id, its own cells,
    the anchor, and the cells it runs (the anchor first); None outside blocks."""
    b = load_manifest(session).get("block")
    if not b or b.get("cells") is None and b.get("anchor") is None:
        return None
    cells = [tuple(c) for c in b.get("cells") or ()]
    anchor = None if b.get("anchor") is None else tuple(b["anchor"])
    return {"id": b["id"], "cells": cells, "anchor": anchor,
            "run_cells": ([anchor] if anchor is not None else []) + cells}


def tag_rows(result: dict, block: dict) -> None:
    """Every ground-truth row carries the block's id and whether it is the
    anchor; the rows must be exactly the block's cells and its anchor. A
    T3.1 row names its grid cell in `cell`."""
    from attnbench.lock.blocks import row_key
    want = {_cell_of(c) for c in block["run_cells"]}
    anchor = None if block["anchor"] is None else _cell_of(block["anchor"])
    for part in ("untraced", "traced"):
        rows = (result.get(part) or {}).get("rows") or []
        got = set()
        for r in rows:
            k = row_key(r)
            r["block"] = block["id"]
            r["anchor"] = k == anchor
            got.add(k)
        if rows and got != want:
            raise StepFailed(f"the {part} rows are cells {sorted(got, key=str)}, not this block's "
                             f"{sorted(want, key=str)}")


def body_late_check(cfg: SessionConfig, args) -> dict:
    """The analysis's own refusal (`load_committed`), run now, against the
    store: the hash, the sync before the session's first clocked block with
    60 s of skew, the write stamp, the clean commit, the training order."""
    from attnbench.predict import commit as C
    session = Path(args.session)
    receipt = read_json(session / RECEIPT)
    start = session_start(session)
    if receipt is None or start is None:
        raise StepFailed("no receipt or no ground-truth start in this session")
    store, prefix = store_for(args)
    banked, files = banked_times(cfg, args)
    # The learned values are refitted from the rows (ruling 1e); a dry
    # record fits nothing.
    rows = training_rows(cfg) if cfg.banked and not args.dry_run else None
    # The pairing's earliest ground-truth start over the prediction's cells
    # (ruling 1c), not only this session's.
    first = ledger_for(cfg, args).earliest_start(receipt.get("cells") or ())
    if first is not None:
        start = min(start, first.timestamp())
    try:
        cp = C.load_committed(store, prefix + receipt["remote_path"],
                              ground_truth_started_at=datetime.fromtimestamp(start, tz=timezone.utc),
                              clock_skew=timedelta(seconds=CLOCK_SKEW_S),
                              banked_at=banked, local=receipt["local"],
                              training_rows=rows, training_files=files or None)
    except C.PredictionCommitError as e:
        raise StepFailed(f"the analysis would refuse this prediction: {e}")
    if cp.sha256 != receipt["sha256"]:
        raise StepFailed(f"the store holds {cp.sha256}, the receipt says {receipt['sha256']}")
    return {"sha256": cp.sha256,
            "synced_at": receipt["synced_at"], "session_start": iso(start),
            "gap_s": start - epoch(receipt["synced_at"]), "clock_skew_s": CLOCK_SKEW_S}


def real_trace_checks(cfg, args) -> dict:
    """T2.7 on every trace the traced ground-truth pass wrote (paths relative
    to the ground-truth step's output)."""
    session = Path(args.session)
    gt = step_dir(session, "ground_truth", cfg)
    traces = read_json(gt / "traces.json", None)
    if not traces:
        raise StepFailed(f"no traces listed in {gt / 'traces.json'}")
    bad = [t for t in traces
           if run(py("check_trace.py", "kineto", str(gt / t),
                     "--out", str(Path(args.out) / "__".join(Path(t).with_suffix("").parts))))]
    if bad:
        raise StepFailed(f"check_trace.py flagged or failed on {len(bad)} trace(s)", {"bad": bad})
    return {"traces": len(traces)}


# --- the mask gates (T2.23, T2.24) ----------------------------------------------------------

# The session's copy of a banked pairing's XAttention masks: P1's as T3.1's
# step 5 wrote them (`<cell>/recording/masks.{npz,json}`, `t31_steps.cell_dir`),
# P2's as its mask gate did (`mask_gate/masks/triton/<cell>/masks.{npz,json}`).
P1_MASKS = "p1_masks"
P2_MASKS = "p2_masks"
_CELL_DIR = re.compile(r"^(?P<arm>.+)_d(?P<d>[^_]+)_L(?P<budget>\d+)_b(?P<batch>\d+)$")


def banked_masks(base: Path) -> list:
    """(budget, MaskRecord) for every XAttention cell banked under `base`,
    the budget read from the nearest cell directory above the bank, each read
    back through its own hashes (`mask_bank.load`)."""
    from attnbench.components import mask_bank
    base, out = Path(base), []
    for m in sorted(base.rglob("masks.json")):
        cell = next((c for c in map(_CELL_DIR.match, (p.name for p in m.relative_to(base).parents))
                     if c), None)
        if cell is None:
            raise StepFailed(f"{m}: not under a T3.1 cell directory")
        rec = mask_bank.load(m.with_suffix(""))
        if rec.code_path.implementation.startswith("xattn:"):
            out.append((int(cell["budget"]), rec))
    if not out:
        raise StepFailed(f"no XAttention masks banked under {base}")
    return out


def mask_gate_bound(path=None, plan_text=None) -> dict:
    """The mask-gate bound from the lock (T4.1, "mask-gate bound"), as
    `mask_gate.Bounds` takes it: one absolute density tolerance, T4.1's
    compiled-versus-eager one (adviser's ruling 3 of the eleventh 9 Nov 2026
    paste), so the lock's two must be equal. Refused with no complete lock."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import t31_steps as st
    from attnbench.lock import checklist
    p = Path(path or st.LOCK_FILE)
    if not p.exists():
        raise StepFailed(f"no lock file at {p}: the mask-gate bound is a lock value (T4.1)")
    lock = json.loads(p.read_text(encoding="utf-8"))
    text = plan_text if plan_text is not None else checklist.PLAN.read_text(encoding="utf-8")
    res = checklist.check(lock, text)
    if not res.complete:
        raise StepFailed(f"{p.name} is not a complete lock, so the mask-gate bound is unset")
    values = lock.get("values") or {}
    v = (values.get("mask_gate_bound") or {}).get("value")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise StepFailed(f"the lock's mask-gate bound is {v!r}, not one density tolerance")
    flag = (values.get(DENSITY_FLAG_ITEM) or {}).get("value")
    if flag != v:
        raise StepFailed(f"the lock's mask-gate bound {v!r} is not its compiled-versus-eager "
                         f"density tolerance {flag!r}: the ruling makes them one value")
    return {"max_density_difference": v}


DENSITY_FLAG_ITEM = "compiled_versus_eager_density_flag_tolerance"


def p1_training_check(path: Optional[Path] = None) -> tuple:
    """(sha256 or None, problem or None): the P1 training file against the
    lock's `p1_training.file_sha256`. Read at step 1 and again by the step
    that trains on it (`training_rows`), so a file swapped between the two is
    refused too."""
    import hashlib
    sys.path.insert(0, str(ROOT / "scripts"))
    import t31_steps as st
    path = Path(path or P1_TRAINING)
    want = ((read_json(st.LOCK_FILE, {}) or {}).get("p1_training") or {}).get("file_sha256")
    if not path.exists():
        return None, f"no P1 training file at {path}"
    if want is None:
        return None, (f"{Path(st.LOCK_FILE).name} names no P1 training file sha256 "
                      f"(p1_training.file_sha256)")
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    return got, (None if got == want else f"the P1 training file's sha256 is {got}, the lock's {want}")


def p2_staged_inputs(cfg: SessionConfig, args, runs: tuple) -> tuple:
    """What the session's later steps read from the staging, checked at step
    1 so that a missing input costs a minute, not the hours before the step
    that reads it (adviser's ruling 2 of the 10 Nov 2026 paste): the lock and
    its mask-gate bound and P1's banked masks (step 6, `mask_gate`), the P1
    training file at the lock's sha256 and the learned configuration (step
    11, `predict`). A block checks only what its own steps read. Returns
    (findings, problems)."""
    from attnbench.predict import learned as LM
    found, bad = {}, []
    if "mask_gate" in runs:
        try:
            found["mask_gate_bound"] = mask_gate_bound()["max_density_difference"]
        except StepFailed as e:
            bad.append(str(e))
        try:
            found["p1_mask_cells"] = len(banked_masks(Path(args.session) / P1_MASKS))
        except Exception as e:   # noqa: BLE001  (a bank that does not read back is refused too)
            bad.append(f"P1's masks: {e}")
    if "predict" in runs:
        got, why = p1_training_check()
        if got:
            found["p1_training_sha256"] = got
        if why:
            bad.append(why)
        if LM.LOCKED_CONFIG is None:
            bad.append("the learned predictor's configuration is a lock value (Gate B): "
                       "learned.LOCKED_CONFIG is not set")
    return found, bad


def record_both_paths(args, cells, out: Path, paths=("torch", "triton")) -> dict:
    """Each XAttention cell's prefill on each of `paths`, untimed
    (`recorder.record_masks`): `use_triton` set on a fresh backend before its
    first run, and the path the estimator took read back into the code path
    (`Harness.code_path`), refused if it is not the one asked for or if the
    prompt is not the banked one. {path: [MaskRecord, ...]}"""
    import torch
    sys.path.insert(0, str(ROOT / "scripts"))
    import t31_steps as st
    from attnbench.components import recorder
    from attnbench.components.shakedown_arms import ARMS
    h = st._harness(args)
    banks = {p: [] for p in paths}
    for arm, budget, batch, prompt_id in cells:
        for path in paths:
            triton = path == "triton"
            be = h._backend(ARMS[arm])
            be.use_triton, be.triton_effective = triton, None
            a = h.prepare(arm, budget, backend=be, density=st.density_setting(arm, None),
                          batch=batch)
            a.run_prefill()
            torch.cuda.synchronize()
            a.code_path = h.code_path(a)
            if a.code_path.implementation != f"xattn:{path}":
                raise StepFailed(f"{arm} L{budget}: asked for the {path} path, the "
                                 f"estimator took {a.code_path.implementation}")
            if a.prompt_id != prompt_id:
                raise StepFailed(f"{arm} L{budget}: prompt {a.prompt_id}, the bank's is {prompt_id}")
            rec = recorder.record_masks(
                a.run_prefill, out / path / st.cell_dir(Path(), arm, None, budget, batch).name,
                code_path=a.code_path, context_length=a.length, batch=batch,
                prompt_id=a.prompt_id)
            banks[path].append(rec.masks)
    h.unwrap()
    return banks


def _comparisons(comps) -> list:
    from dataclasses import asdict
    return [asdict(c) for c in comps]


def real_mask_gate(cfg, args, *, record=None, bound=None) -> dict:
    """T2.23 on P2 (Section 7): XAttention's two paths, through `use_triton`,
    recorded untimed on the prompts P1 banked masks for (`<session>/p1_masks`,
    T3.2d's step 5), each against P1's masks and against each other
    (`mask_gate.p2_mask_gate`). The comparisons are written before the bound
    is read, and the step refuses without one. The verdict is the two paths
    on P2's card; against P1's masks, across cards, a difference beyond the
    bound is a flag. Neither stops the session: the verdict is what the
    per-arm rule reads (Section 5.1), and the masks stay recorded."""
    from attnbench.components import mask_gate as MG
    session, out = Path(args.session), Path(args.out)
    p1 = banked_masks(session / P1_MASKS)
    cells = [(r.code_path.arm, budget, r.batch, r.prompt_id) for budget, r in p1]
    banks = (record or record_both_paths)(args, cells, out / "masks")
    ref = [r for _, r in p1]
    try:
        write_json(out / "mask_comparisons.json", {
            "torch_vs_p1": _comparisons(MG.compare_banks(ref, banks["torch"])),
            "triton_vs_p1": _comparisons(MG.compare_banks(ref, banks["triton"])),
            "triton_vs_torch_on_p2": _comparisons(MG.compare_banks(banks["torch"],
                                                                   banks["triton"]))})
        res = MG.p2_mask_gate(ref, banks["torch"], banks["triton"],
                              MG.Bounds(**(bound if bound is not None else mask_gate_bound())))
    except MG.MaskGateError as e:
        raise StepFailed(f"the mask gate cannot compare these banks: {e}") from e
    from dataclasses import asdict
    verdict = "pass" if res.passed else "fail"
    flags = list(res.cross_card_flags)
    write_json(out / "mask_gate.json", {"verdict": verdict, "cross_card_flags": flags,
                                        **asdict(res)})
    return {"verdict": verdict, "cross_card_flags": flags, "cells": len(cells),
            "failures": {k: len(getattr(res, k).failures)
                         for k in ("pytorch_vs_p1", "triton_vs_p1", "triton_vs_pytorch_on_p2")}}


# --- the P2 session -------------------------------------------------------------------------

# Phase 2 tasks a pairing session calls (plan Section 8). T2.14's AWS half,
# T2.15 and T2.28 read a committed file afterwards, and T2.25 is P1's (Gate
# A), so none of those is required to launch.
PHASE2 = (
    Requirement("T2.1", "host-speed probe at the start, middle and end"),
    Requirement("T2.2", "host components from the null-kernel profile"),
    Requirement("T2.3", "operation sequence and synchronisation map"),
    Requirement("T2.4", "device components from in-order graph replay"),
    Requirement("T2.5", "link microbenchmarks"),
    Requirement("T2.6", "launch-queue depth Q"),
    Requirement("T2.7", "trace checks on the traced ground truth"),
    Requirement("T2.8", "the recursion"),
    Requirement("T2.9", "sum, fitted sum and step-form baselines"),
    Requirement("T2.10", "the learned baseline"),
    Requirement("T2.11", "the prediction commit"),
    Requirement("T2.12", "the precomputed-mask arm in ground truth"),
    Requirement("T2.13", "the untraced ground-truth harness"),
    Requirement("T2.16", "dense-baseline selection (K2)"),
    Requirement("T2.17", "the asserts-removed XAttention arm"),
    Requirement("T2.18", "the structural probe for a new code path"),
    Requirement("T2.19", "naive baselines and intervals in the committed file"),
    Requirement("T2.20", "provenance fields on every row"),
    Requirement("T2.21", "compiled variants", read=False),     # dropped from P2 and P4
    Requirement("T2.22", "the dispatcher's choice in the committed file (H6)"),
    Requirement("T2.26", "the GPU suite record run by the launch scripts"),
)

GATES = (
    Requirement("T3.1", "P1 banked: P2 is trained on it", gate=True),
    Requirement("T3.3", "Gate A recorded", gate=True),
    Requirement("T4.4", "Gate B: the tagged lock commit", gate=True),
)

BUILT_FILES = (
    # the strict first-clocked-block stamp (main, PR #20); absent on this branch's base
    ("attnbench/ground_truth.py", STRICT_KEY),
    ("attnbench/predict/commit.py", "def load_committed"),
)

COMMON_STEPS_HEAD = (
    Step("prereqs", "Section 8", "plan rows Done (or waived, P2); deployed files present"),
    Step("preflight", "T2.20, T2.1", "provenance on the probe's row; host speed (start); "
                                     "Triton; the smoke check, every arm at 2048, untimed"),
    Step("link", "T2.5", "link microbenchmarks, one stamped row"),
    Step("launch_queue", "T2.6", "launch-queue depth Q"),
    Step("probe", "T2.18", "structural probe of the new code path; durations discarded"),
)
COMMON_STEPS_COMPONENTS = (
    Step("host_components", "T2.2", "null-kernel host components; CPU compute at target sizes"),
    Step("device_components", "T2.4", "in-order graph replay, isolated cross-check, "
                                       "replay-versus-trace check (first card of the code path)"),
)
COMMON_STEPS_TAIL = (
    Step("dense_select", "T2.16", "K2 choice per cell, recorded before the timed runs"),
    Step("host_speed_mid", "T2.1", "host-speed probe (middle)"),
    Step("predict", "T2.11", "prediction written, hashed, synced, read back"),
    Step("ground_truth", "T2.13", "held 60 s past the sync; untraced, then traced; strict stamp"),
    Step("late_check", "K9", "load_committed against the store, 60 s skew"),
    Step("trace_checks", "T2.7", "copy-kernel concurrency and UM faults on the traced pass"),
    Step("host_speed_end", "T2.1", "host-speed probe (end)"),
)


def common_bodies() -> tuple[dict, dict]:
    reals = {
        "prereqs": body_prereqs, "preflight": real_preflight, "link": real_link,
        "launch_queue": real_launch_queue,
        "probe": t31_step("probe"), "host_components": t31_step("host_components"),
        "device_components": t31_step("device_components"),
        "dense_select": t31_step("dense_select"),
        "host_speed_mid": real_host_speed("mid"), "predict": body_predict,
        "ground_truth": body_ground_truth, "late_check": body_late_check,
        "trace_checks": real_trace_checks, "host_speed_end": real_host_speed("end"),
    }
    fakes = {
        "prereqs": body_prereqs, "preflight": fake_preflight,
        "link": fake_files("link.json"), "launch_queue": fake_files("launch_queue.json", q=64),
        "probe": fake_files("sequences.json", verdict="same sequence"),
        "host_components": fake_files("host_components.json"),
        "device_components": fake_files("device_components.json", replay_vs_trace="within"),
        "dense_select": fake_files("dense_selection.json", chosen="sdpa_flash"),
        "host_speed_mid": fake_host_speed("mid"), "predict": body_predict,
        "ground_truth": body_ground_truth, "late_check": body_late_check,
        "trace_checks": fake_files("trace_checks.json"), "host_speed_end": fake_host_speed("end"),
    }
    return reals, fakes


def _t61() -> SessionConfig:
    reals, fakes = common_bodies()
    reals["mask_gate"] = real_mask_gate
    fakes["mask_gate"] = fake_files("mask_gate.json", verdict="pass")
    cfg = SessionConfig(
        task="T6.1", pairing="P2", title="The P2 session (T6.1: Cascade Lake + A100)",
        script="t61_session.py", phase_prefix="t61", session_prefix="t61_p2",
        gpu="A100", instance_type="a2-ultragpu-1g", provisioning=("FLEX_START", "SPOT"),
        code_path={"gpu": "A100", "implementation": "xattention:triton",
                   "libraries": [["torch", "dry"], ["triton", "dry"]], "python_optimize": 0},
        banked=("P1",), components_config="configs/t61/p2_session.json",
        steps=COMMON_STEPS_HEAD + (
            Step("mask_gate", "T2.23", "mask gate: Triton against PyTorch on one card; "
                                       "masks recorded untimed if it fails"),
        ) + COMMON_STEPS_COMPONENTS + COMMON_STEPS_TAIL,
        requirements=PHASE2 + (Requirement("T2.23", "the mask gate on P2", gate=True),) + GATES,
        files=BUILT_FILES, fakes=fakes, reals=reals, staged=p2_staged_inputs,
        pin_timed_path=True)
    validate(cfg)
    return cfg


# --- the runner -------------------------------------------------------------------------------

def step_dir(session: Path, step: str, cfg: SessionConfig) -> Path:
    return Path(session) / f"{cfg.names.index(step) + 1:02d}_{step}"


BOOT_ID_FILE = Path("/proc/sys/kernel/random/boot_id")


def boot_id() -> Optional[str]:
    """This boot's id (Linux), or None where there is none (the Mac, a dry run)."""
    try:
        return BOOT_ID_FILE.read_text().strip() or None
    except OSError:
        return None


def interrupted(manifest: dict, now: Optional[str]) -> bool:
    """The session began on another boot of this machine. A Spot VM that was
    preempted and started again, or any restart, lands here."""
    was = manifest.get("boot_id")
    return was is not None and now is not None and was != now


def load_manifest(session: Path) -> dict:
    return read_json(Path(session) / "manifest.json", {"attempts": []})


def last_status(manifest: dict) -> dict:
    return {a["step"]: a["status"] for a in manifest["attempts"]}


def step_command(cfg: SessionConfig, step: str, session: Path, out: Path, extra: list, *,
                 dry_run: bool, sync: bool) -> list:
    cmd = [sys.executable, str(ROOT / "scripts" / cfg.script), "step", step,
           "--session", str(session), "--out", str(out), *extra, *cfg.step_args,
           *(["--dry-run"] if dry_run else [])]
    if sync:
        n = cfg.names.index(step) + 1
        cmd = ["bash", str(ROOT / "scripts" / "run_phase.sh"),
               f"{cfg.phase_prefix}_{n:02d}_{step}", "--", *cmd]
    return cmd


def run_logged(cmd: list, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as fh:
        fh.write(f"# {_utc_now()}\n# $ {' '.join(cmd)}\n")
        fh.flush()
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        for line in p.stdout:
            sys.stdout.write(line)
            fh.write(line)
        rc = p.wait()
        fh.write(f"# exit {rc} at {_utc_now()}\n")
    return rc


def tree_clean() -> bool:
    r = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0 and not r.stdout.strip()


def refuse_start(cfg: SessionConfig, start: str, manifest: dict, dry_run: bool,
                 steps: Optional[tuple] = None) -> Optional[str]:
    """Why `--from start` is refused in this session, or None. `steps`: the
    steps this session holds (a block's own and those it imported)."""
    done = last_status(manifest)
    first = cfg.names.index(start)
    own = steps if steps is not None else cfg.names
    imported = {a["step"] for a in manifest["attempts"] if a.get("imported_from")}
    if done.get("predict") == PASSED and "predict" not in imported \
            and first <= cfg.names.index("predict"):
        return (f"predict has passed in this session, so --from {start} would make a second "
                f"prediction for {cfg.pairing} (or recompute what it was made from); start a "
                f"new session")
    unpassed = [s for s in cfg.names[:first] if s in own and done.get(s) != PASSED]
    if unpassed and not (dry_run and all(done.get(s) == SKIPPED for s in unpassed)):
        return f"earlier steps have not passed in this session: {unpassed}"
    return None


def run_session(cfg: SessionConfig, session: Path, *, start: str, extra: list, dry_run: bool,
                sync: bool, runner=run_logged, block: Optional["BlockRun"] = None) -> int:
    session = Path(session)
    session.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        write_step_layout(cfg, session)
    manifest = load_manifest(session)
    manifest.setdefault("dry_run", dry_run)
    manifest.setdefault("task", cfg.task)
    if cfg.exploratory:
        manifest["exploratory"] = cfg.exploratory
        if not (session / EXPLORATORY).exists():
            (session / EXPLORATORY).write_text(f"# Exploratory: no row here is evidence\n\n"
                                               f"{cfg.title}. {cfg.exploratory}\n")
    if manifest["dry_run"] != dry_run or manifest["task"] != cfg.task:
        print(f"REFUSING: {session} is a {'dry' if manifest['dry_run'] else 'real'} "
              f"{manifest['task']} session", file=sys.stderr)
        return 2
    had = manifest.get("block")
    begun = bool(manifest["attempts"]) or had is not None
    if begun and ((had is None) != (block is None) or (
            block is not None and (had["id"], had["spec_sha256"]) != (block.id, block.spec.sha256))):
        print(f"REFUSING: {session} is "
              f"{'block ' + manifest['block']['id'] if manifest.get('block') else 'not a block'} "
              f"of its spec; this run is {'block ' + block.id if block else 'not a block'}",
              file=sys.stderr)
        return 2
    steps = cfg.names if block is None else block.steps
    if start not in steps:
        print(f"REFUSING --from {start}: block {block.id} does not run it", file=sys.stderr)
        return 2
    if block is not None and manifest.get("block") is None:
        why = start_block(cfg, session, manifest, block, dry_run)
        if why:
            print(f"REFUSING: {why}", file=sys.stderr)
            return 2
    held = steps + tuple(a["step"] for a in manifest["attempts"] if a.get("imported_from"))
    why = refuse_start(cfg, start, manifest, dry_run, held)
    if why:
        print(f"REFUSING --from {start}: {why}", file=sys.stderr)
        return 2
    if (cfg.discard_on_restart or block is not None) and not dry_run:   # every block, P2's too
        now = boot_id()
        if interrupted(manifest, now):
            (session / DISCARD).write_text(
                f"# Discarded in full: the instance restarted\n\n"
                f"{'Block ' + block.id + ' of ' + cfg.pairing + '. ' if block else ''}"
                f"This session began on boot "
                f"`{manifest['boot_id']}` and `--from {start}` was tried on boot `{now}`. A "
                f"session that was interrupted is discarded in full (Appendix A, 7 Nov 2026): "
                f"none of its rows is evidence. Start a new session on a new instance.\n")
            print(f"REFUSING --from {start}: the instance restarted since this session began "
                  f"(boot {manifest['boot_id']} -> {now}); an interrupted session is discarded "
                  f"in full. See {session / 'INTERRUPTED_DISCARD.md'}", file=sys.stderr)
            return 2
        if now is not None and manifest.get("boot_id") is None:
            manifest["boot_id"] = now
            write_json(session / "manifest.json", manifest)
    # One commit per session or block (ruling 1c, 11 Nov 2026): `--from` at
    # another commit refuses.
    head = code_commit(dry_run)
    if manifest["attempts"] and manifest.get("code_commit") not in (None, head):
        print(f"REFUSING --from {start}: this session began at commit "
              f"{manifest['code_commit']} and HEAD is {head}; a session (or block) runs one "
              f"commit: start a new one", file=sys.stderr)
        return 2
    if manifest.get("code_commit") is None:
        manifest["code_commit"] = head
        write_json(session / "manifest.json", manifest)
    done = last_status(manifest)
    total = len(cfg.steps)
    for step in steps[steps.index(start):]:
        d = step_dir(session, step, cfg)
        if d.exists():
            kept = d.with_name(f"{d.name}.{done.get(step, 'previous')}."
                               f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
            shutil.move(str(d), str(kept))
            print(f"kept the earlier attempt at {kept}")
        n = cfg.names.index(step) + 1
        print(f"\n===== {n:02d}/{total} {step}: {cfg.step(step).what} =====", flush=True)
        attempt = {"step": step, "n": n, "started": _utc_now()}
        if step == "ground_truth":
            try:
                attempt["opened_at"] = iso(hold_for_sync(session, dry_run))
            except StepFailed as e:
                attempt.update(status=FAILED, reason=str(e), exit=None, seconds=0.0,
                               dir=d.name, log=None)
                manifest["attempts"].append(attempt)
                write_json(session / "manifest.json", manifest)
                print(f"----- {step}: FAILED: {e}", flush=True)
                return 1
        t0 = time.time()
        rc = runner(step_command(cfg, step, session, d, extra, dry_run=dry_run, sync=sync),
                    d / "step.log")
        res = read_json(d / "result.json")
        if res is not None:
            status, reason = res["status"], res["reason"]
        else:
            status, reason = FAILED, f"no result.json (exit {rc})"
        if status == PASSED and rc != 0:
            status, reason = FAILED, f"result says passed but the step exited {rc} (sync?)"
        if status == SKIPPED and not dry_run:
            status, reason = FAILED, f"skipped on a real run: {reason}"
        attempt.update(status=status, reason=reason, exit=rc, seconds=round(time.time() - t0, 1),
                       dir=d.name, log=f"{d.name}/step.log")
        manifest["attempts"].append(attempt)
        write_json(session / "manifest.json", manifest)
        print(f"----- {step}: {status.upper()}{': ' + reason if reason else ''}", flush=True)
        if status == PASSED and step == "preflight" and block is not None:
            differ = block_identity_problems(cfg, session, block)
            if differ:
                attempt.update(status=FAILED, reason="; ".join(differ))
                write_json(session / "manifest.json", manifest)
                status = FAILED
                print(f"----- {step}: FAILED: {'; '.join(differ)}", flush=True)
        if status == FAILED:
            print(f"\nSTOPPED at step {n} ({step}). Log: {d / 'step.log'}\n"
                  f"After the fix: python3 scripts/{cfg.script} run --from {step} "
                  f"--session {session}", flush=True)
            return 1
    statuses = last_status(manifest)
    print(f"\n{cfg.title}{' block ' + block.id if block else ''}: " +
          ", ".join(f"{s} {statuses.get(s)}" for s in steps))
    return 0


# --- blocks (researcher's rulings, 7 Nov 2026; attnbench.lock.blocks) -------------------------

@dataclass(frozen=True)
class BlockRun:
    spec: object                 # blocks.BlockSpec
    id: str
    steps: tuple                 # what this block runs, in the session's order
    imports: tuple               # (block id, session directory), every earlier block
    lock_sha256: str


def code_commit(dry_run: bool) -> str:
    if dry_run:
        return "dry"
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def verify_import(cfg: SessionConfig, block: BlockRun, block_id: str, d: Path,
                  dry_run: bool) -> list:
    """Why the session directory `d` cannot stand as block `block_id` of this
    spec for this block, or nothing."""
    m = read_json(Path(d) / "manifest.json")
    if not m:
        return [f"{d}: no manifest"]
    b = m.get("block") or {}
    out = []
    if b.get("id") != block_id or b.get("spec_sha256") != block.spec.sha256:
        out.append(f"{d}: it is block {b.get('id')!r} of spec {str(b.get('spec_sha256'))[:12]}, "
                   f"not {block_id} of {block.spec.sha256[:12]}")
    if m.get("task") != cfg.task or bool(m.get("dry_run")) != dry_run:
        out.append(f"{d}: a {'dry' if m.get('dry_run') else 'real'} {m.get('task')} session")
    if (Path(d) / DISCARD).exists():
        out.append(f"{d}: block {block_id} was interrupted and is discarded ({DISCARD})")
    if m.get("code_commit") != code_commit(dry_run):
        out.append(f"{d}: block {block_id} ran commit {m.get('code_commit')}, this one "
                   f"{code_commit(dry_run)}; every block runs one commit")
    done = last_status(m)
    unpassed = [s for s in b.get("steps") or () if done.get(s) != PASSED
                and not (dry_run and done.get(s) == SKIPPED)]   # a dry run skips prereqs
    if unpassed:
        out.append(f"{d}: block {block_id}'s steps {unpassed} did not pass")
    return out


def start_block(cfg: SessionConfig, session: Path, manifest: dict, block: BlockRun,
                dry_run: bool) -> Optional[str]:
    """A new block's manifest: the spec it runs, and every earlier block's
    results imported after `verify_import`; or why not."""
    own = block.spec.block(block.id)
    problems = []
    for bid, d in block.imports:
        problems += verify_import(cfg, block, bid, d, dry_run)
    if problems:
        return "; ".join(problems)
    for bid, d in block.imports:
        earlier = read_json(Path(d) / "manifest.json")
        for step in block.spec.block(bid).steps:
            if step not in cfg.names or step in cfg.boot_steps or step in block.steps:
                continue
            src = step_dir(Path(d), step, cfg)
            shutil.copytree(src, step_dir(session, step, cfg))
            a = next(x for x in reversed(earlier["attempts"]) if x["step"] == step)
            manifest["attempts"].append(dict(a, imported_from=bid))
        if (Path(d) / RECEIPT).exists() and not (session / RECEIPT).exists():
            shutil.copy2(Path(d) / RECEIPT, session / RECEIPT)
    manifest["block"] = {
        "id": block.id, "pairing": cfg.pairing, "spec_sha256": block.spec.sha256,
        "lock_sha256": block.lock_sha256, "steps": list(block.steps),
        "cells": None if own.cells is None else [list(c) for c in own.cells],
        "anchor": None if block.spec.anchor is None or "ground_truth" not in own.steps
        else list(block.spec.anchor),
        "imported": {bid: str(d) for bid, d in block.imports},
        "predict_block": block.spec.predict_block()}
    manifest["code_commit"] = code_commit(dry_run)
    write_json(session / "manifest.json", manifest)
    return None


def block_identity_problems(cfg: SessionConfig, session: Path, block: BlockRun) -> list:
    """This block's preflight against every block it imported: the same
    instance type, GPU, CPU model, zone, boot image, driver and bootstrap
    script, each recorded on both sides, or the blocks are not one pairing."""
    mine = (read_json(step_dir(session, "preflight", cfg) / "result.json") or {}).get("findings", {})
    out = [f"{k} is not recorded here" for k in BLOCK_IDENTITY if mine.get(k) is None]
    for bid, d in block.imports:
        theirs = (read_json(step_dir(Path(d), "preflight", cfg) / "result.json") or {}) \
            .get("findings", {})
        for k in BLOCK_IDENTITY:
            if theirs.get(k) is None:
                out.append(f"{k} is not recorded in block {bid}")
            elif mine.get(k) is not None and mine.get(k) != theirs.get(k):
                out.append(f"{k} is {mine.get(k)!r} here and {theirs.get(k)!r} in block {bid}")
    return out


REAL_PAIRINGS = ("P2", "P4")


def skips_lock(cfg: SessionConfig) -> bool:
    """True for a rehearsal, which runs under its own lock and spec; False for
    any other session. A P2 or P4 config marked exploratory is refused by
    name: a real pairing never skips its lock checks (adviser's ruling 3
    dated 12 Nov 2026)."""
    if not cfg.exploratory:
        return False
    if cfg.pairing in REAL_PAIRINGS:
        raise StepFailed(f"{cfg.pairing} is a real pairing: it runs from the complete lock, "
                         f"and a config marked exploratory may not skip its lock checks")
    return True


def real_block_lock_problem(cfg: SessionConfig, lock: dict, spec,
                            lock_file: Optional[Path] = None) -> Optional[str]:
    """Why a real block may not run from `lock`, or None (ruling 1d, 11 Nov
    2026): it must be a complete lock (`checklist.check`) and its block spec
    the one in `docs/lock_values.json`, the lock every other step reads."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import t31_steps as st
    from attnbench.lock import blocks as B
    from attnbench.lock import checklist
    try:
        res = checklist.check(lock, checklist.PLAN.read_text(encoding="utf-8"))
    except checklist.LockListError as e:
        return f"a real block runs from a complete lock (T4.1): {e}"
    if not res.complete:
        return (f"a real block runs from a complete lock (T4.1); this file is not one "
                f"({len(res.missing)} missing, {len(res.unset)} unset, "
                f"{len(res.mistyped)} mistyped, plan matches: {res.plan_matches})")
    p = Path(lock_file or st.LOCK_FILE)
    held = read_json(p)
    if held is None:
        return f"no lock at {p} to check the block spec against"
    try:
        theirs = B.parse(held, cfg.pairing)
    except (ValueError, B.BlockSpecError) as e:
        return f"{p.name}: {e}"
    if theirs != spec:
        return f"the block spec differs from {p.name}'s, the lock every step reads"
    return None


def block_run(cfg: SessionConfig, args) -> tuple:
    """(BlockRun, None) for `--lock --block`, or (None, why it is refused)."""
    import hashlib
    from attnbench.lock import blocks as B
    if not (args.lock and args.block):
        return None, "--lock and --block go together"
    try:
        own_spec = skips_lock(cfg)                    # a rehearsal runs its own spec
    except StepFailed as e:
        return None, str(e)
    raw = Path(args.lock).read_bytes()
    try:
        spec = B.parse(json.loads(raw), cfg.pairing)
    except (ValueError, B.BlockSpecError) as e:
        return None, str(e)
    if not args.dry_run and not own_spec:
        why = real_block_lock_problem(cfg, json.loads(raw), spec)
        if why:
            return None, why
    problems = B.check(spec, cfg.names, cfg.boot_steps, complete=not args.dry_run)
    if problems:
        return None, f"the block spec in {args.lock}: " + "; ".join(problems)
    try:
        spec.block(args.block)
    except B.BlockSpecError as e:
        return None, str(e)
    given = {}
    for d in args.import_block:
        bid = ((read_json(Path(d) / "manifest.json") or {}).get("block") or {}).get("id")
        if bid is None or bid in given:
            return None, f"--import-block {d}: not a block's session, or a block given twice"
        given[bid] = Path(d)
    need = [b.id for b in spec.before(args.block)]
    if sorted(given) != sorted(need):
        return None, (f"block {args.block} imports every earlier block once: needs {need}, "
                      f"given {sorted(given)}")
    return BlockRun(spec, args.block, spec.run_steps(args.block, cfg.names, cfg.boot_steps),
                    tuple((b, given[b]) for b in need), hashlib.sha256(raw).hexdigest()), None


def run_step(cfg: SessionConfig, name: str, args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    body = (cfg.fakes if args.dry_run else cfg.reals)[name]
    try:
        status, reason, findings = PASSED, "", body(cfg, args)
    except Skip as e:
        status, reason, findings = SKIPPED, str(e), {}
    except StepFailed as e:
        status, reason, findings = FAILED, str(e), e.findings
    except Exception as e:
        traceback.print_exc()
        status, reason, findings = FAILED, f"{type(e).__name__}: {e}", {}
    write_json(out / "result.json", {
        "step": name, "task": cfg.step(name).task, "status": status, "reason": reason,
        "dry_run": args.dry_run, "seconds": round(time.time() - started, 1),
        "findings": findings})
    print(f"[{name}] {status.upper()}{': ' + reason if reason else ''}", flush=True)
    return EXIT[status]


def add_step_options(ap: argparse.ArgumentParser, cfg: SessionConfig) -> None:
    ap.add_argument("--banked-at", action="append", default=[], metavar="PAIRING=ISO",
                    help=f"when a pairing this prediction is trained on was banked "
                         f"({cfg.pairing}: {', '.join(cfg.banked)})")
    if "fa3_import" in cfg.names:
        ap.add_argument("--fa3-sha256", default=None, metavar="HEX",
                        help="the FA3 wheel's sha256, recorded when it was built")


def parser(cfg: SessionConfig) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=cfg.title)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="the session, step by step")
    r.add_argument("--session", default=None,
                   help=f"session directory (default results/{cfg.session_prefix}_<UTC time>)")
    r.add_argument("--from", dest="start", default=cfg.names[0], choices=cfg.names)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--no-sync", action="store_true",
                   help="do not wrap steps in run_phase.sh (implied by --dry-run)")
    r.add_argument("--list", action="store_true", help="print the steps and exit")
    r.add_argument("--lock", default=None, help="the lock file holding the block spec")
    r.add_argument("--block", default=None, help="the block of the spec this session is")
    r.add_argument("--import-block", action="append", default=[], metavar="DIR",
                   help="an earlier block's session directory (every earlier block, once)")
    add_step_options(r, cfg)
    d = sub.add_parser("drift", help="the post-session step: the anchor's drift between blocks")
    d.add_argument("--lock", required=True, help="the lock file (block spec and drift bound)")
    d.add_argument("--block-dir", action="append", required=True, metavar="DIR",
                   help="a block's session directory (every block, once, discarded ones too)")
    d.add_argument("--out", required=True, help="where block_drift.json is written")
    s = sub.add_parser("step", help="one step (the runner calls this)")
    s.add_argument("name", choices=cfg.names)
    s.add_argument("--session", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--dry-run", action="store_true")
    add_step_options(s, cfg)
    return ap


def _passthrough(args, cfg: SessionConfig) -> list:
    extra = []
    for b in args.banked_at:
        extra += ["--banked-at", b]
    for opt in cfg_extra_options(cfg):
        v = getattr(args, opt.replace("-", "_"), None)
        if v is not None:
            extra += [f"--{opt}", str(v)]
    return extra


def cfg_extra_options(cfg: SessionConfig) -> tuple:
    return ("fa3-sha256",) if "fa3_import" in cfg.names else ()


def start_refusal(cfg: SessionConfig, args) -> Optional[str]:
    """What is checked before the first step, so a missing input is refused
    before any GPU time is spent rather than at the step that needs it."""
    if not args.dry_run and os.environ.get(EXIT_WRAPPER_ENV) != "armed":
        return ("a real session runs under scripts/session_exit.sh, so that its last act on "
                "any exit is the sync of its logs and manifests and then the halt")
    if not args.dry_run and not tree_clean():
        return "the tree is dirty, so no row could be banked with git_dirty = False"
    if cfg.pin_timed_path and not args.dry_run:
        why = timed_path_refusal()
        if why:
            return why
    if "fa3_import" in cfg.names:
        sha = getattr(args, "fa3_sha256", None)
        if not (sha and re.match(r"^[0-9a-f]{64}$", sha)):
            return (f"--fa3-sha256 must be the 64-hex sha256 recorded when the FA3 wheel "
                    f"was built, got {sha!r}")
    if not args.dry_run:
        try:
            banked, _ = banked_times(cfg, args)
        except (StepFailed, ValueError) as e:
            return str(e)
        missing = [p for p in cfg.banked if p not in banked]
        if missing:
            return (f"{cfg.pairing} is trained on {list(cfg.banked)}: each is banked in the "
                    f"pairing ledger first (missing: {missing})")
    return None


TIMED_PATH_ITEM = "timed_path_identity_for_p2_and_p4"


def timed_path_refusal(lock_path=None, commit: str = "HEAD") -> Optional[str]:
    """Why the deployed timed path is not the lock's, or None (T4.1 item 37;
    adviser's ruling 2 of the 11 Nov 2026 paste). Checked at session start,
    so on every boot of a block: `scripts/timed_path.py`'s hash at `commit`
    (the tree is clean, so it is the deployed code) against the lock's
    pinned value."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import t31_steps as st
    import timed_path
    p = Path(lock_path or st.LOCK_FILE)
    if not p.exists():
        return f"no lock file at {p}: the timed-path identity is a lock value (T4.1)"
    lock = read_json(p, {}) or {}
    want = ((lock.get("values") or {}).get(TIMED_PATH_ITEM) or {}).get("value")
    if not (isinstance(want, str) and re.fullmatch(r"[0-9a-f]{64}", want)):
        return f"{p.name}'s timed-path identity is {want!r}, not a 64-hex sha256"
    got = timed_path.pin_hash(commit)["sha256"]       # ruling 1f: whole files + configs
    if got != want:
        return (f"the timed path at {commit} hashes to {got}, the lock pins {want}: "
                f"`python3 scripts/timed_path.py <locked commit> {commit} --pin` names what changed")
    return None


def drift_step(cfg: SessionConfig, args) -> int:
    """The post-session step (adviser's ruling 2c of 11 Nov 2026): after the
    last block, the anchor compared between the blocks that ran ground truth
    (`block_drift.pairing_drift`), against the lock's `block_drift_bound`,
    from a complete lock whose spec is `docs/lock_values.json`'s. Writes
    `block_drift.json`, which the analysis reads; a flagged block is a flag
    (0), a refusal is 2."""
    import hashlib
    from attnbench.analysis import block_drift as BD
    from attnbench.lock import blocks as B
    try:
        if skips_lock(cfg):
            raise StepFailed("the rehearsal's drift is read by t61_rehearsal_b.sh at the "
                             "working bound; this step is a real pairing's")
        lock = json.loads(Path(args.lock).read_bytes())
        spec = B.parse(lock, cfg.pairing)
        why = real_block_lock_problem(cfg, lock, spec)
        if why:
            raise StepFailed(why)
        out = BD.pairing_drift(args.block_dir, spec, bound=BD.lock_bound(lock),
                               ground_truth_dir=step_dir(Path(), "ground_truth", cfg).name)
    except (StepFailed, OSError, ValueError, B.BlockSpecError, BD.BlockDriftError) as e:
        print(f"REFUSING: {e}", file=sys.stderr)
        return 2
    write_json(Path(args.out), {**out, "pairing": cfg.pairing, "lock_sha256":
                                hashlib.sha256(Path(args.lock).read_bytes()).hexdigest()})
    d = out["drift"]
    print(f"{cfg.pairing} block drift: {d['status']}; above the bound "
          f"{d.get('bound')}: {d.get('above') or 'none'}")
    return 0


def main(argv=None, cfg: Optional[SessionConfig] = None) -> int:
    cfg = cfg or T61
    args = parser(cfg).parse_args(argv)
    if args.cmd == "step":
        return run_step(cfg, args.name, args)
    if args.cmd == "drift":
        return drift_step(cfg, args)
    if args.list:
        for i, s in enumerate(cfg.steps, 1):
            print(f"{i:2d} {s.name:18s} {s.task:12s} {s.what}")
        return 0
    block = None
    if args.lock or args.block or args.import_block:
        block, why = block_run(cfg, args)
        if why:
            print(f"REFUSING: {why}", file=sys.stderr)
            return 2
    if args.session is None:
        if args.start != cfg.names[0]:
            print("REFUSING: --from needs the --session it continues", file=sys.stderr)
            return 2
        args.session = str(ROOT / "results" /
                           f"{cfg.session_prefix}_{time.strftime('%Y%m%dT%H%MZ', time.gmtime())}"
                           f"{'_' + block.id if block else ''}")
    why = start_refusal(cfg, args)
    if why:
        print(f"REFUSING: {why}", file=sys.stderr)
        return 2
    return run_session(cfg, Path(args.session), start=args.start, extra=_passthrough(args, cfg),
                       dry_run=args.dry_run, sync=not (args.dry_run or args.no_sync), block=block)


T61 = _t61()

if __name__ == "__main__":
    raise SystemExit(main())
