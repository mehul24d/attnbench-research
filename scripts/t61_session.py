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

from attnbench.predict.commit import (PLANNED_SYNC_WAIT_S,  # noqa: E402
                                      REQUIRED_SYNC_GAP_S)

# Plan Section 5.1: the workload starts at least the required gap after the
# sync, and the analysis allows the same gap as skew between the store's clock
# and the instance's. Both come from predict/commit.py, the one source; the
# runner holds for the planned wait, the gap plus 30 s (audit C2, finding 1).
MIN_GAP_S = REQUIRED_SYNC_GAP_S
CLOCK_SKEW_S = REQUIRED_SYNC_GAP_S
STRICT_KEY = "first_clocked_block_started_at"   # Appendix A, 27 Oct 2026
OLD_KEY = "first_timed_block_started_at"        # the 25 Oct 2026 rule; refused

RECEIPT = "prediction_receipt.json"
OPENED = "ground_truth_opened.json"
STAMPS = "ground_truth_stamps.json"
DRY_CLOCK = "dry_clock.json"


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
    """A plan task whose Section 8 status must start with "Done"."""
    task: str
    why: str


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
        return DirStore(session / "dry_store", lambda: now(session, True)), DirStore.PREFIX
    from attnbench.predict.commit import GsutilStore
    return GsutilStore(), results_bucket() + "/"


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


def unmet(requirements, statuses: dict) -> list:
    rows = []
    for r in requirements:
        s = statuses.get(r.task)
        if s is None or not s.startswith("Done"):
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
    rows = unmet(cfg.requirements, plan_statuses(text))
    files = missing_files(cfg.files)
    findings = {"unmet": rows, "missing_files": files,
                "required": [r.task for r in cfg.requirements]}
    if rows or files:
        msg = (f"{len(rows)} task(s) not Done ({', '.join(r['task'] for r in rows)})"
               + (f"; {len(files)} file check(s) failed" if files else ""))
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


def real_preflight(cfg: SessionConfig, args) -> dict:
    """T2.20 provenance and the T2.1 host-speed probe at the start, read from
    the probe's own row; Triton imports (the XAttention code path here)."""
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
    findings = {"provisioning_model": row.get("provisioning_model"),
                "instance_type": row.get("instance_type"), "gpu_name": row.get("gpu_name"),
                "null_launch_us_median": _us(row.get("null_launch_s_median")),
                "triton": tri.stdout.strip() or None, "problems": problems}
    if cfg.preflight_findings is not None:
        findings.update(cfg.preflight_findings(row))
    if problems:
        raise StepFailed(f"{len(problems)} preflight problem(s): {problems[:3]}", findings)
    return findings


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
    findings = {"provisioning_model": row["provisioning_model"], "fake": True}
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


def real_link(cfg, args) -> dict:
    """T2.5: every field of the link row, one provenance-stamped row."""
    out = Path(args.out)
    if run(py("measure_link.py", "--out", str(out))):
        raise StepFailed("measure_link.py failed")
    import pandas as pd
    row = pd.read_parquet(out / "link.parquet").iloc[0].to_dict()
    empty = sorted(k for k, v in row.items()
                   if (k.startswith("s_fixed_") or k.startswith("bw_")) and v is None)
    findings = {"empty_fields": empty, "gpu_numa_node": row.get("gpu_numa_node"),
                "affinity_on_gpu_node": row.get("affinity_on_gpu_node")}
    if empty:
        raise StepFailed(f"link row has empty fields: {empty}", findings)
    return findings


def real_launch_queue(cfg, args) -> dict:
    """T2.6: Q, from the null kernel (Appendix A, 25 Oct 2026)."""
    out = Path(args.out)
    if run(py("measure_launch_queue.py", "--out", str(out))):
        raise StepFailed("measure_launch_queue.py failed")
    import pandas as pd
    row = pd.read_parquet(out / "launch_queue.parquet").iloc[0].to_dict()
    if row.get("sleep0_q") is None:
        raise StepFailed("the null kernel's Q was not agreed across trials")
    return {"q": int(row["sleep0_q"])}


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


def real_record(cfg: SessionConfig, args, written_at: datetime) -> dict:
    """Assembly of every cell from the session's measured components, the
    fitted baselines trained on the banked pairings, the intervals and the
    dispatcher's choice. Not built: T2.11 is Built, but no code assembles a
    pairing's cells from a session's component files."""
    banked = parse_banked_at(args.banked_at)
    missing = [p for p in cfg.banked if p not in banked]
    if missing:
        raise StepFailed(f"{cfg.pairing} is trained on {list(cfg.banked)}; give each one's "
                         f"banking time with --banked-at (missing: {missing})")
    raise NotBuilt("T2.11, T2.9, T2.10, T2.19, T2.22",
                   "assembling the pairing's cells from the session's components")


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
    committed = C.write_prediction(record, Path(args.out))
    store, prefix = store_for(args)
    rel = f"predictions/{cfg.pairing}/{session.name}"
    receipt = C.sync_prediction(committed, store, prefix + rel)
    out = {"pairing": cfg.pairing, "task": cfg.task, "sha256": committed.sha256,
           "local": str(committed.path), "remote_path": f"{rel}/{committed.path.name}",
           "synced_at": receipt.synced_at.isoformat(timespec="microseconds"),
           "written_at": record["written_at"], "store": "dir" if args.dry_run else "gcs",
           "dry_run": args.dry_run}
    write_json(session / RECEIPT, out)
    write_json(Path(args.out) / RECEIPT, out)
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

    def __call__(self, cfg, args) -> dict:
        first = now(self.session, True)
        advance(self.session, True, self.warmup_s + self.rounds_s)
        rows = [{"arm": a, "context_length": 2048, "batch": 1, "median_ms": m, "fake": True}
                for a, m in (("dense", 4.0), ("xattention", 2.5))]
        return {STRICT_KEY: first, "complete": True, "untraced": {"rows": rows},
                "traced": {"rows": rows}, "overhead": {"fake": True}, "traces": []}


def real_harness(cfg, args) -> dict:
    raise NotBuilt("T2.13, with the T2.12, T2.17 and T2.21 arms",
                   "the ground-truth grid driver: every arm, density, length and batch, "
                   "untraced then traced, with the dense control")


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
    result = harness(cfg, args)
    ended = now(session, args.dry_run)
    stamp = result.get(STRICT_KEY)
    if isinstance(stamp, (int, float)) and not isinstance(stamp, bool):
        append_json(session / STAMPS, {"stamp": iso(float(stamp)), "opened_at": iso(opened_at)})
    write_json(Path(args.out) / "ground_truth.json", result)
    findings = check_stamp(result, synced_at=synced, opened_at=opened_at, ended_at=ended,
                           earlier_start=earlier)
    if result.get("complete") is not True:
        raise StepFailed("the ground truth is incomplete: flagged rounds exceeded the reruns",
                         findings)
    write_json(Path(args.out) / "traces.json", result.get("traces", []))
    return findings


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
    banked = parse_banked_at(args.banked_at)
    try:
        cp = C.load_committed(store, prefix + receipt["remote_path"],
                              ground_truth_started_at=datetime.fromtimestamp(start, tz=timezone.utc),
                              clock_skew=timedelta(seconds=CLOCK_SKEW_S),
                              banked_at=banked, local=receipt["local"])
    except C.PredictionCommitError as e:
        raise StepFailed(f"the analysis would refuse this prediction: {e}")
    if cp.sha256 != receipt["sha256"]:
        raise StepFailed(f"the store holds {cp.sha256}, the receipt says {receipt['sha256']}")
    return {"sha256": cp.sha256,
            "synced_at": receipt["synced_at"], "session_start": iso(start),
            "gap_s": start - epoch(receipt["synced_at"]), "clock_skew_s": CLOCK_SKEW_S}


def real_trace_checks(cfg, args) -> dict:
    """T2.7 on every trace the traced ground-truth pass wrote."""
    session = Path(args.session)
    gt = step_dir(session, "ground_truth", cfg)
    traces = read_json(gt / "traces.json", None)
    if not traces:
        raise StepFailed(f"no traces listed in {gt / 'traces.json'}")
    bad = [t for t in traces
           if run(py("check_trace.py", "kineto", t, "--out", str(Path(args.out) / Path(t).stem)))]
    if bad:
        raise StepFailed(f"check_trace.py flagged or failed on {len(bad)} trace(s)", {"bad": bad})
    return {"traces": len(traces)}


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
    Requirement("T2.21", "compiled variants"),
    Requirement("T2.22", "the dispatcher's choice in the committed file (H6)"),
    Requirement("T2.26", "the GPU suite record run by the launch scripts"),
)

GATES = (
    Requirement("T3.1", "P1 banked: P2 is trained on it"),
    Requirement("T3.3", "Gate A recorded"),
    Requirement("T4.4", "Gate B: the tagged lock commit"),
)

BUILT_FILES = (
    # the strict first-clocked-block stamp (main, PR #20); absent on this branch's base
    ("attnbench/ground_truth.py", STRICT_KEY),
    ("attnbench/predict/commit.py", "def load_committed"),
)

COMMON_STEPS_HEAD = (
    Step("prereqs", "Section 8", "plan rows Done; deployed files present"),
    Step("preflight", "T2.20, T2.1", "provenance on the probe's row; host speed (start); Triton"),
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
        "probe": not_built("T2.18", "the structural probe over every arm on this code path"),
        "host_components": not_built("T2.2", "the host components of every arm and cell"),
        "device_components": not_built("T2.4", "the replay of every arm and cell, with the "
                                               "replay-versus-trace check of record"),
        "dense_select": not_built("T2.16", "the K2 selection over the session's cells"),
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
    reals["mask_gate"] = not_built("T2.23", "the mask gate: both XAttention paths through "
                                            "use_triton on the recorded prompts")
    fakes["mask_gate"] = fake_files("mask_gate.json", verdict="pass")
    cfg = SessionConfig(
        task="T6.1", pairing="P2", title="The P2 session (T6.1: Cascade Lake + A100)",
        script="t61_session.py", phase_prefix="t61", session_prefix="t61_p2",
        gpu="A100", instance_type="a2-ultragpu-1g", provisioning=("FLEX_START", "SPOT"),
        code_path={"gpu": "A100", "implementation": "xattention:triton",
                   "libraries": [["torch", "dry"], ["triton", "dry"]], "python_optimize": 0},
        banked=("P1",),
        steps=COMMON_STEPS_HEAD + (
            Step("mask_gate", "T2.23", "mask gate: Triton against PyTorch on one card; "
                                       "masks recorded untimed if it fails"),
        ) + COMMON_STEPS_COMPONENTS + COMMON_STEPS_TAIL,
        requirements=PHASE2 + (Requirement("T2.23", "the mask gate on P2"),) + GATES,
        files=BUILT_FILES, fakes=fakes, reals=reals)
    validate(cfg)
    return cfg


# --- the runner -------------------------------------------------------------------------------

def step_dir(session: Path, step: str, cfg: SessionConfig) -> Path:
    return Path(session) / f"{cfg.names.index(step) + 1:02d}_{step}"


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


def refuse_start(cfg: SessionConfig, start: str, manifest: dict, dry_run: bool) -> Optional[str]:
    """Why `--from start` is refused in this session, or None."""
    done = last_status(manifest)
    first = cfg.names.index(start)
    if done.get("predict") == PASSED and first <= cfg.names.index("predict"):
        return (f"predict has passed in this session, so --from {start} would make a second "
                f"prediction for {cfg.pairing} (or recompute what it was made from); start a "
                f"new session")
    unpassed = [s for s in cfg.names[:first] if done.get(s) != PASSED]
    if unpassed and not (dry_run and all(done.get(s) == SKIPPED for s in unpassed)):
        return f"earlier steps have not passed in this session: {unpassed}"
    return None


def run_session(cfg: SessionConfig, session: Path, *, start: str, extra: list, dry_run: bool,
                sync: bool, runner=run_logged) -> int:
    session = Path(session)
    session.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(session)
    manifest.setdefault("dry_run", dry_run)
    manifest.setdefault("task", cfg.task)
    if manifest["dry_run"] != dry_run or manifest["task"] != cfg.task:
        print(f"REFUSING: {session} is a {'dry' if manifest['dry_run'] else 'real'} "
              f"{manifest['task']} session", file=sys.stderr)
        return 2
    why = refuse_start(cfg, start, manifest, dry_run)
    if why:
        print(f"REFUSING --from {start}: {why}", file=sys.stderr)
        return 2
    done = last_status(manifest)
    total = len(cfg.steps)
    for step in cfg.names[cfg.names.index(start):]:
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
        if status == FAILED:
            print(f"\nSTOPPED at step {n} ({step}). Log: {d / 'step.log'}\n"
                  f"After the fix: python3 scripts/{cfg.script} run --from {step} "
                  f"--session {session}", flush=True)
            return 1
    statuses = last_status(manifest)
    print(f"\n{cfg.title}: " + ", ".join(f"{s} {statuses.get(s)}" for s in cfg.names))
    return 0


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
    add_step_options(r, cfg)
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
    if not args.dry_run and not tree_clean():
        return "the tree is dirty, so no row could be banked with git_dirty = False"
    if "fa3_import" in cfg.names:
        sha = getattr(args, "fa3_sha256", None)
        if not (sha and re.match(r"^[0-9a-f]{64}$", sha)):
            return (f"--fa3-sha256 must be the 64-hex sha256 recorded when the FA3 wheel "
                    f"was built, got {sha!r}")
    if not args.dry_run:
        try:
            banked = parse_banked_at(args.banked_at)
        except (StepFailed, ValueError) as e:
            return str(e)
        missing = [p for p in cfg.banked if p not in banked]
        if missing:
            return (f"{cfg.pairing} is trained on {list(cfg.banked)}: give each banking "
                    f"time with --banked-at PAIRING=ISO (missing: {missing})")
    return None


def main(argv=None, cfg: Optional[SessionConfig] = None) -> int:
    cfg = cfg or T61
    args = parser(cfg).parse_args(argv)
    if args.cmd == "step":
        return run_step(cfg, args.name, args)
    if args.list:
        for i, s in enumerate(cfg.steps, 1):
            print(f"{i:2d} {s.name:18s} {s.task:12s} {s.what}")
        return 0
    if args.session is None:
        if args.start != cfg.names[0]:
            print("REFUSING: --from needs the --session it continues", file=sys.stderr)
            return 2
        args.session = str(ROOT / "results" /
                           f"{cfg.session_prefix}_{time.strftime('%Y%m%dT%H%MZ', time.gmtime())}")
    why = start_refusal(cfg, args)
    if why:
        print(f"REFUSING: {why}", file=sys.stderr)
        return 2
    return run_session(cfg, Path(args.session), start=args.start, extra=_passthrough(args, cfg),
                       dry_run=args.dry_run, sync=not (args.dry_run or args.no_sync))


T61 = _t61()

if __name__ == "__main__":
    raise SystemExit(main())
