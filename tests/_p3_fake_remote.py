"""The fake remote for scripts/aws_p3_driver.sh (tests/test_aws_p3_driver.py).

Nothing here calls AWS or opens a network connection. The Mac side is the
real driver, run from a fresh git repository holding this working tree's
files, with fakes first on PATH: `ssh` and `scp` that run the remote command
in a local directory standing in for the instance's home (`home/`), `aws`
(the driver's describe calls, the instance's state), `curl` (this Mac's
address); the launcher, teardown, cleanup check and bootstrap are stand-ins
that record what they were given. The instance side is real: the deployed
bundle checked out in `home/attnbench_scaffold`, `scripts/aws_p3_instance.sh`,
the real `scripts/t31_session.py` on `configs/t31/p3_session.json` with its
real config load (the lock's values), and `scripts/aws_p3_gsutil.py` as
`gsutil` under the real `commit.GsutilStore`. Only the steps' bodies are
stubs (`STUB_STEPS` below), redirected by a `sitecustomize` when t31_session
runs `scripts/t31_steps.py`: each writes a result, `host_components` writes
P1's 35 cells' bare prices scaled by FAKE_HOST_SCALE, and `predict` and
`delay_injection` sync through the real `GsutilStore`.

Network drops: `ssh` and `scp` fail while `<run>/net_down` exists. It is
created at the FAKE_DROP_AT-th poll of block FAKE_DROP_BLOCK, and removed
after FAKE_DROP_FAILS failed calls (0: never; the network does not come back).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P1_HOST = (Path.home() / "Desktop/research/attnbench_backups/bucket_20261117/"
           "attnbench-l4-t32d-20261109-1237/results/t32d_p1_20261109T0726Z/05_host_components")
DNS = "ec2-fake.ap-southeast-2.compute.amazonaws.com"
INSTANCE = "i-0fa0e0000000000aa"
MY_IP = "203.0.113.7"

FAKE_SSH = r'''#!/usr/bin/env bash
printf '%q ' ssh "$@" >> "$FAKE_RUN/ssh_calls.log"; echo >> "$FAKE_RUN/ssh_calls.log"
key=""; batch=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i) key="$2"; shift 2;;
    -o) [[ "$2" == BatchMode=yes ]] && batch=1; shift 2;;
    -p|-F|-l) shift 2;;
    -*) shift;;
    *) break;;
  esac
done
host="$1"; shift
cmd="$*"
case "$cmd" in *"cat ~/p3/${FAKE_DROP_BLOCK:-none}.exit"*)
  n=$(( $(cat "$FAKE_RUN/drop_polls" 2>/dev/null || echo 0) + 1 )); echo $n > "$FAKE_RUN/drop_polls"
  [[ $n -eq ${FAKE_DROP_AT:-0} ]] && touch "$FAKE_RUN/net_down";;
esac
if [[ -e "$FAKE_RUN/net_down" ]]; then
  f=$(( $(cat "$FAKE_RUN/net_fails" 2>/dev/null || echo 0) + 1 )); echo $f > "$FAKE_RUN/net_fails"
  [[ ${FAKE_DROP_FAILS:-0} -gt 0 && $f -ge ${FAKE_DROP_FAILS:-0} ]] && rm -f "$FAKE_RUN/net_down"
  echo "ssh: connect to host $DNS port 22: Operation timed out (fake)" >&2; exit 255
fi
[[ -n "$key" && -f "$key" ]] || { echo "fake ssh: no key file" >&2; exit 255; }
[[ $batch -eq 1 ]] || { echo "fake ssh: BatchMode=yes missing" >&2; exit 255; }
[[ "$host" == "ubuntu@$FAKE_DNS" ]] || { echo "fake ssh: unknown host '$host'" >&2; exit 255; }
[[ -n "$cmd" ]] || { echo "fake ssh: no command" >&2; exit 255; }
cmd="${cmd//\/usr\/bin\/python3/$FAKE_PY}"
cd "$FAKE_HOME" || exit 255
exec env -i HOME="$FAKE_HOME" USER=ubuntu LANG=en_US.UTF-8 \
  PATH="$FAKE_ROOT/bin_remote:/usr/bin:/bin:/usr/sbin:/sbin" \
  ATTNBENCH_PYTHON="$FAKE_PY" FAKE_RUN="$FAKE_RUN" FAKE_ROOT="$FAKE_ROOT" FAKE_VPY="$FAKE_VPY" \
  FAKE_P1_HOST="$FAKE_P1_HOST" FAKE_HOST_SCALE="${FAKE_HOST_SCALE:-1.05}" \
  FAKE_FAIL_STEP="${FAKE_FAIL_STEP:-}" FAKE_STEP_SLEEP_S="${FAKE_STEP_SLEEP_S:-0}" \
  FAKE_BACKEND_FAIL="${FAKE_BACKEND_FAIL:-}" \
  ATTNBENCH_P3_SHUTDOWN_SCHEDULED="$FAKE_RUN/shutdown_scheduled" ATTNBENCH_P3_BOOT_EPOCH="$FAKE_BOOT_EPOCH" \
  ATTNBENCH_P3_WATCH_S=0.5 \
  bash -c "$cmd"
'''

FAKE_SCP = r'''#!/usr/bin/env bash
printf '%q ' scp "$@" >> "$FAKE_RUN/ssh_calls.log"; echo >> "$FAKE_RUN/ssh_calls.log"
if [[ -e "$FAKE_RUN/net_down" ]]; then
  f=$(( $(cat "$FAKE_RUN/net_fails" 2>/dev/null || echo 0) + 1 )); echo $f > "$FAKE_RUN/net_fails"
  [[ ${FAKE_DROP_FAILS:-0} -gt 0 && $f -ge ${FAKE_DROP_FAILS:-0} ]] && rm -f "$FAKE_RUN/net_down"
  echo "ssh: connect to host $FAKE_DNS port 22: Operation timed out (fake)" >&2; exit 1
fi
key=""; rec=0; args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i) key="$2"; shift 2;;
    -o|-P|-F|-l) shift 2;;
    -r) rec=1; shift;;
    -*) shift;;
    *) args+=("$1"); shift;;
  esac
done
[[ -n "$key" && -f "$key" ]] || { echo "fake scp: no key file" >&2; exit 1; }
map() {
  local a="$1"
  if [[ "$a" == *@*:* ]]; then
    local h="${a%%:*}" p="${a#*:}"
    [[ "$h" == "ubuntu@$FAKE_DNS" ]] || { echo "fake scp: unknown host $h" >&2; return 1; }
    p="${p/#\~/$FAKE_HOME}"; [[ "$p" == /* ]] || p="$FAKE_HOME/$p"; printf '%s' "$p"
  else printf '%s' "$a"; fi
}
n=${#args[@]}; dest="$(map "${args[n-1]}")" || exit 1
rc=0
for ((i=0; i<n-1; i++)); do
  src="$(map "${args[i]}")" || exit 1
  shopt -s nullglob; set -- $src; shopt -u nullglob
  [[ $# -gt 0 ]] || { echo "scp: $src: No such file or directory" >&2; rc=1; continue; }
  for s in "$@"; do
    if [[ -d "$s" && $rec -eq 0 ]]; then echo "scp: $s: not a regular file" >&2; rc=1; continue; fi
    cp -R "$s" "$dest" || rc=1
  done
done
[[ "${args[*]}" == *".tar"* ]] && sleep "${FAKE_SCP_SLEEP_S:-0}"
exit $rc
'''

FAKE_AWS = r'''#!/usr/bin/env bash
printf '%q ' aws "$@" >> "$FAKE_RUN/aws_calls.log"; echo >> "$FAKE_RUN/aws_calls.log"
case "$*" in
  *describe-security-groups*) cat "$FAKE_RUN/sg_cidrs";;
  *"wait instance-running"*) :;;
  *PublicDnsName*) echo "$FAKE_DNS";;
  *Placement.AvailabilityZone*) echo ap-southeast-2a;;
  *InstanceType*) echo "${FAKE_ITYPE:-g6.2xlarge}";;
  *State.Name*) cat "$FAKE_RUN/state" 2>/dev/null || echo running;;
  *) echo "fake aws: unexpected call: $*" >&2; exit 2;;
esac
'''

FAKE_CURL = r'''#!/usr/bin/env bash
echo "curl $*" >> "$FAKE_RUN/aws_calls.log"; cat "$FAKE_RUN/my_ip"
'''

# The launcher's stand-in: what the real one prints, and every setting it was given.
FAKE_LAUNCH = r'''#!/usr/bin/env bash
env | grep -E '^AWS_(SESSION|RATE|COST|HALT|MAX_RUN|LAUNCH_WINDOW|AMI)' | sort > "$FAKE_RUN/launch_env.txt"
read -r typed; echo "typed=$typed" >> "$FAKE_RUN/launch_env.txt"; echo "args=$*" >> "$FAKE_RUN/launch_env.txt"
echo "Pass 1 of at most 36 (window 180 min, a pass every 5 min)."
echo "WINDOW_PASSES=${FAKE_PASSES:-2}"
echo "CAPACITY_REFUSALS=${FAKE_REFUSALS:-3}"
if [[ "${FAKE_LAUNCH_MODE:-ok}" == refused ]]; then
  echo "ZONES_REFUSED=ap-southeast-2b,ap-southeast-2c,ap-southeast-2a"
  echo "LAUNCH_RESULT=EVERY_ZONE_REFUSED_FOR_CAPACITY"; exit 1
fi
echo "Created $FAKE_INSTANCE. Billing now. Setting the hard delete."
echo "INSTANCE_ID=$FAKE_INSTANCE"
echo "LAUNCH_ZONE=ap-southeast-2a"
'''

FAKE_TEARDOWN = r'''#!/usr/bin/env bash
echo "fake teardown: $*  rate=$AWS_RATE_INR_HR" >> "$FAKE_RUN/teardown.log"
mkdir -p "$3"; echo terminated > "$FAKE_RUN/state"
'''
FAKE_CLEANUP = '#!/usr/bin/env bash\necho "fake cleanup check: nothing billing" >> "$FAKE_RUN/cleanup.log"\n'
FAKE_BOOTSTRAP = '#!/usr/bin/env bash\necho "fake bootstrap $*" >> "$FAKE_RUN/bootstrap.log"\necho BOOTSTRAP_OK\n'

# The instance's commands that macOS lacks or that must not act.
REMOTE = {
    "sha256sum": '#!/usr/bin/env bash\nexec shasum -a 256 "$@"\n',
    "sudo": '#!/usr/bin/env bash\necho "$(date -u +%FT%TZ) sudo $*" >> "$FAKE_RUN/sudo.log"\n',
    "logger": '#!/usr/bin/env bash\necho "logger $*" >> "$FAKE_RUN/logger.log"\n',
    "nvidia-smi": '#!/usr/bin/env bash\necho "GPU 0: NVIDIA L4 (fake)"\n',
    "nproc": '#!/usr/bin/env bash\necho 8\n',
    "free": '#!/usr/bin/env bash\necho "Mem: 31 1 30"\n',
    # The image's /usr/bin/python3: the version guard answers 3.10.12 (the Mac's venv
    # is 3.12); everything else is the Mac's venv, with the stub steps' sitecustomize.
    "fakepy": r'''#!/usr/bin/env bash
printf '%q ' python3 "$@" >> "$FAKE_RUN/py_calls.log"; echo >> "$FAKE_RUN/py_calls.log"
if [[ $# -eq 2 && "$1" == -c && "$2" == 'import platform; print(platform.python_version())' ]]; then
  echo 3.10.12; exit 0
fi
export PYTHONPATH="$FAKE_ROOT/site${PYTHONPATH:+:$PYTHONPATH}" FAKE_STUB="$FAKE_ROOT/stub_steps.py"
exec "$FAKE_VPY" "$@"
''',
}

SITECUSTOMIZE = '''"""The fake remote: t31_session's `scripts/t31_steps.py STEP` runs the stub, and
P6's `scripts/p6_backend_check.py` writes a fake report (failed when
FAKE_BACKEND_FAIL is set; no GPU here)."""
import json, os, runpy, sys
if sys.argv and sys.argv[0].endswith("scripts/p6_backend_check.py") and os.environ.get("FAKE_STUB"):
    out = sys.argv[sys.argv.index("--out") + 1]
    bad = os.environ.get("FAKE_BACKEND_FAIL")
    rep = {"fake": True, "gpu": {"name": "NVIDIA H100 80GB HBM3 (fake)"}, "priced_dense_kernel": "sdpa_flash",
           "problems": [bad] if bad else [], "passed": not bad}
    open(out, "w").write(json.dumps(rep) + "\\n")
    print(("P6_BACKEND_CHECK FAILED: " + bad) if bad else "P6_BACKEND_CHECK PASSED (fake)")
    sys.stdout.flush()
    os._exit(1 if bad else 0)
if sys.argv and sys.argv[0].endswith("scripts/t31_steps.py") and os.environ.get("FAKE_STUB"):
    stub = os.environ.pop("FAKE_STUB")
    sys.argv[0] = stub
    code = 0
    try:
        runpy.run_path(stub, run_name="__main__")
    except SystemExit as e:      # an exit in sitecustomize is fatal to the interpreter
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)
'''

STUB_STEPS = r'''"""Stub step bodies for the fake remote (tests/_p3_fake_remote.py)."""
import argparse, hashlib, json, os, shutil, sys, time
from datetime import datetime, timezone
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("step"); ap.add_argument("--out", required=True); ap.add_argument("--session", required=True)
ap.add_argument("--config"); ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
out, session = Path(a.out), Path(a.session)
out.mkdir(parents=True, exist_ok=True)
time.sleep(float(os.environ.get("FAKE_STEP_SLEEP_S", "0")))
cfg = json.loads(Path(a.config).read_text())
assert cfg["pairing"] in ("P3", "P6"), cfg["pairing"]
findings = {"stub": a.step}
sys.path.insert(0, os.getcwd())


def store_put(name, data):
    from attnbench.predict import commit as C
    store = C.GsutilStore()
    base = os.environ["ATTNBENCH_BUCKET"]
    local = out / name
    local.write_bytes(data)
    remote = f"{base}/{cfg['prediction_prefix']}/{session.name}/{name}"
    store.put(local, remote)
    assert store.get(remote) == data, "read-back differs"
    return remote, store.created(remote)


if a.step == "host_components":
    k = float(os.environ.get("FAKE_HOST_SCALE", "1.05"))
    for f in sorted(Path(os.environ["FAKE_P1_HOST"]).glob("*/bare_prices.json")):
        j = json.loads(f.read_text())
        for v in j["operators"].values():
            v["bare_s"] = v["bare_s"] * k
        j["host_total_s"] = j["host_total_s"] * k
        (out / f.parent.name).mkdir(exist_ok=True)
        (out / f.parent.name / "bare_prices.json").write_text(json.dumps(j))
    findings["cells"] = len(list(out.glob("*/bare_prices.json")))
elif a.step == "predict":
    from attnbench.predict import commit as C
    pn = cfg["pairing"]
    rec = json.dumps({"pairing": pn, "stub": True, "at": datetime.now(timezone.utc).isoformat()}).encode()
    p = out / f"prediction_{pn}_stub.json"; p.write_bytes(rec)
    side = out / f"prediction_{pn}_stub.json.sha256"; side.write_text(hashlib.sha256(rec).hexdigest() + "\n")
    receipt = C.sync_prediction(C.CommittedFile(p, side, hashlib.sha256(rec).hexdigest()), C.GsutilStore(),
                                f"{os.environ['ATTNBENCH_BUCKET']}/{cfg['prediction_prefix']}/{session.name}")
    (out / "receipt.json").write_text(json.dumps(receipt.to_dict()))
    findings["synced_at"] = receipt.to_dict()["synced_at"]
elif a.step == "delay_injection":
    remote, created = store_put("leg3_plan.json", b'{"plan": "stub"}\n')
    findings["plan_created"] = created.isoformat()
(out / "rows.txt").write_text(f"{a.step} stub rows\n")
failed = os.environ.get("FAKE_FAIL_STEP") == a.step
(out / "result.json").write_text(json.dumps({
    "step": a.step, "status": "failed" if failed else "passed",
    "reason": "stub failure" if failed else "", "dry_run": False, "findings": findings}))
sys.exit(1 if failed else 0)
'''


def _write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)
    return path


def repo_copy(dest: Path) -> Path:
    """This working tree's files (tracked and new, as they are now) committed
    in a fresh repository: the driver deploys a clean HEAD."""
    files = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.split("\n")
    for f in filter(None, files):
        src = ROOT / f
        if src.is_file():
            (dest / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / f)
    g = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run(["git", "init", "-q", str(dest)], check=True)
    subprocess.run([*g, "-C", str(dest), "add", "-A"], check=True)
    subprocess.run([*g, "-C", str(dest), "commit", "-q", "-m", "fake P3 commit"], check=True)
    return dest


def build(tmp: Path, halt_min: int = 473, **opts) -> dict:
    """The fake remote in `tmp`; returns the paths and the environment."""
    fr = tmp / "fr"
    run = tmp / "run"
    home = run / "home"
    for d in (fr, run, home):
        d.mkdir(parents=True, exist_ok=True)
    repo = repo_copy(tmp / "repo")
    subprocess.run(["git", "clone", "-q", str(repo), str(home / "attnbench_scaffold")], check=True)
    for name, body in (("ssh", FAKE_SSH), ("scp", FAKE_SCP), ("aws", FAKE_AWS), ("curl", FAKE_CURL)):
        _write(fr / "bin_local" / name, body)
    for name, body in REMOTE.items():
        _write(fr / "bin_remote" / name, body)
    _write(fr / "site" / "sitecustomize.py", SITECUSTOMIZE)
    _write(fr / "stub_steps.py", STUB_STEPS)
    launch = _write(fr / "fake_launch.sh", FAKE_LAUNCH)
    teardown = _write(fr / "fake_teardown.sh", FAKE_TEARDOWN)
    cleanup = _write(fr / "fake_cleanup.sh", FAKE_CLEANUP)
    bootstrap = _write(fr / "fake_bootstrap.sh", FAKE_BOOTSTRAP)
    key = run / "key"
    key.write_text("")
    key.chmod(0o600)
    (run / "aws_env.sh").write_text(
        f'export ATTNBENCH_AWS_ACCOUNT=000000000000 AWS_SSH_KEY="{key}" AWS_SECURITY_GROUP_ID=sg-0fake\n')
    (run / "my_ip").write_text(MY_IP + "\n")
    (run / "sg_cidrs").write_text(f"{MY_IP}/32\n")
    # The launcher's user data: the planned halt, pending from boot (+473 min; P6 +312).
    # The fake `sudo` only records, so it stays the pending shutdown throughout.
    boot = int(time.time())
    (run / "shutdown_scheduled").write_text(f"USEC={(boot + halt_min * 60) * 1_000_000}\nMODE=poweroff\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("P3_", "AWS_", "FAKE_"))}
    env.update({
        "PATH": f"{fr / 'bin_local'}:{env['PATH']}", "TMPDIR": str(run),
        "FAKE_ROOT": str(fr), "FAKE_RUN": str(run), "FAKE_HOME": str(home), "FAKE_DNS": DNS,
        "FAKE_VPY": sys.executable, "FAKE_PY": str(fr / "bin_remote" / "fakepy"),
        "FAKE_P1_HOST": str(P1_HOST), "FAKE_INSTANCE": INSTANCE, "FAKE_BOOT_EPOCH": str(boot),
        "P3_DRY": "1", "P3_REC_BASE": str(run / "rec"), "P3_AWS_ENV": str(run / "aws_env.sh"),
        "P3_WINDOWS": str(run / "windows.tsv"), "P3_P1_HOST": str(P1_HOST), "P3_VPY": sys.executable,
        "P3_POLL_S": "1", "P3_LAUNCH": str(launch), "P3_TEARDOWN": str(teardown),
        "P3_CLEANUP": str(cleanup), "P3_BOOTSTRAP": str(bootstrap),
        "P3_LOGIN_CACHE": str(run / "login_cache"),
        "ATTNBENCH_LOGIN_AT": str(int(time.time())),      # as the launch command stamps it
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    (run / "login_cache").mkdir()
    env.update({k: str(v) for k, v in opts.items()})
    return {"repo": repo, "run": run, "home": home, "env": env}


def run_driver(fake: dict, *args: str, timeout: float = 240) -> subprocess.CompletedProcess:
    p = subprocess.run(["bash", str(fake["repo"] / "scripts" / "aws_p3_driver.sh"), *args],
                       cwd=fake["repo"], env=fake["env"], capture_output=True, text=True,
                       timeout=timeout)
    time.sleep(0.2)      # the driver's log pipe (sed) drains after it exits
    return p


def rec_dir(fake: dict) -> Path:
    recs = sorted((fake["run"] / "rec").glob("aws_p[36]_*Z"))
    assert len(recs) == 1, recs
    return recs[0]


def driver_log(fake: dict) -> str:
    return (rec_dir(fake) / "driver.log").read_text()
