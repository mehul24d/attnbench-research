"""The FA3 wheel build and the H100 import check, wired against fakes.

`scripts/build_fa3_wheel.sh` and `scripts/fa3_import_check.sh` run for real
here, with a fake `gcloud` (a directory stands in for the bucket), a fake
`git` (answers as if the pinned commit were fetched), a fake `nvcc`, `ninja`
and `timeout`, and a `python3` that passes everything through to this
interpreter except `setup.py bdist_wheel` (writes a small wheel and records
its environment), `-m pip` (records) and the post-install import check (which
needs an H100 and is unit-tested below with a stand-in module). Nothing
reaches GCP or GitHub: the real cloud CLIs are dropped from PATH.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from attnbench import fa3_wheel as fw
from attnbench.backends.fa3 import BUILD_FLAGS

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / "scripts" / "build_fa3_wheel.sh"
CHECK = REPO / "scripts" / "fa3_import_check.sh"
PIN = REPO / "scripts" / "_fa3_pin.sh"
COMMIT = "49774b31c8d1172a5b28f72e9aed0f322bf4ccba"
BUCKET = "gs://fake-bucket"
WHEEL = "flash_attn_3-3.0.0b1-py3-none-any.whl"   # a tag any interpreter installs

FAKE_GCLOUD = r"""#!/usr/bin/env bash
printf 'gcloud %s\n' "$*" >> "$CALLS"
[[ "$1 $2" == "storage cp" ]] || { echo "fake gcloud: unexpected $*" >&2; exit 9; }
src="$3"; dst="$4"
map() { echo "$FAKE_GCS/${1#gs://}"; }
if [[ "$dst" == gs://* ]]; then
  [[ -n "${FAKE_DENY_WRITE:-}" ]] && { echo "AccessDeniedException: 403" >&2; exit 1; }
  mkdir -p "$(dirname "$(map "$dst")")"; cp "$src" "$(map "$dst")"
else
  [[ -f "$(map "$src")" ]] || { echo "not found: $src" >&2; exit 1; }
  [[ "$dst" == */ ]] && dst="$dst$(basename "$src")"
  cp "$(map "$src")" "$dst"
  [[ -n "${FAKE_CORRUPT_READ:-}" ]] && printf 'x' >> "$dst"
fi
exit 0
"""

FAKE_GIT = r"""#!/usr/bin/env bash
printf 'git %s\n' "$*" >> "$CALLS"
dir="."; if [[ "$1" == "-C" ]]; then dir="$2"; shift 2; fi
case "$1" in
  init|remote|fetch) exit 0 ;;
  checkout) mkdir -p "$dir/hopper"; cp "$FAKE_SETUP_PY" "$dir/hopper/setup.py"; exit 0 ;;
  submodule) mkdir -p "$dir/csrc/cutlass"; exit 0 ;;
  rev-parse) if [[ "$dir" == */cutlass ]]; then echo cafef00d; else echo "${FAKE_HEAD:-$PINNED}"; fi ;;
  ls-remote) printf '%s\trefs/tags/v2.8.3.post2^{}\n' "${FAKE_TAG_COMMIT:-$PINNED}" ;;
  *) echo "fake git: unexpected $*" >&2; exit 9 ;;
esac
"""

FAKE_PYTHON = r"""#!/usr/bin/env bash
case "$*" in
  *"setup.py bdist_wheel"*)
    env | grep '^FLASH_ATTENTION_' | sort > "$SETUP_ENV"
    printf 'python3 %s\n' "$*" >> "$CALLS"
    [[ -n "${FAKE_DOWNLOAD:-}" ]] && echo "Guessing wheel URL:  https://example.invalid/x.whl"
    [[ -n "${FAKE_BUILD_RC:-}" ]] && { echo "nvcc error"; exit "$FAKE_BUILD_RC"; }
    dist="${@: -1}"; mkdir -p "$dist" build/temp.x
    printf 'wheel-bytes-%s' "$RANDOM" > "$dist/$WHEEL"
    printf '# ninja log v5\n0\t60000\t0\ta.o\tdeadbeef\n' > build/temp.x/.ninja_log
    exit 0 ;;
  *"is_ninja_available"*) exit 0 ;;
  "-m pip "*) printf 'python3 %s\n' "$*" >> "$CALLS"; exit 0 ;;
  "-m attnbench.fa3_wheel import") echo "import check ok (fake)"; exit 0 ;;
esac
exec "$REAL_PY" "$@"
"""

FAKE_NVCC = """#!/bin/bash
echo "Cuda compilation tools, release ${FAKE_NVCC_VERSION:-12.9}, V12.9.86"
"""


def _setup_stub(tmp_path, omit=None) -> Path:
    """The pinned setup.py's flag reads, as `os.getenv("NAME", "FALSE")` lines."""
    names = [f.split("=")[0] for f in BUILD_FLAGS if f.split("=")[0] != omit]
    p = tmp_path / "setup_stub.py"
    p.write_text("".join(f'X = os.getenv("{n}", "FALSE") == "TRUE"\n' for n in names))
    return p


def _env(tmp_path, **extra):
    bin_dir = tmp_path / "bin"
    cuda = tmp_path / "cuda" / "bin"
    for d in (bin_dir, cuda):
        d.mkdir(parents=True, exist_ok=True)
    scripts = {bin_dir / "gcloud": FAKE_GCLOUD, bin_dir / "git": FAKE_GIT,
               bin_dir / "python3": FAKE_PYTHON, cuda / "nvcc": FAKE_NVCC,
               bin_dir / "ninja": "#!/bin/bash\nexit 0\n",
               bin_dir / "timeout": '#!/bin/bash\nshift\nexec "$@"\n',
               bin_dir / "sudo": '#!/bin/bash\nprintf "sudo %s\\n" "$*" >> "$CALLS"\nexit ${FAKE_SUDO_RC:-0}\n',
               bin_dir / "curl": "#!/bin/bash\nexit 22\n"}
    for p, body in scripts.items():
        p.write_text(body)
        p.chmod(0o755)
    real = [d for d in os.environ["PATH"].split(os.pathsep)
            if not any((Path(d) / c).exists() for c in ("gcloud", "gsutil", "bq"))]
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("GCP_", "ATTNBENCH_", "CLOUDSDK_", "FA3_", "FLASH_ATTENTION_"))}
    env.pop("PYTHONPATH", None)
    env.update(
        PATH=os.pathsep.join([str(bin_dir), *real]), HOME=str(tmp_path),
        CALLS=str(tmp_path / "calls.log"), FAKE_GCS=str(tmp_path / "gcs"),
        SETUP_ENV=str(tmp_path / "setup_env.txt"), REAL_PY=sys.executable,
        FAKE_SETUP_PY=str(_setup_stub(tmp_path, extra.pop("OMIT_FLAG", None))),
        PINNED=COMMIT, WHEEL=WHEEL, CUDA_HOME=str(tmp_path / "cuda"),
        ATTNBENCH_BUCKET=BUCKET, FA_FREE_GB_OVERRIDE="64",
        FA3_EXPECT_PYTHON="%d.%d" % sys.version_info[:2], FA3_EXPECT_TORCH=torch.__version__,
        FA3_WORK_DIR=str(tmp_path / "work"))
    env.update(extra)
    return env


def _run(script, tmp_path, *args, **extra):
    env = _env(tmp_path, **extra)
    r = subprocess.run(["bash", str(script), *args], capture_output=True, text=True,
                       env=env, cwd=tmp_path, timeout=180)
    calls = (tmp_path / "calls.log").read_text() if (tmp_path / "calls.log").exists() else ""
    return r, calls


def _stored(tmp_path) -> Path:
    return tmp_path / "gcs" / "fake-bucket" / "fa3" / COMMIT


def test_the_pin_is_the_recorded_release_commit():
    text = PIN.read_text()
    assert f'FA3_COMMIT="{COMMIT}"' in text and 'FA3_TAG="v2.8.3.post2"' in text
    assert "main" not in [l.split("=", 1)[1].strip('"') for l in text.splitlines()
                          if l.startswith("FA3_")]


# --- the build ----------------------------------------------------------------------


def test_a_build_stores_the_wheel_its_hash_and_manifest_and_reads_it_back(tmp_path):
    r, calls = _run(BUILD, tmp_path)
    assert r.returncode == 0, r.stderr + r.stdout
    d = _stored(tmp_path)
    wheel = (d / WHEEL).read_bytes()
    sha = hashlib.sha256(wheel).hexdigest()
    manifest = json.loads((d / "manifest.json").read_text())
    assert manifest["sha256"] == sha and manifest["wheel"] == WHEEL
    assert (manifest["commit"], manifest["tag"]) == (COMMIT, "v2.8.3.post2")
    assert manifest["cutlass_commit"] == "cafef00d"
    assert manifest["stack"]["torch"] == torch.__version__
    assert (d / "SHA256SUMS").read_text() == f"{sha}  {WHEEL}\n"
    for f in ("started.json", "build.log", "stack.json", "ninja_log.txt"):
        assert (d / f).exists(), f
    assert f"sha256 : {sha}" in r.stdout
    assert f"git -C {tmp_path}/work/src fetch -q --depth 1 origin {COMMIT}" in calls
    # the read-back is a download of the stored wheel after its upload
    up = calls.index(f"storage cp {tmp_path}/work/out/dist/{WHEEL}")
    assert calls.index(f"storage cp {BUCKET}/fa3/{COMMIT}/{WHEEL}") > up


def test_the_build_gets_exactly_the_backends_flags_and_the_manifest_records_them(tmp_path):
    r, _ = _run(BUILD, tmp_path)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "setup_env.txt").read_text().split() == sorted(BUILD_FLAGS)
    m = json.loads((_stored(tmp_path) / "manifest.json").read_text())
    assert m["build_flags"] == list(BUILD_FLAGS) and m["built_head_dims"] == [128]
    assert (m["max_jobs"], m["gb_per_job"], m["memory_sample_s"]) == (3, 15, 1)


def test_a_flag_the_pinned_setup_py_does_not_read_refuses_before_the_build(tmp_path):
    r, calls = _run(BUILD, tmp_path, OMIT_FLAG="FLASH_ATTENTION_DISABLE_SPLIT")
    assert r.returncode == 1
    assert "FLASH_ATTENTION_DISABLE_SPLIT=TRUE is not read by the pinned" in r.stderr
    assert "setup.py bdist_wheel" not in calls and "sudo" not in calls


@pytest.mark.parametrize("jobs,ok", [("3", True), ("4", False)])
def test_the_memory_guard_uses_15_gb_per_job(tmp_path, jobs, ok):
    """About 58 GB is usable on n2-standard-16 (61 GB available minus the
    3 GB reserve): 3 jobs x 15 GB fit, 4 do not."""
    r, calls = _run(BUILD, tmp_path, MAX_JOBS=jobs, FA_FREE_GB_OVERRIDE="61")
    assert (r.returncode == 0) is ok, r.stdout + r.stderr
    if not ok:
        assert "MAX_JOBS=4 x NVCC_THREADS=1 = 4 concurrent cicc processes needing ~60 GB" in r.stdout
        assert calls == ""


# --- the VM halts itself when the build ends -------------------------------------------


def _halt_index(calls):
    lines = calls.splitlines()
    halts = [i for i, l in enumerate(lines) if l.startswith("sudo ")]
    assert len(halts) == 1, calls
    assert lines[halts[0]] == 'sudo -n shutdown -h +1 attnbench: FA3 build ended'
    return halts[0], lines


@pytest.mark.parametrize("extra,rc", [({}, 0), ({"FAKE_BUILD_RC": "1"}, 2),
                                      ({"FAKE_CORRUPT_READ": "1"}, 3)])
def test_the_vm_halts_after_the_logs_are_stored_on_success_and_failure(tmp_path, extra, rc):
    r, calls = _run(BUILD, tmp_path, **extra)
    assert r.returncode == rc, r.stderr
    i, lines = _halt_index(calls)
    uploads = [j for j, l in enumerate(lines) if l.startswith("gcloud storage cp ") and
               l.split()[3].startswith("/") and "/build.log" in l]
    assert uploads and max(uploads) < i, "build.log must be stored before the halt"
    assert i == len(lines) - 1 or all(not l.startswith("gcloud") for l in lines[i:])


def test_a_refusal_before_the_build_does_not_halt(tmp_path):
    r, calls = _run(BUILD, tmp_path, FAKE_HEAD="0" * 40)
    assert r.returncode == 1 and "sudo" not in calls


def test_a_halt_that_fails_is_reported(tmp_path):
    r, _ = _run(BUILD, tmp_path, FAKE_SUDO_RC="1")
    assert r.returncode == 0
    assert "the VM did NOT halt itself" in r.stderr


def test_a_download_attempt_in_the_log_fails_the_build(tmp_path):
    r, _ = _run(BUILD, tmp_path, FAKE_DOWNLOAD="1")
    assert r.returncode == 2 and "download a prebuilt wheel" in r.stderr
    assert not (_stored(tmp_path) / WHEEL).exists()


def test_no_bucket_refuses_before_anything_runs(tmp_path):
    env_extra = {}
    r, calls = _run(BUILD, tmp_path, ATTNBENCH_BUCKET="", **env_extra)
    assert r.returncode == 4 and "no results bucket" in r.stderr
    assert calls == ""


@pytest.mark.parametrize("extra,match", [
    ({"MAX_JOBS": "20"}, "MAX_JOBS=20"),
    ({"FA3_EXPECT_TORCH": "2.9.1+cu129-not-this"}, "build stack does not match"),
    ({"FAKE_NVCC_VERSION": "12.1"}, "nvcc 12.1, need 12.3"),
    ({"FAKE_DENY_WRITE": "1"}, "cannot write to the results bucket"),
])
def test_cheap_checks_refuse_before_the_source_is_fetched(tmp_path, extra, match):
    r, calls = _run(BUILD, tmp_path, **extra)
    assert r.returncode == 1, r.stderr
    assert match in r.stdout + r.stderr
    assert "git " not in calls and "setup.py" not in calls


@pytest.mark.parametrize("extra,match", [
    ({"FAKE_HEAD": "0" * 40}, "the pin is"),
    ({"FAKE_TAG_COMMIT": "1" * 40}, "now points at"),
])
def test_a_source_that_is_not_the_pin_is_never_built(tmp_path, extra, match):
    r, calls = _run(BUILD, tmp_path, **extra)
    assert r.returncode == 1 and match in r.stderr
    assert "setup.py" not in calls


def test_a_failed_build_stores_its_logs_and_no_wheel(tmp_path):
    r, _ = _run(BUILD, tmp_path, FAKE_BUILD_RC="1")
    assert r.returncode == 2 and "no usable wheel" in r.stderr
    d = _stored(tmp_path)
    failed = [p for p in d.iterdir() if p.name.startswith("failed-")]
    assert len(failed) == 1 and (failed[0] / "build.log").exists()
    assert not (d / WHEEL).exists() and not (d / "manifest.json").exists()


def test_a_wheel_that_reads_back_differently_fails(tmp_path):
    r, _ = _run(BUILD, tmp_path, FAKE_CORRUPT_READ="1")
    assert r.returncode == 3 and "reads back as" in r.stderr


# --- the H100 import check ------------------------------------------------------------


def _built(tmp_path) -> str:
    r, _ = _run(BUILD, tmp_path)
    assert r.returncode == 0, r.stderr
    (tmp_path / "calls.log").write_text("")
    return hashlib.sha256((_stored(tmp_path) / WHEEL).read_bytes()).hexdigest()


def test_the_import_check_installs_the_recorded_wheel_without_deps(tmp_path):
    sha = _built(tmp_path)
    r, calls = _run(CHECK, tmp_path, sha)
    assert r.returncode == 0, r.stderr + r.stdout
    assert "wheel check ok" in r.stdout and "import check ok" in r.stdout
    (pip,) = [l for l in calls.splitlines() if "-m pip" in l]
    assert "--no-deps" in pip and pip.endswith(WHEEL)


def test_the_import_check_refuses_a_wheel_that_is_not_the_recorded_one(tmp_path):
    _built(tmp_path)
    r, calls = _run(CHECK, tmp_path, "a" * 64)
    assert r.returncode == 1 and "is not the recorded" in r.stderr
    assert "-m pip" not in calls


@pytest.mark.parametrize("arg", ["", "abc", "A" * 64])
def test_the_import_check_needs_the_hash_as_an_argument(tmp_path, arg):
    r, calls = _run(CHECK, tmp_path, arg)
    assert r.returncode == 1 and "usage" in r.stderr and calls == ""


# --- fa3_wheel, unit ------------------------------------------------------------------


def _wheel(tmp_path, name=WHEEL, data=b"w"):
    p = tmp_path / name
    p.write_bytes(data)
    return p, hashlib.sha256(data).hexdigest()


def _manifest(name, sha, py="3.10.12", tv="2.9.1+cu129"):
    return {"wheel": name, "sha256": sha, "stack": {"python": py, "torch": tv}}


HERE = {"python": "3.10.12", "torch": "2.9.1+cu129"}


def test_check_stack_names_every_mismatch():
    with pytest.raises(fw.WheelCheckError) as e:
        fw.check_stack({"python": "3.12.1", "torch": "2.5.0", "nvcc": ""},
                       "3.10", "2.9.1+cu129", "12.3")
    msg = str(e.value)
    assert "python 3.12.1" in msg and "torch 2.5.0" in msg and "nvcc not found" in msg
    fw.check_stack({"python": "3.10.12", "torch": "2.9.1+cu129", "nvcc": "12.9"},
                   "3.10", "2.9.1+cu129", "12.3")


def test_check_wheel_accepts_the_recorded_wheel_on_its_stack(tmp_path):
    from packaging.tags import Tag
    name = "flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl"
    p, sha = _wheel(tmp_path, name)
    fw.check_wheel(p, _manifest(name, sha), sha, HERE,
                   supported_tags={Tag("cp39", "abi3", "linux_x86_64")})


@pytest.mark.parametrize("change,match", [
    ("expected", "is not the recorded"),
    ("manifest_sha", "manifest's sha256"),
    ("torch", "torch here is"),
    ("python", "python here is"),
    ("tags", "no tag of"),
])
def test_check_wheel_refuses(tmp_path, change, match):
    from packaging.tags import Tag
    name = "flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl"
    p, sha = _wheel(tmp_path, name)
    m, expected, here = _manifest(name, sha), sha, dict(HERE)
    tags = {Tag("cp39", "abi3", "linux_x86_64")}
    if change == "expected":
        expected = "0" * 64
    elif change == "manifest_sha":
        m["sha256"] = "0" * 64
    elif change in ("torch", "python"):
        here[change] = "other"
    else:
        tags = {Tag("cp39", "abi3", "macosx_14_0_arm64")}
    with pytest.raises(fw.WheelCheckError, match=match):
        fw.check_wheel(p, m, expected, here, supported_tags=tags)


@pytest.mark.parametrize("cc,ok", [((9, 0), True), ((8, 0), False)])
def test_import_check_needs_hopper_and_the_module(cc, ok, monkeypatch):
    mod = types.ModuleType("flash_attn_interface")
    mod.flash_attn_func = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "flash_attn_interface", mod)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: cc)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "NVIDIA H100 80GB HBM3")
    if ok:
        assert fw.import_check()["fa3_available"] is True
    else:
        with pytest.raises(fw.WheelCheckError, match="sm_90a only"):
            fw.import_check()


def test_import_check_fails_without_the_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "flash_attn_interface", None)
    with pytest.raises(ImportError):
        fw.import_check()


# --- the build VM launcher -------------------------------------------------------------

LAUNCHER = REPO / "scripts" / "gcp_launch_fa3_build.sh"

FAKE_LAUNCH_GCLOUD = r"""#!/usr/bin/env bash
printf '%s\n' "$*" >> "$CALLS"
case "$*" in *"config get-value project"*) echo test-project ;; esac
exit 0
"""


def _launch(tmp_path, rate="77.21", answer="launch"):
    bin_dir = tmp_path / "lbin"
    bin_dir.mkdir()
    (bin_dir / "gcloud").write_text(FAKE_LAUNCH_GCLOUD)
    m = __import__("re").search(r'^LAST_GCP_LAUNCH_DATE="([\d-]+)"$',
                                (REPO / "scripts" / "_launch_policy.sh").read_text(), 8)
    (bin_dir / "date").write_text(
        f'#!/bin/bash\nif [[ "$*" == "+%F" ]]; then echo {m.group(1)}; else exec /bin/date "$@"; fi\n')
    for p in bin_dir.iterdir():
        p.chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path),
           "CALLS": str(tmp_path / "launch.log"), "ATTNBENCH_GCP_PROJECT": "test-project",
           "GCP_RATE_INR_HR": rate}
    r = subprocess.run(["bash", str(LAUNCHER), "fa3-inst"], input=answer + "\n",
                       capture_output=True, text=True, env=env, timeout=60)
    log = tmp_path / "launch.log"
    return r, [l for l in (log.read_text() if log.exists() else "").splitlines()
               if "instances create" in l]


def test_the_launcher_creates_the_chosen_cpu_machine_with_the_cap(tmp_path):
    r, creates = _launch(tmp_path)
    assert r.returncode == 0, r.stderr
    (c,) = creates
    for arg in ("--machine-type=n2-standard-16", "--zone=us-central1-a",
                "--provisioning-model=STANDARD", "--max-run-duration=3h",
                "--instance-termination-action=DELETE", "--image=attnbench-env-v5-20260905",
                "--scopes=https://www.googleapis.com/auth/cloud-platform"):
        assert arg in c, arg
    assert "--accelerator" not in c and "SPOT" not in c
    assert "at most Rs 232" in r.stdout and "+150 min" in r.stdout


@pytest.mark.parametrize("rate,answer,match", [
    ("", "launch", "GCP_RATE_INR_HR"), ("77.21", "no", "Aborted")])
def test_the_launcher_creates_nothing_without_a_rate_or_the_typed_word(tmp_path, rate, answer, match):
    r, creates = _launch(tmp_path, rate=rate, answer=answer)
    assert r.returncode == 1 and match in r.stdout + r.stderr and creates == []


def test_the_build_stops_before_the_vm_halts():
    build = BUILD.read_text()
    launcher = LAUNCHER.read_text()
    deadline = int(__import__("re").search(r'FA3_BUILD_TIMEOUT_MIN:-(\d+)', build).group(1))
    halt = int(__import__("re").search(r'^HALT_MINUTES=(\d+)$', launcher, 8).group(1))
    assert halt - deadline >= 30, (deadline, halt)
