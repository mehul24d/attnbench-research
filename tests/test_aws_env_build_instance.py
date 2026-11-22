"""scripts/aws_env_build_instance.sh on the Mac, with fake commands first on
PATH (python, sudo, git, dpkg, sha256sum, the flash-attn build), every call
logged; the real list helper reads the committed v6 lists. Nothing is
installed, cloned or built."""
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "aws_env_build_instance.sh"
WHEEL_NAME = "flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl"

FAKE_PY = r"""#!/usr/bin/env bash
printf 'py %s\n' "$*" >> "$CALL_LOG"
if [[ "$1" == "-c" ]]; then
  case "$2" in
    *platform*) echo "${FAKE_PYVER:-3.10.12}" ;;
    *"torch, triton"*) echo "${FAKE_TORCH:-2.9.1+cu129 3.5.1}" ;;
    *"import attnbench"*) exit "${FAKE_IMPORT_RC:-0}" ;;
    *"import flash_attn, block_sparse_attn"*) [[ -n "${FAKE_FA_IMPORT_FAIL:-}" ]] && { echo "ImportError: libcuda.so.1" >&2; exit 1; }; echo 2.8.3.post1 ;;
    *getusersitepackages*) echo "$FAKE_USERSITE" ;;
    *) echo "UNHANDLED -c" >&2; exit 97 ;;
  esac
  exit 0
fi
if [[ "$1 $2" == "-m pip" ]]; then
  [[ -n "${FAKE_PIP_FAIL:-}" && "$*" == *"$FAKE_PIP_FAIL"* ]] && { echo "ERROR: fake pip refused $FAKE_PIP_FAIL"; exit 1; }
  [[ "$3" == freeze && -n "${FAKE_FREEZE_FAIL:-}" ]] && exit 3
  [[ "$3" == freeze ]] && echo "torch==2.9.1+cu129"
  if [[ "$*" == *"--user --no-deps "*flash_attn_3-*.whl ]]; then
    mkdir -p "$FAKE_USERSITE/flash_attn_3"; : > "$FAKE_USERSITE/flash_attn_3/_C.abi3.so"
  fi
  exit 0
fi
[[ "$1 $2" == "-m pytest" ]] && exit "${FAKE_PYTEST_RC:-0}"
echo "UNHANDLED $*" >&2; exit 97
"""

FAKE_HELPER = r"""#!/usr/bin/env bash
printf 'helper %s\n' "$*" >> "$CALL_LOG"
[[ "$1" == "-c" ]] && exec "$REAL_PY" "$@"
if [[ "$1 $2" == "-m attnbench.build_guards" ]]; then
  [[ -n "${ENV_BUILD_MEMINFO:-}" ]] && exec "$REAL_PY" "$@"
  exec "$REAL_PY" "$@" --free-gb "${FAKE_FREE_GB:-60}"
fi
case "$2" in
  requirements|diff-freeze) exec "$REAL_PY" "$@" ;;
  fa3-wheel) exit "${FAKE_FA3_RC:-0}" ;;
  libraries) echo '{"problems": []}'; exit "${FAKE_LIBS_RC:-0}" ;;
esac
echo "UNHANDLED $*" >&2; exit 97
"""

FAKES = {
    "sudo": 'printf "sudo %s\\n" "$*" >> "$CALL_LOG"; [[ "$1" == -H ]] && shift; exec "$@"',
    "git": r'''printf 'git %s\n' "$*" >> "$CALL_LOG"
case "$*" in
  "status --porcelain") printf '%s' "${FAKE_DIRTY:-}" ;;
  "rev-parse HEAD") echo 0123456789abcdef0123456789abcdef01234567 ;;
  "clone -q "*) mkdir -p "${@: -1}/.git" ;;
  "-C "*" checkout -q --detach "*) ;;
  *) echo "UNHANDLED git $*" >&2; exit 97 ;;
esac''',
    "dpkg": 'echo "ii  fake-package 1.0"',
    "sha256sum": 'for f in "$@"; do echo "0000  $f"; done',
    "apt-get": 'printf "apt-get %s\\n" "$*" >> "$CALL_LOG"',
    "fa_build": 'printf "fa_build %s archs=%s/%s/%s\\n" "$*" "$TORCH_CUDA_ARCH_LIST" "$FLASH_ATTN_CUDA_ARCHS" '
                '"$BLOCK_SPARSE_ATTN_CUDA_ARCHS" >> "$CALL_LOG"; exit "${FAKE_FA_RC:-0}"',
}
CUDA_FAKES = {
    "nvcc": 'echo "Cuda compilation tools, release ${FAKE_NVCC:-12.9}, V12.9.86"',
    "cuobjdump": r'''printf 'cuobjdump %s\n' "$*" >> "$CALL_LOG"
for sm in ${FAKE_SM:-sm_90 sm_80}; do echo "ELF file    1: x.$sm.cubin"; done''',
}


def _exe(p: Path, body: str) -> Path:
    p.write_text(body if body.startswith("#!") else "#!/usr/bin/env bash\n" + body + "\n")
    p.chmod(0o755)
    return p


@pytest.fixture
def env(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for name, body in FAKES.items():
        _exe(bin_ / name, body)
    cuda = tmp_path / "cuda" / "bin"
    cuda.mkdir(parents=True)
    for name, body in CUDA_FAKES.items():
        _exe(cuda / name, body)
    usersite = tmp_path / "usersite"
    usersite.mkdir()
    for lib in ("flash_attn_2_cuda", "block_sparse_attn_cuda"):
        (usersite / f"{lib}.cpython-310-x86_64-linux-gnu.so").write_bytes(b"")
    py = _exe(bin_ / "python3.10", FAKE_PY)
    helper = _exe(bin_ / "helper", FAKE_HELPER)
    wheel = tmp_path / WHEEL_NAME
    wheel.write_bytes(b"wheel")
    keep = {"/bin", "/usr/bin", "/usr/sbin", "/sbin"}
    path = ":".join([str(bin_)] + [d for d in os.environ["PATH"].split(":") if d in keep])
    e = {"PATH": path, "HOME": str(tmp_path / "home"), "CALL_LOG": str(tmp_path / "calls.log"),
         "REAL_PY": sys.executable, "ATTNBENCH_PYTHON": str(py),
         "ENV_BUILD_HELPER_PYTHON": str(helper), "ENV_BUILD_FA_SCRIPT": str(bin_ / "fa_build"),
         "ENV_BUILD_OUT": str(tmp_path / "out"), "XATTN_DIR": str(tmp_path / "x-attention"),
         "FA3_WHEEL_SHA256": hashlib.sha256(b"wheel").hexdigest(), "PYTHONDONTWRITEBYTECODE": "1",
         "CUDA_HOME": str(tmp_path / "cuda"), "FAKE_USERSITE": str(usersite)}
    (tmp_path / "home").mkdir()
    return tmp_path, e, wheel


def run(env, *extra, **over):
    tmp, e, wheel = env
    args = list(extra) if extra else [str(wheel)]
    r = subprocess.run(["bash", str(SCRIPT), *args], cwd=str(tmp), capture_output=True, text=True,
                       env={**e, **over}, timeout=120)
    log = Path(e["CALL_LOG"])
    return r, (log.read_text().splitlines() if log.exists() else [])


def _installs(calls):
    return [c for c in calls if " pip install" in c]


def test_static_bash_syntax():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0
    if shutil.which("shellcheck"):
        assert subprocess.run(["shellcheck", "-S", "error", str(SCRIPT)]).returncode == 0


def test_static_no_fa3_import_and_no_cloud_call():
    text = SCRIPT.read_text()
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    for word in ("flash_attn_interface", "import flash_attn_3", "fa3_import_check", "aws ", "gcloud", "gsutil"):
        assert word not in code, word


def test_full_pass_in_order(env):
    r, calls = run(env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.rstrip().endswith("ENV_BUILD_OK")
    order = [i for i, c in enumerate(calls) if any(k in c for k in (
        "sudo -H", "--user --no-deps --extra-index-url", f"--user --no-deps {env[0]}/{WHEEL_NAME}",
        "fa_build", "block-sparse-attn @", "-e " + str(env[0] / "x-attention"), "-m pytest"))]
    assert len(order) >= 7 and order == sorted(order)
    seq = [c for c in calls if c.startswith(("sudo -H", "fa_build")) or "pip install" in c or "-m pytest" in c]
    assert seq[0].startswith("sudo -H") and "pip install --no-deps --ignore-installed" in seq[0]
    assert "fa_build 2.8.3.post1 archs=8.9;9.0/80;90/80;90" in calls
    assert sum(c.startswith("cuobjdump --list-elf") for c in calls) == 2
    assert (env[0] / "out" / "cuda_archs.txt").read_text() == (
        "flash_attn_2_cuda sm_80 sm_90\nblock_sparse_attn_cuda sm_80 sm_90\n")
    assert any("--manifest configs/t64/fa3_manifest_49774b31.json" in c for c in calls)
    bsa = [c for c in calls if "block-sparse-attn @" in c][0]
    assert "@49d6c39e4dc0303442cda3bb758b3925d4399c49" in bsa and "--no-build-isolation" in bsa
    assert any("checkout -q --detach e37988770b9d1bebd489eba011d615f35587ba08" in c for c in calls)
    for c in _installs(calls):
        if "block-sparse-attn" not in c:
            assert "--no-deps" in c, c
    req = (env[0] / "out" / "req_usersite.txt").read_text()
    assert "xformers==0.0.33\n" in req + "\n" and "5d4b92a5" not in req
    assert "torch==2.9.1+cu129" in (env[0] / "out" / "req_system.txt").read_text()
    out = env[0] / "out"
    for f in ("libraries.json", "dpkg_l.txt", "SHA256SUMS", "diff_vs_v6_user.json",
              "diff_vs_v6_usersite.json", "diff_vs_v6_root.json", "logs/fa_build.log"):
        assert (out / f).exists(), f
    assert not any(c.startswith("apt-get") for c in calls)


@pytest.mark.parametrize("over,args,match", [
    ({"FA3_WHEEL_SHA256": "abc"}, None, "not a sha256"),
    ({}, ["/nonexistent/" + WHEEL_NAME], "no FA3 wheel"),
    ({"FAKE_DIRTY": " M x.py"}, None, "not clean"),
    ({"FAKE_PYVER": "3.12.1"}, None, "3.10.12"),
    ({"FAKE_FREE_GB": "15"}, None, "memory guard"),
    ({"XATTN_DIR": str(ROOT / "x-attention")}, None, "inside the repository"),
    ({"FAKE_NVCC": "11.8"}, None, "no nvcc 12.x"),
])
def test_refusals_install_nothing(env, over, args, match):
    r, calls = run(env, *(args or []), **over)
    assert r.returncode != 0 and match in r.stderr, r.stderr
    assert _installs(calls) == [] and not (env[0] / "out").exists()


def test_existing_record_is_never_overwritten(env):
    (env[0] / "out").mkdir()
    r, calls = run(env)
    assert r.returncode != 0 and "never overwritten" in r.stderr and _installs(calls) == []


def test_two_jobs_fit_sixteen_gib(env):
    r, _ = run(env, FAKE_FREE_GB="15", MAX_JOBS="2")
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("over,step,not_after", [
    ({"FAKE_TORCH": "2.9.0+cu128 3.5.1"}, 2, "flash_attn_3"),
    ({"FAKE_PIP_FAIL": "req_system"}, 2, "req_usersite"),
    ({"FAKE_FA3_RC": "1"}, 3, "flash_attn_3"),
    ({"FAKE_FA_RC": "1"}, 4, "block-sparse-attn"),
    ({"FAKE_SM": "sm_80 sm_86 sm_90"}, 4, "-m pytest"),
    ({"FAKE_FA_IMPORT_FAIL": "1"}, 5, "-m pytest"),
    ({"FAKE_LIBS_RC": "1"}, 5, "-m pytest"),
    ({"FAKE_PYTEST_RC": "1"}, 5, "pip freeze"),
])
def test_failure_stops_at_its_step(env, over, step, not_after):
    r, calls = run(env, **over)
    assert r.returncode != 0 and f"FAILED at step {step}" in r.stderr, r.stderr
    assert "ENV_BUILD_OK" not in r.stdout
    assert not any(not_after in c and ("pip install" in c or "-m pytest" in c or "pip freeze" in c)
                   for c in calls), calls


def test_missing_pip_is_installed_from_apt(env, tmp_path):
    py = Path(env[1]["ATTNBENCH_PYTHON"])
    py.write_text(FAKE_PY.replace('if [[ "$1 $2" == "-m pip" ]]; then',
                                  'if [[ "$*" == "-m pip --version" ]]; then exit 1; fi\n'
                                  'if [[ "$1 $2" == "-m pip" ]]; then'))
    r, calls = run(env)
    assert r.returncode == 0, r.stderr
    assert any(c == "apt-get install -y python3-pip" for c in calls)


def test_no_cuobjdump_anywhere_is_refused_before_any_pip_install(env):
    (env[0] / "cuda" / "bin" / "cuobjdump").unlink()
    r, calls = run(env, ENV_BUILD_CUDA_ROOT=str(env[0] / "empty"))
    assert r.returncode != 0 and "no cuobjdump" in r.stderr and _installs(calls) == []
    assert "-- tool inventory --" in r.stderr


# --- cuobjdump's search order, on the base image as the probe found it ------
# (adviser's rulings dated 22 Nov 2026, probe). The fixture is built from the
# committed inventory: every toolkit, its nvcc release and its bin listing.
PROBE = ROOT / "docs" / "launch_cards" / "aws_env_build_probe_20261121.txt"

FAKE_APT = r'''printf "apt-get %s\n" "$*" >> "$CALL_LOG"
[[ " $* " == *" install "* ]] || exit 0
n=$(cat @COUNTER@); if (( n > 0 )); then echo $((n-1)) > @COUNTER@; exit 100; fi
[[ -n "@NOOP@" ]] && exit 0
pkg="${@: -1}"; v="${pkg#cuda-cuobjdump-}"; v="${v/-/.}"
cp @CUOBJ@ @ROOT@/cuda-$v/bin/cuobjdump'''


def _probe_toolkits():
    """{'12.8': [bin names], ...} and the /usr/local/cuda link target."""
    import re
    text = PROBE.read_text()
    kits = {}
    for m in re.finditer(r"^### /usr/local/cuda-([0-9.]+)\nnvcc: release ([0-9.]+) .*\n"
                         r"cuobjdump: (\w+)\n.*\nbin: (.*)$", text, re.M):
        assert m.group(1) == m.group(2) and m.group(3) == "absent"
        kits[m.group(1)] = m.group(4).split()
    link = re.search(r"^/usr/local/cuda -> /usr/local/cuda-([0-9.]+)$", text, re.M).group(1)
    return kits, link


def _image(env, with_cuobjdump=(), apt_installs=True, apt_fails=0):
    """A /usr/local laid out as the probe read it, cuobjdump added only to the
    toolkits named (each logs under its own name). apt-get install of
    cuda-cuobjdump-X-Y puts one in cuda-X.Y, as the NVIDIA package does,
    after apt_fails failed install calls."""
    tmp = env[0]
    root = tmp / "usrlocal"
    kits, link = _probe_toolkits()
    for v, names in kits.items():
        b = root / f"cuda-{v}" / "bin"
        b.mkdir(parents=True)
        for n in names:
            (b / n).write_text("")
        _exe(b / "nvcc", f'echo "Cuda compilation tools, release {v}, V{v}.0"')
        if v in with_cuobjdump:
            _exe(b / "cuobjdump", CUDA_FAKES["cuobjdump"].replace("'cuobjdump %s", f"'cuobjdump-{v} %s"))
    (root / "cuda").symlink_to(root / f"cuda-{link}")
    apt_cuobj = _exe(tmp / "apt_cuobjdump", CUDA_FAKES["cuobjdump"].replace("'cuobjdump %s", "'cuobjdump-apt %s"))
    counter = tmp / "apt_fail_count"
    counter.write_text(str(apt_fails))
    _exe(tmp / "bin" / "apt-get", FAKE_APT.replace("@COUNTER@", str(counter)).replace(
        "@NOOP@", "" if apt_installs else "1").replace("@CUOBJ@", str(apt_cuobj)).replace("@ROOT@", str(root)))
    return {"ENV_BUILD_CUDA_ROOT": str(root), "CUDA_HOME": ""}


def test_the_probe_fixture_is_the_captured_image():
    kits, link = _probe_toolkits()
    assert sorted(kits) == ["12.8", "12.9", "13.0", "13.2"] and link == "13.2"
    assert all("nvcc" in b and "cuobjdump" not in b for b in kits.values())


def test_on_the_probed_image_apt_installs_the_matching_cuobjdump(env):
    r, calls = run(env, **_image(env))
    assert r.returncode == 0, r.stderr
    assert [c for c in calls if c.startswith("apt-get")] == [
        "apt-get -o DPkg::Lock::Timeout=600 install -y --no-install-recommends cuda-cuobjdump-12-9"]
    assert "nvcc 12.9 (" in r.stdout and "cuda-12.9/bin/cuobjdump" in r.stdout
    assert sum(c.startswith("cuobjdump-apt --list-elf") for c in calls) == 2
    first_pip = next(i for i, c in enumerate(calls) if " pip install" in c)
    assert next(i for i, c in enumerate(calls) if c.startswith("apt-get")) < first_pip


def test_a_stale_apt_cache_is_updated_once_and_retried(env):
    r, calls = run(env, **_image(env, apt_fails=1))
    assert r.returncode == 0, r.stderr
    assert [c for c in calls if c.startswith("apt-get")] == [
        "apt-get -o DPkg::Lock::Timeout=600 install -y --no-install-recommends cuda-cuobjdump-12-9", "apt-get -o DPkg::Lock::Timeout=600 update -q",
        "apt-get -o DPkg::Lock::Timeout=600 install -y --no-install-recommends cuda-cuobjdump-12-9"]


def test_apt_failing_twice_refuses_with_the_inventory(env):
    r, calls = run(env, **_image(env, apt_fails=2))
    assert r.returncode != 0 and "could not install cuda-cuobjdump-12-9" in r.stderr
    assert _installs(calls) == [] and "ENV_BUILD_OK" not in r.stdout
    inv = r.stderr.split("-- tool inventory --", 1)[1]
    for v in ("12.8", "12.9", "13.0", "13.2"):
        assert f"cuda-{v}: nvcc {v}, cuobjdump absent" in inv, inv
    assert "cuda -> " in inv and "bin2c" in inv and "nvcc=12.9" in inv


def test_an_apt_install_that_leaves_no_cuobjdump_refuses(env):
    r, calls = run(env, **_image(env, apt_installs=False))
    assert r.returncode != 0 and "after installing cuda-cuobjdump-12-9" in r.stderr
    assert _installs(calls) == [] and "-- tool inventory --" in r.stderr


@pytest.mark.parametrize("present,used", [
    (("12.9",), "12.9"),                 # nvcc's own toolkit first
    (("12.9", "13.0"), "12.9"),
    (("13.2",), "13.2"),                 # then a toolkit at or above nvcc's version
    (("13.0", "13.2"), "13.0"),          # the lowest of those
])
def test_a_toolkit_cuobjdump_is_used_without_apt(env, present, used):
    r, calls = run(env, **_image(env, with_cuobjdump=present))
    assert r.returncode == 0, r.stderr
    assert not any(c.startswith("apt-get") for c in calls)
    assert sum(c.startswith(f"cuobjdump-{used} --list-elf") for c in calls) == 2


def test_a_toolkit_below_nvcc_is_not_used(env):
    r, calls = run(env, **_image(env, with_cuobjdump=("12.8",)))
    assert r.returncode == 0, r.stderr
    assert any(c.startswith("apt-get") and " install " in c for c in calls)
    assert not any(c.startswith("cuobjdump-12.8 --list-elf") for c in calls)


def test_a_failure_after_the_guards_prints_the_inventory(env):
    r, _ = run(env, FAKE_PYTEST_RC="1")
    assert "FAILED at step 5" in r.stderr and "-- tool inventory --" in r.stderr


def test_committed_fa3_manifest_is_the_recorded_one():
    """The bucket's copy, read 21 Nov 2026: the script's recorded sha256, and
    the wheel, name and build stack the card's FA3 check expects."""
    import json
    import re
    m = ROOT / "configs" / "t64" / "fa3_manifest_49774b31.json"
    want = re.search(r'MANIFEST_SHA256="([0-9a-f]{64})"', SCRIPT.read_text()).group(1)
    assert hashlib.sha256(m.read_bytes()).hexdigest() == want
    d = json.loads(m.read_text())
    assert d["sha256"] == "1676ef0b86c377ae911a7de238c2a1f6b58a7666f1a6eef919d4dc7b0973b64a"
    assert d["wheel"] == WHEEL_NAME and d["commit"].startswith("49774b31")
    assert d["stack"]["python"] == "3.10.12" and d["stack"]["torch"] == "2.9.1+cu129"



# --- the dry-run wave (adviser's task of 22 Nov 2026, item 2) ---------------
# MemAvailable as the probe's `free -m` read it on m7a.xlarge: 14920 MiB.
PROBE_MEMINFO = "MemTotal:       15994880 kB\nMemFree:        15044608 kB\nMemAvailable:   15278080 kB\n"


def test_the_guard_reads_meminfo_and_two_jobs_fit_the_probed_instance(env):
    mi = env[0] / "meminfo"
    mi.write_text(PROBE_MEMINFO)
    r, _ = run(env, ENV_BUILD_MEMINFO=str(mi), MAX_JOBS="2")
    assert r.returncode == 0, r.stderr
    assert "memory check ok: 2 concurrent cicc" in r.stdout and "14.6 GiB available" in r.stdout


def test_a_real_memory_refusal_carries_the_guards_own_words(env):
    mi = env[0] / "meminfo"
    mi.write_text(PROBE_MEMINFO)
    r, calls = run(env, ENV_BUILD_MEMINFO=str(mi), MAX_JOBS="4")
    assert r.returncode != 0 and "does not fit here; the guard said: FATAL: MAX_JOBS=4" in r.stderr
    assert "lower MAX_JOBS to 2" in r.stderr and _installs(calls) == []


def test_a_guard_that_cannot_run_is_not_reported_as_memory(env):
    r, calls = run(env, ENV_BUILD_MEMINFO=str(env[0] / "no_such_meminfo"), MAX_JOBS="2")
    assert r.returncode != 0 and "did not run (exit 2; not a memory verdict)" in r.stderr
    assert "FileNotFoundError" in r.stderr and "does not fit" not in r.stderr
    assert "-- tool inventory --" in r.stderr and _installs(calls) == []


def test_a_failed_step_prints_its_own_log(env):
    r, _ = run(env, FAKE_PIP_FAIL="req_usersite")
    assert "FAILED at step 2: installing the user-site list" in r.stderr
    assert "| ERROR: fake pip refused req_usersite" in r.stderr and "-- tool inventory --" in r.stderr


def test_an_unguarded_command_names_its_line(env):
    r, _ = run(env, FAKE_FREEZE_FAIL="1")
    assert r.returncode == 3 and "ENV_BUILD_OK" not in r.stdout
    assert "FAILED at step 6: line " in r.stderr and "pip freeze --all" in r.stderr
    assert "-- tool inventory --" in r.stderr


def test_stop_before_compile_runs_every_step_up_to_the_first_compile(env):
    r, calls = run(env, "--stop-before-compile", str(env[2]), **_image(env))
    assert r.returncode == 0, r.stderr
    assert r.stdout.rstrip().endswith("ENV_BUILD_DRY_RUN_OK") and "ENV_BUILD_OK" not in r.stdout
    assert any("-r " in c and "req_system" in c for c in calls)            # step 2
    assert any(f"--user --no-deps {env[2]}" in c for c in calls)           # step 3
    assert any(c.startswith("git clone -q") for c in calls)                # step 4, no compile
    assert sum(c.startswith("cuobjdump-apt --list-elf") and "flash_attn_3/_C.abi3.so" in c
               for c in calls) == 1
    # Each source build's metadata step (the interpreter wave), in order, no compile.
    assert [c for c in calls if c.startswith("fa_build")] == [
        "fa_build --metadata-only 2.8.3.post1 archs=8.9;9.0/80;90/80;90"]
    # The build's own pip commands, each with --dry-run, under "$PY".
    meta = [c for c in calls if c.endswith(" --dry-run")]
    assert meta == [
        "py -m pip install --user --no-build-isolation block-sparse-attn @ git+https://github.com/"
        "mit-han-lab/Block-Sparse-Attention.git@49d6c39e4dc0303442cda3bb758b3925d4399c49 --dry-run",
        f"py -m pip install --user --no-deps -e {env[0] / 'x-attention'} --dry-run"]
    first = lambda k: next(n for n, c in enumerate(calls) if c.startswith(k))
    assert first("fa_build") < calls.index(meta[0]) < first("git clone -q") < calls.index(meta[1])
    for word in ("block-sparse-attn @", "-e " + str(env[0] / "x-attention")):
        assert not any(word in c and "--dry-run" not in c for c in calls), word
    for word in ("-m pytest", "pip freeze"):
        assert not any(word in c for c in calls), word
    out = env[0] / "out"
    assert (out / "cuda_archs_fa3.txt").read_text() == "flash_attn_3 sm_80 sm_90\n"
    assert (out / "image_pip_before.txt").exists() and not (out / "SHA256SUMS").exists()
    inv = (out / "interpreters.txt").read_text()
    assert f"{env[1]['ATTNBENCH_PYTHON']} -> " in inv and "chosen: PY=" in inv


@pytest.mark.parametrize("over,match", [
    ({"FAKE_FA_RC": "1"}, "flash-attn 2.8.3.post1's metadata step"),
    ({"FAKE_PIP_FAIL": "block-sparse-attn @"}, "block-sparse-attention's metadata step"),
    ({"FAKE_PIP_FAIL": "-e "}, "x-attention's metadata step"),
])
def test_stop_before_compile_fails_at_a_metadata_step(env, over, match):
    r, calls = run(env, "--stop-before-compile", str(env[2]), **_image(env), **over)
    assert r.returncode != 0 and f"FAILED at step 4: {match}" in r.stderr, r.stderr[-1500:]
    assert "ENV_BUILD_DRY_RUN_OK" not in r.stdout and "-- interpreters" in r.stderr


def test_stop_before_compile_fails_without_the_fa3_library(env):
    py = Path(env[1]["ATTNBENCH_PYTHON"])
    py.write_text(FAKE_PY.replace('mkdir -p "$FAKE_USERSITE/flash_attn_3"', "true"))
    r, calls = run(env, "--stop-before-compile", str(env[2]))
    assert r.returncode != 0 and "FAILED at step 4: no flash_attn_3/_C*.so" in r.stderr, r.stderr[-1500:]
    assert "ENV_BUILD_DRY_RUN_OK" not in r.stdout
    assert all(c.startswith("fa_build --metadata-only") for c in calls if "fa_build" in c)


# --- the pre-compile steps on a clean Python 3.10.12 (same task, item 2) ----
# No Docker on this Mac, so a clean venv (pip only, as the probe found the
# image's python) stands in for python:3.10. Every repository helper runs on
# it for real, with an import recorder (sitecustomize) that writes each
# process's non-stdlib imports into the call log; the CUDA pieces, sudo, git
# and the installs are faked: step 2's install drops a stand-in torch and
# triton into the venv, step 3's drops the FA3 library into the user site.
# A pre-compile step that imports a non-stdlib module the image lacks
# (docs/launch_cards/aws_env_build_image_modules.txt; after step 2, also the
# v6 lists) fails here.
#
# The interpreter wave (same date): the real scripts/build_flash_attn.sh runs
# its --metadata-only path here, and every source build's metadata step runs
# for real on stand-in trees (a setup.py that imports what flash-attn's does).
# The image's second Python is stood in for too: a `python3`, `python`, `pip`
# and `pip3` first on PATH that answer the inventory as the 3.11 would and
# fail anything else with the third attempt's "No module named 'torch'". Any
# pre-compile call that reaches them fails the dry run.
IMAGE_MODULES = ROOT / "docs" / "launch_cards" / "aws_env_build_image_modules.txt"
V6 = ROOT / "docs" / "image_libraries" / "attnbench-env-v6-20260917"
RECORDED_FA3_SHA = "1676ef0b86c377ae911a7de238c2a1f6b58a7666f1a6eef919d4dc7b0973b64a"

RECORDER = r'''import atexit, json, os, sys, sysconfig
_BASE = set(sys.modules)
_STD = os.path.realpath(sysconfig.get_paths()["stdlib"])
def _stdlib_file(f):   # e.g. _sysconfigdata_*, not in stdlib_module_names
    f = os.path.realpath(f) if f else ""
    return f.startswith(_STD + os.sep) and "site-packages" not in f
def _dump():
    log = os.environ.get("CALL_LOG")
    if not log:
        return
    new = {n: getattr(m, "__file__", None) for n, m in list(sys.modules.items())
           if n not in _BASE and "." not in n and n not in sys.stdlib_module_names
           and not _stdlib_file(getattr(m, "__file__", None))}
    with open(log, "a") as f:
        f.write("imports " + json.dumps({"argv": sys.argv[:3], "nonstd": new}) + "\n")
atexit.register(_dump)
'''

# The image's python: real for everything but pip's installs, which are faked.
CLEAN_PY = r'''#!/usr/bin/env bash
printf 'py %s\n' "$*" >> "$CALL_LOG"
if [[ "$1" == -c ]]; then
  [[ "$2" == *shakedown_arms* ]] && exit 0   # needs the real torch; the dry run imports it
  exec "$CLEAN" "$@"
fi
if [[ "$1 $2" == "-m pip" ]]; then
  case "$*" in
    *" list "*|*" --version"*) exec "$CLEAN" "$@" ;;
    *" --dry-run"*)   # for real, as on the instance, less what a venv here cannot do:
      args=()          # --user (refused in a venv), the git URL (the stand-in tree),
      for a in "$@"; do  # build isolation for -e (it needs an index)
        case "$a" in --user) ;; "block-sparse-attn @ git+"*) args+=("$BSA_STANDIN") ;; *) args+=("$a") ;; esac
      done
      [[ " $* " == *" -e "* ]] && args+=(--no-build-isolation)
      exec "$CLEAN" "${args[@]}" ;;
    *req_system*) cp -R "$FAKE_PKGS"/. "$CLEAN_SITE"/ ;;
    *flash_attn_3-*.whl*) us="$("$CLEAN" -c 'import site; print(site.getusersitepackages())')"
       mkdir -p "$us/flash_attn_3"; : > "$us/flash_attn_3/_C.abi3.so" ;;
  esac
  exit 0
fi
exec "$CLEAN" "$@"
'''

# The image's other Python (3.11, its pip first on PATH), as `python3`,
# `python`, `pip` and `pip3`: it answers the inventory, and refuses the rest.
OTHER_PY = r'''#!/usr/bin/env bash
case "$1" in
  --version) [[ "$0" == */pip* ]] && echo "pip 26.3 from /usr/local/lib/python3.11/dist-packages/pip (python 3.11)" \
               || echo "Python 3.11.14"; exit 0 ;;
  -c) [[ "$2" == "# interpreter inventory"* ]] && { echo "3.11.14 /usr/bin/python3.11 torch absent (ModuleNotFoundError)"; exit 0; } ;;
esac
printf 'other-interpreter %s %s\n' "${0##*/}" "$*" >> "$CALL_LOG"
echo "ModuleNotFoundError: No module named 'torch'" >&2; exit 1
'''

# The stand-in source trees' setup.py: what flash-attn's (and, a fork of it,
# block-sparse-attention's) imports at its top level; x-attention's is plain.
STANDIN_SETUP = """import psutil, torch
from torch.utils.cpp_extension import BuildExtension
from setuptools import setup
setup(name="@NAME@", version="0.0.0", py_modules=[])
"""
PLAIN_SETUP = 'from setuptools import setup\nsetup(name="xattn", version="0.0.0", py_modules=[])\n'

# The helper python: the clean interpreter itself; the FA3 check with only
# the wheel's hash and this Mac's platform tag faked (the wheel is a stand-in).
CLEAN_HELPER = r'''#!/usr/bin/env bash
printf 'helper %s\n' "$*" >> "$CALL_LOG"
[[ "${2:-}" == fa3-wheel ]] && exec "$CLEAN" -c "$SHIM" "$@"
exec "$CLEAN" "$@"
'''
SHIM = r'''import runpy, sys
sys.argv = sys.argv[1:]
from attnbench import fa3_wheel as fw
fw.sha256 = lambda p: "@SHA@"
_real = fw._sys_tags
def _tags():
    t = _real()
    return t | {type(next(iter(t)))("cp39", "abi3", "linux_x86_64")}
fw._sys_tags = _tags
runpy.run_path(sys.argv[0], run_name="__main__")
'''.replace("@SHA@", RECORDED_FA3_SHA)


def _clean_python():
    p = os.environ.get("ATTNBENCH_CLEAN_PY310")
    if not p and shutil.which("uv"):
        r = subprocess.run(["uv", "python", "find", "--no-project", "3.10.12"],
                           capture_output=True, text=True)
        p = r.stdout.strip() if r.returncode == 0 else None
    if not p or not Path(p).exists():
        pytest.skip("no Python 3.10.12 (set ATTNBENCH_CLEAN_PY310 or `uv python install 3.10.12`)")
    return p


def _image_sections():
    """{python version: set of names} from the list's `python:` sections."""
    import re
    out, cur = {}, None
    for line in IMAGE_MODULES.read_text().splitlines():
        if m := re.match(r"^python: (\S+)", line):
            cur = out.setdefault(m.group(1), set())
        elif m := re.match(r"^module: (\S+)$", line):
            cur.add(m.group(1))
    return out


def _image_modules():   # the interpreter the build uses, /usr/bin/python3
    return _image_sections()["3.10.12"]


# Distributions whose top-level modules are not only their own name (their
# top_level.txt): setuptools brings pkg_resources and _distutils_hack.
EXTRA_MODULES = {"setuptools": {"pkg_resources", "_distutils_hack"}}


def _with_extra_modules(names):
    return names.union(*(EXTRA_MODULES.get(n, set()) for n in names))


def test_the_image_list_covers_both_interpreters():
    secs = _image_sections()
    assert sorted(secs) == ["3.10.12", "3.11"]
    assert len(secs["3.10.12"]) == 80 and {"pip", "setuptools"} <= secs["3.10.12"]
    assert all("torch" not in names for names in secs.values())   # torch is step 2's


def _v6_modules():
    names = set()
    for f in V6.glob("pip_freeze_*.txt"):
        for line in f.read_text().splitlines():
            n = line.split("==")[0].split(" @ ")[0].strip()
            if n and not n.startswith(("-", "#")):
                names.add(n.lower().replace("-", "_"))
    return names


@pytest.fixture
def clean_env(env, tmp_path):
    base = _clean_python()
    venv = tmp_path / "clean310"
    subprocess.run([base, "-m", "venv", str(venv)], check=True, capture_output=True)
    py = venv / "bin" / "python"
    # The image's wheel 0.37.1 (image_pip_before.txt): setuptools' metadata
    # hook needs bdist_wheel. From uv's cache when it has it, else the index.
    for extra in (["--offline"], []):
        if shutil.which("uv") and subprocess.run(
                ["uv", "pip", "install", "-q", *extra, "--python", str(py), "wheel==0.37.1"],
                capture_output=True).returncode == 0:
            break
    else:
        pytest.skip("no wheel 0.37.1 for the clean venv (uv, offline or from the index)")
    site_dir = Path(subprocess.run([str(py), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                                   capture_output=True, text=True, check=True).stdout.strip())
    (site_dir / "sitecustomize.py").write_text(RECORDER)
    pkgs = tmp_path / "fake_pkgs"
    (pkgs / "torch").mkdir(parents=True)
    (pkgs / "torch" / "__init__.py").write_text('__version__ = "2.9.1+cu129"\nclass version:\n    cuda = "12.9"\n')
    (pkgs / "torch" / "utils").mkdir()
    (pkgs / "torch" / "utils" / "__init__.py").write_text("")
    (pkgs / "torch" / "utils" / "cpp_extension.py").write_text(
        "def is_ninja_available():\n    return True\nclass BuildExtension:\n    pass\n")
    (pkgs / "triton").mkdir()
    (pkgs / "triton" / "__init__.py").write_text('__version__ = "3.5.1"\n')
    (pkgs / "psutil").mkdir()   # v6's user site has psutil 7.2.2
    (pkgs / "psutil" / "__init__.py").write_text('__version__ = "7.2.2"\n')
    tmp, e, wheel = env
    _exe(Path(e["ATTNBENCH_PYTHON"]), CLEAN_PY)
    _exe(Path(e["ENV_BUILD_HELPER_PYTHON"]), CLEAN_HELPER)
    for name in ("python3", "python", "pip", "pip3"):
        _exe(tmp / "bin" / name, OTHER_PY)
    _exe(tmp / "bin" / "ninja", "exit 0")
    trees = {"fa/flash_attn-2.8.3.post1": STANDIN_SETUP.replace("@NAME@", "flash_attn"),
             "bsa_standin": STANDIN_SETUP.replace("@NAME@", "block_sparse_attn"),
             "x-attention": PLAIN_SETUP}
    for d, text in trees.items():
        (tmp / d / ".git").mkdir(parents=True)
        (tmp / d / "setup.py").write_text(text)
    (tmp / "fa" / "flash_attn-2.8.3.post1" / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools", "torch"]\nbuild-backend = "setuptools.build_meta"\n')
    shutil.rmtree(tmp / "fa" / "flash_attn-2.8.3.post1" / ".git")   # an sdist
    mi = tmp / "meminfo"
    mi.write_text(PROBE_MEMINFO)
    e.update({"CLEAN": str(py), "CLEAN_SITE": str(site_dir), "FAKE_PKGS": str(pkgs), "SHIM": SHIM,
              "ENV_BUILD_MEMINFO": str(mi), "MAX_JOBS": "2", "FA3_WHEEL_SHA256": RECORDED_FA3_SHA,
              "ENV_BUILD_FA_SCRIPT": str(ROOT / "scripts" / "build_flash_attn.sh"),
              "FA_SRC_DIR": str(tmp / "fa"), "BSA_STANDIN": str(tmp / "bsa_standin"), "FA_FREE_GB_OVERRIDE": "15"})   # the third attempt read 14.4 GiB
    e.pop("REAL_PY", None)
    return env


def _imports_by_phase(calls):
    """[(phase, argv, {module: file})] in call order; phase 'after' once step
    2's system install has run."""
    import json
    phase, out = "before", []
    for c in calls:
        if c.startswith("py -m pip install") and "req_system" in c:
            phase = "after"
        if c.startswith("imports "):
            d = json.loads(c[len("imports "):])
            out.append((phase, d["argv"], d["nonstd"]))
    return out


def test_the_pre_compile_steps_run_on_a_clean_python_310(clean_env):
    r, calls = run(clean_env, "--stop-before-compile", str(clean_env[2]), **_image(clean_env))
    assert r.returncode == 0, r.stderr.split("-- tool inventory --")[0][-3000:]
    assert r.stdout.rstrip().endswith("ENV_BUILD_DRY_RUN_OK")
    assert "memory check ok: 2 concurrent cicc" in r.stdout
    assert "FA3 wheel flash_attn_3" in r.stdout and "sha256 and tags checked" in r.stdout
    recs = _imports_by_phase(calls)
    seen = " ".join(" ".join(a) for _, a, _ in recs)
    # Not vacuous: the guard, both list reads and the FA3 check were recorded.
    for what in ("attnbench/build_guards.py", "requirements", "fa3-wheel"):
        assert what in seen, (what, [a for _, a, _ in recs])
    # The interpreter wave: the real flash-attn script's metadata path and all
    # three metadata steps ran under "$PY", and nothing reached the 3.11.
    assert "METADATA_OK flash-attn 2.8.3.post1" in (clean_env[0] / "out" / "logs" / "fa_metadata.log").read_text()
    # One metadata process per source build: pip's PEP 517 hook (a tree with
    # a pyproject.toml) or its legacy `setup.py egg_info` (one without).
    hooks = [a for _, a, _ in recs if a and a[0].endswith(("/_in_process.py", "/setup.py"))]
    assert len(hooks) == 3 and all(a[1] in ("dist_info", "egg_info") for a in hooks), hooks
    assert [c for c in calls if c.startswith("other-interpreter")] == []
    inv = (clean_env[0] / "out" / "interpreters.txt").read_text()
    assert "/bin/pip -> " in inv and "python 3.11" in inv and "3.11.14" in inv


def test_no_pre_compile_step_imports_what_the_image_lacks(clean_env):
    r, calls = run(clean_env, "--stop-before-compile", str(clean_env[2]), **_image(clean_env))
    assert r.returncode == 0, r.stderr.split("-- tool inventory --")[0][-3000:]
    image, v6 = _image_modules(), _v6_modules()
    bad = []
    for phase, argv, mods in _imports_by_phase(calls):
        allowed = _with_extra_modules(image | (v6 if phase == "after" else set()))
        for name, f in mods.items():
            if f and str(Path(f).resolve()).startswith(str(ROOT)):
                continue   # the repository's own modules
            if name.lower() not in allowed:
                bad.append((phase, argv, name))
    assert bad == [], bad
