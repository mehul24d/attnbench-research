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
  [[ -n "${FAKE_PIP_FAIL:-}" && "$*" == *"$FAKE_PIP_FAIL"* ]] && exit 1
  [[ "$3" == freeze ]] && echo "torch==2.9.1+cu129"
  exit 0
fi
[[ "$1 $2" == "-m pytest" ]] && exit "${FAKE_PYTEST_RC:-0}"
echo "UNHANDLED $*" >&2; exit 97
"""

FAKE_HELPER = r"""#!/usr/bin/env bash
printf 'helper %s\n' "$*" >> "$CALL_LOG"
[[ "$1" == "-c" ]] && exec "$REAL_PY" "$@"
if [[ "$1 $2" == "-m attnbench.build_guards" ]]; then
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


def test_no_cuobjdump_is_refused_before_any_install(env):
    (env[0] / "cuda" / "bin" / "cuobjdump").unlink()
    r, calls = run(env)
    assert r.returncode != 0 and "no cuobjdump" in r.stderr and _installs(calls) == []


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
