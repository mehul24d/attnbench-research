"""No bare pip, pip3 or python in the build and instance scripts (adviser's
task of 22 Nov 2026, the interpreter wave). The AWS base image has two
Pythons: /usr/bin/python3 (3.10.12, where the environment build installs
torch) and a 3.11 whose `pip` is first on PATH. The third environment-build
attempt's bare `pip` in scripts/build_flash_attn.sh ran flash-attn's setup.py
under 3.11, which has no torch. Every call is now "$PY" (or "$HELPER_PY") with
`-m pip` for installs.

Scope (plan default, flagged): the scripts that build or run on the
environment-build and FA3-build instances. The gcp_* launch and bootstrap
scripts are left as they are (GCP, pinned, already `python3 -m pip`)."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ["build_flash_attn.sh", "build_fa3_wheel.sh", "_fa3_pin.sh", "fa3_import_check.sh",
           "aws_env_build_instance.sh", "aws_env_probe_inventory.sh"]

# A bare interpreter run as a command: pip, pip3, pip3.11, python, python3,
# python3.10 where a command starts (a line, after | ; & ( $( or a backtick,
# after then/do/else/exec/env/xargs/nohup, `timeout N`, `sudo` and its flags,
# and any VAR=value prefixes), not as an argument (`"$PY" -m pip`,
# `apt-get install python3-pip`), a path, a variable's default or prose in a
# message ("(pip list failed)"; a `(` just inside a quote is not a subshell).
CMD_START = (r"(?:^|[|;&`]|(?<!\")\(|\$\(|\b(?:then|do|else|exec|nohup|env|xargs"
             r"|timeout\s+\S+|sudo(?:\s+-\S+)*)\s)")
BARE = re.compile(CMD_START + r"\s*(?:[A-Za-z_]\w*=\S*\s+)*(?:pip3?|python3?)(?:\.\d+)?(?![\w.*-])")
# The variable's own default, removed before the scan.
ALLOWED = [re.compile(r"\$\{ATTNBENCH_PYTHON:-[^}]*\}")]


def bare_uses(text):
    """[(line number, line)] of every bare use outside comments."""
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("#"):
            continue
        code = re.sub(r"(^|\s)#\s.*$", "", line)   # a trailing `# comment`
        for a in ALLOWED:
            code = a.sub("", code)
        if BARE.search(code):
            out.append((n, line.strip()))
    return out


@pytest.mark.parametrize("name", SCRIPTS)
def test_no_bare_pip_or_python(name):
    assert bare_uses((ROOT / "scripts" / name).read_text()) == []


@pytest.mark.parametrize("name", ["build_flash_attn.sh", "build_fa3_wheel.sh", "fa3_import_check.sh"])
def test_one_interpreter_variable(name):
    text = (ROOT / "scripts" / name).read_text()
    assert len(re.findall(r'^PY="\$\{ATTNBENCH_PYTHON:-python3\}"$', text, re.M)) == 1


def test_the_instance_script_passes_its_interpreter_to_the_flash_attn_build():
    text = (ROOT / "scripts" / "aws_env_build_instance.sh").read_text()
    assert re.search(r'^PY="\$\{ATTNBENCH_PYTHON:-/usr/bin/python3\}"\nexport ATTNBENCH_PYTHON="\$PY"', text, re.M)


@pytest.mark.parametrize("line", [
    "pip install --user --no-build-isolation .",                 # the third attempt's
    "python3 -c 'import psutil' 2>/dev/null || pip install --user -q psutil",
    "  pip3 install x", "python setup.py egg_info", "x=$(python3.11 -c 1)",
    "sudo -H pip3.10 install y", "PYTHONPATH=x python3 -m attnbench.y",
    'timeout "${T}m" python3 setup.py bdist_wheel', 'FIELDS="$(python3 - "$OUT/stack.json" <<EOF',
])
def test_the_scan_sees_a_bare_use(line):
    assert bare_uses(line) != []


@pytest.mark.parametrize("line", [
    '"$PY" -m pip install --user --no-build-isolation .', 'PY="${ATTNBENCH_PYTHON:-python3}"',
    'PY="${ATTNBENCH_PYTHON:-/usr/bin/python3}"', "sudo apt-get install -y python3-pip",
    "# pip install in a comment", 'for f in "$d"/pip "$d"/python3*; do', 'echo "  $PY -m pip install ninja"',
    "cuda-gdb-python3.10-tui", "x=1  # bare pip in a trailing comment",
    '"$PY" -m pip list >x 2>&1 || echo "(pip list failed)" >>x', 'sec "python and torch"',
    'pip*) echo "$f: $("$f" --version)" ;;', '|| fail "no pip for $PY"',
])
def test_the_scan_passes_the_one_interpreter(line):
    assert bare_uses(line) == []
