"""`import attnbench` and the pure modules run without torch.

The time model (`attnbench.predict`), the link and trace-parsing components
and the provenance record do not compute with torch, so a machine without it
(a CPU container, an analysis laptop) can import, test and run them. Each
check runs in a fresh interpreter with torch blocked, so it holds whether or
not torch is installed here.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Setting sys.modules["torch"] to None makes any `import torch` raise.
BLOCK = "import sys; sys.modules['torch'] = None\n"


def run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", BLOCK + code], cwd=REPO,
                          capture_output=True, text=True, timeout=120)


def test_the_package_and_the_pure_modules_import_without_torch():
    r = run(
        "import importlib, pkgutil\n"
        "import attnbench, attnbench.predict\n"
        "names = ['attnbench.config', 'attnbench.provenance',\n"
        "         'attnbench.components.link', 'attnbench.components.op_sequence']\n"
        "names += [m.name for m in pkgutil.walk_packages(\n"
        "    attnbench.predict.__path__, 'attnbench.predict.')]\n"
        "for n in names:\n"
        "    importlib.import_module(n)\n"
        "assert 'attnbench.predict.recursion' in names\n"
        "assert 'attnbench.predict.ablations' in names\n"
        "print(len(names))\n")
    assert r.returncode == 0, r.stderr
    assert int(r.stdout) >= 20


def test_the_torch_modules_still_load_on_first_use():
    # attnbench.timing and attnbench.gates are still reachable as attributes;
    # without torch, reaching one raises the import error, not AttributeError.
    r = run(
        "import attnbench\n"
        "for name in ('timing', 'gates'):\n"
        "    try:\n"
        "        getattr(attnbench, name)\n"
        "    except ImportError as e:\n"
        "        assert 'torch' in str(e), e\n"
        "    else:\n"
        "        raise SystemExit(name + ' loaded with torch blocked')\n"
        "try:\n"
        "    attnbench.provenance.capture()\n"
        "except ImportError as e:\n"
        "    assert 'torch' in str(e), e\n"
        "else:\n"
        "    raise SystemExit('capture ran with torch blocked')\n"
        "try:\n"
        "    attnbench.no_such_module\n"
        "except AttributeError:\n"
        "    pass\n")
    assert r.returncode == 0, r.stderr + r.stdout
