"""Fix for the deploy script's temporary file (gcp_deploy_source.sh):
`mktemp -t attnbench-deploy` is macOS (BSD) syntax and GNU mktemp refuses
it ("too few X's"), so on Linux the deploy stopped before its first gcloud
call. The call must work under both. Each implementation is a fake first on
PATH that accepts only what that implementation accepts; no gcloud is real.
Written before the fix.
"""

from __future__ import annotations

import re

import pytest

from test_gcp_project_pin import SCRIPTS, _run

# BSD (macOS) mktemp: options -d -q -u and -t PREFIX; a template's trailing
# X's are replaced; no template behaves as `-t tmp`. No long options, no -p.
FAKE_BSD = r"""#!/usr/bin/env python3
import os, random, string, sys
args = sys.argv[1:]
open(os.environ["MKTEMP_LOG"], "a").write("bsd " + " ".join(args) + "\n")
d = prefix = None
templates = []
i = 0
while i < len(args):
    a = args[i]
    if a == "-d": d = True
    elif a in ("-q", "-u"): pass
    elif a == "-t": i += 1; prefix = args[i]
    elif a.startswith("-"): sys.exit(f"mktemp: illegal option -- {a}")
    else: templates.append(a)
    i += 1
def fill(t):
    n = len(t) - len(t.rstrip("X"))
    return t[:len(t) - n] + "".join(random.choice(string.ascii_letters) for _ in range(n))
if not templates:      # macOS: no template behaves as `-t tmp`
    templates = [os.path.join(os.environ.get("TMPDIR", "/tmp"), (prefix or "tmp") + ".XXXXXXXX")]
for t in templates:
    p = fill(t)
    os.mkdir(p) if d else open(p, "x").close()
    print(p)
"""

# GNU coreutils mktemp: at most one template, which needs 3 or more trailing
# X's; -t means "relative to the temp dir" and still needs the X's.
FAKE_GNU = r"""#!/usr/bin/env python3
import os, random, string, sys
args = sys.argv[1:]
open(os.environ["MKTEMP_LOG"], "a").write("gnu " + " ".join(args) + "\n")
d = t_flag = False
tmpdir = None
templates = []
i = 0
while i < len(args):
    a = args[i]
    if a in ("-d", "--directory"): d = True
    elif a in ("-q", "--quiet", "-u", "--dry-run"): pass
    elif a == "-t": t_flag = True
    elif a == "-p": i += 1; tmpdir = args[i]
    elif a.startswith("--tmpdir"): tmpdir = a.partition("=")[2] or os.environ.get("TMPDIR", "/tmp")
    elif a.startswith("-"): sys.exit(f"mktemp: invalid option -- '{a}'")
    else: templates.append(a)
    i += 1
if len(templates) > 1: sys.exit("mktemp: too many templates")
t = templates[0] if templates else "tmp.XXXXXXXXXX"
if len(t) - len(t.rstrip("X")) < 3: sys.exit(f"mktemp: too few X's in template '{t}'")
if t_flag or tmpdir: t = os.path.join(tmpdir or os.environ.get("TMPDIR", "/tmp"), t)
n = len(t) - len(t.rstrip("X"))
p = t[:len(t) - n] + "".join(random.choice(string.ascii_letters) for _ in range(n))
os.mkdir(p) if d else open(p, "x").close()
print(p)
"""


def _with(tmp_path, body):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "mktemp").write_text(body)
    (bin_dir / "mktemp").chmod(0o755)
    log = tmp_path / "mktemp.log"
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    p, calls = _run("gcp_deploy_source.sh", tmp_path,
                    extra={"MKTEMP_LOG": str(log), "TMPDIR": str(tmp)})
    return p, calls, log.read_text().splitlines(), tmp


@pytest.mark.parametrize("impl, body", [("bsd", FAKE_BSD), ("gnu", FAKE_GNU)],
                         ids=["bsd", "gnu"])
def test_the_deploy_makes_its_bundle_under_either_mktemp(tmp_path, impl, body):
    p, calls, used, tmp = _with(tmp_path, body)
    assert used and used[0].startswith(impl), used
    scp = [c for c in calls if c.startswith("compute scp")]
    assert scp, p.stderr[-2000:]
    bundle = scp[0].split()[2]
    assert bundle.startswith(str(tmp)) and bundle.endswith(".bundle")
    assert list(tmp.iterdir()) == []            # the trap removed what mktemp made


def test_the_mktemp_call_uses_only_what_both_accept():
    text = (SCRIPTS / "gcp_deploy_source.sh").read_text()
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    calls = re.findall(r"mktemp\b[^)\n]*", code)
    assert calls, "no mktemp call"
    for c in calls:
        assert not re.search(r"\s(-t|-p|--tmpdir|--suffix)\b", c), c
        assert re.search(r"X{6,}", c), f"{c}: a template with trailing X's"
