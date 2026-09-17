"""Checks run INSIDE a booted image to decide whether it is usable.

A separate file, shipped with scp, rather than a heredoc nested inside an
`ssh --command='...'` string: that nesting is what produced empty output on
the first attempt, and empty output that the caller read as success is this
project's single most repeated failure. Every check prints exactly one
`CHECK <name> PASS|FAIL <detail>` line, and the caller counts them.
"""
import importlib
import os
import subprocess
import sys

REPO = os.path.expanduser("~/attnbench_scaffold")
results = []


def emit(name, ok, detail=""):
    results.append(ok)
    print(f"CHECK {name} {'PASS' if ok else 'FAIL'} {detail}".rstrip(), flush=True)


def sh(*args):
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True)


if not os.path.isdir(REPO):
    emit("repo", False, f"{REPO} missing")
    sys.exit(1)

emit("repo", True, sh("git", "rev-parse", "--short", "HEAD").stdout.strip())

porcelain = sh("git", "status", "--porcelain").stdout.strip()
emit("clean", porcelain == "", "" if not porcelain else porcelain.splitlines()[0])

tracked = [f for f in sh("git", "ls-files", "results/").stdout.split() if f]
missing = [f for f in tracked if not os.path.isfile(os.path.join(REPO, f))]
emit("tracked_results", not missing and bool(tracked),
     f"{len(tracked)} tracked" if not missing else f"missing {missing[:3]}")

sys.path.insert(0, REPO)
for mod, attr in (("torch", "__version__"), ("transformers", "__version__"),
                  ("flash_attn", "__version__"), ("attnbench.masks", None)):
    try:
        m = importlib.import_module(mod)
        emit(mod, True, getattr(m, attr, "importable") if attr else "importable")
    except Exception as e:
        emit(mod, False, f"{type(e).__name__}: {e}")

# CPU-only host: no GPU is the EXPECTED state here, so it is asserted rather
# than reported, and a GPU showing up would mean this test is not testing what
# it claims to be cheap about.
try:
    import torch
    emit("cuda_absent", not torch.cuda.is_available(), "CPU-only host")
except Exception as e:
    emit("cuda_absent", False, f"{type(e).__name__}: {e}")

sys.exit(0 if all(results) else 1)
