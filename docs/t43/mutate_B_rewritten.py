"""Reader B's five mutations whose text the wave 1 fixes rewrote (M1, M2,
M14, M15, M16), re-expressed on the current code with the same intent and
B's own test lists; B's driver (apply, run, git checkout)."""
import os
import subprocess
import sys

ROOT = sys.argv[1]
M = [
    ("M1' late check drops the skew", "attnbench/predict/commit.py",
     "if start < synced_at + gap + clock_skew:", "if start < synced_at + gap:",
     ["tests/test_predict_commit.py", "tests/test_audit_fixes_commit.py"]),
    ("M2' late check removed", "attnbench/predict/commit.py",
     "if start < synced_at + gap + clock_skew:", "if False:", ["tests/test_predict_commit.py"]),
    ("M14' block identity check returns nothing", "scripts/t61_session.py",
     "            elif mine.get(k) is not None and mine.get(k) != theirs.get(k):",
     "            elif False:", ["tests/test_t61_blocks.py"]),
    ("M15' staged training sha compare removed", "scripts/t61_session.py",
     "(None if got == want else", "(None if True else",
     ["tests/test_t61_session.py", "tests/test_t61_rehearsal.py"]),
    ("M16' GT process ready_to_clock removed", "scripts/t31_steps.py",
     "    so.ready_to_clock(receipt.synced_at, so.utc_now(), clock_skew=skew, guard=guard, gap=gap)\n",
     "", ["tests/test_t31_steps.py", "tests/test_t31_session.py", "tests/test_session_order.py"]),
]
for name, f, old, new, tests in M:
    p = os.path.join(ROOT, f)
    s = open(p).read()
    if s.count(old) != 1:
        print(f"{name}: SKIP: pattern found {s.count(old)} times")
        continue
    open(p, "w").write(s.replace(old, new))
    tests = [t for t in tests if os.path.exists(os.path.join(ROOT, t))]
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-x", *tests],
                       cwd=ROOT, capture_output=True, text=True)
    tail = [ln for ln in r.stdout.splitlines() if "passed" in ln or "failed" in ln][-1:]
    print(f"{name}: {'CAUGHT' if r.returncode else 'SURVIVED'} {tail[0] if tail else ''}")
    subprocess.run(["git", "checkout", "--", f], cwd=ROOT, check=True)
