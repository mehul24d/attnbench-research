import subprocess, sys, json
ROOT = sys.argv[1]
M = [
 ("M1 late check drops the skew", "attnbench/predict/commit.py",
  "if synced_at + clock_skew >= start:", "if synced_at >= start:", ["tests/test_predict_commit.py","tests/test_audit_fixes_commit.py"]),
 ("M2 late check removed", "attnbench/predict/commit.py",
  "if synced_at + clock_skew >= start:", "if False:", ["tests/test_predict_commit.py"]),
 ("M3 write-after-sync check removed", "attnbench/predict/commit.py",
  "if cp.written_at > synced_at + clock_skew:", "if False:", ["tests/test_predict_commit.py","tests/test_audit_fixes_commit.py"]),
 ("M4 own pairing banked-after check removed", "attnbench/predict/commit.py",
  "if b - clock_skew <= cp.written_at:\n                raise LatePredictionError(", "if False:\n                raise LatePredictionError(", ["tests/test_predict_commit.py","tests/test_audit_fixes_commit.py"]),
 ("M5 sync read-back compare removed", "attnbench/predict/commit.py",
  "if store.get(remote) != data:", "if False:", ["tests/test_predict_commit.py"]),
 ("M6 locked learned config compare removed", "attnbench/predict/commit.py",
  "if lm.config != locked:", "if False:", ["tests/test_predict_commit.py","tests/test_predict_learned.py"]),
 ("M7 target-in-banked refusal removed (training)", "attnbench/predict/training.py",
  "    if target in banked:\n        raise CostModelError(", "    if False:\n        raise CostModelError(", ["tests/test_predict_training.py","tests/test_predict_commit.py","tests/test_predict_baselines.py"]),
 ("M8 runner: second predict after pass allowed", "scripts/t61_session.py",
  "if done.get(\"predict\") == PASSED and \"predict\" not in imported \\", "if False and done.get(\"predict\") == PASSED and \"predict\" not in imported \\", ["tests/test_t61_session.py"]),
 ("M9 runner: receipt-exists refusal removed", "scripts/t61_session.py",
  "if (session / RECEIPT).exists():\n        raise StepFailed(", "if False:\n        raise StepFailed(", ["tests/test_t61_session.py"]),
 ("M10 strict stamp before-open check removed", "scripts/t61_session.py",
  "if stamp < opened_at:", "if False:", ["tests/test_t61_session.py"]),
 ("M11 gap check in check_stamp removed", "scripts/t61_session.py",
  "if not gap > MIN_GAP_S:", "if False:", ["tests/test_t61_session.py"]),
 ("M12 import refuses discarded block: removed", "scripts/t61_session.py",
  "if (Path(d) / DISCARD).exists():\n        out.append(", "if False:\n        out.append(", ["tests/test_t61_blocks.py"]),
 ("M13 import commit check removed", "scripts/t61_session.py",
  "if m.get(\"code_commit\") != code_commit(dry_run):", "if False:", ["tests/test_t61_blocks.py"]),
 ("M14 block identity check returns nothing", "scripts/t61_session.py",
  "            if mine.get(k) != theirs.get(k):", "            if False:", ["tests/test_t61_blocks.py"]),
 ("M15 staged training sha compare removed", "scripts/t61_session.py",
  "if got != want:\n                bad.append(", "if False:\n                bad.append(", ["tests/test_t61_session.py","tests/test_t61_rehearsal.py"]),
 ("M16 GT process ready_to_clock removed", "scripts/t31_steps.py",
  "    so.ready_to_clock(receipt.synced_at, so.utc_now(), clock_skew=skew, guard=guard)\n    not_before", "    not_before", ["tests/test_t31_steps.py","tests/test_t31_session.py","tests/test_session_order.py"]),
 ("M17 block_drift ignores DISCARD", "attnbench/analysis/block_drift.py",
  "if (d / DISCARD).exists():", "if False:", ["tests/test_t61_blocks.py","tests/test_block_drift.py"]),
 ("M18 interrupted() always False", "scripts/t61_session.py",
  "return was is not None and now is not None and was != now", "return False", ["tests/test_t61_blocks.py","tests/test_t61_session.py","tests/test_t64_session.py"]),
 ("M19 fill_lock_values completeness check removed", "scripts/t31_steps.py",
  "    if not res.complete:\n        raise StepFailed(f\"{p.name} is not a complete lock, so no pairing's", "    if False:\n        raise StepFailed(f\"{p.name} is not a complete lock, so no pairing's", ["tests/test_lock_fill.py"]),
 ("M20 checklist: _unset never true", "attnbench/lock/checklist.py",
  "return v is None or (isinstance(v, (str, list, dict)) and len(v) == 0)", "return False", ["tests/test_lock_checklist.py","tests/test_lock_fill.py"]),
 ("M21 rows_from_record self-sha check removed", "attnbench/predict/p1_training.py",
  "if rec.get(\"sha256\") != _digest(rec):", "if False:", ["tests/test_p1_training.py"]),
 ("M22 recomputed predictions compare removed", "attnbench/predict/commit.py",
  "            if got[n] != v:", "            if False:", ["tests/test_predict_commit.py"]),
]
import os
res = []
for name, f, old, new, tests in M:
    p = os.path.join(ROOT, f); s = open(p).read()
    n = s.count(old)
    if n != 1:
        res.append((name, f"SKIP: pattern found {n} times")); continue
    open(p, "w").write(s.replace(old, new))
    tests = [t for t in tests if os.path.exists(os.path.join(ROOT, t))]
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-x", *tests],
                       cwd=ROOT, capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or ["?"])[-1]
    failed = [l for l in r.stdout.splitlines() if l.startswith("FAILED")][:1]
    subprocess.run(["git", "checkout", "--", f], cwd=ROOT, check=True)
    res.append((name, ("CAUGHT " if r.returncode else "NOT CAUGHT ") + last + (" | " + failed[0][:120] if failed else "") + f" [{', '.join(t.split('/')[-1] for t in tests)}]"))
for a, b in res:
    print(f"{a}: {b}")
