"""EXPLORATORY: the H100 diagnostic (17 Nov 2026 rulings, item 2d): the
mutation drive for tests/test_h100_diag.py. Each mutation is applied alone,
the tests are run, and the file restored; every one must make them fail.
The Q4 recursion mutations need P4's local data (the H100_DIAG_Q4_* variables
the test names); without it they are reported as not run.

    python3 tools/h100_diag_mutations.py [repo root] [--only-q4]

The tests must pass unmutated first (a failing baseline makes every kill
vacuous). A mutated file's original is kept beside it as <file>.mut_orig until
it is restored; a run that finds one refuses, since a killed run (17 Nov 2026)
once left a mutant in place.
"""
import subprocess, sys, os, re
from pathlib import Path
args = [a for a in sys.argv[1:] if a != '--only-q4']
ONLY_Q4 = '--only-q4' in sys.argv
W = Path(args[0]) if args else Path(__file__).resolve().parents[1]
M, S = 'attnbench/exploratory/h100_diag.py', 'scripts/h100_diag.py'
Q4 = 'q4'
MUT = [
 (M, '            missing += 1\n', '            pass\n'),
 (M, '"total": sum(out[c] for c in COMPONENTS)', '"total": sum(out[c] for c in COMPONENTS if c != "copy")'),
 (M, '        if cur_b is None or a > cur_b:', '        if cur_b is None or a >= cur_b + 1:'),
 (M, '            cur_b = max(cur_b, b)', '            cur_b = b'),
 (M, 'xs = sorted((max(a, lo), min(b, hi))', 'xs = sorted((a, b)'),
 (M, 'last = max((b for a, b in launches if a >= start), default=start)', 'last = max((b for a, b in launches), default=start)'),
 (M, 'blocked = _union(blocking, start, last)', 'blocked = _union(blocking, start, end)'),
 (M, '"blocked_s": blocked, "host_s": submit - blocked,', '"blocked_s": blocked, "host_s": submit,'),
 (M, '"idle_share": (wall - busy) / wall,', '"idle_share": busy / wall,'),
 (M, 'if len(regs) != 1:', 'if not regs:'),
 (M, '("cuda_runtime", "cuda_driver") and lo <= e["ts"] <= hi]', '("cuda_runtime",) and lo <= e["ts"] <= hi]'),
 (M, 'end = max([hi] + [b for _, b in device])', 'end = hi'),
 (M, 'return {k: (v * 1e-6 if k.endswith("_s") else v) for k, v in s.items()}', 'return {k: (v * 1e-6 if k != "launches" else v) for k, v in s.items()}'),
 (M, '"StringIds"):\n            if t not in tables:', '"StringIds"):\n            if False:'),
 (M, 'COALESCE(n.text, t.value)', 'n.text'),
 (M, 'if x and x.startswith(prefix)]', 'if x]'),
 (M, '                other += con.execute', '                other = con.execute'),
 (M, 'blocking = [(x, y) for x, y, n in rt if a <= x <= b and _base(n) in BLOCKING]', 'blocking = [(x, y) for x, y, n in rt if a <= x <= b and n in BLOCKING]'),
 (M, '            if a <= x < b:\n', '            if True:\n'),
 (M, '(v * 1e-9 if k.endswith("_s") else v)', '(v * 1e-6 if k.endswith("_s") else v)'),
 (M, 'return statistics.median(float(r[key]) for r in rows)', 'return statistics.fmean(float(r[key]) for r in rows)'),
 (M, 'ok = abs(r - 1.0) <= SAME', 'ok = r <= 1.0 + SAME'),
 (M, '    if abs(g - 1.0) <= SAME:', '    if g <= 1.0 + SAME:'),
 (M, '    elif g >= GRAPH_EXPLAINS:', '    elif g > GRAPH_EXPLAINS:'),
 (M, '"within": abs(a / b - 1.0) <= SAME}', '"within": a / b - 1.0 <= SAME}'),
 (M, '"comparable": bool(rows) and all(', '"comparable": all('),
 (M, '    if phi >= MOST:', '    if phi > MOST:'),
 (M, '    elif phi <= LITTLE:', '    elif phi < LITTLE:'),
 (M, 'phi = (predicted_all_replaced - predicted_host_scaled) / resid', 'phi = (predicted_all_replaced - predicted_host_scaled) / measured'),
 (S, 'if any(k in e.get("name", "").lower() for k in ATTENTION_NAMES):', 'if "flash" in e.get("name", "").lower():'),
 (S, 'if e.get("cat") == "kernel" and e["ts"] + e["dur"] > lo', 'if e["ts"] + e["dur"] > lo'),
 (S, '    if r.returncode != 0:\n        return {"available": True, "failed": f"nsys profile', '    if False:\n        return {"available": True, "failed": f"nsys profile'),
 (S, 'if not args.cpu_boundary and "H100" not in env["gpu"]:', 'if not args.cpu_boundary and "NVIDIA" not in env["gpu"]:'),
 (S, 'if c.arm != "dense" or c.backend == "sdpa_flash" or boundary:', 'if c.arm != "dense" or boundary:'),
 (S, '"attention_priced_s": c["device_s_priced_traced"]["attention"][0],', '"attention_priced_s": c["device_s_priced_traced"]["attention"][1],'),
 (S, 'if c["batch"] == 1 and c["arm"] in', 'if c["arm"] in'),
 (S, 'TOKENS = {8192: 8075, 16384: 16370, 32768: 32700}', 'TOKENS = {8192: 8075, 16384: 16370, 32768: 32768}'),
 (S, 'if m.get("status") == "passed" and (c.get("arm")', 'if (c.get("arm")'),
 (S, 'ref[("dense", 16370)]["attention_priced_s"])', 'ref[("dense", 16370)]["attention_traced_s"])'),
 (S, 'pairs[L][f"dense_fa3_{L}_wall"] = (d["untraced"]["median_s"]', 'pairs[L][f"dense_fa3_{L}_wall"] = (d["torch_profiler"]["host"]["host_s"]'),
 (S, 'src = "nsys" if "median" in ns else', 'src = "nsys" if ns else'),
 (S, 'tr = ns["median"] if "median" in ns else x["torch_profiler"]["host"]', 'tr = x["torch_profiler"]["host"]'),
 # the 17 Nov card edits: Q3 per length, the Nsight overhead
 (M, 'TRACED_WALL_MAX = 1.0 + SAME', 'TRACED_WALL_MAX = 1.0 + 2 * SAME'),
 (M, 'over = max(0.0, traced_wall_s - p4_wall_s)', 'over = traced_wall_s - p4_wall_s'),
 (M, '"host_s": traced_host_s - over,', '"host_s": traced_host_s,'),
 (M, '"comparable": traced_wall_s <= TRACED_WALL_MAX * p4_wall_s}', '"comparable": traced_wall_s < TRACED_WALL_MAX * p4_wall_s}'),
 (S, 'out["Q1"] = _gate(q1, q3_why[16384])', 'out["Q1"] = _gate(q1, None)'),
 (S, 'out["Q1"]["confirmed"] = None', 'pass'),
 (S, 'q3_why = {L: None if out["Q3"][str(L)]["comparable"]', 'q3_why = {L: None if all(r["comparable"] for r in out["Q3"].values())'),
 (S, 'why = [w for w in (q3_why[L], None if', 'why = [w for w in (None, None if'),
 (S, 'hc = HD.q4_host(tr["wall_s"],', 'hc = HD.q4_host(x["untraced"]["median_s"],'),
 (S, '**q4_inputs[n](hc["host_s"])}', '**q4_inputs[n](tr["host_s"])}'),
 (S, 'return {**res, "comparable": False, "verdict": f"not comparable: {why}",', 'return {**res, "verdict": f"not comparable: {why}",'),
 (S, '            k = host / priced_host', '            k = priced_host / host if host else 1.0', Q4),
 (S, '        priced_host = sum(c.priced.host)', '        priced_host = max(c.priced.host)', Q4),
 # not mutated: the mean over a cell's traces (P4 has one trace per XAttention cell, so
 # taking the first is equivalent on its data)
 (S, '        full = BR.substitute(c.priced, dev, fam)', '        full = BR.substitute(c.priced, dev, fam, "attention")', Q4),
 (S, '        pred_all = recurse(full, cp.queue_depth).end_to_end', '        pred_all = recurse(c.priced, cp.queue_depth).end_to_end', Q4),
]
env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
have_q4 = all(env.get(k) for k in ('H100_DIAG_Q4_INPUTS', 'H100_DIAG_Q4_MIRROR', 'H100_DIAG_Q4_TRACES'))
left = [str(p) for f in (M, S) for p in [W / (f + '.mut_orig')] if p.exists()]
if left:
    sys.exit(f'REFUSED: a mutated file was not restored; compare and restore by hand: {left}')


def pytest(sel):
    return subprocess.run([sys.executable, '-m', 'pytest', '-q', '-x', '-p', 'no:cacheprovider', *sel,
                           'tests/test_h100_diag.py'], cwd=W, env=env, capture_output=True, text=True,
                          timeout=600)


for sel in ([] if ONLY_Q4 else [['-k', 'not recursion']]) + ([['-k', 'recursion']] if have_q4 else []):
    b = pytest(sel)
    if b.returncode != 0:
        sys.exit(f'REFUSED: the unmutated baseline {sel} fails:\n{b.stdout[-2000:]}')
survived, not_run = [], []
for i, (f, old, new, *tag) in enumerate(MUT):
    q4 = tag == [Q4]
    if ONLY_Q4 and not q4:
        continue
    if q4 and not have_q4:
        print(f'#{i:2d} NOT RUN (no P4 data) {old[:60]!r}'); not_run.append(i); continue
    p = W / f; src = p.read_text()
    if src.count(old) != 1:
        print(f'#{i} NOT FOUND ({src.count(old)}) {f}: {old[:60]!r}'); survived.append(i); continue
    bak = W / (f + '.mut_orig')
    bak.write_text(src)
    p.write_text(src.replace(old, new))
    try:
        r = pytest(['-k', 'recursion'] if q4 else ['-k', 'not recursion'])
    finally:
        p.write_text(src)
        bak.unlink()
    killed = bool(re.search(r'[0-9]+ failed|[0-9]+ error', r.stdout))
    print(f'#{i:2d} {"killed" if killed else "SURVIVED"} {Path(f).name}: {old[:70]!r}')
    if not killed: survived.append(i)
print('survived', survived, 'not run', not_run, 'of', len(MUT))
sys.exit(1 if survived else 0)
