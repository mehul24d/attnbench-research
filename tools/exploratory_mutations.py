"""EXPLORATORY wave (16 Nov 2026 rulings, item 6): the mutation drive for
tests/test_exploratory_wave.py. Each mutation is applied alone, the test file
is run, and the file restored; every one must make it fail.

    python3 tools/exploratory_mutations.py [repo root]
"""
import subprocess, sys, os
from pathlib import Path
W = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
T, BRK, SDp, BDp, SGp, EWp = ('attnbench/exploratory/traces.py', 'attnbench/exploratory/breakdown.py',
    'attnbench/exploratory/safe_dispatch.py', 'attnbench/exploratory/bound.py', 'attnbench/exploratory/sign.py',
    'scripts/exploratory_wave.py')
MUT = [
 (T, 'return BUDGET[int(L)], int(b)', 'return int(L), int(b)'),
 (T, 'abs(c.density - d) < 1e-9', 'True'),
 (T, 'return emb[0], syncs[0]', 'return emb[0], syncs[-1]'),
 (T, '            if c not in have:', '            if True:'),
 (T, 'and cat and lo <= e["ts"] <= hi', 'and cat'),
 (T, '("cuda_runtime", "cuda_driver")', '("cuda_runtime",)'),
 (T, 'd2.append(None if d is None else d + carry)', 'd2.append(None if d is None else d)'),
 (T, 'if OPKIND[f.split(":", 1)[0]] != k]', 'if False]'),
 (T, '        if carry:\n', '        if False:\n'),
 (T, 'if len(fam) > len(priced_kinds):', 'if False:'),
 (BRK, 'if "attn" in name or "attention" in name:', 'if "attention" in name:'),
 (BRK, '"aten::mm", "aten::addmm",', '"aten::mm",'),
 (BRK, 'if t is None or comp is None or', 'if comp is None or'),
 (BRK, '            link[i] = t', '            dev[i] = t'),
 (BRK, '(only is not None and comp != only)', 'False'),
 (BRK, 'all_replaced=recurse(substitute(priced, traced, families), queue_depth)', 'all_replaced=recurse(substitute(priced, traced, families, "gemm"), queue_depth)'),
 (BRK, 'return statistics.median(abs(p / m - 1.0)', 'return statistics.fmean(abs(p / m - 1.0)'),
 (BRK, 'out[comp][0] += priced.link[i] if comp == "copy" else priced.device[i]', 'out[comp][0] += priced.device[i]'),
 (BRK, '[(s.all_replaced, s.measured) for s in splits]', '[(s.predicted, s.measured) for s in splits]'),
 (SDp, 'if k.arm != "dense" and k.arm not in NOT_DEPLOYABLE)', 'if k.arm != "dense")'),
 (SDp, 'return best, gap, gap / p.dense.predicted', 'return best, gap, gap / best.predicted'),
 (SDp, 'return ch if rel > margin else p.dense.config', 'return ch if rel >= margin else p.dense.config'),
 (SDp, '        if m[ch] > m[p.dense.config]:', '        if m[ch] < m[p.dense.config]:'),
 (SDp, '        if p.point.batch not in batches:\n            continue\n        ch = DP.dispatcher', '        ch = DP.dispatcher'),
 (SDp, 'out.append((p.point, c.config, a_p, a_m, a_p - a_m))', 'out.append((p.point, c.config, a_p, a_m, abs(a_p - a_m)))'),
 (SDp, 'return max(0.0, max(e[4] for e in errs))', 'return max(e[4] for e in errs)'),
 (SDp, '        if td - m[best] > floor_s:', '        if td - m[best] > 0:'),
 (SDp, 'kept += ch != p.dense.config and td - m[ch] > floor_s', 'kept += ch != p.dense.config'),
 (SDp, 'reg.append(m[ch] / m[best] - 1.0)', 'reg.append(m[ch] / td - 1.0)'),
 (SDp, 'rec, step = points_for(cells, "recursion"), points_for(cells, "step_form")', 'rec, step = points_for(cells, "recursion"), points_for(cells, "recursion")'),
 (SDp, '"bunga_abstain": (rec, {p.point: abstaining(p, admissible, floors, margin)', '"bunga_abstain": (rec, {p.point: DP.dispatcher(p, admissible, floors)'),
 (BDp, 'f = sum(v[0] for k, v in fb.items() if k != "lm_head") * n_layers + fb["lm_head"][0]', 'f = sum(v[0] for k, v in fb.items()) * n_layers'),
 (BDp, 'b = sum(v[1] for k, v in fb.items() if k != "lm_head") * n_layers + fb["lm_head"][1]', 'b = sum(v[1] for k, v in fb.items() if k != "lm_head") * n_layers'),
 (BDp, 'return Bound(phi_c=max(', 'return Bound(phi_c=min('),
 (BDp, 'tau=min(o.measured / o.n_ops for o in train))', 'tau=max(o.measured / o.n_ops for o in train))'),
 (BDp, '"status": "break" if r < 1 else ("close" if r < 1 + close else "clear")', '"status": "break" if r < 1 else ("close" if r <= 1 else "clear")'),
 (BDp, '"host": o.n_ops * self.tau}', '"host": self.tau}'),
 (BDp, '"memory": o.bytes / (self.phi_m * bw)', '"memory": o.bytes / (self.phi_m * p)'),
 (SGp, 'reps = np.median(s[idx], axis=1) - np.median(d[idx], axis=1)', 'reps = np.median(s[rng.integers(0, len(d), size=(n_boot, len(d)))], axis=1) - np.median(d[idx], axis=1)'),
 (SGp, 'beyond_floor=_verdict(lo, hi, floor_s))', 'beyond_floor=_verdict(lo, hi, 0.0))'),
 (SGp, 'p_sparse_faster=float(np.mean(reps < 0))', 'p_sparse_faster=float(np.mean(reps <= 0))'),
 (SGp, '    if result.get("rounds_run") != result.get("clean_rounds"):', '    if False:'),
 (SGp, '    a = (1 - level) / 2', '    a = (1 - level)'),
 (SGp, 'diff_s=float(np.median(s) - np.median(d))', 'diff_s=float(np.mean(s) - np.mean(d))'),
 (EWp, '    if round(m, 5) != RECORDED_MARGIN:', '    if False:'),
 (EWp, '    m = m0 if m0 > 0 else SD.fallback_margin(pts, meas, adm)', '    m = SD.fallback_margin(pts, meas, adm)'),
 (EWp, 'if "gs://" in text or (name and name in text):', 'if "gs://" in text:'),
 (EWp, '               else statistics.fmean(a[1][i] for a in al) for i in range(len(fam))]', '               else al[0][1][i] for i in range(len(fam))]'),
 (EWp, 'dev = [None if any(a[1][i] is None for a in al)', 'dev = [None if all(a[1][i] is None for a in al)'),
 (EWp, 'if any(a[0] != fam for a in al):', 'if False:'),
 (EWp, 'for c in cp1.cells if c.arm != "dense"]', 'for c in cp1.cells]'),
 (EWp, 'why = SG.pairable(result) or', 'why = None or'),
]
env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
import re
survived = []
for i, (f, old, new) in enumerate(MUT):
    p = W / f; src = p.read_text()
    if src.count(old) != 1:
        print(f'#{i} NOT FOUND ({src.count(old)}) {f}: {old[:60]!r}'); survived.append(i); continue
    p.write_text(src.replace(old, new))
    try:
        r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-x', '-p', 'no:cacheprovider', 'tests/test_exploratory_wave.py'],
                           cwd=W, env=env, capture_output=True, text=True, timeout=300)
    finally:
        p.write_text(src)
    killed = bool(re.search(r'[0-9]+ failed|[0-9]+ error', r.stdout))
    print(f'#{i:2d} {"killed" if killed else "SURVIVED"} {Path(f).name}: {old[:70]!r}')
    if not killed: survived.append(i)
print('survived', survived, 'of', len(MUT))
sys.exit(1 if survived else 0)
