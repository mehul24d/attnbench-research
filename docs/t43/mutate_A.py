import subprocess, sys, pathlib
ROOT = pathlib.Path("mut")
T = lambda *n: [f"tests/{x}.py" for x in n]
MUTS = [
 # id, file, old, new, tests
 ("H1 holm: no step-down (level=alpha)", "attnbench/lock/holm.py", "level = alpha / (m - i)", "level = alpha", T("test_lock_holm")),
 ("H2 holm: Bonferroni (alpha/m)", "attnbench/lock/holm.py", "level = alpha / (m - i)", "level = alpha / m", T("test_lock_holm")),
 ("H3 holm: no stop after first failure", "attnbench/lock/holm.py", "ok = (not stopped) and _clears(", "ok = _clears(", T("test_lock_holm")),
 ("H4 holm: order by descending p", "attnbench/lock/holm.py", "ranked = sorted(((ni_p_value", "ranked = sorted(reverse=True, iterable=((ni_p_value", T("test_lock_holm")),
 ("S1 scoring: sign_agreement floor strict (<)", "attnbench/predict/scoring.py", "if abs(m) <= floor:", "if abs(m) < floor:", T("test_predict_crossover","test_gate_a","test_lock_hdbi_fit","test_predict_dispatcher")),
 ("S2 scoring: measured_profitable dense-sparse >= floor", "attnbench/predict/scoring.py", "if dense - sparse > floor:", "if dense - sparse >= floor:", T("test_predict_crossover","test_gate_a","test_predict_dispatcher")),
 ("S3 scoring: predicted tie profitable (<=)", "attnbench/predict/scoring.py", "return bool(admissible) and sparse < dense", "return bool(admissible) and sparse <= dense", T("test_predict_crossover","test_gate_a","test_predict_dispatcher")),
 ("S4 scoring: sign tie goes to sparse", "attnbench/predict/scoring.py", "return (predicted_sparse < predicted_dense) == (m < 0)", "return (predicted_sparse <= predicted_dense) == (m < 0)", T("test_predict_crossover","test_gate_a")),
 ("D1 dispatcher: sparse if gain >= floor", "attnbench/predict/dispatcher.py", "if p.dense.predicted - best.predicted > floor_s", "if p.dense.predicted - best.predicted >= floor_s", T("test_predict_dispatcher")),
 ("D2 dispatcher: ignore floor", "attnbench/predict/dispatcher.py", "if p.dense.predicted - best.predicted > floor_s", "if p.dense.predicted - best.predicted > 0", T("test_predict_dispatcher")),
 ("D3 dispatcher: kernel rule s>=1", "attnbench/predict/dispatcher.py", "if s > 1]", "if s >= 1]", T("test_predict_dispatcher")),
 ("D4 dispatcher: regret vs chosen-best off by dense tie", "attnbench/predict/dispatcher.py", "min(have, key=lambda c: (measured[c], c != p.dense.config, c))", "min(have, key=lambda c: (measured[c], c == p.dense.config, c))", T("test_predict_dispatcher")),
 ("D5 dispatcher: missed-profitable gap >= floor", "attnbench/predict/dispatcher.py", "if gap > floor_s:\n        return MISSED_PROFITABLE", "if gap >= floor_s:\n        return MISSED_PROFITABLE", T("test_predict_dispatcher")),
 ("D6 dispatcher: lengths filter removed", "attnbench/predict/dispatcher.py", "if keep is not None and p.point.context_length not in keep:", "if False:", T("test_predict_dispatcher")),
 ("B1 block_drift: bound >=", "attnbench/analysis/block_drift.py", "> bound]", ">= bound]", T("test_t61_blocks")),
 ("B2 block_drift: claimable drops flagged blocks", "attnbench/analysis/block_drift.py", "    return list(rows)", "    return [r for r in rows if r.get('block') not in set(result.get('above') or ())]", T("test_t61_blocks")),
 ("B3 block_drift: relative to last block", "attnbench/analysis/block_drift.py", "ref = by_block[present[0]]", "ref = by_block[present[-1]]", T("test_t61_blocks")),
 ("B4 block_drift: abs() removed", "attnbench/analysis/block_drift.py", "if abs(per[b][\"relative_difference\"]) > bound]", "if per[b][\"relative_difference\"] > bound]", T("test_t61_blocks")),
 ("F1 floors: ignore between-session", "attnbench/lock/floors.py", "floor = w.floor_ms if between is None else max(w.floor_ms, between)", "floor = w.floor_ms", T("test_lock_floors","test_resolution_floors","test_gate_a_repeats")),
 ("F2 floors: between uses min not max", "attnbench/lock/floors.py", "ranges = [max(m.values()) - min(m.values())", "ranges = [min(m.values()) - min(m.values())", T("test_lock_floors","test_resolution_floors","test_gate_a_repeats")),
 ("F3 floors: pooled deviations from mean not median", "attnbench/lock/floors.py", "med = statistics.median(xs)\n        pool.extend", "med = statistics.mean(xs)\n        pool.extend", T("test_lock_floors","test_resolution_floors","test_gate_a_repeats")),
 ("F4 floors: quantile 0.95 -> max", "attnbench/lock/floors.py", "floor_ms=float(np.quantile(pool, q))", "floor_ms=float(max(pool))", T("test_lock_floors","test_resolution_floors","test_gate_a_repeats")),
 ("G1 gate_a: MIN_CLEAN_ROUNDS 10->3", "attnbench/analysis/gate_a.py", "MIN_CLEAN_ROUNDS = 10", "MIN_CLEAN_ROUNDS = 3", T("test_gate_a","test_gate_a_repeats")),
 ("G2 gate_a: recursion_below non-strict", "attnbench/analysis/gate_a.py", "{p: rec < by[p][\"median_abs_relative\"]", "{p: rec <= by[p][\"median_abs_relative\"]", T("test_gate_a","test_gate_a_repeats")),
 ("G3 gate_a: exempt family <= floor", "attnbench/analysis/gate_a.py", "if r[\"trace_s\"] < floor_s", "if r[\"trace_s\"] <= floor_s", T("test_gate_a")),
 ("G4 gate_a: exempt nothing", "attnbench/analysis/gate_a.py", "if r[\"trace_s\"] < floor_s", "if False", T("test_gate_a")),
 ("G5 gate_a: slope range open end", "attnbench/analysis/gate_a.py", "lo <= got[m] <= hi", "lo <= got[m] <= hi * 2", T("test_gate_a")),
 ("G6 gate_a_repeats: leg1 only any session", "attnbench/analysis/gate_a_repeats.py", "ok = all(all(v.values()) for v in per.values())", "ok = any(all(v.values()) for v in per.values())", T("test_gate_a_repeats")),
 ("G7 gate_a_repeats: gate passes if any leg", "attnbench/analysis/gate_a_repeats.py", "passes = all(v[\"verdict\"] == HOLDS for v in legs.values())", "passes = any(v[\"verdict\"] == HOLDS for v in legs.values())", T("test_gate_a_repeats")),
 ("G8 gate_a_repeats: provisional bound applied", "attnbench/analysis/gate_a_repeats.py", "if bound is None or provisional:", "if bound is None:", T("test_gate_a_repeats")),
 ("R1 replay.check_passes: > bound -> >= ", "attnbench/components/replay.py", "if d is None or abs(d) > bound:", "if d is None or abs(d) >= bound:", T("test_components_replay","test_gate_a")),
 ("R2 replay.check_passes: None passes", "attnbench/components/replay.py", "if d is None or abs(d) > bound:", "if d is not None and abs(d) > bound:", T("test_components_replay","test_gate_a")),
 ("R3 replay.check_passes: no-replay positions ignored", "attnbench/components/replay.py", "if report[\"positions_without_replay\"]:", "if False:", T("test_components_replay","test_gate_a")),
 ("R4 replay.check_passes: structure mismatch ignored", "attnbench/components/replay.py", "if km and (km[\"unmatched_positions\"] or km[\"extra_nodes\"]):", "if False:", T("test_components_replay","test_gate_a")),
 ("X1 crossover: sign_at point >= floor", "attnbench/predict/crossover.py", "if d.lo > 0 and d.point > floor:", "if d.lo > 0 and d.point >= floor:", T("test_predict_crossover","test_crossover")),
 ("X2 crossover: interval ignored (+)", "attnbench/predict/crossover.py", "if d.lo > 0 and d.point > floor:", "if d.point > floor:", T("test_predict_crossover","test_crossover")),
 ("X3 crossover: interval ignored (-)", "attnbench/predict/crossover.py", "if d.hi < 0 and d.point < -floor:", "if d.point < -floor:", T("test_predict_crossover","test_crossover")),
 ("X4 crossover: UNRESOLVED -> none in range", "attnbench/predict/crossover.py", "    elif len(determined) == len(signs):\n        status = NONE_IN_RANGE", "    elif True:\n        status = NONE_IN_RANGE", T("test_predict_crossover","test_crossover")),
 ("X5 crossover_error: contains off by one", "attnbench/predict/crossover.py", "contains=p.lower_index <= m.lower_index and m.upper_index <= p.upper_index", "contains=p.lower_index < m.lower_index and m.upper_index <= p.upper_index", T("test_predict_crossover","test_crossover")),
 ("X6 crossover_error: direction ignored", "attnbench/predict/crossover.py", "direction_agrees=p.direction == m.direction", "direction_agrees=True", T("test_predict_crossover","test_crossover")),
 ("E1 exact NI: alpha not halved", "attnbench/analysis/exact_noninferiority.py", "return cp_lower(b, n, alpha / 2) - cp_upper(c, n, alpha / 2)", "return cp_lower(b, n, alpha) - cp_upper(c, n, alpha)", T("test_exact_noninferiority","test_lock_holm","test_predict_dispatcher")),
 ("E2 exact NI: lower >= -margin", "attnbench/analysis/exact_noninferiority.py", "non_inferior=lower > -margin_pts", "non_inferior=lower >= -margin_pts", T("test_exact_noninferiority")),
 ("E3 holm _clears >= ", "attnbench/lock/holm.py", "return 100 * paired_lower_bound(cnt.b, cnt.c, cnt.n, alpha) > -margin_pts", "return 100 * paired_lower_bound(cnt.b, cnt.c, cnt.n, alpha) >= -margin_pts", T("test_lock_holm")),
 ("N1 hdbi_fit: ties to higher", "attnbench/lock/hdbi_fit.py", "threshold = min(t for t, a in curve if a == best)", "threshold = max(t for t, a in curve if a == best)", T("test_lock_hdbi_fit")),
 ("N2 hdbi_fit: floor inclusive off", "attnbench/lock/hdbi_fit.py", "if abs(d) <= floor:", "if abs(d) < floor:", T("test_lock_hdbi_fit")),
 ("N3 residuals: inside_floor strict", "attnbench/predict/residuals.py", "inside_floor=abs(rel) <= floor", "inside_floor=abs(rel) < floor", T("test_gate_a","test_predict_sensitivity","test_predict_crossover")),
 ("N4 residuals: sync-free class when n<=1", "attnbench/predict/residuals.py", "return SYNC_FREE if n_sync == 0 else SYNCHRONISING", "return SYNC_FREE if n_sync <= 1 else SYNCHRONISING", T("test_gate_a","test_predict_sensitivity")),
 ("I1 intervals: seed ignored", "attnbench/predict/intervals.py", "rng = np.random.Generator(np.random.PCG64(seed))", "rng = np.random.Generator(np.random.PCG64(0))", T("test_predict_intervals")),
 ("I2 intervals: wrong quantiles", "attnbench/predict/intervals.py", "[(1 - level) / 2, (1 + level) / 2]", "[(1 - level), level]", T("test_predict_intervals")),
 ("I3 intervals: point = replicate mean", "attnbench/predict/intervals.py", "return Interval(point=float(point),", "return Interval(point=float(np.mean(r)),", T("test_predict_intervals")),
 ("K1 baselines: step_form drops sync tail", "attnbench/predict/baselines.py", "return float(body + plain_sum(priced.sync_cost))", "return float(body)", T("test_predict_baselines","test_gate_a")),
 ("K2 baselines: step form max->sum", "attnbench/predict/baselines.py", "body = plain_sum(max(h, ld) for _, h, ld in steps(priced))", "body = plain_sum(h + ld for _, h, ld in steps(priced))", T("test_predict_baselines","test_gate_a")),
 ("K3 baselines: sum adds s0", "attnbench/predict/baselines.py", "return float(plain_sum(priced.host) + plain_sum(priced.link) + plain_sum(priced.device))", "return float(plain_sum(priced.host) + plain_sum(priced.link) + plain_sum(priced.device) + plain_sum(priced.sync_cost))", T("test_predict_baselines","test_gate_a")),
 ("K4 baselines: fitted sum nonneg removed", "attnbench/predict/baselines.py", "bounds = [(0, None) if j in identified else (0, 0) for j in range(3)]", "bounds = [(None, None) if j in identified else (0, 0) for j in range(3)]", T("test_predict_baselines")),
 ("K5 baselines: LP loss squared-ish (abs removed from lower side)", "attnbench/predict/baselines.py", "A_ub = np.vstack([np.hstack([X, -eye]), np.hstack([-X, -eye])])", "A_ub = np.vstack([np.hstack([X, -eye]), np.hstack([X, -eye])])", T("test_predict_baselines")),
 ("V1 naive: nearest tie not refused", "attnbench/predict/naive.py", "if len(hits) > 1:", "if False:", T("test_predict_naive")),
 ("V2 naive: nearest host before GPU", "attnbench/predict/naive.py", "for rule, same in ((\"same GPU\", lambda p: p.gpu == t.gpu),\n                       (\"same host platform\",\n                        lambda p: p.host_platform == t.host_platform)):", "for rule, same in ((\"same host platform\", lambda p: p.host_platform == t.host_platform),\n                       (\"same GPU\", lambda p: p.gpu == t.gpu)):", T("test_predict_naive")),
 ("V3 naive: constant host cost sets syncs too", "attnbench/predict/naive.py", "host = tuple(0.0 if s else float(constant) for s in priced.is_sync)", "host = tuple(float(constant) for s in priced.is_sync)", T("test_predict_naive")),
 ("V4 naive: median -> mean launch", "attnbench/predict/naive.py", "return float(statistics.median(xs))", "return float(statistics.mean(xs))", T("test_predict_naive")),
]
only = sys.argv[1:]
res = []
for mid, f, old, new, tests in MUTS:
    if only and mid.split()[0] not in only: continue
    p = ROOT / f
    src = p.read_text()
    if src.count(old) != 1:
        res.append((mid, "NOT-APPLICABLE (old text count %d)" % src.count(old))); continue
    p.write_text(src.replace(old, new))
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *tests],
                           cwd=ROOT, capture_output=True, text=True, timeout=300)
        tail = [l for l in r.stdout.splitlines() if "passed" in l or "failed" in l or "error" in l.lower()][-1:]
        res.append((mid, ("KILLED" if r.returncode != 0 else "SURVIVED") + " " + (tail[0] if tail else "")))
    finally:
        p.write_text(src)
for m, s in res: print(f"{m:62s} {s}")
