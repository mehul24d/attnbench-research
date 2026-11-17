"""EXPLORATORY: the H100 diagnostic session (adviser's rulings of 17 Nov 2026,
item 2). It explains P4's miss and scores nothing; it is not on the timed path
(the pin `0e0fb012` is unchanged) and nothing it writes is a ground-truth row.

    python3 scripts/h100_diag.py run --cells configs/exploratory/h100_diag_cells.json \\
        --out results/h100_diag_<UTC>            # on the H100, under session_exit.sh
    python3 scripts/h100_diag.py run ... --cpu-boundary   # the pre-launch CPU pass
    python3 scripts/h100_diag.py verdict --manifests <pulled dir> --inputs <inputs.json> \\
        --mirror <dir> --traces '<P4 GT trace glob>' --out <file.json>   # on the Mac

`run`, per cell in the list's order, each cell's result in
`cell_<id>_manifest.json` (a name the teardown pulls) and the session's in
`diag_manifest.json`:

  - eager: warm-up, then the untraced wall time of `--reps` prefills, then one
    prefill under torch.profiler (T2.3's tracer, the one P4's sequences were
    recorded with) for the device seconds per component and the Kineto host
    timeline, then `--prefills` prefills in a child process under Nsight
    Systems (`one`) for the host timeline without the profiler's overhead;
  - graph: the arm recorded as step 5 records it, captured and replayed as a
    CUDA graph as step 6 prices it (soak, profiled replay, structural match),
    for the device seconds per component in the replay, and the replay's
    untraced time.

A cell that fails is recorded and the next one runs; nothing reruns. Cells
not started by `--budget-min` are recorded as skipped.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attnbench.exploratory import h100_diag as HD  # noqa: E402

ATTENTION_NAMES = ("flash", "fmha", "attn", "attention", "block_sparse")
SOAK_S = 5.0


def _cuda() -> bool:
    import torch
    return torch.cuda.is_available()


def _sync() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def environment(args) -> dict:
    import torch
    from attnbench.analysis import trace_checks as tc
    from attnbench.components.shakedown_arms import libraries
    nsys = os.environ.get("H100_DIAG_NSYS") or tc.nsys_available()
    env = {"cuda": _cuda(), "gpu": torch.cuda.get_device_name(0) if _cuda() else "cpu",
           "libraries": [list(x) for x in libraries()], "nsys": nsys, "nsys_version": None,
           "cpu_boundary": bool(args.cpu_boundary)}
    if nsys:
        r = subprocess.run([nsys, "--version"], capture_output=True, text=True)
        env["nsys_version"] = (r.stdout or r.stderr).strip() or None
    return env


def dense_backend(c: HD.Cell, boundary: bool):
    """The cell's dense backend: FA3 as ground truth's K2 control builds it,
    or None for the harness's own (`sdpa_flash`, what steps 5 and 6 record
    and price). Behind the CPU boundary there is no FA3: the harness's
    backend stands in, and the manifest says so."""
    if c.arm != "dense" or c.backend == "sdpa_flash" or boundary:
        return None
    from attnbench import dense_selection as ds
    be = ds.registry_candidates("H100").get("fa3")
    if be is None:
        raise HD.DiagError("the fa3 backend is not available on this instance")
    return be


def untraced(run, reps: int) -> dict:
    xs = []
    for _ in range(reps):
        _sync()
        t0 = time.perf_counter()
        run()
        _sync()
        xs.append(time.perf_counter() - t0)
    return {"reps": reps, "median_s": statistics.median(xs), "min_s": min(xs), "max_s": max(xs)}


def by_name(trace: dict, lo: float, hi: float) -> dict:
    """Kineto kernel seconds in [lo, hi] µs whose names look like attention,
    and in all (a cross-check that needs no sequence)."""
    att = tot = 0.0
    for e in trace["traceEvents"]:
        if e.get("cat") == "kernel" and e["ts"] + e["dur"] > lo and e["ts"] < hi:
            d = e["dur"] * 1e-6
            tot += d
            if any(k in e.get("name", "").lower() for k in ATTENTION_NAMES):
                att += d
    return {"attention_s": att, "all_s": tot}


def eager_cell(h, c, a, out: Path, args) -> dict:
    from attnbench.components import op_sequence as ops
    from attnbench.components.trace_recording import record_prefill_trace
    res = {"untraced": untraced(a.run_prefill, args.reps)}
    path = record_prefill_trace(a.run_prefill, out / f"trace_{c.id}.json", layers=a.layers,
                                warmup=1)
    tj = ops.load_trace(path)
    seq = ops.parse_trace(tj, a.code_path, context_length=a.length, batch=c.batch)
    try:
        dd, why = ops.device_durations(tj, seq), None
    except ops.SequenceError as e:     # a dropped Kineto record: by name only
        dd, why = {}, str(e)
    fam = [ops.operation_family(o) for o in seq.ops]
    host = HD.kineto_host(tj, ops.REGION)
    res["torch_profiler"] = {"components_s": HD.component_seconds(fam, dd),
                             "durations_problem": why, "n_ops": len(seq),
                             "by_name": by_name(tj, 0.0, float("inf")), "host": host}
    res["nsys"] = nsys_pass(c, out, args)
    return res


def nsys_pass(c, out: Path, args) -> dict:
    """`one` under Nsight Systems, its export read back (`nsys_prefills`);
    a failure is recorded, not raised (the torch.profiler numbers stand)."""
    nsys = os.environ.get("H100_DIAG_NSYS") or _nsys()
    if not nsys:
        return {"available": False}
    base = out / f"nsys_{c.id}"
    child = [sys.executable, str(Path(__file__).resolve()), "one", "--cells", str(args.cells),
             "--cell", c.id, "--warmup", str(args.warmup), "--prefills", str(args.prefills)]
    if args.cpu_boundary:
        child.append("--cpu-boundary")
    cmd = [nsys, "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
           "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
           "--force-overwrite=true", "-o", str(base), *child]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=args.nsys_timeout_s)
    (out / f"nsys_{c.id}.log").write_text(r.stdout[-20000:] + "\n--- stderr\n" + r.stderr[-20000:])
    if r.returncode != 0:
        return {"available": True, "failed": f"nsys profile rc={r.returncode}"}
    db = base.with_suffix(".sqlite")
    e = subprocess.run([nsys, "export", "--type=sqlite", "--force-overwrite=true",
                        f"--output={db}", str(base.with_suffix(".nsys-rep"))],
                       capture_output=True, text=True, timeout=args.nsys_timeout_s)
    if e.returncode != 0:
        return {"available": True, "failed": f"nsys export rc={e.returncode}: {e.stderr[-500:]}"}
    try:
        rows = HD.nsys_prefills(db)
    except (HD.DiagError, Exception) as x:    # recorded: the torch.profiler numbers stand
        return {"available": True, "failed": f"reading the export: {type(x).__name__}: {x}"}
    att = [sum(v for k, v in r["kernels_s"].items()
               if any(n in k.lower() for n in ATTENTION_NAMES)) for r in rows]
    return {"available": True, "seconds": round(time.time() - t0, 1), "prefills": len(rows),
            "median": {k: HD.median_of(rows, k) for k in
                       ("wall_s", "device_busy_s", "idle_s", "idle_share", "submit_s",
                        "blocked_s", "host_s")},
            "attention_by_name_median_s": statistics.median(att), "rows": rows}


def _nsys():
    from attnbench.analysis import trace_checks as tc
    return tc.nsys_available()


# FA3's forward, which the stand-in's table of GPU custom kernels lacks: it
# launches its kernel itself beside the `zero_` that clears its accumulator,
# and the stand-in took that `zero_` for its device work (the diagnostic of
# 17 Nov 2026, D16_fa3_graph: 28 of 4175 calls). `cpu_boundary.py` is in the
# timed-path pin, so the table is widened here, for this script's recording
# only.
FA3_FWD = "flash_attn_3::fwd"


@contextlib.contextmanager
def fa3_launch_table():
    from attnbench import cpu_boundary
    old = cpu_boundary.GPU_CUSTOM_KERNELS
    cpu_boundary.GPU_CUSTOM_KERNELS = old + (FA3_FWD,)
    try:
        yield
    finally:
        cpu_boundary.GPU_CUSTOM_KERNELS = old


def graph_backend(boundary: bool):
    from attnbench.components import replay
    if boundary:
        from attnbench.cpu_boundary import GraphLikeBackend
        return GraphLikeBackend()
    return replay.CudaGraphBackend()


def graph_cell(h, c, a, out: Path, args) -> dict:
    """Step 6's pricing path for one cell: recorded as step 5 records it,
    captured, soaked, replayed under the profiler and matched by structure."""
    from attnbench.components import op_sequence as ops
    from attnbench.components import recorder, replay
    dev = "cpu" if args.cpu_boundary else "cuda"
    with fa3_launch_table():
        rec = recorder.record_arm_prefill(a.run_prefill, out / f"rec_{c.id}", code_path=a.code_path,
                                          context_length=a.length, batch=c.batch,
                                          prompt_id=a.prompt_id, layers=a.layers)
    fam = [ops.operation_family(o) for o in rec.sequence.ops]
    etj = ops.load_trace(rec.directory / "trace.json")
    eager = HD.component_seconds(fam, ops.device_durations(etj, rec.sequence))
    cap = replay.capture(a.run_prefill, rec.tape, device=dev, masks=rec.masks,
                         backend=graph_backend(args.cpu_boundary))
    t0, n = time.monotonic(), 0
    while n < 2 or time.monotonic() - t0 < SOAK_S:
        cap.replay()
        cap.backend.synchronize()
        n += 1
    path = replay.profile_replay(cap, out / f"replay_{c.id}.json")
    m = replay.match(rec.sequence, replay.replay_nodes(path))
    gtj = ops.load_trace(path)
    return {"recording_eager_components_s": eager,
            "recording_eager_by_name": by_name(etj, 0.0, float("inf")),
            "graph_components_s": HD.component_seconds(fam, m.device_s),
            "graph_by_name": by_name(gtj, 0.0, float("inf")),
            "match": {"complete": m.complete, "unmatched_positions": len(m.unmatched_positions),
                      "extra_nodes": len(m.extra_nodes), "extra_s": m.extra_s,
                      "families_unmatched": m.families_unmatched},
            "soak": {"replays": n, "seconds": round(time.monotonic() - t0, 3)},
            "replay_untraced": replay.time_replay(cap, args.reps)}


def run_cell(h, c, out: Path, args) -> dict:
    be = dense_backend(c, args.cpu_boundary)
    a = h.prepare(c.arm, c.length, backend=be, batch=c.batch)
    for _ in range(args.warmup):
        a.run_prefill()
    _sync()
    a.code_path = h.code_path(a)
    head = {"cell": vars(c), "prompt_tokens": a.length, "implementation": a.code_path.implementation,
            "backend_used": type(a.backend).__name__,
            "stand_in": bool(args.cpu_boundary and c.arm == "dense" and c.backend == "fa3")}
    if c.arm == "xattention" and not args.cpu_boundary and a.code_path.implementation != "xattn:triton":
        raise HD.DiagError(f"{c.id}: XAttention took {a.code_path.implementation}, not Triton")
    body = graph_cell(h, c, a, out, args) if c.mode == "graph" else eager_cell(h, c, a, out, args)
    return {**head, **body}


def _harness(args):
    from attnbench.components.shakedown_arms import Harness
    return Harness(device="cpu" if args.cpu_boundary else "cuda", dry_run=bool(args.cpu_boundary))


def _write(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def cmd_run(args) -> int:
    cells, sha = HD.load_cells(args.cells)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    env = environment(args)
    if not args.cpu_boundary and "H100" not in env["gpu"]:
        print(f"REFUSING: {env['gpu']} is not an H100", file=sys.stderr)
        return 2
    man = {"label": "EXPLORATORY", "rulings": "adviser's rulings of 17 Nov 2026, item 2",
           "cells_file": str(args.cells), "cells_sha256": sha, "environment": env,
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start)), "cells": {}}
    _write(out / "diag_manifest.json", man)
    h = _harness(args)
    for c in cells:
        if (time.time() - t_start) / 60 > args.budget_min:
            man["cells"][c.id] = {"status": "skipped", "why": f"past the {args.budget_min} min budget"}
            _write(out / "diag_manifest.json", man)
            continue
        t0 = time.time()
        print(f"[h100_diag] {c.id} start", flush=True)
        try:
            res = {"status": "passed", **run_cell(h, c, out, args)}
        except Exception as e:
            traceback.print_exc()
            res = {"status": "failed", "cell": vars(c), "error": f"{type(e).__name__}: {e}"}
        h.unwrap()
        res["seconds"] = round(time.time() - t0, 1)
        _write(out / f"cell_{c.id}_manifest.json", res)
        man["cells"][c.id] = {"status": res["status"], "seconds": res["seconds"],
                              **({"error": res["error"]} if "error" in res else {})}
        _write(out / "diag_manifest.json", man)
        print(f"[h100_diag] {c.id} {res['status']} {res['seconds']} s", flush=True)
    man["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write(out / "diag_manifest.json", man)
    return 0 if all(v["status"] == "passed" for v in man["cells"].values()) else 1


def cmd_one(args) -> int:
    """The process Nsight Systems traces: one cell, warmed, then `--prefills`
    prefills each in an NVTX range closed after its synchronise, inside the
    profiler API's capture range."""
    import torch
    cells, _ = HD.load_cells(args.cells)
    c = next(x for x in cells if x.id == args.cell)
    h = _harness(args)
    a = h.prepare(c.arm, c.length, backend=dense_backend(c, args.cpu_boundary), batch=c.batch)
    for _ in range(args.warmup):
        a.run_prefill()
    _sync()
    gpu = torch.cuda.is_available()
    if gpu:
        torch.cuda.profiler.start()
    for i in range(args.prefills):
        if gpu:
            torch.cuda.nvtx.range_push(f"{HD.NVTX_PREFIX}{i}")
        a.run_prefill()
        _sync()
        if gpu:
            torch.cuda.nvtx.range_pop()
    if gpu:
        torch.cuda.profiler.stop()
    return 0


# --- verdict, on the Mac ----------------------------------------------------------------

def p4_reference(path: Path) -> dict:
    """P4's numbers the rules read, from the exploratory wave's committed
    results: per (arm, prompt tokens) at batch 1, the measured seconds and
    the priced and traced attention seconds."""
    d = json.loads(Path(path).read_text())
    out = {}
    for c in d["item2_breakdown"]["P4"]["cells"]:
        if c["batch"] == 1 and c["arm"] in ("dense", "xattention"):
            out[(c["arm"], c["context_length"])] = {
                "measured_s": c["measured_s"],
                "attention_priced_s": c["device_s_priced_traced"]["attention"][0],
                "attention_traced_s": c["device_s_priced_traced"]["attention"][1],
                "rel_error_all_device_replaced": c["rel_error_all_device_replaced"]}
    return out


TOKENS = {8192: 8075, 16384: 16370, 32768: 32700}


def _find(man: dict, arm, backend, length, mode):
    for m in man.values():
        c = m.get("cell", {})
        if m.get("status") == "passed" and (c.get("arm"), c.get("backend"), c.get("length"),
                                            c.get("mode")) == (arm, backend, length, mode):
            return m
    return None


def _att(m) -> float:
    return m["torch_profiler"]["components_s"]["attention"]


def _read(rule, *xs):
    try:
        return rule(*xs)
    except HD.DiagError as e:
        return f"not read: {e}"


def _gate(res, why):
    """A rule read against P4's numbers, marked not comparable for `why`;
    the ungated reading is kept beside it."""
    if not isinstance(res, dict) or why is None:
        return res
    return {**res, "comparable": False, "verdict": f"not comparable: {why}",
            "ungated_verdict": res.get("verdict")}


def verdict(man: dict, ref: dict, q4_inputs=None) -> dict:
    """The four rules over the pulled cell manifests (card, "What each
    outcome means"); a rule whose cells did not pass is "not read". Q3 is
    read per length and gates Q1 (16370 tokens) and Q4 at its own length."""
    out = {"label": "EXPLORATORY"}
    pairs = {L: {} for L in TOKENS}
    for L, n in TOKENS.items():
        d = _find(man, "dense", "fa3", L, "eager")
        if d:
            pairs[L][f"dense_fa3_{L}_attention"] = (_att(d), ref[("dense", n)]["attention_traced_s"])
            pairs[L][f"dense_fa3_{L}_wall"] = (d["untraced"]["median_s"], ref[("dense", n)]["measured_s"])
        x = _find(man, "xattention", None, L, "eager")
        if x:
            pairs[L][f"xattention_{L}_wall"] = (x["untraced"]["median_s"],
                                                ref[("xattention", n)]["measured_s"])
    out["Q3"] = {str(L): HD.q3_comparable(p) for L, p in pairs.items()}
    q3_why = {L: None if out["Q3"][str(L)]["comparable"] else f"Q3 failed at {n} tokens"
              for L, n in TOKENS.items()}
    off = _find(man, "dense", "sdpa_flash", 16384, "eager")
    q1 = (_read(HD.q1_priced_kernel, _att(off), ref[("dense", 16370)]["attention_priced_s"])
          if off else "not read")
    out["Q1"] = _gate(q1, q3_why[16384])
    if isinstance(out["Q1"], dict) and q3_why[16384]:
        out["Q1"]["confirmed"] = None
    g = _find(man, "dense", "fa3", 16384, "graph")
    e = _find(man, "dense", "fa3", 16384, "eager")
    out["Q2"] = (_read(HD.q2_graph, g["graph_components_s"]["attention"], _att(e))
                 if g and e else "not read")
    q4 = {}
    for L, n in TOKENS.items():
        x = _find(man, "xattention", None, L, "eager")
        if x is None or q4_inputs is None or (n not in q4_inputs):
            q4[str(L)] = "not read"
            continue
        ns = x.get("nsys", {})
        src = "nsys" if "median" in ns else "torch.profiler (inflated by its overhead)"
        tr = ns["median"] if "median" in ns else x["torch_profiler"]["host"]
        try:
            hc = HD.q4_host(tr["wall_s"], ref[("xattention", n)]["measured_s"], tr["host_s"])
            res = {"host_source": src, **hc, "host_measured_s": hc["host_s"],
                   **q4_inputs[n](hc["host_s"])}
        except HD.DiagError as e:
            q4[str(L)] = f"not read: {e}"
            continue
        why = [w for w in (q3_why[L], None if hc["comparable"] else
                           f"the traced wall is {hc['wall_ratio']:.3f}x P4's, above "
                           f"{HD.TRACED_WALL_MAX:.2f}x") if w]
        q4[str(L)] = _gate(res, "; ".join(why) or None)
    out["Q4"] = q4
    return out


def q4_functions(inputs: Path, mirror: Path, traces: str) -> dict:
    """For each P4 XAttention cell at batch 1: a function of a measured host
    time (seconds per prefill) giving Q4's fraction, the recursion re-run with
    every device time traced and each priced host cost scaled by measured /
    priced host. Offline, from the scoring mirror and P4's traces."""
    import dataclasses
    import exploratory_wave as EW
    from attnbench.exploratory import breakdown as BR
    from attnbench.exploratory import traces as TR
    from attnbench.predict.recursion import recurse
    D = EW.load(inputs, mirror)
    cp, ms = D["cps"]["P4"], D["ms"]["P4"]
    recs = EW.extract(cp, sorted(glob.glob(traces)))
    out = {}
    for c in cp.cells:
        if c.arm != "xattention" or c.batch != 1:
            continue
        mine = [r for r in recs if r["key"] == c.key]
        if not mine:
            continue
        kinds = [x.value for x in c.priced.kinds]
        al = [TR.align(r["family"], r["device"], kinds) for r in mine]
        fam = al[0][0]
        dev = [None if any(a[1][i] is None for a in al) else
               statistics.fmean(a[1][i] for a in al) for i in range(len(fam))]
        full = BR.substitute(c.priced, dev, fam)
        priced_host = sum(c.priced.host)
        meas = ms["seconds"][c.key]
        pred_all = recurse(full, cp.queue_depth).end_to_end

        def f(host, full=full, priced_host=priced_host, meas=meas, pred_all=pred_all):
            k = host / priced_host
            scaled = dataclasses.replace(full, host=tuple(x * k for x in full.host))
            return {"host_priced_s": priced_host, "host_ratio": k,
                    **HD.q4_fraction(pred_all, recurse(scaled, cp.queue_depth).end_to_end, meas)}
        out[c.context_length] = f
    return out


def cmd_verdict(args) -> int:
    man = {}
    for p in sorted(Path(args.manifests).rglob("cell_*_manifest.json")):
        m = json.loads(p.read_text())
        man[m.get("cell", {}).get("id", p.name)] = m
    ref = p4_reference(args.reference)
    q4 = q4_functions(args.inputs, args.mirror, args.traces) if args.inputs else None
    out = verdict(man, ref, q4)
    text = json.dumps(out, indent=1, default=str)
    if args.inputs:
        import exploratory_wave as EW
        EW.refuse_store(text, json.loads(Path(args.inputs).read_text())["bucket"])
    Path(args.out).write_text(text + "\n")
    print(f"wrote {args.out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "one"):
        p = sub.add_parser(name)
        p.add_argument("--cells", required=True, type=Path)
        p.add_argument("--cpu-boundary", action="store_true")
        p.add_argument("--warmup", type=int, default=5)
        p.add_argument("--prefills", type=int, default=5)
    run = sub.choices["run"]
    run.add_argument("--out", required=True, type=Path)
    run.add_argument("--reps", type=int, default=30)
    run.add_argument("--budget-min", type=float, default=55.0)
    run.add_argument("--nsys-timeout-s", type=float, default=900.0)
    sub.choices["one"].add_argument("--cell", required=True)
    v = sub.add_parser("verdict")
    v.add_argument("--manifests", required=True, type=Path)
    v.add_argument("--reference", type=Path,
                   default=Path(__file__).resolve().parents[1] / "docs" / "exploratory"
                   / "results_20261116.json")
    v.add_argument("--inputs", type=Path)
    v.add_argument("--mirror", type=Path)
    v.add_argument("--traces")
    v.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "verdict":
        return cmd_verdict(a)
    if a.cpu_boundary:
        from attnbench import cpu_boundary
        with cpu_boundary.installed(xattn=False):
            return cmd_run(a) if a.cmd == "run" else cmd_one(a)
    return cmd_run(a) if a.cmd == "run" else cmd_one(a)


if __name__ == "__main__":
    sys.exit(main())
