"""EXPLORATORY: the 16 Nov 2026 exploratory wave, items 2 to 5 (the adviser's
rulings dated 16 Nov 2026). Nothing here scores a hypothesis; the scored H0
to H3 verdicts and the analysis commit 83b2edf are untouched.

CPU only and offline: the committed predictions and the ground truth are read
from the scoring wave's mirror through `score_hypotheses`' own loaders (no
store is contacted), and the traces from local copies.

  2. per-component error: `attnbench.exploratory.breakdown` over the traces
  3. safe dispatch: `attnbench.exploratory.safe_dispatch`, with the abstain
     margin recomputed from P1 alone and refused unless it is the one
     recorded in Appendix A before P2 or P4 was dispatched
  4. the P1-fitted lower bound: `attnbench.exploratory.bound`
  5. the paired sign bootstrap: `attnbench.exploratory.sign`

    python3 scripts/exploratory_wave.py --inputs <inputs.json> --mirror <dir> \\
        --traces P2=<glob> --traces P4=<glob> --out <file.json>

The output names no store and no bucket.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import statistics
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import score_hypotheses as SH  # noqa: E402
from attnbench.components import op_sequence as ops  # noqa: E402
from attnbench.exploratory import bound as BD  # noqa: E402
from attnbench.exploratory import breakdown as BR  # noqa: E402
from attnbench.exploratory import safe_dispatch as SD  # noqa: E402
from attnbench.exploratory import sign as SG  # noqa: E402
from attnbench.exploratory import traces as TR  # noqa: E402
from attnbench.predict import intensity as IN  # noqa: E402

# Appendix A, 17 Nov 2026: fixed from P1 alone before any P2 or P4 dispatch.
RECORDED_MARGIN = 0.02264
PAIRINGS = ("P2", "P4")


def load(inputs: Path, mirror: Path) -> dict:
    spec = json.loads(Path(inputs).read_text())
    tagged = SH.lock_at(SH.TAG)
    locked = SH.locked_from(tagged)
    store = SH.MirrorStore(Path(mirror))
    prefix = spec["bucket"].rstrip("/") + "/"
    banked = {p: b for p in ("P1",) + PAIRINGS
              if (b := SH.Ledger(store, prefix, p).banked()) is not None}
    rows, files = SH.training_inputs(spec, tagged, banked)
    mcr = int(tagged["values"]["rerun_budget"]["value"]["min_clean_rounds"])
    cps, ms, blocks_of = {}, {}, {}
    for p in ("P1",) + PAIRINGS:
        ps = spec[p]
        blocks, starts = [], []
        if p != "P1":
            blocks, starts, _ = SH.passed_sessions(ps["sessions"], ps["ground_truth_dir"])
        cps[p], _ = SH.load(store, prefix, p, ps, starts, banked=banked, rows=rows, files=files)
        if p != "P1":
            ms[p] = SH.measure(cps[p], blocks, min_clean_rounds=mcr)
            blocks_of[p] = blocks
    p1 = {r.cell.key: r.measured for r in rows if r.cell.pairing == "P1"}
    return dict(locked=locked, cps=cps, ms=ms, blocks=blocks_of, p1_measured=p1)


def code_path(cp, c) -> ops.CodePath:
    i = cp.code_paths[c.code_path]
    return ops.CodePath(arm=c.arm, gpu=i["gpu"], implementation=i["implementation"],
                        libraries=tuple(tuple(x) for x in i["libraries"]),
                        python_optimize=i["python_optimize"])


def extract(cp, files) -> list[dict]:
    """Per trace: its cell, the traced families and per-position device
    seconds (None where Kineto dropped the record), plus the closing sync."""
    recs = []
    for f in files:
        c = TR.cell_for(cp.cells, f)
        tj, missing = TR.with_region(ops.load_trace(f))
        seq = ops.parse_trace(tj, code_path(cp, c), context_length=c.context_length,
                              batch=c.batch)
        dd = ops.device_durations(tj, seq)
        recs.append({"trace": Path(f).name, "key": c.key, "placeholders": len(missing),
                     "family": [ops.operation_family(o) for o in seq.ops] + ["sync:closing"],
                     "device": [None if o.correlation in missing else dd.get(i)
                                for i, o in enumerate(seq.ops)] + [None]})
    return recs


def item2(cp, ms, recs) -> dict:
    by = {c.key: c for c in cp.cells}
    groups = collections.defaultdict(list)
    for r in recs:
        groups[r["key"]].append(r)
    splits, cells = [], []
    for k in sorted(groups):
        c = by[k]
        kinds = [x.value for x in c.priced.kinds]
        al = [TR.align(r["family"], r["device"], kinds) for r in groups[k]]
        fam = al[0][0]
        if any(a[0] != fam for a in al):
            raise TR.TraceError(f"{tuple(k)}: its traces differ in sequence")
        dev = [None if any(a[1][i] is None for a in al)
               else statistics.fmean(a[1][i] for a in al) for i in range(len(fam))]
        s = BR.split(c.priced, cp.queue_depth, ms["seconds"][k], dev, fam)
        splits.append(s)
        cells.append({"arm": c.arm, "density": c.density, "context_length": c.context_length,
                      "batch": c.batch, "traces": len(groups[k]), "unpriced_folded": al[0][2],
                      "measured_s": s.measured, "rel_error": s.rel(s.predicted),
                      "rel_error_component_replaced": {q: s.rel(v) for q, v in s.replaced.items()},
                      "rel_error_all_device_replaced": s.rel(s.all_replaced),
                      "device_s_priced_traced": {q: list(v) for q, v in s.device.items()},
                      "predicted_host_bound": sum(c.priced.host) > sum(c.priced.device)
                      + sum(c.priced.link)})
    missing = sorted(set(by) - set(groups))
    return {"summary": BR.summary(splits), "cells": cells,
            "cells_without_trace": [list(k) for k in missing]}


def margin(D) -> dict:
    cells = D["cps"]["P1"].cells
    adm = SD.admissible_of([c.key for c in cells])
    meas = SD.measured_by_point(D["p1_measured"])
    pts = SD.points_for(cells, "recursion")
    floors = D["locked"].floors
    m0, bad = SD.margin_from(pts, meas, adm, floors)
    m = m0 if m0 > 0 else SD.fallback_margin(pts, meas, adm)
    if round(m, 5) != RECORDED_MARGIN:
        raise SystemExit(f"the P1 margin is {m:.6f}, not the {RECORDED_MARGIN} recorded "
                         f"in Appendix A; refusing to dispatch P2 or P4")
    return {"zero_false_profitable_margin": m0, "false_profitable_on_P1": len(bad),
            "margin": m, "rule": "fallback (largest P1 overstatement)" if m0 == 0 else "primary"}


def item3(D, m: float) -> dict:
    out = {}
    floors = D["locked"].floors
    for p in PAIRINGS:
        cells = D["cps"][p].cells
        adm = SD.admissible_of([c.key for c in cells])
        meas = SD.measured_by_point(D["ms"][p]["seconds"])
        out[p] = {}
        for name, (pts, ch) in SD.policies(cells, adm, floors, m).items():
            s = SD.score(pts, ch, meas, adm, floors)
            out[p][name] = {"n": s.n, "mean_regret": s.mean_regret, "worst_regret": s.worst_regret,
                            "mean_vs_dense": s.mean_vs_dense, "worst_vs_dense": s.worst_vs_dense,
                            "wins": s.wins, "wins_kept": s.wins_kept,
                            "choices": {str(k.context_length): [v.arm, v.density]
                                        for k, v in s.choices.items()}}
    return out


def _obs(cp, c, t, shape, n_layers) -> BD.Obs:
    f, b = BD.work(shape, n_layers, c.context_length, c.batch, float(c.density))
    return BD.Obs(cp.code_paths[c.code_path]["gpu"], f, b, len(c.priced), t)


def _name(k) -> str:
    return f"{k.arm}|{k.density:.6g}|{k.context_length}|{k.batch}"


def item4(D) -> dict:
    shape, nl = IN.primary_shape(), BD.layers()
    cp1 = D["cps"]["P1"]
    train = [_obs(cp1, c, D["p1_measured"][c.key], shape, nl) for c in cp1.cells if c.arm != "dense"]
    b = BD.fit(train)
    out = {"fit_on_P1": vars(b), "n_train": len(train)}
    for p in PAIRINGS:
        cp, sec = D["cps"][p], D["ms"][p]["seconds"]
        obs = {_name(c.key): _obs(cp, c, sec[c.key], shape, nl) for c in cp.cells if c.arm != "dense"}
        out[p] = {"cells": BD.check(b, obs),
                  "own_envelope": vars(BD.fit(list(obs.values())))}   # descriptive only
    return out


def item5(D) -> dict:
    floors, bs = D["locked"].floors, D["locked"].bootstrap
    out = {}
    for p in PAIRINGS:
        cp, per = D["cps"][p], D["ms"][p]["per_block"]
        cells, skipped = {}, []
        for bid, results in D["blocks"][p]:
            for name, result in results:
                keys = [SH.G.place_row(cp, r, name).key for r in result["rows"]]
                dense = [k for k in keys if k.arm == "dense"]
                sparse = [k for k in keys if k.arm != "dense"]
                if not sparse:
                    continue
                why = SG.pairable(result) or (None if len(dense) == 1 else f"{len(dense)} dense rows")
                if why:
                    skipped.append({"block": bid, "file": name, "why": why})
                    continue
                for k in sparse:
                    s = SG.paired(per[SH.BlockCell(bid, dense[0])], per[SH.BlockCell(bid, k)],
                                  floor_s=floors.at(k.context_length, k.batch),
                                  n_boot=bs.n_boot, seed=bs.seed, level=bs.level)
                    cells[_name(k)] = {"block": bid, **vars(s)}
        out[p] = {"cells": cells, "skipped": skipped}
    return out


def refuse_store(text: str, bucket: str) -> None:
    """Refuse output that names a store or the inputs' bucket."""
    name = bucket.split("://")[-1].strip("/")
    if "gs://" in text or (name and name in text):
        raise SystemExit("the output names a store; refusing to write it")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inputs", required=True, type=Path)
    ap.add_argument("--mirror", required=True, type=Path)
    ap.add_argument("--traces", action="append", default=[], help="PAIRING=<glob>")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    globs = dict(t.split("=", 1) for t in a.traces)
    if set(globs) != set(PAIRINGS):
        ap.error(f"--traces for exactly {PAIRINGS}")
    D = load(a.inputs, a.mirror)
    files = {p: sorted(glob.glob(globs[p])) for p in PAIRINGS}
    out = {"label": "EXPLORATORY", "rulings": "adviser's rulings dated 16 Nov 2026, exploratory wave",
           "traces": {p: len(f) for p, f in files.items()}}
    out["item2_breakdown"] = {p: item2(D["cps"][p], D["ms"][p], extract(D["cps"][p], files[p]))
                              for p in PAIRINGS}
    mg = margin(D)
    out["item3_safe_dispatch"] = {"margin": mg, **item3(D, mg["margin"])}
    out["item4_bound"] = item4(D)
    out["item5_sign"] = item5(D)
    text = json.dumps(out, indent=1, default=lambda o: list(o) if isinstance(o, tuple) else str(o))
    refuse_store(text, json.loads(a.inputs.read_text())["bucket"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text + "\n")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
