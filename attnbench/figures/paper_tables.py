"""The paper's results, as rows, and their LaTeX booktabs tables (C10).

Each builder returns plain rows from checked `paper_inputs.Inputs`; the
numbers come from `predict` (residuals, ablations, sensitivity, scoring,
crossover) and are not recomputed here. Every row carries the model, the
code path and the result class, and every summary groups by all three, so
no number pools two code paths (rule 3) or two classes (rule 2).

  - `cell_rows`: C5's `t3_tables.cell_table`, one row per measured cell,
    with its class.
  - `predictor_rows`: per predictor, model, code path and class. (C5's
    `predictor_summary` groups by predictor alone, which pools code paths;
    the paper does not use it.)
  - `sync_rows`: residuals by synchronisation count (T7.1), the same keys.
  - `ablation_rows`: Section 5.4's ablations, recomputed from the committed
    components (`predict.ablations`). They are not committed predictions
    and are labelled so; the two lower bounds are bounds, not predictions,
    and a bound above the measured time is flagged.
  - `queue_rows`: the queue sensitivity of the committed recursion
    (`predict.sensitivity`), every committed cell, with the multiples at
    which the predicted sparse-versus-dense sign flips.
  - `decision`: decision error and sign agreement per predictor
    (`predict.scoring.score_cells`), cells inside the resolution floor
    counted apart.
  - `crossover`: the bracket rule on predicted and measured differences,
    and the crossover error (`predict.crossover`), when the spec supplies
    the intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from ..predict import ablations as A
from ..predict import crossover as X
from ..predict import residuals as R
from ..predict import scoring as SC
from ..predict import sensitivity as S
from ..predict.training import partner_of
from . import t3_tables as T
from .paper_inputs import DENSE, Inputs, PaperInputError, class_label, measured_difference

ABLATIONS = ("no_sync_cost", "no_sync_wait")
BOUNDS = ("host_only_bound", "device_only_bound")
ABLATION_LABEL = "recomputed from the committed components; not a committed prediction"


def _residuals(inp: Inputs):
    return R.residuals(inp.cp, inp.measured, statistic=inp.cp.statistic,
                       dense_arms=[inp.dense_arm], relative_floor=inp.relative_floor,
                       predictors=list(inp.predictors))


def cell_rows(inp: Inputs) -> tuple[tuple[str, ...], list[dict]]:
    cols, rows = T.cell_table(inp.cp, inp.measured, dense_arms=[inp.dense_arm],
                              relative_floor=inp.relative_floor, predictors=inp.predictors)
    for r in rows:
        r["result_class"] = inp.classes[r["arm"]]
        r["builder_kind"] = inp.builder(r["arm"])
    return cols + ("result_class", "builder_kind"), rows


def _grouped(inp: Inputs, by: Sequence[str]) -> list[dict]:
    rs = _residuals(inp).residuals
    if not rs:
        raise PaperInputError("no committed cell was measured")
    order = {p: i for i, p in enumerate(inp.predictors)}
    out = []
    kind = lambda r: (inp.classes[r.arm], inp.builder(r.arm) or "")
    for cls, bk in sorted({kind(r) for r in rs}):
        sub = [r for r in rs if kind(r) == (cls, bk)]
        for g in R.summarise(sub, by):
            row = dict(zip(by, g.group))
            row["result_class"] = cls
            row["builder_kind"] = bk or None
            row.update({f: getattr(g, f) for f in T.SUMMARY_FIELDS})
            out.append(row)
    out.sort(key=lambda r: (order[r["predictor"]], r["model"], r["code_path"],
                            r["result_class"], r["builder_kind"] or "")
             + tuple(r[f] for f in by[3:]))
    return out


def predictor_rows(inp: Inputs) -> list[dict]:
    return _grouped(inp, ("predictor", "model", "code_path"))


def sync_rows(inp: Inputs) -> list[dict]:
    return _grouped(inp, ("predictor", "model", "code_path", "n_synchronisations"))


def ablation_rows(inp: Inputs) -> list[dict]:
    out = []
    Q = inp.cp.queue_depth
    for c in sorted(inp.cp.cells, key=lambda c: c.key):
        if c.key not in inp.measured:
            continue
        m = inp.measured[c.key]
        vals = {"recursion": c.predictions["recursion"],
                "no_sync_cost": A.no_sync_cost(c.priced, Q).end_to_end,
                "no_sync_wait": A.no_sync_wait(c.priced, Q).end_to_end,
                "host_only_bound": A.host_only_bound(c.priced),
                "device_only_bound": A.device_only_bound(c.priced)}
        row = {"key": c.key, "result_class": inp.classes[c.arm],
               "builder_kind": inp.builder(c.arm), "measured_s": m,
               "n_synchronisations": R.synchronisations_in_step(c)}
        for n, v in vals.items():
            row[f"{n}_s"] = v
            row[f"{n}_relative"] = (v - m) / m
        row["bound_above_measured"] = [b for b in BOUNDS if vals[b] > m]
        out.append(row)
    if not out:
        raise PaperInputError("no committed cell was measured")
    return out


def queue_rows(inp: Inputs) -> list[dict]:
    qs = S.queue_sensitivity(inp.cp, "recursion")
    flips = {d.sparse_key: d for d in S.decision_sensitivity(qs, dense_arm=inp.dense_arm)}
    out = []
    for r in sorted(qs, key=lambda r: r.key):
        f = flips.get(r.key)
        out.append({"key": r.key, "result_class": inp.classes[r.key.arm],
                    "builder_kind": inp.builder(r.key.arm),
                    "measured_q": r.measured_q, "committed_s": r.committed,
                    "points": [(p.multiple, p.queue_depth, p.relative_change)
                               for p in r.points],
                    "flips_at": None if f is None else list(f.flips_at),
                    "committed_sparse_faster": None if f is None
                    else f.committed_sparse_faster})
    return out


def _dense_measured(inp: Inputs) -> dict:
    return S.dense_partners([k for k in inp.measured if k.arm == inp.dense_arm],
                            inp.dense_arm)


@dataclass(frozen=True)
class DecisionGroup:
    predictor: str
    model: str
    code_path: str
    result_class: str
    builder_kind: Optional[str]
    scored: SC.ScoredCells


def decision(inp: Inputs) -> list[DecisionGroup]:
    dense = _dense_measured(inp)
    pairs = []
    for k in sorted(inp.measured):
        if k.arm == inp.dense_arm or partner_of(k) not in dense:
            continue
        if k not in inp.admissible:
            raise PaperInputError(f"no admissibility stated for {k}; decision error needs it "
                                  f"(no default)")
        pairs.append((k, dense[partner_of(k)]))
    out = []
    for pred in inp.predictors:
        groups: dict = {}
        for k, d in pairs:
            groups.setdefault((k.model, k.code_path, inp.classes[k.arm],
                               inp.builder(k.arm) or ""), []).append((k, d))
        for (model, cpath, cls, bk), ps in sorted(groups.items()):
            cells = []
            for k, d in ps:
                p_s = inp.cp.cell(*k).predictions[pred]
                p_d = inp.cp.cell(*d).predictions[pred]
                m_s, m_d = inp.measured[k], inp.measured[d]
                cells.append((k, p_s, p_d, m_s, m_d, inp.admissible[k]))
            out.append(DecisionGroup(pred, model, cpath, cls, bk or None,
                                     SC.score_cells(cells, floor=inp.resolution_floor_s)))
    return out


@dataclass(frozen=True)
class CrossoverSeries:
    predictor: str
    series: tuple              # (model, code path, arm, density, batch)
    result_class: str
    builder_kind: Optional[str]
    lengths: tuple
    predicted: X.CrossoverResult
    measured: X.CrossoverResult
    predicted_diffs: tuple     # GridDifference per length
    measured_diffs: tuple
    error: X.CrossoverError


def series_of(k) -> tuple:
    return (k.model, k.code_path, k.arm, k.density, k.batch)


def crossover(inp: Inputs) -> Optional[list[CrossoverSeries]]:
    if inp.crossover is None:
        return None
    pred_iv = inp.crossover["predicted"]
    dense = _dense_measured(inp)
    series: dict = {}
    for k in inp.measured:
        if k.arm != inp.dense_arm and partner_of(k) in dense:
            series.setdefault(series_of(k), []).append(k)
    out = []
    for pred in inp.predictors:
        for s, ks in sorted(series.items()):
            ks = sorted(ks, key=lambda k: k.context_length)
            if len(ks) < 2 or not any((pred, k) in pred_iv for k in ks):
                continue
            pd, md = [], []
            for k in ks:
                if (pred, k) not in pred_iv:
                    raise PaperInputError(f"no predicted {pred} interval at {k}; a series "
                                          f"is read whole or not at all")
                dk, iv = pred_iv[(pred, k)]
                if dk != dense[partner_of(k)]:
                    raise PaperInputError(f"the {pred} interval at {k} is against {dk}, "
                                          f"not the measured dense cell")
                pd.append(X.GridDifference(k.context_length, iv.point, iv.lo, iv.hi))
                m = measured_difference(inp, k, dk)
                md.append(X.GridDifference(k.context_length, m.point, m.lo, m.hi))
            p = X.crossover_brackets(pd, floor=inp.resolution_floor_s)
            m = X.crossover_brackets(md, floor=inp.resolution_floor_s)
            out.append(CrossoverSeries(pred, s, inp.classes[s[2]], inp.builder(s[2]),
                                       p.lengths, p, m,
                                       tuple(pd), tuple(md), X.crossover_error(p, m)))
    return out


# --------------------------------------------------------------------------
# LaTeX
# --------------------------------------------------------------------------

_TEX = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}", "|": r"\textbar{}", "<": r"\textless{}",
        ">": r"\textgreater{}"}


def tex_escape(s) -> str:
    return "".join(_TEX.get(ch, ch) for ch in str(s))


def code_path_short(label: str) -> str:
    """gpu, implementation, optimise flag and the label's hash, from a
    `commit.code_path_label`."""
    return " / ".join(label.split("|"))


def builder_label(b: Optional[str]) -> str:
    return {"cpu": "CPU", "gpu": "GPU", None: "--"}[b]


def fmt_s(v: float) -> str:
    return f"{v:.4f}"


def fmt_pct(v: float, signed: bool = True) -> str:
    return f"{100 * v:+.2f}" if signed else f"{100 * v:.2f}"


def booktabs(header: Sequence[str], align: str, rows: Sequence[Sequence[str]], *,
             caption: str, label: str) -> str:
    """A table environment. Cells are already LaTeX (escape with
    `tex_escape`); headers and the caption are escaped here."""
    if len(header) != len(align):
        raise PaperInputError("one alignment per column")
    for r in rows:
        if len(r) != len(header):
            raise PaperInputError(f"a row of {len(r)} cells under {len(header)} columns")
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             rf"\caption{{{tex_escape(caption)}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(tex_escape(h) for h in header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _floor_note(inp: Inputs) -> str:
    p = "provisional " if inp.floor_provisional else ""
    return (f"{p}noise floor {inp.relative_floor:g} (relative), {p}resolution floor "
            f"{inp.resolution_floor_s:g} s")


def _caption(inp: Inputs, what: str) -> str:
    return (f"{what}. {inp.cp.pairing}, {inp.cp.statistic}; {_floor_note(inp)}; "
            f"prediction sha256 {inp.cp.sha256[:12]}.")


def _key_cells(k) -> list[str]:
    return [tex_escape(k.model), tex_escape(code_path_short(k.code_path)), tex_escape(k.arm),
            f"{k.density:g}", str(k.context_length), str(k.batch)]


KEY_HEADER = ["model", "code path", "arm", "density", "length", "batch"]


def tex_cells(inp: Inputs) -> str:
    _, rows = cell_rows(inp)
    head = KEY_HEADER + ["class", "builder", "syncs", "measured (s)"]
    for p in inp.predictors:
        head += [f"{p} (s)", f"{p} err (%)"]
    body = []
    for r in rows:
        cells = [tex_escape(r["model"]), tex_escape(code_path_short(r["code_path"])),
                 tex_escape(r["arm"]), f"{r['density']:g}", str(r["context_length"]),
                 str(r["batch"]), tex_escape(class_label(r["result_class"])),
                 builder_label(r["builder_kind"]),
                 str(r["n_synchronisations"]), fmt_s(r["measured_s"])]
        for p in inp.predictors:
            mark = r"\dag" if r[f"{p}_inside_floor"] else ""
            cells += [fmt_s(r[f"{p}_s"]), fmt_pct(r[f"{p}_relative"]) + mark]
        body.append(cells)
    return booktabs(head, "lllrrrllrr" + "rr" * len(inp.predictors), body,
                    caption=_caption(inp, "Committed prediction against measured time per "
                                          "cell; a dagger marks an error inside the noise "
                                          "floor, not interpreted"),
                    label="tab:cells")


def _summary_tex(inp: Inputs, rows: list[dict], extra: Sequence[str], what: str,
                 label: str) -> str:
    head = ["predictor", "model", "code path", "class", "builder"] + list(extra) + [
        "n", "inside floor", "median |err| (%)", "mean err (%)", "max |err| (%)"]
    body = []
    for r in rows:
        body.append([tex_escape(r["predictor"]), tex_escape(r["model"]),
                     tex_escape(code_path_short(r["code_path"])),
                     tex_escape(class_label(r["result_class"])),
                     builder_label(r["builder_kind"])]
                    + [str(r[e]) for e in extra]
                    + [str(r["n"]), str(r["n_inside_floor"]),
                       fmt_pct(r["median_abs_relative"], False), fmt_pct(r["mean_relative"]),
                       fmt_pct(r["max_abs_relative"], False)])
    return booktabs(head, "lllll" + "r" * len(extra) + "rrrrr", body,
                    caption=_caption(inp, what), label=label)


def tex_predictors(inp: Inputs) -> str:
    return _summary_tex(inp, predictor_rows(inp), [],
                        "Absolute relative error per predictor, model, code path and "
                        "result class; never pooled across code paths", "tab:predictors")


def tex_sync(inp: Inputs) -> str:
    return _summary_tex(inp, sync_rows(inp), ["n_synchronisations"],
                        "Residuals by synchronisation count inside the step (the closing "
                        "synchronise not counted)", "tab:residuals-sync")


def tex_ablations(inp: Inputs) -> str:
    rows = ablation_rows(inp)
    head = KEY_HEADER + ["class", "syncs", "measured (s)", "recursion err (%)",
                         "no sync cost err (%)", "no sync wait err (%)",
                         "host bound (s)", "device bound (s)", "bound above measured"]
    body = []
    for r in rows:
        body.append(_key_cells(r["key"]) + [
            tex_escape(class_label(r["result_class"])), str(r["n_synchronisations"]),
            fmt_s(r["measured_s"]), fmt_pct(r["recursion_relative"]),
            fmt_pct(r["no_sync_cost_relative"]), fmt_pct(r["no_sync_wait_relative"]),
            fmt_s(r["host_only_bound_s"]), fmt_s(r["device_only_bound_s"]),
            tex_escape(", ".join(r["bound_above_measured"]) or "none")])
    return booktabs(head, "lllrrrlrrrrrrrl", body,
                    caption=_caption(inp, f"Model ablations ({ABLATION_LABEL}) and the "
                                          f"host-only and device-only lower bounds"),
                    label="tab:ablations")


def _mult(m) -> str:
    return "no queue" if m is None else f"{m:g}x"


def tex_queue(inp: Inputs) -> str:
    rows = queue_rows(inp)
    mults = [m for m, _, _ in rows[0]["points"]]
    head = KEY_HEADER + ["class", "Q", "recursion (s)"] + [
        f"{_mult(m)} (%)" for m in mults] + ["sign flips at"]
    body = []
    for r in rows:
        if r["flips_at"] is None:
            flips = "n/a"
        else:
            flips = ", ".join(_mult(m) for m in r["flips_at"]) or "none"
        body.append(_key_cells(r["key"]) + [
            tex_escape(class_label(r["result_class"])), str(r["measured_q"]),
            fmt_s(r["committed_s"])] + [fmt_pct(rc) for _, _, rc in r["points"]]
            + [tex_escape(flips)])
    depths = ", ".join(f"{_mult(m)}: Q={q}" for m, q, _ in rows[0]["points"] if m is not None)
    return booktabs(head, "lllrrrlrr" + "r" * len(mults) + "l", body,
                    caption=_caption(inp, f"Queue sensitivity of the committed recursion, "
                                          f"relative change per multiple of the measured Q "
                                          f"({depths})"),
                    label="tab:queue")


def tex_decision(inp: Inputs, groups: list[DecisionGroup]) -> str:
    head = ["predictor", "model", "code path", "class", "builder", "scored", "correct",
            "false profitable", "missed profitable", "sign agrees", "sign disagrees",
            "inside floor"]
    body = []
    for g in groups:
        d = g.scored.decision
        body.append([tex_escape(g.predictor), tex_escape(g.model),
                     tex_escape(code_path_short(g.code_path)),
                     tex_escape(class_label(g.result_class)), builder_label(g.builder_kind),
                     str(g.scored.n_scored),
                     str(d[SC.CORRECT]), str(d[SC.FALSE_PROFITABLE]),
                     str(d[SC.MISSED_PROFITABLE]), str(g.scored.sign_agree),
                     str(g.scored.sign_disagree), str(len(g.scored.inside_floor))])
    return booktabs(head, "lllll" + "r" * 7, body,
                    caption=_caption(inp, "Decision error; a predicted tie is dense, and "
                                          "cells inside the resolution floor are not scored"),
                    label="tab:decision")


def _brackets(r: X.CrossoverResult) -> str:
    if not r.brackets:
        return r.status
    return "; ".join(f"[{b.lower}, {b.upper}] {b.direction}" for b in r.brackets)


def tex_crossover(inp: Inputs, series: list[CrossoverSeries]) -> str:
    head = ["predictor", "model", "code path", "arm", "density", "batch", "class",
            "predicted", "measured", "error", "inside", "overlaps", "direction",
            "shift (steps)"]
    body = []
    for s in series:
        model, cpath, arm, density, batch = s.series
        cmp = s.error.comparisons
        yes = lambda f: ", ".join("yes" if getattr(c, f) else "no" for c in cmp) or "n/a"
        body.append([tex_escape(s.predictor), tex_escape(model),
                       tex_escape(code_path_short(cpath)), tex_escape(arm), f"{density:g}",
                       str(batch), tex_escape(class_label(s.result_class)),
                       tex_escape(_brackets(s.predicted)), tex_escape(_brackets(s.measured)),
                       tex_escape(s.error.status), yes("contains"), yes("overlaps"),
                       yes("direction_agrees"),
                       ", ".join(f"{c.shift_grid_steps:+g}" for c in cmp) or "n/a"])
    return booktabs(head, "llllrrllllllll", body,
                    caption=_caption(inp, f"Crossover brackets, widened by the bootstrap "
                                          f"interval of the difference "
                                          f"(level {inp.crossover['level']:g})"),
                    label="tab:crossover")
