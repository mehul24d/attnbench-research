"""The paper's PDF figures (C10), from `paper_tables`' rows.

Each panel is one series, a (model, code path, arm, density, batch), so a
panel never holds two code paths or two result classes, and its title names
the class (rule 2, rule 3). Lines inside a panel are predictors, coloured
in a fixed order that follows the predictor, never its rank, with a marker
as a second encoding. Fonts are embedded as TrueType (`pdf.fonttype` 42)
and the PDF carries no creation date, so the same inputs give the same
bytes.

  - `error_vs_length`: signed relative error against context length, the
    noise floor shaded.
  - `decision_error`: per predictor, the count of each decision outcome,
    one panel per (model, code path, class); cells inside the resolution
    floor are a separate, hatched bar, not an outcome.
  - `crossover`: sparse minus dense against length, predicted and measured,
    each with its bootstrap interval, the resolution floor shaded and the
    brackets marked.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..predict import scoring as SC  # noqa: E402
from . import paper_tables as PT  # noqa: E402
from .paper_inputs import Inputs, class_label  # noqa: E402

RC = {"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8, "axes.spines.top": False,
      "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
      "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984", "lines.linewidth": 1.5}
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
MARKERS = ("o", "s", "^", "D", "v", "P")
FLOOR = "#d9d8d3"
INK = "#3b3a37"
OUTCOMES = ((SC.CORRECT, "correct", SERIES[0], ""),
            (SC.FALSE_PROFITABLE, "false profitable", SERIES[1], ""),
            (SC.MISSED_PROFITABLE, "missed profitable", SERIES[3], ""),
            ("inside", "inside floor (not scored)", FLOOR, "///"))


def _style(i: int) -> dict:
    if i >= len(SERIES):
        raise ValueError("more predictors than fixed colours; draw them in another figure")
    return {"color": SERIES[i], "marker": MARKERS[i], "markersize": 4}


def _title(series: tuple, cls: str, builder=None) -> str:
    model, cpath, arm, density, batch = series
    b = "" if builder is None else f", {PT.builder_label(builder)} builder"
    return (f"{class_label(cls)}{b}\n{model}, {PT.code_path_short(cpath)}\n"
            f"{arm}, density {density:g}, batch {batch}")


def _grid(n: int):
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False)
    flat = [a for r in axes for a in r]
    for a in flat[n:]:
        a.set_visible(False)
    return fig, flat[:n]


def _legend(fig, ax) -> None:
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=min(len(lab), 5), fontsize=6,
               frameon=False, bbox_to_anchor=(0.5, 0.0))


def _save(fig, path: Union[str, Path]) -> Path:
    path = Path(path)
    if path.exists():
        plt.close(fig)
        raise FileExistsError(f"{path.name} exists; artifacts are never overwritten")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, format="pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    return path


def error_vs_length(inp: Inputs, path) -> Path:
    _, rows = PT.cell_rows(inp)
    series: dict = {}
    for r in rows:
        s = (r["model"], r["code_path"], r["arm"], r["density"], r["batch"])
        series.setdefault(s, []).append(r)
    with plt.rc_context(RC):
        fig, axes = _grid(len(series))
        for ax, (s, rs) in zip(axes, sorted(series.items())):
            rs = sorted(rs, key=lambda r: r["context_length"])
            L = [r["context_length"] for r in rs]
            ax.axhspan(-100 * inp.relative_floor, 100 * inp.relative_floor, color=FLOOR,
                       lw=0, label="noise floor")
            for i, p in enumerate(inp.predictors):
                ax.plot(L, [100 * r[f"{p}_relative"] for r in rs], label=p, **_style(i))
            ax.axhline(0, color=INK, lw=0.6)
            ax.set_xscale("log", base=2)
            ax.set_xticks(L, [str(x) for x in L])
            ax.set_xlabel("context length (tokens)")
            ax.set_ylabel("relative error (%)")
            ax.set_title(_title(s, inp.classes[s[2]], inp.builder(s[2])), fontsize=7)
        _legend(fig, axes[0])
        return _save(fig, path)


def decision_error(inp: Inputs, groups: Sequence[PT.DecisionGroup], path) -> Path:
    panels: dict = {}
    for g in groups:
        panels.setdefault((g.model, g.code_path, g.result_class, g.builder_kind or ""),
                          []).append(g)
    with plt.rc_context(RC):
        fig, axes = _grid(len(panels))
        for ax, ((model, cpath, cls, bk), gs) in zip(axes, sorted(panels.items())):
            x = range(len(gs))
            w = 0.8 / len(OUTCOMES)
            for j, (key, label, colour, hatch) in enumerate(OUTCOMES):
                ys = [len(g.scored.inside_floor) if key == "inside"
                      else g.scored.decision[key] for g in gs]
                ax.bar([i + (j - 1.5) * w for i in x], ys, width=w * 0.9, color=colour,
                       hatch=hatch, edgecolor=INK if hatch else colour, linewidth=0.4,
                       label=label)
            ax.set_xticks(list(x), [g.predictor for g in gs], rotation=30, ha="right")
            ax.set_ylabel("cells")
            ax.yaxis.get_major_locator().set_params(integer=True)
            b = f", {PT.builder_label(bk)} builder" if bk else ""
            ax.set_title(f"{class_label(cls)}{b}\n{model}, {PT.code_path_short(cpath)}",
                         fontsize=7)
        _legend(fig, axes[0])
        return _save(fig, path)


def crossover(inp: Inputs, series: Sequence[PT.CrossoverSeries], path) -> Path:
    if not series:
        raise ValueError("no crossover series to draw")
    order = {p: i for i, p in enumerate(inp.predictors)}
    with plt.rc_context(RC):
        fig, axes = _grid(len(series))
        for ax, s in zip(axes, series):
            L = list(s.lengths)
            f = inp.resolution_floor_s
            ax.axhspan(-f, f, color=FLOOR, lw=0, label="resolution floor")
            ax.axhline(0, color=INK, lw=0.6)
            for diffs, label, style in (
                    (s.measured_diffs, "measured", {"color": INK, "marker": "o",
                                                    "markersize": 4}),
                    (s.predicted_diffs, f"predicted ({s.predictor})",
                     _style(order[s.predictor]))):
                y = [d.point for d in diffs]
                err = [[d.point - d.lo for d in diffs], [d.hi - d.point for d in diffs]]
                ax.errorbar(L, y, yerr=err, capsize=2, label=label, **style)
            for b in s.measured.brackets:
                ax.axvspan(b.lower, b.upper, color=INK, alpha=0.08, lw=0)
            for b in s.predicted.brackets:
                ax.axvline(b.lower, color=SERIES[order[s.predictor]], lw=0.8, ls="--")
                ax.axvline(b.upper, color=SERIES[order[s.predictor]], lw=0.8, ls="--")
            ax.set_xscale("log", base=2)
            ax.set_xticks(L, [str(x) for x in L])
            ax.set_xlabel("context length (tokens)")
            ax.set_ylabel("sparse minus dense (s)")
            ax.set_title(_title(s.series, s.result_class, s.builder_kind)
                         + f"\nerror: {s.error.status}", fontsize=7)
        _legend(fig, axes[0])
        return _save(fig, path)
