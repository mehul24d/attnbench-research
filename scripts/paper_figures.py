"""The paper's six tables and figures, generated from banked data (the
researcher's paper skeleton wave of 29 Nov 2026, item 4). CPU only and
offline. Run by scripts/paper_reproduce.py on its own regenerated outputs, or
directly on the committed copies:

    python3 scripts/paper_figures.py \\
        --scored docs/scoring/hypotheses_scored_20261116.json \\
        --tables docs/paper/exploratory/paper_tables.json --out paper

It writes, under --out:

  generated/numbers.tex           the macros the paper's text may use
  generated/tab_pairings.tex      T1  pairing and lock status        PRE-REGISTERED verdicts
  generated/fig_pred_vs_meas.tex  F1  predicted against measured     PRE-REGISTERED
  generated/fig_error_by_cell.tex F2  signed relative error (H1's) and signed error (ms)   PRE-REGISTERED (intervals EXPLORATORY)
  generated/tab_miss.tex          T2  the miss decomposition         EXPLORATORY / POST HOC
  generated/tab_baselines.tex     T3  baselines and knockouts        PRE-REGISTERED rivals, EXPLORATORY knockouts
  generated/tab_threats.tex       T4  threats to validity            a list, not a result
  figures/fig_*.pdf, figures/fig_*.csv  (each figure's plotted values, as a table view)

Nothing is fitted, rescored or changed. The per-cell values come from the
paper-quality wave's tables; before anything is written the script checks that
they reproduce the scored values: the median absolute relative error over the
35 cells must equal H1's scored median on each pairing, and each rival's median
must equal its scored median. A mismatch refuses, and nothing is written.

Every caption starts with its type. No path, repository, branch or tag name is
typeset: names a reviewer could trace sit in numbers.tex macros (\\LockTag and
the like) so they can be blanked in one place for the anonymised submission.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRINGS = ("P2", "P4")
GPU = {"P2": "A100", "P4": "H100"}

# The plan's arms, in a fixed order. The two XAttention arms share one hue
# (the dataviz rule caps a scatter's hues at four; shape tells them apart).
ARMS = ("dense", "meanpool", "oracle", "xattention", "xattention_noassert")
ARM_LABEL = {"dense": "dense", "meanpool": "mean-pool", "oracle": "oracle",
             "xattention": "XAttention", "xattention_noassert": "XAttention (no assert)"}
ARM_COLOR = {"dense": "#2a78d6", "meanpool": "#eb6834", "oracle": "#1baf7a",
             "xattention": "#6250d6", "xattention_noassert": "#6250d6"}
ARM_MARKER = {"dense": "o", "meanpool": "s", "oracle": "^", "xattention": "D",
              "xattention_noassert": "d"}
# Narrow tables only; the caption names the arms in full.
SHORT_ARM = {"xattention": "XAttn.", "xattention_noassert": "XAttn. (no assert)"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#d9d8d3"

# Pairing status beyond the scored ones. The lock's scored_pairings name only
# P2 and P4; these rows restate docs/paper/claim_status.md (section C, the P3
# row) and docs/paper/threats_to_validity.md (item 14). Not data the lock holds.
STATUS_OUTSIDE_LOCK = {
    "P1": "Development pairing; never counted as a test",
    "P3": "Run after the lock; exploratory only",
    "P5": "Not run",
    "P6": "Not run",
}

RIVALS = ("sum", "fitted_sum", "step_form", "learned")
KNOCKOUTS = ("no_queue", "no_fixed_sync", "no_wait", "no_sync", "host_only",
             "device_only", "fitted_sum_plus_s0", "sum_plus_s0")
PREDICTOR_LABEL = {
    "recursion": "recursion (committed)", "sum": "sum", "fitted_sum": "fitted sum",
    "step_form": "step form", "learned": "learned",
    "no_queue": "without queue", "no_fixed_sync": "without fixed sync cost",
    "no_wait": "without sync waits", "no_sync": "without any sync term",
    "host_only": "host time only", "device_only": "device chain only",
    "fitted_sum_plus_s0": "fitted sum plus $s_0$", "sum_plus_s0": "sum plus $s_0$",
}


class Refused(SystemExit):
    pass


# ---------------------------------------------------------------- helpers

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cell_parts(key: str):
    arm, density, length, batch = key.split("|")
    return arm, float(density), int(length), int(batch.lstrip("b"))


def tex_escape(s: str) -> str:
    """Plain text (with `code` spans) to LaTeX that XeTeX and pdfTeX both take."""
    out = []
    for i, part in enumerate(re.split(r"`([^`]*)`", s)):
        if i % 2:
            out.append(r"\texttt{" + _esc(part) + "}")
        else:
            out.append(_esc(part))
    return "".join(out)


def _esc(s: str) -> str:
    s = (s.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
          .replace("$", r"\$").replace("#", r"\#").replace("_", r"\_")
          .replace("{", r"\{").replace("}", r"\}").replace("~", r"\textasciitilde{}"))
    s = re.sub(r'"([^"]*)"', r"``\1''", s)
    for a, b in (("µ", r"$\mu$"), ("±", r"$\pm$"), ("÷", r"$\div$"),
                 ("≤", r"$\le$"), ("≥", r"$\ge$"), ("×", r"$\times$"),
                 ("–", "--"), ("−", "$-$"), ("“", "``"), ("”", "''"),
                 ("’", "'"), ("‘", "`"), ("φ", r"$\varphi$")):
        s = s.replace(a, b)
    if "—" in s:
        raise Refused("an em dash reached a table; the paper takes none")
    if any(ord(c) > 127 for c in s):
        raise Refused(f"a character this script does not map reached a table: {s!r}")
    return s


def fmt(x, nd=3, signed=False):
    if x is None:
        return "--"
    s = f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def header(kind: str, inputs: dict) -> str:
    lines = [f"% GENERATED by scripts/paper_figures.py. Do not edit; rerun the script.",
             f"% {kind}", "% Inputs (sha256):"]
    lines += [f"%   {k}: {v}" for k, v in inputs.items()]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- checks

def check_against_scores(scores: dict, tables: dict) -> None:
    """The per-cell tables must reproduce what was scored, or nothing is drawn."""
    bad = []
    for p in PAIRINGS:
        h1, t = scores["H1"][p], tables[p]
        rows = t["miss"]
        if len(rows) != h1["n_cells"]:
            bad.append(f"{p}: {len(rows)} cells in the tables, {h1['n_cells']} scored")
        med = statistics.median(abs(r["rel"]) for r in rows)
        if abs(med - h1["recursion_median_abs_relative"]) > 1e-12:
            bad.append(f"{p}: median |rel| {med:.6f} != scored {h1['recursion_median_abs_relative']:.6f}")
        for r in rows:
            rel = (r["predicted"] - r["measured"]) / r["measured"]
            if abs(rel - r["rel"]) > 1e-12:
                bad.append(f"{p} {r['cell']}: rel is not (predicted - measured) / measured")
                break
        ks = t["knockout_summary"]
        for rv in RIVALS:
            want = h1["rivals_median_abs_relative"][rv]
            if abs(ks[rv]["all"]["median_abs"] - want) > 1e-12:
                bad.append(f"{p}: rival {rv} median {ks[rv]['all']['median_abs']:.6f} != scored {want:.6f}")
        cells_u = {u["cell"] for u in t["uncertainty"]}
        if cells_u != {r["cell"] for r in rows}:
            bad.append(f"{p}: the uncertainty rows name other cells than the miss rows")
    if bad:
        raise Refused("refused, the tables do not reproduce the scored values:\n  " + "\n  ".join(bad))


# ---------------------------------------------------------------- tables

def plan_pairings(plan_text: str) -> list[tuple[str, str, str, str]]:
    rows = []
    for line in plan_text.splitlines():
        m = re.match(r"\| (P[1-6]) \| ([^|]+) \| ([^|]+) \| ([^|]+) \|", line)
        if m and m.group(1) not in {r[0] for r in rows}:
            rows.append(tuple(g.strip() for g in m.groups()))
        if len(rows) == 6:
            break
    if [r[0] for r in rows] != ["P1", "P2", "P4", "P3", "P5", "P6"]:
        raise Refused(f"plan Section 5.2's pairing table did not parse: {[r[0] for r in rows]}")
    return sorted(rows, key=lambda r: r[0])


def verdict(scores: dict, h: str, p: str) -> str:
    """The verdict as scored. H1's is also checked against its tolerance."""
    v = scores[h][p]["verdict"]
    if h == "H1":
        s = scores["H1"][p]
        want = "fails" if s["recursion_median_abs_relative"] > s["tolerance"] else "holds"
        if v != want:
            raise Refused(f"H1 {p}: scored verdict {v!r} disagrees with its median and tolerance")
    return v


def tab_pairings(scores, lock, plan_rows) -> str:
    scored = {h: v for h, v in lock["scored_pairings"].items() if isinstance(v, list)}
    out = [r"\begin{table*}[t]", r"\centering", r"\small",
           r"\caption{\textbf{Pre-registered.} Pairings and their status at the lock "
           r"(\LockTag; analysis commit \AnalysisCommit; timed-path pin \TimedPathPin). "
           r"A verdict is the pre-registered hypothesis as scored; ``not scored'' means its "
           r"pre-registered test could not be read. Rows without a scored hypothesis carry "
           r"no result.}",
           r"\label{tab:pairings}",
           r"\begin{tabular}{@{}llllp{0.42\textwidth}@{}}", r"\toprule",
           r"Pairing & Host CPU & GPU & Provider & Status \\", r"\midrule"]
    for pid, host, gpu, prov in plan_rows:
        hs = [h for h, ps in scored.items() if pid in ps]
        if hs:
            status = "; ".join(f"{h} {verdict(scores, h, pid)}" for h in sorted(hs))
        else:
            status = STATUS_OUTSIDE_LOCK[pid]
        out.append(f"{pid} & {tex_escape(host)} & {tex_escape(gpu)} & {tex_escape(prov)} & "
                   f"{tex_escape(status)} \\\\")
    withdrawn = [h for h, v in lock["scored_pairings"].items() if not isinstance(v, list)]
    out += [r"\midrule",
            r"\multicolumn{5}{@{}l@{}}{" + ", ".join(withdrawn)
            + r": withdrawn before scoring (no pairing scored).} \\",
            r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(out) + "\n"


def tab_miss(tables) -> str:
    out = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
           r"\caption{\textbf{Exploratory / post hoc.} Where the committed model's miss "
           r"lies, summed over cells (ms). Error = device part (priced device chain "
           r"minus traced device-busy time) + idle part (predicted minus measured idle "
           r"time), exact by construction. Traced times come from profiled runs. "
           r"\emph{Host $>$ meas.} counts cells whose priced host time alone exceeds the "
           r"whole measured prefill. XAttn.\ is XAttention. Nothing here rescores H1.}",
           r"\label{tab:miss}",
           r"\setlength{\tabcolsep}{3pt}",
           r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
           r" & Arm & $n$ & Error & Device & Idle & \shortstack[r]{Host $>$\\meas.} \\", r"\midrule"]
    for p in PAIRINGS:
        rows, summ = tables[p]["miss"], tables[p]["miss_summary"]
        tot = {"n": 0, "e": 0.0, "d": 0.0, "i": 0.0, "h": 0}
        for k, arm in enumerate(ARMS):
            rs = [r for r in rows if r["arm"] == arm]
            e = sum(r["error_s"] for r in rs) * 1e3
            d = sum(r["device_part_s"] for r in rs) * 1e3
            i = sum(r["idle_part_s"] for r in rs) * 1e3
            h = sum(1 for r in rs if r["host"] > r["measured"])
            for key, val in (("n", len(rs)), ("e", e), ("d", d), ("i", i), ("h", h)):
                tot[key] += val
            lead = p if k == 0 else ""
            out.append(f"{lead} & {tex_escape(SHORT_ARM.get(arm, ARM_LABEL[arm]))} & {len(rs)} & {fmt(e, 1, True)} & "
                       f"{fmt(d, 1, True)} & {fmt(i, 1, True)} & {h} \\\\")
        # The per-arm sums must add up to the wave's own summary.
        if (tot["n"] != summ["n"] or tot["h"] != summ["cells_host_above_measured"]
                or abs(tot["e"] - summ["sum_error_s"] * 1e3) > 1e-6
                or abs(tot["d"] - summ["sum_device_part_s"] * 1e3) > 1e-6
                or abs(tot["i"] - summ["sum_idle_part_s"] * 1e3) > 1e-6):
            raise Refused(f"{p}: the miss decomposition's arms do not add up to its summary")
        out.append(f" & all & {tot['n']} & {fmt(tot['e'], 1, True)} & {fmt(tot['d'], 1, True)} & "
                   f"{fmt(tot['i'], 1, True)} & {tot['h']} \\\\")
        out.append(r"\midrule" if p != PAIRINGS[-1] else r"\bottomrule")
    out += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def tab_baselines(scores, tables) -> str:
    out = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
           r"\caption{Median relative error of each predictor over 35 cells per pairing. "
           r"\textbf{Pre-registered} above the rule: the committed recursion and H1's "
           r"pre-registered rivals, as scored (tolerance \HoneTolerance). "
           r"\textbf{Exploratory, in-sample} below it: the committed model with one term "
           r"removed or replaced, read on the same pairing's ground truth, so a knockout "
           r"that lowers the error was found on the data it is scored on.}",
           r"\label{tab:baselines}",
           r"\setlength{\tabcolsep}{3pt}",
           r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
           r" & \multicolumn{2}{c}{P2 (A100)} & \multicolumn{2}{c}{P4 (H100)} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
           r"Predictor & $|$median$|$ & signed & $|$median$|$ & signed \\", r"\midrule"]

    def row(name, preregistered):
        cells = []
        for p in PAIRINGS:
            ks = tables[p]["knockout_summary"][name]["all"]
            if preregistered:
                h1 = scores["H1"][p]
                a = (h1["recursion_median_abs_relative"] if name == "recursion"
                     else h1["rivals_median_abs_relative"][name])
            else:
                a = ks["median_abs"]
            cells += [fmt(a, 4), fmt(ks["median_signed"], 4, True)]
        return f"{PREDICTOR_LABEL[name]} & " + " & ".join(cells) + r" \\"

    out.append(row("recursion", True))
    out += [row(r, True) for r in RIVALS]
    out.append(r"\midrule")
    out += [row(k, False) for k in KNOCKOUTS]
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def parse_threats(md: str) -> list[tuple[str, int, str, bool]]:
    rows, cls, cur = [], None, None
    for line in md.splitlines():
        m = re.match(r"## (\w+) validity", line)
        if m:
            cls = m.group(1)
            continue
        m = re.match(r"(\d+)\. \*\*(.+?)\*\*", line)
        if m and cls:
            cur = [cls, int(m.group(1)), m.group(2).rstrip(".:"), line]
            rows.append(cur)
        elif cur is not None and line.strip():
            cur[3] += "\n" + line
    out = []
    for c, n, lead, body in rows:
        expl = bool(re.search(r"exploratory|post hoc|post-hoc", body, re.I))
        out.append((c, n, lead, expl))
    if [r[1] for r in out] != list(range(1, len(out) + 1)) or len(out) < 10:
        raise Refused("the threats list did not parse as one numbered list")
    return out


def tab_threats(threats) -> str:
    out = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
           r"\caption{Threats to validity (a list, not a result). \emph{Exploratory} marks a "
           r"threat whose entry in the source list cites numbers labelled "
           r"\textbf{exploratory or post hoc}; the mark follows the list's own labels, and an "
           r"unmarked threat concerns design or scope, or cites \textbf{pre-registered} results.}",
           r"\label{tab:threats}",
           r"\begin{tabular}{@{}llp{0.70\textwidth}l@{}}", r"\toprule",
           r"Validity & \# & Threat & Exploratory \\", r"\midrule"]
    last = None
    for c, n, lead, expl in threats:
        out.append(f"{c if c != last else ''} & {n} & {tex_escape(lead)} & "
                   f"{'yes' if expl else ''} \\\\")
        last = c
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(out) + "\n"


def numbers(scores, lock) -> str:
    v = lock["values"]
    pin = v["timed_path_identity_for_p2_and_p4"]["value"]
    pin = pin if isinstance(pin, str) else json.dumps(pin)
    m = re.search(r"\b([0-9a-f]{8})", pin)
    h1 = scores["H1"]
    defs = {
        # Names a reviewer could trace; blank or replace for the submission.
        "LockTag": tex_escape(scores_lock_tag(scores)),
        "AnalysisCommit": r"\texttt{" + v["analysis_commit"]["value"][:7] + "}",
        "TimedPathPin": r"\texttt{" + (m.group(1) if m else "?") + "}",
        # Pre-registered values as scored.
        "HoneTolerance": fmt(h1["P2"]["tolerance"], 2),
        "HoneCellsPtwo": str(h1["P2"]["n_cells"]),
        "HoneCellsPfour": str(h1["P4"]["n_cells"]),
        "HoneMedianPtwo": fmt(h1["P2"]["recursion_median_abs_relative"], 4),
        "HoneMedianPfour": fmt(h1["P4"]["recursion_median_abs_relative"], 4),
        "HoneFittedSumPtwo": fmt(h1["P2"]["rivals_median_abs_relative"]["fitted_sum"], 4),
        "HoneFittedSumPfour": fmt(h1["P4"]["rivals_median_abs_relative"]["fitted_sum"], 4),
        "HoneStepFormPtwo": fmt(h1["P2"]["rivals_median_abs_relative"]["step_form"], 4),
        "HoneStepFormPfour": fmt(h1["P4"]["rivals_median_abs_relative"]["step_form"], 4),
    }
    out = ["% Macros the paper's text may use. Generated; never type a number into the text."]
    out += [f"\\newcommand{{\\{k}}}{{{val}\\xspace}}" for k, val in defs.items()]
    return "\n".join(out) + "\n"


def scores_lock_tag(scored_file: dict) -> str:
    return scored_file["_lock_tag"]


# ---------------------------------------------------------------- figures

def _plt():
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "pdf.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 7,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "legend.frameon": False, "axes.spines.top": False,
        "axes.spines.right": False, "svg.hashsalt": "attnbench",
    })
    return plt


def _save(fig, path: Path):
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.02,
                metadata={"CreationDate": None, "ModDate": None, "Producer": None,
                          "Creator": None})


def _cells_in_order(rows):
    return sorted(rows, key=lambda r: (cell_parts(r["cell"])[3], cell_parts(r["cell"])[2],
                                       ARMS.index(r["arm"]), cell_parts(r["cell"])[1]))


def fig_pred_vs_meas(tables, out: Path) -> list[list]:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    data = [["pairing", "cell", "arm", "predicted_ms", "measured_ms"]]
    for ax, p in zip(axes, PAIRINGS):
        rows = tables[p]["miss"]
        lo = min(min(r["predicted"], r["measured"]) for r in rows) * 1e3 / 1.3
        hi = max(max(r["predicted"], r["measured"]) for r in rows) * 1e3 * 1.3
        tol = tables[p]["h1_tolerance"]
        ax.fill_between([lo, hi], [lo * (1 - tol), hi * (1 - tol)], [lo * (1 + tol), hi * (1 + tol)],
                        color=GRID, alpha=0.6, lw=0, label=f"within $\\pm${tol:g}")
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8)
        for arm in ARMS:
            rs = [r for r in rows if r["arm"] == arm]
            ax.scatter([r["measured"] * 1e3 for r in rs], [r["predicted"] * 1e3 for r in rs],
                       s=16, marker=ARM_MARKER[arm], color=ARM_COLOR[arm],
                       edgecolors="white", linewidths=0.5, label=ARM_LABEL[arm], zorder=3)
        for r in _cells_in_order(rows):
            data.append([p, r["cell"], r["arm"], f"{r['predicted'] * 1e3:.4f}", f"{r['measured'] * 1e3:.4f}"])
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("measured prefill (ms)"); ax.set_ylabel("predicted (ms)")
        ax.set_title(f"{p} ({GPU[p]})", color=INK, fontsize=8, loc="left")
        ax.grid(True, which="major", color=GRID, lw=0.4)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.06), fontsize=7)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, out)
    plt.close(fig)
    return data


def fig_error_by_cell(tables, out: Path) -> list[list]:
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.6), sharex="col",
                             gridspec_kw={"height_ratios": (3, 2)})
    data = [["pairing", "cell", "arm", "signed_rel", "rel_lo", "rel_hi", "signed_error_ms"]]
    for col, p in enumerate(PAIRINGS):
        rows = _cells_in_order(tables[p]["miss"])
        unc = {u["cell"]: u for u in tables[p]["uncertainty"]}
        tol = tables[p]["h1_tolerance"]
        xs = list(range(len(rows)))
        top, bot = axes[0][col], axes[1][col]
        top.axhspan(-tol, tol, color=GRID, alpha=0.6, lw=0)
        top.axhline(0, color=MUTED, lw=0.6)
        bot.axhline(0, color=MUTED, lw=0.6)
        for x, r in zip(xs, rows):
            u = unc[r["cell"]]
            ms = (r["predicted"] - r["measured"]) * 1e3
            kw = dict(marker=ARM_MARKER[r["arm"]], color=ARM_COLOR[r["arm"]], ms=4,
                      mec="white", mew=0.4, zorder=3)
            top.errorbar([x], [r["rel"]], yerr=[[r["rel"] - u["rel_lo"]], [u["rel_hi"] - r["rel"]]],
                         ecolor=INK, elinewidth=0.6, capsize=0, **kw)
            bot.plot([x], [ms], ls="none", **kw)
            data.append([p, r["cell"], r["arm"], f"{r['rel']:.6f}", f"{u['rel_lo']:.6f}",
                         f"{u['rel_hi']:.6f}", f"{ms:.4f}"])
        # Group ticks by context length (and batch).
        groups = {}
        for x, r in zip(xs, rows):
            _, _, n, b = cell_parts(r["cell"])
            groups.setdefault((b, n), []).append(x)
        ticks = [statistics.mean(v) for v in groups.values()]
        labels = [f"{n}" if b == 1 else f"$\\times${b}" for b, n in groups]
        for v in list(groups.values())[1:]:
            for ax in (top, bot):
                ax.axvline(v[0] - 0.5, color=GRID, lw=0.5)
        bot.set_xticks(ticks, labels)
        bot.set_xlabel("cells, by realised prompt length (tokens)")
        top.set_title(f"{p} ({GPU[p]})", color=INK, fontsize=8, loc="left")
        for ax in (top, bot):
            ax.margins(y=0.08)
        if col == 0:
            top.set_ylabel("signed relative error")
            bot.set_ylabel("signed error (ms)")
        for ax in (top, bot):
            ax.grid(True, axis="y", color=GRID, lw=0.4)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], ls="none", marker=ARM_MARKER[a], color=ARM_COLOR[a], mec="white",
                      mew=0.4, ms=5, label=ARM_LABEL[a]) for a in ARMS]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04), fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, out)
    plt.close(fig)
    return data


def fig_env(name: str, caption: str, label: str) -> str:
    return "\n".join([r"\begin{figure*}[t]", r"\centering",
                      r"\paperfigure[\textwidth]{" + name + ".pdf}",
                      r"\caption{" + caption + "}", r"\label{" + label + "}",
                      r"\end{figure*}"]) + "\n"


def write_csv(rows: list[list]) -> str:
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    return buf.getvalue()


# ---------------------------------------------------------------- main

def build(scored_path: Path, tables_path: Path, lock_path: Path, plan_path: Path,
          threats_path: Path, out: Path) -> list[Path]:
    scored_file = json.loads(scored_path.read_text())
    scores = dict(scored_file["scores"])
    scores["_lock_tag"] = scored_file["lock"]["tag"]
    tables = json.loads(tables_path.read_text())
    lock = json.loads(lock_path.read_text())
    check_against_scores(scores, tables)
    # The scored file's other fields name local paths, which differ between
    # machines; the hash covers only what is read (its scores and lock tag).
    used = json.dumps({"scores": scored_file["scores"], "lock_tag": scored_file["lock"]["tag"]},
                      sort_keys=True).encode()
    inputs = {"scored (scores and lock tag)": hashlib.sha256(used).hexdigest(),
              "tables": sha(tables_path), "lock values": sha(lock_path),
              "plan": sha(plan_path), "threats": sha(threats_path)}
    gen, figs = out / "generated", out / "figures"
    gen.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    plan_rows = plan_pairings(plan_path.read_text())
    texts = {
        "numbers.tex": numbers(scores, lock),
        "tab_pairings.tex": tab_pairings(scores, lock, plan_rows),
        "tab_miss.tex": tab_miss(tables),
        "tab_baselines.tex": tab_baselines(scores, tables),
        "tab_threats.tex": tab_threats(parse_threats(threats_path.read_text())),
        "fig_pred_vs_meas.tex": fig_env(
            "fig_pred_vs_meas",
            r"\textbf{Pre-registered.} The committed prediction against the measured "
            r"prefill (median of rounds) for each of the \HoneCellsPtwo cells per pairing; "
            r"log scales. The diagonal is a perfect prediction and the band is H1's tolerance "
            r"($\pm$\HoneTolerance). The median absolute relative error over these cells is "
            r"H1's scored value: \HoneMedianPtwo on P2 and \HoneMedianPfour on P4 (both fail).",
            "fig:predmeas"),
        "fig_error_by_cell.tex": fig_env(
            "fig_error_by_cell",
            r"\textbf{Pre-registered} errors, with \textbf{exploratory} intervals. Top: H1's "
            r"pre-registered relative error for each cell, signed, (predicted $-$ measured) / "
            r"measured; the band is H1's tolerance, and the median of its absolute value over "
            r"the cells is H1's scored value (\HoneMedianPtwo on P2, \HoneMedianPfour on P4). "
            r"Whiskers are post hoc percentile bootstrap intervals over the cell's rounds "
            r"(within one session; often narrower than the marker). Bottom: the signed error "
            r"(ms), predicted $-$ measured, not a scored quantity. Cells are grouped by "
            r"realised prompt length; the last cell ($\times$4) is the dense cell at batch 4.",
            "fig:errcell"),
    }
    written = []
    for name, body in texts.items():
        kind = {"numbers.tex": "macros"}.get(name, "table or figure environment")
        path = gen / name
        path.write_text(header(kind, inputs) + body)
        written.append(path)
    for name, fn in (("fig_pred_vs_meas", fig_pred_vs_meas), ("fig_error_by_cell", fig_error_by_cell)):
        rows = fn(tables, figs / f"{name}.pdf")
        (figs / f"{name}.csv").write_text(write_csv(rows))
        written += [figs / f"{name}.pdf", figs / f"{name}.csv"]
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scored", type=Path, default=ROOT / "docs/scoring/hypotheses_scored_20261116.json")
    ap.add_argument("--tables", type=Path, default=ROOT / "docs/paper/exploratory/paper_tables.json")
    ap.add_argument("--lock", type=Path, default=ROOT / "docs/lock_values.json")
    ap.add_argument("--plan", type=Path, default=ROOT / "docs/RESEARCH_PLAN.md")
    ap.add_argument("--threats", type=Path, default=ROOT / "docs/paper/threats_to_validity.md")
    ap.add_argument("--out", type=Path, required=True, help="directory that gets generated/ and figures/")
    a = ap.parse_args(argv)
    try:
        written = build(a.scored, a.tables, a.lock, a.plan, a.threats, a.out)
    except Refused as e:
        print(e, file=sys.stderr)
        return 2
    for p in written:
        print(p.relative_to(a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
