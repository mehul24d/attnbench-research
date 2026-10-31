#!/usr/bin/env python3
"""
Figure 1 for BUNGA: host / link / device lanes for one layer on two GPUs.

Usage
  python make_fig1.py --events events.csv --layer 14 \\
      [--meta L4=meta_l4.json --meta A100=meta_a100.json] \\
      [--note A100="kernel ...x; whole prefill ...x"] --out fig1.pdf
  python make_fig1.py --demo --out fig1_layout_test.pdf     # layout test only

events.csv columns (times in milliseconds from the layer's start, from a
measured trace; written by scripts/figures/fig1_events.py):
  gpu,lane,op,start_ms,end_ms,kind
    gpu   : label of the card, e.g. L4 or A100 (one panel per label, in order of appearance)
    lane  : host | link | device
    op    : short label drawn on the bar (may be empty)
    kind  : mask_build | sync | copy | kernel | other

--meta GPU=meta.json, one per panel that shows a predicted end: the figure's
metadata from scripts/figures/figure_meta.py, made with --layer. The dashed
"predicted end" line is read through attnbench.figures.meta.predicted_end_ms,
which returns the recursion's predicted end of --layer, in ms from the
layer's start, and refuses metadata with no per-layer value, for another
layer, or whose value is the whole prefill's. The metadata's code path must
be for the panel's GPU.

--note GPU=TEXT prints TEXT in the panel's corner, as given. The script never
computes or invents a value.

--demo draws synthetic bars with a "LAYOUT TEST, NOT DATA" watermark so the
layout can be checked before any measured trace exists. Do not use its output
in the paper. It takes no --events, --meta or --note, so a real figure is never
drawn in demo mode.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# TrueType (Type 42) fonts in the PDF, not Type 3, as venues require.
matplotlib.rcParams["pdf.fonttype"] = 42

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from attnbench.figures.meta import MetaError, predicted_end_ms  # noqa: E402

LANES = ["host", "link", "device"]
WATERMARK = "LAYOUT TEST, NOT DATA"
# Colour-blind-safe palette plus hatches, so the figure also reads in grayscale.
STYLE = {
    "mask_build": dict(color="#D55E00", hatch="//"),
    "sync":       dict(color="#CC79A7", hatch="xx"),
    "copy":       dict(color="#E69F00", hatch=".."),
    "kernel":     dict(color="#0072B2", hatch=""),
    "other":      dict(color="#BBBBBB", hatch=""),
}
LEGEND = [
    ("mask_build", "mask build (host)"),
    ("sync", "synchronization"),
    ("copy", "host-device copy"),
    ("kernel", "GPU kernel"),
    ("other", "other op"),
]


def read_events(path):
    panels = {}
    with open(path, newline="", encoding="utf8") as f:
        for row in csv.DictReader(f):
            lane = row["lane"].strip().lower()
            if lane not in LANES:
                sys.exit(f"unknown lane {lane!r}; use host, link or device")
            kind = row["kind"].strip().lower()
            if kind not in STYLE:
                sys.exit(f"unknown kind {kind!r}; use one of {list(STYLE)}")
            panels.setdefault(row["gpu"].strip(), []).append(dict(
                lane=lane, op=row["op"].strip(), kind=kind,
                start=float(row["start_ms"]), end=float(row["end_ms"])))
    if not panels:
        sys.exit("no events found")
    return panels


def _pairs(items, flag):
    out = {}
    for item in items or ():
        gpu, sep, value = item.partition("=")
        if not sep or not gpu.strip() or not value:
            sys.exit(f"{flag} takes GPU=VALUE, got {item!r}")
        if gpu.strip() in out:
            sys.exit(f"{flag} names {gpu.strip()!r} twice")
        out[gpu.strip()] = value
    return out


def predicted_ends(meta_paths, layer, panels):
    """{gpu: predicted end in ms}, each read through predicted_end_ms()."""
    ends = {}
    for gpu, path in meta_paths.items():
        if gpu not in panels:
            sys.exit(f"--meta names {gpu!r}, which has no events")
        meta = json.loads(Path(path).read_text(encoding="utf8"))
        card = str(meta.get("cell", {}).get("code_path", "")).split("|")[0]
        if gpu.lower() not in card.lower():
            sys.exit(f"{path} is for a code path on {card!r}, not the {gpu} panel")
        try:
            ends[gpu] = predicted_end_ms(meta, layer)
        except MetaError as e:
            sys.exit(f"{path}: {e}")
    return ends


def demo_events():
    """Synthetic bars for a layout check only. Not measurements."""
    def trace(scale_dev, gap):
        ev = [("host", "mask", "mask_build", 0.0, 3.0),
              ("host", "sync", "sync", 3.0, 3.4),
              ("host", "launch", "other", 3.4, 4.0),
              ("link", "copy", "copy", 3.0, 3.4),
              ("device", "prior", "other", 0.0, 3.0 * scale_dev),
              ("device", "attn", "kernel", 3.4 + gap, 3.4 + gap + 2.0)]
        return [dict(lane=l, op=o, kind=k, start=s, end=e) for l, o, k, s, e in ev]
    return {"Card A": trace(1.0, 0.0), "Card B": trace(0.4, 0.6)}


def draw(panels, ends, notes, out, demo):
    """Draw and save; returns the figure. `ends` and `notes` are by gpu."""
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(3.4, 1.55 * n + 0.6), sharex=True)
    axes = [axes] if n == 1 else list(axes)
    xmax = max([e["end"] for evs in panels.values() for e in evs] + list(ends.values()))

    for ax, (gpu, evs) in zip(axes, panels.items()):
        for e in evs:
            y = LANES.index(e["lane"])
            st = STYLE[e["kind"]]
            ax.barh(y, e["end"] - e["start"], left=e["start"], height=0.62,
                    color=st["color"], hatch=st["hatch"], edgecolor="black",
                    linewidth=0.4)
            if e["op"] and (e["end"] - e["start"]) > 0.07 * xmax:
                ax.text((e["start"] + e["end"]) / 2, y, e["op"], ha="center",
                        va="center", fontsize=5.5, color="black",
                        bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.4))
        ax.set_yticks(range(len(LANES)))
        ax.set_yticklabels(LANES, fontsize=6.5)
        ax.invert_yaxis()
        ax.set_xlim(0, xmax * 1.02)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.set_title(gpu, fontsize=7.5, loc="left", pad=2, fontweight="bold")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

        if gpu in ends:
            ax.axvline(ends[gpu], color="black", ls="--", lw=0.8)
            ax.text(ends[gpu], -0.62, "predicted end",
                    fontsize=5.5, ha="right", va="bottom")
        if gpu in notes:
            ax.text(0.99, 0.04, notes[gpu], transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=6,
                    bbox=dict(fc="white", ec="black", lw=0.4, pad=1.5))
        if demo:
            ax.text(0.5, 0.5, WATERMARK, transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="red",
                    alpha=0.45, rotation=12, fontweight="bold")

    axes[-1].set_xlabel("time within one layer (ms)", fontsize=7)
    handles = [Patch(fc=STYLE[k]["color"], hatch=STYLE[k]["hatch"], ec="black",
                     lw=0.4, label=lab) for k, lab in LEGEND]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=5.8,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print("wrote", out)
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--events")
    ap.add_argument("--meta", action="append", metavar="GPU=META_JSON")
    ap.add_argument("--layer", type=int, help="the layer the events and metadata are of")
    ap.add_argument("--note", action="append", metavar="GPU=TEXT")
    ap.add_argument("--out", default="fig1.pdf")
    ap.add_argument("--demo", action="store_true",
                    help="synthetic bars for a layout check only")
    a = ap.parse_args(argv)
    if a.demo:
        if a.events or a.meta or a.note:
            sys.exit("--demo draws synthetic bars only; it takes no --events, --meta or --note")
        draw(demo_events(), {}, {}, a.out, True)
        return 0
    if not a.events:
        sys.exit("give --events events.csv, or --demo for a layout test")
    panels = read_events(a.events)
    metas = _pairs(a.meta, "--meta")
    if metas and a.layer is None:
        sys.exit("--meta needs --layer: the predicted end is read for that layer")
    ends = predicted_ends(metas, a.layer, panels)
    notes = _pairs(a.note, "--note")
    for gpu in notes:
        if gpu not in panels:
            sys.exit(f"--note names {gpu!r}, which has no events")
    draw(panels, ends, notes, a.out, False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
