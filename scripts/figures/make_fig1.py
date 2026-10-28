#!/usr/bin/env python3
"""
Figure 1 for BUNGA: host / link / device lanes for one layer on two GPUs.

Usage
  python make_fig1.py --events events.csv [--meta meta.json] --out fig1.pdf
  python make_fig1.py --demo --out fig1_layout_test.pdf     # layout test only

events.csv columns (times in milliseconds, from a measured trace):
  gpu,lane,op,start_ms,end_ms,kind
    gpu   : label of the card, e.g. L4 or A100 (one panel per label, in order of appearance)
    lane  : host | link | device
    op    : short label drawn on the bar (may be empty)
    kind  : mask_build | sync | copy | kernel | other

meta.json (optional, all values are strings or numbers you supply):
  {
    "A100": {"kernel_gain": "kernel 1.96x", "net": "whole prefill 0.63x",
             "predicted_end_ms": 41.2},
    "L4":   {...}
  }
  The script never computes or invents these values. Whatever you put in
  meta.json is printed as given.

--demo draws synthetic bars with a "LAYOUT TEST, NOT DATA" watermark so the
layout can be checked before any measured trace exists. Do not use its output
in the paper.
"""
import argparse
import csv
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

LANES = ["host", "link", "device"]
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


def draw(panels, meta, out, demo):
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(3.4, 1.55 * n + 0.6), sharex=True)
    axes = [axes] if n == 1 else list(axes)
    xmax = max(e["end"] for evs in panels.values() for e in evs)

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

        m = meta.get(gpu, {})
        if "predicted_end_ms" in m:
            ax.axvline(float(m["predicted_end_ms"]), color="black", ls="--", lw=0.8)
            ax.text(float(m["predicted_end_ms"]), -0.62, "predicted end",
                    fontsize=5.5, ha="right", va="bottom")
        notes = [m[k] for k in ("kernel_gain", "net") if k in m]
        if notes:
            ax.text(0.99, 0.04, "; ".join(notes), transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=6,
                    bbox=dict(fc="white", ec="black", lw=0.4, pad=1.5))
        if demo:
            ax.text(0.5, 0.5, "LAYOUT TEST, NOT DATA", transform=ax.transAxes,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events")
    ap.add_argument("--meta")
    ap.add_argument("--out", default="fig1.pdf")
    ap.add_argument("--demo", action="store_true",
                    help="synthetic bars for a layout check only")
    a = ap.parse_args()
    if a.demo:
        panels, meta = demo_events(), {}
    else:
        if not a.events:
            sys.exit("give --events events.csv, or --demo for a layout test")
        panels = read_events(a.events)
        meta = json.load(open(a.meta, encoding="utf8")) if a.meta else {}
    draw(panels, meta, a.out, a.demo)


if __name__ == "__main__":
    main()