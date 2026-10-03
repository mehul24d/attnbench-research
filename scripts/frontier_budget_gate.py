#!/usr/bin/env python
"""The budget gate of the estimator-frontier pre-registration (sec. 8.3).

Run before every launch. It reads the spend ledger and the plan state, and
prints PROCEED or STOP for one session:

    spent + U(next) + sum U(remaining never-cut sessions)
          + sum U(higher-priority cuttable items not yet run) + S_res + R_B  <=  Rs 12,000

The rule and the arithmetic are `attnbench.analysis.frontier_prereg.gate`.
This script only reads and writes the ledger around it. The launchers call it
through `scripts/frontier_gate.sh` and refuse to create an instance unless it
exits 0.

    frontier_budget_gate.py --init
    frontier_budget_gate.py --session I1 --card A100
    frontier_budget_gate.py --fund SL
    frontier_budget_gate.py --record-launch INSTANCE --session I1 --card A100
    frontier_budget_gate.py --record-teardown INSTANCE --minutes 131 --status complete

Exit codes: 0 PROCEED, 1 STOP, 2 the gate could not give a verdict.

Sessions are bracket line names ("I1", "C", "XL0"), or "LINE:unit" where a
line is more than one session ("X primary:32768", "R A100x3:1").

The plan state is an optional JSON file with any of: "cut" and "released"
(line names), "xa_withheld", "h6a_passed", "months_remaining", and "inputs"
(measured overrides of BracketInputs, written at DP1 and DP2).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from attnbench.analysis import frontier_prereg as fp  # noqa: E402

LEDGER = REPO / "results" / "frontier_spend.csv"
STATE = REPO / "results" / "frontier_plan_state.json"


def read_rows(path: Path) -> list:
    if not path.exists():
        raise fp.GateRefusal(f"no ledger at {path}; create it with --init. "
                             "A missing ledger is not read as zero spend.")
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != fp.LEDGER_COLUMNS:
            raise fp.GateRefusal(f"ledger header is {reader.fieldnames}, not {list(fp.LEDGER_COLUMNS)}")
        return list(reader)


def read_state(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        state = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise fp.GateRefusal(f"plan state {path} is not JSON: {e}") from None
    if not isinstance(state, dict):
        raise fp.GateRefusal(f"plan state {path} is not an object")
    return state


def append(path: Path, row: dict) -> None:
    with path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=fp.LEDGER_COLUMNS).writerow(row)


def today() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--session")
    ap.add_argument("--card", choices=sorted(fp.RATE))
    ap.add_argument("--fund", metavar="ARM", help="funding decision for an arm (no launch)")
    ap.add_argument("--init", action="store_true", help="create an empty ledger")
    ap.add_argument("--record-launch", metavar="INSTANCE")
    ap.add_argument("--record-teardown", metavar="INSTANCE")
    ap.add_argument("--minutes", type=float)
    ap.add_argument("--status", choices=("complete", "failed"))
    ap.add_argument("--ledger", type=Path, default=LEDGER)
    ap.add_argument("--state", type=Path, default=STATE)
    a = ap.parse_args(argv)
    try:
        if a.init:
            if a.ledger.exists():
                raise fp.GateRefusal(f"{a.ledger} exists; not overwriting it")
            a.ledger.parent.mkdir(parents=True, exist_ok=True)
            a.ledger.write_text(",".join(fp.LEDGER_COLUMNS) + "\n")
            print(f"created {a.ledger}")
            return 0
        rows, state = read_rows(a.ledger), read_state(a.state)
        if a.record_teardown:
            opened = [r for r in rows if r["instance"] == a.record_teardown and r["status"] == "launched"]
            closed = [r for r in rows if r["instance"] == a.record_teardown and r["status"] != "launched"]
            if len(opened) != 1 or closed or a.minutes is None or a.minutes < 0 or a.status is None:
                raise fp.GateRefusal("--record-teardown needs one open launch for that instance, "
                                     "--minutes and --status")
            o = opened[0]
            append(a.ledger, {"date": today(), "session": o["session"], "instance": o["instance"],
                              "card": o["card"], "minutes": f"{a.minutes:g}",
                              "est_inr": f"{a.minutes / 60 * fp.RATE[o['card']]:.0f}",
                              "status": a.status, "source": "frontier_budget_gate --record-teardown"})
            print(f"recorded {a.status}: {o['session']} on {o['instance']}, {a.minutes:g} min")
            return 0
        session = a.fund or a.session
        if not session or (a.fund and (a.card or a.record_launch)) or (a.session and not a.card):
            raise fp.GateRefusal("give --session and --card, or --fund ARM")
        g = fp.gate(session, a.card, rows, state)
    except fp.GateRefusal as e:
        print(f"GATE REFUSED: {e}", file=sys.stderr)
        return 2
    kind = "never-cut" if g["star"] else f"cuttable, cut-order item {g['cut_item']}"
    print(f"frontier budget gate: {g['session']} ({kind})")
    for label, key in (("spent", "spent"), ("U(this session)", "next_u"),
                       ("remaining never-cut sessions", "remaining_star"),
                       ("higher-priority cuttable items", "reserved_cuttable"),
                       ("storage reserve S_res", "s_res"), ("R_B (I2c-B)", "r_b")):
        print(f"  {label:32s} Rs {g[key]:>9,.0f}")
    print(f"  {'needed':32s} Rs {g['need']:>9,.0f}   cap Rs {g['cap']:,}")
    print(f"VERDICT={g['verdict']}")
    if g["verdict"] != "PROCEED":
        if not g["star"]:
            print(f"Cut it (item {g['cut_item']}) and continue down sec. 8.4.")
        else:
            print("Cut the remaining cuttable items in sec. 8.4 order. If none is left, the study stops.")
        return 1
    print(f"HALT_MINUTES={g['halt_minutes']}")
    print(f"MAX_RUN_MINUTES={g['max_run_minutes']}")
    if a.record_launch:
        line_card = a.card
        append(a.ledger, {"date": today(), "session": session, "instance": a.record_launch,
                          "card": line_card, "minutes": str(g["max_run_minutes"]),
                          "est_inr": f"{g['max_run_minutes'] / 60 * fp.RATE[line_card]:.0f}",
                          "status": "launched", "source": "frontier_budget_gate --record-launch"})
        print(f"recorded launch: {session} on {a.record_launch}, reserved to its hard cap")
    return 0


if __name__ == "__main__":
    sys.exit(main())
