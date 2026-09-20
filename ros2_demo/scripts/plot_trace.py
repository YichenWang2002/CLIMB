#!/usr/bin/env python3
"""Plot a paper-style execution timeline from a bt_runtime trace JSON.

Input: the JSON written by runner_node.py's `trace_json` parameter:

    {
      "success": true, "reason": "goal_reached", "ticks": 42, "recoveries": 1,
      "trajectory": {
        "alpha": [{"tick": 3, "kind": "action",
                   "label": "MoveTo(robot=alpha,from=shelf_b,to=charge_bay)",
                   "status": "SUCCESS"}, ...],
        "beta":  [...]
      }
    }

(A bare {"<agent>": [events...]} mapping without the envelope is accepted too.)

Output: one PNG with one swimlane per robot, events as markers on the tick
axis, rendezvous (Handover / Co* joint completions and SignalReady/WaitReady
barriers) marked with vertical dashed lines.

Pure offline script: matplotlib only, NO ROS required.

Usage:
    python3 plot_trace.py TRACE.json -o out.png [--show-conditions]
    python3 plot_trace.py --demo -o demo_trace.png   # self-test with a fake trace
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# tag -> (color, marker, display name)
STYLE = {
    "MoveTo": ("#1f77b4", "o", "MoveTo"),
    "PickUp": ("#2ca02c", "^", "PickUp"),
    "PlaceDown": ("#2ca02c", "v", "PlaceDown"),
    "Handover": ("#9467bd", "D", "Handover"),
    "HandoverGive": ("#9467bd", "D", "Handover"),
    "HandoverTake": ("#9467bd", "D", "Handover"),
    "CoPickUp": ("#8c564b", "s", "CoPickUp"),
    "CoMoveTo": ("#8c564b", "s", "CoMoveTo"),
    "CoPlaceDown": ("#8c564b", "s", "CoPlaceDown"),
    "ClearPath": ("#ff7f0e", "*", "ClearPath"),
    "Recharge": ("#d4af37", "P", "Recharge"),
    "SignalReady": ("#7f7f7f", ">", "Signal/Wait"),
    "WaitReady": ("#7f7f7f", "<", "Signal/Wait"),
    "CatalogBook": ("#17becf", "h", "CatalogBook"),
    "WaterPlants": ("#17becf", "h", "WaterPlants"),
    "InspectPlants": ("#17becf", "h", "InspectPlants"),
}
DEFAULT_STYLE = ("#7f7f7f", "x", "other")
# recovery leaves get a status annotation in the summary box
RECOVERY_TAGS = ("ClearPath", "Recharge")

_TAG_RE = re.compile(r"^([A-Za-z0-9_]+)\((.*)\)$")


def parse_label(label: str):
    """'MoveTo(robot=alpha,from=a,to=b)' -> ('MoveTo', {robot: alpha, ...})."""
    m = _TAG_RE.match(label.strip())
    if not m:
        return label.strip(), {}
    tag, argstr = m.groups()
    attrs = {}
    for part in argstr.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            attrs[k.strip()] = v.strip()
    return tag, attrs


def load_trace(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    if "trajectory" in data:
        return data["trajectory"], data
    return data, {}  # bare per-agent mapping


def find_rendezvous(traj: dict):
    """Group joint/sync events that belong to the same rendezvous.

    Returns [(tick, name)] where tick is the completion tick (max of the
    group's ticks) and name a short annotation."""
    groups = {}
    for agent, events in traj.items():
        for ev in events:
            if ev["kind"] not in ("joint", "sync"):
                continue
            tag, attrs = parse_label(ev["label"])
            if ev["kind"] == "joint":
                canon = ("Handover" if tag in ("HandoverGive", "HandoverTake")
                         else tag)
                key = ("joint", canon,
                       tuple(sorted((k, v) for k, v in attrs.items()
                                    if k in ("item", "location", "from", "to"))))
                name = f"rendezvous: {canon}"
            else:  # SignalReady / WaitReady barrier
                key = ("signal", attrs.get("item"),
                       frozenset([attrs.get("robot"), attrs.get("to"),
                                  attrs.get("from")]))
                name = f"sync: {attrs.get('item', '?')}"
            groups.setdefault(key, {"ticks": [], "name": name})
            groups[key]["ticks"].append(ev["tick"])
    out = []
    for g in groups.values():
        if len(g["ticks"]) >= 2:  # both sides posted
            out.append((max(g["ticks"]), g["name"]))
    return sorted(out)


def plot(traj: dict, meta: dict, out_path: str, show_conditions: bool,
         title: str = None):
    agents = sorted(traj)
    lane_y = {a: len(agents) - 1 - i for i, a in enumerate(agents)}  # alpha on top

    fig, ax = plt.subplots(figsize=(max(9, 0.55 * max(
        (ev["tick"] for evs in traj.values() for ev in evs), default=10)), 1.6 * len(agents) + 2))

    used_styles = {}
    for agent in agents:
        y = lane_y[agent]
        ax.axhline(y, color="#dddddd", lw=1.0, zorder=0)
        for ev in traj[agent]:
            if ev["kind"] == "condition" and not show_conditions:
                continue
            tag, _attrs = parse_label(ev["label"])
            color, marker, disp = STYLE.get(tag, DEFAULT_STYLE)
            used_styles.setdefault(disp, (color, marker))
            failed = ev["status"] != "SUCCESS"
            ax.scatter(ev["tick"], y, s=140 if tag in RECOVERY_TAGS else 90,
                       marker=marker,
                       facecolors="none" if failed else color,
                       edgecolors="#d62728" if failed else color,
                       linewidths=2.0 if failed else 1.2,
                       zorder=3)
            short = tag if tag not in ("SignalReady", "WaitReady") else None
            if short:
                ax.annotate(short, (ev["tick"], y), textcoords="offset points",
                            xytext=(0, 9), ha="center", fontsize=7, color="#444444")

    for tick, name in find_rendezvous(traj):
        ax.axvline(tick, color="#9467bd", ls="--", lw=1.2, alpha=0.7, zorder=1)
        ax.annotate(name, (tick, -1.45), rotation=90, fontsize=7,
                    color="#9467bd", va="bottom", ha="right")

    ax.set_yticks([lane_y[a] for a in agents])
    ax.set_yticklabels(agents, fontsize=11)
    ax.set_ylim(-1.6, len(agents) - 0.2)
    ax.set_xlabel("BT tick", fontsize=11)
    ax.set_title(title or meta.get("reason", "BT execution trace"), fontsize=12)
    ax.grid(axis="x", color="#eeeeee")

    legend = [Line2D([0], [0], marker=m, color="w", markerfacecolor=c,
                     markeredgecolor=c, markersize=9, label=name)
              for name, (c, m) in sorted(used_styles.items())]
    legend.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                         markeredgecolor="#d62728", markersize=9,
                         label="FAILED event"))
    ax.legend(handles=legend, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=min(6, len(legend)),
              fontsize=8, frameon=False)

    if meta:
        info = (f"success={meta.get('success')}  reason={meta.get('reason')}  "
                f"ticks={meta.get('ticks')}  recoveries={meta.get('recoveries')}")
        fig.text(0.99, 0.01, info, ha="right", fontsize=8, color="#666666")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return out_path


def demo_trace():
    """Fake trace mimicking demo_B_relay_blocked: blocked edge -> ClearPath,
    then a Handover rendezvous at shelf_b."""
    def ev(tick, kind, label, status="SUCCESS"):
        return {"tick": tick, "kind": kind, "label": label, "status": status}
    return {
        "success": True, "reason": "goal_reached", "ticks": 40, "recoveries": 1,
        "trajectory": {
            "alpha": [
                ev(1, "action", "MoveTo(robot=alpha,from=shelf_b,to=charge_bay)", "FAILURE"),
                ev(2, "action", "ClearPath(robot=alpha,edge_from=shelf_b,edge_to=charge_bay)"),
                ev(6, "action", "MoveTo(robot=alpha,from=shelf_b,to=charge_bay)"),
                ev(7, "condition", "IsAtLocation(robot=alpha,location=charge_bay)"),
                ev(8, "condition", "IsItemAt(item=crate_blue,location=charge_bay)"),
                ev(9, "action", "PickUp(robot=alpha,item=crate_blue,location=charge_bay)"),
                ev(13, "action", "MoveTo(robot=alpha,from=charge_bay,to=shelf_b)"),
                ev(15, "sync", "SignalReady(robot=alpha,item=crate_blue,to=beta)"),
                ev(16, "joint", "HandoverGive(giver=alpha,receiver=beta,item=crate_blue,location=shelf_b)"),
            ],
            "beta": [
                ev(5, "action", "MoveTo(robot=beta,from=shelf_a,to=shelf_b)"),
                ev(16, "sync", "WaitReady(robot=beta,item=crate_blue,from=alpha)"),
                ev(17, "joint", "HandoverTake(receiver=beta,giver=alpha,item=crate_blue,location=shelf_b)"),
                ev(22, "action", "MoveTo(robot=beta,from=shelf_b,to=shelf_a)"),
                ev(23, "condition", "IsCarrying(robot=beta,item=crate_blue)"),
                ev(24, "action", "PlaceDown(robot=beta,item=crate_blue,location=shelf_a)"),
            ],
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", nargs="?", help="trace JSON from runner_node trace_json")
    ap.add_argument("-o", "--output", default="trace_timeline.png")
    ap.add_argument("--show-conditions", action="store_true",
                    help="also plot IsAtLocation/IsItemAt/IsCarrying checks")
    ap.add_argument("--title", default=None)
    ap.add_argument("--demo", action="store_true",
                    help="ignore input; plot a built-in fake trace (self-test)")
    ap.add_argument("--demo-json", default=None,
                    help="with --demo: also write the fake trace JSON here")
    args = ap.parse_args(argv)

    if args.demo:
        data = demo_trace()
        traj, meta = data["trajectory"], data
        if args.demo_json:
            with open(args.demo_json, "w") as f:
                json.dump(data, f, indent=2)
    else:
        if not args.trace:
            ap.error("trace JSON path required (or use --demo)")
        traj, meta = load_trace(args.trace)

    if not traj:
        print("empty trace, nothing to plot", file=sys.stderr)
        return 1
    out = plot(traj, meta, args.output, args.show_conditions, args.title)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
