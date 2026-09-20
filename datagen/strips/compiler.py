"""Compile a grounded STRIPS plan (+ fault spec) into a BehaviorTree.CPP v4
multi-agent XML tree. Deterministic: the output is correct by construction
(and is verified by executor.py anyway).

XML conventions:
  <root BTCPP_format="4" main_tree_to_execute="MainTree">
  MainTree: Parallel over per-agent SubTrees.
  Joint actions become rendezvous node pairs (SignalReady/WaitReady,
  HandoverGive/HandoverTake, Co* leaves appear in both agent subtrees).
  Faults become Fallback recovery branches.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import deque
from xml.dom import minidom

from .core import GroundAction


# ------------------------------------------------------------ utilities ---

def bfs_path(edges: set, src: str, dst: str) -> list:
    """Shortest location path src->...->dst over directed (a, b) edge pairs."""
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
    prev = {src: None}
    dq = deque([src])
    while dq:
        u = dq.popleft()
        if u == dst:
            break
        for v in adj.get(u, []):
            if v not in prev:
                prev[v] = u
                dq.append(v)
    if dst not in prev:
        return []
    path = []
    u = dst
    while u is not None:
        path.append(u)
        u = prev[u]
    return path[::-1]


def _el(tag: str, attrs: dict = None) -> ET.Element:
    return ET.Element(tag, {k: str(v) for k, v in (attrs or {}).items()})


def _seq(children: list) -> ET.Element:
    e = _el("Sequence")
    for c in children:
        e.append(c)
    return e


def _fallback(children: list) -> ET.Element:
    e = _el("Fallback")
    for c in children:
        e.append(c)
    return e


# ------------------------------------------------------------- compiler ---

class BTCompiler:
    def __init__(self, task: dict):
        self.task = task
        self.domain = task["domain_obj"]
        self.edges = {f for f in self.domain.static_facts if f[0] == "connected"}
        self.can_reach = {(f[1], f[2]) for f in self.domain.static_facts if f[0] == "can_reach"}
        self.charge_locs = [f[1] for f in self.domain.static_facts if f[0] == "charge_station_at"]
        self.blocked_edges = {tuple(f["edge"]) for f in task["faults"] if f["type"] == "blocked_edge"}
        self.battery_robots = {f["robot"] for f in task["faults"] if f["type"] == "battery"}
        self.handover_fail = [f for f in task["faults"] if f["type"] == "handover_fail"]

    # -- leaf builders ----------------------------------------------------

    def _move(self, r, l1, l2) -> ET.Element:
        return _el("MoveTo", {"robot": r, "from": l1, "to": l2})

    def _clear(self, r, l1, l2) -> ET.Element:
        return _el("ClearPath", {"robot": r, "edge_from": l1, "edge_to": l2})

    def _recharge_route(self, r, from_loc, resume_edge) -> list:
        """Nodes: navigate from_loc -> nearest reachable charge station -> Recharge -> back.

        Routes only through locations the robot can reach (can_reach target rule).
        """
        allowed = {(a, b) for _, a, b in self.edges if (r, b) in self.can_reach}

        def path(src, dst):
            return bfs_path(allowed, src, dst)

        def station_of(loc):
            best, best_len = None, 1 << 30
            for s in self.charge_locs:
                p = path(loc, s)
                if p and len(p) < best_len:
                    best, best_len = s, len(p)
            return best

        station = station_of(from_loc)
        if station is None:
            return None
        nodes = []
        for a, b in zip(path(from_loc, station)[0:], path(from_loc, station)[1:]):
            nodes.append(self._move(r, a, b))
        nodes.append(_el("Recharge", {"robot": r, "station": station}))
        back = path(station, from_loc)
        for a, b in zip(back, back[1:]):
            nodes.append(self._move(r, a, b))
        return nodes

    def _move_with_recovery(self, r, l1, l2) -> ET.Element:
        move = self._move(r, l1, l2)
        children = [move]
        if (l1, l2) in self.blocked_edges or (l2, l1) in self.blocked_edges:
            children.append(_seq([self._clear(r, l1, l2), self._move(r, l1, l2)]))
        if r in self.battery_robots:
            route = self._recharge_route(r, l1, (l1, l2))
            if route:
                children.append(_seq(route + [self._move(r, l1, l2)]))
        if len(children) == 1:
            return move
        return _fallback(children)

    # -- per-action subtree ------------------------------------------------

    def compile_action(self, a: GroundAction, agent: str) -> ET.Element:
        b = a.binding
        n = a.name
        if n == "MoveTo":
            return self._move_with_recovery(b["r"], b["l1"], b["l2"])
        if n == "PickUp":
            return _seq([
                _el("IsAtLocation", {"robot": b["r"], "location": b["l"]}),
                _el("IsItemAt", {"item": b["i"], "location": b["l"]}),
                _el("PickUp", {"robot": b["r"], "item": b["i"], "location": b["l"]}),
            ])
        if n == "PlaceDown":
            return _seq([
                _el("IsCarrying", {"robot": b["r"], "item": b["i"]}),
                _el("PlaceDown", {"robot": b["r"], "item": b["i"], "location": b["l"]}),
            ])
        if n == "Handover":
            g, t, i, l = b["g"], b["t"], b["i"], b["l"]
            retry = any(f["item"] == i and f.get("giver") == g for f in self.handover_fail)
            if agent == g:
                leaf = _el("HandoverGive", {"giver": g, "receiver": t, "item": i, "location": l})
                body = [_el("IsAtLocation", {"robot": g, "location": l}),
                        _el("SignalReady", {"robot": g, "item": i, "to": t})]
                body.append(_fallback([leaf, _el("HandoverGive", {"giver": g, "receiver": t, "item": i, "location": l})]) if retry else leaf)
                return _seq(body)
            else:
                leaf = _el("HandoverTake", {"receiver": t, "giver": g, "item": i, "location": l})
                body = [_el("WaitReady", {"robot": t, "item": i, "from": g})]
                body.append(_fallback([leaf, _el("HandoverTake", {"receiver": t, "giver": g, "item": i, "location": l})]) if retry else leaf)
                return _seq(body)
        if n in ("CoPickUp", "CoPlaceDown"):
            r1, r2, i, l = b["r1"], b["r2"], b["i"], b["l"]
            return _seq([
                _el("IsAtLocation", {"robot": agent, "location": l}),
                _el(n, {"robot1": r1, "robot2": r2, "item": i, "location": l}),
            ])
        if n == "CoMoveTo":
            r1, r2, i, l1, l2 = b["r1"], b["r2"], b["i"], b["l1"], b["l2"]
            return _el(n, {"robot1": r1, "robot2": r2, "item": i, "from": l1, "to": l2})
        if n == "Recharge":
            return _seq([
                _el("IsAtLocation", {"robot": b["r"], "location": b["l"]}),
                _el("Recharge", {"robot": b["r"], "station": b["l"]}),
            ])
        if n == "ClearPath":
            return self._clear(b["r"], b["l1"], b["l2"])
        # domain-unique actions (CatalogBook / WaterPlants / InspectPlants)
        attrs = {"robot": b["r"]}
        if "i" in b:
            attrs["item"] = b["i"]
        if "l" in b:
            attrs["location"] = b["l"]
        return _el(n, attrs)

    # -- whole tree ---------------------------------------------------------

    def compile(self) -> str:
        agents = sorted({ag for a in self.task["plan"] for ag in a.agents})
        per_agent = {ag: [] for ag in agents}
        for a in self.task["plan"]:
            for ag in a.agents:
                per_agent[ag].append(self.compile_action(a, ag))

        root = _el("root", {"BTCPP_format": "4", "main_tree_to_execute": "MainTree"})
        main = _el("BehaviorTree", {"ID": "MainTree"})
        par = _el("Parallel", {"success_threshold": str(len(agents)), "failure_threshold": "1"})
        for ag in agents:
            par.append(_el("SubTree", {"ID": f"Agent_{ag}"}))
        main.append(par)
        root.append(main)
        for ag in agents:
            tree = _el("BehaviorTree", {"ID": f"Agent_{ag}"})
            tree.append(_seq(per_agent[ag]) if len(per_agent[ag]) > 1 else per_agent[ag][0])
            root.append(tree)

        xml = ET.tostring(root, encoding="unicode")
        return minidom.parseString(xml).toprettyxml(indent="    ")
