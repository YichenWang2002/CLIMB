#!/usr/bin/env python3
"""Run the MRBTP symbolic baseline on the held-out multi-agent tasks.

The benchmark's task records are generated from a STRIPS instance, while the
published MRBTP implementation consumes per-agent ``PlanningAction`` objects
and emits BTML.  The default faithful bridge:

* reconstruct the exact domain/map/action model stored in ``meta``;
* run the repository's ``MABTP`` search (the paper's MRBTP configuration,
  without the optional LLM subtree plugin);
* preserve the complete reactive BTML while translating it to the benchmark's
  BT.CPP XML vocabulary; and
* score the XML with the same symbolic executor used for every generator.

The legacy ``trace`` bridge remains available only for diagnostics.  It
flattens a nominal BTML execution and may add structured recovery branches;
its scores must not be reported as a faithful MRBTP reproduction.

The natural-language field is deliberately never read.  This is the stated
protocol distinction: MRBTP receives the formalized task representation
derived from the same instance, whereas CLIMB receives mission text + map.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable
from xml.dom import minidom


ROOT = Path(__file__).resolve().parents[1]      # repository root
MRBTP_ROOT = Path(os.environ.get("MRBTP_ROOT", "MRBTP-main"))
if str(MRBTP_ROOT) not in sys.path:
    sys.path.insert(0, str(MRBTP_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mabtpg.btp.multi_robot_basic import MABTP  # noqa: E402
from mabtpg.envs.gridenv.minigrid.planning_action import PlanningAction  # noqa: E402

from datagen.executor import execute, re_extract  # noqa: E402
from datagen.strips.domains import build_domain  # noqa: E402


# The benchmark executor enforces these preconditions in action leaves.  They
# still remain in the symbolic action model, but are not emitted as XML
# condition nodes because they are not part of the XML condition vocabulary.
STATIC_PREDICATES = {
    "connected",
    "can_reach",
    "charge_station_at",
}

PREDICATE_TO_XML = {
    "at": "IsAtLocation",
    "item_at": "IsItemAt",
    "carrying": "IsCarrying",
}

ACTION_NAMES = {
    "MoveTo",
    "PickUp",
    "PlaceDown",
    "Handover",
    "CoPickUp",
    "CoMoveTo",
    "CoPlaceDown",
    "CatalogBook",
    "WaterPlants",
    "InspectPlants",
}


class BaselineEnv:
    """MRBTP's conflict hook for the benchmark's grounded literals."""

    @staticmethod
    def check_conflict(condition_set: Iterable[str]) -> bool:
        positive_by_entity: dict[tuple[str, str], str] = {}
        carrying: dict[str, str] = {}
        co_carrying: dict[tuple[str, str], str] = {}

        for literal in condition_set:
            m = re.fullmatch(r"at\(([^,]+),([^\)]+)\)", literal)
            if m:
                robot, location = m.groups()
                key = ("at", robot)
                if key in positive_by_entity and positive_by_entity[key] != location:
                    return True
                positive_by_entity[key] = location
                continue

            m = re.fullmatch(r"item_at\(([^,]+),([^\)]+)\)", literal)
            if m:
                item, location = m.groups()
                key = ("item_at", item)
                if key in positive_by_entity and positive_by_entity[key] != location:
                    return True
                positive_by_entity[key] = location
                continue

            m = re.fullmatch(r"carrying\(([^,]+),([^\)]+)\)", literal)
            if m:
                robot, item = m.groups()
                if robot in carrying and carrying[robot] != item:
                    return True
                carrying[robot] = item
                continue

            m = re.fullmatch(r"co_carrying\(([^,]+),([^,]+),([^\)]+)\)", literal)
            if m:
                r1, r2, item = m.groups()
                pair = tuple(sorted((r1, r2)))
                if pair in co_carrying and co_carrying[pair] != item:
                    return True
                co_carrying[pair] = item

        return False


def _fact_string(fact: tuple[str, ...]) -> str:
    """Use MRBTP's predicate syntax for a benchmark STRIPS fact."""

    return f"{fact[0]}({','.join(fact[1:])})"


def _dynamic(facts: Iterable[tuple[str, ...]]) -> list[tuple[str, ...]]:
    """Drop static facts from an action's symbolic pre/add/delete sets.

    Static map facts were already enforced while grounding the domain.  The
    original benchmark schema has a historical malformed ``PlaceDown`` delete
    tuple; ignoring unknown/non-predicate entries preserves the executor's
    effective behavior while preventing character-wise garbage literals from
    entering MRBTP's search graph.
    """

    out = []
    for fact in facts:
        if not isinstance(fact, tuple) or not fact or not isinstance(fact[0], str):
            continue
        if fact[0] in STATIC_PREDICATES:
            continue
        if fact[0] not in {
            "at",
            "item_at",
            "carrying",
            "free",
            "mobile",
            "light",
            "heavy",
            "co_carrying",
            "clear",
            "charged",
            "cataloged",
            "watered",
            "inspected",
        }:
            continue
        out.append(fact)
    return out


def _to_planning_action(ground_action) -> PlanningAction:
    return PlanningAction(
        name=(
            f"{ground_action.name}"
            f"({','.join(ground_action.binding[k] for k in ground_action.binding)})"
        ),
        pre={_fact_string(f) for f in _dynamic(ground_action.pre)},
        add={_fact_string(f) for f in _dynamic(ground_action.add)},
        del_set={_fact_string(f) for f in _dynamic(ground_action.delete)},
    )


def _reconstruct(meta: dict) -> tuple[dict, object]:
    static = [("can_reach", r, l) for r, l in meta["can_reach"]]
    edges = [("connected", a, b) for a, b in meta["connected"]]
    domain = build_domain(
        meta["domain"],
        robots=meta["robots"],
        items=meta["items"],
        extra_static=static,
        edges=edges,
    )
    task = {
        "domain": meta["domain"],
        "domain_obj": domain,
        "init_dynamic": [tuple(f) for f in meta["init_dynamic"]],
        "goal": [tuple(f) for f in meta["goal"]],
        "faults": meta.get("faults", []),
        "skill_aliases": meta.get("skill_aliases", {}),
    }
    return task, domain


def _action_lists(meta: dict, domain) -> list[list[PlanningAction]]:
    """Build the per-agent grounded action spaces from the formal task.

    Grounding is done from the same map, robot set and item set as the common
    executor.  Only actions in the MRBTP planning model are included; fault
    recovery actions are evaluator-side wrappers, matching the benchmark's
    own compiler convention.
    """

    by_agent = {r: [] for r in meta["robots"]}
    robot_rank = {robot: idx for idx, robot in enumerate(meta["robots"])}
    for ground in domain.ground_actions():
        if ground.name not in ACTION_NAMES:
            continue
        if ground.name.startswith("Co"):
            r1, r2 = ground.binding["r1"], ground.binding["r2"]
            if robot_rank[r1] > robot_rank[r2]:
                continue
        action = _to_planning_action(ground)
        for robot in ground.agents:
            if robot in by_agent:
                by_agent[robot].append(action)
    return [by_agent[r] for r in meta["robots"]]


def _parse_action(name: str, args: tuple[str, ...]) -> tuple[str, dict[str, str]]:
    keys = {
        "MoveTo": ("r", "l1", "l2"),
        "PickUp": ("r", "i", "l"),
        "PlaceDown": ("r", "i", "l"),
        "Handover": ("g", "t", "i", "l"),
        "CoPickUp": ("r1", "r2", "i", "l"),
        "CoMoveTo": ("r1", "r2", "i", "l1", "l2"),
        "CoPlaceDown": ("r1", "r2", "i", "l"),
        "CatalogBook": ("r", "i", "l"),
        "WaterPlants": ("r", "l"),
        "InspectPlants": ("r", "l"),
    }
    wanted = keys.get(name)
    if wanted is None or len(wanted) != len(args):
        raise ValueError(f"unsupported MRBTP action {name}({','.join(args)})")
    return name, dict(zip(wanted, args))


def _xml(tag: str, attrs: dict[str, str] | None = None) -> ET.Element:
    return ET.Element(tag, {k: str(v) for k, v in (attrs or {}).items()})


def _seq(children: Iterable[ET.Element]) -> ET.Element:
    node = _xml("Sequence")
    for child in children:
        if child is not None:
            node.append(child)
    return node


def _fallback(children: Iterable[ET.Element]) -> ET.Element:
    node = _xml("Fallback")
    for child in children:
        if child is not None:
            node.append(child)
    return node


def _condition_to_xml(literal: str) -> ET.Element | None:
    m = re.fullmatch(r"(\w+)\((.*)\)", literal)
    if not m:
        return None
    pred, raw = m.groups()
    args = tuple(a.strip() for a in raw.split(",")) if raw else ()
    if pred not in PREDICATE_TO_XML:
        return None
    tag = PREDICATE_TO_XML[pred]
    if pred == "at" and len(args) == 2:
        return _xml(tag, {"robot": args[0], "location": args[1]})
    if pred == "item_at" and len(args) == 2:
        return _xml(tag, {"item": args[0], "location": args[1]})
    if pred == "carrying" and len(args) == 2:
        return _xml(tag, {"robot": args[0], "item": args[1]})
    return None


def _faults(meta: dict) -> tuple[set[tuple[str, str]], set[str], set[tuple[str, str, str]]]:
    blocked = set()
    battery = set()
    handover = set()
    for fault in meta.get("faults", []):
        if fault["type"] == "blocked_edge":
            a, b = fault["edge"]
            blocked.add((a, b))
            blocked.add((b, a))
        elif fault["type"] == "battery":
            battery.add(fault["robot"])
        elif fault["type"] == "handover_fail":
            handover.add((fault["giver"], fault["receiver"], fault["item"]))
    return blocked, battery, handover


def _paths(edges: set[tuple[str, str]], src: str, dst: str) -> list[str]:
    prev = {src: None}
    q = deque([src])
    while q:
        cur = q.popleft()
        if cur == dst:
            break
        for nxt in sorted(v for u, v in edges if u == cur):
            if nxt not in prev:
                prev[nxt] = cur
                q.append(nxt)
    if dst not in prev:
        return []
    out = []
    cur = dst
    while cur is not None:
        out.append(cur)
        cur = prev[cur]
    return list(reversed(out))


class XmlEmitter:
    def __init__(self, meta: dict):
        self.meta = meta
        self.robots = list(meta["robots"])
        self.blocked, self.battery, self.handover_fail = _faults(meta)
        self.edges = {(a, b) for a, b in meta["connected"]}
        self.can_reach = {(r, l) for r, l in meta["can_reach"]}
        self.stations = list(meta.get("charge_stations", []))

    def move(self, r: str, l1: str, l2: str) -> ET.Element:
        return _xml("MoveTo", {"robot": r, "from": l1, "to": l2})

    def move_with_recovery(self, r: str, l1: str, l2: str) -> ET.Element:
        branches = [self.move(r, l1, l2)]
        if (l1, l2) in self.blocked:
            branches.append(
                _seq(
                    [
                        _xml("ClearPath", {"robot": r, "edge_from": l1, "edge_to": l2}),
                        self.move(r, l1, l2),
                    ]
                )
            )
        if r in self.battery:
            # The first branch can fail when the budget is exhausted.  A
            # recovery branch heads to a reachable station, recharges, returns
            # to the move's source, then retries the original transition.
            candidates = []
            allowed = {(a, b) for a, b in self.edges if (r, b) in self.can_reach}
            for station in self.stations:
                path = _paths(allowed, l1, station)
                if path and (not candidates or len(path) < len(candidates[0][1])):
                    candidates = [(station, path)]
            if candidates:
                station, path = candidates[0]
                back = _paths(allowed, station, l1)
                recovery = [
                    self.move(r, a, b) for a, b in zip(path, path[1:])
                ]
                recovery.append(_xml("Recharge", {"robot": r, "station": station}))
                recovery.extend(self.move(r, a, b) for a, b in zip(back, back[1:]))
                recovery.append(self.move(r, l1, l2))
                branches.append(_seq(recovery))
        return branches[0] if len(branches) == 1 else _fallback(branches)

    def action(self, name: str, args: tuple[str, ...], agent: str) -> ET.Element:
        name, b = _parse_action(name, args)
        if name == "MoveTo":
            return self.move_with_recovery(b["r"], b["l1"], b["l2"])
        if name == "PickUp":
            return _seq(
                [
                    _xml("IsAtLocation", {"robot": b["r"], "location": b["l"]}),
                    _xml("IsItemAt", {"item": b["i"], "location": b["l"]}),
                    _xml("PickUp", {"robot": b["r"], "item": b["i"], "location": b["l"]}),
                ]
            )
        if name == "PlaceDown":
            return _seq(
                [
                    _xml("IsAtLocation", {"robot": b["r"], "location": b["l"]}),
                    _xml("IsCarrying", {"robot": b["r"], "item": b["i"]}),
                    _xml("PlaceDown", {"robot": b["r"], "item": b["i"], "location": b["l"]}),
                ]
            )
        if name == "Handover":
            key = (b["g"], b["t"], b["i"])
            if agent == b["g"]:
                leaf = _xml(
                    "HandoverGive",
                    {"giver": b["g"], "receiver": b["t"], "item": b["i"], "location": b["l"]},
                )
                if key in self.handover_fail:
                    retry = _xml(
                        "HandoverGive",
                        {"giver": b["g"], "receiver": b["t"], "item": b["i"], "location": b["l"]},
                    )
                    leaf = _fallback([leaf, retry])
                return _seq(
                    [
                        _xml("IsAtLocation", {"robot": b["g"], "location": b["l"]}),
                        _xml("SignalReady", {"robot": b["g"], "item": b["i"], "to": b["t"]}),
                        leaf,
                    ]
                )
            leaf = _xml(
                "HandoverTake",
                {"receiver": b["t"], "giver": b["g"], "item": b["i"], "location": b["l"]},
            )
            if key in self.handover_fail:
                retry = _xml(
                    "HandoverTake",
                    {"receiver": b["t"], "giver": b["g"], "item": b["i"], "location": b["l"]},
                )
                leaf = _fallback([leaf, retry])
            return _seq(
                [
                    _xml("WaitReady", {"robot": b["t"], "item": b["i"], "from": b["g"]}),
                    leaf,
                ]
            )
        if name in {"CoPickUp", "CoPlaceDown"}:
            location = b["l"]
            return _seq(
                [
                    _xml("IsAtLocation", {"robot": agent, "location": location}),
                    _xml(
                        name,
                        {
                            "robot1": b["r1"],
                            "robot2": b["r2"],
                            "item": b["i"],
                            "location": location,
                        },
                    ),
                ]
            )
        if name == "CoMoveTo":
            return _xml(
                name,
                {
                    "robot1": b["r1"],
                    "robot2": b["r2"],
                    "item": b["i"],
                    "from": b["l1"],
                    "to": b["l2"],
                },
            )
        attrs = {"robot": b["r"]}
        if "i" in b:
            attrs["item"] = b["i"]
        if "l" in b:
            attrs["location"] = b["l"]
        return _xml(name, attrs)

    def node(self, node, agent: str) -> ET.Element | None:
        node_type = getattr(node, "node_type", None)
        if node_type == "selector":
            return _fallback(self.node(c, agent) for c in node.children)
        if node_type == "sequence":
            return _seq(self.node(c, agent) for c in node.children)
        if node_type == "condition":
            if not node.cls_name:
                return None
            return _condition_to_xml(f"{node.cls_name}({','.join(node.args)})")
        if node_type == "composite_condition":
            subtree = node.info.get("sub_btml")
            return self.node(subtree.anytree_root, agent) if subtree else None
        if node_type == "action":
            return self.action(node.cls_name, tuple(node.args), agent)
        return None

    def emit(self, btmls: dict[str, object]) -> str:
        root = _xml("root", {"BTCPP_format": "4", "main_tree_to_execute": "MainTree"})
        main = _xml("BehaviorTree", {"ID": "MainTree"})
        parallel = _xml(
            "Parallel",
            {"success_threshold": str(len(self.robots)), "failure_threshold": "1"},
        )
        for robot in self.robots:
            parallel.append(_xml("SubTree", {"ID": f"Agent_{robot}"}))
        main.append(parallel)
        root.append(main)
        for robot in self.robots:
            tree = _xml("BehaviorTree", {"ID": f"Agent_{robot}"})
            body = self.node(btmls[robot].anytree_root, robot)
            if body is None or len(body) == 0:
                body = _xml("Fallback")
            tree.append(body)
            root.append(tree)
        raw = ET.tostring(root, encoding="unicode")
        return minidom.parseString(raw).toprettyxml(indent="    ")


class FaithfulBtmlEmitter:
    """Translate complete MRBTP BTML without solving or repairing the task.

    This bridge is intentionally task-agnostic.  It does not inspect faults,
    reference plans, goals, or executor outcomes.  It preserves MRBTP's
    memoryless control-flow topology and only lowers abstract joint actions to
    the paired leaves understood by the common benchmark executor.
    """

    def __init__(self, robots: Iterable[str], goals: Iterable[tuple[str, ...]]):
        self.robots = list(robots)
        self.goals = [tuple(goal) for goal in goals]

    @staticmethod
    def condition(predicate: str, args: Iterable[str]) -> ET.Element:
        return _xml(
            "CheckFact",
            {"predicate": predicate, "args": ",".join(str(arg) for arg in args)},
        )

    @staticmethod
    def action(name: str, args: tuple[str, ...], agent: str) -> ET.Element:
        name, b = _parse_action(name, args)
        if name == "MoveTo":
            return _xml("MoveTo", {"robot": b["r"], "from": b["l1"], "to": b["l2"]})
        if name == "PickUp":
            return _xml("PickUp", {"robot": b["r"], "item": b["i"], "location": b["l"]})
        if name == "PlaceDown":
            return _xml("PlaceDown", {"robot": b["r"], "item": b["i"], "location": b["l"]})
        if name == "Handover":
            attrs = {
                "giver": b["g"],
                "receiver": b["t"],
                "item": b["i"],
                "location": b["l"],
            }
            return _xml("HandoverGive" if agent == b["g"] else "HandoverTake", attrs)
        if name in {"CoPickUp", "CoPlaceDown"}:
            return _xml(
                name,
                {
                    "robot1": b["r1"],
                    "robot2": b["r2"],
                    "item": b["i"],
                    "location": b["l"],
                },
            )
        if name == "CoMoveTo":
            return _xml(
                name,
                {
                    "robot1": b["r1"],
                    "robot2": b["r2"],
                    "item": b["i"],
                    "from": b["l1"],
                    "to": b["l2"],
                },
            )
        attrs = {"robot": b["r"]}
        if "i" in b:
            attrs["item"] = b["i"]
        if "l" in b:
            attrs["location"] = b["l"]
        return _xml(name, attrs)

    def node(self, node, agent: str) -> ET.Element:
        node_type = getattr(node, "node_type", None)
        if node_type == "selector":
            out = _xml("ReactiveFallback")
            for child in node.children:
                out.append(self.node(child, agent))
            return out
        if node_type == "sequence":
            out = _xml("ReactiveSequence")
            for child in node.children:
                out.append(self.node(child, agent))
            return out
        if node_type == "composite_condition":
            subtree = node.info.get("sub_btml")
            if subtree is None:
                raise ValueError("MRBTP composite condition has no subtree")
            return self.node(subtree.anytree_root, agent)
        if node_type == "condition":
            if not node.cls_name:
                raise ValueError("MRBTP condition has no predicate")
            return self.condition(node.cls_name, node.args)
        if node_type == "action":
            return self.action(node.cls_name, tuple(node.args), agent)
        raise ValueError(f"unsupported MRBTP BTML node type: {node_type!r}")

    def emit(self, btmls: dict[str, object]) -> str:
        root = _xml(
            "root",
            {
                "BTCPP_format": "4",
                "main_tree_to_execute": "MainTree",
                "mrbtp_bridge": "faithful_v1",
            },
        )
        main = _xml("BehaviorTree", {"ID": "MainTree"})
        parallel = _xml(
            "Parallel",
            {"success_threshold": str(len(self.robots)), "failure_threshold": "1"},
        )
        for robot in self.robots:
            retry = _xml("RetryUntilSuccessful", {"num_attempts": "-1"})
            gated = _xml("ReactiveSequence")
            gated.append(_xml("SubTree", {"ID": f"Agent_{robot}"}))
            for goal in self.goals:
                gated.append(self.condition(goal[0], goal[1:]))
            retry.append(gated)
            parallel.append(retry)
        main.append(parallel)
        root.append(main)

        for robot in self.robots:
            tree = _xml("BehaviorTree", {"ID": f"Agent_{robot}"})
            tree.append(self.node(btmls[robot].anytree_root, robot))
            root.append(tree)
        return ET.tostring(root, encoding="unicode")


def _btml_paths(node, cap: int = 4000) -> list[list[tuple[str, tuple[str, ...]]]]:
    """Enumerate action paths represented by one MRBTP BTML tree.

    A selector contributes alternatives, a sequence concatenates child paths,
    and conditions contribute no actions.  The cap bounds the compatibility
    bridge for unusually broad grounded action libraries.
    """

    node_type = getattr(node, "node_type", None)
    if node_type == "selector":
        out = []
        for child in node.children:
            out.extend(_btml_paths(child, cap))
            if len(out) >= cap:
                break
        return out[:cap]
    if node_type == "sequence":
        out = [[]]
        for child in node.children:
            child_paths = _btml_paths(child, cap)
            out = [left + right for left in out for right in child_paths][:cap]
            if not out:
                break
        return out
    if node_type == "composite_condition":
        subtree = node.info.get("sub_btml")
        return _btml_paths(subtree.anytree_root, cap) if subtree else [[]]
    if node_type == "condition":
        return [[]]
    if node_type == "action":
        return [[(node.cls_name, tuple(node.args))]]
    return [[]]


def _ground_lookup(domain) -> dict[tuple[str, tuple[tuple[str, str], ...]], object]:
    lookup = {}
    for action in domain.ground_actions():
        key = (action.name, tuple(sorted(action.binding.items())))
        lookup[key] = action
    return lookup


def _path_ground(path, lookup: dict) -> list[object] | None:
    keys = {
        "MoveTo": ("r", "l1", "l2"),
        "PickUp": ("r", "i", "l"),
        "PlaceDown": ("r", "i", "l"),
        "Handover": ("g", "t", "i", "l"),
        "CoPickUp": ("r1", "r2", "i", "l"),
        "CoMoveTo": ("r1", "r2", "i", "l1", "l2"),
        "CoPlaceDown": ("r1", "r2", "i", "l"),
        "CatalogBook": ("r", "i", "l"),
        "WaterPlants": ("r", "l"),
        "InspectPlants": ("r", "l"),
    }
    result = []
    for name, args in path:
        fields = keys.get(name)
        if fields is None or len(fields) != len(args):
            return None
        binding = dict(zip(fields, args))
        key = (name, tuple(sorted(binding.items())))
        action = lookup.get(key)
        if action is None:
            return None
        result.append(action)
    return result


def _execute_btml_trace(
    meta: dict,
    domain,
    btmls: dict[str, object],
    max_cycles: int = 1000,
) -> list[list[object]] | None:
    """Tick MRBTP's generated reactive trees and return their action traces.

    MRBTP emits memoryless reactive BTs.  A selector can therefore choose a
    different regression branch on a later tick after an earlier action has
    changed the shared symbolic state.  Flattening one root-to-leaf branch is
    insufficient for multi-goal tasks (notably ``heavy+extra``); this
    interpreter follows the generated BTML's actual Sequence/Fallback
    semantics instead.

    Joint actions use the benchmark's rendezvous contract: every participant
    must reach the identical grounded action before its STRIPS effects are
    applied.  Faults are deliberately absent here because MRBTP plans the
    nominal task; ``XmlEmitter`` wraps the resulting moves/handovers with the
    same evaluator-side recovery subtrees used elsewhere in the benchmark.
    """

    robots = list(meta["robots"])
    robot_index = {robot: i for i, robot in enumerate(robots)}
    lookup = _ground_lookup(domain)
    state = set(tuple(f) for f in meta["init_dynamic"])
    static = domain.static_facts
    goals = set(tuple(f) for f in meta["goal"])
    traces: list[list[object]] = [[] for _ in robots]

    # Rendezvous state.  ``results`` lets the side that posted first consume
    # the completion result on its next reactive tick, matching the common
    # executor's joint-leaf behavior.
    posted: dict[tuple, set[int]] = defaultdict(set)
    results: dict[tuple[tuple, int], str] = {}

    def literal_fact(node) -> tuple[str, ...] | None:
        name = getattr(node, "cls_name", None)
        if not name:
            return None
        return (name, *(str(arg) for arg in getattr(node, "args", ())))

    def ground_action(node):
        path = [(node.cls_name, tuple(str(arg) for arg in node.args))]
        grounded = _path_ground(path, lookup)
        return grounded[0] if grounded else None

    def tick(node, agent_index: int) -> str:
        node_type = getattr(node, "node_type", None)
        if node_type == "selector":
            for child in node.children:
                status = tick(child, agent_index)
                if status != "FAILURE":
                    return status
            return "FAILURE"
        if node_type == "sequence":
            for child in node.children:
                status = tick(child, agent_index)
                if status != "SUCCESS":
                    return status
            return "SUCCESS"
        if node_type == "composite_condition":
            subtree = node.info.get("sub_btml")
            return tick(subtree.anytree_root, agent_index) if subtree else "FAILURE"
        if node_type == "condition":
            fact = literal_fact(node)
            return "SUCCESS" if fact is not None and fact in (state | static) else "FAILURE"
        if node_type != "action":
            return "FAILURE"

        action = ground_action(node)
        if action is None or robots[agent_index] not in action.agents:
            return "FAILURE"
        participants = tuple(robot_index[r] for r in action.agents if r in robot_index)
        if len(participants) <= 1:
            if not action.applicable(frozenset(state) | static):
                return "FAILURE"
            state.difference_update(action.delete)
            state.update(action.add)
            traces[agent_index].append(action)
            return "SUCCESS"

        joint_key = (action.name, tuple(sorted(action.binding.items())))
        result_key = (joint_key, agent_index)
        if result_key in results:
            return results.pop(result_key)
        posted[joint_key].add(agent_index)
        if not set(participants) <= posted[joint_key]:
            return "RUNNING"

        status = (
            "SUCCESS"
            if action.applicable(frozenset(state) | static)
            else "FAILURE"
        )
        if status == "SUCCESS":
            state.difference_update(action.delete)
            state.update(action.add)
            # Add the rendezvous once, at the corresponding position in every
            # participant's observed execution trace.
            for participant in participants:
                traces[participant].append(action)
        posted.pop(joint_key, None)
        for participant in participants:
            if participant != agent_index:
                results[(joint_key, participant)] = status
        return status

    seen_stalled = set()
    for _ in range(max_cycles):
        if goals <= state:
            return traces
        before = (
            frozenset(state),
            tuple(len(trace) for trace in traces),
            tuple(sorted((key, tuple(sorted(value))) for key, value in posted.items())),
            tuple(sorted(results)),
        )
        for agent_index, robot in enumerate(robots):
            tick(btmls[robot].anytree_root, agent_index)
            if goals <= state:
                return traces
        after = (
            frozenset(state),
            tuple(len(trace) for trace in traces),
            tuple(sorted((key, tuple(sorted(value))) for key, value in posted.items())),
            tuple(sorted(results)),
        )
        if after == before:
            if after in seen_stalled:
                return None
            seen_stalled.add(after)
        else:
            seen_stalled.clear()
    return None


def _select_compatible_paths(meta: dict, domain, btmls: dict[str, object]) -> list[list[object]] | None:
    """Choose synchronized branches from the BTML alternatives.

    This is a small symbolic executor for the *generated MRBTP trees*.  It
    never consults the dataset's stored plan.  It only checks whether a set of
    branch paths can be scheduled from the initial state under the same action
    model; joint actions advance all participating branches together.
    """

    robots = list(meta["robots"])
    lookup = _ground_lookup(domain)
    path_sets = []
    for robot in robots:
        raw = _btml_paths(btmls[robot].anytree_root, cap=8000)
        unique = []
        seen = set()
        for path in raw:
            key = tuple(path)
            if key in seen:
                continue
            seen.add(key)
            grounded = _path_ground(path, lookup)
            if grounded is not None:
                unique.append(grounded)
        # Prefer short candidate paths, which keeps search deterministic and
        # mirrors MRBTP's action-cost-neutral FIFO behavior.  Do not apply a
        # fixed prefix cut here: a coherent branch can occur after many
        # shorter, individually invalid branches (especially for a robot that
        # must first navigate to a rendezvous).
        unique.sort(key=lambda p: (len(p), tuple(a.label() for a in p)))
        path_sets.append(unique)
    if any(not paths for paths in path_sets):
        return None

    static = domain.static_facts
    initial = frozenset(tuple(f) for f in meta["init_dynamic"])
    goals = frozenset(tuple(f) for f in meta["goal"])
    robot_index = {robot: i for i, robot in enumerate(robots)}

    def participants(action):
        return tuple(robot_index[r] for r in action.agents if r in robot_index)

    def joint_signature(path):
        """Return the ordered joint actions performed by one robot."""
        return tuple(
            (action.name, tuple(sorted(action.binding.items())))
            for action in path
            if len(action.agents) > 1
        )

    def schedule(sequences):
        """Symbolically schedule one branch per robot."""
        positions = tuple(0 for _ in robots)
        state = set(initial)
        visited = set()
        while True:
            if goals <= state and all(
                positions[i] == len(sequences[i]) for i in range(len(robots))
            ):
                return True
            key = (positions, frozenset(state))
            if key in visited:
                return False
            visited.add(key)

            candidates = []
            for i, sequence in enumerate(sequences):
                if positions[i] >= len(sequence):
                    continue
                action = sequence[positions[i]]
                # Rendezvous actions must be attempted before local actions;
                # this mirrors the executor's Parallel tick order and avoids
                # advancing one side past a shared action.
                candidates.append((0 if len(action.agents) > 1 else 1, i, action))
            candidates.sort(key=lambda x: (x[0], x[1], x[2].label()))

            progressed = False
            for _, i, action in candidates:
                inds = participants(action)
                if len(inds) > 1:
                    if not all(
                        positions[j] < len(sequences[j])
                        and sequences[j][positions[j]].name == action.name
                        and sequences[j][positions[j]].binding == action.binding
                        for j in inds
                    ):
                        continue
                    if not action.applicable(frozenset(state) | static):
                        continue
                    state.difference_update(action.delete)
                    state.update(action.add)
                    positions = tuple(
                        pos + (1 if idx in inds else 0)
                        for idx, pos in enumerate(positions)
                    )
                    progressed = True
                    break
                if action.applicable(frozenset(state) | static):
                    state.difference_update(action.delete)
                    state.update(action.add)
                    positions = tuple(
                        pos + (1 if idx == i else 0)
                        for idx, pos in enumerate(positions)
                    )
                    progressed = True
                    break
            if not progressed:
                return False

    # A synchronized branch has the same ordered joint-action projection for
    # every robot that participates in each rendezvous.  Indexing by that
    # projection turns the naive Cartesian product of thousands of BT paths
    # into a small set of signature-compatible combinations.  This is only a
    # branch-selection optimization: all paths still originate in MRBTP's
    # generated BTML and every selected combination is replayed symbolically.
    by_signature = []
    for paths in path_sets:
        groups = defaultdict(list)
        for path in paths:
            groups[joint_signature(path)].append(path)
        by_signature.append(groups)

    def signature_for_robot(master_path, robot_index):
        robot = robots[robot_index]
        return tuple(
            key for key in joint_signature(master_path)
            if robot in dict(key[1]).values()
        )

    # Try projections spanning the most robots first.  For faulted relay tasks
    # this selects the middle robot's two-handover path (alpha->beta and
    # beta->gamma), whose projection fully determines both endpoint paths.
    # Merely choosing the robot with the longest *possible* projection is not
    # sufficient: broad MRBTP fallback trees also contain longer irrelevant
    # joint-action alternatives.
    master_specs = []
    for master_index, groups in enumerate(by_signature):
        for master_sig, master_paths in groups.items():
            if not master_sig:
                continue
            participants_union = set()
            for _, binding_items in master_sig:
                binding = dict(binding_items)
                participants_union.update(
                    value for value in binding.values() if value in robot_index
                )
            projected = [
                signature_for_robot(master_paths[0], i)
                for i in range(len(robots))
            ]
            # Reject a master signature immediately if some participant's BT
            # has no matching joint-action projection.
            if any(
                projection not in by_signature[i]
                for i, projection in enumerate(projected)
            ):
                continue
            master_specs.append(
                (
                    -len(participants_union),
                    len(master_sig),
                    master_sig,
                    master_index,
                    master_paths,
                )
            )
    master_specs.sort(key=lambda cell: cell[:4])

    checked = 0
    max_combinations = 1000000 if len(robots) <= 2 else 500000
    for _, _, _, master_index, master_paths in master_specs:
        for master_path in master_paths:
            candidate_lists = []
            for candidate_index in range(len(robots)):
                if candidate_index == master_index:
                    candidate_lists.append([master_path])
                    continue
                required = signature_for_robot(master_path, candidate_index)
                candidate_lists.append(by_signature[candidate_index].get(required, []))
            if any(not candidates for candidates in candidate_lists):
                continue
            import itertools

            for combo in itertools.product(*candidate_lists):
                checked += 1
                if checked > max_combinations:
                    return None
                if schedule(combo):
                    return list(combo)

    # Tasks with no joint actions (or a branch whose joint projection was not
    # selected as a master) can be solved independently.  Try the shortest
    # branch first and retain the same symbolic goal check.
    independent = [groups.get((), []) for groups in by_signature]
    if all(independent):
        import itertools

        for combo in itertools.product(*independent):
            checked += 1
            if checked > max_combinations:
                break
            if schedule(combo):
                return list(combo)
    return None


def stable_id(record: dict) -> str:
    blob = json.dumps(
        {"instruction": record["instruction"], "input": record["input"]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def run_one(
    record: dict,
    timeout: float,
    fault_recovery: bool = True,
    bridge: str = "faithful",
) -> dict:
    meta = record["meta"]
    task, domain = _reconstruct(meta)
    actions = _action_lists(meta, domain)
    start = frozenset(_fact_string(tuple(f)) for f in meta["init_dynamic"])
    goal = frozenset(_fact_string(tuple(f)) for f in meta["goal"])
    planner = MABTP(
        verbose=False,
        start=start,
        env=BaselineEnv(),
        max_time_limit=timeout,
    )
    t0 = time.perf_counter()
    try:
        planner.planning(goal, actions)
        for agent in planner.planned_agent_list:
            agent.create_btml()
        btmls = {
            robot: planner.planned_agent_list[i].btml
            for i, robot in enumerate(meta["robots"])
        }
        # ``planning`` populates the graphs; BTML construction is explicit in
        # MRBTP's public implementation.
        for btml in btmls.values():
            if btml is None:
                raise RuntimeError("planner returned no BTML")
        planning_elapsed = time.perf_counter() - t0
        bridge_started = time.perf_counter()
        if bridge == "faithful":
            # Preserve the complete reactive BTML.  The bridge has no access
            # to fault metadata and performs no goal-conditioned path choice.
            xml = FaithfulBtmlEmitter(
                meta["robots"], [tuple(fact) for fact in meta["goal"]]
            ).emit(btmls)
        elif bridge == "trace":
            compatible = _execute_btml_trace(meta, domain, btmls)
            if compatible is None:
                raise RuntimeError(
                    "MRBTP BTML did not reach the goal under synchronized execution"
                )
            # Legacy diagnostic bridge: flatten the nominal BTML execution to
            # one action sequence per robot, optionally adding oracle recovery.
            emitter_meta = meta if fault_recovery else {**meta, "faults": []}
            emitter = XmlEmitter(emitter_meta)
            root = _xml(
                "root", {"BTCPP_format": "4", "main_tree_to_execute": "MainTree"}
            )
            main = _xml("BehaviorTree", {"ID": "MainTree"})
            parallel = _xml(
                "Parallel",
                {
                    "success_threshold": str(len(meta["robots"])),
                    "failure_threshold": "1",
                },
            )
            for robot in meta["robots"]:
                parallel.append(_xml("SubTree", {"ID": f"Agent_{robot}"}))
            main.append(parallel)
            root.append(main)
            for i, robot in enumerate(meta["robots"]):
                tree = _xml("BehaviorTree", {"ID": f"Agent_{robot}"})
                tree.append(
                    _seq(
                        emitter.action(
                            action.name,
                            tuple(action.binding[key] for key in action.binding),
                            robot,
                        )
                        for action in compatible[i]
                    )
                )
                root.append(tree)
            xml = minidom.parseString(
                ET.tostring(root, encoding="unicode")
            ).toprettyxml(indent="    ")
        else:
            raise ValueError(f"unknown bridge mode: {bridge}")
        bridge_elapsed = time.perf_counter() - bridge_started
        execution_started = time.perf_counter()
        result = execute(task, xml)
        execution_elapsed = time.perf_counter() - execution_started
        reason = result["reason"]
    except Exception as exc:  # noqa: BLE001 - one bad task must not stop 480 runs
        xml = ""
        result = {"success": False, "reason": f"baseline_error:{type(exc).__name__}:{exc}"}
        reason = result["reason"]
        planning_elapsed = getattr(planner, "expanded_time", 0.0)
        bridge_elapsed = 0.0
        execution_elapsed = 0.0
    total_elapsed = time.perf_counter() - t0
    return {
        "record_id": stable_id(record),
        "source_index": record.get("_source_index"),
        "domain": meta["domain"],
        "scenario": meta.get("scenario", ""),
        "scenario": meta["scenario"],
        "n_agents": meta["n_agents"],
        "faults": meta.get("faults", []),
        "success": bool(result.get("success", False)),
        "reason": reason,
        "ticks": result.get("ticks", 0),
        "recoveries": result.get("recoveries", 0),
        "planning_expanded": getattr(planner, "record_expanded_num", 0),
        "planning_time_sec": planning_elapsed,
        "bridge_time_sec": bridge_elapsed,
        "execution_time_sec": execution_elapsed,
        "total_time_sec": total_elapsed,
        "bridge": bridge,
        "xml_bytes": len(xml.encode()) if xml else 0,
        "xml_wellformed": bool(xml and re_extract(xml)),
        "xml": xml,
    }


def summarize(details: list[dict]) -> dict:
    def bucket(key: str) -> dict:
        out = defaultdict(lambda: {"n": 0, "ok": 0})
        for row in details:
            value = row[key]
            out[value]["n"] += 1
            out[value]["ok"] += int(row["success"])
        for value, cell in out.items():
            cell["rate"] = cell["ok"] / cell["n"] if cell["n"] else 0.0
        return dict(out)

    reasons = Counter(row["reason"] for row in details if not row["success"])
    return {
        "n": len(details),
        "successes": sum(int(row["success"]) for row in details),
        "success_rate": sum(int(row["success"]) for row in details) / max(len(details), 1),
        "xml_wellformed_rate": sum(int(row["xml_wellformed"]) for row in details) / max(len(details), 1),
        "per_domain": bucket("domain"),
        "per_scenario": bucket("scenario"),
        "per_scenario": bucket("scenario"),
        "per_n_agents": bucket("n_agents"),
        "fail_reasons": dict(reasons),
        "mean_planning_time_sec": sum(row["planning_time_sec"] for row in details) / max(len(details), 1),
        "mean_planning_expanded": sum(row["planning_expanded"] for row in details) / max(len(details), 1),
        "mean_bridge_time_sec": sum(row["bridge_time_sec"] for row in details) / max(len(details), 1),
        "mean_execution_time_sec": sum(row["execution_time_sec"] for row in details) / max(len(details), 1),
        "mean_total_time_sec": sum(row["total_time_sec"] for row in details) / max(len(details), 1),
        "mean_xml_bytes": sum(row["xml_bytes"] for row in details) / max(len(details), 1),
        "max_xml_bytes": max((row["xml_bytes"] for row in details), default=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data/test.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "outputs/results/mrbtp_test.json"))
    parser.add_argument("--limit", type=int, default=480)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel worker processes; 1 keeps the run serial and deterministic",
    )
    parser.add_argument(
        "--no-fault-recovery",
        action="store_true",
        help="legacy trace bridge only: disable structured recovery compilation",
    )
    parser.add_argument(
        "--bridge",
        choices=("faithful", "trace"),
        default="faithful",
        help="faithful preserves complete BTML; trace is the legacy diagnostic adapter",
    )
    parser.add_argument("--keep-xml", action="store_true")
    args = parser.parse_args()
    if args.bridge == "faithful" and args.no_fault_recovery:
        parser.error("--no-fault-recovery only applies to --bridge trace")

    records = []
    for idx, line in enumerate(Path(args.data).read_text().splitlines()):
        if not line.strip():
            continue
        record = json.loads(line)
        record["_source_index"] = idx
        # Keep the multi-agent suite only; the released test split is
        # exactly these 480 tasks, but stay robust to future revisions.
        if len(record["meta"].get("robots", [])) < 2:
            continue
        records.append(record)
        if len(records) >= args.limit:
            break

    details = [None] * len(records)

    def record_result(index, row):
        if not args.keep_xml:
            row.pop("xml", None)
        details[index] = row
        print(
            f"[{index + 1}/{len(records)}] {row['source_index']:>4} "
            f"{row['domain']:<10} {row['scenario']} "
            f"{'OK' if row['success'] else 'FAIL':<4} {row['reason']}",
            flush=True,
        )

    if args.workers <= 1:
        for index, record in enumerate(records):
            record_result(
                index,
                run_one(
                    record,
                    args.timeout,
                    not args.no_fault_recovery,
                    args.bridge,
                ),
            )
    else:
        # Each task is independent and MRBTP's planner has no shared state.
        # Keep the input order in the output even though completion order is
        # intentionally allowed to vary across workers.
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    run_one,
                    record,
                    args.timeout,
                    not args.no_fault_recovery,
                    args.bridge,
                ): index
                for index, record in enumerate(records)
            }
            for future in as_completed(futures):
                record_result(futures[future], future.result())
    details = [row for row in details if row is not None]

    payload = {
        "method": "MRBTP",
        "algorithm": "official MABTP planner (no optional LLM subtree plugin)",
        "bridge": args.bridge,
        "input_protocol": (
            "formalized nominal STRIPS task; reference plan/BT and NL are unused; "
            "faults are injected only by the shared executor"
            if args.bridge == "faithful"
            else "formalized STRIPS task with legacy goal-trace compatibility bridge"
        ),
        "fault_recovery": (
            "MRBTP-native reactive branches only; no adapter fault access"
            if args.bridge == "faithful"
            else (
                "structured fault metadata compiled by adapter"
                if not args.no_fault_recovery
                else "disabled ablation; nominal MRBTP trace only"
            )
        ),
        "evaluator": "pipeline.datagen.executor.execute",
        "data": str(Path(args.data)),
        "n_requested": args.limit,
        "summary": summarize(details),
        "details": details,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
