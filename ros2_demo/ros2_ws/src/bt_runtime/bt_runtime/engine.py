"""Pure-Python BehaviorTree.CPP v4 multi-agent BT engine.

NO rclpy import in this module: it runs on any Python 3.7+ interpreter, so the
control-flow semantics can be unit-tested on machines without ROS 2.

The tick semantics are a 1:1 port of the pipeline's ground-truth symbolic
executor (datagen/executor.py):

  * Sequence / Fallback: standard BT.CPP semantics with progress memory
    (a child that returned RUNNING is re-ticked first on the next cycle;
    the index resets when the node halts with SUCCESS/FAILURE).
  * Parallel: BT.CPP semantics -- a child that returned SUCCESS/FAILURE is
    halted and its status LATCHED; only RUNNING children keep being ticked.
    success_threshold defaults to N, failure_threshold to N - s_th + 1.
  * One <BehaviorTree> per robot; MainTree is a Parallel of <SubTree> nodes.
  * SignalReady/WaitReady: rendezvous keyed by
    ("signal", item, signaler, waiter); both sides must post before either
    completes.
  * HandoverGive/HandoverTake and Co* leaves: joint rendezvous. The canonical
    key is defined by JOINT_PAIRS; the side identity is the leaf name for
    Handover and the node id for Co* leaves. The completing side applies the
    joint effect via world.apply_joint(); the partner side picks up the
    stored result on its next tick.
  * Global tick limit of 2000 (deadlock guard), identical to executor.py.

World interaction is delegated to a duck-typed `world` object (see
leaves.py for the mock and ROS implementations):

  world.condition(tag, attrs, agent) -> bool
  world.run_simple(name, attrs, node_id, agent) -> "SUCCESS"|"FAILURE"|"RUNNING"
  world.apply_joint(canon, binding, agent) -> bool
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

S, F, R = "SUCCESS", "FAILURE", "RUNNING"

TICK_LIMIT = 2000

# XML attribute -> STRIPS binding name, per action (identical to executor.py).
BINDING_MAP = {
    "MoveTo": {"robot": "r", "from": "l1", "to": "l2"},
    "PickUp": {"robot": "r", "item": "i", "location": "l"},
    "PlaceDown": {"robot": "r", "item": "i", "location": "l"},
    "Recharge": {"robot": "r", "station": "l"},
    "ClearPath": {"robot": "r", "edge_from": "l1", "edge_to": "l2"},
    "HandoverGive": {"giver": "g", "receiver": "t", "item": "i", "location": "l"},
    "HandoverTake": {"receiver": "t", "giver": "g", "item": "i", "location": "l"},
    "CoPickUp": {"robot1": "r1", "robot2": "r2", "item": "i", "location": "l"},
    "CoPlaceDown": {"robot1": "r1", "robot2": "r2", "item": "i", "location": "l"},
    "CoMoveTo": {"robot1": "r1", "robot2": "r2", "item": "i", "from": "l1", "to": "l2"},
    "CatalogBook": {"robot": "r", "item": "i", "location": "l"},
    "WaterPlants": {"robot": "r", "location": "l"},
    "InspectPlants": {"robot": "r", "location": "l"},
}

# Joint leaf tag -> (canonical action name, key attribute order).
JOINT_PAIRS = {
    "HandoverGive": ("Handover", ("giver", "receiver", "item")),
    "HandoverTake": ("Handover", ("giver", "receiver", "item")),
    "CoPickUp": ("CoPickUp", ("robot1", "robot2", "item", "location")),
    "CoPlaceDown": ("CoPlaceDown", ("robot1", "robot2", "item", "location")),
    "CoMoveTo": ("CoMoveTo", ("robot1", "robot2", "item", "from", "to")),
}

CONDITION_TAGS = ("IsAtLocation", "IsItemAt", "IsCarrying")


class ExecutionError(Exception):
    pass


def re_extract(xml_string: str) -> str:
    """Extract the <root>...</root> block (LLM outputs sometimes add prose)."""
    m = re.search(r"<root[^>]*>.*?</root>", xml_string, re.DOTALL)
    if not m:
        raise ExecutionError("no <root>...</root> block found")
    return m.group(0)


class BTEngine:
    """Tick a BTCPP v4 multi-agent XML against a `world` backend."""

    def __init__(self, xml_string: str, world, aliases: dict = None):
        self.world = world
        # skill aliases: renamed leaf tag -> standard action name
        # (pipeline rename_skills.py augmentation; empty for regular records)
        self.aliases = aliases or {}

        self.trees = {}
        self.root = self._parse(xml_string)
        self.node_state = {}   # id(element) -> Sequence/Fallback progress index
        self.sync_posted = {}  # rendezvous key -> set of sides posted
        self.sync_result = {}  # (key, side) -> latched result for the partner
        self.ticks = 0
        self.trace = []        # (tick, agent, kind, label, status) leaf events

    # ------------------------------------------------------------- parse --
    def _parse(self, xml_string: str):
        try:
            root = ET.fromstring(re_extract(xml_string))
        except ExecutionError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ExecutionError(f"XML parse error: {e}")
        if root.tag != "root":
            raise ExecutionError("top tag is not <root>")
        for bt in root.findall("BehaviorTree"):
            self.trees[bt.get("ID")] = bt
        main_id = root.get("main_tree_to_execute", "MainTree")
        if main_id not in self.trees:
            raise ExecutionError(f"main tree '{main_id}' not found")
        return root

    # ------------------------------------------------------------ helpers --
    @staticmethod
    def _label(tag: str, attrs: dict) -> str:
        args = ",".join(f"{k}={v}" for k, v in attrs.items())
        return f"{tag}({args})"

    def _record(self, agent: str, kind: str, label: str, status: str):
        self.trace.append((self.ticks, agent, kind, label, status))

    # ------------------------------------------------------- rendezvous ---
    def _run_joint(self, name: str, attrs: dict, node_id: int, agent: str) -> str:
        """Joint action rendezvous -- 1:1 port of executor.py:_run_joint."""
        canon, key_attrs = JOINT_PAIRS[name]
        # malformed model output: missing attributes -> hard error, scored as failure
        for k in list(key_attrs) + list(BINDING_MAP[name]):
            if k not in attrs:
                raise ExecutionError(f"joint action '{name}' missing attr '{k}'")
        key = (canon, tuple(attrs[k] for k in key_attrs))
        # side identity: Handover -> leaf name; Co* -> node id
        side = name if canon == "Handover" else node_id
        if (key, side) in self.sync_result:
            res = self.sync_result.pop((key, side))
            self._record(agent, "joint", self._label(name, attrs), res)
            return res
        posted = self.sync_posted.setdefault(key, set())
        posted.add(side)
        if canon == "Handover":
            ready = posted >= {"HandoverGive", "HandoverTake"}
            others = posted - {side}
        else:
            ready = len(posted) >= 2
            others = posted - {side}
        if not ready:
            return R
        # complete the joint action: result for the partner side(s) is stored,
        # this side gets it immediately; posted is cleared for the next round.
        binding = {}
        for xml_attr, bind_name in BINDING_MAP[name].items():
            if xml_attr not in attrs:
                raise ExecutionError(f"action '{name}' missing attr '{xml_attr}'")
            binding[bind_name] = attrs[xml_attr]
        ok = self.world.apply_joint(canon, binding, agent)
        res = S if ok else F
        self.sync_posted.pop(key, None)
        for o in others:
            self.sync_result[(key, o)] = res
        self._record(agent, "joint", self._label(name, attrs), res)
        return res

    # -------------------------------------------------------------- tick ---
    def tick(self, node: ET.Element, agent: str = None) -> str:
        self.ticks += 1
        if self.ticks > TICK_LIMIT:
            raise ExecutionError("tick limit exceeded (deadlock)")
        tag = node.tag
        tag = self.aliases.get(tag, tag)
        if tag == "BehaviorTree":
            if len(node) == 0:
                raise ExecutionError("empty <BehaviorTree>")
            return self.tick(node[0], agent)
        if tag == "SubTree":
            tid = node.get("ID")
            if tid not in self.trees:
                raise ExecutionError(f"SubTree '{tid}' not defined")
            # Per-robot subtrees are named Agent_<robot>; use that for tracing.
            if agent is None and tid and tid.startswith("Agent_"):
                agent = tid[len("Agent_"):]
            return self.tick(self.trees[tid], agent)
        if tag == "Sequence":
            i = self.node_state.get(id(node), 0)
            while i < len(node):
                st = self.tick(node[i], agent)
                if st == R:
                    self.node_state[id(node)] = i
                    return R
                if st == F:
                    self.node_state[id(node)] = 0
                    return F
                i += 1
            self.node_state[id(node)] = 0
            return S
        if tag == "Fallback":
            i = self.node_state.get(id(node), 0)
            while i < len(node):
                st = self.tick(node[i], agent)
                if st == R:
                    self.node_state[id(node)] = i
                    return R
                if st == S:
                    self.node_state[id(node)] = 0
                    return S
                i += 1
            self.node_state[id(node)] = 0
            return F
        if tag == "Parallel":
            # BT.CPP semantics: a child that returned S/F is halted and its
            # status latched; only RUNNING children keep being ticked.
            latched = self.node_state.setdefault(("par", id(node)), {})
            succ = fail = 0
            for ci, child in enumerate(node):
                if ci in latched:
                    st = latched[ci]
                else:
                    st = self.tick(child, agent)
                    if st != R:
                        latched[ci] = st
                if st == S:
                    succ += 1
                elif st == F:
                    fail += 1
            s_th = int(node.get("success_threshold") or len(node))
            f_th = int(node.get("failure_threshold") or (len(node) - s_th + 1))
            if succ >= s_th:
                return S
            if fail >= f_th:
                return F
            return R
        if tag in CONDITION_TAGS:
            ok = self.world.condition(tag, dict(node.attrib), agent)
            st = S if ok else F
            self._record(agent, "condition", self._label(tag, dict(node.attrib)), st)
            return st
        if tag in ("SignalReady", "WaitReady"):
            # Rendezvous barrier keyed exactly like executor.py.
            if tag == "SignalReady":
                key = ("signal", node.get("item"), node.get("robot"), node.get("to"))
                side = "ready"
            else:
                key = ("signal", node.get("item"), node.get("from"), node.get("robot"))
                side = "wait"
            if (key, side) in self.sync_result:
                res = self.sync_result.pop((key, side))
                self._record(agent, "sync", self._label(tag, dict(node.attrib)), res)
                return res
            posted = self.sync_posted.setdefault(key, set())
            posted.add(side)
            if posted >= {"ready", "wait"}:
                self.sync_posted.pop(key, None)
                self.sync_result[(key, "wait" if side == "ready" else "ready")] = S
                self._record(agent, "sync", self._label(tag, dict(node.attrib)), S)
                return S
            return R
        if tag in JOINT_PAIRS:
            return self._run_joint(tag, dict(node.attrib), id(node), agent)
        if tag in BINDING_MAP:
            attrs = dict(node.attrib)
            for xml_attr in BINDING_MAP[tag]:
                if xml_attr not in attrs:
                    raise ExecutionError(f"action '{tag}' missing attr '{xml_attr}'")
            st = self.world.run_simple(tag, attrs, id(node), agent)
            if st != R:
                self._record(agent, "action", self._label(tag, attrs), st)
            return st
        raise ExecutionError(f"unknown node tag '{tag}'")

    # --------------------------------------------------------------- run ---
    def tick_once(self) -> str:
        """Tick the main tree once and return its status."""
        main = self.trees[self.root.get("main_tree_to_execute", "MainTree")]
        return self.tick(main)

    def goal_reached(self) -> bool:
        gr = getattr(self.world, "goal_reached", None)
        return gr() if callable(gr) else True

    def run(self, max_cycles: int = 500) -> dict:
        """Drive the main tree to a terminal status (blocking; mock/offline use)."""
        try:
            for _ in range(max_cycles):
                st = self.tick_once()
                if st == S:
                    ok = self.goal_reached()
                    return {"success": ok,
                            "reason": "goal_reached" if ok else "tree_done_goal_missing",
                            "ticks": self.ticks,
                            "recoveries": getattr(self.world, "recoveries_fired", 0)}
                if st == F:
                    return {"success": False, "reason": "tree_failed",
                            "ticks": self.ticks,
                            "recoveries": getattr(self.world, "recoveries_fired", 0)}
            return {"success": False, "reason": "cycle_limit", "ticks": self.ticks,
                    "recoveries": getattr(self.world, "recoveries_fired", 0)}
        except ExecutionError as e:
            return {"success": False, "reason": f"error: {e}", "ticks": self.ticks,
                    "recoveries": getattr(self.world, "recoveries_fired", 0)}

    def trajectory_by_agent(self) -> dict:
        """Trace events grouped per robot, for runner_node printing."""
        out = {}
        for tick, agent, kind, label, status in self.trace:
            out.setdefault(agent or "?", []).append(
                {"tick": tick, "kind": kind, "label": label, "status": status})
        return out
