"""Symbolic executor: tick a BehaviorTree.CPP v4 XML against the STRIPS world
model with fault injection and multi-agent rendezvous semantics.

Used both to VALIDATE generated ground-truth trees (datagen) and to SCORE
model-generated trees (eval). Success = MainTree returns SUCCESS and the
goal facts hold in the final world state.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import deque

S, F, R = "SUCCESS", "FAILURE", "RUNNING"
MAX_NODE_TICKS = 2_000_000

# XML attribute -> STRIPS binding name, per action
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
    "SanitizeCounter": {"robot": "r", "location": "l"},
}
JOINT_PAIRS = {
    "HandoverGive": ("Handover", ("giver", "receiver", "item")),
    "HandoverTake": ("Handover", ("giver", "receiver", "item")),
    "CoPickUp": ("CoPickUp", ("robot1", "robot2", "item", "location")),
    "CoPlaceDown": ("CoPlaceDown", ("robot1", "robot2", "item", "location")),
    "CoMoveTo": ("CoMoveTo", ("robot1", "robot2", "item", "from", "to")),
}


class ExecutionError(Exception):
    pass


class Executor:
    def __init__(self, task: dict, xml_string: str):
        self.task = task
        self.domain = task["domain_obj"]
        self.state = set(task["init_dynamic"])
        self.goal = set(task["goal"])
        self.static = self.domain.static_facts
        self.charge_locs = [f[1] for f in self.static if f[0] == "charge_station_at"]
        self.edges = {(f[1], f[2]) for f in self.static if f[0] == "connected"}
        self.can_reach = {(f[1], f[2]) for f in self.static if f[0] == "can_reach"}

        # faults
        self.blocked = set()
        self.battery_budget = {}
        self.battery_dead = set()
        self.handover_fails = {}  # (giver, receiver, item) -> remaining fail times
        for f_ in task.get("faults", []):
            if f_["type"] == "blocked_edge":
                a, b = f_["edge"]
                self.blocked.add((a, b))
                self.blocked.add((b, a))
            elif f_["type"] == "battery":
                self.battery_budget[f_["robot"]] = f_["budget"]
            elif f_["type"] == "handover_fail":
                k = (f_["giver"], f_["receiver"], f_["item"])
                self.handover_fails[k] = self.handover_fails.get(k, 0) + f_.get("times", 1)

        # skill aliases: renamed simple-action leaf tag -> standard action name
        # (rename_skills.py augmentation; empty for regular records)
        self.aliases = task.get("skill_aliases", {})

        # rendezvous: key -> set of sides posted; results: key -> status
        self.sync_posted = {}
        self.sync_result = {}

        self._ground = {}
        for ga in self.domain.ground_actions():
            self._ground[(ga.name, tuple(sorted(ga.binding.items())))] = ga

        self.trees = {}
        self.root = self._parse(xml_string)
        self.detect_stall = self.root.get("mrbtp_bridge") == "faithful_v1"
        self.node_state = {}  # id(element) -> progress index
        self.recoveries_fired = 0
        self.ticks = 0

    # ------------------------------------------------------------- parse --
    def _parse(self, xml_string: str):
        try:
            m = re_extract(xml_string)
            root = ET.fromstring(m)
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
    def _lookup(self, name: str, attrs: dict):
        if name not in BINDING_MAP:
            raise ExecutionError(f"unknown action '{name}'")
        binding = {}
        for xml_attr, bind_name in BINDING_MAP[name].items():
            if xml_attr not in attrs:
                raise ExecutionError(f"action '{name}' missing attr '{xml_attr}'")
            binding[bind_name] = attrs[xml_attr]
        ga = self._ground.get((name, tuple(sorted(binding.items()))))
        return ga

    def _dist_to_station(self, loc: str, robot: str) -> int:
        adj = {}
        for a, b in self.edges:
            if (robot, b) in self.can_reach:
                adj.setdefault(a, set()).add(b)
        best = 1 << 30
        for s in self.charge_locs:
            prev = {loc}
            dq = deque([(loc, 0)])
            while dq:
                u, d = dq.popleft()
                if u == s:
                    best = min(best, d)
                    break
                for v in adj.get(u, ()):
                    if v not in prev:
                        prev.add(v)
                        dq.append((v, d + 1))
        return best

    # --------------------------------------------------------- action run --
    def _run_simple(self, ga) -> str:
        """Non-joint action: check preconditions (+ faults), apply effects."""
        b = ga.binding
        if ga.name == "MoveTo":
            r, l1, l2 = b["r"], b["l1"], b["l2"]
            if (r, l2) not in self.can_reach:
                return F
            if (l1, l2) in self.blocked and ("clear", l1, l2) not in self.state:
                return F
            if r in self.battery_dead:
                # limp mode: only moves that approach a charge station succeed
                if l2 not in self.charge_locs and self._dist_to_station(l2, r) >= self._dist_to_station(l1, r):
                    return F
            if not ga.applicable(frozenset(self.state) | self.static):
                return F
            if r in self.battery_budget and r not in self.battery_dead:
                self.battery_budget[r] -= 1
                if self.battery_budget[r] < 0:
                    self.battery_dead.add(r)
                    return F  # this move drains the battery -> fails
            self.state -= ga.delete
            self.state |= ga.add
            return S
        if ga.name == "ClearPath":
            if not ga.applicable(frozenset(self.state) | self.static):
                return F
            self.state -= ga.delete
            self.state |= ga.add
            l1, l2 = b["l1"], b["l2"]
            self.blocked.discard((l1, l2))
            self.blocked.discard((l2, l1))
            self.recoveries_fired += 1
            return S
        if ga.name == "Recharge":
            if not ga.applicable(frozenset(self.state) | self.static):
                return F
            self.state -= ga.delete
            self.state |= ga.add
            r = b["r"]
            self.battery_dead.discard(r)
            if r in self.battery_budget:
                self.battery_budget[r] = 99
            self.recoveries_fired += 1
            return S
        if not ga.applicable(frozenset(self.state) | self.static):
            return F
        self.state -= ga.delete
        self.state |= ga.add
        return S

    def battery_dead_check(self, r):
        return r in self.battery_dead

    # ------------------------------------------------------- rendezvous ---
    def _run_joint(self, name: str, attrs: dict, node_id: int) -> str:
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
        amap = BINDING_MAP[name]
        for xml_attr, bind_name in amap.items():
            binding[bind_name] = attrs[xml_attr]
        ga = self._ground.get((canon, tuple(sorted(binding.items()))))
        hf_key = (binding.get("g"), binding.get("t"), binding.get("i"))
        if ga is None or not ga.applicable(frozenset(self.state) | self.static):
            res = F
        elif canon == "Handover" and self.handover_fails.get(hf_key, 0) > 0:
            self.handover_fails[hf_key] -= 1
            res = F  # this specific handover fumbles; fallback retry will succeed
        else:
            self.state -= ga.delete
            self.state |= ga.add
            res = S
        self.sync_posted.pop(key, None)
        for o in others:
            self.sync_result[(key, o)] = res
        return res

    # -------------------------------------------------------------- tick ---
    def tick(self, node: ET.Element, agent: str = None) -> str:
        self.ticks += 1
        if self.ticks > MAX_NODE_TICKS:
            raise ExecutionError("node tick limit exceeded")
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
        if tag == "ReactiveSequence":
            # MRBTP builds memoryless py_trees sequences.  Re-evaluate every
            # child from the beginning on each tick so earlier conditions can
            # react to state changes while a later child is RUNNING.
            for child in node:
                st = self.tick(child, agent)
                if st != S:
                    return st
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
        if tag == "ReactiveFallback":
            # Equivalent to py_trees Selector(memory=False), which is the
            # control node emitted by MRBTP's public implementation.
            for child in node:
                st = self.tick(child, agent)
                if st != F:
                    return st
            return F
        if tag == "RetryUntilSuccessful":
            if len(node) != 1:
                raise ExecutionError("RetryUntilSuccessful requires one child")
            st = self.tick(node[0], agent)
            if st == S:
                self.node_state.pop(("retry", id(node)), None)
                return S
            if st == R:
                return R
            attempts = self.node_state.get(("retry", id(node)), 0) + 1
            self.node_state[("retry", id(node))] = attempts
            raw_limit = node.get("num_attempts", "-1")
            try:
                limit = int(raw_limit)
            except ValueError as exc:
                raise ExecutionError(
                    f"invalid RetryUntilSuccessful num_attempts '{raw_limit}'"
                ) from exc
            return F if limit >= 0 and attempts >= limit else R
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
        if tag == "CheckFact":
            predicate = node.get("predicate")
            if not predicate:
                raise ExecutionError("CheckFact missing predicate")
            raw_args = node.get("args", "")
            args = tuple(arg for arg in raw_args.split(",") if arg)
            return S if (predicate, *args) in (self.state | self.static) else F
        if tag in ("IsAtLocation", "IsItemAt", "IsCarrying"):
            fact = {
                "IsAtLocation": ("at", node.get("robot"), node.get("location")),
                "IsItemAt": ("item_at", node.get("item"), node.get("location")),
                "IsCarrying": ("carrying", node.get("robot"), node.get("item")),
            }[tag]
            return S if fact in self.state else F
        if tag in ("SignalReady", "WaitReady"):
            if tag == "SignalReady":
                key = ("signal", node.get("item"), node.get("robot"), node.get("to"))
                side = "ready"
            else:
                key = ("signal", node.get("item"), node.get("from"), node.get("robot"))
                side = "wait"
            if (key, side) in self.sync_result:
                return self.sync_result.pop((key, side))
            posted = self.sync_posted.setdefault(key, set())
            posted.add(side)
            if posted >= {"ready", "wait"}:
                self.sync_posted.pop(key, None)
                self.sync_result[(key, "wait" if side == "ready" else "ready")] = S
                return S
            return R
        if tag in JOINT_PAIRS:
            return self._run_joint(tag, dict(node.attrib), id(node))
        if tag in BINDING_MAP:
            ga = self._lookup(tag, dict(node.attrib))
            if ga is None:
                return F
            return self._run_simple(ga)
        raise ExecutionError(f"unknown node tag '{tag}'")

    # --------------------------------------------------------------- run ---
    def _progress_snapshot(self):
        """Return world/rendezvous state for reactive no-progress checks."""

        return (
            frozenset(self.state),
            tuple(sorted(self.blocked)),
            tuple(sorted(self.battery_budget.items())),
            tuple(sorted(self.battery_dead)),
            tuple(sorted(self.handover_fails.items())),
            tuple(
                sorted(
                    (repr(key), tuple(sorted(map(repr, value))))
                    for key, value in self.sync_posted.items()
                )
            ),
            tuple(sorted(map(repr, self.sync_result.items()))),
        )

    def run(self, max_cycles: int = 500) -> dict:
        main = self.trees[self.root.get("main_tree_to_execute", "MainTree")]
        stalled = 0
        try:
            for _ in range(max_cycles):
                before = self._progress_snapshot() if self.detect_stall else None
                st = self.tick(main)
                if st == S:
                    ok = self.goal <= self.state
                    return {"success": ok, "reason": "goal_reached" if ok else "tree_done_goal_missing",
                            "ticks": self.ticks, "recoveries": self.recoveries_fired}
                if st == F:
                    return {"success": False, "reason": "tree_failed",
                            "ticks": self.ticks, "recoveries": self.recoveries_fired}
                if self.detect_stall:
                    after = self._progress_snapshot()
                    stalled = stalled + 1 if after == before else 0
                    if stalled >= 2:
                        return {
                            "success": False,
                            "reason": "stalled_no_progress",
                            "ticks": self.ticks,
                            "recoveries": self.recoveries_fired,
                        }
            return {"success": False, "reason": "cycle_limit", "ticks": self.ticks,
                    "recoveries": self.recoveries_fired}
        except ExecutionError as e:
            return {"success": False, "reason": f"error: {e}", "ticks": self.ticks,
                    "recoveries": self.recoveries_fired}


def re_extract(xml_string: str) -> str:
    import re
    m = re.search(r"<root[^>]*>.*?</root>", xml_string, re.DOTALL)
    if not m:
        raise ExecutionError("no <root>...</root> block found")
    return m.group(0)


def execute(task: dict, xml_string: str) -> dict:
    try:
        ex = Executor(task, xml_string)
    except ExecutionError as e:
        return {"success": False, "reason": f"error: {e}", "ticks": 0, "recoveries": 0}
    return ex.run()
