"""Leaf-node action layer for the BT engine, in two modes:

* ``MockWorld`` -- fully simulated STRIPS world, NO ROS required. Reproduces
  the ground-truth semantics of datagen/executor.py exactly:
  preconditions/effects per action schema, blocked_edge / battery /
  handover_fail fault injection, ClearPath & Recharge recovery counting,
  battery "limp mode" (only moves that approach a charge station succeed).
  Optional per-node latency simulation (actions stay RUNNING for N ticks).

* ``RosWorld`` -- ROS 2 backend. rclpy (and only here) is imported lazily
  inside this class. MoveTo is wired to Nav2's NavigateToPose action; every
  other leaf (PickUp/PlaceDown/Recharge/ClearPath/Handover/Co*/CatalogBook/
  WaterPlants/InspectPlants) is a replaceable stub: it logs and consults a
  user-provided service/action callback table.

The world interface consumed by engine.BTEngine:

  condition(tag, attrs, agent) -> bool
  run_simple(name, attrs, node_id, agent) -> "SUCCESS"|"FAILURE"|"RUNNING"
  apply_joint(canon, binding, agent) -> bool
"""
from __future__ import annotations

from collections import deque

S, F, R = "SUCCESS", "FAILURE", "RUNNING"

# --------------------------------------------------------------------------
# Action schemas, ported 1:1 from datagen/strips/domains.py.
# Fact templates use binding names (r/l1/l2/i/l/g/t/r1/r2); values are
# substituted from the XML attributes via engine.BINDING_MAP.
# --------------------------------------------------------------------------
ACTION_SCHEMAS = {
    # -- shared simple actions -------------------------------------------
    "MoveTo": dict(
        pre=(("at", "r", "l1"), ("connected", "l1", "l2"),
             ("can_reach", "r", "l2"), ("mobile", "r")),
        add=(("at", "r", "l2"),), delete=(("at", "r", "l1"),)),
    "PickUp": dict(
        pre=(("at", "r", "l"), ("item_at", "i", "l"), ("free", "r"), ("light", "i")),
        add=(("carrying", "r", "i"),), delete=(("item_at", "i", "l"), ("free", "r"))),
    "PlaceDown": dict(
        pre=(("at", "r", "l"), ("carrying", "r", "i")),
        add=(("item_at", "i", "l"), ("free", "r")), delete=(("carrying", "r", "i"))),
    "Recharge": dict(
        pre=(("at", "r", "l"), ("charge_station_at", "l")),
        add=(("charged", "r"),), delete=()),
    "ClearPath": dict(
        pre=(("at", "r", "l1"), ("connected", "l1", "l2")),
        add=(("clear", "l1", "l2"),), delete=()),
    # -- joint canonical actions (reached via engine rendezvous) ---------
    "Handover": dict(
        pre=(("at", "g", "l"), ("at", "t", "l"), ("carrying", "g", "i"), ("free", "t")),
        add=(("carrying", "t", "i"),), delete=(("carrying", "g", "i"), ("free", "t"))),
    "CoPickUp": dict(
        pre=(("at", "r1", "l"), ("at", "r2", "l"), ("item_at", "i", "l"),
             ("free", "r1"), ("free", "r2"), ("heavy", "i")),
        add=(("co_carrying", "r1", "r2", "i"),),
        delete=(("item_at", "i", "l"), ("free", "r1"), ("free", "r2"),
                ("mobile", "r1"), ("mobile", "r2"))),
    "CoMoveTo": dict(
        pre=(("co_carrying", "r1", "r2", "i"), ("at", "r1", "l1"), ("at", "r2", "l1"),
             ("connected", "l1", "l2"), ("can_reach", "r1", "l2"), ("can_reach", "r2", "l2")),
        add=(("at", "r1", "l2"), ("at", "r2", "l2")),
        delete=(("at", "r1", "l1"), ("at", "r2", "l1"))),
    "CoPlaceDown": dict(
        pre=(("co_carrying", "r1", "r2", "i"), ("at", "r1", "l"), ("at", "r2", "l")),
        add=(("item_at", "i", "l"), ("free", "r1"), ("free", "r2"),
             ("mobile", "r1"), ("mobile", "r2")),
        delete=(("co_carrying", "r1", "r2", "i"),)),
    # -- domain-unique actions (library / greenhouse test domains) -------
    "CatalogBook": dict(
        pre=(("at", "r", "l"), ("item_at", "i", "l")),
        add=(("cataloged", "i"),), delete=()),
    "WaterPlants": dict(
        pre=(("at", "r", "l"),), add=(("watered", "l"),), delete=()),
    "InspectPlants": dict(
        pre=(("at", "r", "l"),), add=(("inspected", "l"),), delete=()),
}

# Condition tag -> fact template (identical to executor.py).
CONDITION_FACTS = {
    "IsAtLocation": lambda a: ("at", a["robot"], a["location"]),
    "IsItemAt": lambda a: ("item_at", a["item"], a["location"]),
    "IsCarrying": lambda a: ("carrying", a["robot"], a["item"]),
}


def _ground(name: str, binding: dict):
    """Instantiate a schema into concrete pre/add/delete fact sets."""
    schema = ACTION_SCHEMAS[name]

    def sub(tpl):
        return tuple(binding.get(t, t) for t in tpl)

    return (frozenset(sub(t) for t in schema["pre"]),
            frozenset(sub(t) for t in schema["add"]),
            frozenset(sub(t) for t in schema["delete"]))


# --------------------------------------------------------------------------
# Mock world
# --------------------------------------------------------------------------
class MockWorld:
    """Simulated STRIPS world matching executor.py semantics exactly.

    Parameters
    ----------
    static_facts : iterable of tuples
        Static facts: ("connected", a, b), ("charge_station_at", l),
        ("can_reach", robot, loc).
    init_state : iterable of tuples
        Dynamic facts true at t=0 (at/free/mobile/item_at/light/heavy/...).
    goal : iterable of tuples
        Goal facts; goal_reached() checks goal <= state.
    faults : list of dicts, pipeline fault spec:
        {"type": "blocked_edge", "edge": [a, b]}
        {"type": "battery", "robot": r, "budget": n}
        {"type": "handover_fail", "giver": g, "receiver": t, "item": i, "times": n}
    delays : dict action_name -> int, optional
        If > 0, the action stays RUNNING for that many ticks before
        resolving (per BT node). Default 0 = atomic, like executor.py.
    """

    def __init__(self, static_facts=(), init_state=(), goal=(), faults=(),
                 delays=None, verbose=False):
        self.static = frozenset(tuple(f) for f in static_facts)
        self.state = set(tuple(f) for f in init_state)
        self.goal = set(tuple(f) for f in goal)
        self.verbose = verbose

        self.charge_locs = [f[1] for f in self.static if f[0] == "charge_station_at"]
        self.edges = {(f[1], f[2]) for f in self.static if f[0] == "connected"}
        self.can_reach = {(f[1], f[2]) for f in self.static if f[0] == "can_reach"}

        # faults -- identical handling to executor.py
        self.blocked = set()
        self.battery_budget = {}
        self.battery_dead = set()
        self.handover_fails = {}  # (giver, receiver, item) -> remaining fail times
        for f_ in faults or ():
            if f_["type"] == "blocked_edge":
                a, b = f_["edge"]
                self.blocked.add((a, b))
                self.blocked.add((b, a))
            elif f_["type"] == "battery":
                self.battery_budget[f_["robot"]] = f_["budget"]
            elif f_["type"] == "handover_fail":
                k = (f_["giver"], f_["receiver"], f_["item"])
                self.handover_fails[k] = self.handover_fails.get(k, 0) + f_.get("times", 1)

        self.delays = dict(delays or {})
        self._pending = {}        # node_id -> ticks left before resolving
        self.recoveries_fired = 0
        self.log = []             # human-readable event log

    # ------------------------------------------------------ construction --
    @classmethod
    def from_task_meta(cls, meta: dict, **kw):
        """Build from a train.jsonl `meta` record (domain/robots/init/goal/
        faults/connected/charge_stations/can_reach)."""
        static = [("connected", a, b) for a, b in meta["connected"]]
        static += [("charge_station_at", l) for l in meta.get("charge_stations", [])]
        static += [("can_reach", r, l) for r, l in meta.get("can_reach", [])]
        return cls(static_facts=static,
                   init_state=[tuple(f) for f in meta["init_dynamic"]],
                   goal=[tuple(f) for f in meta["goal"]],
                   faults=meta.get("faults", []), **kw)

    # ------------------------------------------------------- fault hooks --
    def inject_blocked_edge(self, a: str, b: str):
        """Runtime fault-injection hook (demo Fallback recovery)."""
        self.blocked.add((a, b))
        self.blocked.add((b, a))
        self.log.append(f"FAULT injected: blocked_edge {a}<->{b}")

    def inject_battery(self, robot: str, budget: int):
        self.battery_budget[robot] = budget
        self.log.append(f"FAULT injected: battery {robot} budget={budget}")

    def inject_handover_fail(self, giver: str, receiver: str, item: str, times: int = 1):
        k = (giver, receiver, item)
        self.handover_fails[k] = self.handover_fails.get(k, 0) + times
        self.log.append(f"FAULT injected: handover_fail {k} x{times}")

    # ---------------------------------------------------------- interface --
    def goal_reached(self) -> bool:
        return self.goal <= self.state

    def condition(self, tag: str, attrs: dict, agent: str = None) -> bool:
        return CONDITION_FACTS[tag](attrs) in self.state

    def run_simple(self, name: str, attrs: dict, node_id: int = None,
                   agent: str = None) -> str:
        # optional latency simulation: stay RUNNING for `delays[name]` ticks
        d = self.delays.get(name, 0)
        if d > 0:
            left = self._pending.get(node_id)
            if left is None:
                left = self._pending[node_id] = d
            if left > 0:
                self._pending[node_id] = left - 1
                return R
            self._pending.pop(node_id, None)
        binding = _binding_of(name, attrs)
        st = self._run_simple(name, binding)
        if self.verbose and st != R:
            self.log.append(f"{agent or '?'}: {name}{attrs} -> {st}")
        return st

    def apply_joint(self, canon: str, binding: dict, agent: str = None) -> bool:
        """Apply the joint effect once the engine rendezvous completes.
        Mirrors executor.py:_run_joint completion (incl. handover_fail)."""
        pre, add, delete = _ground(canon, binding)
        hf_key = (binding.get("g"), binding.get("t"), binding.get("i"))
        if not pre <= (frozenset(self.state) | self.static):
            if self.verbose:
                self.log.append(f"{agent or '?'}: joint {canon} -> FAILURE (pre)")
            return False
        if canon == "Handover" and self.handover_fails.get(hf_key, 0) > 0:
            self.handover_fails[hf_key] -= 1
            if self.verbose:
                self.log.append(f"{agent or '?'}: joint Handover -> FAILURE (fumble)")
            return False  # this specific handover fumbles; fallback retry succeeds
        self.state -= delete
        self.state |= add
        if self.verbose:
            self.log.append(f"{agent or '?'}: joint {canon} -> SUCCESS")
        return True

    # --------------------------------------------------------- action run --
    def _run_simple(self, name: str, b: dict) -> str:
        """Non-joint action: check preconditions (+ faults), apply effects.
        1:1 port of executor.py:_run_simple."""
        pre, add, delete = _ground(name, b)
        facts = frozenset(self.state) | self.static
        if name == "MoveTo":
            r, l1, l2 = b["r"], b["l1"], b["l2"]
            if (r, l2) not in self.can_reach:
                return F
            if (l1, l2) in self.blocked and ("clear", l1, l2) not in self.state:
                return F
            if r in self.battery_dead:
                # limp mode: only moves that approach a charge station succeed
                if l2 not in self.charge_locs and \
                        self._dist_to_station(l2, r) >= self._dist_to_station(l1, r):
                    return F
            if not pre <= facts:
                return F
            if r in self.battery_budget and r not in self.battery_dead:
                self.battery_budget[r] -= 1
                if self.battery_budget[r] < 0:
                    self.battery_dead.add(r)
                    return F  # this move drains the battery -> fails
            self.state -= delete
            self.state |= add
            return S
        if name == "ClearPath":
            if not pre <= facts:
                return F
            self.state -= delete
            self.state |= add
            l1, l2 = b["l1"], b["l2"]
            self.blocked.discard((l1, l2))
            self.blocked.discard((l2, l1))
            self.recoveries_fired += 1
            return S
        if name == "Recharge":
            if not pre <= facts:
                return F
            self.state -= delete
            self.state |= add
            r = b["r"]
            self.battery_dead.discard(r)
            if r in self.battery_budget:
                self.battery_budget[r] = 99
            self.recoveries_fired += 1
            return S
        if not pre <= facts:
            return F
        self.state -= delete
        self.state |= add
        return S

    def _dist_to_station(self, loc: str, robot: str) -> int:
        """BFS distance to nearest charge station over edges the robot may
        traverse (can_reach target rule) -- same as executor.py."""
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


# XML attribute -> binding name for simple leaves (must match engine.BINDING_MAP;
# duplicated here so leaves.py stays importable stand-alone).
from .engine import BINDING_MAP as _BINDING_MAP  # noqa: E402


def _binding_of(name: str, attrs: dict) -> dict:
    return {bn: attrs[xa] for xa, bn in _BINDING_MAP[name].items()}


# --------------------------------------------------------------------------
# ROS 2 world (rclpy imported lazily -- only in this class)
# --------------------------------------------------------------------------
class RosWorld:
    """ROS 2 backend.

    * MoveTo -> Nav2 NavigateToPose action goal per robot namespace; the pose
      is looked up in the waypoints table (loc id -> x, y, yaw).
    * All other leaves are replaceable stubs: the call is logged and the
      result comes from ``stub_handlers[name]`` if provided, else SUCCESS.
      ``stub_handlers[name](attrs, agent, node) -> "SUCCESS"|"FAILURE"|"RUNNING"``
      is where a real manipulation/docking service or action client plugs in.
    * Conditions come from ``condition_provider(tag, attrs, agent) -> bool``
      if given (e.g. backed by a blackboard/perception node); otherwise they
      optimistically return True -- this is a PLACEHOLDER, see README.
    * Joint effects (Handover/Co*) are coordination-only at this level: the
      engine rendezvous guarantees both sides arrived; apply_joint logs and
      returns True. Physical co-manipulation must be implemented via
      stub_handlers by the integrator.

    NOT verified on a live Nav2 stack -- see README.md.
    """

    def __init__(self, node, waypoints: dict, robots: list,
                 stub_handlers: dict = None, condition_provider=None,
                 goal_timeout_sec: float = 120.0):
        import rclpy  # noqa: F401  (lazy: leaves.py is importable without ROS)
        from rclpy.action import ActionClient
        from nav2_msgs.action import NavigateToPose

        self.node = node
        self.waypoints = waypoints  # loc id -> {"x":..,"y":..,"yaw":..}
        self.robots = list(robots)
        self.stub_handlers = dict(stub_handlers or {})
        self.condition_provider = condition_provider
        self.goal_timeout_sec = goal_timeout_sec
        self.recoveries_fired = 0

        self._nav_clients = {
            r: ActionClient(node, NavigateToPose, f"/{r}/navigate_to_pose")
            for r in self.robots}
        self._nav_jobs = {}  # node_id -> {"robot":r, "goal_handle_future":..., "result_future":...}

    # ---------------------------------------------------------- interface --
    def goal_reached(self) -> bool:
        # Goal checking against a symbolic state is a mock-world concept; in
        # ROS mode the tree's SUCCESS is taken as mission success.
        return True

    def condition(self, tag: str, attrs: dict, agent: str = None) -> bool:
        if self.condition_provider is not None:
            return bool(self.condition_provider(tag, attrs, agent))
        self.node.get_logger().warn(
            f"condition {tag}{attrs} has no provider; returning True (placeholder)")
        return True

    def run_simple(self, name: str, attrs: dict, node_id: int = None,
                   agent: str = None) -> str:
        if name == "MoveTo":
            return self._run_move_to(attrs, node_id)
        handler = self.stub_handlers.get(name)
        if handler is not None:
            return handler(attrs, agent, self.node)
        self.node.get_logger().info(
            f"[stub] {name}({', '.join(f'{k}={v}' for k, v in attrs.items())}) -> SUCCESS")
        return S

    def apply_joint(self, canon: str, binding: dict, agent: str = None) -> bool:
        handler = self.stub_handlers.get(canon)
        if handler is not None:
            return handler(binding, agent, self.node) != F
        self.node.get_logger().info(
            f"[stub] joint {canon}({', '.join(f'{k}={v}' for k, v in binding.items())}) -> SUCCESS")
        return True

    # --------------------------------------------------------------- Nav2 --
    def _run_move_to(self, attrs: dict, node_id: int) -> str:
        robot = attrs["robot"]
        target = attrs["to"]
        wp = self.waypoints.get(target)
        log = self.node.get_logger()
        if wp is None:
            log.error(f"MoveTo: no waypoint for location '{target}'")
            return F
        if robot not in self._nav_clients:
            log.error(f"MoveTo: no Nav2 client for robot '{robot}'")
            return F

        job = self._nav_jobs.get(node_id)
        if job is None:
            # send the goal once per BT node activation
            import math
            from geometry_msgs.msg import PoseStamped, Quaternion
            from nav2_msgs.action import NavigateToPose
            goal = NavigateToPose.Goal()
            goal.pose = PoseStamped()
            goal.pose.header.frame_id = "map"
            goal.pose.header.stamp = self.node.get_clock().now().to_msg()
            goal.pose.pose.position.x = float(wp["x"])
            goal.pose.pose.position.y = float(wp["y"])
            yaw = float(wp.get("yaw", 0.0))
            goal.pose.pose.orientation = Quaternion(
                x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))
            client = self._nav_clients[robot]
            if not client.wait_for_server(timeout_sec=5.0):
                log.error(f"Nav2 action server not available for '{robot}'")
                return F
            log.info(f"MoveTo {robot}: {attrs['from']} -> {target} "
                     f"({wp['x']:.2f}, {wp['y']:.2f})")
            job = self._nav_jobs[node_id] = {
                "robot": robot,
                "goal_handle_future": client.send_goal_async(goal),
            }
            return R

        if job.get("result_future") is None:
            gh_future = job["goal_handle_future"]
            if not gh_future.done():
                return R
            goal_handle = gh_future.result()
            if not goal_handle.accepted:
                log.warn(f"MoveTo goal rejected ({robot} -> {target})")
                self._nav_jobs.pop(node_id, None)
                return F
            job["result_future"] = goal_handle.get_result_async()
            return R

        res_future = job["result_future"]
        if not res_future.done():
            return R
        self._nav_jobs.pop(node_id, None)
        # nav2_msgs/NavigateToPose result is empty; status 4 == SUCCEEDED
        status = res_future.result().status
        if status == 4:
            log.info(f"MoveTo {robot} -> {target}: SUCCEEDED")
            return S
        log.warn(f"MoveTo {robot} -> {target}: nav status {status} -> FAILURE")
        return F
