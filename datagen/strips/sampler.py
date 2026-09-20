"""Task sampler: scenario templates -> STRIPS task (init/goal/plan/faults/tier).

Scenarios:
  independent  (T1): each robot delivers its own light item. No coordination.
  relay        (T2 k=2, T3 k=3): zone chain forces Handover at shared handoffs.
  heavy        (T2/T3): heavy item forces CoPickUp/CoMoveTo/CoPlaceDown;
               T3 adds an independent third-robot delivery.
  service      (test/external domains): delivery + domain-unique action goal
               (CatalogBook / WaterPlants / InspectPlants / SanitizeCounter).
Faults (T2: 0-1, T3: 1-2): blocked_edge / battery / handover_fail.
"""
from __future__ import annotations

import random
from collections import deque

from .core import Planner, replay
from .domains import DOMAIN_SPECS, ROBOTS, build_domain, item_facts, random_graph

MAX_TRIES = 60

NOMINAL_SHARED = {"MoveTo", "PickUp", "PlaceDown", "Handover",
                  "CoPickUp", "CoMoveTo", "CoPlaceDown"}


def _planning_domain(domain_name, robots, items, can_reach, goal, edges):
    """Domain for nominal planning: ClearPath/Recharge excluded (their effects
    are never preconditions or goals — recovery is compiled in later), and
    domain-unique schemas included only when the goal actually needs them.
    This keeps the search space small. The FULL domain (all schemas) is built
    separately for the executor."""
    goal_preds = {g[0] for g in goal}
    spec = DOMAIN_SPECS[domain_name]
    extra_ids = {id(s) for s in spec["extra_schemas"]
                 if any(f[0] in goal_preds for f in s.add)}
    static = [("can_reach", r, l) for (r, l) in sorted(can_reach)]
    dom = build_domain(domain_name, robots=robots, items=items, extra_static=static,
                       edges=edges)
    dom.schemas = [s for s in dom.schemas
                   if s.name in NOMINAL_SHARED or id(s) in extra_ids]
    return dom


# ------------------------------------------------------------- graph utils --

def _adj(edges: list) -> dict:
    adj = {}
    for _, a, b in edges:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def _bfs(adj: dict, src: str, dst: str):
    prev = {src: None}
    dq = deque([src])
    while dq:
        u = dq.popleft()
        if u == dst:
            break
        for v in adj.get(u, ()):
            if v not in prev:
                prev[v] = u
                dq.append(v)
    if dst not in prev:
        return None
    path = []
    u = dst
    while u is not None:
        path.append(u)
        u = prev[u]
    return path[::-1]


def _simple_paths(adj: dict, n_nodes: int, rng: random.Random, cap: int = 400):
    paths = []
    starts = sorted(adj)
    rng.shuffle(starts)

    def dfs(path):
        if len(path) == n_nodes:
            paths.append(tuple(path))
            return len(paths) >= cap
        nbs = sorted(adj.get(path[-1], ()))
        rng.shuffle(nbs)
        for v in nbs:
            if v not in path and dfs(path + [v]):
                return True
        return False

    for s in starts:
        if dfs([s]):
            break
    return paths


# ------------------------------------------------------------ task builder --

class Sampler:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    # -- faults -------------------------------------------------------------

    def _apply_faults(self, task: dict, n_faults: int):
        plan = task["plan"]
        spec = DOMAIN_SPECS[task["domain"]]
        adj = _adj(task["edges"])
        rng = self.rng
        options = []
        moves = [(a.binding["r"], (a.binding["l1"], a.binding["l2"]))
                 for a in plan if a.name == "MoveTo"]
        if moves:
            options.append("blocked_edge")
        movers = {}
        for a in plan:
            if a.name == "MoveTo":
                movers.setdefault(a.binding["r"], 0)
                movers[a.binding["r"]] += 1
        movers = {r: c for r, c in movers.items() if c >= 2}
        if movers:
            options.append("battery")
        handovers = [a for a in plan if a.name == "Handover"]
        if handovers:
            options.append("handover_fail")

        rng.shuffle(options)
        chosen = options[:n_faults]
        used_edges = set()
        for ft in chosen:
            if ft == "blocked_edge":
                cands = [e for _, e in moves if e not in used_edges]
                if not cands:
                    continue
                edge = rng.choice(cands)
                used_edges.add(edge)
                task["faults"].append({"type": "blocked_edge", "edge": list(edge)})
            elif ft == "battery":
                r = rng.choice(list(movers))
                budget = rng.randint(1, movers[r] - 1)
                task["faults"].append({"type": "battery", "robot": r, "budget": budget})
                # make sure r can limp to a charge station from its zone
                zone = task["zones"].get(r, spec["locations"])
                for loc in zone:
                    p = _bfs(adj, loc, spec["charge_stations"][0])
                    best = p
                    for s in spec["charge_stations"]:
                        q = _bfs(adj, loc, s)
                        if q and (best is None or len(q) < len(best)):
                            best = q
                    if best:
                        for node in best:
                            task["can_reach"].add((r, node))
            elif ft == "handover_fail":
                a = rng.choice(handovers)
                task["faults"].append({
                    "type": "handover_fail", "giver": a.binding["g"],
                    "receiver": a.binding["t"], "item": a.binding["i"], "times": 1})

    # -- finalize -------------------------------------------------------------

    def _finalize(self, domain_name, robots, items, can_reach, init_dynamic,
                  goal, tier, scenario, zones, edges):
        plan_dom = _planning_domain(domain_name, robots, items, can_reach, goal, edges)
        planner = Planner(plan_dom, max_expansions=8000)
        plan = planner.plan(init_dynamic, goal)
        if plan is None or not replay(plan, init_dynamic, plan_dom.static_facts, goal):
            return None
        used_robots = {ag for a in plan for ag in a.agents}
        if used_robots != set(robots):
            return None
        static = [("can_reach", r, l) for (r, l) in sorted(can_reach)]
        full_dom = build_domain(domain_name, robots=robots, items=items,
                                extra_static=static, edges=edges)
        return {
            "domain": domain_name, "domain_obj": full_dom, "robots": robots,
            "items": items, "init_dynamic": init_dynamic, "goal": goal,
            "plan": plan, "faults": [], "tier": tier, "scenario": scenario,
            "zones": zones, "can_reach": can_reach, "edges": edges,
        }

    def _rebuild_with_faults(self, task: dict, n_faults: int):
        """Faults only affect execution; can_reach augmentation (battery) only
        ADDS static facts, so the original plan stays valid — rebuild the domain
        and confirm by replay instead of re-planning (keeps faults aligned with
        the exact plan the compiler will use)."""
        if n_faults == 0:
            return task
        self._apply_faults(task, n_faults)
        static = [("can_reach", r, l) for (r, l) in sorted(task["can_reach"])]
        domain = build_domain(task["domain"], robots=task["robots"],
                              items=task["items"], extra_static=static,
                              edges=task["edges"])
        if not replay(task["plan"], task["init_dynamic"], domain.static_facts, task["goal"]):
            return None
        task["domain_obj"] = domain
        return task

    # -- scenarios ------------------------------------------------------------

    def _independent(self, domain_name: str, tier: str, n_faults: int = 0):
        spec = DOMAIN_SPECS[domain_name]
        rng = self.rng
        robots = ROBOTS[:2]
        lights = [i for i in spec["items"] if i not in spec["heavy_items"]]
        if len(lights) < 2:
            return None
        items = rng.sample(lights, 2)
        locs = spec["locations"]
        edges = random_graph(locs, rng, rng.randint(1, 2))
        can_reach = {(r, l) for r in robots for l in locs}
        init, goal = [], []
        zones = {}
        for r, i in zip(robots, items):
            src, dst = rng.sample(locs, 2)
            init.append(("item_at", i, src))
            goal.append(("item_at", i, dst))
            zones[r] = locs
        for r in robots:
            init.append(("at", r, rng.choice(locs)))
            init.append(("free", r))
            init.append(("mobile", r))
        init += item_facts(domain_name)
        task = self._finalize(domain_name, robots, items, can_reach, init, goal,
                              tier, "independent", zones, edges)
        if task is None or not (3 <= len(task["plan"]) <= 12):
            return None
        return self._rebuild_with_faults(task, n_faults)

    def _relay(self, domain_name: str, tier: str, k: int, n_faults: int,
               service: bool = False):
        spec = DOMAIN_SPECS[domain_name]
        rng = self.rng
        edges = random_graph(spec["locations"], rng, rng.randint(1, 2))
        adj = _adj(edges)
        robots = ROBOTS[:k]
        paths = _simple_paths(adj, k + 1, rng)
        if not paths:
            return None
        path = list(rng.choice(paths))
        lights = [i for i in spec["items"] if i not in spec["heavy_items"]]
        item = rng.choice(lights)
        can_reach = set()
        zones = {}
        for idx, r in enumerate(robots):
            zone = [path[idx], path[idx + 1]]
            zones[r] = zone
            for l in zone:
                can_reach.add((r, l))
        init = [("item_at", item, path[0])]
        goal = [("item_at", item, path[k])]
        for idx, r in enumerate(robots):
            init.append(("at", r, rng.choice(zones[r])))
            init.append(("free", r))
            init.append(("mobile", r))
        init += item_facts(domain_name)
        if service and domain_name == "library":
            goal.append(("cataloged", item))
        elif service and domain_name == "kitchen":
            goal.append(("sanitized", path[k]))
        task = self._finalize(domain_name, robots, [item], can_reach, init, goal,
                              tier, "relay" + ("+service" if service else ""), zones, edges)
        if task is None:
            return None
        n_hand = sum(1 for a in task["plan"] if a.name == "Handover")
        if n_hand < k - 1:
            return None
        return self._rebuild_with_faults(task, n_faults)

    def _heavy(self, domain_name: str, tier: str, extra_robot: bool, n_faults: int,
               service: bool = False):
        spec = DOMAIN_SPECS[domain_name]
        rng = self.rng
        robots = ROBOTS[:3] if extra_robot else ROBOTS[:2]
        carriers = robots[:2]
        heavy = rng.choice(spec["heavy_items"])
        locs = spec["locations"]
        edges = random_graph(locs, rng, rng.randint(1, 2))
        src, dst = rng.sample(locs, 2)
        items = [heavy]
        can_reach = {(r, l) for r in carriers for l in locs}
        zones = {r: locs for r in carriers}
        init = [("item_at", heavy, src)]
        goal = [("item_at", heavy, dst)]
        for r in robots:
            init.append(("at", r, rng.choice(locs)))
            init.append(("free", r))
            init.append(("mobile", r))
        if extra_robot:
            r3 = robots[2]
            lights = [i for i in spec["items"] if i not in spec["heavy_items"]]
            light = rng.choice(lights)
            items.append(light)
            s3, d3 = rng.sample(locs, 2)
            init.append(("item_at", light, s3))
            goal.append(("item_at", light, d3))
            for l in locs:
                can_reach.add((r3, l))
            zones[r3] = locs
        if service and domain_name == "greenhouse":
            goal.append((rng.choice(["watered", "inspected"]), dst))
        elif service and domain_name == "kitchen":
            goal.append(("sanitized", dst))
        init += item_facts(domain_name)
        task = self._finalize(domain_name, robots, items, can_reach, init, goal,
                              tier, "heavy" + ("+extra" if extra_robot else "")
                              + ("+service" if service else ""), zones, edges)
        if task is None:
            return None
        names = [a.name for a in task["plan"]]
        if not ({"CoPickUp", "CoMoveTo", "CoPlaceDown"} <= set(names)):
            return None
        return self._rebuild_with_faults(task, n_faults)

    # -- entry ------------------------------------------------------------------

    def sample(self, domain_name: str, tier: str):
        spec = DOMAIN_SPECS[domain_name]
        is_test = spec["split"] in {"test", "external"}
        for _ in range(MAX_TRIES):
            try:
                if tier == "T1":
                    task = self._independent(domain_name, "T1", n_faults=0)
                elif tier == "T2":
                    if self.rng.random() < 0.5:
                        task = self._relay(domain_name, "T2", k=2, n_faults=self.rng.randint(0, 1))
                    else:
                        task = self._heavy(domain_name, "T2", extra_robot=False,
                                           n_faults=self.rng.randint(0, 1),
                                           service=is_test and self.rng.random() < 0.5)
                else:  # T3
                    roll = self.rng.random()
                    if roll < 0.5:
                        task = self._relay(domain_name, "T3", k=3,
                                           n_faults=self.rng.randint(1, 2),
                                           service=is_test and self.rng.random() < 0.5)
                    else:
                        task = self._heavy(domain_name, "T3", extra_robot=True,
                                           n_faults=self.rng.randint(1, 2),
                                           service=is_test and self.rng.random() < 0.5)
                if task is not None:
                    return task
            except Exception:  # noqa: BLE001 - resample on any construction failure
                continue
        return None
