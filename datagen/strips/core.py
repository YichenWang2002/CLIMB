"""Minimal STRIPS core: ground actions, forward search planner, plan replay.

Facts are tuples of strings, e.g. ("at", "alpha", "shelf_A").
Action schemas are lifted with typed params; grounding enumerates bindings.
Static predicates (e.g. "connected") never change during planning; any ground
action whose static precondition is not in static_facts is pruned up front.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass

Fact = tuple


@dataclass(frozen=True)
class ActionSchema:
    name: str
    params: tuple  # (("r", "robot"), ("l1", "location"), ...)
    pre: tuple  # fact templates, e.g. (("at", "r", "l1"),)
    add: tuple
    delete: tuple
    agents: tuple = ("r",)  # param names of robots performing the action

    def ground(self, binding: dict) -> "GroundAction":
        def sub(tpl):
            return tuple(binding.get(t, t) for t in tpl)

        return GroundAction(
            name=self.name,
            binding=dict(binding),
            pre=frozenset(sub(t) for t in self.pre),
            add=frozenset(sub(t) for t in self.add),
            delete=frozenset(sub(t) for t in self.delete),
            agents=tuple(binding[a] for a in self.agents),
        )


@dataclass(frozen=True)
class GroundAction:
    name: str
    binding: dict
    pre: frozenset
    add: frozenset
    delete: frozenset
    agents: tuple

    def applicable(self, state: frozenset) -> bool:
        return self.pre <= state

    def apply(self, state: frozenset) -> frozenset:
        return (state - self.delete) | self.add

    def label(self) -> str:
        args = ",".join(f"{k}={v}" for k, v in self.binding.items())
        return f"{self.name}({args})"


class Domain:
    def __init__(self, name: str, types: dict, schemas: list,
                 static_facts: list, static_preds: tuple = ("connected", "charge_station_at", "can_reach")):
        self.name = name
        self.types = types  # {"robot": [...], "location": [...], "item": [...]}
        self.schemas = schemas
        self.static_facts = frozenset(static_facts)
        self.static_preds = frozenset(static_preds)
        self._ground = None

    def ground_actions(self) -> list:
        if self._ground is not None:
            return self._ground
        actions = []
        for schema in self.schemas:
            pools = [self.types[t] for _, t in schema.params]
            names = [n for n, _ in schema.params]
            for combo in itertools.product(*pools):
                binding = dict(zip(names, combo))
                # joint actions require distinct agents (no self-handover/co-carry)
                agent_vals = [binding[a] for a in schema.agents]
                if len(set(agent_vals)) < len(agent_vals):
                    continue
                ga = schema.ground(binding)
                # prune: static preconditions must hold in the closed static world
                ok = all(
                    f[0] not in self.static_preds or f in self.static_facts
                    for f in ga.pre
                )
                if ok:
                    actions.append(ga)
        self._ground = actions
        return actions


class Planner:
    """Greedy best-first search (plans need to be valid, not optimal)."""

    def __init__(self, domain: Domain, max_expansions: int = 20000):
        self.domain = domain
        self.max_expansions = max_expansions

    def plan(self, init_dynamic: list, goal: list):
        static = self.domain.static_facts
        init = frozenset(init_dynamic) | static
        goal_set = frozenset(goal)
        actions = self.domain.ground_actions()

        def h(state):
            return len(goal_set - state)

        if h(init) == 0:
            return []
        counter = itertools.count()
        openq = [(h(init), 0, next(counter), init, [])]
        best = {init: 0}
        expansions = 0
        while openq and expansions < self.max_expansions:
            _, g, _, state, path = heapq.heappop(openq)
            expansions += 1
            for a in actions:
                if not a.applicable(state):
                    continue
                ns = a.apply(state)
                ng = g + 1
                if ng < best.get(ns, 1 << 30):
                    best[ns] = ng
                    npath = path + [a]
                    nh = len(goal_set - ns)
                    if nh == 0:
                        return npath
                    heapq.heappush(openq, (nh * 10 + ng, ng, next(counter), ns, npath))
        return None


def replay(plan: list, init_dynamic: list, static_facts: frozenset, goal: list) -> bool:
    state = frozenset(init_dynamic) | static_facts
    for a in plan:
        if not a.applicable(state):
            return False
        state = a.apply(state)
    return frozenset(goal) <= state
