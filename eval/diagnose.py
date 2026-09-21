"""Executor diagnosis extension: instrumented re-execution of failed generations.

For each failed record, re-runs the Executor with tracing and emits a
structured diagnosis per the SPCL-ER spec:
  first_failed_node / missing_preconditions / robot_positions /
  valid_outgoing_locations / goal_remaining + coarse error_type label.
Never reveals the correct plan.
"""
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import os as _os; sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from datagen.executor import Executor, ExecutionError, re_extract, BINDING_MAP, JOINT_PAIRS
from eval.evaluate import reconstruct_task, load_jsonl


class DiagExecutor(Executor):
    def __init__(self, task, xml):
        super().__init__(task, xml)
        self.trace = []  # per leaf action: dict(name, binding, result, missing_pre, note)

    def _run_simple(self, ga):
        snap = frozenset(self.state) | self.static
        res = super()._run_simple(ga)
        if res == "FAILURE":
            missing = sorted(map(str, ga.pre - snap))
            note = ""
            b = ga.binding
            if ga.name == "MoveTo":
                if (b["r"], b["l2"]) not in self.can_reach:
                    note = f"robot {b['r']} cannot_reach {b['l2']}"
                elif (b["l1"], b["l2"]) in self.blocked:
                    note = f"edge {b['l1']}->{b['l2']} blocked, needs ClearPath"
                elif b["r"] in self.battery_dead:
                    note = f"{b['r']} battery dead, needs Recharge path"
                elif b["r"] in self.battery_budget and self.battery_budget[b["r"]] < 0:
                    note = f"{b['r']} battery exhausted on this move"
            self.trace.append({"name": ga.name, "binding": dict(ga.binding),
                               "result": res, "missing_pre": missing, "note": note})
        else:
            self.trace.append({"name": ga.name, "binding": dict(ga.binding),
                               "result": res, "missing_pre": [], "note": ""})
        return res

    def _run_joint(self, name, attrs, node_id):
        res = super()._run_joint(name, attrs, node_id)
        if res == "FAILURE":
            canon, key_attrs = JOINT_PAIRS[name]
            binding = {bn: attrs[xa] for xa, bn in BINDING_MAP[name].items() if xa in attrs}
            ga = self._ground.get((canon, tuple(sorted(binding.items()))))
            missing = sorted(map(str, ga.pre - (frozenset(self.state) | self.static))) if ga else ["<ungroundable>"]
            self.trace.append({"name": name, "binding": binding, "result": res,
                               "missing_pre": missing, "note": "joint"})
        return res

    def _lookup(self, name, attrs):
        ga = super()._lookup(name, attrs)
        if ga is None:
            self.trace.append({"name": name, "binding": dict(attrs), "result": "FAILURE",
                               "missing_pre": ["<ungroundable binding>"],
                               "note": "ungroundable"})
        return ga

    # ---- diagnosis helpers ----
    def robot_positions(self):
        return {r: f[2] for f in self.state if f[0] == "at" for r in [f[1]]}

    def valid_outgoing(self, loc, robot):
        outs = []
        for a, b in self.edges:
            if a == loc and (robot, b) in self.can_reach and (a, b) not in self.blocked:
                outs.append(b)
            elif b == loc and (robot, a) in self.can_reach and (b, a) not in self.blocked:
                outs.append(a)
        return sorted(outs)

    def goal_remaining(self):
        return sorted(map(str, self.goal - self.state))



    def tick(self, node, agent=None):
        tag = node.tag
        if tag in ("IsAtLocation", "IsItemAt", "IsCarrying"):
            res = super().tick(node, agent)
            if res == "FAILURE":
                fact = {
                    "IsAtLocation": ("at", node.get("robot"), node.get("location")),
                    "IsItemAt": ("item_at", node.get("item"), node.get("location")),
                    "IsCarrying": ("carrying", node.get("robot"), node.get("item")),
                }[tag]
                self.trace.append({"name": tag, "binding": dict(node.attrib),
                                   "result": res, "missing_pre": [str(fact)], "note": "condition"})
            return res
        return super().tick(node, agent)


def classify(task, gen, res, ex):
    """Coarse error-type label aligned with corruption taxonomy."""
    reason = res["reason"]
    vocab_locs = {f[1] for f in task["domain_obj"].static_facts if f[0]=="connected"} 
    vocab_locs |= {f[2] for f in task["domain_obj"].static_facts if f[0]=="connected"}
    vocab_items = set(task_meta_items) if False else None
    if reason.startswith("error:"):
        r = reason.lower()
        if "parse" in r or "no <root" in r or "top tag" in r:
            return "structural_parse"
        if "unknown action" in r or "unknown node tag" in r:
            return "hallucinated_node"
        if "missing attr" in r:
            return "missing_attr"
        if "subtree" in r or "main tree" in r or "empty" in r:
            return "structural_tree"
        if "tick limit" in r or "deadlock" in r:
            return "deadlock"
        return "structural_other"
    if reason == "cycle_limit":
        return "deadlock"
    if reason == "tree_done_goal_missing":
        return "goal_segment_missing"   # tree ran clean to SUCCESS but goal facts absent
    # tree_failed: inspect first FAILURE event
    fev = next((e for e in ex.trace if e["result"] == "FAILURE"), None)
    if fev is None:
        return "other"
    if fev.get("note") == "condition":
        return "condition_guard_failed"
    if fev.get("note") == "ungroundable":
        if fev["name"] == "MoveTo":
            b = fev["binding"]
            locs = {f[1] for f in task["domain_obj"].static_facts if f[0]=="connected"} | \
                   {f[2] for f in task["domain_obj"].static_facts if f[0]=="connected"}
            l1, l2, r = b.get("from"), b.get("to"), b.get("robot")
            if l1 not in locs or l2 not in locs or l1 == l2:
                return "attr_value_hallucinated"
            if (r, l2) not in ex.can_reach:
                return "attr_wrong_location"
            return "missing_intermediate_leg"
        return "attr_value_hallucinated"
    if fev.get("note") == "joint":
        return "handover_miscoord"
    mp = " ".join(fev["missing_pre"])
    note = fev["note"]
    if "cannot_reach" in note:
        return "attr_wrong_location"     # to/from not valid for this robot (attr error)
    if "blocked" in note:
        return "fault_recovery_missing"  # needs ClearPath inserted
    if "battery" in note:
        return "battery_recovery_missing"
    if "('at'" in mp or "('at'," in mp:
        # robot/item not positioned -> either missing move leg or wrong from attr
        b = fev["binding"]
        l1 = b.get("l1") or b.get("l")
        # if robot has never successfully moved and l1 != its init pos -> wrong-from-ish;
        # else missing leg insertion
        return "position_precondition_missing"
    if mp:
        return "other_precondition_missing"  # carrying/free/item_at/co_carrying etc.
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ev = json.load(open(a.eval_json))
    recs = load_jsonl(a.data)
    gens, details = ev["generations"], ev["details"]
    assert len(gens) == len(recs) == len(details)

    out, stats = [], Counter()
    scen_stats = defaultdict(Counter)
    n_succ = 0
    for rec, g, det in zip(recs, gens, details):
        task = reconstruct_task(rec["meta"])
        try:
            xml = re_extract(g)
            ex = DiagExecutor(task, xml)
            res = ex.run()
        except ExecutionError as e:
            res = {"success": False, "reason": f"error: {e}", "ticks": 0, "recoveries": 0}
            ex = None
        if res["success"]:
            n_succ += 1
            continue
        etype = classify(task, g, res, ex) if ex else classify(task, g, res, DummyEx())
        scen = rec["meta"].get("scenario", "?")
        stats[etype] += 1
        scen_stats[scen][etype] += 1
        n_fail = sum(1 for e in (ex.trace if ex else []) if e["result"] == "FAILURE")
        n_ok = sum(1 for e in (ex.trace if ex else []) if e["result"] == "SUCCESS")
        fev = next((e for e in (ex.trace if ex else []) if e["result"] == "FAILURE"), None)
        diag = {
            "id": det.get("id"), "scenario": scen, "domain": rec["meta"].get("domain"),
            "reason": res["reason"], "error_type": etype,
            "executed_actions": n_ok, "failed_actions": n_fail,
            "first_failed_node": fev, 
            "robot_positions": ex.robot_positions() if ex else {},
            "goal_remaining": ex.goal_remaining() if ex else [],
        }
        out.append(diag)

    Path(a.out).write_text(json.dumps({
        "source": a.eval_json, "n": len(recs), "n_success": n_succ, "n_fail": len(out),
        "error_type_counts": dict(stats.most_common()),
        "error_type_by_scenario": {s: dict(c.most_common()) for s, c in scen_stats.items()},
        "diagnoses": out,
    }, ensure_ascii=False, indent=1))
    print(f"success {n_succ}/{len(recs)}  fail {len(out)}")
    for k, v in stats.most_common():
        print(f"  {k:32s} {v:4d}  {v/len(out)*100:.1f}%")
    for s, c in scen_stats.items():
        print(f"[{s}]", dict(c.most_common()))


class DummyEx:
    trace = []

if __name__ == "__main__":
    main()
