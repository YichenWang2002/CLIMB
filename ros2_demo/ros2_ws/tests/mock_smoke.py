#!/usr/bin/env python3
"""Mock smoke test: execute real dataset XMLs with bt_runtime's pure-Python
engine + MockWorld, and cross-check against the ground-truth symbolic
executor (datagen/executor.py in the repository root).

Runs on any machine -- no ROS, no Gazebo.

Usage (from ros2_demo/):
    python3 ros2_ws/tests/mock_smoke.py
Exit code 0 = all samples passed with goal_reached, matching executor.py.
"""
import json
import os
import sys

THIS = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(THIS)                       # ros2_ws/
ROOT = os.path.dirname(os.path.dirname(WS))      # repository root
sys.path.insert(0, os.path.join(WS, "src", "bt_runtime"))  # bt_runtime pkg
sys.path.insert(0, ROOT)                                   # datagen pkg

from bt_runtime.engine import BTEngine            # noqa: E402
from bt_runtime.leaves import MockWorld           # noqa: E402

# ground truth: symbolic executor + domain builder
from datagen.executor import execute as gt_execute        # noqa: E402
from datagen.strips.domains import build_domain           # noqa: E402

TRAIN_JSONL = os.path.join(ROOT, "data", "train.jsonl")


def pick_samples(n_want=3):
    """Pick: 1st warehouse, 1st hospital, 1st record with faults."""
    picks, have = [], set()
    with open(TRAIN_JSONL) as f:
        for line in f:
            d = json.loads(line)
            m = d["meta"]
            dom = m["domain"]
            key = None
            if dom == "warehouse" and "warehouse" not in have:
                key = "warehouse"
            elif dom == "hospital" and "hospital" not in have:
                key = "hospital"
            elif m.get("faults") and "faults" not in have:
                key = "faults"
            if key:
                have.add(key)
                picks.append((key, d))
            if len(picks) >= n_want:
                break
    return picks


def gt_result(meta, xml):
    """Rebuild the executor task from the jsonl meta and run executor.py.
    Reconstruction mirrors pipeline/eval/evaluate.py:reconstruct_task exactly:
    edges must be full ("connected", a, b) facts, not bare pairs."""
    static = [("can_reach", r, l) for r, l in meta["can_reach"]]
    edges = [("connected", a, b) for a, b in meta["connected"]]
    domain = build_domain(meta["domain"], robots=meta["robots"],
                          items=meta["items"], extra_static=static, edges=edges)
    task = {"domain": meta["domain"], "domain_obj": domain,
            "init_dynamic": [tuple(f) for f in meta["init_dynamic"]],
            "goal": [tuple(f) for f in meta["goal"]],
            "faults": meta["faults"],
            "skill_aliases": meta.get("skill_aliases", {})}
    return gt_execute(task, xml)


def run_one(tag, record):
    meta, xml = record["meta"], record["output"]
    print(f"\n=== sample [{tag}] domain={meta['domain']} scenario={meta.get('scenario', '-')} "
          f"scenario={meta['scenario']} robots={meta['robots']} "
          f"faults={[f['type'] for f in meta['faults']]}")

    gt = gt_result(meta, xml)
    print(f"  ground truth (executor.py): {gt}")

    world = MockWorld.from_task_meta(meta)
    engine = BTEngine(xml, world)
    res = engine.run()
    print(f"  bt_runtime  (engine+mock):  {res}")

    assert res["success"], f"bt_runtime failed: {res['reason']}"
    assert res["reason"] == "goal_reached", res
    assert world.goal_reached(), "goal facts not in final mock state"
    assert gt["success"], f"executor.py ground truth failed?! {gt}"
    assert res["recoveries"] == gt["recoveries"], \
        f"recovery count mismatch: {res['recoveries']} vs gt {gt['recoveries']}"

    for agent, events in sorted(engine.trajectory_by_agent().items()):
        labels = [f"{e['label'].split('(')[0]}:{e['status'][0]}" for e in events]
        print(f"  trajectory[{agent}]: {' -> '.join(labels)}")
    return res


def main():
    picks = pick_samples()
    assert len(picks) == 3, f"could not find 3 suitable samples ({len(picks)} found)"
    n_ok = 0
    for tag, record in picks:
        run_one(tag, record)
        n_ok += 1
    print(f"\nSMOKE OK: {n_ok}/3 real dataset XMLs executed to goal_reached, "
          f"cross-checked against executor.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
