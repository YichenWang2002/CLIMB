"""rclpy runner node: load one BTCPP v4 XML file and drive all robot trees
to a terminal status, then print the per-robot trajectory.

Modes (parameter `mode`):
  * mock -- simulated world (no Nav2/Gazebo needed). Optionally takes a
    train.jsonl-style meta JSON via `task_meta_json` to reproduce a dataset
    task, or synthesizes a permissive world from the XML alone.
  * ros  -- MoveTo goes to Nav2 NavigateToPose; all other leaves are stubs
    (see leaves.RosWorld).

Example:
  ros2 run bt_runtime bt_runner --ros-args \
      -p xml_path:=/path/to/tree.xml -p mode:=mock -p task_meta_json:=/path/to/meta.json
"""
from __future__ import annotations

import json
import sys

import rclpy
from rclpy.node import Node

from .engine import BTEngine, S, F, R, ExecutionError
from .leaves import MockWorld, RosWorld


def _load_waypoints(path: str) -> dict:
    import yaml  # available in any ROS 2 Humble install (python3-yaml)
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("waypoints", data) if isinstance(data, dict) else {}


def _synthesize_mock_world(xml_string: str) -> MockWorld:
    """Build a permissive mock world when no task meta is provided: every
    location mentioned in the XML is mutually connected, every robot can
    reach every location and starts free/mobile/charged at its first `from`.
    Items start at the locations referenced by PickUp/IsItemAt nodes.
    Good enough to smoke-test tree structure, not semantic correctness."""
    import xml.etree.ElementTree as ET
    from .engine import re_extract

    root = ET.fromstring(re_extract(xml_string))
    locs, robots, items_at = set(), set(), set()
    for el in root.iter():
        for a in ("from", "to", "location", "station", "edge_from", "edge_to"):
            if el.get(a):
                locs.add(el.get(a))
        for a in ("robot", "giver", "receiver", "robot1", "robot2"):
            if el.get(a):
                robots.add(el.get(a))
        if el.tag == "PickUp":
            items_at.add((el.get("item"), el.get("location")))
    static = set()
    for a in locs:
        for b in locs:
            if a != b:
                static.add(("connected", a, b))
        for r in robots:
            static.add(("can_reach", r, a))
    init = set()
    for r in robots:
        init |= {("free", r), ("mobile", r)}
    for i, l in items_at:
        init.add(("item_at", i, l))
        init.add(("light", i))
    return MockWorld(static_facts=static, init_state=init, goal=(), faults=[])


class BTRunnerNode(Node):
    def __init__(self):
        super().__init__("bt_runner")
        self.declare_parameter("xml_path", "")
        self.declare_parameter("mode", "mock")          # mock | ros
        self.declare_parameter("waypoints_file", "")
        self.declare_parameter("robots", ["alpha", "beta"])
        self.declare_parameter("task_meta_json", "")    # optional, mock mode
        self.declare_parameter("tick_period", 0.1)      # seconds per BT cycle
        self.declare_parameter("max_cycles", 500)
        # optional: dump the per-robot trace to a JSON file on finish
        # (consumed by demo_pack/scripts/plot_trace.py)
        self.declare_parameter("trace_json", "")

        xml_path = self.get_parameter("xml_path").value
        if not xml_path:
            raise RuntimeError("parameter 'xml_path' is required")
        with open(xml_path, "r") as f:
            xml_string = f.read()

        mode = self.get_parameter("mode").value
        if mode == "ros":
            waypoints = _load_waypoints(self.get_parameter("waypoints_file").value)
            robots = list(self.get_parameter("robots").value)
            self.world = RosWorld(self, waypoints, robots)
        else:
            meta_path = self.get_parameter("task_meta_json").value
            if meta_path:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                self.world = MockWorld.from_task_meta(meta, verbose=True)
            else:
                self.world = _synthesize_mock_world(xml_string)
                self.get_logger().warn(
                    "no task_meta_json given; using a permissive synthesized world")

        self.engine = BTEngine(xml_string, self.world)
        self.cycles = 0
        self.max_cycles = int(self.get_parameter("max_cycles").value)
        period = float(self.get_parameter("tick_period").value)
        self.timer = self.create_timer(period, self._on_tick)
        self.get_logger().info(f"BT runner started: mode={mode} xml={xml_path}")

    def _on_tick(self):
        self.cycles += 1
        try:
            st = self.engine.tick_once()
        except ExecutionError as e:
            self._finish(False, f"error: {e}")
            return
        if st in (S, F):
            ok = st == S and self.engine.goal_reached()
            reason = ("goal_reached" if ok else
                      "tree_done_goal_missing" if st == S else "tree_failed")
            self._finish(ok, reason)
            return
        if self.cycles >= self.max_cycles:
            self._finish(False, "cycle_limit")

    def _finish(self, success: bool, reason: str):
        self.timer.cancel()
        self.get_logger().info(
            f"BT finished: success={success} reason={reason} ticks={self.engine.ticks} "
            f"recoveries={getattr(self.world, 'recoveries_fired', 0)}")
        for agent, events in sorted(self.engine.trajectory_by_agent().items()):
            self.get_logger().info(f"--- trajectory [{agent}] ---")
            for ev in events:
                self.get_logger().info(
                    f"  t={ev['tick']:4d} {ev['kind']:9s} {ev['status']:7s} {ev['label']}")
        for line in getattr(self.world, "log", []):
            self.get_logger().info(f"  world: {line}")
        trace_json = self.get_parameter("trace_json").value
        if trace_json:
            payload = {
                "success": success,
                "reason": reason,
                "ticks": self.engine.ticks,
                "recoveries": getattr(self.world, "recoveries_fired", 0),
                "trajectory": self.engine.trajectory_by_agent(),
            }
            try:
                with open(trace_json, "w") as f:
                    json.dump(payload, f, indent=2)
                self.get_logger().info(f"trace written to {trace_json}")
            except OSError as e:
                self.get_logger().error(f"could not write trace_json '{trace_json}': {e}")
        # stop spinning; the process exits once the executor is dropped
        raise SystemExit(0 if success else 1)


def main(args=None):
    rclpy.init(args=args)
    node = BTRunnerNode()
    try:
        rclpy.spin(node)
    except SystemExit as e:
        code = e.code or 0
    except KeyboardInterrupt:
        code = 130
    finally:
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
