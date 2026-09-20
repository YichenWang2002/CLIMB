#!/usr/bin/env python3
"""Gazebo Classic fault-injection helper for the bt_runtime demo.

Demonstrates the "blocked edge -> ClearPath recovery" loop on a live system:

  1. This node watches the trigger robot's odometry. When the robot gets
     within `trigger_distance` of the midpoint of the blocked edge, it spawns
     a box (`box_name`) at the midpoint via gazebo_ros' /spawn_entity service.
     Nav2 then fails the MoveTo across that edge, the BT Fallback fires, and
     the ClearPath recovery leaf runs.
  2. When the bt_runtime ClearPath stub fires, it publishes one
     std_msgs/Empty on `clear_topic` (/fault_cleared). This node then calls
     /delete_entity to remove the box, so the retried MoveTo succeeds.

HOW TO WIRE ClearPath (do NOT edit leaves.py):
  In your own bringup script/node, after creating the BTRunnerNode, register
  a stub handler on its world:

      from std_msgs.msg import Empty
      clear_pub = runner_node.create_publisher(Empty, "/fault_cleared", 10)

      def clear_path_stub(attrs, agent, node):
          clear_pub.publish(Empty())
          node.get_logger().info(f"ClearPath stub fired: {attrs} -> /fault_cleared")
          return "SUCCESS"

      runner_node.world.stub_handlers["ClearPath"] = clear_path_stub

  (See demo_pack/README.md for a copy-paste launch snippet. RosWorld looks up
  stub_handlers["ClearPath"](attrs, agent, node) -> "SUCCESS"|"FAILURE"|"RUNNING".)

Parameters
----------
waypoints_file : str   bt_runtime waypoints YAML; used with edge_from/edge_to
edge_from/edge_to : str  location ids of the blocked edge (lookup in YAML)
x1,y1,x2,y2 : double   explicit edge endpoint coords (used if no waypoints_file)
robot : str            trigger robot name ("alpha")
odom_topic : str       odometry topic (default "/<robot>/odom")
trigger_distance : double  spawn when robot is this close to the edge midpoint
box_name : str         Gazebo model name ("fault_box")
box_size : double      box edge length in metres (0.6)
spawn_source : str     "inline" (generated box SDF) | "model" (model:// URI)
model_uri : str        used when spawn_source=="model" ("model://cardboard_box")
clear_topic : str      Empty topic that deletes the box ("/fault_cleared")
frame_id : str         pose reference frame for spawn ("world")

Only rclpy + gazebo_msgs + std_msgs + nav_msgs are required.
"""
from __future__ import annotations

import math

import rclpy
from rclpy.node import Node

from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty

INLINE_BOX_SDF = """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>false</static>
    <link name="link">
      <collision name="collision">
        <geometry><box><size>{s} {s} {s}</size></box></geometry>
      </collision>
      <visual name="visual">
        <geometry><box><size>{s} {s} {s}</size></box></geometry>
        <material>
          <ambient>0.8 0.2 0.2 1</ambient>
          <diffuse>0.8 0.2 0.2 1</diffuse>
        </material>
      </visual>
      <inertial>
        <mass>5.0</mass>
        <inertia>
          <ixx>0.1</ixx><iyy>0.1</iyy><izz>0.1</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
    </link>
  </model>
</sdf>
"""


def _load_waypoint_midpoint(path: str, edge_from: str, edge_to: str):
    import yaml  # python3-yaml, present in any ROS 2 Humble install
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    wps = data.get("waypoints", data) if isinstance(data, dict) else {}
    a, b = wps.get(edge_from), wps.get(edge_to)
    if a is None or b is None:
        raise RuntimeError(
            f"waypoints '{edge_from}'/'{edge_to}' not found in {path}")
    return ((float(a["x"]) + float(b["x"])) / 2.0,
            (float(a["y"]) + float(b["y"])) / 2.0)


class GazeboFaultNode(Node):
    def __init__(self):
        super().__init__("gazebo_fault")
        self.declare_parameter("waypoints_file", "")
        self.declare_parameter("edge_from", "charge_bay")
        self.declare_parameter("edge_to", "shelf_b")
        self.declare_parameter("x1", 0.0)
        self.declare_parameter("y1", 0.0)
        self.declare_parameter("x2", 0.0)
        self.declare_parameter("y2", 0.0)
        self.declare_parameter("robot", "alpha")
        self.declare_parameter("odom_topic", "")     # default: /<robot>/odom
        self.declare_parameter("trigger_distance", 1.0)
        self.declare_parameter("box_name", "fault_box")
        self.declare_parameter("box_size", 0.6)
        self.declare_parameter("spawn_source", "inline")   # inline | model
        self.declare_parameter("model_uri", "model://cardboard_box")
        self.declare_parameter("clear_topic", "/fault_cleared")
        self.declare_parameter("frame_id", "world")

        gp = self.get_parameter
        # --- resolve the blocked-edge midpoint ---------------------------
        wp_file = gp("waypoints_file").value
        if wp_file:
            self.mid_x, self.mid_y = _load_waypoint_midpoint(
                wp_file, gp("edge_from").value, gp("edge_to").value)
        else:
            self.mid_x = (gp("x1").value + gp("x2").value) / 2.0
            self.mid_y = (gp("y1").value + gp("y2").value) / 2.0

        self.robot = gp("robot").value
        self.trigger_distance = float(gp("trigger_distance").value)
        self.box_name = gp("box_name").value
        self.box_size = float(gp("box_size").value)
        self.spawn_source = gp("spawn_source").value
        self.model_uri = gp("model_uri").value
        self.frame_id = gp("frame_id").value
        odom_topic = gp("odom_topic").value or f"/{self.robot}/odom"

        # --- ROS interfaces ----------------------------------------------
        self.spawn_cli = self.create_client(SpawnEntity, "/spawn_entity")
        self.delete_cli = self.create_client(DeleteEntity, "/delete_entity")
        self.create_subscription(Odometry, odom_topic, self._on_odom, 10)
        self.create_subscription(
            Empty, gp("clear_topic").value, self._on_clear, 10)

        self.spawned = False       # box currently present in the world
        self._busy = False         # a spawn/delete call is in flight

        self.get_logger().info(
            f"gazebo_fault armed: edge midpoint=({self.mid_x:.2f}, {self.mid_y:.2f}), "
            f"trigger={self.robot} within {self.trigger_distance:.2f} m on {odom_topic}, "
            f"box='{self.box_name}' ({self.spawn_source})")

    # ---------------------------------------------------------- odometry --
    def _on_odom(self, msg: Odometry):
        if self.spawned or self._busy:
            return
        dx = msg.pose.pose.position.x - self.mid_x
        dy = msg.pose.pose.position.y - self.mid_y
        if math.hypot(dx, dy) <= self.trigger_distance:
            self._spawn_box()

    # ------------------------------------------------------------- spawn --
    def _spawn_box(self):
        if not self.spawn_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/spawn_entity not available")
            return
        req = SpawnEntity.Request()
        req.name = self.box_name
        req.reference_frame = self.frame_id
        req.initial_pose.position.x = self.mid_x
        req.initial_pose.position.y = self.mid_y
        req.initial_pose.position.z = self.box_size / 2.0
        if self.spawn_source == "model":
            req.xml = (
                f'<?xml version="1.0"?><sdf version="1.6"><model name="{self.box_name}">'
                f"<include><uri>{self.model_uri}</uri></include></model></sdf>")
        else:
            req.xml = INLINE_BOX_SDF.format(name=self.box_name, s=self.box_size)
        self._busy = True
        self.get_logger().info(
            f"spawning '{self.box_name}' at ({self.mid_x:.2f}, {self.mid_y:.2f})")
        future = self.spawn_cli.call_async(req)
        future.add_done_callback(self._spawn_done)

    def _spawn_done(self, future):
        self._busy = False
        try:
            resp = future.result()
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"spawn call failed: {e}")
            return
        if resp.success:
            self.spawned = True
            self.get_logger().info(f"FAULT ACTIVE: '{self.box_name}' spawned")
        else:
            self.get_logger().warn(f"spawn refused: {resp.status_message}")

    # ------------------------------------------------------------ clear ---
    def _on_clear(self, _msg: Empty):
        """bt_runtime's ClearPath stub published on clear_topic -> delete box."""
        if not self.spawned or self._busy:
            self.get_logger().info("clear requested but no box is spawned; ignoring")
            return
        if not self.delete_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("/delete_entity not available")
            return
        req = DeleteEntity.Request()
        req.name = self.box_name
        self._busy = True
        self.get_logger().info(f"ClearPath received -> deleting '{self.box_name}'")
        future = self.delete_cli.call_async(req)
        future.add_done_callback(self._delete_done)

    def _delete_done(self, future):
        self._busy = False
        try:
            resp = future.result()
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"delete call failed: {e}")
            return
        if resp.success:
            self.spawned = False  # re-arm: the fault can trigger again
            self.get_logger().info(
                f"FAULT CLEARED: '{self.box_name}' deleted (re-armed)")
        else:
            self.get_logger().warn(f"delete refused: {resp.status_message}")


def main(args=None):
    rclpy.init(args=args)
    node = GazeboFaultNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
