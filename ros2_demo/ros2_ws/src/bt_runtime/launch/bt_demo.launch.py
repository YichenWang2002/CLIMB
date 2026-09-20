"""Example bringup: AWS small-warehouse Gazebo world + Nav2 for two
namespaced robots + the bt_runtime runner.

!!! THIS FILE IS AN ANNOTATED TEMPLATE -- NEVER RUN ON A LIVE SYSTEM !!!
It shows the intended wiring for a ROS 2 Humble + Gazebo Classic + Nav2
setup. You must adapt:
  * the AWS world package launch name (check `ros2 pkg list` /
    `ros2 launch <pkg> --show-args` after cloning the AWS repos),
  * the robot model / spawn launch (TurtleBot3 assumed below),
  * the Nav2 map YAML and params files (SLAM or pre-mapped -- your choice),
  * robot namespaces ("alpha"/"beta" match the dataset robot ids).

Build the AWS worlds first (see README.md):
  aws-robomaker-small-warehouse-world-ros2  (warehouse domain)
  aws-robomaker-hospital-world-ros2         (hospital domain)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace

ROBOTS = ["alpha", "beta"]


def generate_launch_description():
    world = LaunchConfiguration("world")
    xml_path = LaunchConfiguration("xml_path")
    waypoints_file = LaunchConfiguration("waypoints_file")

    declare = [
        # Absolute path to the AWS world .world file, e.g.
        # <aws_ws>/install/aws_robomaker_small_warehouse_world/share/
        #   aws_robomaker_small_warehouse_world/worlds/small_warehouse.world
        DeclareLaunchArgument("world", description="Gazebo .world file (AWS warehouse/hospital)"),
        DeclareLaunchArgument("xml_path", description="BTCPP v4 XML file to execute"),
        DeclareLaunchArgument("waypoints_file",
                              description="waypoints_{warehouse,hospital}.yaml"),
    ]

    # -- 1. Gazebo Classic (server + client) with the AWS world ------------
    # The AWS repos ship their own launch files; if you prefer them, replace
    # this IncludeLaunchDescription with theirs, e.g.:
    #   IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(
    #       get_package_share_directory("aws_robomaker_small_warehouse_world"),
    #       "launch", "view_small_warehouse.launch.py")))
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("gazebo_ros"),
                         "launch", "gazebo.launch.py")),
        launch_arguments={"world": world}.items(),
    )

    # -- 2. Per-robot stacks ------------------------------------------------
    # For each robot: spawn a TurtleBot3 in its namespace, then Nav2 bringup
    # in the same namespace. Assumes:
    #   * turtlebot3_gazebo provides a spawn launch accepting x/y/yaw/ns
    #   * you have per-robot nav2 params YAML with use_namespace:=true and
    #     robot_base_frame etc. adjusted (see nav2_bringup multi-robot docs:
    #     bringup_launch.py with namespace + params_file)
    nav2_bringup = os.path.join(
        get_package_share_directory("nav2_bringup"), "launch", "bringup_launch.py")

    robot_groups = []
    spawn_xy = {"alpha": (-1.0, -0.5), "beta": (-1.0, 0.5)}  # adapt to your world
    for r in ROBOTS:
        x, y = spawn_xy[r]
        group = GroupAction([
            PushRosNamespace(r),
            # 2a. spawn the robot model (REPLACE with your robot's spawn launch)
            Node(
                package="gazebo_ros", executable="spawn_entity.py",
                arguments=["-entity", r,
                           "-robot_namespace", r,
                           "-x", str(x), "-y", str(y),
                           # point at a TurtleBot3 model in GAZEBO_MODEL_PATH
                           "-file", os.path.join(
                               get_package_share_directory("turtlebot3_description"),
                               "urdf", "turtlebot3_burger.urdf")],
                output="screen"),
            # 2b. Nav2 in this namespace (provide your own map + params!)
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(nav2_bringup),
                launch_arguments={
                    "use_namespace": "true",
                    # "map": "/path/to/warehouse_map.yaml",     # pre-mapped, or
                    # "slam": "true",                            # run SLAM instead
                    "params_file": f"/path/to/nav2_params_{r}.yaml",
                    "use_sim_time": "true",
                    "autostart": "true",
                }.items()),
        ])
        robot_groups.append(group)

    # -- 3. BT runner --------------------------------------------------------
    # MoveTo leaves map to /<robot>/navigate_to_pose via waypoints YAML;
    # all other leaves are logging stubs until you wire real services.
    runner = Node(
        package="bt_runtime", executable="bt_runner", output="screen",
        parameters=[{
            "mode": "ros",
            "xml_path": xml_path,
            "waypoints_file": waypoints_file,
            "robots": ROBOTS,
            "tick_period": 0.2,
            "use_sim_time": True,
        }])

    return LaunchDescription(
        declare
        + [SetEnvironmentVariable("TURTLEBOT3_MODEL", "burger"),
           gazebo]
        + robot_groups
        + [runner])
