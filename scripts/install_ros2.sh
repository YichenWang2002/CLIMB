#!/bin/bash
# Install ROS2 Humble + Gazebo Classic 11 + Nav2 + TurtleBot3 on Ubuntu 22.04 (root).
set -e
export DEBIAN_FRONTEND=noninteractive

echo "=== apt base deps ==="
apt-get update -qq
apt-get install -y -qq curl gnupg lsb-release software-properties-common

echo "=== add ROS2 apt repo ==="
if [ ! -f /usr/share/keyrings/ros-archive-keyring.gpg ]; then
  curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
fi
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" > /etc/apt/sources.list.d/ros2.list
apt-get update -qq

echo "=== install ROS2 Humble + Gazebo Classic + Nav2 + TB3 ==="
apt-get install -y -qq \
  ros-humble-ros-base \
  gazebo \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-gazebo-plugins \
  ros-humble-navigation2 \
  ros-humble-nav2-bringup \
  ros-humble-turtlebot3-gazebo \
  ros-humble-turtlebot3-description \
  ros-humble-turtlebot3-msgs \
  ros-humble-xacro \
  ros-humble-robot-state-publisher \
  ros-humble-behaviortree-cpp-v3 \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-vcstool

echo "=== rosdep init ==="
rosdep init 2>/dev/null || true
rosdep update || true

echo "=== verify ==="
source /opt/ros/humble/setup.bash
ros2 --version
which gzserver gazebo
echo "=== DONE ==="
