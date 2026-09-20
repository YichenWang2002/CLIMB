from glob import glob

from setuptools import setup

package_name = "bt_runtime"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "pyyaml"],
    zip_safe=True,
    maintainer="CLIMB authors",
    maintainer_email="user@example.com",
    description="Multi-agent BehaviorTree.CPP v4 XML runtime (pure-Python engine + mock/ROS leaves).",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "bt_runner = bt_runtime.runner_node:main",
        ],
    },
)
