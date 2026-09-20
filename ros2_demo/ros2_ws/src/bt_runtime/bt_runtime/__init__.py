"""bt_runtime: pure-Python BehaviorTree.CPP v4 multi-agent BT engine +
mock/ROS leaf layers. engine.py is importable without ROS."""
from .engine import BTEngine, ExecutionError, S, F, R  # noqa: F401
