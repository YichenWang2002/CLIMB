# ROS 2 / Gazebo Demo Pack

在本地 Ubuntu（ROS 2 Humble + Gazebo Classic + Nav2 + TurtleBot3）上实际执行
1B 模型生成的多智能体行为树（BTCPP v4 XML），证明可落地。

## 目录结构

```
ros2_demo/
├── xml/                          # 4 个演示 XML（模型生成）+ .meta.json + 3 个人工 prompt
├── ros2_ws/
│   ├── src/bt_runtime/           # 执行引擎（rclpy 包；engine.py 纯 Python）
│   └── tests/mock_smoke.py       # 无 ROS 冒烟测试（使用仓库根目录的 data/ 与 datagen/）
└── scripts/
    ├── gazebo_fault.py           # Gazebo 堵路故障注入 / 清除辅助节点
    └── plot_trace.py             # trace JSON -> 论文用时间线 PNG（纯离线）
```

## 1. 冒烟（无需 ROS/Gazebo，先验证 XML 可执行）

```bash
cd ros2_demo
python3 - <<'EOF'
import json, sys
sys.path.insert(0, "ros2_ws/src/bt_runtime")
from bt_runtime.engine import BTEngine
from bt_runtime.leaves import MockWorld
xml  = open("xml/demo_B_relay_blocked.xml").read()
meta = json.load(open("xml/demo_B_relay_blocked.meta.json"))
world = MockWorld.from_task_meta(meta)
print(BTEngine(xml, world).run())   # {'success': True, 'reason': 'goal_reached', ...}
EOF
```

（`ros2_ws/tests/mock_smoke.py` 会自动定位仓库根目录的 `data/train.jsonl` 与
`datagen/` 符号执行器，在仓库内任何位置克隆后均可直接运行：
`python3 ros2_ws/tests/mock_smoke.py`。）

## 2. Mock 模式跑 runner（需 ROS 2，无需 Gazebo/Nav2）

```bash
cd ros2_demo/ros2_ws && colcon build && source install/setup.bash
ros2 run bt_runtime bt_runner --ros-args \
  -p xml_path:=$PWD/../xml/demo_B_relay_blocked.xml \
  -p mode:=mock \
  -p task_meta_json:=$PWD/../xml/demo_B_relay_blocked.meta.json \
  -p trace_json:=/tmp/trace_demo_B.json
python3 ../scripts/plot_trace.py /tmp/trace_demo_B.json -o trace_demo_B.png
```

`trace_json` 参数（本次新增，默认空=关闭）在结束时把每机器人轨迹事件落盘 JSON，
格式：`{success, reason, ticks, recoveries, trajectory: {agent: [{tick, kind, label, status}]}}`。

## 3. ROS 模式（Gazebo + Nav2 实机执行）

`ros2_ws/src/bt_runtime/launch/bt_demo.launch.py` 是**注释模板**，按文件头说明改：
AWS world 包 launch 名、TurtleBot3 spawn、每机器人 Nav2 params、地图。
MoveTo → `/<robot>/navigate_to_pose`（Nav2），位姿查 waypoints YAML；
其余叶子是 logging stub（见 `leaves.py: RosWorld`）。

```bash
ros2 launch bt_runtime bt_demo.launch.py \
  world:=/path/to/small_warehouse.world \
  xml_path:=<demo_pack>/xml/demo_B_relay_blocked.xml \
  waypoints_file:=<bt_runtime>/config/waypoints_warehouse.yaml
```

## 4. 堵路→ClearPath 恢复演示（scripts/gazebo_fault.py）

rclpy 节点：监视触发机器人里程计，机器人接近被堵边中点时经 gazebo_ros
`/spawn_entity` 在边中点生成箱子（Nav2 的 MoveTo 因此失败 → BT Fallback →
ClearPath）；收到 `/fault_cleared`（std_msgs/Empty）后经 `/delete_entity` 删箱。

### 接口

- **订阅** `/<robot>/odom`（nav_msgs/Odometry，默认 `/alpha/odom`）；
  `clear_topic`（std_msgs/Empty，默认 `/fault_cleared`）。
- **服务客户端** `/spawn_entity`、`/delete_entity`（gazebo_msgs）。
- **参数**：
  - `waypoints_file` + `edge_from`/`edge_to`（位置 id，从 YAML 算中点）
    **或**直接 `x1,y1,x2,y2`（无 waypoints_file 时生效）；
  - `robot`（alpha）、`odom_topic`、`trigger_distance`（1.0 m）；
  - `box_name`（fault_box）、`box_size`（0.6）；
  - `spawn_source`：`inline`（内联 SDF 红箱）| `model`（用 `model_uri`，
    默认 `model://cardboard_box`）；`frame_id`（world）。

例（demo_B 的 charge_bay↔shelf_b 边）：

```bash
ros2 run <pkg> gazebo_fault.py --ros-args \
  -p waypoints_file:=<bt_runtime>/config/waypoints_warehouse.yaml \
  -p edge_from:=charge_bay -p edge_to:=shelf_b -p robot:=alpha
```

### ClearPath 挂接（不要改 leaves.py）

`RosWorld.run_simple` 会查 `world.stub_handlers["ClearPath"]`。在你的 bringup
脚本里（拿到 runner 节点后）注册一个发布 `/fault_cleared` 的回调即可：

```python
from std_msgs.msg import Empty
clear_pub = runner_node.create_publisher(Empty, "/fault_cleared", 10)

def clear_path_stub(attrs, agent, node):
    clear_pub.publish(Empty())
    return "SUCCESS"

runner_node.world.stub_handlers["ClearPath"] = clear_path_stub
```

若要把它做成常驻节点而非内嵌，也可以让 bt_demo launch 里多起一个
wrapper node 做同样的事。

## 5. 画时间线（scripts/plot_trace.py）

```bash
python3 scripts/plot_trace.py TRACE.json -o out.png [--show-conditions] [--title T]
python3 scripts/plot_trace.py --demo -o demo.png --demo-json demo.json   # 自测
```

横轴 BT tick，纵轴每机器人一条泳道；MoveTo/PickUp/PlaceDown/Handover/Co*/
ClearPath/Recharge 有各自 marker，失败事件红色空心，rendezvous
（Handover/Co* 完成、SignalReady/WaitReady 屏障）画紫色虚线。
只依赖 matplotlib，无需 ROS。

## 本地（用户机器）待办清单

1. **依赖安装**：`sudo apt install ros-humble-gazebo-ros ros-humble-gazebo-msgs
   ros-humble-nav2-bringup ros-humble-turtlebot3-gazebo python3-yaml python3-matplotlib`；
   `pip` 侧无需额外包。
2. **AWS world 仓库**：clone 并 colcon build
   `aws-robomaker-small-warehouse-world-ros2`（warehouse 域）/
   `aws-robomaker-hospital-world-ros2`（hospital 域），确认 GAZEBO_MODEL_PATH。
3. **waypoint 标定**：`config/waypoints_*.yaml` 全部是占位坐标（文件头有警告），
   必须在自己的 Gazebo 里用 rviz "Publish Point" 逐点重测后替换，
   否则 Nav2 目标点会落在货架/墙里。
4. **launch 模板落地**：改 `launch/bt_demo.launch.py` 中 AWS world launch 名、
   TurtleBot3 spawn 方式、每机器人 nav2 params YAML（use_namespace + 帧名）、
   地图（先 SLAM 建图或用现成 map yaml）；alpha/beta 出生点避开货架。
5. **ClearPath 接线**：按第 4 节注册 stub handler（PickUp/PlaceDown/Handover
   目前只是 logging stub，真实抓取/交接需自行接 service/action；
   条件叶子 IsAtLocation 等默认乐观返回 True，是占位实现——见 leaves.py 注释）。
6. **定位**：多机器人各自 AMCL/SLAM 的 initial pose 要在 map 里设好，
   否则 /alpha/odom→map 链路不通，gazebo_fault 的 odom 触发也依赖它。
7. **use_sim_time**：runner 与 gazebo_fault 都建议 `-p use_sim_time:=true`。
