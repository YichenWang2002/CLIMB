# mBT ROS2 仿真演示 —— 本地执行指南（给本地 Kimi 的操作指令）

## 你是谁、要做什么

你在用户的本地 Ubuntu 机器上工作（ROS2 Humble + Gazebo Classic 已装）。你的任务是：**把我们 1B 模型（ours_v3）从自然语言生成的多智能体行为树 XML，在 Gazebo 仿真里真实执行出来**，证明模型输出可落地运行。这是 ICLR 论文的演示材料。

## 最高原则（违反即演示作废）

1. `xml/` 目录下的所有 `.xml` 文件是**模型推理的原样输出**，已经过符号执行器验证（goal_reached）。**绝对不许修改、格式化、"修复"这些 XML 的任何字节**。遇到执行问题，检查的是你的环境/配置，不是 XML。
2. 执行引擎的语义以 `ros2_ws/src/bt_runtime/bt_runtime/engine.py` 为准，它已与数据生成侧的执行器 300 条样本交叉验证零不一致。**不许改 engine.py 的语义**；发现疑似 bug 时停下来报告，不要自行"修正"。
3. 演示目标：每个 demo 跑完得到 `success=True / goal_reached` 的执行轨迹 + 录屏视频。

## 文件清单

```
demo_pack/
  prompts.md                  本文件
  xml/
    demo_A_relay.xml          【场景1】双机接力 Handover（warehouse，无故障）
    demo_B_relay_blocked.xml  【场景2】接力 + 走廊堵塞→ClearPath 恢复（warehouse）
    demo_C_heavy_blocked.xml  【场景3】双机协同搬运 CoPickUp/CoMoveTo + 堵塞恢复（warehouse）
    demo_D_relay_battery.xml  【场景4】接力 + 低电量→Recharge 恢复（warehouse）
    *.meta.json               每个 XML 对应的任务结构化描述（仅供 mock 模式/故障注入参数用，不给模型）
    human_*.prompt.txt        3 条人工撰写的自然语言任务（Phase 3 用）
    human_*.xml               对应的模型真实推理输出（执行器已验证 goal_reached）
  ros2_ws/src/bt_runtime/     BT 执行运行时（rclpy 包）
    bt_runtime/engine.py      BTCPP v4 tick 引擎（纯 Python，无 ROS 依赖）
    bt_runtime/leaves.py      叶节点层：MockWorld（无 Gazebo 自测）/ RosWorld（Nav2）
    bt_runtime/runner_node.py 执行节点：加载 XML 驱动全部机器人树，trace_json 参数可落盘轨迹
    config/waypoints_{warehouse,hospital}.yaml   地点 id → 世界坐标（**占位值，必须标定**）
    launch/bt_demo.launch.py  参考模板（需按本机实际改）
  scripts/
    gazebo_fault.py           故障注入节点：机器人接近被堵走廊时在 Gazebo 动态生成障碍箱；
                              收到 /fault_cleared（std_msgs/Empty）后删除（对应 ClearPath 动作）
    plot_trace.py             离线画论文用执行轨迹图（trace JSON → PNG，无需 ROS）
  README.md                   包内补充说明
```

另外需要从服务器下载：**AWS 仿真世界仓库两个目录**（`aws-robomaker-small-warehouse-world-ros2`、`aws-robomaker-hospital-world-ros2`），它们是 Gazebo 世界模型。

## 前置依赖

```bash
sudo apt install -y ros-humble-gazebo-ros ros-humble-gazebo-msgs \
  ros-humble-nav2-bringup ros-humble-turtlebot3-gazebo ros-humble-navigation2
pip3 install pyyaml matplotlib
```

## 步骤

### 1. 构建（10 分钟）
```bash
cd demo_pack/ros2_ws && colcon build --packages-select bt_runtime
source install/setup.bash   # 每个新终端都要 source，另加 /opt/ros/humble
```
构建 AWS 世界仓库（按其各自 README），并设置 `GAZEBO_MODEL_PATH` 指向其模型目录。

### 2. mock 模式自测（5 分钟，不开 Gazebo）
```bash
python3 ros2_ws/tests/mock_smoke.py   # 注意先改文件里 PIPELINE 常量为本机路径
ros2 run bt_runtime bt_runner --ros-args \
  -p xml_path:=$PWD/../xml/demo_B_relay_blocked.xml \
  -p task_meta_json:=$PWD/../xml/demo_B_relay_blocked.meta.json -p mode:=mock
```
预期：`success=True, goal_reached`，且能看到 ClearPath 恢复事件。不通 = 环境问题，先修这里。

### 3. waypoint 标定（30-60 分钟，唯一的手工活）
`config/waypoints_warehouse.yaml`（和 hospital）里是**占位坐标**，必须替换为 AWS 世界里的真实点位：
- 启动 AWS warehouse 世界，用 rviz 的 "Publish Point" 在 6 个地点逐点读数（x, y, yaw）：`shelf_a / shelf_b / shelf_c / packing_station / loading_dock / charge_bay`（warehouse）；hospital 为 `pharmacy / ward_east / ward_west / lab / nurses_station / charge_nook`。
- 选点原则：开阔、Nav2 可通行、离货架/墙体 ≥0.5m；同一地点两台机器人都要能停（必要时给会合点留白）。
- 替换 yaml 里对应 id 的 x/y/yaw，**不要改 id 名**。

### 4. 双机器人 + Nav2 启动（按 launch 模板落地）
两台 TurtleBot3，命名空间 `alpha`、`beta`，各自独立 Nav2 栈（注意每台的 map/base/odom 帧名前缀、初始位姿与 AMCL）。所有节点 `-p use_sim_time:=true`。`launch/bt_demo.launch.py` 只是模板，按本机实际的 AWS world launch 文件名、spawn 位置改写。

### 5. 跑四个场景（每个流程相同）
```bash
ros2 run bt_runtime bt_runner --ros-args -p use_sim_time:=true \
  -p xml_path:=<场景.xml> -p waypoints_file:=<标定后的warehouse yaml> \
  -p robots:=alpha,beta -p trace_json:=/tmp/trace_<场景>.json
```
- **场景1 demo_A**：直接跑。看点：两机装载码头会合，SignalReady/WaitReady 同步后 Beta 接力。
- **场景4 demo_D**：直接跑。看点：电量预算耗尽 → 机器人去充电点 Recharge → 继续任务。
- **场景2/3 demo_B/C**：先启动故障注入再跑 runner：
  ```bash
  ros2 run bt_runtime gazebo_fault --ros-args -p use_sim_time:=true \
    -p waypoints_file:=<warehouse yaml> -p edge_from:=charge_bay -p edge_to:=shelf_b -p robot:=alpha
  ```
  并在 runner 启动前按 README 第 4 节注册 ClearPath → 发布 `/fault_cleared` 的 stub 回调。
  看点：机器人走向被堵走廊 → Gazebo 里出现障碍箱 → 导航失败 → **Fallback 触发 ClearPath（箱子消失）→ 重试成功**。这是论文最重要的 30 秒。
  （注意 demo_B 的堵塞边是 charge_bay↔shelf_b，以各自 .meta.json 里的 faults 字段为准。）
- PickUp/PlaceDown/Handover 是日志 stub（论文里如实写"导航级执行、操作符号化"），不用管机械动作。

### 6. 出图与录屏
```bash
python3 scripts/plot_trace.py /tmp/trace_demo_B.json -o trace_demo_B.png
```
录屏：Gazebo 画面 + rviz 双窗口，每场景 20-40 秒（场景1 接力、场景2 或 3 堵路恢复、场景4 充电恢复），拼成 1-2 分钟视频。

### 7. Phase 3（人工 prompt 端到端）
`xml/human_*.prompt.txt` 是人工撰写的任务描述，`human_*.xml` 是模型对它们的真实推理输出（服务器侧已用执行器验证 goal_reached，md5 留档备查）。按步骤 5 同样执行，重点强调"这是随手写的人话 → 模型生成 → 直接跑"。human_3 用 hospital 世界。

## 常见问题

- **导航卡住不动**：先查 use_sim_time 是否全链路一致；再查 AMCL 初始位姿；再查 waypoint 是否落在不可通行区。
- **会合不同步**：两机器人必须都到达会合 waypoint 附近 SignalReady/WaitReady 才配对成功；检查两机的会合点坐标是否一致。
- **故障箱不出现**：确认 `/spawn_entity` 服务存在（gazebo_ros 已启动）、odom topic 名与实际命名空间一致。
- **ClearPath 不删箱**：检查 stub 回调是否注册、`/fault_cleared` topic 名两端一致。

## 验收标准（全部达成才算完成）

1. 场景 1/2/3/4 均 `success=True / goal_reached`，trace JSON 落盘；
2. 场景 2 或 3 的 trace 里能看到 `MoveTo:F → ClearPath:S → MoveTo:S` 恢复序列，场景 4 能看到 Recharge；
3. 每场景一张轨迹 PNG + 每场景一段录屏；
4. 全部 XML 与服务器侧原件逐字节一致（`md5sum` 对比）。

完成后向用户汇报：各场景结果、trace/PNG/视频路径、遇到的任何 XML 执行异常（如有，原样上报，不要自行改 XML）。
