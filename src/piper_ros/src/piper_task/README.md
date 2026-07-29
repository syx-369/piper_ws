# piper_task

Piper 机械臂正式比赛抓取/放置 ROS1 包。

## 设计基准

`src/piper_task/vision_grasp_core.py` 是
`juesai/grab2016_7_21.py` 的原样副本。目标卡片识别、六类物品检测、
RealSense 深度、手眼矩阵、平滑笛卡尔移动以及抓放偏移均保持不变。

正式节点只负责：

- 将连续测试流程拆成卡片点、三个抓取候选点和三个放置候选点；
- 使用原测试脚本中的取货拍照位、运输归零位和卸货拍照位；
- 保存本轮所抓物品类别，确保卸货时寻找相同类别的置物框图片；
- 发布任务状态和结果，并管理比赛的两轮抓放。
- 与 `waypoint_tools/follow_waypoints.py` 的外部任务协议联动。

原测试脚本不会被此包修改。

## 重要安全约束

机械臂固定姿态和笛卡尔抓放动作来自已经实测的原脚本，但正式联调时仍应保持
急停可用，先空载验证，再测试持物流程。

## 编译

```bash
conda activate piper
cd ~/piper_ws
PYTHONNOUSERSITE=1 catkin build piper_task
source devel/setup.bash
```

## 启动正式任务节点

```bash
roslaunch piper_task piper_task.launch
```

节点启动时会等待 `/end_pose`，随后加载 YOLO 并打开 RealSense。脱离车辆导航
单独联调时，可以依次手动发布：

```bash
rostopic pub -1 /piper_task/command std_msgs/String "data: 'card'"
rostopic pub -1 /piper_task/command std_msgs/String "data: 'pick1'"
rostopic pub -1 /piper_task/command std_msgs/String "data: 'pick2'"
rostopic pub -1 /piper_task/command std_msgs/String "data: 'pick3'"
rostopic pub -1 /piper_task/command std_msgs/String "data: 'place1'"
rostopic pub -1 /piper_task/command std_msgs/String "data: 'place2'"
rostopic pub -1 /piper_task/command std_msgs/String "data: 'place3'"
```

正常比赛不需要手动发布这些命令，车辆 CSV 的七个外部任务会自动触发。其他调试
命令：

```bash
rostopic pub -1 /piper_task/command std_msgs/String "data: 'stow'"
rostopic pub -1 /piper_task/command std_msgs/String "data: 'status'"
rostopic pub -1 /piper_task/command std_msgs/String "data: 'reset'"
```

监看状态：

```bash
rostopic echo /piper_task/state
rostopic echo /piper_task/result
rostopic echo /piper_task/target
```

`reset` 只清除任务轮次和持物状态，不会驱动机械臂运动。

## 仅机械臂功能验证（一条命令启动跟踪与抓放）

`arm_function_test.launch` 只启动普通 CSV 路径跟踪与 `piper_task`：不启动重定位、
避障、红绿灯或完整比赛总控。机械臂驱动、逆运动学以及 `/Odometry` 必须事先存在。

```bash
conda activate piper
source /opt/ros/noetic/setup.bash
source ~/fastlio_ws/devel/setup.bash
source ~/piper_ws/devel/setup.bash

roslaunch piper_task arm_function_test.launch \
  csv_path:=/home/user/fastlio_ws/src/waypoint_tools/data/new_route.csv \
  target_speed:=0.20
```

测试跟踪节点仅保留 `ext:piper_stop_1`～`ext:piper_stop_7`，CSV 中的其他任务标记
全部按普通航点处理。CSV 缺少任意一个机械臂任务点时会拒绝启动。抓取或放置成功
后，本轮剩余候选点会动态改成普通航点，不停车。

## 7 个车辆停靠航点

车辆航点和机械臂关节路径是两类不同数据：

- 车辆航点保存在 `fastlio_ws` 的路线 CSV 中；
- 七点任务映射保存在本包的 `config/task_config.yaml` 中。

录制车辆路线时，可在 7 个停车位置依次写入：

```text
ext:piper_stop_1
ext:piper_stop_2
ext:piper_stop_3
ext:piper_stop_4
ext:piper_stop_5
ext:piper_stop_6
ext:piper_stop_7
```

例如在录制车辆航点时标记第一个点：

```bash
rostopic pub -1 /waypoint_task std_msgs/String "data: 'ext:piper_stop_1'"
```

七个点需要在车辆分别到达对应实际位置时逐条发送，不能连续发送：

```bash
# 点1：裁判目标卡片识别位
rostopic pub -1 /waypoint_task std_msgs/String "data: 'ext:piper_stop_1'"

# 点2～4：三个实物识别/抓取候选位
rostopic pub -1 /waypoint_task std_msgs/String "data: 'ext:piper_stop_2'"
rostopic pub -1 /waypoint_task std_msgs/String "data: 'ext:piper_stop_3'"
rostopic pub -1 /waypoint_task std_msgs/String "data: 'ext:piper_stop_4'"

# 点5～7：三个置物框识别/放置候选位
rostopic pub -1 /waypoint_task std_msgs/String "data: 'ext:piper_stop_5'"
rostopic pub -1 /waypoint_task std_msgs/String "data: 'ext:piper_stop_6'"
rostopic pub -1 /waypoint_task std_msgs/String "data: 'ext:piper_stop_7'"
```

比赛需要两轮抓放时，在返程后的第二组对应位置再次按相同顺序标记这七个任务。

车辆回放到该点后会保持停车，并发布：

```text
start:piper_stop_1:idx<航点序号>
```

机械臂成功完成配置动作后自动回复：

```text
done:piper_stop_1
```

随后车辆才继续。固定流程为：

1. `piper_stop_1`：识别裁判目标卡片并保存类别；
2. `piper_stop_2`：第一个实物识别/抓取候选点；
3. `piper_stop_3`：第二个实物识别/抓取候选点；
4. `piper_stop_4`：第三个实物识别/抓取候选点；
5. `piper_stop_5`：第一个置物框识别/放置候选点；
6. `piper_stop_6`：第二个置物框识别/放置候选点；
7. `piper_stop_7`：第三个置物框识别/放置候选点。

若在某个抓取候选点完成抓取，后续抓取点收到事件时会立即确认，不再执行机械臂
动作；放置区同理。节点同时在 `/piper_task/navigation_skip` 发布无需停车的任务
名，供最终路径跟踪节点在到点前将这些点按普通导航点处理。

若前两个候选点未识别到目标，机械臂回到安全姿态并让车辆继续；第三个候选点仍
未识别到时不发送完成确认，车辆保持停车并报告失败，等待人工安全处理。
