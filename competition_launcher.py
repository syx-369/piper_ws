#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""比赛启动器：在独立终端启动 ROS 服务，并显示启动状态。"""

import base64
import csv
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QCheckBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QStackedWidget, QVBoxLayout, QWidget


STATUS_DIR = Path("/tmp/competition_launcher_status")
STARTUP_GRACE_SECONDS = 5
# 避障区实测起始位姿：车辆停在该位置、朝向附近后可直接启动重定位。
# CSV 仅用于生成 avoid_start..avoid_end 的短测试路线，不参与重定位初值。
AVOIDANCE_TEST_INITIAL_POS = (2.406303099617625, -26.38992998139038, -0.16290923251009404)
AVOIDANCE_TEST_INITIAL_ROT = (-0.007025715288537381, 0.007590029144768787, 0.7048832757145482, 0.7092479103953668)

# 仅用于独立避障测试的完整可覆盖参数集合。未写在这里的任务/机械臂参数不会影响
# 这条没有 ext 任务的短路线。每次启动都会生成临时 YAML，绝不改比赛配置文件。
AVOIDANCE_TEST_PARAM_TEMPLATE = """# 每行格式：参数名: 值。# 开头的行是注释。
# 速度与跟踪
max_linear: 1.00
max_angular: 0.80
max_linear_accel: 0.25
max_linear_decel: 1.00
max_angular_accel: 0.60
min_tracking_speed: 0.08
control_rate: 20.0
lookahead_distance: 0.80
min_lookahead: 0.60
max_lookahead: 1.80
lookahead_speed_ratio: 1.80
execute_points: 7
reference_horizon: 6.0
reference_spacing: 0.15
reference_smoothing_window: 7
heading_slow_angle: 0.35
heading_hard_slow_angle: 0.55
rotate_in_place_angle: 0.70
rotate_k_angular: 1.20
final_approach_dist: 0.80
goal_tolerance_default: 0.30
# A* 规划与车体模型
planner_enabled: true
zone_only: true
require_zone_markers: true
grid_resolution: 0.10
max_grid_cells: 16000
xy_margin: 2.00
footprint_front: 0.45
footprint_rear: 0.45
footprint_half_width: 0.35
start_clear_padding: 0.06
inflation_radius: 0.65
min_clearance: 0.12
goal_clear_radius: 0.25
goal_search_radius: 1.20
planning_goal_index_ahead: 12
planning_goal_min_dist: 1.20
planning_goal_min_forward: 0.20
plan_goal_max_dist: 4.00
skip_behind_waypoint_x: -0.25
skip_behind_waypoint_dist: 1.00
allow_behind_target_dist: 0.80
local_goal_tolerance: 0.20
local_waypoint_spacing: 0.30
min_valid_plan_dist: 0.60
min_valid_plan_points: 2
min_forward_target: -0.05
detour_forward_samples: [1.20, 2.00, 3.00]
detour_lateral_offsets: [-1.20, -0.90, -0.60, -0.30, 0.30, 0.60, 0.90, 1.20]
replan_min_interval: 0.20
blocked_replan_delay: 0.30
detour_lock_time: 4.00
blocked_rotate_angle: 0.35
escape_angular: 0.35
# 点云、障碍与安全层
obstacle_timeout: 0.60
max_obstacle_points: 2500
cloud_stride: 2
cloud_keep_radius: 6.00
scan_min_range: 0.15
scan_max_range: 6.00
obstacle_min_z: -1.20
obstacle_max_z: 0.60
front_half_width: 0.42
side_safety_radius: 0.25
safety_slow_dist: 0.70
safety_stop_dist: 0.28
safety_emergency_dist: 0.16
near_obstacle_angular: 0.35
# 避障区进入、退出与进度保护
zone_entry_margin: 0.05
zone_exit_margin: 0.00
zone_min_valid_plan_dist: 0.25
zone_handoff_distance: 0.60
zone_handoff_early_release: 0.60
zone_handoff_lateral_tolerance: 0.10
progress_search_ahead: 12
progress_search_back: 10
progress_lateral_limit: 3.00
progress_heading_tolerance: 0.70
progress_pass_margin: 0.08
progress_max_advance: 1.00
progress_zone_search_ahead: 20
progress_zone_max_advance: 3.60
progress_zone_confirm_samples: 3
progress_zone_confirm_tolerance: 0.30
progress_recovery_max_advance: 1.80
progress_recovery_samples: 3
progress_recovery_tolerance: 0.30
progress_recovery_speed: 0.25
odom_jump_threshold: 0.80
odom_recovery_samples: 5
"""


@dataclass(frozen=True)
class LaunchItem:
    name: str
    command: str
    persistent: bool = True
    accept_successful_exit: bool = False


ITEMS = [
    LaunchItem("1. ROS Core", "source /opt/ros/noetic/setup.bash && (rosnode list >/dev/null 2>&1 && echo '检测到已有 ROS Master，复用现有服务。' || roscore)", True, True),
    LaunchItem("2. 配置 CAN", "cd /home/user/fastlio_ws && ./start_can.sh", False),
    LaunchItem("3. Bunker 底盘", "cd ~/bunker_ws && source devel/setup.bash && roslaunch bunker_bringup bunker_robot_base.launch publish_tf:=false"),
    LaunchItem("4. Piper 底层", "cd ~/piper_ws && source /home/user/miniconda3/etc/profile.d/conda.sh && conda activate piper && source devel/setup.bash && roslaunch piper start_single_piper.launch can_port:=can1 auto_enable:=true"),
    LaunchItem("5. Piper 运动学", "cd ~/piper_ws && source /home/user/miniconda3/etc/profile.d/conda.sh && conda activate piper && source devel/setup.bash && python ~/piper_ws/src/piper_ros/src/piper/scripts/piper_pinocchio/piper_pinocchio.py"),
    LaunchItem("6. 机械臂任务节点", "source /opt/ros/noetic/setup.bash && source /home/user/miniconda3/etc/profile.d/conda.sh && conda activate piper && source /home/user/fastlio_ws/devel/setup.bash && source /home/user/piper_ws/devel/setup.bash && roslaunch piper_task piper_task.launch enable_camera_relay:=true"),
    LaunchItem("7. MID360 雷达", "cd ~/livox_ws && source devel/setup.bash && roslaunch livox_ros_driver2 msg_MID360.launch"),
    LaunchItem("8. S-FAST_LIO 重定位", "cd ~/fastlio_ws && source devel/setup.bash && source ~/livox_ws/devel/setup.bash --extend && roslaunch sfast_lio mapping_mid360_relocalization.launch rviz:=true"),
    LaunchItem("9. 比赛总控", "source /opt/ros/noetic/setup.bash && source /home/user/fastlio_ws/devel/setup.bash && RACE_CSV=/home/user/fastlio_ws/src/waypoint_tools/data/{route} && CRUISE_SPEED={speed} && POST_PLACE_SPEED={post_place_speed} && AVOIDANCE_SPEED={avoid_speed} && roslaunch final_mission final_race.launch csv_path:=${RACE_CSV} target_speed:=${CRUISE_SPEED} post_place_speed:=${POST_PLACE_SPEED} avoidance_speed:=${AVOIDANCE_SPEED} wait_for_start:=true auto_start:=false enable_vision:=true show_image:=true"),
]


RUNNER = r'''import base64,json,os,subprocess,sys,time
status_file, encoded, persistent, accept_successful_exit, gate_file = sys.argv[1], sys.argv[2], sys.argv[3] == "1", sys.argv[4] == "1", sys.argv[5]
command = base64.b64decode(encoded).decode("utf-8")
def report(state, detail=""):
    os.makedirs(os.path.dirname(status_file), exist_ok=True)
    tmp = status_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f: json.dump({"state":state,"detail":detail,"time":time.time()}, f)
    os.replace(tmp, status_file)
report("waiting", "标签页已就绪，等待启动指令")
while True:
    while not os.path.exists(gate_file): time.sleep(0.1)
    os.unlink(gate_file)
    report("starting")
    # 实际启动命令必须使用非交互 shell：否则命令失败后 bash 仍驻留，
    # 会被错误判断为服务仍在运行。
    child = subprocess.Popen(["bash", "-lc", command])
    if persistent:
        time.sleep(5)
        if child.poll() is None:
            report("success", "进程已稳定运行")
            code = child.wait()
            report("stopped" if code == 0 else "failed", "进程已退出，返回码 %s" % code)
        else:
            if child.returncode == 0 and accept_successful_exit:
                report("success", "检测到并复用了已有服务")
            else:
                report("failed", "启动进程提前退出，返回码 %s" % child.returncode)
    else:
        code = child.wait()
        report("success" if code == 0 else "failed", "CAN 配置完成" if code == 0 else "返回码 %s" % code)
    print("\n启动器状态：本次执行结束，标签页会保留并等待下次启动。")
    # 保留本次成功/失败状态，供界面轮询和一键启动继续处理；
    # 下次收到 gate 时会直接覆盖为 starting。
'''


class Launcher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("比赛启动器")
        self.resize(650, 520)
        self.buttons = []
        self.status_files = []
        self.gate_files = []
        self.terminal_window_created = False
        self.minimize_attempts = 0
        self.batch_index = None
        self.batch_waiting = None
        STATUS_DIR.mkdir(parents=True, exist_ok=True)

        root = QWidget()
        layout = QVBoxLayout(root)
        hint = QLabel("绿色 = 命令成功完成 / 服务进程稳定运行（等待 5 秒）。所有命令会在同一终端窗口的不同标签页中运行。")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        tuning = QHBoxLayout()
        tuning.addWidget(QLabel("测试速度："))
        self.avoid_speed_input = QDoubleSpinBox()
        self.avoid_speed_input.setRange(0.03, 1.50)
        self.avoid_speed_input.setSingleStep(0.05)
        self.avoid_speed_input.setValue(0.20)
        self.avoid_speed_input.setSuffix(" m/s")
        tuning.addWidget(self.avoid_speed_input)
        tuning.addWidget(QLabel("障碍膨胀："))
        self.avoid_inflation_input = QDoubleSpinBox()
        self.avoid_inflation_input.setRange(0.10, 1.50)
        self.avoid_inflation_input.setSingleStep(0.05)
        self.avoid_inflation_input.setValue(0.65)
        self.avoid_inflation_input.setSuffix(" m")
        tuning.addWidget(self.avoid_inflation_input)
        reset_avoid_params = QPushButton("恢复避障默认参数")
        reset_avoid_params.clicked.connect(self.reset_avoidance_params)
        tuning.addWidget(reset_avoid_params)
        tuning.addStretch()
        layout.addLayout(tuning)
        self.avoid_params_input = QPlainTextEdit()
        self.avoid_params_input.setPlainText(AVOIDANCE_TEST_PARAM_TEMPLATE)
        self.avoid_params_input.setPlaceholderText("YAML 参数覆盖")
        self.avoid_params_input.setToolTip("仅影响下一次独立避障测试。参数名必须来自默认列表，避免误改 ROS 话题或任务流程。")
        self.avoid_params_input.setMaximumHeight(145)
        layout.addWidget(QLabel("高级避障参数覆盖（可滚动编辑，速度和膨胀半径以上方控件为准）："))
        layout.addWidget(self.avoid_params_input)
        self.all_button = QPushButton("一键顺序启动全部比赛服务")
        self.all_button.setMinimumHeight(48)
        self.all_button.clicked.connect(self.start_all)
        speed_row = QHBoxLayout()
        speed_row.addWidget(self.all_button)
        speed_row.addWidget(QLabel("比赛速度："))
        self.speed_input = QDoubleSpinBox()
        self.speed_input.setRange(0.05, 2.00)
        self.speed_input.setSingleStep(0.05)
        self.speed_input.setDecimals(2)
        self.speed_input.setValue(1)
        self.speed_input.setSuffix(" m/s")
        self.speed_input.setToolTip("仅用于第 9 项“比赛总控”的 target_speed")
        speed_row.addWidget(self.speed_input)
        speed_row.addWidget(QLabel("避障区速度："))
        self.race_avoid_speed_input = QDoubleSpinBox()
        self.race_avoid_speed_input.setRange(0.08, 2.00)
        self.race_avoid_speed_input.setSingleStep(0.05)
        self.race_avoid_speed_input.setDecimals(2)
        self.race_avoid_speed_input.setValue(0.70)
        self.race_avoid_speed_input.setSuffix(" m/s")
        self.race_avoid_speed_input.setToolTip(
            "仅在 CSV 的 avoid_start 到 avoid_end 避障区间生效；其余路段仍使用比赛速度"
        )
        speed_row.addWidget(self.race_avoid_speed_input)
        speed_row.addWidget(QLabel("路径文件："))
        self.race_file_input = QLineEdit("final08.csv")
        self.race_file_input.setPlaceholderText("例如 final03.csv")
        self.race_file_input.setToolTip("第 9 项总控使用的 CSV，位于 waypoint_tools/data/")
        speed_row.addWidget(self.race_file_input)
        layout.addLayout(speed_row)
        post_place_speed_row = QHBoxLayout()
        post_place_speed_row.addWidget(QLabel("放置完成 → 避障区起点速度："))
        self.post_place_speed_input = QDoubleSpinBox()
        self.post_place_speed_input.setRange(0.08, 2.00)
        self.post_place_speed_input.setSingleStep(0.05)
        self.post_place_speed_input.setDecimals(2)
        self.post_place_speed_input.setValue(0.70)
        self.post_place_speed_input.setSuffix(" m/s")
        self.post_place_speed_input.setToolTip(
            "piper_stop_7 任务完成后生效，到下一处 avoid_start 为止"
        )
        post_place_speed_row.addWidget(self.post_place_speed_input)
        post_place_speed_row.addStretch()
        layout.addLayout(post_place_speed_row)
        grid = QGridLayout()
        for i, item in enumerate(ITEMS):
            button = QPushButton(item.name)
            button.setMinimumHeight(42)
            button.clicked.connect(lambda checked=False, n=i: self.start_one(n))
            self.buttons.append(button)
            status_file = STATUS_DIR / ("item_%d.json" % i)
            # 状态文件只用于本次界面运行；清除上一次遗留的绿色/红色状态。
            status_file.unlink(missing_ok=True)
            self.status_files.append(status_file)
            gate_file = STATUS_DIR / ("item_%d.start" % i)
            gate_file.unlink(missing_ok=True)
            self.gate_files.append(gate_file)
            grid.addWidget(button, i // 2, i % 2)
        layout.addLayout(grid)
        self.message = QLabel("就绪")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        navigation = QHBoxLayout()
        navigation.addStretch()
        debug_button = QPushButton("进入调试界面 →")
        debug_button.clicked.connect(self.show_debug_page)
        navigation.addWidget(debug_button)
        test_button = QPushButton("单独任务测试 →")
        test_button.clicked.connect(lambda: self.pages.setCurrentIndex(2))
        navigation.addWidget(test_button)
        layout.addLayout(navigation)

        self.pages = QStackedWidget()
        self.pages.addWidget(root)
        self.pages.addWidget(self.create_debug_page())
        self.pages.addWidget(self.create_single_test_page())
        self.setCentralWidget(self.pages)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(350)

    def create_debug_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel("调试界面 / 录制命令")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)
        hint = QLabel("录制航迹点会持续运行，完成时请在对应终端按 Ctrl+C；检查按钮使用比赛路线文件。")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("保存航迹文件名："))
        self.route_file_input = QLineEdit("final03.csv")
        self.route_file_input.setPlaceholderText("例如 final03.csv")
        self.route_file_input.setToolTip("文件会保存到 ~/fastlio_ws/src/waypoint_tools/data/")
        file_row.addWidget(self.route_file_input)
        layout.addLayout(file_row)

        commands = [
            ("开始录制航迹点", self.record_waypoints),
            ("启动红绿灯检测", lambda: self.run_piper_task_command("启动红绿灯检测", "rosrun piper_task traffic_light_test.py")),
            ("记录红绿灯检测点", lambda: self.publish_waypoint_task("traffic_light")),
            ("启动抓取识别测试", lambda: self.run_piper_task_command("启动抓取识别测试", "rosrun piper_task pick_object_depth_ir_test.py")),
            ("抓取测试（关闭视觉）", lambda: self.run_piper_task_command("抓取测试（关闭视觉）", "rosrun piper_task pick_object_depth_ir_test.py _enable_vision:=false")),
            ("点1：目标卡片识别位置", lambda: self.publish_waypoint_task("piper_stop_1")),
            ("点2：第一个抓取识别位置", lambda: self.publish_waypoint_task("piper_stop_2")),
            ("点3：第二个抓取识别位置", lambda: self.publish_waypoint_task("piper_stop_3")),
            ("点4：第三个抓取识别位置", lambda: self.publish_waypoint_task("piper_stop_4")),
            ("启动放置识别测试", lambda: self.run_piper_task_command("启动放置识别测试", "rosrun piper_task place_bottle_depth_ir_test.py")),
            ("放置测试（关闭视觉）", lambda: self.run_piper_task_command("放置测试（关闭视觉）", "rosrun piper_task place_bottle_depth_ir_test.py _enable_vision:=false")),
            ("点5：第一个放置识别位置", lambda: self.publish_waypoint_task("piper_stop_5")),
            ("点6：第二个放置识别位置", lambda: self.publish_waypoint_task("piper_stop_6")),
            ("点7：第三个放置识别位置", lambda: self.publish_waypoint_task("piper_stop_7")),
            ("记录避障区起点", lambda: self.publish_waypoint_task("avoid_start", external=False)),
            ("记录避障区终点", lambda: self.publish_waypoint_task("avoid_end", external=False)),
            ("设置最后点为终点", self.set_last_waypoint_as_finish),
            ("一键地图轨迹检查", self.plot_route_on_map),
        ]
        grid = QGridLayout()
        for i, (name, callback) in enumerate(commands):
            button = QPushButton(name)
            button.setMinimumHeight(48)
            button.clicked.connect(lambda checked=False, c=callback: c())
            grid.addWidget(button, i // 2, i % 2)
        layout.addLayout(grid)
        self.debug_message = QLabel("就绪")
        layout.addWidget(self.debug_message)
        layout.addStretch()
        back = QPushButton("← 返回比赛启动界面")
        back.clicked.connect(lambda: self.pages.setCurrentIndex(0))
        layout.addWidget(back)
        return page

    def show_debug_page(self):
        self.pages.setCurrentIndex(1)

    def create_single_test_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel("单独任务测试")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)
        hint = QLabel("勾选要测试的任务。未勾选的外部任务会写入临时航迹并改为普通巡航点，以 1.00 m/s 通过；避障区和终点始终保留。")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.task_checkboxes = {}
        tasks = [("traffic_light", "红绿灯检测点")] + [("piper_stop_%d" % i, "机械臂任务点 %d" % i) for i in range(1, 8)]
        grid = QGridLayout()
        for i, (task, label) in enumerate(tasks):
            check = QCheckBox(label)
            self.task_checkboxes[task] = check
            grid.addWidget(check, i // 2, i % 2)
        layout.addLayout(grid)
        actions = QHBoxLayout()
        all_button = QPushButton("全选")
        all_button.clicked.connect(lambda: [box.setChecked(True) for box in self.task_checkboxes.values()])
        none_button = QPushButton("全不选")
        none_button.clicked.connect(lambda: [box.setChecked(False) for box in self.task_checkboxes.values()])
        start_button = QPushButton("以 1.00 m/s 启动单独测试")
        start_button.clicked.connect(self.start_single_task_test)
        avoid_button = QPushButton("避障区独立测试（自动重定位）")
        avoid_button.clicked.connect(self.start_avoidance_test)
        avoid_confirm_button = QPushButton("确认定位后，开始避障")
        avoid_confirm_button.clicked.connect(self.enable_avoidance_tracking)
        actions.addWidget(all_button)
        actions.addWidget(none_button)
        actions.addWidget(start_button)
        actions.addWidget(avoid_button)
        actions.addWidget(avoid_confirm_button)
        layout.addLayout(actions)
        self.single_test_message = QLabel(
            "路径文件使用主界面的“路径文件”输入框。避障区独立测试会截取首个 "
            "avoid_start/avoid_end 区间前后短路线，并使用已实测保存的 S-FAST-LIO 初始位姿。"
            "启动后车辆保持静止，确认 RViz 定位正确后再点击“确认定位后，开始避障”。"
        )
        self.single_test_message.setWordWrap(True)
        layout.addWidget(self.single_test_message)
        layout.addStretch()
        back = QPushButton("← 返回比赛启动界面")
        back.clicked.connect(lambda: self.pages.setCurrentIndex(0))
        layout.addWidget(back)
        return page

    def start_single_task_test(self):
        filename = self.race_file_name(show_error=True)
        if not filename:
            return
        source = Path("/home/user/fastlio_ws/src/waypoint_tools/data") / filename
        selected = {name for name, box in self.task_checkboxes.items() if box.isChecked()}
        if QMessageBox.question(self, "确认单独测试", "将以 1.00 m/s 自动启动；未选外部任务会直接通过。确认周边安全后继续？", QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            with source.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fields, rows = reader.fieldnames, list(reader)
            if not fields or "task" not in fields:
                raise ValueError("CSV 缺少 task 列")
            for row in rows:
                task = (row.get("task") or "").strip()
                name = task[4:] if task.startswith("ext:") else task
                if task.startswith("ext:") and name != "finish" and name not in selected:
                    row["task"] = "none"
            test_path = STATUS_DIR / ("single_test_%d.csv" % int(time.time()))
            with test_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "创建测试航迹失败", str(error))
            return
        command = "source /opt/ros/noetic/setup.bash && source /home/user/fastlio_ws/devel/setup.bash && roslaunch final_mission final_race.launch csv_path:=%s target_speed:=1.00 wait_for_start:=false auto_start:=true enable_vision:=true show_image:=true" % shlex.quote(str(test_path))
        self.run_debug_command("单独任务测试", command)
        names = "、".join(sorted(selected)) if selected else "无（纯巡航）"
        self.single_test_message.setText("已生成 %s；测试任务：%s" % (test_path.name, names))

    @staticmethod
    def _task_name(row):
        """返回规范化任务名；避障标记不带 ext:，仍兼容手工写入的 ext:。"""
        task = (row.get("task") or "").strip().lower()
        return task[4:] if task.startswith("ext:") else task

    @staticmethod
    def _window_after(rows, index, distance):
        """找出出口后至少 distance 米的 CSV 终点。"""
        end, covered = index, 0.0
        while end < len(rows) - 1 and covered < distance:
            current, following = rows[end], rows[end + 1]
            covered += math.hypot(
                float(following["x"]) - float(current["x"]),
                float(following["y"]) - float(current["y"]),
            )
            end += 1
        return end

    def start_avoidance_test(self):
        """从首个避障区附近启动重定位与仅跟踪/避障的临时 launch。"""
        filename = self.race_file_name(show_error=True)
        if not filename:
            return
        source = Path("/home/user/fastlio_ws/src/waypoint_tools/data") / filename
        try:
            with source.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fields, rows = reader.fieldnames, list(reader)
            required = {"x", "y", "z", "qx", "qy", "qz", "qw", "task"}
            if not fields or not required.issubset(fields):
                raise ValueError("CSV 缺少避障测试所需的位姿或 task 列")

            start_index = end_index = None
            for index, row in enumerate(rows):
                task = self._task_name(row)
                if task == "avoid_start" and start_index is None:
                    start_index = index
                elif task == "avoid_end" and start_index is not None:
                    end_index = index
                    break
            if start_index is None or end_index is None:
                raise ValueError("未找到一对完整的 avoid_start / avoid_end 标记")

            # 固定实测位姿位于 avoid_start 附近（已在避障区入口处），短路线必须
            # 直接从入口标记开始。若保留入口前 2m，跟踪器会把初始投影视为不合理
            # 的进度跳变并安全锁止。
            first = start_index
            last = self._window_after(rows, end_index, 1.5)
            test_rows = [dict(row) for row in rows[first:last + 1]]
            for row in test_rows:
                if self._task_name(row) not in ("avoid_start", "avoid_end"):
                    row["task"] = "none"

            timestamp = int(time.time())
            test_path = STATUS_DIR / ("avoidance_test_%d.csv" % timestamp)
            with test_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(test_rows)

            # 外层 launch 在 include 完成后覆盖初始位姿；roslaunch 会先设置完
            # 全部参数再启动节点。此处使用实车确认过的固定避障测试位姿，
            # 而不是用 CSV 航点近似车辆的实际停放位姿。
            init_pos = AVOIDANCE_TEST_INITIAL_POS
            init_rot = AVOIDANCE_TEST_INITIAL_ROT
            launch_path = STATUS_DIR / ("avoidance_test_%d.launch" % timestamp)
            overrides_path = STATUS_DIR / ("avoidance_params_%d.yaml" % timestamp)
            overrides = self.avoidance_param_overrides()
            overrides["target_speed"] = round(self.avoid_speed_input.value(), 3)
            try:
                configured_max_linear = float(overrides.get("max_linear", 0.0))
            except ValueError:
                raise ValueError("max_linear 必须是数值")
            overrides["max_linear"] = max(configured_max_linear, overrides["target_speed"])
            overrides["inflation_radius"] = round(self.avoid_inflation_input.value(), 3)
            overrides_path.write_text(
                "\n".join("%s: %s" % (key, value) for key, value in overrides.items()) + "\n",
                encoding="utf-8",
            )
            launch_path.write_text("""<launch>
  <!-- 自动生成：仅用于避障区实车测试，退出本 launch 会同时停止重定位和跟踪。 -->
  <include file=\"$(find sfast_lio)/launch/mapping_mid360_relocalization.launch\">
    <arg name=\"rviz\" value=\"true\"/>
  </include>
  <rosparam param=\"/mapping/init_pos\">[%s]</rosparam>
  <rosparam param=\"/mapping/init_rot\">[%s]</rosparam>
  <node pkg=\"final_mission\" type=\"final_tracker.py\" name=\"final_tracker\" output=\"screen\" required=\"true\">
    <rosparam command=\"load\" file=\"$(find final_mission)/config/tracker.yaml\"/>
    <rosparam command=\"load\" file=\"%s\"/>
    <param name=\"csv_path\" value=\"%s\"/>
    <param name=\"cmd_topic\" value=\"/smoother_cmd_vel\"/>
    <!-- 必须由界面中的“确认定位后，开始避障”显式放行。 -->
    <param name=\"enabled\" value=\"false\"/>
    <param name=\"enable_topic\" value=\"/avoidance_test/tracker_enable\"/>
  </node>
</launch>
""" % (
                ", ".join("%.8f" % value for value in init_pos),
                ", ".join("%.8f" % value for value in init_rot),
                str(overrides_path),
                str(test_path),
            ), encoding="utf-8")
        except (OSError, ValueError, KeyError) as error:
            QMessageBox.critical(self, "创建避障测试失败", str(error))
            return

        if QMessageBox.question(
            self, "确认避障区独立测试",
            "将启动新的 S-FAST-LIO 重定位和避障跟踪，但跟踪器初始保持禁用、车辆不会自动行驶。"
            "请确认底盘、MID360、ROS Master 已运行，"
            "且已有重定位/跟踪/速度控制节点均已停止；车辆应停在已保存的避障测试起始位姿附近。继续？",
            QMessageBox.Yes | QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        command = (
            "source /opt/ros/noetic/setup.bash && source /home/user/fastlio_ws/devel/setup.bash "
            "&& source /home/user/livox_ws/devel/setup.bash --extend && roslaunch %s"
        ) % shlex.quote(str(launch_path))
        self.run_debug_command("避障区独立测试", command)
        self.single_test_message.setText(
            "已生成 %s：使用固定实测位姿重定位；短路线从原路线第 %d 个航点开始。"
            "请保持车辆静止，确认 RViz 定位正确后点击“确认定位后，开始避障”。"
            % (test_path.name, first + 1)
        )

    def enable_avoidance_tracking(self):
        """由人工确认定位后才打开临时避障跟踪器的速度输出。"""
        if QMessageBox.question(
            self, "确认开始避障",
            "确认 RViz 中实时点云与地图重合，且车辆位置、朝向与短路线首点一致？"
            "确认后跟踪器将开始向 /smoother_cmd_vel 输出低速指令。",
            QMessageBox.Yes | QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        command = (
            "source /opt/ros/noetic/setup.bash && rostopic pub -1 "
            "/avoidance_test/tracker_enable std_msgs/Bool \"data: true\""
        )
        self.run_debug_command("开始避障跟踪", command)
        self.single_test_message.setText("已发送避障跟踪使能；如需停车，请在避障测试终端按 Ctrl+C 或使用底盘急停。")

    def reset_avoidance_params(self):
        self.avoid_speed_input.setValue(0.20)
        self.avoid_inflation_input.setValue(0.65)
        self.avoid_params_input.setPlainText(AVOIDANCE_TEST_PARAM_TEMPLATE)
        self.single_test_message.setText("避障测试参数已恢复为默认值；仅影响下一次独立测试。")

    def avoidance_param_overrides(self):
        """校验高级编辑框，仅允许覆盖已暴露的避障参数。"""
        allowed = set()
        for line in AVOIDANCE_TEST_PARAM_TEMPLATE.splitlines():
            if ":" in line and not line.lstrip().startswith("#"):
                allowed.add(line.split(":", 1)[0].strip())
        overrides = {}
        for number, raw_line in enumerate(self.avoid_params_input.toPlainText().splitlines(), 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError("高级参数第 %d 行缺少冒号" % number)
            key, value = (part.strip() for part in line.split(":", 1))
            if key not in allowed:
                raise ValueError("高级参数第 %d 行不支持 %s" % (number, key))
            if not value:
                raise ValueError("高级参数第 %d 行的 %s 没有值" % (number, key))
            overrides[key] = value
        return overrides

    def selected_route_file(self):
        """返回 data 目录内的安全文件名，避免误写到其它目录。"""
        filename = self.route_file_input.text().strip()
        if not filename or filename != Path(filename).name or filename in (".", ".."):
            QMessageBox.warning(self, "文件名无效", "请输入文件名，例如 final02.csv；不要包含路径。")
            return None
        return filename if filename.endswith(".csv") else filename + ".csv"

    def race_file_name(self, show_error=False):
        filename = self.race_file_input.text().strip()
        if not filename or filename != Path(filename).name or filename in (".", ".."):
            if show_error:
                QMessageBox.warning(self, "路径文件名无效", "请输入 CSV 文件名，例如 final03.csv；不要包含路径。")
            return None
        return filename if filename.endswith(".csv") else filename + ".csv"

    def record_waypoints(self):
        filename = self.selected_route_file()
        if not filename:
            return
        command = "cd ~/fastlio_ws && source devel/setup.bash && rosrun waypoint_tools record_waypoints.py _odom_topic:=/Odometry _file_name:=%s _output_dir:=/home/user/fastlio_ws/src/waypoint_tools/data _min_distance:=0.15 _min_yaw_change:=0.17 _default_tol:=0.25 _frame_id:=camera_init" % shlex.quote(filename)
        self.run_debug_command("开始录制航迹点", command)

    def publish_waypoint_task(self, task, external=True):
        payload = "ext:" + task if external else task
        command = "source /opt/ros/noetic/setup.bash && source /home/user/fastlio_ws/devel/setup.bash && source /home/user/piper_ws/devel/setup.bash && rostopic pub -1 /waypoint_task std_msgs/String \"data: '%s'\"" % payload
        self.run_debug_command("记录任务点 " + task, command)

    def run_piper_task_command(self, name, ros_command):
        command = "source /opt/ros/noetic/setup.bash && source /home/user/miniconda3/etc/profile.d/conda.sh && conda activate piper && source /home/user/piper_ws/devel/setup.bash && " + ros_command
        self.run_debug_command(name, command)

    def plot_route_on_map(self):
        filename = self.selected_route_file()
        if not filename:
            return
        route = shlex.quote("/home/user/fastlio_ws/src/waypoint_tools/data/" + filename)
        command = "python3 /home/user/piper_ws/plot_route_on_map.py %s" % route
        self.run_debug_command("一键地图轨迹检查", command)

    def set_last_waypoint_as_finish(self):
        filename = self.selected_route_file()
        if not filename:
            return
        route = Path("/home/user/fastlio_ws/src/waypoint_tools/data") / filename
        try:
            with route.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fields, rows = reader.fieldnames, list(reader)
            if not fields or "task" not in fields or not rows:
                raise ValueError("CSV 为空或缺少 task 列")
            backup = route.with_suffix(route.suffix + ".before_finish.bak")
            if not backup.exists():
                shutil.copy2(route, backup)
            rows[-1]["task"] = "ext:finish"
            with route.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "设置终点失败", str(error))
            return
        self.debug_message.setText("已将 %s 的最后点设置为 ext:finish；备份：%s" % (filename, backup.name))
        QMessageBox.information(self, "终点已设置", "最后一个航点已标记为 ext:finish。")

    def run_debug_command(self, name, command):
        """调试命令使用独立终端，结束后保留 shell 供查看输出。"""
        shell_command = command + "; result=$?; echo; echo \"调试命令结束，返回码: $result\"; exec bash"
        args = ["gnome-terminal", "--title=调试：" + name, "--window", "--command=" + shlex.join(["bash", "-lc", shell_command])]
        try:
            subprocess.Popen(args, start_new_session=True)
            self.debug_message.setText("已在终端运行：%s" % name)
        except FileNotFoundError:
            QMessageBox.critical(self, "无法启动终端", "未找到 gnome-terminal。")

    def start_one(self, index):
        if index == 8 and not self.race_file_name(show_error=True):
            return False
        if not self.ensure_terminal_tabs():
            return False
        status_file = self.status_files[index]
        status_file.unlink(missing_ok=True)
        self.gate_files[index].touch()
        self.set_state(index, "starting")
        self.message.setText("正在启动：%s" % ITEMS[index].name)
        return True

    def ensure_terminal_tabs(self):
        if self.terminal_window_created:
            return True
        runner = base64.b64encode(RUNNER.encode()).decode()
        script = "import base64;exec(base64.b64decode(%s))" % repr(runner)
        # --command=... 将命令绑定到紧邻的 --window/--tab。不能使用
        # "-- 命令 --tab ..."：那个 -- 会终止 gnome-terminal 的选项解析。
        args = ["gnome-terminal"]
        for i, item in enumerate(ITEMS):
            command = (
                item.command
                .replace("{speed}", "%.2f" % self.speed_input.value())
                .replace("{post_place_speed}", "%.2f" % self.post_place_speed_input.value())
                .replace("{avoid_speed}", "%.2f" % self.race_avoid_speed_input.value())
                .replace("{route}", self.race_file_name() or "final03.csv")
            )
            runner_args = ["python3", "-c", script, str(self.status_files[i]), base64.b64encode(command.encode()).decode(), "1" if item.persistent else "0", "1" if item.accept_successful_exit else "0", str(self.gate_files[i])]
            args.extend(["--title=比赛启动终端", "--window" if i == 0 else "--tab", "--command=" + shlex.join(runner_args)])
        try:
            subprocess.Popen(args, start_new_session=True)
        except FileNotFoundError:
            QMessageBox.critical(self, "无法启动终端", "未找到 gnome-terminal。请安装它或在代码中替换终端命令。")
            return False
        self.terminal_window_created = True
        self.minimize_attempts = 0
        QTimer.singleShot(250, self.minimize_terminal)
        return True

    def minimize_terminal(self):
        """等待窗口出现后最小化；GNOME Terminal 创建标签页会有短暂延迟。"""
        try:
            result = subprocess.run(
                ["xdotool", "search", "--name", "比赛启动终端", "windowminimize"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )
        except FileNotFoundError:
            self.message.setText("终端标签页已创建；未安装 xdotool，无法自动最小化终端窗口。")
            return
        if result.returncode != 0 and self.minimize_attempts < 8:
            self.minimize_attempts += 1
            QTimer.singleShot(350, self.minimize_terminal)


    def start_all(self):
        if self.batch_index is not None:
            return
        self.all_button.setEnabled(False)
        self.all_button.setStyleSheet("")
        self.batch_index = 0
        self.batch_waiting = None
        self.message.setText("一键启动中：将按顺序启动各服务。")
        self.advance_batch()

    def advance_batch(self):
        if self.batch_index is None:
            return
        if self.batch_index >= len(ITEMS):
            self.all_button.setStyleSheet("background:#43a047; color:white; font-weight:bold;")
            self.all_button.setEnabled(True)
            self.message.setText("全部比赛服务已成功启动。")
            self.batch_index = self.batch_waiting = None
            return
        self.batch_waiting = self.batch_index
        self.batch_index += 1
        if not self.start_one(self.batch_waiting):
            self.finish_batch_failed(self.batch_waiting)

    def finish_batch_failed(self, index):
        self.all_button.setEnabled(True)
        self.all_button.setStyleSheet("background:#c62828; color:white;")
        self.message.setText("一键启动已停止：%s 启动失败。请查看对应终端。" % ITEMS[index].name)
        self.batch_index = self.batch_waiting = None

    def set_state(self, index, state, detail=""):
        button = self.buttons[index]
        if state == "success":
            button.setStyleSheet("background:#43a047; color:white; font-weight:bold;")
        elif state == "starting":
            button.setStyleSheet("background:#f9a825; color:black;")
        elif state == "failed":
            button.setStyleSheet("background:#c62828; color:white;")
        elif state == "stopped":
            button.setStyleSheet("background:#757575; color:white;")
        if detail:
            button.setToolTip(detail)

    def poll(self):
        for i, path in enumerate(self.status_files):
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            state, detail = data.get("state", ""), data.get("detail", "")
            self.set_state(i, state, detail)
            if self.batch_waiting == i and state in ("success", "failed", "stopped"):
                if state == "success":
                    self.batch_waiting = None
                    self.advance_batch()
                else:
                    self.finish_batch_failed(i)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Launcher()
    window.show()
    sys.exit(app.exec_())
