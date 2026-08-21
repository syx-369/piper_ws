#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""比赛启动器：在独立终端启动 ROS 服务，并显示启动状态。"""

import base64
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QCheckBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton, QStackedWidget, QVBoxLayout, QWidget


STATUS_DIR = Path("/tmp/competition_launcher_status")
STARTUP_GRACE_SECONDS = 5


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
    LaunchItem("9. 比赛总控", "source /opt/ros/noetic/setup.bash && source /home/user/fastlio_ws/devel/setup.bash && RACE_CSV=/home/user/fastlio_ws/src/waypoint_tools/data/{route} && CRUISE_SPEED={speed} && roslaunch final_mission final_race.launch csv_path:=${RACE_CSV} target_speed:=${CRUISE_SPEED} wait_for_start:=true auto_start:=false enable_vision:=true show_image:=true"),
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
        self.speed_input.setValue(0.80)
        self.speed_input.setSuffix(" m/s")
        self.speed_input.setToolTip("仅用于第 9 项“比赛总控”的 target_speed")
        speed_row.addWidget(self.speed_input)
        speed_row.addWidget(QLabel("路径文件："))
        self.race_file_input = QLineEdit("final03.csv")
        self.race_file_input.setPlaceholderText("例如 final03.csv")
        self.race_file_input.setToolTip("第 9 项总控使用的 CSV，位于 waypoint_tools/data/")
        speed_row.addWidget(self.race_file_input)
        layout.addLayout(speed_row)
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
        actions.addWidget(all_button)
        actions.addWidget(none_button)
        actions.addWidget(start_button)
        layout.addLayout(actions)
        self.single_test_message = QLabel("路径文件使用主界面的“路径文件”输入框。")
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
            command = item.command.replace("{speed}", "%.2f" % self.speed_input.value()).replace("{route}", self.race_file_name() or "final03.csv")
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
