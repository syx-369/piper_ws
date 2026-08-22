#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""正式比赛用 Piper 抓取/放置任务节点。

视觉识别、深度定位、手眼变换、笛卡尔插值以及抓放偏移全部调用
vision_grasp_core.py；该文件是 grab2016_7_21.py 的原样副本。
本节点只增加：任务触发、抓放阶段拆分、已录制路径播放以及两轮状态管理。
"""

import os
import queue
import sys
import threading
import time

import numpy as np
import rospy
import rospkg
import tf.transformations as tf_trans
from geometry_msgs.msg import PoseStamped
from piper_msgs.msg import PosCmd
from piper_msgs.srv import GoZero
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String

# ===== 改动 2026-07-30（1/3）：接入 final_mission 的相机按需中继 =====
# 原因：全车只有一台 D435i 且装在机械臂上，pyrealsense2 设备独占，
# 本节点在 init_camera_and_detector() 就 pipeline.start() 并全程持有，
# 所以 final_mission 的红旗/红绿灯识别没法自己开相机。
# 解法：本节点保持唯一相机持有者，按需把帧发成 ROS 话题。
#
# 默认关闭（enable_camera_relay=false）。不打开时本文件行为与改动前完全一致。
# 打开方式：roslaunch piper_task piper_task.launch enable_camera_relay:=true
#           或 rosrun 时加 _enable_camera_relay:=true
#
# 若 final_mission 包不存在（例如单独用 piper_ws），这里静默跳过，不影响启动。
try:
    _FINAL_MISSION_SCRIPTS = os.path.join(
        rospkg.RosPack().get_path("final_mission"), "scripts"
    )
    if _FINAL_MISSION_SCRIPTS not in sys.path:
        sys.path.insert(0, _FINAL_MISSION_SCRIPTS)
    from camera_relay import CameraRelayMixin
    _HAS_CAMERA_RELAY = True
except Exception:
    _HAS_CAMERA_RELAY = False

    class CameraRelayMixin:
        """找不到 final_mission 时的空实现，保证本节点照常启动。"""

        def start_frame_relay(self):
            pass

        def set_relay_paused(self, paused):
            pass
# ===== 改动结束（1/3）=====

from piper_task.vision_grasp_core import (
    LABEL_DESCRIPTION,
    T_EE_TO_TOOL,
    T_OBJECT_TO_GRASP,
    PiperVisionController,
)


# 三个固定关节姿态与 2026-07-21 测试脚本完全一致。
PICK_SCAN_JOINTS = [-1.530, 0.446, 0.0, 0.0, -0.115, 0.0, 0.0]
TRANSPORT_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
PLACE_SCAN_JOINTS = [-1.602, 0.641, -0.509, 0.0, 0.324, 0.0, 0.0]
# 改动 2026-07-30（2/3）：混入 CameraRelayMixin（原为 PiperVisionController 单继承）
class CompetitionVisionController(CameraRelayMixin, PiperVisionController):
    """使用固定 ROS 节点名初始化原测试脚本的视觉控制器。"""

    def __init__(self):
        # 正式包需要稳定的 /piper_task 私有参数空间，因此这里只替换原类中
        # anonymous=True 的节点初始化；其余话题、反馈等待、相机和模型初始化一致。
        rospy.init_node("piper_task", anonymous=False)

        self.pub = rospy.Publisher("/pin_pos_cmd", PosCmd, queue_size=1)
        self.joint_pub = rospy.Publisher("/joint_states", JointState, queue_size=1)
        self.detection_image_topic = rospy.get_param(
            "~detection_image_topic", "/piper_task/detection_image"
        )
        self.detection_status_topic = rospy.get_param(
            "~detection_status_topic", "/piper_task/detection_status"
        )
        self.detection_image_pub = rospy.Publisher(
            self.detection_image_topic, Image, queue_size=1
        )
        self.detection_status_pub = rospy.Publisher(
            self.detection_status_topic, String, queue_size=10
        )

        self.current_ee_pose_mat = np.eye(4)
        self.pose_received = False
        self.pose_received_at = None
        self.startup_joint_positions = None
        self.startup_joint_received_at = None
        rospy.Subscriber("/end_pose", PoseStamped, self.pose_callback)
        self.startup_joint_sub = rospy.Subscriber(
            rospy.get_param("~startup_joint_state_topic", "/joint_states_single"),
            JointState,
            self.startup_joint_callback,
            queue_size=10,
        )

        rospy.loginfo("等待获取机械臂实时位姿 (/end_pose)...")
        while not self.pose_received and not rospy.is_shutdown():
            rospy.sleep(0.1)
        if rospy.is_shutdown():
            raise rospy.ROSInterruptException("等待 /end_pose 时节点被关闭")
        rospy.loginfo("成功接入实时位姿反馈！")

        if bool(rospy.get_param("~startup_go_zero", True)):
            self.go_zero_and_wait()
        else:
            rospy.logwarn("startup_go_zero=false：跳过启动回零。")

        self.init_camera_and_detector()

        # 改动 2026-07-30（2/3）：相机就绪后启动按需中继。
        # 必须在 init_camera_and_detector() 之后 —— 中继要用 self.pipeline。
        self.start_frame_relay()
        rospy.loginfo(
            "机械臂检测调试输出：image=%s status=%s",
            self.detection_image_topic,
            self.detection_status_topic,
        )

    def pose_callback(self, msg):
        """保存 /end_pose 实际反馈，并记录到达时间供运动日志判断新鲜度。"""
        super().pose_callback(msg)
        self.pose_received_at = time.monotonic()

    def startup_joint_callback(self, msg):
        if len(msg.position) < 6:
            return
        self.startup_joint_positions = tuple(float(value) for value in msg.position[:6])
        self.startup_joint_received_at = time.monotonic()

    def go_zero_and_wait(self):
        """启动时调用驱动回零服务，并等待六个关节真实收敛到零位。"""
        service_name = rospy.get_param("~go_zero_service", "/go_zero_srv")
        wait_service_timeout = max(
            0.1, float(rospy.get_param("~go_zero_service_timeout", 10.0))
        )
        motion_timeout = max(
            0.1, float(rospy.get_param("~go_zero_motion_timeout", 30.0))
        )
        tolerance = max(
            0.001, float(rospy.get_param("~go_zero_joint_tolerance", 0.05))
        )
        stable_samples_required = max(
            1, int(rospy.get_param("~go_zero_stable_samples", 5))
        )
        is_mit_mode = bool(rospy.get_param("~go_zero_is_mit_mode", False))

        rospy.loginfo(
            "启动回零：等待服务 %s（service %.1fs / motion %.1fs），"
            "is_mit_mode=%s，关节容差=%.3frad，稳定帧=%d",
            service_name,
            wait_service_timeout,
            motion_timeout,
            is_mit_mode,
            tolerance,
            stable_samples_required,
        )
        try:
            rospy.wait_for_service(service_name, timeout=wait_service_timeout)
            response = rospy.ServiceProxy(service_name, GoZero)(
                is_mit_mode=is_mit_mode
            )
        except (rospy.ROSException, rospy.ServiceException) as exc:
            raise RuntimeError("启动回零服务调用失败：%s" % exc)

        if not response.status:
            raise RuntimeError(
                "启动回零服务拒绝请求：status=false code=%s" % response.code
            )

        rospy.loginfo(
            "回零指令已发送（code=%s），等待关节反馈实际到零位。", response.code
        )
        deadline = time.monotonic() + motion_timeout
        stable_samples = 0
        last_max_error = float("inf")
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            positions = self.startup_joint_positions
            received_at = self.startup_joint_received_at
            feedback_fresh = (
                positions is not None
                and received_at is not None
                and time.monotonic() - received_at <= 1.0
            )
            if feedback_fresh:
                last_max_error = max(abs(value) for value in positions)
                if last_max_error <= tolerance:
                    stable_samples += 1
                    if stable_samples >= stable_samples_required:
                        rospy.loginfo(
                            "机械臂启动回零完成：最大关节误差 %.4frad，连续 %d 帧合格。",
                            last_max_error,
                            stable_samples,
                        )
                        # 保留关节反馈订阅。比赛中的失败恢复会再次核验机械臂
                        # 确实回到运输零位，不能只凭“命令已发送”就允许车辆倒车。
                        return
                else:
                    stable_samples = 0
            rate.sleep()

        if rospy.is_shutdown():
            raise rospy.ROSInterruptException("等待机械臂启动回零时节点被关闭")
        raise RuntimeError(
            "机械臂启动回零超时 %.1fs：最大关节误差 %s，"
            "未达到 %.3frad×%d 帧；拒绝开放 piper_task。"
            % (
                motion_timeout,
                "无有效反馈"
                if not np.isfinite(last_max_error)
                else "%.4frad" % last_max_error,
                tolerance,
                stable_samples_required,
            )
        )


class CompetitionTaskNode:
    """把原测试程序的一次连续抓放拆成车辆可触发的 pick/place。"""

    VALID_COMMANDS = {
        "card",
        "pick", "pick1", "pick2", "pick3",
        "place", "place1", "place2", "place3",
        "prepare_pick_scan", "prepare_place_scan", "place_any",
        "abort_round", "discard_place",
        "stow", "reset", "status", "continue",
    }

    def __init__(self, vision_controller):
        self.arm = vision_controller
        self.command_topic = rospy.get_param("~command_topic", "/piper_task/command")
        self.state_topic = rospy.get_param("~state_topic", "/piper_task/state")
        self.result_topic = rospy.get_param("~result_topic", "/piper_task/result")
        self.target_topic = rospy.get_param("~target_topic", "/piper_task/target")
        self.waypoint_event_topic = rospy.get_param(
            "~waypoint_event_topic", "/waypoint_task_event"
        )
        self.waypoint_done_topic = rospy.get_param(
            "~waypoint_done_topic", "/waypoint_task_done"
        )
        self.navigation_skip_topic = rospy.get_param(
            "~navigation_skip_topic", "/piper_task/navigation_skip"
        )

        self.max_rounds = int(rospy.get_param("~max_rounds", 2))
        self.card_confirm_hold_time = max(
            0.0, float(rospy.get_param("~card_confirm_hold_time", 0.15))
        )
        # 比赛动作节拍可在 task_config.yaml 中继续调整。限制最大倍率，避免
        # 配置笔误使笛卡尔插值或延迟进入不安全范围。
        self.cartesian_speed_scale = min(
            2.5, max(0.25, float(rospy.get_param("~cartesian_speed_scale", 2.25)))
        )
        self.joint_speed_percent = min(
            100.0, max(1.0, float(rospy.get_param("~joint_speed_percent", 40.0)))
        )
        self.observation_joint_speed_percent = min(
            100.0,
            max(
                1.0,
                float(rospy.get_param("~observation_joint_speed_percent", 60.0)),
            ),
        )
        self.place_stow_joint_speed_percent = min(
            100.0,
            max(
                1.0,
                float(rospy.get_param("~place_stow_joint_speed_percent", 70.0)),
            ),
        )
        self.place_cartesian_speed_scale = min(
            1.5,
            max(
                1.0,
                float(rospy.get_param("~place_cartesian_speed_scale", 1.25)),
            ),
        )
        self.place_delay_scale = min(
            1.0, max(0.5, float(rospy.get_param("~place_delay_scale", 0.7)))
        )
        self.gripper_close_wait = max(
            0.5, float(rospy.get_param("~gripper_close_wait", 0.65))
        )
        self.gripper_open_wait = max(
            0.3, float(rospy.get_param("~gripper_open_wait", 0.4))
        )
        self.motion_delay_scale = min(
            2.0, max(0.2, float(rospy.get_param("~motion_delay_scale", 0.3)))
        )
        self.joint_motion_verify_timeout = max(
            0.5, float(rospy.get_param("~joint_motion_verify_timeout", 4.0))
        )
        self.joint_settle_time = max(
            0.0, float(rospy.get_param("~joint_settle_time", 0.1))
        )
        self.vehicle_stop_actions = rospy.get_param("~vehicle_stop_actions", {})

        self.state_pub = rospy.Publisher(
            self.state_topic, String, queue_size=1, latch=True
        )
        self.result_pub = rospy.Publisher(
            self.result_topic, String, queue_size=10, latch=True
        )
        self.target_pub = rospy.Publisher(
            self.target_topic, String, queue_size=1, latch=True
        )
        self.waypoint_done_pub = rospy.Publisher(
            self.waypoint_done_topic, String, queue_size=10
        )
        self.navigation_skip_pub = rospy.Publisher(
            self.navigation_skip_topic, String, queue_size=1, latch=True
        )

        self.command_queue = queue.Queue(maxsize=1)
        self.queue_lock = threading.Lock()
        self.busy = False
        self.selected_target_label = None
        self.carried_target_label = None
        self.no_payload = False
        self.completed_rounds = 0
        self.navigation_skip_tasks = set()

        rospy.Subscriber(
            self.command_topic, String, self.command_callback, queue_size=10
        )
        rospy.Subscriber(
            self.waypoint_event_topic,
            String,
            self.waypoint_event_callback,
            queue_size=10,
        )
        rospy.on_shutdown(self.on_shutdown)

        self.publish_target("")
        self.publish_navigation_skip(())
        self.publish_state("idle")
        rospy.loginfo(
            "机械臂节拍：笛卡尔速度×%.2f，关节速度=%.0f%%，"
            "观察位速度=%.0f%%，放置笛卡尔额外×%.2f，放置回位速度=%.0f%%，"
            "动作等待×%.2f，放置等待额外×%.2f "
            "（关节实时到位确认 timeout=%.1fs，settle=%.2fs）",
            self.cartesian_speed_scale,
            self.joint_speed_percent,
            self.observation_joint_speed_percent,
            self.place_cartesian_speed_scale,
            self.place_stow_joint_speed_percent,
            self.motion_delay_scale,
            self.place_delay_scale,
            self.joint_motion_verify_timeout,
            self.joint_settle_time,
        )
        rospy.loginfo(
            "piper_task ready: command topic=%s, "
            "commands=card/pick1-3/place1-3/prepare_pick_scan/"
            "prepare_place_scan/place_any/abort_round/discard_place/"
            "stow/reset/status",
            self.command_topic,
        )

    def publish_state(self, state):
        rospy.loginfo("piper_task state: %s", state)
        self.state_pub.publish(String(data=state))

    def publish_result(self, result):
        rospy.loginfo("piper_task result: %s", result)
        self.result_pub.publish(String(data=result))

    def publish_target(self, label):
        self.target_pub.publish(String(data=label or ""))

    def publish_navigation_skip(self, task_names):
        """发布已经无需停车的后续车辆任务名。"""
        self.navigation_skip_tasks = set(task_names)
        self.navigation_skip_pub.publish(
            String(data=",".join(sorted(self.navigation_skip_tasks)))
        )

    def command_callback(self, msg):
        command = msg.data.strip().lower()
        if command not in self.VALID_COMMANDS:
            self.publish_result("rejected:unknown_command:%s" % command)
            return

        if command == "status":
            selected = self.selected_target_label or "none"
            carried = self.carried_target_label or "none"
            self.publish_result(
                "status:busy=%s,selected=%s,carried=%s,no_payload=%s,rounds=%d/%d"
                % (
                    self.busy,
                    selected,
                    carried,
                    self.no_payload,
                    self.completed_rounds,
                    self.max_rounds,
                )
            )
            return

        with self.queue_lock:
            if self.busy or not self.command_queue.empty():
                self.publish_result("rejected:busy:%s" % command)
                return
            self.command_queue.put_nowait((command, None))

    def waypoint_event_callback(self, msg):
        """接收 follow_waypoints.py 发布的 start:<task>:idx<N> 事件。"""
        fields = msg.data.strip().split(":", 2)
        if len(fields) < 2 or fields[0] != "start":
            return
        task_name = fields[1].strip()
        if task_name not in self.vehicle_stop_actions:
            return

        if task_name in self.navigation_skip_tasks:
            done = "done:%s" % task_name
            self.waypoint_done_pub.publish(String(data=done))
            self.publish_result("success:waypoint:%s:skip_without_arm" % task_name)
            rospy.loginfo("候选点 %s 已不需要机械臂动作，立即通知车辆继续", task_name)
            return

        action = str(self.vehicle_stop_actions.get(task_name, "")).strip().lower()
        if not action:
            rospy.logerr(
                "车辆已到达 %s，但该停靠点尚未配置机械臂动作；车辆保持停车。",
                task_name,
            )
            self.publish_result("failed:waypoint:%s:action_not_configured" % task_name)
            return
        if action not in self.VALID_COMMANDS or action == "status":
            rospy.logerr("停靠点 %s 配置了无效动作: %s", task_name, action)
            self.publish_result("failed:waypoint:%s:invalid_action" % task_name)
            return

        with self.queue_lock:
            if self.busy or not self.command_queue.empty():
                rospy.logerr("机械臂忙，拒绝停靠事件 %s", task_name)
                self.publish_result("failed:waypoint:%s:arm_busy" % task_name)
                return
            self.command_queue.put_nowait((action, task_name))
        rospy.loginfo("车辆停靠事件 %s -> 机械臂动作 %s", task_name, action)

    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            try:
                command, waypoint_task_name = self.command_queue.get_nowait()
            except queue.Empty:
                rate.sleep()
                continue

            self.busy = True
            # 改动 2026-07-30（3/3）：机械臂动作期间暂停相机中继。
            # vision_grasp_core 只在 :443 和 :609 两处 wait_for_frames，
            # 都在本区间内；pyrealsense2 不允许并发调用，必须让中继避让。
            self.arm.set_relay_paused(True)
            try:
                ok = self.execute_command(command)
                if ok and waypoint_task_name:
                    done = "done:%s" % waypoint_task_name
                    self.waypoint_done_pub.publish(String(data=done))
                    rospy.loginfo("已通知车辆继续: %s", done)
            except Exception as exc:
                rospy.logerr("任务 %s 异常: %s", command, exc)
                self.publish_result("failed:%s:exception" % command)
            finally:
                self.busy = False
                # 改动 2026-07-30（3/3）：动作结束，允许中继继续取帧
                self.arm.set_relay_paused(False)
                self.publish_state("idle:no_payload" if self.no_payload else "idle")
                self.command_queue.task_done()

    def execute_command(self, command):
        if command == "card":
            ok, reason = self.execute_card()
        elif command in ("pick", "pick1", "pick2", "pick3"):
            candidate = 1 if command == "pick" else int(command[-1])
            ok, reason = self.execute_pick_candidate(candidate)
        elif command in ("place", "place1", "place2", "place3"):
            candidate = 1 if command == "place" else int(command[-1])
            ok, reason = self.execute_place_candidate(candidate)
        elif command == "prepare_pick_scan":
            ok, reason = self.execute_prepare_pick_scan()
        elif command == "prepare_place_scan":
            ok, reason = self.execute_prepare_place_scan()
        elif command == "place_any":
            ok, reason = self.execute_place_any()
        elif command == "abort_round":
            ok, reason = self.execute_abort_round()
        elif command == "discard_place":
            ok, reason = self.execute_discard_place()
        elif command == "stow":
            ok, reason = self.execute_stow()
        elif command == "reset":
            self.selected_target_label = None
            self.carried_target_label = None
            self.no_payload = False
            self.completed_rounds = 0
            self.publish_target("")
            self.publish_navigation_skip(())
            ok, reason = True, "state_cleared"
        elif command == "continue":
            # 仅确认车辆停靠点，不驱动机械臂。
            ok, reason = True, "waypoint_acknowledged"
        else:
            ok, reason = False, "unknown_command"

        prefix = "success" if ok else "failed"
        self.publish_result("%s:%s:%s" % (prefix, command, reason))
        return ok

    def scaled_cartesian_speed(self, base_speed):
        """按比赛配置提高笛卡尔插值速度。"""
        return float(base_speed) * self.cartesian_speed_scale

    def wait_after_motion(self, base_seconds):
        """按比赛配置缩短运动、夹爪动作后的稳定等待。"""
        duration = max(0.0, float(base_seconds) * self.motion_delay_scale)
        if duration > 0.0:
            rospy.sleep(duration)

    def wait_after_place_motion(self, base_seconds):
        """在全局等待倍率基础上进一步缩短放置阶段等待。"""
        self.wait_after_motion(float(base_seconds) * self.place_delay_scale)

    def move_joint_pose(self, positions, speed_percent=None):
        """发送固定关节姿态，并按加速后的节拍等待。"""
        speed_percent = (
            self.joint_speed_percent
            if speed_percent is None
            else min(100.0, max(1.0, float(speed_percent)))
        )
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = [""]
        msg.position = list(positions)
        msg.velocity = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            speed_percent,
        ]
        msg.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
        self.arm.joint_pub.publish(msg)
        # 不再使用按最慢速度估算的固定 sleep。实时反馈一旦连续稳定到位便
        # 立即继续；若未到位则拒绝进入视觉/抓放阶段，避免提速损失成功率。
        if not self.wait_joint_pose(
            positions, timeout=self.joint_motion_verify_timeout
        ):
            return False
        if self.joint_settle_time > 0.0:
            rospy.sleep(self.joint_settle_time)
        return not rospy.is_shutdown()

    def log_pick_pose_target(self, stage, target_pose):
        """记录即将发送给逆解节点的抓取末端目标。"""
        target_xyz = target_pose[0:3, 3]
        rospy.loginfo(
            "抓取位姿[%s] 指令目标 EE XYZ: X=%.4f Y=%.4f Z=%.4f m",
            stage,
            target_xyz[0],
            target_xyz[1],
            target_xyz[2],
        )

    def log_pick_pose_feedback(self, stage, target_pose):
        """将 /end_pose 实际反馈与本阶段抓取目标进行对比。"""
        actual_pose = self.arm.current_ee_pose_mat.copy()
        actual_xyz = actual_pose[0:3, 3]
        target_xyz = target_pose[0:3, 3]
        error_mm = (actual_xyz - target_xyz) * 1000.0
        error_norm_mm = float(np.linalg.norm(error_mm))
        received_at = getattr(self.arm, "pose_received_at", None)
        feedback_age = (
            float("inf")
            if received_at is None
            else max(0.0, time.monotonic() - received_at)
        )
        age_text = "未知" if not np.isfinite(feedback_age) else "%.3fs" % feedback_age
        rospy.loginfo(
            "抓取位姿[%s] 实际反馈 EE XYZ: X=%.4f Y=%.4f Z=%.4f m; "
            "误差(实际-目标): dX=%+.1f dY=%+.1f dZ=%+.1f mm, 总误差=%.1f mm; "
            "/end_pose反馈年龄=%s",
            stage,
            actual_xyz[0],
            actual_xyz[1],
            actual_xyz[2],
            error_mm[0],
            error_mm[1],
            error_mm[2],
            error_norm_mm,
            age_text,
        )

    def wait_joint_pose(self, positions, timeout=None):
        """用真实关节反馈确认前六轴到达指定姿态。"""
        if timeout is None:
            timeout = max(
                0.1,
                float(rospy.get_param("~runtime_joint_verify_timeout", 3.0)),
            )
        else:
            timeout = max(0.1, float(timeout))
        tolerance = max(
            0.001, float(rospy.get_param("~runtime_joint_tolerance", 0.05))
        )
        stable_required = max(
            1, int(rospy.get_param("~runtime_joint_stable_samples", 3))
        )
        target = tuple(float(value) for value in positions[:6])
        deadline = time.monotonic() + timeout
        stable = 0
        last_error = float("inf")
        rate = rospy.Rate(20)

        while not rospy.is_shutdown() and time.monotonic() < deadline:
            feedback = self.arm.startup_joint_positions
            received_at = self.arm.startup_joint_received_at
            fresh = (
                feedback is not None
                and received_at is not None
                and time.monotonic() - received_at <= 1.0
            )
            if fresh:
                last_error = max(
                    abs(actual - desired)
                    for actual, desired in zip(feedback[:6], target)
                )
                if last_error <= tolerance:
                    stable += 1
                    if stable >= stable_required:
                        return True
                else:
                    stable = 0
            rate.sleep()

        rospy.logerr(
            "关节姿态确认失败：最大误差=%s，要求 <= %.3frad 连续 %d 帧。",
            (
                "无有效反馈"
                if not np.isfinite(last_error)
                else "%.4frad" % last_error
            ),
            tolerance,
            stable_required,
        )
        return False

    def is_joint_pose_reached(self, positions):
        """当前新鲜关节反馈是否已在目标姿态容差内。"""
        feedback = self.arm.startup_joint_positions
        received_at = self.arm.startup_joint_received_at
        if feedback is None or received_at is None:
            return False
        if time.monotonic() - received_at > 1.0:
            return False

        tolerance = max(
            0.001, float(rospy.get_param("~runtime_joint_tolerance", 0.05))
        )
        target = tuple(float(value) for value in positions[:6])
        error = max(
            abs(actual - desired)
            for actual, desired in zip(feedback[:6], target)
        )
        return error <= tolerance

    def close_gripper_at_current_pose(self):
        curr_pose = self.arm.current_ee_pose_mat.copy()
        roll, pitch, yaw = tf_trans.euler_from_matrix(curr_pose, axes="sxyz")
        cmd = PosCmd()
        cmd.x, cmd.y, cmd.z = curr_pose[0:3, 3]
        cmd.roll, cmd.pitch, cmd.yaw = roll, pitch, yaw
        cmd.gripper = 0
        cmd.mode1 = 1
        cmd.mode2 = 0
        self.arm.pub.publish(cmd)

    def open_gripper_at_current_pose(self):
        curr_pose = self.arm.current_ee_pose_mat.copy()
        roll, pitch, yaw = tf_trans.euler_from_matrix(curr_pose, axes="sxyz")
        cmd = PosCmd()
        cmd.x, cmd.y, cmd.z = curr_pose[0:3, 3]
        cmd.roll, cmd.pitch, cmd.yaw = roll, pitch, yaw
        cmd.gripper = 200
        cmd.mode1 = 1
        cmd.mode2 = 0
        self.arm.pub.publish(cmd)

    def execute_card(self):
        stage_started_at = time.monotonic()
        if self.carried_target_label is not None:
            return False, "already_carrying=%s" % self.carried_target_label
        if self.completed_rounds >= self.max_rounds:
            return False, "all_rounds_completed"

        self.selected_target_label = None
        self.no_payload = False
        self.publish_target("")
        self.publish_navigation_skip(())

        self.publish_state("card:moving_to_scan")
        if not self.move_joint_pose(
            PICK_SCAN_JOINTS,
            speed_percent=self.observation_joint_speed_percent,
        ):
            return False, "reference_scan_failed"
        scan_ready_at = time.monotonic()

        self.publish_state("card:recognizing_reference")
        self.arm.detection_context = "card"
        selected_label = self.arm.recognize_reference_target()
        recognized_at = time.monotonic()
        target_type = self.arm.target_type_from_label(selected_label)
        if target_type is None:
            return False, "reference_not_recognized"

        # 识别结果已经连续三帧确认；短暂停留用于展示结果和移走卡片。
        rospy.loginfo("卡片识别结果保持 %.1fs。", self.card_confirm_hold_time)
        rospy.sleep(self.card_confirm_hold_time)
        self.selected_target_label = selected_label
        self.publish_target(selected_label)

        rospy.loginfo(
            "卡牌阶段耗时：到观察位 %.2fs，识别 %.2fs，确认停留 %.2fs，总计 %.2fs。",
            scan_ready_at - stage_started_at,
            recognized_at - scan_ready_at,
            self.card_confirm_hold_time,
            time.monotonic() - stage_started_at,
        )

        rospy.loginfo(
            "目标卡片识别完成: %s",
            LABEL_DESCRIPTION.get(selected_label, selected_label),
        )
        return True, selected_label

    def execute_pick_candidate(self, candidate):
        if candidate not in (1, 2, 3):
            return False, "invalid_pick_candidate"
        if self.carried_target_label is not None:
            return True, "skip_already_picked=%s" % self.carried_target_label
        if self.selected_target_label is None:
            return False, "card_target_not_selected"

        selected_label = self.selected_target_label
        target_type = self.arm.target_type_from_label(selected_label)
        if target_type is None:
            return False, "invalid_selected_target=%s" % selected_label

        target_text = "方块" if target_type == "block" else "瓶子"
        rospy.loginfo(
            "已选择抓取目标: %s", LABEL_DESCRIPTION.get(selected_label, selected_label)
        )

        self.publish_state("pick%d:locating_object" % candidate)
        self.arm.detection_context = "pick%d" % candidate
        T_base_object = self.arm.get_object_pose(target_type, selected_label)
        if T_base_object is None:
            vision_failure = str(
                getattr(self.arm, "last_object_detection_failure", "not_found")
                or "not_found"
            )
            if vision_failure.startswith("target_at_edge="):
                edge_direction = vision_failure.split("=", 1)[1] or "unknown"
                if candidate < 3:
                    return True, "target_at_edge_%s_continue_to_pick%d" % (
                        edge_direction,
                        candidate + 1,
                    )
                return False, "target_at_edge_at_final_pick_point=%s" % (
                    edge_direction
                )
            if candidate < 3:
                return True, "not_here_continue_to_pick%d" % (candidate + 1)
            return False, "object_not_found_at_all_pick_points"

        # 抓取偏移沿用 grab2016_7_16_1.py 中已验证的实机参数。
        self.publish_state("pick%d:moving_to_pregrasp" % candidate)
        T_pre = T_OBJECT_TO_GRASP.copy()
        # 抓取时夹爪偏左，沿 -Y（向右）补偿 1 cm；预抓取位额外上抬 1 cm，
        # 避免横向接近时蹭到桌面。+X 向下，因此 x 减小表示上抬。
        T_pre[0:3, 3] = [-0.04, -0.03, -0.05]
        target_ee_pre = T_base_object @ T_pre @ T_EE_TO_TOOL
        self.log_pick_pose_target("pregrasp", target_ee_pre)
        self.arm.move_to_target_smooth(
            target_ee_pre,
            v=self.scaled_cartesian_speed(0.06),
            gripper_val=100,
        )
        # 缩短预抓取稳定等待，更快进入最终夹取位。
        self.wait_after_motion(2.0)
        self.log_pick_pose_feedback("pregrasp", target_ee_pre)

        self.publish_state("pick%d:approaching" % candidate)
        T_approach = T_OBJECT_TO_GRASP.copy()
        # 最终夹取点向左（+Y）补偿 1 cm，并累计沿 +X 向下补偿 2 cm。
        # 瓶子抓取时将前探方向（+Z）由 60 mm 改为 57 mm，使夹爪后退
        # 2 mm；方块仍使用原位置，避免影响已经调好的方块抓取。
        approach_forward = 0.057 if target_type == "bottle" else 0.06
        T_approach[0:3, 3] = [-0.01, -0.02, approach_forward]
        target_ee_approach = T_base_object @ T_approach @ T_EE_TO_TOOL
        self.log_pick_pose_target("approach", target_ee_approach)
        self.arm.move_to_target_smooth(
            target_ee_approach,
            v=self.scaled_cartesian_speed(0.02),
            gripper_val=100,
        )
        self.wait_after_motion(2.5)
        self.log_pick_pose_feedback("approach", target_ee_approach)

        self.publish_state("pick%d:closing_gripper" % candidate)
        self.close_gripper_at_current_pose()
        # 夹爪需要完整闭合才能抬升；使用独立下限，不随全局提速继续压缩。
        rospy.sleep(self.gripper_close_wait)

        self.publish_state("pick%d:lifting" % candidate)
        T_up = T_OBJECT_TO_GRASP.copy()
        T_up[0:3, 3] = [-0.15, 0.0, 0.04]
        self.arm.move_to_target_smooth(
            T_base_object @ T_up @ T_EE_TO_TOOL,
            v=self.scaled_cartesian_speed(0.05),
            gripper_val=0,
        )
        # 抬起完成后缩短等待，更快回到运输零位。
        self.wait_after_motion(1.5)

        self.publish_state("pick%d:moving_to_transport" % candidate)
        if not self.move_joint_pose(TRANSPORT_JOINTS):
            return False, "transport_path_failed_after_grasp"

        # 7_21 连续测试在抓取归位后原本额外等待 2 秒；七点功能测试随后虽会
        # 行车，仍保留这段等待，但按 motion_delay_scale 缩短。
        self.wait_after_motion(2.0)

        self.carried_target_label = selected_label
        self.no_payload = False
        self.publish_target(selected_label)
        # 例如在第2个车辆点（candidate=1）抓取成功，则车辆点3、4无需停车。
        remaining_pick_stops = tuple(
            "piper_stop_%d" % stop_number
            for stop_number in range(candidate + 2, 5)
        )
        self.publish_navigation_skip(remaining_pick_stops)
        rospy.loginfo("%s抓取完成，机械臂已进入运输姿态", target_text)
        return True, selected_label

    def execute_place_candidate(self, candidate):
        if candidate not in (1, 2, 3):
            return False, "invalid_place_candidate"
        if self.no_payload:
            return True, "skip_no_payload"
        if self.carried_target_label is None:
            # 若已经在前面的放置候选点成功，后续候选点直接跳过。
            if self.selected_target_label is None:
                return True, "skip_already_placed"
            # 本轮没有抓到货物时，卸货点不执行机械臂动作，车辆继续完成路线。
            return True, "skip_no_payload"

        target_label = self.carried_target_label
        target_type = self.arm.target_type_from_label(target_label)
        if target_type is None:
            return False, "invalid_carried_target=%s" % target_label

        self.publish_state("place%d:moving_to_scan" % candidate)
        # 第5个车辆点对应 candidate=1。若未发现目标，机械臂保持该姿态，
        # 车辆逐点移动到 candidate=2/3，完全符合“逐个走点”的流程。
        if candidate == 1 and not self.move_joint_pose(
            PLACE_SCAN_JOINTS,
            speed_percent=self.observation_joint_speed_percent,
        ):
            return False, "place_scan_failed"

        # 与 7_21 脚本一致：只定位任务开始时锁定的物品类型+颜色图片。
        self.publish_state("place%d:locating_target" % candidate)
        self.arm.detection_context = "place%d" % candidate
        T_base_place_image = self.arm.get_place_pose_by_image_style(target_label)
        if T_base_place_image is None:
            if candidate < 3:
                return True, "not_here_continue_to_place%d" % (candidate + 1)
            return False, "target_not_found_at_all_place_points"

        return self.execute_place_at_pose(
            T_base_place_image,
            state_prefix="place%d" % candidate,
            placed_on_label=target_label,
            candidate=candidate,
        )

    def execute_place_at_pose(
        self, T_base_place_image, state_prefix, placed_on_label, candidate=None
    ):
        """复用正常放置轨迹；candidate=None 表示任意目标降级放置。"""

        # 放置时末端偏右，沿 +Y（向左）补偿 1 cm。
        self.publish_state("%s:moving_to_release" % state_prefix)
        T_pre = T_OBJECT_TO_GRASP.copy()
        T_pre[0:3, 3] = [-0.12, -0.04, -0.06]
        target_ee_pre = T_base_place_image @ T_pre @ T_EE_TO_TOOL
        self.arm.move_to_target_smooth(
            target_ee_pre,
            v=(
                self.scaled_cartesian_speed(0.06)
                * self.place_cartesian_speed_scale
            ),
            gripper_val=0,
        )
        self.wait_after_place_motion(3.5)

        self.publish_state("%s:opening_gripper" % state_prefix)
        self.open_gripper_at_current_pose()
        # 保留可靠松爪时间，避免全局等待倍率过低时物品尚未释放便开始收臂。
        rospy.sleep(self.gripper_open_wait)

        # 松开物品后不再执行额外上抬，直接回运输姿态。
        self.publish_state("%s:moving_to_stow" % state_prefix)
        if not self.move_joint_pose(
            TRANSPORT_JOINTS,
            speed_percent=self.place_stow_joint_speed_percent,
        ):
            # 物品已经放下，保留准确状态，防止误以为仍在持物。
            self.carried_target_label = None
            self.publish_target("")
            return False, "stow_failed_after_release"

        self.carried_target_label = None
        self.selected_target_label = None
        self.no_payload = False
        self.completed_rounds += 1
        self.publish_target("")
        # 正常前进放置成功后跳过剩余候选点；回退降级放置发生在三个点均已
        # 消费之后，不再发布历史点跳过名单。
        remaining_place_stops = (
            tuple(
                "piper_stop_%d" % stop_number
                for stop_number in range(candidate + 5, 8)
            )
            if candidate is not None
            else ()
        )
        self.publish_navigation_skip(remaining_place_stops)
        if candidate is None:
            return True, "fallback_any:placed_on=%s:round=%d/%d" % (
                placed_on_label or "unknown",
                self.completed_rounds,
                self.max_rounds,
            )
        return True, "round=%d/%d" % (self.completed_rounds, self.max_rounds)

    def execute_prepare_place_scan(self):
        """放置回退前/到点后确认机械臂保持持物观察姿态。"""
        if self.no_payload or self.carried_target_label is None:
            return False, "place_recovery_without_payload"

        self.publish_state("place_recovery:moving_to_scan")
        if self.is_joint_pose_reached(PLACE_SCAN_JOINTS):
            rospy.loginfo("放置回退：机械臂已在放置观察位，跳过重复关节运动。")
            return True, "place_scan_already_ready"
        if not self.move_joint_pose(
            PLACE_SCAN_JOINTS,
            speed_percent=self.observation_joint_speed_percent,
        ):
            return False, "place_scan_failed"
        return True, "place_scan_ready"

    def execute_place_any(self):
        """正确目标全部失败后，放到任意稳定且具备安全位姿的已知物体处。"""
        if self.no_payload or self.carried_target_label is None:
            return False, "fallback_any_without_payload"

        self.publish_state("place_any:locating_safe_object")
        self.arm.detection_context = "place_any"
        T_base_place, detected_label = self.arm.get_any_place_pose()
        if T_base_place is None:
            return False, "no_safe_fallback_object"

        rospy.logwarn(
            "正确目标复查失败：手持 %s，降级放置到 %s。",
            self.carried_target_label,
            detected_label or "unknown",
        )
        return self.execute_place_at_pose(
            T_base_place,
            state_prefix="place_any",
            placed_on_label=detected_label,
            candidate=None,
        )

    def execute_prepare_pick_scan(self):
        """回退车辆已经停稳后，重新展开到取货观察姿态。"""
        if self.carried_target_label is not None:
            return False, "already_carrying=%s" % self.carried_target_label
        if self.selected_target_label is None:
            return False, "card_target_not_selected"
        if self.no_payload:
            return False, "round_already_aborted"

        self.publish_state("recovery:moving_to_pick_scan")
        # 回退期间机械臂通常已保持在该观察位。若反馈确认已到位，直接放行，
        # 避免每次恢复都重复发送关节运动并再次等待实时反馈收敛。
        if self.is_joint_pose_reached(PICK_SCAN_JOINTS):
            rospy.loginfo("回退恢复：机械臂已在抓取观察位，跳过重复关节运动。")
            return True, "pick_scan_already_ready"
        if not self.move_joint_pose(
            PICK_SCAN_JOINTS,
            speed_percent=self.observation_joint_speed_percent,
        ):
            return False, "pick_scan_failed"
        return True, "pick_scan_ready"

    def execute_abort_round(self):
        """抓取恢复也失败：收臂、标记空载并跳过本轮全部卸货动作。"""
        self.publish_state("abort_round:moving_to_transport")
        stowed, stow_reason = self.execute_stow()
        if not stowed:
            return False, "stow_failed_during_abort=%s" % stow_reason

        self.carried_target_label = None
        self.selected_target_label = None
        self.no_payload = True
        self.publish_target("")
        # 车辆仍沿原路线经过卸货区，但三个卸货候选点都不再停车或动臂。
        self.publish_navigation_skip(
            ("piper_stop_5", "piper_stop_6", "piper_stop_7")
        )
        self.publish_state("no_payload")
        return True, "no_payload"

    def execute_discard_place(self):
        """三个放置候选点均失败：松开物品，确认收臂后结束本轮。

        carried_target_label 在张爪后立即清除，避免回零重试期间仍把已经丢弃的
        物品当成在手。selected_target_label 则保留到回零确认成功，既标记本轮
        尚未安全收尾，也保证重复执行本命令时不会重复累计 completed_rounds。
        """
        if self.carried_target_label is not None:
            self.publish_state("discard_place:opening_gripper")
            rospy.logwarn("三个放置候选点均失败：在第三点张开夹爪丢弃物品。")
            self.open_gripper_at_current_pose()
            self.wait_after_motion(2.0)
            self.carried_target_label = None
            self.publish_target("")

        self.publish_state("discard_place:moving_to_transport")
        stowed, stow_reason = self.execute_stow()
        if not stowed:
            return False, "stow_failed_after_discard=%s" % stow_reason

        # 只有仍有本轮锁定目标时才累计一次。若回零失败后重发本命令，目标会
        # 一直保留到本次成功，因此不会少计；成功后清空，也不会重复累计。
        if self.selected_target_label is not None:
            self.completed_rounds = min(
                self.max_rounds, self.completed_rounds + 1
            )
        self.selected_target_label = None
        self.no_payload = False
        self.publish_target("")
        # 清除上一阶段的跳点名单，保证下一轮任务点正常执行。
        self.publish_navigation_skip(())
        self.publish_state("idle")
        return True, "discarded_and_stowed:round=%d/%d" % (
            self.completed_rounds,
            self.max_rounds,
        )

    def execute_stow(self):
        if not self.move_joint_pose(TRANSPORT_JOINTS):
            return False, "stow_path_failed"
        return True, "stowed_verified"

    def on_shutdown(self):
        try:
            self.arm.pipeline.stop()
        except Exception:
            pass
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            pass


def main():
    arm = CompetitionVisionController()
    node = CompetitionTaskNode(arm)
    node.run()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except RuntimeError as exc:
        rospy.logfatal("piper_task 启动失败：%s", exc)
        raise
