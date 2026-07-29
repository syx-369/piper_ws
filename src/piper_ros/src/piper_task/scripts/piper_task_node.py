#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""正式比赛用 Piper 抓取/放置任务节点。

视觉识别、深度定位、手眼变换、笛卡尔插值以及抓放偏移全部调用
vision_grasp_core.py；该文件是 grab2016_7_21.py 的原样副本。
本节点只增加：任务触发、抓放阶段拆分、已录制路径播放以及两轮状态管理。
"""

import queue
import threading
import time

import numpy as np
import rospy
import tf.transformations as tf_trans
from geometry_msgs.msg import PoseStamped
from piper_msgs.msg import PosCmd
from sensor_msgs.msg import JointState
from std_msgs.msg import String

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
JOINT_MOVE_WAIT = 8.0


class CompetitionVisionController(PiperVisionController):
    """使用固定 ROS 节点名初始化原测试脚本的视觉控制器。"""

    def __init__(self):
        # 正式包需要稳定的 /piper_task 私有参数空间，因此这里只替换原类中
        # anonymous=True 的节点初始化；其余话题、反馈等待、相机和模型初始化一致。
        rospy.init_node("piper_task", anonymous=False)

        self.pub = rospy.Publisher("/pin_pos_cmd", PosCmd, queue_size=1)
        self.joint_pub = rospy.Publisher("/joint_states", JointState, queue_size=1)

        self.current_ee_pose_mat = np.eye(4)
        self.pose_received = False
        rospy.Subscriber("/end_pose", PoseStamped, self.pose_callback)

        rospy.loginfo("等待获取机械臂实时位姿 (/end_pose)...")
        while not self.pose_received and not rospy.is_shutdown():
            rospy.sleep(0.1)
        if rospy.is_shutdown():
            raise rospy.ROSInterruptException("等待 /end_pose 时节点被关闭")
        rospy.loginfo("成功接入实时位姿反馈！")

        self.init_camera_and_detector()


class CompetitionTaskNode:
    """把原测试程序的一次连续抓放拆成车辆可触发的 pick/place。"""

    VALID_COMMANDS = {
        "card",
        "pick", "pick1", "pick2", "pick3",
        "place", "place1", "place2", "place3",
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
            "piper_task ready: command topic=%s, commands=card/pick1-3/place1-3/stow/reset/status",
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
                "status:busy=%s,selected=%s,carried=%s,rounds=%d/%d"
                % (
                    self.busy,
                    selected,
                    carried,
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
                self.publish_state("idle")
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
        elif command == "stow":
            ok, reason = self.execute_stow()
        elif command == "reset":
            self.selected_target_label = None
            self.carried_target_label = None
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

    def move_joint_pose(self, positions):
        """发送原测试脚本格式的单个关节姿态并等待8秒。"""
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = [""]
        msg.position = list(positions)
        msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]
        msg.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
        self.arm.joint_pub.publish(msg)
        time.sleep(JOINT_MOVE_WAIT)
        return not rospy.is_shutdown()

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
        if self.carried_target_label is not None:
            return False, "already_carrying=%s" % self.carried_target_label
        if self.completed_rounds >= self.max_rounds:
            return False, "all_rounds_completed"

        self.selected_target_label = None
        self.publish_target("")
        self.publish_navigation_skip(())

        self.publish_state("card:moving_to_scan")
        if not self.move_joint_pose(PICK_SCAN_JOINTS):
            return False, "reference_scan_failed"

        self.publish_state("card:recognizing_reference")
        selected_label = self.arm.recognize_reference_target()
        target_type = self.arm.target_type_from_label(selected_label)
        if target_type is None:
            return False, "reference_not_recognized"

        # 7_21 脚本在识别卡片后固定等待10秒，并在此锁定图片样式。
        time.sleep(10.0)
        self.selected_target_label = selected_label
        self.publish_target(selected_label)

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
        T_base_object = self.arm.get_object_pose(target_type, selected_label)
        if T_base_object is None:
            if candidate < 3:
                return True, "not_here_continue_to_pick%d" % (candidate + 1)
            return False, "object_not_found_at_all_pick_points"

        # 以下偏移、速度、夹爪值和等待时间严格沿用原测试脚本。
        self.publish_state("pick%d:moving_to_pregrasp" % candidate)
        T_pre = T_OBJECT_TO_GRASP.copy()
        T_pre[0:3, 3] = [-0.04, -0.02, -0.05]
        target_ee_pre = T_base_object @ T_pre @ T_EE_TO_TOOL
        self.arm.move_to_target_smooth(target_ee_pre, v=0.06, gripper_val=100)
        time.sleep(3.5)

        self.publish_state("pick%d:approaching" % candidate)
        T_approach = T_OBJECT_TO_GRASP.copy()
        T_approach[0:3, 3] = [-0.04, -0.02, 0.06]
        self.arm.move_to_target_smooth(
            T_base_object @ T_approach @ T_EE_TO_TOOL,
            v=0.02,
            gripper_val=100,
        )
        time.sleep(2.5)

        self.publish_state("pick%d:closing_gripper" % candidate)
        self.close_gripper_at_current_pose()
        time.sleep(2.0)

        self.publish_state("pick%d:lifting" % candidate)
        T_up = T_OBJECT_TO_GRASP.copy()
        T_up[0:3, 3] = [-0.15, 0.0, 0.06]
        self.arm.move_to_target_smooth(
            T_base_object @ T_up @ T_EE_TO_TOOL,
            v=0.05,
            gripper_val=0,
        )
        time.sleep(3.0)

        self.publish_state("pick%d:moving_to_transport" % candidate)
        if not self.move_joint_pose(TRANSPORT_JOINTS):
            return False, "transport_path_failed_after_grasp"

        # 7_21 连续测试在抓取归位后额外等待2秒，再进入放置阶段。
        # 七点功能测试随后虽会行车，仍保留该机械臂流程等待时间。
        time.sleep(2.0)

        self.carried_target_label = selected_label
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
        if candidate == 1 and not self.move_joint_pose(PLACE_SCAN_JOINTS):
            return False, "place_scan_failed"

        # 与 7_21 脚本一致：只定位任务开始时锁定的物品类型+颜色图片。
        self.publish_state("place%d:locating_target" % candidate)
        T_base_place_image = self.arm.get_place_pose_by_image_style(target_label)
        if T_base_place_image is None:
            if candidate < 3:
                return True, "not_here_continue_to_place%d" % (candidate + 1)
            return False, "target_not_found_at_all_place_points"

        # 以下放置偏移、速度、夹爪值和等待时间严格沿用原测试脚本。
        self.publish_state("place%d:moving_to_release" % candidate)
        T_pre = T_OBJECT_TO_GRASP.copy()
        T_pre[0:3, 3] = [-0.10, -0.04, -0.06]
        target_ee_pre = T_base_place_image @ T_pre @ T_EE_TO_TOOL
        self.arm.move_to_target_smooth(target_ee_pre, v=0.06, gripper_val=0)
        time.sleep(3.5)

        self.publish_state("place%d:opening_gripper" % candidate)
        self.open_gripper_at_current_pose()
        time.sleep(2.0)

        self.publish_state("place%d:moving_to_stow" % candidate)
        if not self.move_joint_pose(TRANSPORT_JOINTS):
            # 物品已经放下，保留准确状态，防止误以为仍在持物。
            self.carried_target_label = None
            self.publish_target("")
            return False, "stow_failed_after_release"

        self.carried_target_label = None
        self.selected_target_label = None
        self.completed_rounds += 1
        self.publish_target("")
        # candidate=1 对应车辆点5；成功后点6、7无需停车。
        remaining_place_stops = tuple(
            "piper_stop_%d" % stop_number
            for stop_number in range(candidate + 5, 8)
        )
        self.publish_navigation_skip(remaining_place_stops)
        return True, "round=%d/%d" % (self.completed_rounds, self.max_rounds)

    def execute_stow(self):
        if not self.move_joint_pose(TRANSPORT_JOINTS):
            return False, "stow_path_failed"
        return True, "stowed"

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
