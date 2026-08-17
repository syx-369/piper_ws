#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""机械臂零位下的独立红绿灯识别测试。

复用 final_mission/final_vision.py 的比赛 YOLO + HSV 红绿灯算法，直接独占
D435i 彩色相机。本节点不发布机械臂命令或车辆速度，识别结果仅发布到测试
话题 ``/piper_task_test/traffic_light``，不会驱动比赛总控。

运行：
  rosrun piper_task traffic_light_test.py

按 q、Esc 或 Ctrl+C 退出；退出时只释放相机，不移动机械臂。
"""

import os
import sys

import cv2
import rosnode
import rospy
import rospkg
from std_msgs.msg import String


WINDOW_NAME = "Traffic Light Test"


def append_default_ros_arg(name, value):
    prefix = "_%s:=" % name
    if not any(argument.startswith(prefix) for argument in sys.argv[1:]):
        sys.argv.append("%s%s" % (prefix, value))


# FinalVision 自己调用 rospy.init_node()，所以必须在构造前以 ROS 私有参数形式
# 指定测试模式。保留 conf/imgsz/device 等参数由操作者按需覆盖。
if not any(argument.startswith("__name:=") for argument in sys.argv[1:]):
    sys.argv.append("__name:=traffic_light_test")
append_default_ros_arg("vision_source", "standalone")
append_default_ros_arg("initial_mode", "light")
append_default_ros_arg("show_image", "true")
append_default_ros_arg(
    "traffic_weights",
    "/home/user/fastlio_ws/src/waypoint_tools/config/traffic_light.pt",
)

try:
    FINAL_MISSION_SCRIPTS = os.path.join(
        rospkg.RosPack().get_path("final_mission"), "scripts"
    )
except rospkg.ResourceNotFound:
    FINAL_MISSION_SCRIPTS = "/home/user/fastlio_ws/src/final_mission/scripts"

if FINAL_MISSION_SCRIPTS not in sys.path:
    sys.path.insert(0, FINAL_MISSION_SCRIPTS)

from final_vision import FinalVision  # noqa: E402


class TrafficLightTest(FinalVision):
    """只保留正式红绿灯推理，隔离正式总控话题和控制命令。"""

    def __init__(self):
        super().__init__()
        if self.display_ok:
            try:
                cv2.destroyWindow("Final Mission Vision")
            except cv2.error:
                pass
        if self.source != "standalone":
            raise ValueError("traffic_light_test 固定要求 vision_source=standalone")
        if self.mode != "light":
            raise ValueError("traffic_light_test 固定要求 initial_mode=light")

        self.ensure_camera_is_available()
        self.light_pub = rospy.Publisher(
            "/piper_task_test/traffic_light", String, queue_size=1
        )
        rospy.loginfo(
            "红绿灯测试就绪：不移动机械臂、不控制车辆，结果话题=%s",
            self.light_pub.resolved_name,
        )

    def ensure_camera_is_available(self):
        """在打开相机前拒绝与其他视觉节点并发，避免设备争用。"""
        try:
            active_nodes = set(rosnode.get_node_names())
        except Exception as exc:
            raise RuntimeError("无法检查相机冲突节点: %s" % exc)

        conflicts = []
        for node_name in sorted(active_nodes):
            if node_name == rospy.get_name():
                continue
            base_name = node_name.rsplit("/", 1)[-1].lower()
            if base_name in {
                "piper_task",
                "piper_task_plane",
                "place_plane_depth_test",
                "place_bottle_depth_ir_test",
                "pick_object_depth_ir_test",
                "final_vision",
            } or "realsense2_camera" in base_name:
                conflicts.append(node_name)
        if conflicts:
            raise RuntimeError(
                "检测到占用D435i的节点 %s；请先关闭后再运行红绿灯测试。"
                % conflicts
            )

    def control_callback(self, msg):
        """忽略正式总控的模式切换，测试过程始终保持 light。"""
        rospy.logwarn_throttle(
            5.0,
            "红绿灯独立测试忽略 /final_mission/vision_control=%s",
            (msg.data or "").strip(),
        )

    def publish_status(self):
        """不写正式 /final_mission/vision_status。"""

    def spin(self):
        rate = rospy.Rate(30)
        window_created = False
        try:
            while not rospy.is_shutdown():
                frame = self.grab_frame()
                if frame is None:
                    rate.sleep()
                    continue

                display = frame.copy()
                self.process_light(frame, display)

                if self.display_ok:
                    try:
                        cv2.imshow(WINDOW_NAME, display)
                        window_created = True
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            rospy.signal_shutdown("用户退出红绿灯测试")
                        elif (
                            window_created
                            and cv2.getWindowProperty(
                                WINDOW_NAME, cv2.WND_PROP_VISIBLE
                            ) < 1.0
                        ):
                            rospy.signal_shutdown("用户关闭红绿灯测试窗口")
                    except cv2.error as exc:
                        rospy.logwarn("红绿灯测试窗口不可用，停止显示: %s", exc)
                        self.display_ok = False
                rate.sleep()
        finally:
            self.on_shutdown()
            rospy.loginfo(
                "红绿灯测试已退出；相机已释放，未发送机械臂或车辆控制命令。"
            )


def main():
    TrafficLightTest().spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except Exception as exc:
        try:
            rospy.logfatal("红绿灯独立测试启动/运行失败: %s", exc)
        except Exception:
            pass
        raise
