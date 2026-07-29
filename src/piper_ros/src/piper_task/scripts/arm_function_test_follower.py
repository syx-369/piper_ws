#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""机械臂功能验证专用的普通路径跟踪节点。

复用 waypoint_tools/follow_waypoints.py 的原有跟踪和外部任务握手，不启用
任何避障。唯一扩展是：收到 piper_task 的 navigation_skip 后，将本轮后续
抓取/放置候选点改为普通航点，从而连续通过、不停车。
"""

import os
import sys

import rospkg
import rospy
from std_msgs.msg import String


WAYPOINT_TOOLS_SCRIPTS = os.path.join(
    rospkg.RosPack().get_path("waypoint_tools"), "scripts"
)
if WAYPOINT_TOOLS_SCRIPTS not in sys.path:
    sys.path.insert(0, WAYPOINT_TOOLS_SCRIPTS)

from follow_waypoints import WaypointFollower  # noqa: E402


ARM_EXTERNAL_TASKS = {
    "piper_stop_1",
    "piper_stop_2",
    "piper_stop_3",
    "piper_stop_4",
    "piper_stop_5",
    "piper_stop_6",
    "piper_stop_7",
}


class ArmFunctionTestFollower(WaypointFollower):
    def __init__(self):
        super().__init__()

        self.skip_topic = rospy.get_param(
            "~skip_topic", "/piper_task/navigation_skip"
        )
        self.arm_state_topic = rospy.get_param(
            "~arm_state_topic", "/piper_task/state"
        )
        self.arm_ready_timeout = float(rospy.get_param("~arm_ready_timeout", 120.0))
        self.required_task_sets = max(
            1, int(rospy.get_param("~required_task_sets", 1))
        )

        self.keep_only_arm_tasks()
        self.validate_arm_task_points()
        rospy.Subscriber(self.skip_topic, String, self.skip_callback, queue_size=10)

        rospy.loginfo("等待机械臂任务节点就绪: %s", self.arm_state_topic)
        try:
            state = rospy.wait_for_message(
                self.arm_state_topic, String, timeout=self.arm_ready_timeout
            )
        except rospy.ROSException as exc:
            raise RuntimeError("等待 piper_task 就绪超时: %s" % exc)
        rospy.loginfo("机械臂任务节点已就绪，state=%s，开始普通路径跟踪", state.data)

    @staticmethod
    def external_name(task):
        task = (task or "").strip()
        if not task.startswith("ext:"):
            return None
        return task[4:].strip()

    def keep_only_arm_tasks(self):
        """功能验证时忽略红绿灯、避障区等其他 CSV 任务标记。"""
        ignored = 0
        for waypoint in self.waypoints:
            task = waypoint.get("task", "none")
            external_name = self.external_name(task)
            if external_name in ARM_EXTERNAL_TASKS:
                continue
            if task not in ("", "none"):
                ignored += 1
            waypoint["task"] = "none"
        rospy.loginfo("机械臂功能验证：已将 %d 个非机械臂任务点作为普通航点", ignored)

    def validate_arm_task_points(self):
        counts = {name: 0 for name in ARM_EXTERNAL_TASKS}
        for waypoint in self.waypoints:
            external_name = self.external_name(waypoint.get("task", "none"))
            if external_name in counts:
                counts[external_name] += 1

        missing = [
            name
            for name, count in sorted(counts.items())
            if count < self.required_task_sets
        ]
        if missing:
            raise RuntimeError(
                "CSV 缺少机械臂测试任务点（每个至少%d次）: %s"
                % (self.required_task_sets, ", ".join(missing))
            )
        rospy.loginfo("七个机械臂任务点校验通过: %s", counts)

    def skip_callback(self, msg):
        skip_names = {
            name.strip() for name in msg.data.split(",") if name.strip()
        }
        if not skip_names:
            return

        changed = []
        # 只修改当前这一轮；遇到下一次 piper_stop_1 就停止，避免影响第二轮。
        for index in range(self.current_index, len(self.waypoints)):
            waypoint = self.waypoints[index]
            external_name = self.external_name(waypoint.get("task", "none"))
            if index > self.current_index and external_name == "piper_stop_1":
                break
            if external_name in skip_names:
                waypoint["task"] = "none"
                changed.append("%s@seq%d" % (external_name, waypoint["seq"]))

        if changed:
            rospy.loginfo("后续候选点并入普通跟踪，不停车: %s", ", ".join(changed))


if __name__ == "__main__":
    rospy.init_node("arm_function_test_follower", anonymous=False)
    try:
        node = ArmFunctionTestFollower()
        node.spin()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        rospy.logfatal("机械臂功能测试跟踪节点启动失败: %s", exc)
        raise
