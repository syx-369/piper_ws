#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""独立的抓取/放置候选点观察与 D435i 诊断节点。

运行顺序：
1. 将 Piper 移动到比赛主控使用的抓取或放置观察位；
2. 启动 D435i 的 RGB、Depth、Left IR、Right IR 四路图像；
3. 显示模型支持的全部目标及深度信息，辅助录制车辆候选点。

本节点不会控制车辆，也不会执行抓取或放置。Ctrl+C 或关闭窗口后默认把
机械臂送回主控运输零位；正在运行的航迹录制节点不受影响。
放置模式在确认到达观察位后默认发送 gripper=200 张开夹爪；抓取模式不
额外发送张爪命令。

启动参数 ``_enable_vision:=false`` 可关闭模型和相机，只移动到观察位并
保持；按 Ctrl+C 后仍按默认设置返回零位。

示例：
  rosrun piper_task place_bottle_depth_ir_test.py

旧命令中的 _target_label 会被忽略，避免 ROS 参数残留导致漏检其他类别。
"""

import os
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import rosnode
import rospy
import tf.transformations as tf_trans
from geometry_msgs.msg import PoseStamped
from piper_msgs.msg import PosCmd
from sensor_msgs.msg import JointState
from ultralytics import YOLO

from piper_task.vision_grasp_core import (
    DEPTH_FALLBACK_MIN_VALID_RATIO,
    DEPTH_STABILITY_MAX_SPREAD,
    DEPTH_STABLE_SAMPLES,
    PICK_DEPTH_MAX,
    PICK_DEPTH_MIN,
    SHRINK_RATIO,
    get_depth_percentile,
    get_filtered_depth_in_range,
    is_depth_in_range,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_SCRIPT_DIR = "/home/user/piper_ws/src/piper_ros/src/piper_task/scripts"
for candidate_dir in (SCRIPT_DIR, SOURCE_SCRIPT_DIR):
    if candidate_dir not in sys.path:
        sys.path.insert(0, candidate_dir)

# 复用现有独立平面测试文件中的纯算法和绘图函数。导入不会启动 ROS 节点、
# 相机或机械臂；该文件的 main() 有 __name__ 保护。
from place_plane_depth_test import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    DEFAULT_PLACE_SCAN_JOINTS,
    LABEL_COLORS,
    SUPPORTED_LABELS,
    PlacePlaneEstimator,
    StablePlaneDepth,
    apply_depth_filters,
    build_depth_filters,
    clip_rect,
    draw_estimator_result,
    make_depth_colormap,
)


COLOR_WIDTH = 640
COLOR_HEIGHT = 480
DEPTH_WIDTH = 640
DEPTH_HEIGHT = 480
IR_WIDTH = 640
IR_HEIGHT = 480
FPS = 30

WINDOW_RGB_DEPTH = "Place Test - RGB + Depth"
WINDOW_STEREO_IR = "Place Test - Left IR + Right IR"
DEFAULT_SAVE_DIR = "~/.ros/place_bottle_depth_ir_test"
DEFAULT_PICK_SCAN_JOINTS = [-1.530, 0.446, 0.0, 0.0, -0.115, 0.0, 0.0]

# 与 task_plane_config.yaml 当前比赛参数保持一致。仅当调用者没有显式传入
# 对应私有参数时才设置默认值，便于 rosrun 单文件启动也能复现比赛算法。
COMPETITION_PLANE_DEFAULTS = {
    "depth_min": 0.30,
    "depth_max": 0.70,
    "ring_pad_ratio_x": 0.35,
    "ring_pad_ratio_y": 0.25,
    "ring_min_pad_x": 12,
    "ring_min_pad_y": 10,
    "ring_max_pad_x": 55,
    "ring_max_pad_y": 45,
    "ring_gap": 2,
    "min_patch_valid_ratio": 0.05,
    "min_patch_valid_pixels": 20,
    "max_patch_mad": 0.035,
    "patch_consistency": 0.080,
    "min_consistent_patches": 2,
    "patch_plane_consistency": 0.055,
    "max_plane_points": 2400,
    "ransac_iterations": 100,
    "plane_inlier_threshold": 0.015,
    "min_plane_inliers": 60,
    "min_plane_inlier_ratio": 0.45,
    "min_patch_plane_inliers": 10,
    "max_plane_rmse": 0.015,
    "min_abs_plane_normal_z": 0.45,
}

LABEL_NAMES_ZH = {
    "block-r": "红色方块",
    "block-y": "黄色方块",
    "block-b": "蓝色方块",
    "bottle-b": "黑色瓶子",
    "bottle-y": "黄色瓶子",
    "bottle-g": "绿色瓶子",
}


def set_competition_plane_defaults():
    for name, value in COMPETITION_PLANE_DEFAULTS.items():
        private_name = "~" + name
        if not rospy.has_param(private_name):
            rospy.set_param(private_name, value)


def edge_directions(
    box,
    width,
    height,
    margin_x,
    margin_y,
    observation_mode="place",
    pick_hard_border_margin=8,
):
    x1, y1, x2, y2 = box
    if observation_mode == "pick":
        center_x = 0.5 * (x1 + x2)
        directions = []
        if x1 < pick_hard_border_margin or center_x < margin_x:
            directions.append("left")
        if (
            x2 > width - 1 - pick_hard_border_margin
            or center_x > width - 1 - margin_x
        ):
            directions.append("right")
        return directions

    directions = []
    if x1 < margin_x:
        directions.append("left")
    if x2 > width - 1 - margin_x:
        directions.append("right")
    if y1 < margin_y:
        directions.append("top")
    if y2 > height - 1 - margin_y:
        directions.append("bottom")
    return directions


def collect_matching_detections(
    results,
    model_names,
    target_label,
    width,
    height,
    margin_x,
    margin_y,
    observation_mode="place",
    pick_hard_border_margin=8,
):
    """收集全部支持类别（或可选指定类别）；贴边目标也继续显示。"""
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return []

    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    candidates = []
    for raw_box, confidence, class_id in zip(boxes, confidences, class_ids):
        label = str(model_names[int(class_id)])
        if label not in SUPPORTED_LABELS:
            continue
        if target_label and label != target_label:
            continue
        x1, y1, x2, y2 = clip_rect(raw_box, width, height)
        if x2 <= x1 or y2 <= y1:
            continue
        center_x = 0.5 * (x1 + x2)
        center_y = 0.5 * (y1 + y2)
        center_penalty = abs(center_x - width / 2.0) / width + abs(
            center_y - height / 2.0
        ) / height
        area_ratio = float((x2 - x1) * (y2 - y1)) / float(width * height)
        candidates.append(
            {
                "box": (x1, y1, x2, y2),
                "confidence": float(confidence),
                "label": label,
                "score": float(confidence) + 0.10 * area_ratio - 0.08 * center_penalty,
                "edge_directions": edge_directions(
                    (x1, y1, x2, y2),
                    width,
                    height,
                    margin_x,
                    margin_y,
                    observation_mode,
                    pick_hard_border_margin,
                ),
            }
        )
    if not candidates:
        return []
    # 固定按画面从左到右、从上到下编号，便于用 _target_index 手动选实例。
    candidates.sort(
        key=lambda item: (
            0.5 * (item["box"][0] + item["box"][2]),
            0.5 * (item["box"][1] + item["box"][3]),
        )
    )
    for index, candidate in enumerate(candidates):
        candidate["index"] = index
    return candidates


def center_roi_depth_stats(
    depth_m,
    box,
    roi_ratio,
    task_depth_min,
    task_depth_max,
):
    """同时保留传感器原始中位数和比赛有效范围内的中心 ROI 统计。"""
    height, width = depth_m.shape[:2]
    x1, y1, x2, y2 = clip_rect(box, width, height)
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    roi_width = max(5, int(round(max(1, x2 - x1) * roi_ratio)))
    roi_height = max(5, int(round(max(1, y2 - y1) * roi_ratio)))
    roi_x1 = max(0, int(round(center_x - 0.5 * roi_width)))
    roi_y1 = max(0, int(round(center_y - 0.5 * roi_height)))
    roi_x2 = min(width, max(roi_x1 + 1, roi_x1 + roi_width))
    roi_y2 = min(height, max(roi_y1 + 1, roi_y1 + roi_height))
    roi = depth_m[roi_y1:roi_y2, roi_x1:roi_x2]

    center_ix = int(np.clip(round(center_x), 0, width - 1))
    center_iy = int(np.clip(round(center_y), 0, height - 1))
    center_pixel = float(depth_m[center_iy, center_ix])
    if not np.isfinite(center_pixel) or center_pixel <= 0.0:
        center_pixel = None

    sensor_values = roi[
        np.isfinite(roi) & (roi > 0.10) & (roi < 8.0)
    ]
    task_values = roi[
        np.isfinite(roi)
        & (roi > float(task_depth_min))
        & (roi < float(task_depth_max))
    ]
    task_count = int(task_values.size)
    task_ratio = float(task_count) / float(roi.size) if roi.size else 0.0
    sensor_median = (
        float(np.median(sensor_values)) if sensor_values.size else None
    )
    task_median = float(np.median(task_values)) if task_values.size else None
    task_p20 = (
        float(np.percentile(task_values, 20.0)) if task_values.size else None
    )
    task_mad = (
        float(np.median(np.abs(task_values - task_median)))
        if task_values.size
        else None
    )
    return {
        "center": (center_x, center_y),
        "center_pixel": center_pixel,
        "roi": (roi_x1, roi_y1, roi_x2, roi_y2),
        "roi_pixels": int(roi.size),
        "sensor_count": int(sensor_values.size),
        "sensor_median": sensor_median,
        "task_count": task_count,
        "task_ratio": task_ratio,
        "task_median": task_median,
        "task_p20": task_p20,
        "task_mad": task_mad,
    }


def format_depth(value):
    return "--" if value is None else "%.3f" % float(value)


def draw_panel_title(image, title):
    cv2.rectangle(image, (0, 0), (image.shape[1] - 1, 30), (0, 0, 0), -1)
    cv2.putText(
        image,
        title,
        (10, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_bottom_lines(image, lines, color=(255, 255, 255)):
    if not lines:
        return
    line_height = 20
    block_height = 8 + line_height * len(lines)
    y1 = max(0, image.shape[0] - block_height)
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (0, y1),
        (image.shape[1] - 1, image.shape[0] - 1),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.68, image, 0.32, 0.0, image)
    y = y1 + 18
    for line in lines:
        cv2.putText(
            image,
            str(line),
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            color,
            1,
            cv2.LINE_AA,
        )
        y += line_height


class PlaceBottleDepthIrTestNode:
    def __init__(self):
        self.observation_mode = str(
            rospy.get_param("~observation_mode", "place")
        ).strip().lower()
        if self.observation_mode not in {"pick", "place"}:
            raise ValueError("observation_mode 只能是 pick 或 place")
        self.region_name_zh = "抓取" if self.observation_mode == "pick" else "放置"
        self.window_rgb_depth = "%s Test - RGB + Depth" % self.observation_mode.title()
        self.window_stereo_ir = "%s Test - Left IR + Right IR" % self.observation_mode.title()
        self.enable_vision = bool(rospy.get_param("~enable_vision", True))

        legacy_target_label = str(
            rospy.get_param("~target_label", "")
        ).strip()
        if legacy_target_label:
            rospy.logwarn(
                "候选点观察固定使用全类别模式，忽略旧参数 target_label=%s。",
                legacy_target_label,
            )
        self.target_label = ""

        self.model_path = str(rospy.get_param("~model_path", DEFAULT_MODEL_PATH))
        self.conf_threshold = float(rospy.get_param("~conf_threshold", 0.50))
        self.image_size = int(rospy.get_param("~image_size", 640))
        self.target_index = int(rospy.get_param("~target_index", -1))
        if self.target_index < -1:
            raise ValueError("target_index 必须为-1（自动）或从0开始的实例编号")
        self.margin_x = max(0, int(rospy.get_param("~safe_margin_x", 40)))
        self.margin_y = max(0, int(rospy.get_param("~safe_margin_y", 30)))
        self.pick_hard_border_margin = max(
            0, int(rospy.get_param("~pick_hard_border_margin", 8))
        )
        self.center_roi_ratio = min(
            0.80,
            max(0.05, float(rospy.get_param("~center_roi_ratio", 0.30))),
        )
        self.center_min_valid_pixels = max(
            1, int(rospy.get_param("~center_min_valid_pixels", 25))
        )
        self.center_min_valid_ratio = min(
            1.0,
            max(0.0, float(rospy.get_param("~center_min_valid_ratio", 0.10))),
        )
        self.center_max_mad = max(
            0.0, float(rospy.get_param("~center_max_mad", 0.035))
        )

        self.show_window = bool(rospy.get_param("~show_window", True))
        self.display_scale = min(
            1.50, max(0.25, float(rospy.get_param("~display_scale", 0.75)))
        )
        self.display_depth_min = float(rospy.get_param("~display_depth_min", 0.20))
        self.display_depth_max = float(rospy.get_param("~display_depth_max", 4.00))
        self.save_dir = os.path.expanduser(
            str(
                rospy.get_param(
                    "~save_dir",
                    "~/.ros/%s_observation_depth_ir_test" % self.observation_mode,
                )
            )
        )
        self.device_serial = str(rospy.get_param("~device_serial", "")).strip()
        self.warmup_frames = max(0, int(rospy.get_param("~warmup_frames", 30)))
        self.allow_conflicting_nodes = bool(
            rospy.get_param("~allow_conflicting_nodes", False)
        )

        self.move_arm_to_observation = bool(
            rospy.get_param("~move_arm_to_observation", True)
        )
        self.arm_was_commanded_to_observation = False
        self.open_gripper_at_observation = bool(
            rospy.get_param(
                "~open_gripper_at_observation",
                self.observation_mode == "place",
            )
        )
        self.return_to_zero_on_exit = bool(
            rospy.get_param("~return_to_zero_on_exit", True)
        )
        self.return_zero_timeout = max(
            0.5, float(rospy.get_param("~return_zero_timeout", 12.0))
        )
        self.arm_feedback_timeout = max(
            0.1, float(rospy.get_param("~arm_feedback_timeout", 10.0))
        )
        self.arm_move_wait = max(
            0.0, float(rospy.get_param("~arm_move_wait", 8.0))
        )
        self.arm_verify_timeout = max(
            0.1, float(rospy.get_param("~arm_verify_timeout", 5.0))
        )
        self.arm_joint_tolerance = max(
            0.001, float(rospy.get_param("~arm_joint_tolerance", 0.05))
        )
        self.arm_stable_samples = max(
            1, int(rospy.get_param("~arm_stable_samples", 3))
        )
        self.arm_feedback_max_age = max(
            0.1, float(rospy.get_param("~arm_feedback_max_age", 1.0))
        )
        self.gripper_open_value = float(
            rospy.get_param("~gripper_open_value", 200.0)
        )
        self.gripper_open_wait = max(
            0.0, float(rospy.get_param("~gripper_open_wait", 2.0))
        )
        self.joint_command_topic = str(
            rospy.get_param("~joint_command_topic", "/joint_states")
        )
        self.joint_feedback_topic = str(
            rospy.get_param("~joint_feedback_topic", "/joint_states_single")
        )
        self.end_pose_topic = str(rospy.get_param("~end_pose_topic", "/end_pose"))
        self.cartesian_command_topic = str(
            rospy.get_param("~cartesian_command_topic", "/pin_pos_cmd")
        )

        default_scan_joints = (
            DEFAULT_PICK_SCAN_JOINTS
            if self.observation_mode == "pick"
            else DEFAULT_PLACE_SCAN_JOINTS
        )
        legacy_param = "~%s_scan_joints" % self.observation_mode
        scan_joints = rospy.get_param(
            "~observation_joints",
            rospy.get_param(legacy_param, list(default_scan_joints)),
        )
        if not isinstance(scan_joints, (list, tuple)) or len(scan_joints) != 7:
            raise ValueError("observation_joints 必须是含7个数值的关节姿态")
        self.observation_joints = [float(value) for value in scan_joints]
        if not np.all(np.isfinite(np.asarray(self.observation_joints, dtype=float))):
            raise ValueError("observation_joints 不能包含NaN或无穷值")
        self.observation_gripper_value = float(
            rospy.get_param(
                "~observation_gripper_value", self.observation_joints[6]
            )
        )
        if not 0.0 <= self.observation_gripper_value <= 200.0:
            raise ValueError("observation_gripper_value 必须位于0到200之间")
        self.observation_joints[6] = self.observation_gripper_value

        if self.enable_vision and not os.path.exists(self.model_path):
            raise FileNotFoundError("未找到YOLO模型: %s" % self.model_path)
        if not 0.05 <= self.conf_threshold <= 1.0:
            raise ValueError("conf_threshold 必须位于0.05到1.0之间")
        if not 0.05 < self.display_depth_min < self.display_depth_max <= 8.0:
            raise ValueError("显示深度范围配置无效")

        if self.observation_mode == "pick" and not rospy.has_param("~depth_min"):
            rospy.set_param("~depth_min", 0.25)
        set_competition_plane_defaults()
        self.estimator = PlacePlaneEstimator()
        self.plane_stability = StablePlaneDepth(
            rospy.get_param("~stable_samples", 4),
            rospy.get_param("~stable_max_depth_spread", 0.050),
            rospy.get_param("~stable_max_center_spread", 25.0),
        )
        self.center_stability = StablePlaneDepth(
            rospy.get_param("~center_stable_samples", 4),
            rospy.get_param("~center_stable_max_depth_spread", 0.050),
            rospy.get_param("~stable_max_center_spread", 25.0),
        )
        self.pick_depth_min = float(
            rospy.get_param("~pick_depth_min", PICK_DEPTH_MIN)
        )
        self.pick_depth_max = float(
            rospy.get_param("~pick_depth_max", PICK_DEPTH_MAX)
        )
        self.pick_fallback_min_valid_ratio = float(
            rospy.get_param(
                "~depth_fallback_min_valid_ratio",
                DEPTH_FALLBACK_MIN_VALID_RATIO,
            )
        )
        self.pick_stability = StablePlaneDepth(
            rospy.get_param("~depth_stable_samples", DEPTH_STABLE_SAMPLES),
            rospy.get_param(
                "~depth_stability_max_spread", DEPTH_STABILITY_MAX_SPREAD
            ),
            60.0,
        )
        self.pick_stability_label = None

        self.joint_pub = rospy.Publisher(
            self.joint_command_topic, JointState, queue_size=1
        )
        self.cartesian_pub = rospy.Publisher(
            self.cartesian_command_topic, PosCmd, queue_size=1
        )
        self.joint_positions = None
        self.joint_received_at = None
        self.current_ee_pose_mat = None
        self.pose_received_at = None
        self.joint_sub = rospy.Subscriber(
            self.joint_feedback_topic,
            JointState,
            self.joint_feedback_callback,
            queue_size=10,
        )
        self.pose_sub = rospy.Subscriber(
            self.end_pose_topic,
            PoseStamped,
            self.pose_callback,
            queue_size=10,
        )

        self.model = None
        self.pipeline = None
        self.profile = None
        self.pipeline_started = False
        self.align = None
        self.filters = None
        self.depth_scale = None
        self.last_snapshot = None
        self.shutdown_done = False
        self.windows_positioned = False
        rospy.on_shutdown(self.shutdown)

    def ensure_no_conflicting_nodes(self):
        """在机械臂运动前阻止比赛节点或其他 RealSense 节点抢占资源。"""
        if self.allow_conflicting_nodes:
            rospy.logwarn("已显式允许冲突节点；操作者需自行保证控制权和相机独占。")
            return
        try:
            active_nodes = set(rosnode.get_node_names())
        except Exception as exc:
            raise RuntimeError("无法检查ROS节点冲突，拒绝移动机械臂: %s" % exc)

        self_name = rospy.get_name()
        conflicts = []
        for node_name in sorted(active_nodes):
            if node_name == self_name:
                continue
            base_name = node_name.rsplit("/", 1)[-1]
            if base_name in {
                "piper_task",
                "piper_task_plane",
                "place_plane_depth_test",
                "place_bottle_depth_ir_test",
                "pick_object_depth_ir_test",
            }:
                conflicts.append(node_name)
            elif self.enable_vision and "realsense2_camera" in base_name.lower():
                conflicts.append(node_name)
        if conflicts:
            raise RuntimeError(
                "检测到会争用机械臂或D435i的节点 %s；请先停止它们后再测试。"
                % conflicts
            )

    def joint_feedback_callback(self, msg):
        if len(msg.position) < 6:
            return
        self.joint_positions = tuple(float(value) for value in msg.position[:6])
        self.joint_received_at = time.monotonic()

    def pose_callback(self, msg):
        position = msg.pose.position
        orientation = msg.pose.orientation
        quaternion = [
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        ]
        pose_mat = tf_trans.quaternion_matrix(quaternion)
        pose_mat[0:3, 3] = [position.x, position.y, position.z]
        self.current_ee_pose_mat = pose_mat
        self.pose_received_at = time.monotonic()

    def feedback_is_fresh(self, received_at):
        return (
            received_at is not None
            and time.monotonic() - received_at <= self.arm_feedback_max_age
        )

    def wait_for_arm_interfaces(self):
        if not self.move_arm_to_observation and not self.open_gripper_at_observation:
            return
        deadline = time.monotonic() + self.arm_feedback_timeout
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            joint_ready = (
                not self.move_arm_to_observation
                or (
                    self.joint_pub.get_num_connections() > 0
                    and self.joint_positions is not None
                    and self.feedback_is_fresh(self.joint_received_at)
                )
            )
            gripper_ready = (
                not self.open_gripper_at_observation
                or (
                    self.cartesian_pub.get_num_connections() > 0
                    and self.current_ee_pose_mat is not None
                    and self.feedback_is_fresh(self.pose_received_at)
                )
            )
            if joint_ready and gripper_ready:
                return
            rate.sleep()

        missing = []
        if self.move_arm_to_observation and self.joint_pub.get_num_connections() == 0:
            missing.append("%s无订阅者" % self.joint_command_topic)
        if self.move_arm_to_observation and (
            self.joint_positions is None
            or not self.feedback_is_fresh(self.joint_received_at)
        ):
            missing.append("%s无新鲜反馈" % self.joint_feedback_topic)
        if self.open_gripper_at_observation and self.cartesian_pub.get_num_connections() == 0:
            missing.append("%s无订阅者" % self.cartesian_command_topic)
        if self.open_gripper_at_observation and (
            self.current_ee_pose_mat is None
            or not self.feedback_is_fresh(self.pose_received_at)
        ):
            missing.append("%s无新鲜反馈" % self.end_pose_topic)
        raise RuntimeError(
            "机械臂接口未就绪（%s），请先启动Piper底层驱动。"
            % ("、".join(missing) if missing else "等待超时")
        )

    def publish_observation_pose(self):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = [""]
        msg.position = list(self.observation_joints)
        msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]
        msg.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
        self.joint_pub.publish(msg)

    def wait_for_observation_pose(self):
        target = tuple(self.observation_joints[:6])
        deadline = time.monotonic() + self.arm_verify_timeout
        stable = 0
        last_error = float("inf")
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if (
                self.joint_positions is not None
                and self.feedback_is_fresh(self.joint_received_at)
            ):
                last_error = max(
                    abs(actual - desired)
                    for actual, desired in zip(self.joint_positions, target)
                )
                if last_error <= self.arm_joint_tolerance:
                    stable += 1
                    if stable >= self.arm_stable_samples:
                        return
                else:
                    stable = 0
            rate.sleep()
        error_text = (
            "无有效反馈" if not np.isfinite(last_error) else "%.4frad" % last_error
        )
        raise RuntimeError(
            "机械臂未确认到达%s观察位：最大误差=%s，要求<=%.3frad连续%d帧"
            % (
                self.region_name_zh,
                error_text,
                self.arm_joint_tolerance,
                self.arm_stable_samples,
            )
        )

    def publish_open_gripper(self):
        if self.current_ee_pose_mat is None or not self.feedback_is_fresh(
            self.pose_received_at
        ):
            raise RuntimeError("张开夹爪前未获得新鲜的末端位姿反馈")
        pose_mat = self.current_ee_pose_mat.copy()
        roll, pitch, yaw = tf_trans.euler_from_matrix(pose_mat, axes="sxyz")
        cmd = PosCmd()
        cmd.x, cmd.y, cmd.z = pose_mat[0:3, 3]
        cmd.roll, cmd.pitch, cmd.yaw = roll, pitch, yaw
        cmd.gripper = self.gripper_open_value
        cmd.mode1 = 1
        cmd.mode2 = 0
        self.cartesian_pub.publish(cmd)

    def prepare_arm(self):
        if not self.move_arm_to_observation and not self.open_gripper_at_observation:
            rospy.logwarn("已关闭机械臂准备动作，直接启动视觉测试。")
            return
        rospy.loginfo("等待Piper控制接口和关节反馈...")
        self.wait_for_arm_interfaces()
        if self.move_arm_to_observation:
            rospy.loginfo(
                "机械臂移动到比赛%s观察位: %s（关节命令夹爪值=%.1f）",
                self.region_name_zh,
                self.observation_joints,
                self.observation_gripper_value,
            )
            self.publish_observation_pose()
            self.arm_was_commanded_to_observation = True
            if self.arm_move_wait > 0.0:
                rospy.sleep(self.arm_move_wait)
            self.wait_for_observation_pose()
            rospy.loginfo(
                "机械臂已通过反馈确认到达%s观察位。", self.region_name_zh
            )
        if self.open_gripper_at_observation:
            rospy.logwarn(
                "机械臂已到%s观察位，现在张开夹爪；若夹有物品，此动作会释放物品。",
                self.region_name_zh,
            )
            self.publish_open_gripper()
            if self.gripper_open_wait > 0.0:
                rospy.sleep(self.gripper_open_wait)
            rospy.loginfo(
                "%s观察位夹爪张开命令已完成: gripper=%.1f",
                self.region_name_zh,
                self.gripper_open_value,
            )
        else:
            rospy.logwarn(
                "持瓶诊断模式：不额外发送张爪命令；观察位关节命令已使用"
                "夹爪值 %.1f（默认0为闭合）。",
                self.observation_gripper_value,
            )

    def load_model(self):
        rospy.loginfo("加载YOLO模型: %s", self.model_path)
        self.model = YOLO(self.model_path)
        rospy.loginfo("模型类别: %s", self.model.names)

    def start_camera(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        if self.device_serial:
            config.enable_device(self.device_serial)
        config.enable_stream(
            rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, FPS
        )
        config.enable_stream(
            rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS
        )
        config.enable_stream(
            rs.stream.infrared, 1, IR_WIDTH, IR_HEIGHT, rs.format.y8, FPS
        )
        config.enable_stream(
            rs.stream.infrared, 2, IR_WIDTH, IR_HEIGHT, rs.format.y8, FPS
        )
        try:
            self.profile = self.pipeline.start(config)
            self.pipeline_started = True
        except RuntimeError as exc:
            raise RuntimeError(
                "无法同时启动D435i RGB/Depth/IR1/IR2。请停止占用相机的"
                "piper_task或realsense2_camera节点，并确认使用USB3：%s" % exc
            )

        device = self.profile.get_device()
        try:
            device_name = device.get_info(rs.camera_info.name)
            serial = device.get_info(rs.camera_info.serial_number)
            usb_type = device.get_info(rs.camera_info.usb_type_descriptor)
        except RuntimeError:
            device_name = "RealSense"
            serial = "unknown"
            usb_type = "unknown"
        depth_sensor = device.first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())
        self.align = rs.align(rs.stream.color)
        self.filters = build_depth_filters()
        rospy.loginfo(
            "D435i四路图像已启动: device=%s serial=%s USB=%s scale=%.6f",
            device_name,
            serial,
            usb_type,
            self.depth_scale,
        )
        rospy.loginfo(
            "注意：LEFT/RIGHT IR是双目原始灰度图，只有一张Depth由双目计算得到。"
        )

    def warmup(self):
        for _ in range(self.warmup_frames):
            if rospy.is_shutdown():
                return
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            left_ir = frames.get_infrared_frame(1)
            right_ir = frames.get_infrared_frame(2)
            aligned = self.align.process(frames)
            depth = aligned.get_depth_frame()
            color = aligned.get_color_frame()
            if depth and color and left_ir and right_ir:
                apply_depth_filters(depth, self.filters)
        rospy.loginfo("D435i预热完成，开始%s区候选点观察。", self.region_name_zh)

    def compose_and_store_views(
        self,
        annotated,
        depth_view,
        left_ir_image,
        right_ir_image,
        color_image,
        raw_depth_z16,
        raw_depth_m,
        filtered_depth_m,
        detections,
        detection,
    ):
        """生成四路显示与截图缓存，供抓取和放置深度分支共用。"""
        left_ir_view = cv2.cvtColor(left_ir_image, cv2.COLOR_GRAY2BGR)
        right_ir_view = cv2.cvtColor(right_ir_image, cv2.COLOR_GRAY2BGR)
        draw_panel_title(annotated, "RGB / YOLO: ALL SUPPORTED LABELS")
        draw_panel_title(
            depth_view,
            "ALIGNED FILTERED DEPTH %.2f-%.2f m"
            % (self.display_depth_min, self.display_depth_max),
        )
        draw_panel_title(left_ir_view, "LEFT IR (raw Y8, no RGB bbox)")
        draw_panel_title(right_ir_view, "RIGHT IR (raw Y8, no RGB bbox)")

        rgb_depth_view = np.hstack((annotated, depth_view))
        stereo_ir_view = np.hstack((left_ir_view, right_ir_view))
        composite = np.vstack((rgb_depth_view, stereo_ir_view))
        if self.display_scale != 1.0:
            rgb_depth_display = cv2.resize(
                rgb_depth_view,
                None,
                fx=self.display_scale,
                fy=self.display_scale,
                interpolation=cv2.INTER_AREA,
            )
            stereo_ir_display = cv2.resize(
                stereo_ir_view,
                None,
                fx=self.display_scale,
                fy=self.display_scale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            rgb_depth_display = rgb_depth_view
            stereo_ir_display = stereo_ir_view

        self.last_snapshot = {
            "composite": composite.copy(),
            "rgb_raw": color_image.copy(),
            "rgb_annotated": annotated.copy(),
            "depth_colormap": depth_view.copy(),
            "ir_left": left_ir_image.copy(),
            "ir_right": right_ir_image.copy(),
            "aligned_depth_z16": raw_depth_z16.copy(),
            "aligned_raw_depth_m": raw_depth_m.copy(),
            "aligned_filtered_depth_m": filtered_depth_m.copy(),
            "detection_boxes": (
                np.asarray([item["box"] for item in detections], dtype=np.int32)
                if detections
                else np.empty((0, 4), dtype=np.int32)
            ),
            "detection_box": (
                np.asarray(detection["box"], dtype=np.int32)
                if detection is not None
                else np.empty((0,), dtype=np.int32)
            ),
        }
        return rgb_depth_display, stereo_ir_display

    def process_pick_depth_overlay(
        self,
        annotated,
        depth_view,
        color_image,
        raw_depth_z16,
        raw_depth_m,
        filtered_depth_m,
        raw_depth_frame,
        filtered_depth_frame,
        left_ir_image,
        right_ir_image,
        detections,
        detection,
        selection_failure,
    ):
        """用主控抓取的框内P20/滤波恢复/5帧稳定逻辑显示深度。"""
        for candidate in detections:
            x1, y1, x2, y2 = candidate["box"]
            raw_depth = get_depth_percentile(
                filtered_depth_frame,
                raw_depth_frame,
                x1,
                y1,
                x2,
                y2,
                self.depth_scale,
            )
            accepted_depth = raw_depth
            source = "raw_p20"
            valid_ratio = 1.0
            if not is_depth_in_range(
                raw_depth, self.pick_depth_min, self.pick_depth_max
            ):
                accepted_depth, valid_ratio = get_filtered_depth_in_range(
                    filtered_depth_frame,
                    x1,
                    y1,
                    x2,
                    y2,
                    self.depth_scale,
                    self.pick_depth_min,
                    self.pick_depth_max,
                    self.pick_fallback_min_valid_ratio,
                )
                source = "filtered_retry" if accepted_depth is not None else "reject"
            candidate["main_raw_depth"] = float(raw_depth)
            candidate["main_depth"] = accepted_depth
            candidate["main_depth_source"] = source
            candidate["main_valid_ratio"] = float(valid_ratio)

            label = "#%d %s raw=%.3f main=%s %s" % (
                candidate["index"],
                candidate["label"],
                raw_depth,
                format_depth(accepted_depth),
                source,
            )
            color = (0, 255, 0) if candidate is detection else (255, 0, 255)
            for image in (annotated, depth_view):
                cv2.putText(
                    image,
                    label,
                    (x1, min(image.shape[0] - 8, y2 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        if detection is None:
            self.pick_stability.clear()
            self.pick_stability_label = None
            lines = [
                "MAIN PICK DEPTH: NO TARGET",
                selection_failure or "target_not_found",
            ]
            draw_bottom_lines(annotated, lines, color=(0, 0, 255))
            draw_bottom_lines(depth_view, lines, color=(0, 0, 255))
        else:
            x1, y1, x2, y2 = detection["box"]
            center = (0.5 * (x1 + x2), 0.5 * (y1 + y2))
            accepted_depth = detection["main_depth"]
            safe = not detection["edge_directions"]
            if self.pick_stability_label != detection["label"]:
                self.pick_stability.clear()
                self.pick_stability_label = detection["label"]

            stable = None
            if accepted_depth is not None and safe:
                stable = self.pick_stability.add(accepted_depth, center)
            else:
                self.pick_stability.clear()

            bw, bh = x2 - x1, y2 - y1
            rx1 = max(0, x1 + int(bw * SHRINK_RATIO))
            ry1 = max(0, y1 + int(bh * SHRINK_RATIO))
            rx2 = min(annotated.shape[1] - 1, x2 - int(bw * SHRINK_RATIO))
            ry2 = min(annotated.shape[0] - 1, y2 - int(bh * SHRINK_RATIO))
            for image in (annotated, depth_view):
                cv2.rectangle(image, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
                cv2.drawMarker(
                    image,
                    (int(round(center[0])), int(round(center[1]))),
                    (255, 255, 255),
                    cv2.MARKER_CROSS,
                    14,
                    2,
                )

            if not safe:
                state = "EDGE REJECT"
            elif accepted_depth is None:
                state = "DEPTH REJECT"
            elif stable is None:
                state = "WAIT %d/%d" % (
                    len(self.pick_stability.samples),
                    self.pick_stability.sample_count,
                )
            else:
                state = "PASS %.3fm spread=%.1fmm" % (
                    stable["depth"],
                    stable["depth_spread"] * 1000.0,
                )
            edge_text = "+".join(detection["edge_directions"]) or "safe"
            lines = [
                "MAIN PICK #%d %s conf=%.2f edge=%s"
                % (
                    detection["index"],
                    detection["label"],
                    detection["confidence"],
                    edge_text,
                ),
                "raw P20=%.3fm accepted=%sm source=%s"
                % (
                    detection["main_raw_depth"],
                    format_depth(accepted_depth),
                    detection["main_depth_source"],
                ),
                "range=%.2f-%.2fm fallback_valid=%.1f%%"
                % (
                    self.pick_depth_min,
                    self.pick_depth_max,
                    100.0 * detection["main_valid_ratio"],
                ),
                state,
            ]
            color = (0, 255, 0) if stable is not None else (0, 165, 255)
            draw_bottom_lines(annotated, lines, color=color)
            draw_bottom_lines(depth_view, lines, color=color)
            rospy.loginfo_throttle(
                1.0,
                "[抓取主控同源深度] label=%s raw_p20=%.3f accepted=%s "
                "source=%s edge=%s state=%s",
                detection["label"],
                detection["main_raw_depth"],
                format_depth(accepted_depth),
                detection["main_depth_source"],
                edge_text,
                state,
            )

        return self.compose_and_store_views(
            annotated,
            depth_view,
            left_ir_image,
            right_ir_image,
            color_image,
            raw_depth_z16,
            raw_depth_m,
            filtered_depth_m,
            detections,
            detection,
        )

    def process_frame(
        self,
        color_image,
        raw_depth_z16,
        raw_depth_m,
        filtered_depth_m,
        raw_depth_frame,
        filtered_depth_frame,
        intrinsics,
        left_ir_image,
        right_ir_image,
    ):
        height, width = color_image.shape[:2]
        annotated = color_image.copy()
        cv2.rectangle(
            annotated,
            (
                self.margin_x,
                0 if self.observation_mode == "pick" else self.margin_y,
            ),
            (
                width - 1 - self.margin_x,
                height - 1
                if self.observation_mode == "pick"
                else height - 1 - self.margin_y,
            ),
            (0, 255, 255),
            1,
        )

        results = self.model.predict(
            source=color_image,
            imgsz=self.image_size,
            conf=self.conf_threshold,
            verbose=False,
        )
        detections = collect_matching_detections(
            results,
            self.model.names,
            self.target_label,
            width,
            height,
            self.margin_x,
            self.margin_y,
            self.observation_mode,
            self.pick_hard_border_margin,
        )
        selection_failure = None
        if not detections:
            detection = None
            selection_failure = "target_not_found"
        elif self.target_index >= 0:
            if self.target_index < len(detections):
                detection = detections[self.target_index]
            else:
                detection = None
                selection_failure = "target_index_%d_missing" % self.target_index
        elif self.observation_mode == "pick":
            # 正式抓取会先丢弃左右边缘不安全框，再在安全候选中评分。
            safe_detections = [
                item for item in detections if not item["edge_directions"]
            ]
            detection = max(
                safe_detections or detections,
                key=lambda item: item["score"],
            )
        else:
            detection = max(detections, key=lambda item: item["score"])

        depth_view = make_depth_colormap(
            filtered_depth_m,
            self.display_depth_min,
            self.display_depth_max,
        )
        plane_estimate = None
        plane_stable = None
        center_stable = None
        raw_stats = None
        filtered_stats = None

        # 默认显示模型支持的全部类别并从左到右编号；平面算法只对自动最佳框
        # 或 _target_index 指定框运行，避免多个目标共用稳定性缓存。
        candidate_summaries = []
        for candidate in detections:
            candidate_raw = center_roi_depth_stats(
                raw_depth_m,
                candidate["box"],
                self.center_roi_ratio,
                self.estimator.depth_min,
                self.estimator.depth_max,
            )
            candidate_filtered = center_roi_depth_stats(
                filtered_depth_m,
                candidate["box"],
                self.center_roi_ratio,
                self.estimator.depth_min,
                self.estimator.depth_max,
            )
            candidate["raw_stats"] = candidate_raw
            candidate["filtered_stats"] = candidate_filtered
            is_selected = candidate is detection
            color = (0, 255, 0) if is_selected else (255, 0, 255)
            thickness = 3 if is_selected else 1
            x1, y1, x2, y2 = candidate["box"]
            if self.observation_mode == "pick":
                instance_text = "#%d %s %.2f" % (
                    candidate["index"],
                    candidate["label"],
                    candidate["confidence"],
                )
            else:
                instance_text = "#%d %s %.2f raw=%sm" % (
                    candidate["index"],
                    candidate["label"],
                    candidate["confidence"],
                    format_depth(candidate_raw["sensor_median"]),
                )
            for image in (annotated, depth_view):
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(
                    image,
                    instance_text,
                    (x1, max(50, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            candidate_summaries.append(
                "#%d %s box=%s conf=%.2f raw=%s task=%s/%.1f%% edge=%s"
                % (
                    candidate["index"],
                    candidate["label"],
                    candidate["box"],
                    candidate["confidence"],
                    format_depth(candidate_raw["sensor_median"]),
                    format_depth(candidate_filtered["task_median"]),
                    100.0 * candidate_filtered["task_ratio"],
                    "+".join(candidate["edge_directions"]) or "safe",
                )
            )
        if len(candidate_summaries) > 1:
            rospy.logwarn_throttle(
                1.0,
                "[%s候选点观察] 共检测到%d个目标（左到右编号）: %s",
                self.region_name_zh,
                len(candidate_summaries),
                " | ".join(candidate_summaries),
            )

        if self.observation_mode == "pick":
            return self.process_pick_depth_overlay(
                annotated,
                depth_view,
                color_image,
                raw_depth_z16,
                raw_depth_m,
                filtered_depth_m,
                raw_depth_frame,
                filtered_depth_frame,
                left_ir_image,
                right_ir_image,
                detections,
                detection,
                selection_failure,
            )

        if detection is None:
            self.plane_stability.clear()
            self.center_stability.clear()
            if selection_failure == "target_not_found":
                target_text = self.target_label or "all-supported-labels"
                status_lines = [
                    "target=%s NOT FOUND" % target_text,
                    "conf>=%.2f" % self.conf_threshold,
                ]
                rospy.logwarn_throttle(
                    1.0,
                    "[%s候选点观察] 未检测到支持的目标 %s (conf>=%.2f)",
                    self.region_name_zh,
                    target_text,
                    self.conf_threshold,
                )
            else:
                status_lines = [
                    "target=%s instances=%d"
                    % (self.target_label or "all", len(detections)),
                    "%s; choose 0..%d"
                    % (selection_failure, max(0, len(detections) - 1)),
                ]
                rospy.logwarn_throttle(
                    1.0,
                    "[%s候选点观察] %s，当前检测框数量=%d",
                    self.region_name_zh,
                    selection_failure,
                    len(detections),
                )
            draw_bottom_lines(annotated, status_lines, color=(0, 0, 255))
            draw_bottom_lines(depth_view, status_lines, color=(0, 0, 255))
        else:
            raw_stats = detection["raw_stats"]
            filtered_stats = detection["filtered_stats"]
            plane_estimate = self.estimator.estimate(
                filtered_depth_m,
                detection["box"],
                intrinsics,
            )

            if plane_estimate["ok"]:
                plane_stable = self.plane_stability.add(
                    plane_estimate["depth"], plane_estimate["center"]
                )
            else:
                self.plane_stability.clear()

            center_ok = bool(
                filtered_stats["task_median"] is not None
                and filtered_stats["task_count"] >= self.center_min_valid_pixels
                and filtered_stats["task_ratio"] >= self.center_min_valid_ratio
                and filtered_stats["task_mad"] is not None
                and filtered_stats["task_mad"] <= self.center_max_mad
            )
            if center_ok:
                center_stable = self.center_stability.add(
                    filtered_stats["task_median"], filtered_stats["center"]
                )
            else:
                self.center_stability.clear()

            for image in (annotated, depth_view):
                draw_estimator_result(
                    image,
                    detection,
                    plane_estimate,
                    plane_stable,
                    len(self.plane_stability.samples),
                )
                rx1, ry1, rx2, ry2 = filtered_stats["roi"]
                cv2.rectangle(
                    image,
                    (rx1, ry1),
                    (max(rx1, rx2 - 1), max(ry1, ry2 - 1)),
                    (255, 255, 0),
                    2,
                )
                cx = int(round(filtered_stats["center"][0]))
                cy = int(round(filtered_stats["center"][1]))
                cv2.drawMarker(
                    image,
                    (cx, cy),
                    (255, 255, 255),
                    cv2.MARKER_CROSS,
                    14,
                    2,
                )

            edge_text = (
                "+".join(detection["edge_directions"])
                if detection["edge_directions"]
                else "safe"
            )
            plane_text = (
                "%.3f" % plane_estimate["depth"]
                if plane_estimate["ok"]
                else "REJECT:%s" % plane_estimate.get("reason", "unknown")
            )
            center_state = "PASS" if center_stable is not None else (
                "WAIT" if center_ok else "REJECT"
            )
            status_lines = [
                "#%d %s conf=%.2f box=%s edge=%s"
                % (
                    detection["index"],
                    detection["label"],
                    detection["confidence"],
                    detection["box"],
                    edge_text,
                ),
                "RAW center_px=%sm roi_median=%sm"
                % (
                    format_depth(raw_stats["center_pixel"]),
                    format_depth(raw_stats["sensor_median"]),
                ),
                "FILTER task=%sm support=%d/%d(%.1f%%) %s"
                % (
                    format_depth(filtered_stats["task_median"]),
                    filtered_stats["task_count"],
                    filtered_stats["roi_pixels"],
                    100.0 * filtered_stats["task_ratio"],
                    center_state,
                ),
                "PLANE=%s" % plane_text,
            ]
            status_color = (
                (0, 255, 0)
                if plane_stable is not None or center_stable is not None
                else (0, 165, 255)
            )
            draw_bottom_lines(annotated, status_lines, color=status_color)
            draw_bottom_lines(depth_view, status_lines, color=status_color)

            log_message = (
                "index=%d label=%s conf=%.2f box=%s edge=%s raw_pixel=%s "
                "raw_roi=%s filtered_task=%s support=%d/%d(%.1f%%) "
                "center=%s plane=%s"
                % (
                    detection["index"],
                    detection["label"],
                    detection["confidence"],
                    detection["box"],
                    edge_text,
                    format_depth(raw_stats["center_pixel"]),
                    format_depth(raw_stats["sensor_median"]),
                    format_depth(filtered_stats["task_median"]),
                    filtered_stats["task_count"],
                    filtered_stats["roi_pixels"],
                    100.0 * filtered_stats["task_ratio"],
                    center_state,
                    plane_text,
                )
            )
            if plane_stable is not None or center_stable is not None:
                rospy.loginfo_throttle(
                    1.0, "[%s候选点观察] %s", self.region_name_zh, log_message
                )
            else:
                rospy.logwarn_throttle(
                    1.0, "[%s候选点观察] %s", self.region_name_zh, log_message
                )

        left_ir_view = cv2.cvtColor(left_ir_image, cv2.COLOR_GRAY2BGR)
        right_ir_view = cv2.cvtColor(right_ir_image, cv2.COLOR_GRAY2BGR)
        draw_panel_title(
            annotated,
            "RGB / YOLO: %s"
            % (
                "%s (%s)"
                % (
                    self.target_label,
                    LABEL_NAMES_ZH.get(self.target_label, "target"),
                )
                if self.target_label
                else "ALL SUPPORTED LABELS"
            ),
        )
        draw_panel_title(
            depth_view,
            "ALIGNED FILTERED DEPTH %.2f-%.2f m"
            % (self.display_depth_min, self.display_depth_max),
        )
        draw_panel_title(left_ir_view, "LEFT IR (raw Y8, no RGB bbox)")
        draw_panel_title(right_ir_view, "RIGHT IR (raw Y8, no RGB bbox)")

        rgb_depth_view = np.hstack((annotated, depth_view))
        stereo_ir_view = np.hstack((left_ir_view, right_ir_view))
        # 保存时仍保留完整四路组合图；屏幕显示则拆成两个独立窗口。
        composite = np.vstack((rgb_depth_view, stereo_ir_view))
        if self.display_scale != 1.0:
            rgb_depth_display = cv2.resize(
                rgb_depth_view,
                None,
                fx=self.display_scale,
                fy=self.display_scale,
                interpolation=cv2.INTER_AREA,
            )
            stereo_ir_display = cv2.resize(
                stereo_ir_view,
                None,
                fx=self.display_scale,
                fy=self.display_scale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            rgb_depth_display = rgb_depth_view
            stereo_ir_display = stereo_ir_view

        self.last_snapshot = {
            "composite": composite.copy(),
            "rgb_raw": color_image.copy(),
            "rgb_annotated": annotated.copy(),
            "depth_colormap": depth_view.copy(),
            "ir_left": left_ir_image.copy(),
            "ir_right": right_ir_image.copy(),
            "aligned_depth_z16": raw_depth_z16.copy(),
            "aligned_raw_depth_m": raw_depth_m.copy(),
            "aligned_filtered_depth_m": filtered_depth_m.copy(),
            "detection_boxes": (
                np.asarray([item["box"] for item in detections], dtype=np.int32)
                if detections
                else np.empty((0, 4), dtype=np.int32)
            ),
            "detection_box": (
                np.asarray(detection["box"], dtype=np.int32)
                if detection is not None
                else np.empty((0,), dtype=np.int32)
            ),
        }
        return rgb_depth_display, stereo_ir_display

    def save_snapshot(self):
        snapshot = self.last_snapshot
        if snapshot is None:
            rospy.logwarn("当前还没有可保存的完整四路图像。")
            return
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            now = time.time()
            timestamp = "%s-%03d" % (
                time.strftime("%Y%m%d-%H%M%S", time.localtime(now)),
                int((now % 1.0) * 1000.0),
            )
            prefix = os.path.join(
                self.save_dir,
                "%s-%s" % (timestamp, self.target_label or "all"),
            )
            outputs = {
                prefix + "-composite.jpg": snapshot["composite"],
                prefix + "-rgb-raw.png": snapshot["rgb_raw"],
                prefix + "-rgb-annotated.png": snapshot["rgb_annotated"],
                prefix + "-depth-color.png": snapshot["depth_colormap"],
                prefix + "-ir-left.png": snapshot["ir_left"],
                prefix + "-ir-right.png": snapshot["ir_right"],
                prefix + "-depth-z16.png": snapshot["aligned_depth_z16"],
            }
            failed = []
            for filename, image in outputs.items():
                if not cv2.imwrite(filename, image):
                    failed.append(filename)
            np.savez_compressed(
                prefix + "-depth-data.npz",
                aligned_depth_z16=snapshot["aligned_depth_z16"],
                aligned_raw_depth_m=snapshot["aligned_raw_depth_m"],
                aligned_filtered_depth_m=snapshot["aligned_filtered_depth_m"],
                detection_boxes=snapshot["detection_boxes"],
                detection_box=snapshot["detection_box"],
                depth_scale=np.asarray([self.depth_scale], dtype=np.float64),
            )
            if failed:
                rospy.logerr("部分截图保存失败: %s", failed)
            else:
                rospy.loginfo("已保存四路截图和原始深度数据: %s-*", prefix)
        except (OSError, ValueError, RuntimeError, cv2.error) as exc:
            rospy.logerr("保存测试数据失败，但节点继续运行: %s", exc)

    def run(self):
        rospy.logwarn(
            "这是独立诊断节点：请勿同时运行 piper_task、piper_task_plane "
            "或 realsense2_camera。"
        )
        rospy.loginfo(
            "%s区检测范围: %s，检测阈值=%.2f",
            self.region_name_zh,
            (
                "%s (%s)"
                % (
                    self.target_label,
                    LABEL_NAMES_ZH.get(self.target_label, self.target_label),
                )
                if self.target_label
                else "全部支持类别 %s" % sorted(SUPPORTED_LABELS)
            ),
            self.conf_threshold,
        )
        try:
            # 按用户要求，先移动机械臂，再启动模型和相机。
            self.ensure_no_conflicting_nodes()
            self.prepare_arm()
            if rospy.is_shutdown():
                return
            if not self.enable_vision:
                rospy.loginfo(
                    "%s区仅到位模式：机械臂已到观察位，未加载模型、未打开相机；"
                    "按 Ctrl+C 退出并回零。",
                    self.region_name_zh,
                )
                while not rospy.is_shutdown():
                    rospy.sleep(0.2)
                return
            self.load_model()
            self.start_camera()
            self.warmup()
            rospy.loginfo(
                "已启用两个显示窗口：RGB+Depth、Left IR+Right IR；"
                "q/Esc退出，s保存完整数据。"
            )

            while not rospy.is_shutdown():
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                except RuntimeError as exc:
                    rospy.logwarn("等待D435i四路图像超时: %s", exc)
                    continue

                left_ir_frame = frames.get_infrared_frame(1)
                right_ir_frame = frames.get_infrared_frame(2)
                aligned = self.align.process(frames)
                raw_depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()
                if (
                    not left_ir_frame
                    or not right_ir_frame
                    or not raw_depth_frame
                    or not color_frame
                ):
                    rospy.logwarn_throttle(1.0, "D435i四路图像有缺帧，等待完整帧组。")
                    continue

                filtered_depth_frame = apply_depth_filters(
                    raw_depth_frame, self.filters
                )
                color_image = np.asanyarray(color_frame.get_data())
                left_ir_image = np.asanyarray(left_ir_frame.get_data())
                right_ir_image = np.asanyarray(right_ir_frame.get_data())
                raw_depth_z16 = np.asanyarray(raw_depth_frame.get_data())
                raw_depth_m = raw_depth_z16.astype(np.float32) * self.depth_scale
                filtered_depth_m = (
                    np.asanyarray(filtered_depth_frame.get_data()).astype(np.float32)
                    * self.depth_scale
                )
                intrinsics = (
                    filtered_depth_frame.profile.as_video_stream_profile().intrinsics
                )

                rgb_depth_display, stereo_ir_display = self.process_frame(
                    color_image,
                    raw_depth_z16,
                    raw_depth_m,
                    filtered_depth_m,
                    raw_depth_frame,
                    filtered_depth_frame,
                    intrinsics,
                    left_ir_image,
                    right_ir_image,
                )

                if self.show_window:
                    try:
                        cv2.imshow(self.window_rgb_depth, rgb_depth_display)
                        cv2.imshow(self.window_stereo_ir, stereo_ir_display)
                        if not self.windows_positioned:
                            display_height = int(
                                round(COLOR_HEIGHT * self.display_scale)
                            )
                            cv2.moveWindow(self.window_rgb_depth, 20, 30)
                            cv2.moveWindow(
                                self.window_stereo_ir,
                                20,
                                30 + display_height + 55,
                            )
                            self.windows_positioned = True
                        key = cv2.waitKey(1) & 0xFF
                    except cv2.error as exc:
                        rospy.logwarn("OpenCV窗口不可用，停止显示: %s", exc)
                        self.show_window = False
                        key = 0xFF
                    if key in (ord("q"), 27):
                        rospy.signal_shutdown(
                            "用户退出%s区候选点观察" % self.region_name_zh
                        )
                    elif key == ord("s"):
                        self.save_snapshot()
                    try:
                        rgb_depth_visible = cv2.getWindowProperty(
                            self.window_rgb_depth, cv2.WND_PROP_VISIBLE
                        )
                        stereo_ir_visible = cv2.getWindowProperty(
                            self.window_stereo_ir, cv2.WND_PROP_VISIBLE
                        )
                    except cv2.error:
                        rgb_depth_visible = 1.0
                        stereo_ir_visible = 1.0
                    if rgb_depth_visible < 1.0 or stereo_ir_visible < 1.0:
                        rospy.signal_shutdown(
                            "用户关闭%s区候选点观察窗口" % self.region_name_zh
                        )
        finally:
            self.shutdown()

    def return_arm_to_zero(self):
        """退出观察测试时发送主控运输零位，并尽量用真实反馈确认。"""
        if (
            not self.arm_was_commanded_to_observation
            or not self.return_to_zero_on_exit
        ):
            return False
        if self.joint_pub.get_num_connections() <= 0:
            rospy.logwarn("退出回零失败：%s 没有订阅者。", self.joint_command_topic)
            return False

        zero_joints = [0.0] * 7
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = [""]
        msg.position = zero_joints
        msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]
        msg.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
        rospy.loginfo("测试退出：机械臂开始返回主控运输零位 %s。", zero_joints)
        # Ctrl+C 已把 rospy 标为 shutdown，不能使用 rospy.sleep；用短间隔重复
        # 发布提高命令在ROS连接拆除前送达的可靠性。
        for _ in range(3):
            msg.header.stamp = rospy.Time.now()
            self.joint_pub.publish(msg)
            time.sleep(0.08)

        deadline = time.monotonic() + self.return_zero_timeout
        stable = 0
        last_error = float("inf")
        while time.monotonic() < deadline:
            positions = self.joint_positions
            if positions is not None:
                last_error = max(abs(value) for value in positions[:6])
                if last_error <= self.arm_joint_tolerance:
                    stable += 1
                    if stable >= self.arm_stable_samples:
                        rospy.loginfo(
                            "机械臂退出回零完成：最大关节误差 %.4frad。",
                            last_error,
                        )
                        return True
                else:
                    stable = 0
            time.sleep(0.05)
        rospy.logwarn(
            "已发送退出回零命令，但 %.1fs 内未确认到零位；最后最大误差=%s。",
            self.return_zero_timeout,
            "无反馈" if not np.isfinite(last_error) else "%.4frad" % last_error,
        )
        return True

    def shutdown(self):
        if self.shutdown_done:
            return
        self.shutdown_done = True
        if self.pipeline_started and self.pipeline is not None:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
            self.pipeline_started = False
        zero_requested = self.return_arm_to_zero()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        if zero_requested:
            rospy.loginfo(
                "%s区候选点观察节点已退出；机械臂已请求返回零位，"
                "航迹录制节点不受影响。",
                self.region_name_zh,
            )
        else:
            rospy.loginfo(
                "%s区候选点观察节点已退出；本次未发送退出回零命令，"
                "航迹录制节点不受影响。",
                self.region_name_zh,
            )


def main():
    rospy.init_node("place_bottle_depth_ir_test", anonymous=False)
    node = PlaceBottleDepthIrTestNode()
    node.run()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except Exception as exc:
        try:
            rospy.logfatal("放置区候选点观察启动/运行失败: %s", exc)
        except Exception:
            pass
        raise
