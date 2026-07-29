#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import rospy
import tf.transformations as tf_trans
from geometry_msgs.msg import PoseStamped
from piper_msgs.msg import PosCmd
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from ultralytics import YOLO


# Camera and detector defaults. Most values come from the existing
# xiaosai/grab2016_5_28_3.py tuning.
MODEL_PATH = "/home/user/piper_ws/src/realsense-D455-YOLOV5/weights/bottle.pt"
CONF_THRES = 0.5
IMG_SIZE = 640
COLOR_WIDTH = 640
COLOR_HEIGHT = 480
DEPTH_WIDTH = 640
DEPTH_HEIGHT = 480
FPS = 30
DEPTH_MIN = 0.1
DEPTH_MAX = 8.0
SHRINK_RATIO = 0.20
DEPTH_PERCENTILE = 20
COLOR_RATIO_THRES = 0.15

T_EE_TO_CAM = np.array([
    [-0.0482, 0.9987, 0.0147, -0.0699],
    [-0.9979, -0.0488, 0.0416, 0.0301],
    [0.0423, -0.0127, 0.9990, 0.0675],
    [0.0, 0.0, 0.0, 1.0],
], dtype=float)

T_EE_TO_TOOL = np.eye(4)
T_EE_TO_TOOL[0:3, 3] = [0.0, 0.0, -0.14]
T_BOTTLE_TO_GRASP = np.eye(4)

COLOR_DEFS = {
    "bottle-y": {
        "hsv_lower": np.array([5, 80, 80]),
        "hsv_upper": np.array([25, 255, 255]),
        "bgr": (0, 140, 255),
    },
    "bottle-g": {
        "hsv_lower": np.array([35, 50, 50]),
        "hsv_upper": np.array([85, 255, 255]),
        "bgr": (0, 200, 0),
    },
    "bottle-b": {
        "hsv_lower": np.array([0, 0, 0]),
        "hsv_upper": np.array([180, 255, 60]),
        "bgr": (80, 80, 80),
    },
}


def normalize_vector(v, eps=1e-9):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        return None
    return v / n


def build_level_grasp_rotation(reference_R=None):
    grasp_x = np.array([0.0, 0.0, -1.0], dtype=float)

    candidate_refs = []
    if reference_R is not None:
        reference_R = np.array(reference_R, dtype=float)
        candidate_refs.extend([
            reference_R[0:3, 2],
            reference_R[0:3, 1],
            reference_R[0:3, 0],
        ])

    candidate_refs.extend([
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
    ])

    grasp_z = None
    for ref in candidate_refs:
        z_candidate = ref - np.dot(ref, grasp_x) * grasp_x
        z_candidate = normalize_vector(z_candidate)
        if z_candidate is not None:
            grasp_z = z_candidate
            break

    if grasp_z is None:
        grasp_z = np.array([1.0, 0.0, 0.0], dtype=float)

    grasp_y = normalize_vector(np.cross(grasp_z, grasp_x))
    return np.column_stack((grasp_x, grasp_y, grasp_z))


def classify_bottle_color(color_image, x1, y1, x2, y2):
    h, w = color_image.shape[:2]
    rx1 = max(int(x1), 0)
    ry1 = max(int(y1), 0)
    rx2 = min(int(x2), w - 1)
    ry2 = min(int(y2), h - 1)
    if rx2 <= rx1 or ry2 <= ry1:
        return None

    roi = color_image[ry1:ry2, rx1:rx2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]
    if total == 0:
        return None

    priority_order = ["bottle-y", "bottle-g", "bottle-b"]
    claimed = np.zeros((hsv.shape[0], hsv.shape[1]), dtype=bool)
    scores = {}

    for label in priority_order:
        cfg = COLOR_DEFS[label]
        mask = cv2.inRange(hsv, cfg["hsv_lower"], cfg["hsv_upper"])
        exclusive = (mask > 0) & (~claimed)
        scores[label] = np.count_nonzero(exclusive) / total
        claimed |= exclusive

    best_label = max(scores, key=scores.get)
    if scores[best_label] >= COLOR_RATIO_THRES:
        return best_label
    return None


def build_filters():
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 1)

    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    spatial.set_option(rs.option.holes_fill, 1)

    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal.set_option(rs.option.filter_smooth_delta, 20)

    hole_filling = rs.hole_filling_filter()
    hole_filling.set_option(rs.option.holes_fill, 1)
    return [decimation, spatial, temporal, hole_filling]


def apply_filters(depth_frame, filters):
    for f in filters:
        depth_frame = f.process(depth_frame)
    return depth_frame


def get_depth_percentile(depth_frame, raw_depth_frame, x1, y1, x2, y2, depth_scale):
    bw = x2 - x1
    bh = y2 - y1
    sx = int(bw * SHRINK_RATIO)
    sy = int(bh * SHRINK_RATIO)

    rx1 = max(x1 + sx, 0)
    ry1 = max(y1 + sy, 0)
    rx2 = min(x2 - sx, depth_frame.get_width() - 1)
    ry2 = min(y2 - sy, depth_frame.get_height() - 1)

    cx = int((x1 + x2) // 2)
    cy = int((y1 + y2) // 2)
    if rx2 <= rx1 or ry2 <= ry1:
        return depth_frame.get_distance(cx, cy)

    depth_image = np.asanyarray(raw_depth_frame.get_data())
    roi = depth_image[ry1:ry2, rx1:rx2].astype(np.float32) * depth_scale
    valid = roi[(roi > DEPTH_MIN) & (roi < DEPTH_MAX)]
    if len(valid) == 0:
        return depth_frame.get_distance(cx, cy)
    return float(np.percentile(valid, DEPTH_PERCENTILE))


def deproject(intrinsics, px, py, depth_m):
    return rs.rs2_deproject_pixel_to_point(intrinsics, [float(px), float(py)], float(depth_m))


class ArmTaskServer:
    def __init__(self):
        rospy.init_node("arm_task_server", anonymous=False)

        self.cmd_topic = rospy.get_param("~cmd_topic", "/arm_task_cmd")
        self.result_topic = rospy.get_param("~result_topic", "/arm_task_result")
        self.state_topic = rospy.get_param("~state_topic", "/arm_task_state")
        self.pin_pos_topic = rospy.get_param("~pin_pos_topic", "/pin_pos_cmd")
        self.joint_cmd_topic = rospy.get_param("~joint_cmd_topic", "/joint_states")
        self.end_pose_topic = rospy.get_param("~end_pose_topic", "/end_pose")

        self.model_path = rospy.get_param("~model_path", MODEL_PATH)
        self.target_label = rospy.get_param("~target_label", "")
        self.show_image = bool(rospy.get_param("~show_image", True))
        self.release_camera_after_task = bool(rospy.get_param("~release_camera_after_task", True))
        self.place_use_vision = bool(rospy.get_param("~place_use_vision", False))

        self.scan_joint_position = self.get_list_param(
            "~scan_joint_position",
            [1.521, 0.403, -0.521, 0.0, 0.808, 0.0, 0.0],
        )
        self.stow_joint_position = self.get_list_param(
            "~stow_joint_position",
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        self.place_joint_position = self.get_list_param("~place_joint_position", self.scan_joint_position)
        self.place_pose = self.get_list_param("~place_pose", [])
        self.pick_pre_offset = self.get_list_param("~pick_pre_offset", [0.0, -0.022, 0.0])
        self.pick_approach_offset = self.get_list_param("~pick_approach_offset", [0.0, -0.022, 0.04])
        self.pick_lift_offset = self.get_list_param("~pick_lift_offset", [-0.15, -0.022, 0.04])

        self.pos_pub = rospy.Publisher(self.pin_pos_topic, PosCmd, queue_size=1)
        self.joint_pub = rospy.Publisher(self.joint_cmd_topic, JointState, queue_size=1)
        self.result_pub = rospy.Publisher(self.result_topic, String, queue_size=10)
        self.state_pub = rospy.Publisher(self.state_topic, String, queue_size=1, latch=True)

        self.current_ee_pose_mat = np.eye(4)
        self.pose_received = False
        self.busy = False
        self.model = None
        self.pipeline = None
        self.align = None
        self.filters = None
        self.depth_scale = 0.001

        rospy.Subscriber(self.end_pose_topic, PoseStamped, self.pose_callback, queue_size=20)
        rospy.Subscriber(self.cmd_topic, String, self.command_callback, queue_size=10)
        rospy.on_shutdown(self.on_shutdown)

        self.publish_state("idle")
        rospy.loginfo("ArmTaskServer ready. Waiting for /arm_task_cmd pick/place.")

    def get_list_param(self, name, default):
        value = rospy.get_param(name, default)
        if isinstance(value, str):
            value = [x.strip() for x in value.split(",") if x.strip()]
        return [float(x) for x in value]

    def publish_state(self, state):
        self.state_pub.publish(String(data=state))

    def publish_result(self, result):
        rospy.loginfo("Arm task result: %s", result)
        self.result_pub.publish(String(data=result))

    def pose_callback(self, msg):
        q = [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ]
        T = tf_trans.quaternion_matrix(q)
        T[0:3, 3] = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ]
        self.current_ee_pose_mat = T
        self.pose_received = True

    def wait_for_pose(self, timeout=10.0):
        start = rospy.Time.now()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and not self.pose_received:
            if timeout > 0.0 and (rospy.Time.now() - start).to_sec() > timeout:
                return False
            rate.sleep()
        return True

    def command_callback(self, msg):
        task = msg.data.strip().lower()
        if not task:
            return
        if self.busy:
            rospy.logwarn("Arm is busy. Reject task: %s", task)
            self.publish_result("fail:%s" % task)
            return

        self.busy = True
        self.publish_state("running:%s" % task)
        try:
            if task == "pick":
                ok = self.execute_pick()
            elif task == "place":
                ok = self.execute_place()
            elif task == "stow":
                ok = self.move_joint(self.stow_joint_position, sleep_s=6.0)
            else:
                rospy.logwarn("Unknown arm task: %s", task)
                ok = False

            self.publish_result(("done:%s" if ok else "fail:%s") % task)
        except Exception as exc:
            rospy.logerr("Arm task '%s' crashed: %s", task, exc)
            self.publish_result("fail:%s" % task)
        finally:
            if self.release_camera_after_task:
                self.release_camera()
            self.busy = False
            self.publish_state("idle")

    # ------------------------------------------------------------------
    # Camera and detector
    # ------------------------------------------------------------------
    def ensure_camera_and_model(self):
        if self.model is None:
            if not os.path.exists(self.model_path):
                raise RuntimeError("YOLO model not found: %s" % self.model_path)
            rospy.loginfo("Loading bottle model: %s", self.model_path)
            self.model = YOLO(self.model_path)
            rospy.loginfo("Bottle model loaded.")

        if self.pipeline is not None:
            return

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, FPS)
        config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)
        profile = self.pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)
        self.filters = build_filters()
        rospy.loginfo("Arm RealSense opened. depth_scale=%.6f", self.depth_scale)

    def release_camera(self):
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
            self.align = None
            self.filters = None
            rospy.loginfo("Arm RealSense released.")
        if self.show_image:
            try:
                cv2.destroyWindow("Arm Bottle Detection")
            except cv2.error:
                pass

    def make_bottle_pose_in_base(self, bottle_xyz_cam):
        T_cam_bottle = np.eye(4)
        T_cam_bottle[0:3, 3] = np.array(bottle_xyz_cam, dtype=float)
        T_base_point = self.current_ee_pose_mat @ T_EE_TO_CAM @ T_cam_bottle

        T_base_bottle = np.eye(4)
        T_base_bottle[0:3, 3] = T_base_point[0:3, 3]
        T_base_bottle[0:3, 0:3] = build_level_grasp_rotation(self.current_ee_pose_mat[0:3, 0:3])
        return T_base_bottle

    def get_bottle_pose(self, max_frames=60):
        self.ensure_camera_and_model()
        target_label = self.target_label if self.target_label else None
        label_text = target_label if target_label else "any bottle"
        rospy.loginfo("Detecting target: %s", label_text)

        best_candidate = None
        best_score = -1.0

        for frame_idx in range(max_frames):
            if rospy.is_shutdown():
                return None
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            aligned = self.align.process(frames)
            raw_depth = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not raw_depth or not color_frame:
                continue

            filtered_depth = apply_filters(raw_depth, self.filters).as_depth_frame()
            color_image = np.asanyarray(color_frame.get_data())
            annotated = color_image.copy()
            intrinsics = filtered_depth.profile.as_video_stream_profile().intrinsics

            results = self.model.predict(
                source=color_image,
                imgsz=IMG_SIZE,
                conf=CONF_THRES,
                verbose=False,
            )

            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                confs = results[0].boxes.conf.cpu().numpy()
                H, W = color_image.shape[:2]

                for box, conf in zip(boxes, confs):
                    x1, y1, x2, y2 = box.tolist()
                    x1 = max(0, min(x1, W - 1))
                    y1 = max(0, min(y1, H - 1))
                    x2 = max(0, min(x2, W - 1))
                    y2 = max(0, min(y2, H - 1))
                    color_label = classify_bottle_color(color_image, x1, y1, x2, y2)
                    if color_label is None:
                        continue
                    if target_label is not None and color_label != target_label:
                        continue

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    dist_m = get_depth_percentile(filtered_depth, raw_depth, x1, y1, x2, y2, self.depth_scale)
                    if not (DEPTH_MIN < dist_m < DEPTH_MAX):
                        continue

                    xyz_cam = deproject(intrinsics, cx, cy, dist_m)
                    T_base_bottle = self.make_bottle_pose_in_base(xyz_cam)
                    center_penalty = abs(cx - W / 2.0) / W + abs(cy - H / 2.0) / H
                    score = float(conf) - 0.10 * center_penalty
                    if score > best_score:
                        best_score = score
                        best_candidate = T_base_bottle

                    bgr = COLOR_DEFS[color_label]["bgr"]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, 2)
                    cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(annotated, "%s %.2f %.3fm" % (color_label, conf, dist_m),
                                (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)

            if self.show_image:
                cv2.imshow("Arm Bottle Detection", annotated)
                cv2.waitKey(1)

            if best_candidate is not None and frame_idx >= 10:
                xyz_base = best_candidate[0:3, 3]
                rospy.loginfo("Target in arm base: x=%.4f y=%.4f z=%.4f",
                              xyz_base[0], xyz_base[1], xyz_base[2])
                return best_candidate

        rospy.logwarn("No bottle target detected.")
        return None

    # ------------------------------------------------------------------
    # Arm commands
    # ------------------------------------------------------------------
    def move_joint(self, positions, sleep_s=5.0, speed=10.0, effort=0.5):
        if len(positions) < 7:
            rospy.logerr("Joint command needs 7 positions, got %d", len(positions))
            return False
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
        msg.position = list(positions[:7])
        msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, speed]
        msg.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, effort]
        self.joint_pub.publish(msg)
        rospy.sleep(sleep_s)
        return True

    def publish_pose_cmd(self, T_target, gripper_val=0.0):
        roll, pitch, yaw = tf_trans.euler_from_matrix(T_target, axes="sxyz")
        cmd = PosCmd()
        cmd.x, cmd.y, cmd.z = T_target[0:3, 3]
        cmd.roll = roll
        cmd.pitch = pitch
        cmd.yaw = yaw
        cmd.gripper = gripper_val
        cmd.mode1 = 1
        cmd.mode2 = 0
        self.pos_pub.publish(cmd)

    def move_to_target_smooth(self, T_target, v=0.05, rate_hz=100, gripper_val=0.0):
        if not self.wait_for_pose(timeout=10.0):
            rospy.logerr("No /end_pose received; cannot do cartesian move.")
            return False

        rate = rospy.Rate(rate_hz)
        T_start = self.current_ee_pose_mat.copy()
        start_pos = T_start[0:3, 3]
        target_pos = T_target[0:3, 3]
        dist = np.linalg.norm(target_pos - start_pos)
        if dist < 0.001:
            self.publish_pose_cmd(T_target, gripper_val)
            return True

        duration = max(dist / max(v, 0.001), 0.5)
        steps = max(int(duration * rate_hz), 1)
        q_start = tf_trans.quaternion_from_matrix(T_start)
        q_target = tf_trans.quaternion_from_matrix(T_target)

        for i in range(1, steps + 1):
            if rospy.is_shutdown():
                return False
            alpha = i / float(steps)
            curr_pos = start_pos + (target_pos - start_pos) * alpha
            curr_q = tf_trans.quaternion_slerp(q_start, q_target, alpha)
            T_interp = tf_trans.quaternion_matrix(curr_q)
            T_interp[0:3, 3] = curr_pos
            self.publish_pose_cmd(T_interp, gripper_val)
            rate.sleep()
        return True

    def gripper_cmd_at_current_pose(self, gripper_val):
        if not self.wait_for_pose(timeout=5.0):
            return False
        self.publish_pose_cmd(self.current_ee_pose_mat.copy(), gripper_val=gripper_val)
        rospy.sleep(2.0)
        return True

    # ------------------------------------------------------------------
    # High-level tasks
    # ------------------------------------------------------------------
    def execute_pick(self):
        rospy.loginfo("Pick task started.")
        if not self.wait_for_pose(timeout=10.0):
            rospy.logerr("No /end_pose feedback from Piper.")
            return False

        self.move_joint(self.scan_joint_position, sleep_s=8.0)
        T_base_bottle = self.get_bottle_pose(max_frames=70)
        if T_base_bottle is None:
            return False

        T_pre = T_BOTTLE_TO_GRASP.copy()
        T_pre[0:3, 3] = self.pick_pre_offset
        if not self.move_to_target_smooth(T_base_bottle @ T_pre @ T_EE_TO_TOOL, v=0.06, gripper_val=100):
            return False
        rospy.sleep(3.5)

        T_approach = T_BOTTLE_TO_GRASP.copy()
        T_approach[0:3, 3] = self.pick_approach_offset
        if not self.move_to_target_smooth(T_base_bottle @ T_approach @ T_EE_TO_TOOL, v=0.02, gripper_val=100):
            return False
        rospy.sleep(2.5)

        rospy.loginfo("Closing gripper.")
        self.gripper_cmd_at_current_pose(gripper_val=0)

        T_up = T_BOTTLE_TO_GRASP.copy()
        T_up[0:3, 3] = self.pick_lift_offset
        if not self.move_to_target_smooth(T_base_bottle @ T_up @ T_EE_TO_TOOL, v=0.05, gripper_val=0):
            return False
        rospy.sleep(3.0)

        self.move_joint(self.stow_joint_position, sleep_s=8.0)
        rospy.loginfo("Pick task finished.")
        return True

    def execute_place(self):
        rospy.loginfo("Place task started.")
        if not self.wait_for_pose(timeout=10.0):
            rospy.logerr("No /end_pose feedback from Piper.")
            return False

        self.move_joint(self.place_joint_position, sleep_s=8.0)

        if self.place_use_vision:
            T_base_target = self.get_bottle_pose(max_frames=70)
            if T_base_target is None:
                return False
            T_pre = T_BOTTLE_TO_GRASP.copy()
            T_pre[0:3, 3] = [-0.08, -0.08, -0.06]
            if not self.move_to_target_smooth(T_base_target @ T_pre @ T_EE_TO_TOOL, v=0.06, gripper_val=0):
                return False
            rospy.sleep(3.5)
        elif len(self.place_pose) == 6:
            x, y, z, roll, pitch, yaw = self.place_pose
            T_place = tf_trans.euler_matrix(roll, pitch, yaw, axes="sxyz")
            T_place[0:3, 3] = [x, y, z]
            if not self.move_to_target_smooth(T_place, v=0.05, gripper_val=0):
                return False
        else:
            rospy.logwarn("No place_pose set. Opening gripper at place_joint_position.")

        rospy.loginfo("Opening gripper.")
        self.gripper_cmd_at_current_pose(gripper_val=200)
        self.move_joint(self.stow_joint_position, sleep_s=8.0)
        rospy.loginfo("Place task finished.")
        return True

    def on_shutdown(self):
        self.release_camera()
        self.publish_state("shutdown")


if __name__ == "__main__":
    try:
        ArmTaskServer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
