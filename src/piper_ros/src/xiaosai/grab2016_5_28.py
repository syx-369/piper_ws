#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from ultralytics import YOLO

# ===============================
# 瓶子检测参数
# ===============================
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

# None 表示抓取任意检测到的瓶子；也可以改成 "bottle-y"、"bottle-g"、"bottle-b"
TARGET_BOTTLE_LABEL = None

# 是否显示检测窗口。
SHOW_DETECTION_WINDOW = True

COLOR_DEFS = {
    "bottle-y": {  # 橘色
        "hsv_lower": np.array([5, 80, 80]),
        "hsv_upper": np.array([25, 255, 255]),
        "bgr": (0, 140, 255),
    },
    "bottle-g": {  # 绿色
        "hsv_lower": np.array([35, 50, 50]),
        "hsv_upper": np.array([85, 255, 255]),
        "bgr": (0, 200, 0),
    },
    "bottle-b": {  # 黑色
        "hsv_lower": np.array([0, 0, 0]),
        "hsv_upper": np.array([180, 255, 60]),
        "bgr": (80, 80, 80),
    },
}

# ===============================
# 手眼位姿矩阵
# ===============================
# 相机相对于机械臂末端的位姿：T_EE_TO_CAM
T_EE_TO_CAM = np.array([
    [-0.0482,  0.9987,  0.0147, -0.0699],
    [-0.9979, -0.0488,  0.0416,  0.0301],
    [ 0.0423, -0.0127,  0.9990,  0.0675],
    [ 0.0,     0.0,     0.0,     1.0]
], dtype=float)

# 夹爪相对于机械臂末端的位姿
T_EE_TO_TOOL = np.eye(4)
T_EE_TO_TOOL[0:3, 3] = [0.0, 0.0, -0.14]

T_BOTTLE_TO_GRASP = np.eye(4)


def classify_bottle_color(color_image, x1, y1, x2, y2):
    """根据检测框内 HSV 颜色占比，返回 bottle-y/bottle-g/bottle-b 或 None。"""
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
    """在检测框中心区域取深度百分位，避免直接取中心点受空洞/反光影响。"""
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


def draw_detection(img, x1, y1, x2, y2, color_label, conf, dist_m, xyz):
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    bgr = COLOR_DEFS[color_label]["bgr"]

    cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)
    cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
    cv2.putText(img, f"{color_label} {conf:.2f} | {dist_m:.3f}m",
                (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)

    if xyz is not None:
        X, Y, Z = xyz
        cv2.putText(img, f"X:{X:.3f} Y:{Y:.3f} Z:{Z:.3f}",
                    (x1, min(y2 + 20, img.shape[0] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


class PiperVisionController:
    def __init__(self):
        rospy.init_node("piper_bottle_vision_grasp", anonymous=True)

        self.pub = rospy.Publisher("/pin_pos_cmd", PosCmd, queue_size=1)
        self.joint_pub = rospy.Publisher("/joint_states", JointState, queue_size=1)

        self.current_ee_pose_mat = np.eye(4)
        self.pose_received = False
        rospy.Subscriber("/end_pose", PoseStamped, self.pose_callback)

        rospy.loginfo("等待获取机械臂实时位姿 (/end_pose)...")
        while not self.pose_received and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("成功接入实时位姿反馈！")

        self.init_camera_and_detector()

    def pose_callback(self, msg):
        px = msg.pose.position.x
        py = msg.pose.position.y
        pz = msg.pose.position.z
        q = [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ]
        T_mat = tf_trans.quaternion_matrix(q)
        T_mat[0:3, 3] = [px, py, pz]
        self.current_ee_pose_mat = T_mat
        self.pose_received = True

    def init_camera_and_detector(self):
        if not os.path.exists(MODEL_PATH):
            rospy.logwarn(f"未在当前路径找到模型: {MODEL_PATH}。如果模型在别处，请修改 MODEL_PATH 为绝对路径。")

        rospy.loginfo(f"正在加载 YOLO 瓶子检测模型: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)
        rospy.loginfo("YOLO 模型加载完成。")

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, FPS)
        config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS)
        profile = self.pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        rospy.loginfo(f"RealSense depth scale: {self.depth_scale:.6f} m/unit")
        '''
        if depth_sensor.supports(rs.option.visual_preset):
            try:
                depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy
                rospy.loginfo("RealSense visual_preset -> High Accuracy")
            except Exception as e:
                rospy.logwarn(f"设置 RealSense visual_preset 失败: {e}")
        '''
        self.align = rs.align(rs.stream.color)
        self.filters = build_filters()

    def move_to_target_smooth(self, T_target, v=0.05, rate_hz=100, gripper_val=0):
        """平滑移动：位置线性插值 + 四元数球面插值。"""
        rate = rospy.Rate(rate_hz)
        T_start = self.current_ee_pose_mat.copy()
        start_pos = T_start[0:3, 3]
        target_pos = T_target[0:3, 3]

        dist = np.linalg.norm(target_pos - start_pos)
        if dist < 0.001:
            return

        duration = dist / v
        steps = max(int(duration * rate_hz), 1)

        q_start = tf_trans.quaternion_from_matrix(T_start)
        q_target = tf_trans.quaternion_from_matrix(T_target)

        for i in range(1, steps + 1):
            if rospy.is_shutdown():
                break
            alpha = i / float(steps)
            curr_pos = start_pos + (target_pos - start_pos) * alpha
            curr_q = tf_trans.quaternion_slerp(q_start, q_target, alpha)

            T_interp = tf_trans.quaternion_matrix(curr_q)
            roll, pitch, yaw = tf_trans.euler_from_matrix(T_interp, axes="sxyz")

            cmd = PosCmd()
            cmd.x, cmd.y, cmd.z = curr_pos
            cmd.roll, cmd.pitch, cmd.yaw = roll, pitch, yaw
            cmd.gripper = gripper_val
            cmd.mode1 = 1
            cmd.mode2 = 0
            self.pub.publish(cmd)
            rate.sleep()

    def make_bottle_pose_in_base(self, bottle_xyz_cam):
        """
        将相机坐标系下瓶子 3D 点转换为基座坐标系位姿。
        注意：深度检测只有位置没有姿态，所以这里让目标姿态默认沿用当前末端姿态。
        """
        T_cam_bottle_point = np.eye(4)
        T_cam_bottle_point[0:3, 3] = np.array(bottle_xyz_cam, dtype=float)

        # 先用手眼矩阵把瓶子中心点从相机系转换到基座系
        T_base_bottle_point = self.current_ee_pose_mat @ T_EE_TO_CAM @ T_cam_bottle_point

        # 只取位置，姿态沿用当前末端姿态，避免 YOLO 点目标导致姿态乱跳
        T_base_bottle = self.current_ee_pose_mat.copy()
        T_base_bottle[0:3, 3] = T_base_bottle_point[0:3, 3]
        return T_base_bottle

    def get_bottle_pose(self, target_label=TARGET_BOTTLE_LABEL, max_frames=60):
        """检测瓶子并返回瓶子在基座坐标系下的位姿矩阵。"""
        label_text = target_label if target_label is not None else "任意颜色瓶子"
        rospy.loginfo(f"正在尝试检测目标瓶子: {label_text}")

        best_candidate = None
        best_score = -1.0
        last_annotated = None

        for _ in range(max_frames):
            if rospy.is_shutdown():
                break

            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError as e:
                rospy.logwarn(f"RealSense 等待图像超时: {e}")
                continue

            aligned = self.align.process(frames)
            raw_depth = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not raw_depth or not color_frame:
                continue

            filtered_depth = apply_filters(raw_depth, self.filters).as_depth_frame()
            color_image = np.asanyarray(color_frame.get_data())
            annotated = color_image.copy()
            depth_intrinsics = filtered_depth.profile.as_video_stream_profile().intrinsics

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
                    dist_m = get_depth_percentile(
                        filtered_depth, raw_depth, x1, y1, x2, y2, self.depth_scale
                    )
                    if not (DEPTH_MIN < dist_m < DEPTH_MAX):
                        continue

                    xyz_cam = deproject(depth_intrinsics, cx, cy, dist_m)
                    T_base_bottle = self.make_bottle_pose_in_base(xyz_cam)
                    xyz_base = T_base_bottle[0:3, 3]

                    draw_detection(annotated, x1, y1, x2, y2, color_label, float(conf), dist_m, xyz_cam)

                    # 候选目标打分：优先置信度高，轻微偏好画面中心附近，避免误抓边缘目标
                    center_penalty = abs(cx - W / 2.0) / W + abs(cy - H / 2.0) / H
                    score = float(conf) - 0.10 * center_penalty
                    if score > best_score:
                        best_score = score
                        best_candidate = {
                            "label": color_label,
                            "conf": float(conf),
                            "center": (cx, cy),
                            "dist_m": float(dist_m),
                            "xyz_cam": np.array(xyz_cam, dtype=float),
                            "xyz_base": np.array(xyz_base, dtype=float),
                            "T_base_bottle": T_base_bottle,
                        }

            last_annotated = annotated
            if SHOW_DETECTION_WINDOW:
                try:
                    cv2.imshow("Bottle Detection For Grasp", annotated)
                    cv2.waitKey(1)
                except cv2.error:
                    pass

            if best_candidate is not None and _ >= 10:
                c = best_candidate
                rospy.loginfo("========================================")
                rospy.loginfo(f"检测成功！瓶子类别: {c['label']}  conf={c['conf']:.2f}")
                rospy.loginfo(f"像素中心: {c['center']}  深度: {c['dist_m']:.4f} m")
                rospy.loginfo("相机坐标系下瓶子中心 XYZ:")
                rospy.loginfo(f"  X: {c['xyz_cam'][0]:.4f} m")
                rospy.loginfo(f"  Y: {c['xyz_cam'][1]:.4f} m")
                rospy.loginfo(f"  Z: {c['xyz_cam'][2]:.4f} m")
                rospy.loginfo("基座坐标系下瓶子中心 XYZ:")
                rospy.loginfo(f"  X: {c['xyz_base'][0]:.4f} m")
                rospy.loginfo(f"  Y: {c['xyz_base'][1]:.4f} m")
                rospy.loginfo(f"  Z: {c['xyz_base'][2]:.4f} m")
                rospy.loginfo("========================================")
                return c["T_base_bottle"]

        if last_annotated is not None and SHOW_DETECTION_WINDOW:
            try:
                cv2.imshow("Bottle Detection For Grasp", last_annotated)
                cv2.waitKey(1)
            except cv2.error:
                pass

        rospy.logwarn(f"超时：未检测到目标瓶子: {label_text}")
        return None

    def execute_vision_grasp(self):
        rospy.loginfo(">>> 开始瓶子实物视觉抓取任务")

        # 1. 移动到拍照位置，沿用原脚本 JointState 初始姿态
        rospy.loginfo(">>> 移动到初始位")
        joint_msg = JointState()
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.name = [""]
        #joint_msg.position = [0.0, 0.424, 0.0, 0.0, -0.099, 0.0, 0.0]
        joint_msg.position = [0.0, 0.912, -1.283, 0.0, 1.220, 0.0, 0.0]
        joint_msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]
        joint_msg.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
        self.joint_pub.publish(joint_msg)
        time.sleep(6.0)

        # 2. 第一次识别瓶子中心位置
        T_base_bottle = self.get_bottle_pose(TARGET_BOTTLE_LABEL)
        if T_base_bottle is None:
            return

        # 3. 移动到预抓取位
        # T_pre[0:3, 3] = [+x（下）, +y（左）, +z（前）]
        rospy.loginfo(">>> 移动到预抓取位")
        T_pre = T_BOTTLE_TO_GRASP.copy()
        T_pre[0:3, 3] = [0.0, -0.015, 0.0]
        target_ee_pre = T_base_bottle @ T_pre @ T_EE_TO_TOOL
        self.move_to_target_smooth(target_ee_pre, v=0.06, gripper_val=100)
        time.sleep(3.5)

        '''
        # 4. 第二次近距离识别，提高位置精度
        T_base_bottle_fine = self.get_bottle_pose(TARGET_BOTTLE_LABEL, max_frames=45)
        if T_base_bottle_fine is not None:
            T_base_bottle = T_base_bottle_fine
        

        # 5a. 移动到抓取点上方
        T_t1 = T_BOTTLE_TO_GRASP.copy()
        T_t1[0:3, 3] = [0.0, 0.0, 0.07]
        self.move_to_target_smooth(T_base_bottle @ T_t1 @ T_EE_TO_TOOL, v=0.05, gripper_val=100)
        time.sleep(2.5)
        '''

        # 5b. 慢速靠近瓶子
        rospy.loginfo(">>> 前探夹爪")
        T_t2 = T_BOTTLE_TO_GRASP.copy()
        T_t2[0:3, 3] = [0.0, -0.015, 0.04]
        self.move_to_target_smooth(T_base_bottle @ T_t2 @ T_EE_TO_TOOL, v=0.02, gripper_val=100)
        time.sleep(2.5)

        # 5c. 闭合夹爪
        rospy.loginfo(">>> 闭合夹爪")
        curr_pose = self.current_ee_pose_mat.copy()
        r, p, yaw = tf_trans.euler_from_matrix(curr_pose, axes="sxyz")
        cmd_close = PosCmd()
        cmd_close.x, cmd_close.y, cmd_close.z = curr_pose[0:3, 3]
        cmd_close.roll, cmd_close.pitch, cmd_close.yaw = r, p, yaw
        cmd_close.gripper = 0
        cmd_close.mode1 = 1
        cmd_close.mode2 = 0
        self.pub.publish(cmd_close)
        time.sleep(2.0)

        # 5d. 提升
        rospy.loginfo(">>> 提起瓶子")
        T_up = T_BOTTLE_TO_GRASP.copy()
        T_up[0:3, 3] = [-0.05, -0.015, 0.04]
        self.move_to_target_smooth(T_base_bottle @ T_up @ T_EE_TO_TOOL, v=0.05, gripper_val=0)

        rospy.loginfo(">>> 瓶子实物视觉抓取任务结束")

    def run(self):
        try:
            self.execute_vision_grasp()
        finally:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


if __name__ == "__main__":
    try:
        node = PiperVisionController()
        node.run()
    except rospy.ROSInterruptException:
        pass
