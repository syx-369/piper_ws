#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import rospy
import rospkg
import tf.transformations as tf_trans
from geometry_msgs.msg import PoseStamped
from piper_msgs.msg import PosCmd
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from ultralytics import YOLO

# ===============================
# 目标检测参数
# ===============================
MODEL_PATH = os.path.join(
    rospkg.RosPack().get_path("piper_task"),
    "config",
    "weights",
    "bb_0816.pt",
)
CONF_THRES = 0.5
# 红色方块容易受反光和曝光影响，先用较低阈值保留候选框，
# 再在下面通过 HSV 红色占比进行二次确认。
RED_BLOCK_CONF_THRES = 0.20
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

# 比赛抓放位置的合理相机深度。正常情况下仍使用原始深度算法；只有原始
# 深度超出这里的任务范围时，才尝试从滤波帧中恢复前景深度并留在原地重试。
PICK_DEPTH_MIN = 0.25
PICK_DEPTH_MAX = 0.70
PLACE_DEPTH_MIN = 0.30
PLACE_DEPTH_MAX = 0.70
DEPTH_RETRY_EXTRA_FRAMES = 45
DEPTH_STABLE_SAMPLES = 5
DEPTH_STABILITY_MAX_SPREAD = 0.03
DEPTH_FALLBACK_MIN_VALID_RATIO = 0.05

# 抓取时这两个参数限制检测框中心；检测框本身可以进入软边距，但必须与
# 真实画面边缘保持 PICK_HARD_BORDER_MARGIN，防止使用已经被裁切的目标。
# 放置仍沿用整个检测框必须位于软安全区域内的保守规则。
OBJECT_EDGE_MARGIN_X = 40
OBJECT_EDGE_MARGIN_Y = 30
PICK_HARD_BORDER_MARGIN = 8

# 红色在 OpenCV HSV 中跨越 0/180，因此需要两个色相区间。
RED_HSV_LOWER_1 = np.array([0, 60, 40])
RED_HSV_UPPER_1 = np.array([12, 255, 255])
RED_HSV_LOWER_2 = np.array([165, 60, 40])
RED_HSV_UPPER_2 = np.array([180, 255, 255])
RED_BLOCK_MIN_RATIO = 0.06
COLOR_ROI_SHRINK_RATIO = 0.10

TRACK_MAX_PIXEL_DISTANCE = 60.0
TRACK_MAX_DEPTH_DIFF = 0.15

# 检测画面统一发布到 /piper_task/detection_image，由 final_vision 在
# "Final Mission Vision" 窗口显示；这里不再创建第二个本地窗口。
SHOW_DETECTION_WINDOW = False

# bb_0816.pt 中的类别以及检测框显示颜色（OpenCV BGR）。
CLASS_BGR = {
    "block-r": (0, 0, 255),
    "block-y": (0, 255, 255),
    "block-b": (255, 0, 0),
    "bottle-b": (80, 80, 80),
    "bottle-y": (0, 140, 255),
    "bottle-g": (0, 200, 0),
}

VALID_TARGET_LABELS = {
    "block": {"block-r", "block-y", "block-b"},
    "bottle": {"bottle-b", "bottle-y", "bottle-g"},
}

# ===============================
# 抓取前的相机图像识别参数
# ===============================
REFERENCE_CONF_THRES = 0.20
REFERENCE_MAX_FRAMES = 90
REFERENCE_WARMUP_FRAMES = 15
REFERENCE_MIN_STABLE_DETECTIONS = 3

LABEL_DESCRIPTION = {
    "block-r": "红色方块",
    "block-y": "黄色方块",
    "block-b": "蓝色方块",
    "bottle-b": "黑色瓶子",
    "bottle-y": "黄色瓶子",
    "bottle-g": "绿色瓶子",
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

T_CAM_SHIFT = np.array([
    [1,0,0,0],
    # OpenCV 相机坐标 Y 向下；支架累计上移 20 mm，因此沿相机 -Y 平移 0.02 m。
    [0,1,0,-0.02],
    [0,0,1,0],
    [0,0,0,1]
])
T_EE_TO_CAM =T_EE_TO_CAM @ T_CAM_SHIFT
# 夹爪相对于机械臂末端的位姿
T_EE_TO_TOOL = np.eye(4)
T_EE_TO_TOOL[0:3, 3] = [0.0, 0.0, -0.14]

# 这个矩阵后面只作为“相对目标中心的平移偏置”使用。
# 目标抓取姿态不再由这里的旋转决定，而是在 make_object_pose_in_base()
# 中通过 build_level_grasp_rotation() 动态构造。
T_OBJECT_TO_GRASP = np.eye(4)


def normalize_vector(v, eps=1e-9):
    """向量归一化，避免除零。"""
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        return None
    return v / n


def build_level_grasp_rotation(reference_R=None):
    """
    构造“只约束夹爪水平”的目标旋转矩阵。

    约束条件：
        夹爪/抓取坐标系 X 轴 = 基坐标系 -Z 方向
        即 grasp_x = [0, 0, -1]

    不再强制：
        grasp_z = base_x 或 grasp_y = base_y

    做法：
        1. 固定 grasp_x 为 [0, 0, -1]，保证夹爪 X 轴与基坐标系 Z 轴反向重合；
        2. 从当前末端姿态 reference_R 中取一个水平面内的参考方向，
           尽量保留当前末端在水平面内的朝向；
        3. 用右手系补全 grasp_y、grasp_z。

    返回：
        R_base_grasp，3x3 旋转矩阵。矩阵三列分别是：
        grasp_x、grasp_y、grasp_z 在基坐标系下的方向。
    """
    # 只强制这一条：夹爪 X 轴朝向基坐标系 -Z。
    grasp_x = np.array([0.0, 0.0, -1.0], dtype=float)

    candidate_refs = []
    if reference_R is not None:
        reference_R = np.array(reference_R, dtype=float)
        # 优先沿用当前末端的 Z 轴作为水平面内朝向参考。
        # 如果当前 Z 轴接近竖直，投影后会退化，再尝试当前 Y/X 轴。
        candidate_refs.extend([
            reference_R[0:3, 2],
            reference_R[0:3, 1],
            reference_R[0:3, 0],
        ])

    # 最后的兜底方向。只有在当前末端几个轴都不适合时才会用到。
    candidate_refs.extend([
        np.array([1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float),
    ])

    grasp_z = None
    for ref in candidate_refs:
        # 去掉 ref 在 grasp_x 方向上的分量，只保留水平面内分量。
        z_candidate = ref - np.dot(ref, grasp_x) * grasp_x
        z_candidate = normalize_vector(z_candidate)
        if z_candidate is not None:
            grasp_z = z_candidate
            break

    if grasp_z is None:
        # 理论上不会走到这里，只是为了安全。
        grasp_z = np.array([1.0, 0.0, 0.0], dtype=float)

    # 右手系要求：grasp_x × grasp_y = grasp_z
    # 因此 grasp_y = grasp_z × grasp_x
    grasp_y = np.cross(grasp_z, grasp_x)
    grasp_y = normalize_vector(grasp_y)

    R_base_grasp = np.column_stack((grasp_x, grasp_y, grasp_z))
    return R_base_grasp


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


def get_filtered_depth_in_range(
    depth_frame,
    x1,
    y1,
    x2,
    y2,
    depth_scale,
    depth_min,
    depth_max,
    min_valid_ratio,
):
    """
    原始深度异常时，从滤波后的检测框中心区域寻找任务距离内的前景深度。

    只有有效像素达到一定比例才返回结果，避免用一两个噪点代替整件物体。
    返回 (depth_m, valid_ratio)；无法可靠恢复时 depth_m 为 None。
    """
    bw = x2 - x1
    bh = y2 - y1
    sx = int(bw * SHRINK_RATIO)
    sy = int(bh * SHRINK_RATIO)

    rx1 = max(x1 + sx, 0)
    ry1 = max(y1 + sy, 0)
    rx2 = min(x2 - sx, depth_frame.get_width() - 1)
    ry2 = min(y2 - sy, depth_frame.get_height() - 1)
    if rx2 <= rx1 or ry2 <= ry1:
        return None, 0.0

    depth_image = np.asanyarray(depth_frame.get_data())
    roi = depth_image[ry1:ry2, rx1:rx2].astype(np.float32) * depth_scale
    if roi.size == 0:
        return None, 0.0

    valid = roi[(roi > depth_min) & (roi < depth_max)]
    valid_ratio = float(valid.size) / float(roi.size)
    min_valid_pixels = max(20, int(np.ceil(roi.size * min_valid_ratio)))
    if valid.size < min_valid_pixels:
        return None, valid_ratio

    return float(np.percentile(valid, DEPTH_PERCENTILE)), valid_ratio


def is_depth_in_range(depth_m, depth_min, depth_max):
    """任务深度必须有限且严格位于安全范围内。"""
    return bool(np.isfinite(depth_m) and depth_min < depth_m < depth_max)


def deproject(intrinsics, px, py, depth_m):
    return rs.rs2_deproject_pixel_to_point(intrinsics, [float(px), float(py)], float(depth_m))


def get_red_ratio(color_image, x1, y1, x2, y2):
    """计算检测框中心区域内的红色像素占比，用于确认低置信度红方块。"""
    h, w = color_image.shape[:2]
    bw = max(int(x2 - x1), 0)
    bh = max(int(y2 - y1), 0)
    sx = int(bw * COLOR_ROI_SHRINK_RATIO)
    sy = int(bh * COLOR_ROI_SHRINK_RATIO)

    rx1 = max(int(x1) + sx, 0)
    ry1 = max(int(y1) + sy, 0)
    rx2 = min(int(x2) - sx, w)
    ry2 = min(int(y2) - sy, h)
    if rx2 <= rx1 or ry2 <= ry1:
        return 0.0

    roi = color_image[ry1:ry2, rx1:rx2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_1 = cv2.inRange(hsv, RED_HSV_LOWER_1, RED_HSV_UPPER_1)
    mask_2 = cv2.inRange(hsv, RED_HSV_LOWER_2, RED_HSV_UPPER_2)
    red_mask = cv2.bitwise_or(mask_1, mask_2)

    # 去除少量孤立噪点，避免背景中的单个红色像素触发候选。
    kernel = np.ones((3, 3), dtype=np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    return float(np.count_nonzero(red_mask)) / float(red_mask.size)


def draw_detection(img, x1, y1, x2, y2, class_label, conf, dist_m, xyz):
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    bgr = CLASS_BGR.get(class_label, (255, 255, 255))

    cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)
    cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
    cv2.putText(img, f"{class_label} {conf:.2f} | {dist_m:.3f}m",
                (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)

    if xyz is not None:
        X, Y, Z = xyz
        cv2.putText(img, f"X:{X:.3f} Y:{Y:.3f} Z:{Z:.3f}",
                    (x1, min(y2 + 20, img.shape[0] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


def is_box_inside_safe_region(x1, y1, x2, y2, width, height, margin_x, margin_y):
    """检查整个检测框是否完整位于定位安全区域内。"""
    return (
        x1 >= margin_x
        and y1 >= margin_y
        and x2 <= width - 1 - margin_x
        and y2 <= height - 1 - margin_y
    )


def box_outside_safe_edges(x1, y1, x2, y2, width, height, margin_x, margin_y):
    """返回检测框侵入的安全边界方向，供失败恢复选择复查位置。"""
    edges = []
    if x1 < margin_x:
        edges.append("left")
    if x2 > width - 1 - margin_x:
        edges.append("right")
    if y1 < margin_y:
        edges.append("top")
    if y2 > height - 1 - margin_y:
        edges.append("bottom")
    return tuple(edges)


def pick_box_unsafe_edges(
    x1,
    y1,
    x2,
    y2,
    width,
    height,
    center_margin_x,
    center_margin_y,
    hard_border_margin,
):
    """抓取专用边缘判断：只保留左、右限制，上下不做区域限制。"""
    cx = 0.5 * (x1 + x2)
    hard_right = width - 1 - hard_border_margin
    center_right = width - 1 - center_margin_x
    edges = []
    if x1 < hard_border_margin or cx < center_margin_x:
        edges.append("left")
    if x2 > hard_right or cx > center_right:
        edges.append("right")
    return tuple(edges)


def draw_safe_detection_region(
    img,
    margin_x,
    margin_y,
    allow_top_edge=False,
    allow_bottom_edge=False,
):
    """在调试画面中画出允许生成抓放位姿的区域。"""
    height, width = img.shape[:2]
    right = width - 1 - margin_x
    top = 0 if allow_top_edge else margin_y
    bottom = height - 1 if allow_bottom_edge else height - 1 - margin_y
    if right <= margin_x or bottom <= top:
        return
    cv2.rectangle(
        img,
        (margin_x, top),
        (right, bottom),
        (0, 255, 255),
        1,
    )


def draw_edge_rejection(img, x1, y1, x2, y2, class_label, conf):
    """标记因贴近画面边缘而不能用于抓放定位的目标。"""
    height, width = img.shape[:2]
    color = (0, 165, 255)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    text_x = max(5, min(x1, width - 220))
    text_y = max(22, min(y1 + 22, height - 5))
    cv2.putText(
        img,
        f"EDGE REJECT: {class_label} {conf:.2f}",
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )


def draw_depth_retry(img, x1, y1, x2, y2, class_label, depth_m):
    """标记类别正确但深度异常、正在原地重试的目标。"""
    height, width = img.shape[:2]
    color = (255, 0, 255)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    text_x = max(5, min(x1, width - 250))
    text_y = max(22, min(y1 + 22, height - 5))
    depth_text = "nan" if not np.isfinite(depth_m) else f"{depth_m:.3f}m"
    cv2.putText(
        img,
        f"DEPTH RETRY: {class_label} {depth_text}",
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )


def draw_reference_detection(img, x1, y1, x2, y2, class_label, conf):
    """绘制参考图片识别结果；该结果只决定抓取类别，不参与抓取定位。"""
    bgr = CLASS_BGR.get(class_label, (255, 255, 255))
    cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 3)
    text = f"TARGET: {class_label} {conf:.2f}"
    cv2.putText(
        img,
        text,
        (x1, max(y1 - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        bgr,
        2,
    )


class PiperVisionController:
    def __init__(self):
        rospy.init_node("piper_object_vision_grasp", anonymous=True)

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

    def publish_detection_status(self, status):
        """发布机械臂视觉阶段/结果，供 final_race 终端直接显示。"""
        publisher = getattr(self, "detection_status_pub", None)
        if publisher is not None:
            publisher.publish(String(data=str(status)))

    def publish_detection_image(self, image):
        """发布当前检测标注图；相机仍只由本进程读取，不引入并发取帧。"""
        publisher = getattr(self, "detection_image_pub", None)
        if publisher is None or image is None:
            return
        image = np.ascontiguousarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            return
        message = Image()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = "hand_cam"
        message.height = image.shape[0]
        message.width = image.shape[1]
        message.encoding = "bgr8"
        message.is_bigendian = 0
        message.step = image.shape[1] * 3
        message.data = image.tobytes()
        publisher.publish(message)

    def init_camera_and_detector(self):
        if not os.path.exists(MODEL_PATH):
            rospy.logwarn(f"未在当前路径找到模型: {MODEL_PATH}。如果模型在别处，请修改 MODEL_PATH 为绝对路径。")

        rospy.loginfo(f"正在加载 YOLO 方块/瓶子检测模型: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)
        rospy.loginfo(f"YOLO 模型加载完成，类别: {self.model.names}")

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
        self.object_edge_margin_x = max(
            0, int(rospy.get_param("~object_edge_margin_x", OBJECT_EDGE_MARGIN_X))
        )
        self.object_edge_margin_y = max(
            0, int(rospy.get_param("~object_edge_margin_y", OBJECT_EDGE_MARGIN_Y))
        )
        self.pick_hard_border_margin = max(
            0,
            int(
                rospy.get_param(
                    "~pick_hard_border_margin", PICK_HARD_BORDER_MARGIN
                )
            ),
        )
        self.pick_depth_min = float(
            rospy.get_param("~pick_depth_min", PICK_DEPTH_MIN)
        )
        self.pick_depth_max = float(
            rospy.get_param("~pick_depth_max", PICK_DEPTH_MAX)
        )
        self.place_depth_min = float(
            rospy.get_param("~place_depth_min", PLACE_DEPTH_MIN)
        )
        self.place_depth_max = float(
            rospy.get_param("~place_depth_max", PLACE_DEPTH_MAX)
        )
        self.depth_retry_extra_frames = max(
            0,
            int(
                rospy.get_param(
                    "~depth_retry_extra_frames", DEPTH_RETRY_EXTRA_FRAMES
                )
            ),
        )
        self.depth_stable_samples = max(
            3, int(rospy.get_param("~depth_stable_samples", DEPTH_STABLE_SAMPLES))
        )
        self.depth_stability_max_spread = max(
            0.001,
            float(
                rospy.get_param(
                    "~depth_stability_max_spread", DEPTH_STABILITY_MAX_SPREAD
                )
            ),
        )
        self.depth_fallback_min_valid_ratio = min(
            1.0,
            max(
                0.0,
                float(
                    rospy.get_param(
                        "~depth_fallback_min_valid_ratio",
                        DEPTH_FALLBACK_MIN_VALID_RATIO,
                    )
                ),
            ),
        )
        if not (
            DEPTH_MIN < self.pick_depth_min < self.pick_depth_max < DEPTH_MAX
        ):
            raise ValueError("抓取深度范围配置无效")
        if not (
            DEPTH_MIN < self.place_depth_min < self.place_depth_max < DEPTH_MAX
        ):
            raise ValueError("放置深度范围配置无效")
        rospy.loginfo(
            "抓取画面安全规则: 中心边距 left/right=%d px, "
            "top/bottom=不限制；检测框距 left/right 真实边界至少%d px",
            self.object_edge_margin_x,
            self.pick_hard_border_margin,
        )
        rospy.loginfo(
            "放置画面安全规则: 整个检测框位于 left/right=%d px, "
            "top/bottom=%d px 以内",
            self.object_edge_margin_x,
            self.object_edge_margin_y,
        )
        rospy.loginfo(
            "抓取深度范围: %.3f~%.3f m；放置深度范围: %.3f~%.3f m；"
            "连续有效=%d帧，最大跨度=%.3f m，异常额外重试=%d帧",
            self.pick_depth_min,
            self.pick_depth_max,
            self.place_depth_min,
            self.place_depth_max,
            self.depth_stable_samples,
            self.depth_stability_max_spread,
            self.depth_retry_extra_frames,
        )

    def detect_target_in_reference_image(self, image):
        """
        从一帧参考图片中识别目标物品及颜色。

        模型的六个类别已经同时编码物品类型和颜色。红色方块另外沿用原脚本的
        HSV 复核，以降低反光导致模型把红色误判成其他颜色方块的概率。

        返回：
            (label, confidence, annotated_image)，未识别到时 label 为 None。
        """
        annotated = image.copy()
        results = self.model.predict(
            source=image,
            imgsz=IMG_SIZE,
            conf=REFERENCE_CONF_THRES,
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None, 0.0, annotated

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        height, width = image.shape[:2]
        valid_labels = set().union(*VALID_TARGET_LABELS.values())
        best = None

        for box, conf, class_id in zip(boxes, confs, class_ids):
            label = str(self.model.names[int(class_id)])
            if label not in valid_labels:
                continue

            x1, y1, x2, y2 = box.tolist()
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            # 与抓取目标检测保持一致：方块候选中红色占比足够时按红色方块处理。
            if label in VALID_TARGET_LABELS["block"]:
                red_ratio = get_red_ratio(image, x1, y1, x2, y2)
                if red_ratio >= RED_BLOCK_MIN_RATIO:
                    label = "block-r"

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            area_ratio = ((x2 - x1) * (y2 - y1)) / float(width * height)
            center_penalty = abs(cx - width / 2.0) / width + abs(cy - height / 2.0) / height
            # 参考图片通常只有一个主体；优先高置信度、大面积、靠近画面中心的目标。
            score = float(conf) + 0.20 * area_ratio - 0.05 * center_penalty
            candidate = {
                "label": label,
                "conf": float(conf),
                "score": score,
                "box": (x1, y1, x2, y2),
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate

        if best is None:
            return None, 0.0, annotated

        draw_reference_detection(
            annotated,
            *best["box"],
            best["label"],
            best["conf"],
        )
        return best["label"], best["conf"], annotated

    def recognize_reference_target(self):
        """在机械臂执行抓取前，从 RealSense 实时画面确定目标。"""
        rospy.loginfo(
            ">>> 请将只包含一个目标物品的参考图片置于 RealSense 镜头前；"
            "正在识别图片中的物品类型和颜色..."
        )
        self.publish_detection_status("card:start:recognizing_reference")
        stable_label = None
        stable_hits = 0

        for frame_index in range(REFERENCE_MAX_FRAMES + REFERENCE_WARMUP_FRAMES):
            if rospy.is_shutdown():
                return None
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError as exc:
                rospy.logwarn(f"RealSense 等待参考图片超时: {exc}")
                continue

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            if frame_index < REFERENCE_WARMUP_FRAMES:
                continue

            image = np.asanyarray(color_frame.get_data())
            label, conf, annotated = self.detect_target_in_reference_image(image)
            if label is not None and label == stable_label:
                stable_hits += 1
            elif label is not None:
                stable_label = label
                stable_hits = 1
            else:
                stable_label = None
                stable_hits = 0

            self.publish_detection_image(annotated)

            if SHOW_DETECTION_WINDOW:
                try:
                    cv2.imshow("Reference Image Target", annotated)
                    cv2.waitKey(1)
                except cv2.error:
                    pass

            if stable_label is not None and stable_hits >= REFERENCE_MIN_STABLE_DETECTIONS:
                self.publish_detection_status(
                    "card:success:label=%s:conf=%.2f:hits=%d"
                    % (stable_label, conf, stable_hits)
                )
                rospy.loginfo(
                    f">>> 参考图片识别成功: {LABEL_DESCRIPTION[stable_label]} "
                    f"({stable_label}), conf={conf:.2f}, 连续命中={stable_hits}帧"
                )
                return stable_label

        rospy.logerr("未能稳定识别参考图片中的物品类型和颜色，任务终止")
        self.publish_detection_status("card:failed:reference_not_recognized")
        return None

    @staticmethod
    def target_type_from_label(label):
        """将模型类别转换为原抓取代码使用的 target_type。"""
        for target_type, labels in VALID_TARGET_LABELS.items():
            if label in labels:
                return target_type
        return None

    def get_place_pose_by_image_style(self, initial_image_style, max_frames=90):
        """
        定位与任务开始时识别结果完全一致的图片，作为放置位置。

        initial_image_style 同时包含物品类型和颜色，例如 block-r、bottle-g。
        放置阶段只使用任务开始时保存的这个值，不重新选择目标样式，防止在
        放置区存在多张图片时放到其他类型或其他颜色的图片处。
        """
        target_type = self.target_type_from_label(initial_image_style)
        if target_type is None:
            rospy.logerr(f"无法识别的初始图片样式: {initial_image_style}")
            return None

        rospy.loginfo(
            f">>> 正在定位与最初识别结果一致的放置图片: "
            f"{LABEL_DESCRIPTION[initial_image_style]} ({initial_image_style})"
        )
        return self.get_object_pose(
            target_type,
            target_label=initial_image_style,
            max_frames=max_frames,
        )

    def get_any_place_pose(self, max_frames=90):
        """寻找任意可安全放置的已知物体，并返回位姿和实际标签。

        这是正确目标全部复查失败后的降级策略。检测仍完整复用放置场景的
        边缘、深度和连续帧约束，只放宽类别/颜色必须与手中物品一致的要求。
        """
        first_budget = max(15, int(max_frames) // 2)
        second_budget = max(15, int(max_frames) - first_budget)
        for target_type, frame_budget in (
            ("block", first_budget),
            ("bottle", second_budget),
        ):
            pose = self.get_object_pose(
                target_type,
                target_label=None,
                max_frames=frame_budget,
            )
            if pose is not None:
                return pose, self.last_detected_object_label
        return None, None

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

    def make_object_pose_in_base(self, object_xyz_cam):
        """
        将相机坐标系下目标物体 3D 点转换为基座坐标系下的目标抓取位姿。

        YOLO + 深度图只能稳定得到物体中心点，不能像 ArUco 一样直接得到物体的
        完整姿态。因此这里的处理方式是：

        1. 位置：
           用 T_base_ee @ T_ee_cam @ T_cam_object_point
           把物体中心从相机坐标系转换到基坐标系。

        2. 姿态：
           不再沿用当前末端完整姿态，也不强制某个固定水平朝向；
           只约束夹爪/抓取坐标系 X 轴 = 基坐标系 -Z 方向，保证夹爪水平。
           水平面内的朝向由当前末端姿态投影得到，避免每次都锁死到 base_x/base_y。
        """
        T_cam_object_point = np.eye(4)
        T_cam_object_point[0:3, 3] = np.array(object_xyz_cam, dtype=float)

        # 目标中心点：相机坐标系 -> 基坐标系
        T_base_object_point = self.current_ee_pose_mat @ T_EE_TO_CAM @ T_cam_object_point

        # 方块和瓶子采用相同的水平抓取姿态。
        T_base_object = np.eye(4)
        T_base_object[0:3, 3] = T_base_object_point[0:3, 3]
        T_base_object[0:3, 0:3] = build_level_grasp_rotation(
            self.current_ee_pose_mat[0:3, 0:3]
        )
        return T_base_object

    def get_object_pose(self, target_type, target_label=None, max_frames=90):
        """检测指定类型/颜色的方块或瓶子，并返回其在基座坐标系下的位姿。"""
        # 调用方在返回 None 时读取该字段，区分“完全没看见”和“正确目标在边缘”。
        # 每次检测开始必须先清空，避免沿用上一个候选点的失败原因。
        self.last_object_detection_failure = "not_found"
        self.last_detected_object_label = None
        if target_type not in VALID_TARGET_LABELS:
            rospy.logerr(f"不支持的目标类型: {target_type}，只能是 block 或 bottle")
            return None

        if target_label is not None and target_label not in VALID_TARGET_LABELS[target_type]:
            rospy.logerr(
                f"目标类别 {target_label} 与目标类型 {target_type} 不匹配；"
                f"可选类别: {sorted(VALID_TARGET_LABELS[target_type])}"
            )
            return None

        object_text = "方块" if target_type == "block" else "瓶子"
        label_text = target_label if target_label is not None else f"任意颜色{object_text}"
        predict_conf = (
            RED_BLOCK_CONF_THRES if target_label == "block-r" else CONF_THRES
        )
        rospy.loginfo(
            f"正在尝试检测目标: {label_text}，推理置信度阈值: {predict_conf:.2f}"
        )
        detection_context = str(getattr(self, "detection_context", "object"))
        is_pick_context = detection_context.startswith("pick")
        if detection_context.startswith("place"):
            task_depth_min = self.place_depth_min
            task_depth_max = self.place_depth_max
        else:
            task_depth_min = self.pick_depth_min
            task_depth_max = self.pick_depth_max
        self.publish_detection_status(
            "%s:start:label=%s:conf_threshold=%.2f:depth_range=%.3f-%.3f"
            % (
                detection_context,
                label_text,
                predict_conf,
                task_depth_min,
                task_depth_max,
            )
        )
        rospy.loginfo(
            "%s 本次允许深度范围: %.3f~%.3f m",
            detection_context,
            task_depth_min,
            task_depth_max,
        )

        candidate_tracks = []
        last_annotated = None
        edge_rejection_count = 0
        edge_direction_counts = {}
        safe_target_detection_count = 0
        invalid_depth_count = 0
        filtered_depth_recovery_count = 0
        depth_retry_extended = False
        total_frame_budget = max_frames + self.depth_retry_extra_frames

        for frame_index in range(total_frame_budget):
            # 未发现深度异常时维持原来的检测时长；只有目标类别和位置正确、
            # 但深度异常或波动过大时，才启用额外帧数并留在当前任务点重试。
            if frame_index >= max_frames and not depth_retry_extended:
                break
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
            H, W = color_image.shape[:2]
            margin_x = min(self.object_edge_margin_x, max((W - 2) // 2, 0))
            margin_y = min(self.object_edge_margin_y, max((H - 2) // 2, 0))
            draw_safe_detection_region(
                annotated,
                margin_x,
                margin_y,
                allow_top_edge=is_pick_context,
                allow_bottom_edge=is_pick_context,
            )
            depth_intrinsics = filtered_depth.profile.as_video_stream_profile().intrinsics

            results = self.model.predict(
                source=color_image,
                imgsz=IMG_SIZE,
                conf=predict_conf,
                verbose=False,
            )

            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                confs = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                for box, conf, class_id in zip(boxes, confs, class_ids):
                    model_class_label = str(self.model.names[int(class_id)])

                    # 先过滤掉与目标物体类型无关的检测框。
                    if model_class_label not in VALID_TARGET_LABELS[target_type]:
                        continue

                    x1, y1, x2, y2 = box.tolist()
                    x1 = max(0, min(x1, W - 1))
                    y1 = max(0, min(y1, H - 1))
                    x2 = max(0, min(x2, W - 1))
                    y2 = max(0, min(y2, H - 1))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    class_label = model_class_label
                    if target_label == "block-r":
                        red_ratio = get_red_ratio(color_image, x1, y1, x2, y2)

                        # 接受两类红方块候选：
                        # 1. 模型高置信度直接判断为 block-r；
                        # 2. 任意 block-* 低阈值候选，但检测框内红色占比足够。
                        model_strong_red = (
                            model_class_label == "block-r" and float(conf) >= CONF_THRES
                        )
                        hsv_confirmed_red = red_ratio >= RED_BLOCK_MIN_RATIO
                        if not (model_strong_red or hsv_confirmed_red):
                            continue
                        class_label = "block-r"
                    elif target_label is not None and model_class_label != target_label:
                        continue

                    # 抓取采用“中心软边距 + 检测框真实硬边界”：只保留左、右
                    # 限制；上下均不限制，目标框贴住画面上下边缘仍可继续抓取。
                    # 放置继续使用整个检测框位于软安全区内的保守规则。
                    if is_pick_context:
                        edge_directions = pick_box_unsafe_edges(
                            x1,
                            y1,
                            x2,
                            y2,
                            W,
                            H,
                            margin_x,
                            margin_y,
                            self.pick_hard_border_margin,
                        )
                        target_is_safe = not edge_directions
                    else:
                        target_is_safe = is_box_inside_safe_region(
                            x1, y1, x2, y2, W, H, margin_x, margin_y
                        )
                        edge_directions = box_outside_safe_edges(
                            x1, y1, x2, y2, W, H, margin_x, margin_y
                        )
                    if not target_is_safe:
                        edge_rejection_count += 1
                        edge_key = "+".join(edge_directions) or "unknown"
                        edge_direction_counts[edge_key] = (
                            edge_direction_counts.get(edge_key, 0) + 1
                        )
                        draw_edge_rejection(
                            annotated,
                            x1,
                            y1,
                            x2,
                            y2,
                            class_label,
                            float(conf),
                        )
                        if is_pick_context:
                            edge_rule_text = (
                                "中心边距 left/right=%d, top/bottom=不限制；"
                                "抓取框 left/right 硬边界=%d"
                                % (
                                    margin_x,
                                    self.pick_hard_border_margin,
                                )
                            )
                        else:
                            edge_rule_text = "整个框安全边距=(%d,%d)" % (
                                margin_x,
                                margin_y,
                            )
                        rospy.logwarn_throttle(
                            1.0,
                            "%s：拒绝画面边缘目标 %s conf=%.2f "
                            "box=(%d,%d,%d,%d)，%s，方向=%s"
                            % (
                                detection_context,
                                class_label,
                                float(conf),
                                x1,
                                y1,
                                x2,
                                y2,
                                edge_rule_text,
                                edge_key,
                            ),
                        )
                        continue

                    safe_target_detection_count += 1
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    raw_dist_m = get_depth_percentile(
                        filtered_depth, raw_depth, x1, y1, x2, y2, self.depth_scale
                    )
                    dist_m = raw_dist_m
                    depth_source = "raw"

                    if not is_depth_in_range(
                        raw_dist_m, task_depth_min, task_depth_max
                    ):
                        if not depth_retry_extended:
                            depth_retry_extended = True
                            self.publish_detection_status(
                                "%s:depth_retry:start:raw_depth=%.3f:range=%.3f-%.3f:extra_frames=%d"
                                % (
                                    detection_context,
                                    raw_dist_m,
                                    task_depth_min,
                                    task_depth_max,
                                    self.depth_retry_extra_frames,
                                )
                            )
                            rospy.logwarn(
                                "%s 检测到正确目标但原始深度 %.3f m 异常；"
                                "车辆和机械臂保持不动，额外重试最多%d帧",
                                detection_context,
                                raw_dist_m,
                                self.depth_retry_extra_frames,
                            )

                        recovered_depth, valid_ratio = get_filtered_depth_in_range(
                            filtered_depth,
                            x1,
                            y1,
                            x2,
                            y2,
                            self.depth_scale,
                            task_depth_min,
                            task_depth_max,
                            self.depth_fallback_min_valid_ratio,
                        )
                        if recovered_depth is None:
                            invalid_depth_count += 1
                            draw_depth_retry(
                                annotated,
                                x1,
                                y1,
                                x2,
                                y2,
                                class_label,
                                raw_dist_m,
                            )
                            rospy.logwarn_throttle(
                                1.0,
                                "%s 深度异常，正在原地重试：raw=%.3f m，"
                                "允许=%.3f~%.3f m，滤波有效像素=%.1f%%"
                                % (
                                    detection_context,
                                    raw_dist_m,
                                    task_depth_min,
                                    task_depth_max,
                                    valid_ratio * 100.0,
                                ),
                            )
                            continue

                        dist_m = recovered_depth
                        depth_source = "filtered_retry"
                        filtered_depth_recovery_count += 1

                    xyz_cam = deproject(depth_intrinsics, cx, cy, dist_m)
                    T_base_object = self.make_object_pose_in_base(xyz_cam)
                    xyz_base = T_base_object[0:3, 3]

                    draw_detection(
                        annotated, x1, y1, x2, y2,
                        class_label, float(conf), dist_m, xyz_cam
                    )

                    # 候选目标打分：优先置信度高，轻微偏好画面中心附近，避免误抓边缘目标
                    center_penalty = abs(cx - W / 2.0) / W + abs(cy - H / 2.0) / H
                    score = float(conf) - 0.10 * center_penalty
                    candidate = {
                        "label": class_label,
                        "conf": float(conf),
                        "score": score,
                        "center": (cx, cy),
                        "dist_m": float(dist_m),
                        "xyz_cam": np.array(xyz_cam, dtype=float),
                        "xyz_base": np.array(xyz_base, dtype=float),
                        "T_base_object": T_base_object,
                        "hits": 1,
                        "last_seen_frame": frame_index,
                        "depth_samples": [float(dist_m)],
                        "center_samples": [np.array((cx, cy), dtype=float)],
                        "xyz_cam_samples": [np.array(xyz_cam, dtype=float)],
                        "used_filtered_depth": depth_source == "filtered_retry",
                    }

                    # 将相邻帧中位置、深度接近的候选合并为同一目标轨迹。
                    matched_track = None
                    matched_pixel_distance = float("inf")
                    for track in candidate_tracks:
                        pixel_distance = float(np.linalg.norm(
                            np.array(track["center"], dtype=float)
                            - np.array(candidate["center"], dtype=float)
                        ))
                        depth_diff = abs(track["dist_m"] - candidate["dist_m"])
                        if (
                            track["label"] == candidate["label"]
                            and track["last_seen_frame"] < frame_index
                            and pixel_distance <= TRACK_MAX_PIXEL_DISTANCE
                            and depth_diff <= TRACK_MAX_DEPTH_DIFF
                            and pixel_distance < matched_pixel_distance
                        ):
                            matched_track = track
                            matched_pixel_distance = pixel_distance

                    if matched_track is None:
                        candidate_tracks.append(candidate)
                    else:
                        is_consecutive = (
                            matched_track["last_seen_frame"] == frame_index - 1
                        )
                        if not is_consecutive:
                            # 中间出现漏检、边缘框或异常深度后，必须重新累计连续
                            # 正常帧，不能沿用较早的命中次数直接触发机械臂动作。
                            matched_track.clear()
                            matched_track.update(candidate)
                        else:
                            matched_track["hits"] += 1
                            matched_track["last_seen_frame"] = frame_index
                            matched_track["depth_samples"].append(float(dist_m))
                            matched_track["center_samples"].append(
                                np.array((cx, cy), dtype=float)
                            )
                            matched_track["xyz_cam_samples"].append(
                                np.array(xyz_cam, dtype=float)
                            )

                            # 只保留最近的连续有效帧，使偶发异常退出窗口后可以
                            # 在同一任务点恢复，而不必立即前往下一个候选点。
                            sample_window = self.depth_stable_samples
                            matched_track["depth_samples"] = matched_track[
                                "depth_samples"
                            ][-sample_window:]
                            matched_track["center_samples"] = matched_track[
                                "center_samples"
                            ][-sample_window:]
                            matched_track["xyz_cam_samples"] = matched_track[
                                "xyz_cam_samples"
                            ][-sample_window:]

                            median_center = np.median(
                                np.stack(matched_track["center_samples"], axis=0),
                                axis=0,
                            )
                            median_xyz_cam = np.median(
                                np.stack(matched_track["xyz_cam_samples"], axis=0),
                                axis=0,
                            )
                            matched_track["center"] = tuple(
                                np.rint(median_center).astype(int)
                            )
                            matched_track["dist_m"] = float(
                                np.median(matched_track["depth_samples"])
                            )
                            matched_track["xyz_cam"] = median_xyz_cam
                            matched_track["T_base_object"] = (
                                self.make_object_pose_in_base(median_xyz_cam)
                            )
                            matched_track["xyz_base"] = matched_track[
                                "T_base_object"
                            ][0:3, 3]
                            matched_track["used_filtered_depth"] = (
                                matched_track["used_filtered_depth"]
                                or depth_source == "filtered_retry"
                            )
                            if candidate["score"] > matched_track["score"]:
                                matched_track["score"] = candidate["score"]
                                matched_track["conf"] = candidate["conf"]

            last_annotated = annotated
            self.publish_detection_image(annotated)
            if SHOW_DETECTION_WINDOW:
                try:
                    cv2.imshow("Object Detection For Grasp", annotated)
                    cv2.waitKey(1)
                except cv2.error:
                    pass

            current_tracks = [
                track
                for track in candidate_tracks
                if track["last_seen_frame"] == frame_index
                and len(track["depth_samples"]) >= self.depth_stable_samples
            ]
            unstable_depth_tracks = [
                track
                for track in current_tracks
                if max(track["depth_samples"]) - min(track["depth_samples"])
                > self.depth_stability_max_spread
            ]
            if unstable_depth_tracks:
                if not depth_retry_extended:
                    depth_retry_extended = True
                    self.publish_detection_status(
                        "%s:depth_retry:start:reason=unstable:extra_frames=%d"
                        % (detection_context, self.depth_retry_extra_frames)
                    )
                worst_spread = max(
                    max(track["depth_samples"]) - min(track["depth_samples"])
                    for track in unstable_depth_tracks
                )
                rospy.logwarn_throttle(
                    1.0,
                    "%s 连续深度波动 %.3f m，超过 %.3f m，保持原地继续采样"
                    % (
                        detection_context,
                        worst_spread,
                        self.depth_stability_max_spread,
                    ),
                )

            stable_tracks = [
                track
                for track in current_tracks
                if max(track["depth_samples"]) - min(track["depth_samples"])
                <= self.depth_stability_max_spread
            ]
            if stable_tracks and frame_index >= 10:
                c = max(stable_tracks, key=lambda track: (track["hits"], track["score"]))
                depth_spread = max(c["depth_samples"]) - min(c["depth_samples"])
                depth_source = (
                    "raw+filtered_retry" if c["used_filtered_depth"] else "raw"
                )
                self.publish_detection_status(
                    "%s:success:label=%s:conf=%.2f:hits=%d:depth=%.3f:"
                    "depth_samples=%d:spread=%.3f:source=%s"
                    % (
                        detection_context,
                        c["label"],
                        c["conf"],
                        c["hits"],
                        c["dist_m"],
                        len(c["depth_samples"]),
                        depth_spread,
                        depth_source,
                    )
                )
                rospy.loginfo("========================================")
                rospy.loginfo(
                    f"检测成功！目标类别: {c['label']}  conf={c['conf']:.2f}  "
                    f"连续命中={c['hits']}帧"
                )
                rospy.loginfo(
                    "深度确认: 样本=%d帧  中位数=%.4f m  跨度=%.4f m  来源=%s",
                    len(c["depth_samples"]),
                    c["dist_m"],
                    depth_spread,
                    depth_source,
                )
                rospy.loginfo(f"像素中心: {c['center']}  深度: {c['dist_m']:.4f} m")
                rospy.loginfo("相机坐标系下目标中心 XYZ:")
                rospy.loginfo(f"  X: {c['xyz_cam'][0]:.4f} m")
                rospy.loginfo(f"  Y: {c['xyz_cam'][1]:.4f} m")
                rospy.loginfo(f"  Z: {c['xyz_cam'][2]:.4f} m")
                rospy.loginfo("基座坐标系下目标中心 XYZ:")
                rospy.loginfo(f"  X: {c['xyz_base'][0]:.4f} m")
                rospy.loginfo(f"  Y: {c['xyz_base'][1]:.4f} m")
                rospy.loginfo(f"  Z: {c['xyz_base'][2]:.4f} m")
                grasp_x_base = c["T_base_object"][0:3, 0]
                rospy.loginfo("目标抓取姿态校验：夹爪 X 轴在基坐标系下应接近 [0, 0, -1]")
                rospy.loginfo(
                    f"  grasp_x_base: [{grasp_x_base[0]:.4f}, "
                    f"{grasp_x_base[1]:.4f}, {grasp_x_base[2]:.4f}]"
                )
                rospy.loginfo("========================================")
                self.last_object_detection_failure = None
                self.last_detected_object_label = c["label"]
                return c["T_base_object"]

        if last_annotated is not None and SHOW_DETECTION_WINDOW:
            try:
                cv2.imshow("Object Detection For Grasp", last_annotated)
                cv2.waitKey(1)
            except cv2.error:
                pass

        max_hits = max((track["hits"] for track in candidate_tracks), default=0)
        if edge_rejection_count > 0 and safe_target_detection_count == 0:
            edge_direction = max(
                edge_direction_counts,
                key=lambda key: edge_direction_counts[key],
            )
            self.last_object_detection_failure = "target_at_edge=%s" % (
                edge_direction
            )
        elif invalid_depth_count > 0:
            self.last_object_detection_failure = "invalid_depth"
        else:
            self.last_object_detection_failure = "not_found"

        self.publish_detection_status(
            "%s:failed:label=%s:reason=%s:max_hits=%d:edge_rejects=%d:"
            "invalid_depths=%d:filtered_recoveries=%d:depth_retry=%s"
            % (
                detection_context,
                label_text,
                self.last_object_detection_failure,
                max_hits,
                edge_rejection_count,
                invalid_depth_count,
                filtered_depth_recovery_count,
                str(depth_retry_extended).lower(),
            )
        )
        rospy.logwarn(
            f"超时：未稳定检测到目标: {label_text}，"
            f"失败类型={self.last_object_detection_failure}，"
            f"最大连续命中={max_hits}帧，边缘拒绝={edge_rejection_count}次，"
            f"无效深度={invalid_depth_count}次，"
            f"滤波恢复={filtered_depth_recovery_count}次"
        )
        return None

    def execute_vision_grasp(self):
        # 1. 先移动到拍照位，再从 RealSense 画面自动选择抓取目标。
        rospy.loginfo(">>> 移动到拍照位")
        joint_msg = JointState()
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.name = [""]
        joint_msg.position = [-1.530, 0.446, 0.0, 0.0, -0.115, 0.0, 0.0]
        joint_msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]
        joint_msg.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
        self.joint_pub.publish(joint_msg)
        time.sleep(8.0)

        selected_label = self.recognize_reference_target()
        target_type = self.target_type_from_label(selected_label)
        if target_type is None:
            return

        # 初始图片样式在此处锁定，抓取和放置全过程都使用这个值。
        # 后续即使画面中出现其他图片，也不允许重新选择放置样式。
        initial_image_style = selected_label

        time.sleep(10.0)

        target_label = initial_image_style
        target_text = "方块" if target_type == "block" else "瓶子"
        rospy.loginfo(
            f">>> 已根据相机图像选择抓取目标: "
            f"{LABEL_DESCRIPTION[target_label]}"
        )
        rospy.loginfo(
            f">>> 开始{target_text}实物视觉抓取任务，颜色类别: {target_label}"
        )
        # T_pre[0:3, 3] = [+x（下）, +y（左）, +z（前）]

        # 2. 第一次识别目标中心位置
        T_base_object = self.get_object_pose(target_type, target_label)
        if T_base_object is None:
            return

        # 3. 移动到预抓取位
        # 固定抓取坐标系方向：
        #   +x = 基坐标系 -Z 方向，也就是向下
        #   -x = 基坐标系 +Z 方向，也就是向上
        #   +y = 基坐标系 +Y 方向
        #   +z = 基坐标系 +X 方向
        rospy.loginfo(">>> 移动到预抓取位")
        T_pre = T_OBJECT_TO_GRASP.copy()
        T_pre[0:3, 3] = [-0.040, -0.02, -0.05]
        target_ee_pre = T_base_object @ T_pre @ T_EE_TO_TOOL
        self.move_to_target_smooth(target_ee_pre, v=0.06, gripper_val=100)
        time.sleep(3.5)


        # 4a. 慢速靠近目标
        rospy.loginfo(">>> 前探夹爪")
        T_t2 = T_OBJECT_TO_GRASP.copy()
        T_t2[0:3, 3] = [-0.040, -0.02, 0.06]
        self.move_to_target_smooth(T_base_object @ T_t2 @ T_EE_TO_TOOL, v=0.02, gripper_val=100)
        time.sleep(2.5)

        # 4b. 闭合夹爪
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

        # 4c. 提升
        rospy.loginfo(f">>> 提起{target_text}")
        T_up = T_OBJECT_TO_GRASP.copy()
        T_up[0:3, 3] = [-0.15, 0.0, 0.06]
        self.move_to_target_smooth(T_base_object @ T_up @ T_EE_TO_TOOL, v=0.05, gripper_val=0)
        time.sleep(3.0)
        
        # 5. 回到初始姿态
        rospy.loginfo(">>> 移动到初始位")
        joint_msg = JointState()
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.name = [""]
        joint_msg.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        joint_msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]
        joint_msg.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
        self.joint_pub.publish(joint_msg)
        time.sleep(8.0)

        rospy.loginfo(">>> 视觉抓取任务结束")
        
        time.sleep(2.0)
        
        # 放置
        # 1. 移动到拍照位
        rospy.loginfo(">>> 放置任务开始")
        rospy.loginfo(">>> 移动到初始位")
        joint_msg = JointState()
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.name = [""]
        joint_msg.position = [-1.602, 0.641, -0.509, 0.0, 0.324, 0.0, 0.0]
        joint_msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]
        joint_msg.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
        self.joint_pub.publish(joint_msg)
        time.sleep(8.0)
        
        # 2. 在放置区定位与任务开始时识别到的样式完全一致的图片。
        T_base_place_image = self.get_place_pose_by_image_style(initial_image_style)
        if T_base_place_image is None:
            rospy.logerr(
                f"未找到与最初识别样式一致的放置图片: {initial_image_style}，"
                "取消松开夹爪"
            )
            return
        
        # 3. 移动到图片前（盒子上方）
        rospy.loginfo(">>> 移动到图片前")
        T_pre = T_OBJECT_TO_GRASP.copy()
        T_pre[0:3, 3] = [-0.12, -0.04, -0.06]
        target_ee_pre = T_base_place_image @ T_pre @ T_EE_TO_TOOL
        self.move_to_target_smooth(target_ee_pre, v=0.06, gripper_val=0)
        time.sleep(3.5)
        
        # 4. 松开夹爪
        rospy.loginfo(">>> 松开夹爪")
        curr_pose = self.current_ee_pose_mat.copy()
        r, p, yaw = tf_trans.euler_from_matrix(curr_pose, axes="sxyz")
        cmd_close = PosCmd()
        cmd_close.x, cmd_close.y, cmd_close.z = curr_pose[0:3, 3]
        cmd_close.roll, cmd_close.pitch, cmd_close.yaw = r, p, yaw
        cmd_close.gripper = 200
        cmd_close.mode1 = 1
        cmd_close.mode2 = 0
        self.pub.publish(cmd_close)
        time.sleep(2.0)


        # 6. 归位
        joint_msg = JointState()
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.name = [""]
        joint_msg.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        joint_msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]
        joint_msg.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
        self.joint_pub.publish(joint_msg)
        time.sleep(8.0)
        
        rospy.loginfo(">>> 放置完成")

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
