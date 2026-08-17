#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""独立的放置卡片平面深度测试节点。

启动后先把机械臂移动到正式流程使用的放置观察位，并在当前位置张开夹爪；
确认机械臂到位后才读取 D435i 和运行 YOLO。放置深度不从瓶子/方块图案内部
提取，而是从检测框四周的卡片区域寻找一致平面，随后通过三维 RANSAC 平面
拟合估计检测框中心深度。本节点不会控制车辆，也不会执行伸臂或放置动作。

示例：
  rosrun piper_task place_plane_depth_test.py _target_label:=bottle-b
"""

import os
import time
from collections import deque

import cv2
import numpy as np
import pyrealsense2 as rs
import rospy
import tf.transformations as tf_trans
from geometry_msgs.msg import PoseStamped
from piper_msgs.msg import PosCmd
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, String
from ultralytics import YOLO

from piper_task.vision_grasp_core import MODEL_PATH as DEFAULT_MODEL_PATH

COLOR_WIDTH = 640
COLOR_HEIGHT = 480
DEPTH_WIDTH = 640
DEPTH_HEIGHT = 480
FPS = 30

# 与正式比赛节点 piper_task_node.py 使用的放置观察位保持一致。这里保留独立
# 副本，避免测试节点依赖或修改正式任务代码。
DEFAULT_PLACE_SCAN_JOINTS = [-1.602, 0.641, -0.509, 0.0, 0.324, 0.0, 0.0]
DEFAULT_OPEN_GRIPPER_VALUE = 200.0

SUPPORTED_LABELS = {
    "block-r",
    "block-y",
    "block-b",
    "bottle-b",
    "bottle-y",
    "bottle-g",
}

LABEL_COLORS = {
    "block-r": (0, 0, 255),
    "block-y": (0, 255, 255),
    "block-b": (255, 0, 0),
    "bottle-b": (80, 80, 80),
    "bottle-y": (0, 140, 255),
    "bottle-g": (0, 200, 0),
}

PATCH_COLORS = {
    "top": (255, 255, 0),
    "bottom": (255, 0, 255),
    "left": (0, 255, 255),
    "right": (0, 165, 255),
}


def build_depth_filters():
    """创建与正式节点相同类型的 D4xx 深度滤波链。"""
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


def apply_depth_filters(depth_frame, filters):
    for depth_filter in filters:
        depth_frame = depth_filter.process(depth_frame)
    return depth_frame.as_depth_frame()


def clip_rect(rect, width, height):
    x1, y1, x2, y2 = [int(value) for value in rect]
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    return x1, y1, x2, y2


def rect_area(rect):
    x1, y1, x2, y2 = rect
    return max(0, x2 - x1) * max(0, y2 - y1)


def make_ring_patches(
    box,
    width,
    height,
    pad_ratio_x,
    pad_ratio_y,
    min_pad_x,
    min_pad_y,
    max_pad_x,
    max_pad_y,
    gap,
):
    """在 YOLO 框外生成上、下、左、右四块卡片平面候选区域。"""
    x1, y1, x2, y2 = clip_rect(box, width, height)
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    pad_x = int(np.clip(round(box_width * pad_ratio_x), min_pad_x, max_pad_x))
    pad_y = int(np.clip(round(box_height * pad_ratio_y), min_pad_y, max_pad_y))
    gap = max(0, int(gap))

    patches = {
        "top": clip_rect((x1, y1 - pad_y, x2, y1 - gap), width, height),
        "bottom": clip_rect((x1, y2 + gap, x2, y2 + pad_y), width, height),
        "left": clip_rect((x1 - pad_x, y1, x1 - gap, y2), width, height),
        "right": clip_rect((x2 + gap, y1, x2 + pad_x, y2), width, height),
    }
    return {name: rect for name, rect in patches.items() if rect_area(rect) > 0}


def calculate_patch_stats(
    depth_m,
    rect,
    depth_min,
    depth_max,
    min_valid_ratio,
    min_valid_pixels,
    max_mad,
):
    """计算单个邻域的任务范围深度支持率和鲁棒统计量。"""
    x1, y1, x2, y2 = rect
    roi = depth_m[y1:y2, x1:x2]
    if roi.size == 0:
        return {
            "rect": rect,
            "valid": False,
            "valid_count": 0,
            "valid_ratio": 0.0,
            "median": None,
            "mad": None,
            "raw_median": None,
        }

    finite_sensor = roi[np.isfinite(roi) & (roi > 0.10) & (roi < 8.0)]
    task_mask = np.isfinite(roi) & (roi > depth_min) & (roi < depth_max)
    task_values = roi[task_mask]
    valid_count = int(task_values.size)
    valid_ratio = float(valid_count) / float(roi.size)
    raw_median = (
        float(np.median(finite_sensor)) if finite_sensor.size > 0 else None
    )

    if valid_count == 0:
        return {
            "rect": rect,
            "valid": False,
            "valid_count": 0,
            "valid_ratio": valid_ratio,
            "median": None,
            "mad": None,
            "raw_median": raw_median,
        }

    median = float(np.median(task_values))
    mad = float(np.median(np.abs(task_values - median)))
    valid = bool(
        valid_count >= min_valid_pixels
        and valid_ratio >= min_valid_ratio
        and mad <= max_mad
    )
    return {
        "rect": rect,
        "valid": valid,
        "valid_count": valid_count,
        "valid_ratio": valid_ratio,
        "median": median,
        "mad": mad,
        "raw_median": raw_median,
        "task_mask": task_mask,
    }


def largest_depth_consensus(patch_stats, tolerance):
    """从可靠邻域中寻找深度最一致的一组，避免前景或背景单区主导。"""
    reliable = [
        (name, stats)
        for name, stats in patch_stats.items()
        if stats["valid"] and stats["median"] is not None
    ]
    best_group = []
    best_span = float("inf")
    for _, seed_stats in reliable:
        seed = seed_stats["median"]
        group = [
            (name, stats)
            for name, stats in reliable
            if abs(stats["median"] - seed) <= tolerance
        ]
        medians = [stats["median"] for _, stats in group]
        span = max(medians) - min(medians) if medians else float("inf")
        if len(group) > len(best_group) or (
            len(group) == len(best_group) and span < best_span
        ):
            best_group = group
            best_span = span
    return [name for name, _ in best_group], best_span


def collect_patch_points(
    depth_m,
    patch_stats,
    selected_names,
    max_points,
):
    """均衡地从每个一致邻域采样，防止一个大区域压过其他区域。"""
    if not selected_names:
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=object)

    per_patch_limit = max(20, int(max_points) // len(selected_names))
    samples = []
    sample_patch_names = []
    for name in selected_names:
        stats = patch_stats[name]
        x1, y1, _, _ = stats["rect"]
        ys, xs = np.nonzero(stats["task_mask"])
        if len(xs) > per_patch_limit:
            indices = np.linspace(0, len(xs) - 1, per_patch_limit).astype(int)
            xs = xs[indices]
            ys = ys[indices]
        image_x = xs + x1
        image_y = ys + y1
        values = depth_m[image_y, image_x]
        samples.append(
            np.column_stack(
                (
                    image_x.astype(np.float64),
                    image_y.astype(np.float64),
                    values.astype(np.float64),
                )
            )
        )
        sample_patch_names.extend([name] * len(image_x))

    if not samples:
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=object)
    return np.vstack(samples), np.asarray(sample_patch_names, dtype=object)


def pixels_to_camera_points(pixel_depth_samples, intrinsics):
    u = pixel_depth_samples[:, 0]
    v = pixel_depth_samples[:, 1]
    z = pixel_depth_samples[:, 2]
    x = (u - float(intrinsics.ppx)) * z / float(intrinsics.fx)
    y = (v - float(intrinsics.ppy)) * z / float(intrinsics.fy)
    return np.column_stack((x, y, z))


def ray_for_pixel(pixel, intrinsics):
    u, v = pixel
    return np.array(
        [
            (float(u) - float(intrinsics.ppx)) / float(intrinsics.fx),
            (float(v) - float(intrinsics.ppy)) / float(intrinsics.fy),
            1.0,
        ],
        dtype=np.float64,
    )


def depth_on_plane(normal, offset, pixel, intrinsics):
    ray = ray_for_pixel(pixel, intrinsics)
    denominator = float(np.dot(normal, ray))
    if abs(denominator) < 1e-8:
        return None
    depth = -float(offset) / denominator
    if not np.isfinite(depth):
        return None
    return float(depth)


def fit_plane_ransac(
    pixel_depth_samples,
    sample_patch_names,
    intrinsics,
    center_pixel,
    iterations,
    inlier_threshold,
    min_inliers,
    min_inlier_ratio,
    min_patch_inliers,
    max_rmse,
    min_abs_normal_z,
    random_state,
):
    """在相机三维坐标中拟合平面，并计算中心像素与平面的交点深度。"""
    if len(pixel_depth_samples) < max(3, min_inliers):
        return None, "too_few_plane_points"

    points = pixels_to_camera_points(pixel_depth_samples, intrinsics)
    best_inliers = None
    for _ in range(max(1, int(iterations))):
        indices = random_state.choice(len(points), size=3, replace=False)
        p1, p2, p3 = points[indices]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-8:
            continue
        normal /= norm
        offset = -float(np.dot(normal, p1))
        distances = np.abs(points @ normal + offset)
        inliers = distances <= inlier_threshold
        if best_inliers is None or int(np.count_nonzero(inliers)) > int(
            np.count_nonzero(best_inliers)
        ):
            best_inliers = inliers

    if best_inliers is None:
        return None, "plane_ransac_failed"

    inlier_count = int(np.count_nonzero(best_inliers))
    inlier_ratio = float(inlier_count) / float(len(points))
    if inlier_count < min_inliers:
        return None, "plane_inliers=%d<%d" % (inlier_count, min_inliers)
    if inlier_ratio < min_inlier_ratio:
        return None, "plane_inlier_ratio=%.2f<%.2f" % (
            inlier_ratio,
            min_inlier_ratio,
        )

    inlier_points = points[best_inliers]
    centroid = np.mean(inlier_points, axis=0)
    _, _, vh = np.linalg.svd(inlier_points - centroid, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    offset = -float(np.dot(normal, centroid))

    residuals = np.abs(points @ normal + offset)
    refined_inliers = residuals <= inlier_threshold
    inlier_count = int(np.count_nonzero(refined_inliers))
    inlier_ratio = float(inlier_count) / float(len(points))
    if inlier_count < min_inliers or inlier_ratio < min_inlier_ratio:
        return None, "plane_refine_support_lost"

    patch_inliers = {}
    for name in np.unique(sample_patch_names):
        patch_mask = sample_patch_names == name
        patch_inliers[str(name)] = int(
            np.count_nonzero(refined_inliers & patch_mask)
        )
    weak_patches = [
        name for name, count in patch_inliers.items() if count < min_patch_inliers
    ]
    if weak_patches:
        return None, "weak_plane_patches=%s" % "+".join(sorted(weak_patches))

    rmse = float(np.sqrt(np.mean(np.square(residuals[refined_inliers]))))
    if rmse > max_rmse:
        return None, "plane_rmse=%.4f>%.4f" % (rmse, max_rmse)
    if abs(float(normal[2])) < min_abs_normal_z:
        return None, "plane_too_oblique=nz%.2f" % abs(float(normal[2]))

    center_depth = depth_on_plane(normal, offset, center_pixel, intrinsics)
    if center_depth is None:
        return None, "center_ray_parallel_to_plane"

    return {
        "normal": normal,
        "offset": offset,
        "center_depth": center_depth,
        "inlier_count": inlier_count,
        "point_count": len(points),
        "inlier_ratio": inlier_ratio,
        "rmse": rmse,
        "patch_inliers": patch_inliers,
    }, None


class PlacePlaneEstimator:
    def __init__(self):
        self.depth_min = float(rospy.get_param("~depth_min", 0.30))
        self.depth_max = float(rospy.get_param("~depth_max", 0.70))
        self.pad_ratio_x = float(rospy.get_param("~ring_pad_ratio_x", 0.35))
        self.pad_ratio_y = float(rospy.get_param("~ring_pad_ratio_y", 0.25))
        self.min_pad_x = int(rospy.get_param("~ring_min_pad_x", 12))
        self.min_pad_y = int(rospy.get_param("~ring_min_pad_y", 10))
        self.max_pad_x = int(rospy.get_param("~ring_max_pad_x", 55))
        self.max_pad_y = int(rospy.get_param("~ring_max_pad_y", 45))
        self.ring_gap = int(rospy.get_param("~ring_gap", 2))

        self.min_patch_valid_ratio = float(
            rospy.get_param("~min_patch_valid_ratio", 0.08)
        )
        self.min_patch_valid_pixels = int(
            rospy.get_param("~min_patch_valid_pixels", 30)
        )
        self.max_patch_mad = float(rospy.get_param("~max_patch_mad", 0.025))
        self.patch_consistency = float(
            rospy.get_param("~patch_consistency", 0.040)
        )
        self.min_consistent_patches = int(
            rospy.get_param("~min_consistent_patches", 3)
        )
        self.patch_plane_consistency = float(
            rospy.get_param("~patch_plane_consistency", 0.035)
        )

        self.max_plane_points = int(rospy.get_param("~max_plane_points", 2400))
        self.ransac_iterations = int(rospy.get_param("~ransac_iterations", 100))
        self.plane_inlier_threshold = float(
            rospy.get_param("~plane_inlier_threshold", 0.010)
        )
        self.min_plane_inliers = int(rospy.get_param("~min_plane_inliers", 90))
        self.min_plane_inlier_ratio = float(
            rospy.get_param("~min_plane_inlier_ratio", 0.55)
        )
        self.min_patch_plane_inliers = int(
            rospy.get_param("~min_patch_plane_inliers", 15)
        )
        self.max_plane_rmse = float(rospy.get_param("~max_plane_rmse", 0.010))
        self.min_abs_normal_z = float(
            rospy.get_param("~min_abs_plane_normal_z", 0.45)
        )
        self.random_state = np.random.RandomState(20260808)

        if not 0.10 < self.depth_min < self.depth_max < 8.0:
            raise ValueError("深度范围配置无效")
        if not 1 <= self.min_consistent_patches <= 4:
            raise ValueError("min_consistent_patches 必须为1到4")

    def estimate(self, filtered_depth_m, box, intrinsics):
        height, width = filtered_depth_m.shape[:2]
        patches = make_ring_patches(
            box,
            width,
            height,
            self.pad_ratio_x,
            self.pad_ratio_y,
            self.min_pad_x,
            self.min_pad_y,
            self.max_pad_x,
            self.max_pad_y,
            self.ring_gap,
        )
        patch_stats = {
            name: calculate_patch_stats(
                filtered_depth_m,
                rect,
                self.depth_min,
                self.depth_max,
                self.min_patch_valid_ratio,
                self.min_patch_valid_pixels,
                self.max_patch_mad,
            )
            for name, rect in patches.items()
        }
        selected_names, consensus_span = largest_depth_consensus(
            patch_stats, self.patch_consistency
        )
        base_result = {
            "patches": patches,
            "patch_stats": patch_stats,
            "selected_patches": selected_names,
            "consensus_span": consensus_span,
        }
        if len(selected_names) < self.min_consistent_patches:
            base_result["ok"] = False
            base_result["reason"] = "patch_consensus=%d/%d" % (
                len(selected_names),
                self.min_consistent_patches,
            )
            return base_result

        samples, sample_names = collect_patch_points(
            filtered_depth_m,
            patch_stats,
            selected_names,
            self.max_plane_points,
        )
        x1, y1, x2, y2 = box
        center = ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)
        plane, reason = fit_plane_ransac(
            samples,
            sample_names,
            intrinsics,
            center,
            self.ransac_iterations,
            self.plane_inlier_threshold,
            self.min_plane_inliers,
            self.min_plane_inlier_ratio,
            self.min_patch_plane_inliers,
            self.max_plane_rmse,
            self.min_abs_normal_z,
            self.random_state,
        )
        if plane is None:
            base_result["ok"] = False
            base_result["reason"] = reason
            return base_result

        center_depth = float(plane["center_depth"])
        if not self.depth_min < center_depth < self.depth_max:
            base_result["ok"] = False
            base_result["reason"] = "plane_depth=%.3f_out_of_range" % center_depth
            base_result["plane"] = plane
            return base_result

        patch_plane_errors = {}
        for name in selected_names:
            px1, py1, px2, py2 = patches[name]
            patch_center = ((px1 + px2) / 2.0, (py1 + py2) / 2.0)
            predicted_depth = depth_on_plane(
                plane["normal"], plane["offset"], patch_center, intrinsics
            )
            if predicted_depth is None:
                base_result["ok"] = False
                base_result["reason"] = "patch_ray_parallel=%s" % name
                base_result["plane"] = plane
                return base_result
            error = abs(predicted_depth - patch_stats[name]["median"])
            patch_plane_errors[name] = error
        worst_patch_error = max(patch_plane_errors.values())
        if worst_patch_error > self.patch_plane_consistency:
            base_result["ok"] = False
            base_result["reason"] = "patch_plane_error=%.3f" % worst_patch_error
            base_result["plane"] = plane
            base_result["patch_plane_errors"] = patch_plane_errors
            return base_result

        base_result.update(
            {
                "ok": True,
                "reason": "ok",
                "depth": center_depth,
                "center": center,
                "plane": plane,
                "patch_plane_errors": patch_plane_errors,
            }
        )
        return base_result


class StablePlaneDepth:
    """要求平面深度和YOLO中心连续多帧稳定。"""

    def __init__(self, sample_count, max_depth_spread, max_center_spread):
        self.sample_count = max(3, int(sample_count))
        self.max_depth_spread = float(max_depth_spread)
        self.max_center_spread = float(max_center_spread)
        self.samples = deque(maxlen=self.sample_count)

    def clear(self):
        self.samples.clear()

    def add(self, depth, center):
        self.samples.append((float(depth), float(center[0]), float(center[1])))
        if len(self.samples) < self.sample_count:
            return None
        values = np.asarray(self.samples, dtype=np.float64)
        depth_spread = float(np.ptp(values[:, 0]))
        center_spread_x = float(np.ptp(values[:, 1]))
        center_spread_y = float(np.ptp(values[:, 2]))
        if depth_spread > self.max_depth_spread:
            return None
        if max(center_spread_x, center_spread_y) > self.max_center_spread:
            return None
        return {
            "depth": float(np.median(values[:, 0])),
            "center": (
                float(np.median(values[:, 1])),
                float(np.median(values[:, 2])),
            ),
            "depth_spread": depth_spread,
            "center_spread": max(center_spread_x, center_spread_y),
            "samples": len(values),
        }


def box_inside_safe_region(box, width, height, margin_x, margin_y):
    x1, y1, x2, y2 = box
    return bool(
        x1 >= margin_x
        and y1 >= margin_y
        and x2 <= width - 1 - margin_x
        and y2 <= height - 1 - margin_y
    )


def select_target_detection(
    results,
    model_names,
    target_label,
    width,
    height,
    margin_x,
    margin_y,
):
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None, 0
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    confidences = results[0].boxes.conf.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    candidates = []
    edge_rejects = 0
    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        label = str(model_names[int(class_id)])
        if target_label != "auto" and label != target_label:
            continue
        if target_label == "auto" and label not in SUPPORTED_LABELS:
            continue
        x1, y1, x2, y2 = clip_rect(box, width, height)
        if x2 <= x1 or y2 <= y1:
            continue
        if not box_inside_safe_region(
            (x1, y1, x2, y2), width, height, margin_x, margin_y
        ):
            edge_rejects += 1
            continue
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        center_penalty = abs(center_x - width / 2.0) / width + abs(
            center_y - height / 2.0
        ) / height
        area_ratio = float((x2 - x1) * (y2 - y1)) / float(width * height)
        score = float(confidence) + 0.10 * area_ratio - 0.08 * center_penalty
        candidates.append(
            {
                "box": (x1, y1, x2, y2),
                "confidence": float(confidence),
                "label": label,
                "score": score,
            }
        )
    if not candidates:
        return None, edge_rejects
    return max(candidates, key=lambda item: item["score"]), edge_rejects


def draw_text_lines(image, lines, origin=(12, 24), color=(255, 255, 255)):
    x, y = origin
    for line in lines:
        cv2.putText(
            image,
            str(line),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(line),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 23


def draw_estimator_result(image, detection, estimate, stable, stable_count):
    x1, y1, x2, y2 = detection["box"]
    label = detection["label"]
    label_color = LABEL_COLORS.get(label, (255, 255, 255))
    cv2.rectangle(image, (x1, y1), (x2, y2), label_color, 2)
    cv2.putText(
        image,
        "%s %.2f" % (label, detection["confidence"]),
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        label_color,
        2,
        cv2.LINE_AA,
    )

    selected = set(estimate.get("selected_patches", []))
    for name, rect in estimate.get("patches", {}).items():
        px1, py1, px2, py2 = rect
        stats = estimate["patch_stats"][name]
        base_color = PATCH_COLORS[name]
        if name in selected:
            color = (0, 255, 0)
            thickness = 2
        elif stats["valid"]:
            color = base_color
            thickness = 1
        else:
            color = (0, 0, 255)
            thickness = 1
        cv2.rectangle(image, (px1, py1), (px2, py2), color, thickness)
        median_text = "--" if stats["median"] is None else "%.3f" % stats["median"]
        cv2.putText(
            image,
            "%s %s %.0f%%" % (
                name[0].upper(),
                median_text,
                100.0 * stats["valid_ratio"],
            ),
            (px1, max(14, py1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    lines = []
    if estimate.get("ok"):
        plane = estimate["plane"]
        lines.append("PLANE %.3f m" % estimate["depth"])
        lines.append(
            "patches=%s inliers=%d/%d rmse=%.1fmm"
            % (
                "+".join(estimate["selected_patches"]),
                plane["inlier_count"],
                plane["point_count"],
                plane["rmse"] * 1000.0,
            )
        )
        if stable is None:
            lines.append("WAIT STABLE %d" % stable_count)
            status_color = (0, 255, 255)
        else:
            lines.append(
                "PASS %.3f m spread=%.1fmm"
                % (stable["depth"], stable["depth_spread"] * 1000.0)
            )
            status_color = (0, 255, 0)
    else:
        lines.append("REJECT %s" % estimate.get("reason", "unknown"))
        lines.append("patches=%s" % "+".join(estimate.get("selected_patches", [])))
        status_color = (0, 0, 255)
    draw_text_lines(image, lines, color=status_color)


def make_depth_colormap(depth_m, display_min, display_max):
    valid = np.isfinite(depth_m) & (depth_m > display_min) & (depth_m < display_max)
    normalized = np.zeros(depth_m.shape, dtype=np.uint8)
    if np.any(valid):
        clipped = np.clip(depth_m, display_min, display_max)
        normalized[valid] = np.asarray(
            255.0
            * (display_max - clipped[valid])
            / max(display_max - display_min, 1e-6),
            dtype=np.uint8,
        )
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def patch_summary(estimate):
    entries = []
    for name in ("top", "bottom", "left", "right"):
        stats = estimate.get("patch_stats", {}).get(name)
        if stats is None:
            continue
        median = "--" if stats["median"] is None else "%.3f" % stats["median"]
        entries.append(
            "%s=%s/%.0f%%" % (name, median, stats["valid_ratio"] * 100.0)
        )
    return " ".join(entries)


class PlacePlaneDepthTestNode:
    def __init__(self):
        self.model_path = str(rospy.get_param("~model_path", DEFAULT_MODEL_PATH))
        self.target_label = str(rospy.get_param("~target_label", "bottle-b"))
        self.conf_threshold = float(rospy.get_param("~conf_threshold", 0.50))
        self.image_size = int(rospy.get_param("~image_size", 640))
        self.margin_x = max(0, int(rospy.get_param("~safe_margin_x", 40)))
        self.margin_y = max(0, int(rospy.get_param("~safe_margin_y", 30)))
        self.show_window = bool(rospy.get_param("~show_window", True))
        self.device_serial = str(rospy.get_param("~device_serial", "")).strip()
        self.warmup_frames = max(0, int(rospy.get_param("~warmup_frames", 30)))
        self.save_dir = os.path.expanduser(
            str(rospy.get_param("~save_dir", "~/.ros/place_plane_depth_test"))
        )
        self.display_depth_min = float(rospy.get_param("~display_depth_min", 0.20))
        self.display_depth_max = float(rospy.get_param("~display_depth_max", 4.00))

        self.move_arm_to_observation = bool(
            rospy.get_param("~move_arm_to_observation", True)
        )
        self.open_gripper_at_observation = bool(
            rospy.get_param("~open_gripper_at_observation", True)
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
            rospy.get_param("~gripper_open_value", DEFAULT_OPEN_GRIPPER_VALUE)
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
        place_scan_joints = rospy.get_param(
            "~place_scan_joints", list(DEFAULT_PLACE_SCAN_JOINTS)
        )
        if not isinstance(place_scan_joints, (list, tuple)) or len(place_scan_joints) != 7:
            raise ValueError("place_scan_joints 必须是含7个数值的关节姿态")
        self.place_scan_joints = [float(value) for value in place_scan_joints]

        if self.target_label != "auto" and self.target_label not in SUPPORTED_LABELS:
            raise ValueError(
                "target_label=%s 不受支持，可选: %s 或 auto"
                % (self.target_label, sorted(SUPPORTED_LABELS))
            )
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("未找到YOLO模型: %s" % self.model_path)

        self.estimator = PlacePlaneEstimator()
        self.stability = StablePlaneDepth(
            rospy.get_param("~stable_samples", 8),
            rospy.get_param("~stable_max_depth_spread", 0.020),
            rospy.get_param("~stable_max_center_spread", 15.0),
        )
        self.status_pub = rospy.Publisher("~status", String, queue_size=10)
        self.depth_pub = rospy.Publisher("~depth", Float32, queue_size=10)

        self.current_ee_pose_mat = np.eye(4)
        self.pose_received_at = None
        self.joint_positions = None
        self.joint_received_at = None
        self.joint_pub = rospy.Publisher(
            self.joint_command_topic, JointState, queue_size=1
        )
        self.cartesian_pub = rospy.Publisher(
            self.cartesian_command_topic, PosCmd, queue_size=1
        )
        self.pose_sub = rospy.Subscriber(
            self.end_pose_topic, PoseStamped, self.pose_callback, queue_size=10
        )
        self.joint_sub = rospy.Subscriber(
            self.joint_feedback_topic,
            JointState,
            self.joint_feedback_callback,
            queue_size=10,
        )

        rospy.loginfo("加载独立平面深度测试模型: %s", self.model_path)
        self.model = YOLO(self.model_path)
        rospy.loginfo("模型类别: %s", self.model.names)

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
        try:
            self.profile = self.pipeline.start(config)
        except RuntimeError as exc:
            raise RuntimeError(
                "无法启动D435i。请先停止会占用相机的原piper_task节点: %s" % exc
            )

        device = self.profile.get_device()
        try:
            device_name = device.get_info(rs.camera_info.name)
            serial = device.get_info(rs.camera_info.serial_number)
        except RuntimeError:
            device_name = "RealSense"
            serial = "unknown"
        depth_sensor = device.first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())
        self.align = rs.align(rs.stream.color)
        self.filters = build_depth_filters()
        self.last_composite = None

        rospy.loginfo(
            "独立放置平面深度测试已启动: device=%s serial=%s scale=%.6f",
            device_name,
            serial,
            self.depth_scale,
        )
        rospy.loginfo(
            "target=%s range=%.3f~%.3fm patches>=%d stable=%d帧",
            self.target_label,
            self.estimator.depth_min,
            self.estimator.depth_max,
            self.estimator.min_consistent_patches,
            self.stability.sample_count,
        )
        rospy.logwarn(
            "本节点启动时只执行一次放置观察位和张开夹爪；"
            "不会控制车辆，也不会执行伸臂或放置动作。"
        )

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

    def joint_feedback_callback(self, msg):
        if len(msg.position) < 6:
            return
        self.joint_positions = tuple(float(value) for value in msg.position[:6])
        self.joint_received_at = time.monotonic()

    def feedback_is_fresh(self, received_at):
        return (
            received_at is not None
            and time.monotonic() - received_at <= self.arm_feedback_max_age
        )

    def wait_for_arm_interfaces(self):
        """确认控制订阅者和实时反馈均已就绪，避免盲发机械臂命令。"""
        need_joint = self.move_arm_to_observation
        need_cartesian = self.open_gripper_at_observation
        deadline = time.monotonic() + self.arm_feedback_timeout
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            joint_command_ready = (
                not need_joint or self.joint_pub.get_num_connections() > 0
            )
            joint_feedback_ready = (
                not need_joint
                or (
                    self.joint_positions is not None
                    and self.feedback_is_fresh(self.joint_received_at)
                )
            )
            cartesian_ready = (
                not need_cartesian or self.cartesian_pub.get_num_connections() > 0
            )
            pose_ready = (
                not need_cartesian or self.feedback_is_fresh(self.pose_received_at)
            )
            if (
                joint_command_ready
                and joint_feedback_ready
                and cartesian_ready
                and pose_ready
            ):
                return True
            rate.sleep()

        missing = []
        if need_joint and self.joint_pub.get_num_connections() == 0:
            missing.append("%s无订阅者" % self.joint_command_topic)
        if need_joint and (
            self.joint_positions is None
            or not self.feedback_is_fresh(self.joint_received_at)
        ):
            missing.append("%s无新鲜反馈" % self.joint_feedback_topic)
        if need_cartesian and self.cartesian_pub.get_num_connections() == 0:
            missing.append("%s无订阅者" % self.cartesian_command_topic)
        if need_cartesian and not self.feedback_is_fresh(self.pose_received_at):
            missing.append("%s无新鲜反馈" % self.end_pose_topic)
        raise RuntimeError(
            "机械臂接口未就绪（%s）。请先启动Piper底层驱动。"
            % ("、".join(missing) if missing else "等待超时")
        )

    def publish_observation_pose(self):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = [""]
        msg.position = list(self.place_scan_joints)
        msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]
        msg.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
        self.joint_pub.publish(msg)

    def wait_for_observation_pose(self):
        target = tuple(self.place_scan_joints[:6])
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
                        return True
                else:
                    stable = 0
            rate.sleep()

        error_text = (
            "无有效反馈" if not np.isfinite(last_error) else "%.4frad" % last_error
        )
        raise RuntimeError(
            "机械臂未确认到达放置观察位：最大误差=%s，要求<=%.3frad连续%d帧"
            % (error_text, self.arm_joint_tolerance, self.arm_stable_samples)
        )

    def publish_open_gripper(self):
        if not self.feedback_is_fresh(self.pose_received_at):
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

    def prepare_arm_for_test(self):
        """完成观察姿态和开夹爪后，才允许进入视觉测试循环。"""
        if not self.move_arm_to_observation and not self.open_gripper_at_observation:
            rospy.logwarn("已通过参数关闭机械臂准备动作，直接开始视觉测试。")
            return

        self.status_pub.publish(String(data="ARM_PREP:waiting_interfaces"))
        rospy.loginfo("等待Piper控制接口及实时反馈...")
        self.wait_for_arm_interfaces()

        if self.move_arm_to_observation:
            self.status_pub.publish(String(data="ARM_PREP:moving_to_place_scan"))
            rospy.loginfo("机械臂移动到放置观察位: %s", self.place_scan_joints)
            self.publish_observation_pose()
            if self.arm_move_wait > 0.0:
                rospy.sleep(self.arm_move_wait)
            self.wait_for_observation_pose()
            rospy.loginfo("机械臂已通过关节反馈确认到达放置观察位。")

        if self.open_gripper_at_observation:
            self.status_pub.publish(String(data="ARM_PREP:opening_gripper"))
            self.publish_open_gripper()
            rospy.loginfo(
                "已在放置观察位发送夹爪张开命令: gripper=%.1f",
                self.gripper_open_value,
            )
            if self.gripper_open_wait > 0.0:
                rospy.sleep(self.gripper_open_wait)

        self.status_pub.publish(String(data="ARM_PREP:ready"))
        rospy.loginfo("机械臂准备完成，现在开始D435i深度和图像识别。")

    def warmup(self):
        for _ in range(self.warmup_frames):
            if rospy.is_shutdown():
                return
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            aligned = self.align.process(frames)
            depth = aligned.get_depth_frame()
            if depth:
                apply_depth_filters(depth, self.filters)
        rospy.loginfo("D435i预热完成，开始平面深度测试。")

    def save_snapshot(self):
        if self.last_composite is None:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(
            self.save_dir,
            "%s-%s.jpg" % (timestamp, self.target_label.replace("/", "_")),
        )
        if cv2.imwrite(filename, self.last_composite):
            rospy.loginfo("已保存平面深度测试截图: %s", filename)
        else:
            rospy.logerr("保存测试截图失败: %s", filename)

    def process_frame(self, color_image, filtered_depth_m, intrinsics):
        height, width = color_image.shape[:2]
        annotated = color_image.copy()
        cv2.rectangle(
            annotated,
            (self.margin_x, self.margin_y),
            (width - 1 - self.margin_x, height - 1 - self.margin_y),
            (0, 255, 255),
            1,
        )
        results = self.model.predict(
            source=color_image,
            imgsz=self.image_size,
            conf=self.conf_threshold,
            verbose=False,
        )
        detection, edge_rejects = select_target_detection(
            results,
            self.model.names,
            self.target_label,
            width,
            height,
            self.margin_x,
            self.margin_y,
        )

        stable = None
        estimate = None
        if detection is None:
            self.stability.clear()
            reason = "target_at_edge" if edge_rejects else "target_not_found"
            status = "REJECT:%s:edge_rejects=%d" % (reason, edge_rejects)
            draw_text_lines(annotated, [status], color=(0, 0, 255))
            rospy.logwarn_throttle(1.0, "[平面深度] %s", status)
        else:
            estimate = self.estimator.estimate(
                filtered_depth_m, detection["box"], intrinsics
            )
            if estimate["ok"]:
                stable = self.stability.add(estimate["depth"], estimate["center"])
                if stable is None:
                    status = (
                        "WAIT:label=%s:plane=%.3f:stable=%d/%d:%s"
                        % (
                            detection["label"],
                            estimate["depth"],
                            len(self.stability.samples),
                            self.stability.sample_count,
                            patch_summary(estimate),
                        )
                    )
                    rospy.loginfo_throttle(1.0, "[平面深度] %s", status)
                else:
                    status = (
                        "PASS:label=%s:depth=%.3f:spread=%.3f:patches=%s:"
                        "inlier_ratio=%.2f:rmse=%.4f"
                        % (
                            detection["label"],
                            stable["depth"],
                            stable["depth_spread"],
                            "+".join(estimate["selected_patches"]),
                            estimate["plane"]["inlier_ratio"],
                            estimate["plane"]["rmse"],
                        )
                    )
                    self.depth_pub.publish(Float32(data=stable["depth"]))
                    rospy.loginfo_throttle(1.0, "[平面深度] %s", status)
            else:
                self.stability.clear()
                status = "REJECT:label=%s:reason=%s:%s" % (
                    detection["label"],
                    estimate["reason"],
                    patch_summary(estimate),
                )
                rospy.logwarn_throttle(1.0, "[平面深度] %s", status)

            draw_estimator_result(
                annotated,
                detection,
                estimate,
                stable,
                len(self.stability.samples),
            )

        self.status_pub.publish(String(data=status))
        depth_view = make_depth_colormap(
            filtered_depth_m, self.display_depth_min, self.display_depth_max
        )
        if detection is not None and estimate is not None:
            draw_estimator_result(
                depth_view,
                detection,
                estimate,
                stable,
                len(self.stability.samples),
            )
        cv2.putText(
            annotated,
            "RGB / q:quit s:save",
            (12, height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            depth_view,
            "FILTERED DEPTH %.1f-%.1fm" % (
                self.display_depth_min,
                self.display_depth_max,
            ),
            (12, height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        self.last_composite = np.hstack((annotated, depth_view))

    def run(self):
        try:
            self.prepare_arm_for_test()
            self.warmup()
            while not rospy.is_shutdown():
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                except RuntimeError as exc:
                    rospy.logwarn("等待D435i图像超时: %s", exc)
                    continue
                aligned = self.align.process(frames)
                raw_depth = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()
                if not raw_depth or not color_frame:
                    continue
                filtered_depth = apply_depth_filters(raw_depth, self.filters)
                color_image = np.asanyarray(color_frame.get_data())
                filtered_depth_m = (
                    np.asanyarray(filtered_depth.get_data()).astype(np.float32)
                    * self.depth_scale
                )
                intrinsics = (
                    filtered_depth.profile.as_video_stream_profile().intrinsics
                )
                self.process_frame(color_image, filtered_depth_m, intrinsics)

                if self.show_window and self.last_composite is not None:
                    try:
                        cv2.imshow("Place Plane Depth Test", self.last_composite)
                        key = cv2.waitKey(1) & 0xFF
                    except cv2.error as exc:
                        rospy.logwarn("OpenCV窗口不可用，继续使用ROS话题输出: %s", exc)
                        self.show_window = False
                        key = 0xFF
                    if key in (ord("q"), 27):
                        rospy.signal_shutdown("用户退出平面深度测试")
                    elif key == ord("s"):
                        self.save_snapshot()
        finally:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


def main():
    rospy.init_node("place_plane_depth_test", anonymous=False)
    node = PlacePlaneDepthTestNode()
    node.run()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except Exception as exc:
        try:
            rospy.logfatal("独立放置平面深度测试启动/运行失败: %s", exc)
        except Exception:
            pass
        raise
