#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""使用卡片平面深度算法的独立比赛机械臂任务节点。

这个文件不替换也不修改原 piper_task_node.py。它复用原比赛流程的回零、
卡片识别、抓取、放置动作、失败处理、车辆事件和相机中继，只覆盖放置阶段
的图像定位入口：优先用检测框四周卡片区域拟合三维平面；外围平面持续
不可用时，再用检测框中心小区域的稳定中位深度完成低精度放置。

对应入口：
  roslaunch piper_task piper_task_plane.launch enable_camera_relay:=true

原入口仍然可用：
  roslaunch piper_task piper_task.launch enable_camera_relay:=true
"""

import os
import sys

import cv2
import numpy as np
import pyrealsense2 as rs
import rospy


# piper_task_node.py 和 place_plane_depth_test.py 都是 scripts 下的可执行文件，
# 不是 Python package 模块。把当前目录加入路径后只导入类和函数；两个文件的
# main() 都有 __name__ 保护，导入不会启动第二个 ROS 节点或第二条相机管线。
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from piper_task_node import CompetitionTaskNode, CompetitionVisionController
from place_plane_depth_test import (
    PlacePlaneEstimator,
    StablePlaneDepth,
    draw_estimator_result,
)
from piper_task.vision_grasp_core import (
    CONF_THRES,
    IMG_SIZE,
    LABEL_DESCRIPTION,
    RED_BLOCK_CONF_THRES,
    RED_BLOCK_MIN_RATIO,
    VALID_TARGET_LABELS,
    apply_filters,
    box_outside_safe_edges,
    draw_edge_rejection,
    draw_safe_detection_region,
    get_red_ratio,
    is_box_inside_safe_region,
)


def clip_box(box, width, height):
    """把 YOLO xyxy 框裁剪到图像内。"""
    x1, y1, x2, y2 = [int(value) for value in box]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    return x1, y1, x2, y2


def estimate_center_roi_depth(
    depth_m,
    box,
    depth_min,
    depth_max,
    roi_ratio,
    min_valid_ratio,
    min_valid_pixels,
    max_mad,
):
    """从检测框中心小区域提取稳健深度，避免依赖单个中心像素。"""
    height, width = depth_m.shape[:2]
    x1, y1, x2, y2 = clip_box(box, width, height)
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    roi_width = max(5, int(round(max(1, x2 - x1) * roi_ratio)))
    roi_height = max(5, int(round(max(1, y2 - y1) * roi_ratio)))
    roi_x1 = max(0, int(round(center_x - 0.5 * roi_width)))
    roi_y1 = max(0, int(round(center_y - 0.5 * roi_height)))
    roi_x2 = min(width, max(roi_x1 + 1, roi_x1 + roi_width))
    roi_y2 = min(height, max(roi_y1 + 1, roi_y1 + roi_height))
    roi = depth_m[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_rect = (roi_x1, roi_y1, roi_x2, roi_y2)

    if roi.size == 0:
        return {
            "ok": False,
            "reason": "empty_center_roi",
            "center": (center_x, center_y),
            "roi": roi_rect,
        }

    valid_mask = (
        np.isfinite(roi)
        & (roi > float(depth_min))
        & (roi < float(depth_max))
    )
    values = roi[valid_mask]
    valid_count = int(values.size)
    valid_ratio = float(valid_count) / float(roi.size)
    result = {
        "ok": False,
        "center": (center_x, center_y),
        "roi": roi_rect,
        "valid_count": valid_count,
        "valid_ratio": valid_ratio,
        "depth": None,
        "mad": None,
    }
    if valid_count < int(min_valid_pixels) or valid_ratio < float(min_valid_ratio):
        result["reason"] = "center_support=%d/%.2f" % (
            valid_count,
            valid_ratio,
        )
        return result

    depth = float(np.median(values))
    mad = float(np.median(np.abs(values - depth)))
    result["depth"] = depth
    result["mad"] = mad
    if mad > float(max_mad):
        result["reason"] = "center_mad=%.3f" % mad
        return result

    result["ok"] = True
    result["reason"] = "ok"
    return result


def draw_status(image, text, color):
    """在比赛检测画面底部显示平面算法阶段。"""
    height, _ = image.shape[:2]
    origin = (12, max(24, height - 14))
    cv2.putText(
        image,
        str(text),
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        str(text),
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        1,
        cv2.LINE_AA,
    )


class PlanePlaceCompetitionVisionController(CompetitionVisionController):
    """原比赛视觉控制器的并行版本，仅替换放置定位算法。"""

    def __init__(self):
        super().__init__()

        # PlacePlaneEstimator 读取本节点私有参数 depth_min、ring_*、patch_*、
        # plane_*；这些参数只写在新的 task_plane_config.yaml 中。
        self.place_plane_estimator = PlacePlaneEstimator()
        self.place_plane_stable_samples = max(
            3, int(rospy.get_param("~place_plane_stable_samples", 8))
        )
        self.place_plane_max_depth_spread = max(
            0.001,
            float(rospy.get_param("~place_plane_max_depth_spread", 0.020)),
        )
        self.place_plane_max_center_spread = max(
            1.0,
            float(rospy.get_param("~place_plane_max_center_spread", 15.0)),
        )
        self.place_plane_retry_extra_frames = max(
            0, int(rospy.get_param("~place_plane_retry_extra_frames", 45))
        )
        self.place_plane_max_invalid_gap_frames = max(
            0,
            int(rospy.get_param("~place_plane_max_invalid_gap_frames", 8)),
        )
        self.place_plane_conf_threshold = min(
            1.0,
            max(
                0.01,
                float(
                    rospy.get_param(
                        "~place_plane_conf_threshold", CONF_THRES
                    )
                ),
            ),
        )
        self.place_plane_red_conf_threshold = min(
            1.0,
            max(
                0.01,
                float(
                    rospy.get_param(
                        "~place_plane_red_conf_threshold",
                        RED_BLOCK_CONF_THRES,
                    )
                ),
            ),
        )
        self.place_plane_image_size = max(
            64, int(rospy.get_param("~place_plane_image_size", IMG_SIZE))
        )
        self.place_center_fallback_enabled = bool(
            rospy.get_param("~place_center_fallback_enabled", True)
        )
        self.place_center_fallback_after_rejects = max(
            1, int(rospy.get_param("~place_center_fallback_after_rejects", 6))
        )
        self.place_center_fallback_roi_ratio = min(
            0.80,
            max(
                0.10,
                float(rospy.get_param("~place_center_fallback_roi_ratio", 0.40)),
            ),
        )
        self.place_center_fallback_min_valid_ratio = min(
            1.0,
            max(
                0.01,
                float(
                    rospy.get_param(
                        "~place_center_fallback_min_valid_ratio", 0.08
                    )
                ),
            ),
        )
        self.place_center_fallback_min_valid_pixels = max(
            5,
            int(rospy.get_param("~place_center_fallback_min_valid_pixels", 20)),
        )
        self.place_center_fallback_max_mad = max(
            0.001,
            float(rospy.get_param("~place_center_fallback_max_mad", 0.045)),
        )
        self.place_center_fallback_stable_samples = max(
            3,
            int(rospy.get_param("~place_center_fallback_stable_samples", 3)),
        )
        self.place_center_fallback_max_depth_spread = max(
            0.001,
            float(
                rospy.get_param(
                    "~place_center_fallback_max_depth_spread", 0.070
                )
            ),
        )

        rospy.logwarn(
            "已启动独立比赛节点 piper_task_plane_node.py："
            "抓取仍用原算法，只有放置定位使用卡片平面深度算法。"
        )
        rospy.loginfo(
            "放置平面参数: depth=%.3f~%.3fm patches>=%d stable=%d帧 "
            "depth_spread<=%.3fm center_spread<=%.1fpx extra=%d帧 "
            "invalid_gap<=%d帧",
            self.place_plane_estimator.depth_min,
            self.place_plane_estimator.depth_max,
            self.place_plane_estimator.min_consistent_patches,
            self.place_plane_stable_samples,
            self.place_plane_max_depth_spread,
            self.place_plane_max_center_spread,
            self.place_plane_retry_extra_frames,
            self.place_plane_max_invalid_gap_frames,
        )
        rospy.loginfo(
            "放置中心后备: enabled=%s plane_rejects>=%d roi=%.2f "
            "support>=%d/%.2f mad<=%.3fm stable=%d帧 spread<=%.3fm",
            self.place_center_fallback_enabled,
            self.place_center_fallback_after_rejects,
            self.place_center_fallback_roi_ratio,
            self.place_center_fallback_min_valid_pixels,
            self.place_center_fallback_min_valid_ratio,
            self.place_center_fallback_max_mad,
            self.place_center_fallback_stable_samples,
            self.place_center_fallback_max_depth_spread,
        )

    def select_place_detection(
        self,
        results,
        color_image,
        target_label,
        margin_x,
        margin_y,
        annotated,
    ):
        """选择指定放置图片，并保留原比赛节点的红方块 HSV 复核。"""
        height, width = color_image.shape[:2]
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None, 0, ()

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        candidates = []
        edge_rejects = 0
        edge_directions = []

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            model_label = str(self.model.names[int(class_id)])
            class_label = model_label

            if target_label == "block-r":
                if model_label not in VALID_TARGET_LABELS["block"]:
                    continue
                x1, y1, x2, y2 = clip_box(box, width, height)
                if x2 <= x1 or y2 <= y1:
                    continue
                red_ratio = get_red_ratio(color_image, x1, y1, x2, y2)
                model_strong_red = (
                    model_label == "block-r"
                    and float(confidence) >= self.place_plane_conf_threshold
                )
                if not (model_strong_red or red_ratio >= RED_BLOCK_MIN_RATIO):
                    continue
                class_label = "block-r"
            else:
                if model_label != target_label:
                    continue
                x1, y1, x2, y2 = clip_box(box, width, height)
                if x2 <= x1 or y2 <= y1:
                    continue

            box_tuple = (x1, y1, x2, y2)
            if not is_box_inside_safe_region(
                x1,
                y1,
                x2,
                y2,
                width,
                height,
                margin_x,
                margin_y,
            ):
                edge_rejects += 1
                directions = box_outside_safe_edges(
                    x1,
                    y1,
                    x2,
                    y2,
                    width,
                    height,
                    margin_x,
                    margin_y,
                )
                edge_directions.extend(directions)
                draw_edge_rejection(
                    annotated,
                    x1,
                    y1,
                    x2,
                    y2,
                    class_label,
                    float(confidence),
                )
                continue

            center_x = 0.5 * (x1 + x2)
            center_y = 0.5 * (y1 + y2)
            center_penalty = abs(center_x - width / 2.0) / width + abs(
                center_y - height / 2.0
            ) / height
            area_ratio = float((x2 - x1) * (y2 - y1)) / float(width * height)
            score = float(confidence) + 0.10 * area_ratio - 0.08 * center_penalty
            candidates.append(
                {
                    "box": box_tuple,
                    "confidence": float(confidence),
                    "label": class_label,
                    "score": score,
                }
            )

        if not candidates:
            return None, edge_rejects, tuple(edge_directions)
        return (
            max(candidates, key=lambda item: item["score"]),
            edge_rejects,
            tuple(edge_directions),
        )

    def get_place_pose_by_image_style(self, initial_image_style, max_frames=90):
        """用卡片四周平面深度定位指定放置图片并返回基座位姿。"""
        target_type = self.target_type_from_label(initial_image_style)
        if target_type is None:
            rospy.logerr("无法识别的初始图片样式: %s", initial_image_style)
            self.last_object_detection_failure = "invalid_target_label"
            return None

        detection_context = str(getattr(self, "detection_context", "place"))
        predict_conf = (
            self.place_plane_red_conf_threshold
            if initial_image_style == "block-r"
            else self.place_plane_conf_threshold
        )
        stable_depth = StablePlaneDepth(
            self.place_plane_stable_samples,
            self.place_plane_max_depth_spread,
            self.place_plane_max_center_spread,
        )
        center_fallback_depth = StablePlaneDepth(
            self.place_center_fallback_stable_samples,
            self.place_center_fallback_max_depth_spread,
            self.place_plane_max_center_spread,
        )
        base_frames = max(1, int(max_frames))
        total_frame_budget = base_frames + self.place_plane_retry_extra_frames

        self.last_object_detection_failure = "not_found"
        target_detection_count = 0
        valid_plane_count = 0
        plane_rejection_count = 0
        edge_rejection_count = 0
        edge_direction_counts = {}
        camera_timeout_count = 0
        center_fallback_valid_count = 0
        center_fallback_rejection_count = 0
        plane_invalid_streak = 0
        last_annotated = None

        rospy.loginfo(
            ">>> [新平面算法] 定位放置图片: %s (%s)，conf=%.2f，"
            "最多=%d+%d帧",
            LABEL_DESCRIPTION.get(initial_image_style, initial_image_style),
            initial_image_style,
            predict_conf,
            base_frames,
            self.place_plane_retry_extra_frames,
        )
        self.publish_detection_status(
            "%s:plane:start:label=%s:conf=%.2f:depth=%.3f-%.3f:stable=%d"
            % (
                detection_context,
                initial_image_style,
                predict_conf,
                self.place_plane_estimator.depth_min,
                self.place_plane_estimator.depth_max,
                self.place_plane_stable_samples,
            )
        )

        for frame_index in range(total_frame_budget):
            # 与原流程一致：完全没看到正确目标时只使用基础90帧；正确目标已经
            # 出现、但平面深度不稳定时，才在当前停车点增加重试帧数。
            if frame_index >= base_frames and target_detection_count == 0:
                break
            if rospy.is_shutdown():
                break

            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError as exc:
                camera_timeout_count += 1
                stable_depth.clear()
                center_fallback_depth.clear()
                plane_invalid_streak = self.place_plane_max_invalid_gap_frames + 1
                rospy.logwarn("%s 等待D435i图像超时: %s", detection_context, exc)
                continue

            aligned = self.align.process(frames)
            raw_depth = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not raw_depth or not color_frame:
                stable_depth.clear()
                center_fallback_depth.clear()
                plane_invalid_streak = self.place_plane_max_invalid_gap_frames + 1
                continue

            filtered_depth = apply_filters(raw_depth, self.filters).as_depth_frame()
            color_image = np.asanyarray(color_frame.get_data())
            filtered_depth_m = (
                np.asanyarray(filtered_depth.get_data()).astype(np.float32)
                * self.depth_scale
            )
            intrinsics = filtered_depth.profile.as_video_stream_profile().intrinsics
            annotated = color_image.copy()
            height, width = annotated.shape[:2]
            margin_x = min(self.object_edge_margin_x, max((width - 2) // 2, 0))
            margin_y = min(self.object_edge_margin_y, max((height - 2) // 2, 0))
            draw_safe_detection_region(annotated, margin_x, margin_y)

            results = self.model.predict(
                source=color_image,
                imgsz=self.place_plane_image_size,
                conf=predict_conf,
                verbose=False,
            )
            detection, edge_rejects, edge_directions = self.select_place_detection(
                results,
                color_image,
                initial_image_style,
                margin_x,
                margin_y,
                annotated,
            )
            edge_rejection_count += edge_rejects
            for direction in edge_directions:
                edge_direction_counts[direction] = (
                    edge_direction_counts.get(direction, 0) + 1
                )

            if detection is None:
                plane_invalid_streak += 1
                if plane_invalid_streak > self.place_plane_max_invalid_gap_frames:
                    stable_depth.clear()
                center_fallback_depth.clear()
                reason = "target_at_edge" if edge_rejects else "target_not_found"
                status = "%s:plane:reject:%s:edge_rejects=%d" % (
                    detection_context,
                    reason,
                    edge_rejects,
                )
                draw_status(annotated, "PLANE REJECT: %s" % reason, (0, 0, 255))
                self.publish_detection_status(status)
                rospy.logwarn_throttle(
                    1.0,
                    "[%s 新平面算法] %s",
                    detection_context,
                    status,
                )
                last_annotated = annotated
                self.publish_detection_image(annotated)
                continue

            target_detection_count += 1
            estimate = self.place_plane_estimator.estimate(
                filtered_depth_m,
                detection["box"],
                intrinsics,
            )
            if not estimate["ok"]:
                plane_rejection_count += 1
                plane_invalid_streak += 1
                if plane_invalid_streak > self.place_plane_max_invalid_gap_frames:
                    stable_depth.clear()
                fallback = None
                fallback_stable = None
                if (
                    self.place_center_fallback_enabled
                    and plane_rejection_count
                    >= self.place_center_fallback_after_rejects
                ):
                    fallback = estimate_center_roi_depth(
                        filtered_depth_m,
                        detection["box"],
                        self.place_plane_estimator.depth_min,
                        self.place_plane_estimator.depth_max,
                        self.place_center_fallback_roi_ratio,
                        self.place_center_fallback_min_valid_ratio,
                        self.place_center_fallback_min_valid_pixels,
                        self.place_center_fallback_max_mad,
                    )
                    fx1, fy1, fx2, fy2 = fallback["roi"]
                    cv2.rectangle(
                        annotated,
                        (fx1, fy1),
                        (max(fx1, fx2 - 1), max(fy1, fy2 - 1)),
                        (255, 255, 0),
                        2,
                    )
                    if fallback["ok"]:
                        center_fallback_valid_count += 1
                        fallback_stable = center_fallback_depth.add(
                            fallback["depth"], fallback["center"]
                        )
                    else:
                        center_fallback_rejection_count += 1
                        center_fallback_depth.clear()

                status = "%s:plane:reject:label=%s:reason=%s" % (
                    detection_context,
                    detection["label"],
                    estimate["reason"],
                )
                draw_estimator_result(
                    annotated,
                    detection,
                    estimate,
                    None,
                    0,
                )
                draw_status(annotated, "COMPETITION PLANE PLACE", (0, 165, 255))
                self.publish_detection_status(status)
                if fallback is not None:
                    if fallback["ok"]:
                        fallback_status = (
                            "%s:center_fallback:wait:label=%s:depth=%.3f:"
                            "mad=%.3f:stable=%d/%d"
                            % (
                                detection_context,
                                detection["label"],
                                fallback["depth"],
                                fallback["mad"],
                                len(center_fallback_depth.samples),
                                center_fallback_depth.sample_count,
                            )
                        )
                    else:
                        fallback_status = (
                            "%s:center_fallback:reject:label=%s:reason=%s"
                            % (
                                detection_context,
                                detection["label"],
                                fallback["reason"],
                            )
                        )
                    draw_status(
                        annotated,
                        "CENTER FALLBACK: %s" % fallback["reason"],
                        (255, 255, 0) if fallback["ok"] else (0, 0, 255),
                    )
                    self.publish_detection_status(fallback_status)
                    rospy.loginfo_throttle(
                        1.0,
                        "[%s 放置中心后备] %s",
                        detection_context,
                        fallback_status,
                    )
                rospy.logwarn_throttle(
                    1.0,
                    "[%s 新平面算法] %s",
                    detection_context,
                    status,
                )
                last_annotated = annotated
                self.publish_detection_image(annotated)
                if fallback_stable is not None:
                    center_x, center_y = fallback_stable["center"]
                    xyz_cam = rs.rs2_deproject_pixel_to_point(
                        intrinsics,
                        [float(center_x), float(center_y)],
                        float(fallback_stable["depth"]),
                    )
                    T_base_place = self.make_object_pose_in_base(xyz_cam)
                    self.last_object_detection_failure = None
                    success_status = (
                        "%s:center_fallback:success:label=%s:depth=%.3f:"
                        "spread=%.3f:center_spread=%.1f:mad=%.3f"
                        % (
                            detection_context,
                            detection["label"],
                            fallback_stable["depth"],
                            fallback_stable["depth_spread"],
                            fallback_stable["center_spread"],
                            fallback["mad"],
                        )
                    )
                    self.publish_detection_status(success_status)
                    rospy.loginfo("========================================")
                    rospy.logwarn(
                        "[新放置平面算法] 中心后备最终 PASS: %s",
                        success_status,
                    )
                    rospy.loginfo(
                        "中心后备相机坐标: X=%.4f Y=%.4f Z=%.4f m",
                        xyz_cam[0],
                        xyz_cam[1],
                        xyz_cam[2],
                    )
                    rospy.loginfo(
                        "中心后备基座坐标: X=%.4f Y=%.4f Z=%.4f m",
                        T_base_place[0, 3],
                        T_base_place[1, 3],
                        T_base_place[2, 3],
                    )
                    rospy.loginfo("========================================")
                    return T_base_place
                continue

            valid_plane_count += 1
            plane_invalid_streak = 0
            stable = stable_depth.add(estimate["depth"], estimate["center"])
            draw_estimator_result(
                annotated,
                detection,
                estimate,
                stable,
                len(stable_depth.samples),
            )
            draw_status(annotated, "COMPETITION PLANE PLACE", (255, 255, 255))
            last_annotated = annotated
            self.publish_detection_image(annotated)

            if stable is None:
                status = (
                    "%s:plane:wait:label=%s:depth=%.3f:stable=%d/%d:"
                    "invalid_gap<=%d"
                ) % (
                    detection_context,
                    detection["label"],
                    estimate["depth"],
                    len(stable_depth.samples),
                    stable_depth.sample_count,
                    self.place_plane_max_invalid_gap_frames,
                )
                self.publish_detection_status(status)
                rospy.loginfo_throttle(
                    1.0,
                    "[%s 新平面算法] %s",
                    detection_context,
                    status,
                )
                continue

            center_x, center_y = stable["center"]
            xyz_cam = rs.rs2_deproject_pixel_to_point(
                intrinsics,
                [float(center_x), float(center_y)],
                float(stable["depth"]),
            )
            T_base_place = self.make_object_pose_in_base(xyz_cam)
            self.last_object_detection_failure = None
            status = (
                "%s:plane:success:label=%s:depth=%.3f:spread=%.3f:"
                "center_spread=%.1f:patches=%s:inlier_ratio=%.2f:rmse=%.4f"
                % (
                    detection_context,
                    detection["label"],
                    stable["depth"],
                    stable["depth_spread"],
                    stable["center_spread"],
                    "+".join(estimate["selected_patches"]),
                    estimate["plane"]["inlier_ratio"],
                    estimate["plane"]["rmse"],
                )
            )
            self.publish_detection_status(status)
            rospy.loginfo("========================================")
            rospy.loginfo("[新放置平面算法] 最终 PASS: %s", status)
            rospy.loginfo(
                "相机坐标目标中心: X=%.4f Y=%.4f Z=%.4f m",
                xyz_cam[0],
                xyz_cam[1],
                xyz_cam[2],
            )
            rospy.loginfo(
                "基座坐标目标中心: X=%.4f Y=%.4f Z=%.4f m",
                T_base_place[0, 3],
                T_base_place[1, 3],
                T_base_place[2, 3],
            )
            rospy.loginfo("========================================")
            return T_base_place

        if last_annotated is not None:
            self.publish_detection_image(last_annotated)

        if edge_rejection_count > 0 and target_detection_count == 0:
            if edge_direction_counts:
                direction = max(
                    edge_direction_counts,
                    key=lambda key: edge_direction_counts[key],
                )
            else:
                direction = "unknown"
            self.last_object_detection_failure = "target_at_edge=%s" % direction
        elif target_detection_count > 0:
            self.last_object_detection_failure = "invalid_depth"
        else:
            self.last_object_detection_failure = "not_found"

        status = (
            "%s:plane:failed:label=%s:reason=%s:target_frames=%d:"
            "valid_planes=%d:plane_rejects=%d:edge_rejects=%d:camera_timeouts=%d:"
            "center_valid=%d:center_rejects=%d"
            % (
                detection_context,
                initial_image_style,
                self.last_object_detection_failure,
                target_detection_count,
                valid_plane_count,
                plane_rejection_count,
                edge_rejection_count,
                camera_timeout_count,
                center_fallback_valid_count,
                center_fallback_rejection_count,
            )
        )
        self.publish_detection_status(status)
        rospy.logwarn("[新放置平面算法] %s", status)
        return None


def main():
    arm = PlanePlaceCompetitionVisionController()
    node = CompetitionTaskNode(arm)
    node.run()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except RuntimeError as exc:
        rospy.logfatal("piper_task_plane 启动失败：%s", exc)
        raise
