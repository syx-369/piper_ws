#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import pyrealsense2 as rs
from numpy import random

from models.experimental import attempt_load
from utils.general import (
    check_img_size, non_max_suppression, scale_coords,
    plot_one_box, set_logging)
from utils.torch_utils import select_device
from utils.datasets import letterbox

import rospy
from geometry_msgs.msg import PoseStamped
from piper_msgs.msg import PosCmd
from tf.transformations import quaternion_matrix
from pyzbar import pyzbar


# ============ 全局变量 ============
current_T_ee2base = np.eye(4)
_last_pose_time = None

# ============ 回调函数 ============
def end_pose_callback(msg):
    """订阅 /end_pose，将 geometry_msgs/PoseStamped 转换为 4x4 齐次变换矩阵"""
    global current_T_ee2base, _last_pose_time
    try:
        pose = msg.pose  # PoseStamped 内部含 Pose
        qx, qy, qz, qw = pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
        tx, ty, tz = pose.position.x, pose.position.y, pose.position.z

        T = quaternion_matrix([qx, qy, qz, qw])
        T[0, 3], T[1, 3], T[2, 3] = tx, ty, tz
        current_T_ee2base = T
        _last_pose_time = rospy.Time.now()
        
        rospy.loginfo_throttle(1.0, f"[end_pose_cb] x={tx:.3f}, y={ty:.3f}, z={tz:.3f}")
    except Exception as e:
        rospy.logerr(f"end_pose_callback error: {e}")


# ============ 核心检测函数 ============
def detect_once(pub, align_to_color):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    pipeline.start(config)
    rospy.loginfo("RealSense 已启动，等待二维码...")

    while not rospy.is_shutdown():
        frames = pipeline.wait_for_frames()
        frames = align_to_color.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        qrcodes = pyzbar.decode(gray)
        if len(qrcodes) == 0:
            continue

        qr = qrcodes[0]  # 只取第一个二维码
        pts = qr.polygon
        if len(pts) < 4:
            continue

        # 计算二维码中心像素
        pixel_x = int(sum(p.x for p in pts) / len(pts))
        pixel_y = int(sum(p.y for p in pts) / len(pts))
        mid_pos = [pixel_x, pixel_y]

        # 深度多点采样（和 YOLO 一样）
        distance_list = []
        for _ in range(30):
            bias = random.randint(-5, 5)
            dist = depth_frame.get_distance(pixel_x + bias, pixel_y + bias)
            if dist > 0:
                distance_list.append(dist)

        if len(distance_list) == 0:
            continue

        distance_list = np.sort(distance_list)
        mean_dist = np.mean(distance_list[len(distance_list)//4 : len(distance_list)*3//4])

        # ===== 核心：像素 → 相机坐标系 =====
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        X, Y, Z = rs.rs2_deproject_pixel_to_point(
            depth_intrin, mid_pos, mean_dist
        )

        rospy.loginfo(f"[QR] 相机坐标: X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}")

        # ===== 手眼标定 =====
        T_cam2ee = np.array([
            [-0.0482,  0.9987,  0.0147, -0.0699],
            [-0.9979, -0.0488,  0.0416,  0.0301],
            [ 0.0423, -0.0127,  0.9990,  0.0675],
            [ 0.0,     0.0,     0.0,     1.0]
        ])

        T_ee2base = current_T_ee2base.copy()

        P_cam = np.array([[X], [Y], [Z], [1]])
        P_base = T_ee2base @ T_cam2ee @ P_cam
        
        rospy.loginfo(
            f"P_base = x:{P_base[0,0]:.4f}, y:{P_base[1,0]:.4f}, z:{P_base[2,0]:.4f}"
        )
        x_dst, y_dst, z_dst= float(P_base[0]), float(P_base[1]), float(P_base[2])
        x_act, y_act, z_act = pose.position.x, pose.position.y, pose.position.z

        # ===== 发布控制指令 =====
        msg = PosCmd()
        msg.x = float(P_base[0])
        msg.y = float(P_base[1])
        msg.z = float(P_base[2]) + 0.15
        msg.roll = 0
        msg.pitch = 1.57
        msg.yaw = 0.0
        msg.gripper = 0.035
        msg.mode1 = 1
        msg.mode2 = 0

        pub.publish(msg)
        rospy.loginfo("二维码目标已发布 /pin_pos_cmd")

        pipeline.stop()
        return True

# ============ 主函数 ============
def detect():
    if not rospy.core.is_initialized():
        rospy.init_node('realsense_detect_once', anonymous=True)

    pub = rospy.Publisher('/pin_pos_cmd', PosCmd, queue_size=1)
    rospy.Subscriber('/end_pose', PoseStamped, end_pose_callback)
    rospy.sleep(0.5)

    align_to_color = rs.align(rs.stream.color)

    rospy.loginfo("开始第一次检测...")
    success = detect_once(pub, align_to_color)

    if not success:
        rospy.logwarn("第一次未检测到二维码，重试一次...")
        success = detect_once(pub, align_to_color)

    if success:
        rospy.loginfo("检测并发布完成，程序退出。")
    else:
        rospy.logwarn("两次检测均未找到二维码，程序退出。")



# ============ 启动入口 ============
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    opt = parser.parse_args()

    with torch.no_grad():
        detect()
