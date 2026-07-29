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

# ============ 全局变量 ============
current_T_ee2base = np.eye(4)
_last_pose_time = None
tx_init = ty_init = tz_init = 0
is_first = True

# ============ 回调函数 ============
def end_pose_callback(msg):
    """订阅 /end_pose，将 geometry_msgs/PoseStamped 转换为 4x4 齐次变换矩阵"""
    global current_T_ee2base, _last_pose_time
    global tx_init,ty_init,tz_init
    global is_first
    try:
        pose = msg.pose  # PoseStamped 内部含 Pose
        qx, qy, qz, qw = pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
        tx, ty, tz = pose.position.x, pose.position.y, pose.position.z

        if is_first:
            tx_init = tx
            ty_init = ty
            tz_init = tz
            is_first = False
        T = quaternion_matrix([qx, qy, qz, qw])
        T[0, 3], T[1, 3], T[2, 3] = tx, ty, tz
        current_T_ee2base = T
        _last_pose_time = rospy.Time.now()
        rospy.loginfo_throttle(1.0, f"[end_pose_cb] x={tx:.3f}, y={ty:.3f}, z={tz:.3f}")
    except Exception as e:
        rospy.logerr(f"end_pose_callback error: {e}")


# ============ 核心检测函数 ============
def detect_once(pub, model, device, imgsz, names, colors, align_to_color):

    """拍一帧 -> 检测 -> 发布一次"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    
    #启动检测
    pipeline.start(config)
    rospy.loginfo("RealSense 已启动，捕获一帧图像...")

    distance_list_flag = True
    while distance_list_flag:
        # 获取一帧对齐后的彩色和深度图
        frames = pipeline.wait_for_frames()
        frames = align_to_color.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            rospy.logwarn("无法获取 RealSense 帧")
            pipeline.stop()
            return False

        color_image = np.asanyarray(color_frame.get_data())

        # YOLO 输入预处理
        im0 = color_image.copy() # 复制一份备用
        img = letterbox(im0, new_shape=imgsz)[0] # 将图像缩放到指定大小，并返回缩放后的图像和缩放比例
        img = img[:, :, ::-1].transpose(2, 0, 1)  # 前一半是BGR to RGB, 后一半是将图像的维度顺序从(1, 416, 416, 3)转换为(1, 3, 416, 416)，即HWC → CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0 # 将图像转换为连续的内存布局，并归一化到0.0 - 1.0
        
        # 转换设备
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 取pred为检测到的第一张图像，并用NMS筛选画框
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.25, 0.45, agnostic=False)

        found = False

        
        
        det = pred[0]
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round() # 将预测框从模型输入尺寸映射回原始图像尺寸
            # 取第一个检测框
            *xyxy, conf, cls = det[0]
            mid_pos = [int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)]

            # 获取深度
            distance_list = []
            randnum = 30
            min_val = min(abs(int(xyxy[2] - xyxy[0])), abs(int(xyxy[3] - xyxy[1])))
            #print(min_val)
            for j in range(randnum):
                bias = random.randint(-min_val // 4, min_val // 4)
                dist = depth_frame.get_distance(int(mid_pos[0] + bias), int(mid_pos[1] + bias))
                if dist:
                    distance_list.append(dist)
                    
            distance_list = np.array(distance_list)
            distance_list = np.sort(distance_list)[
                            randnum // 2 - randnum // 4:randnum // 2 + randnum // 4]  # 冒泡排序+中值滤波

            mean_dist = np.mean(distance_list) 
            print(mean_dist)

            if len(distance_list) == 0:
                continue
            else:
                distance_list_flag = False
            
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics # 获取深度相机的 相机内参
            X, Y, Z = rs.rs2_deproject_pixel_to_point(depth_intrin, mid_pos, mean_dist) # 将2D坐标和深度信息转换为3D坐标
            rospy.loginfo(f"检测到 [{names[int(cls)]}]：相机坐标 X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}")

            # 相机->末端标定矩阵
            T_cam2ee = np.array([
                [-0.0482, 0.9987,  0.0147, -0.0699],
                [-0.9979, -0.0488, 0.0416, 0.0301],
                [0.0423, -0.0127,  0.9990, 0.0675],
                [0.0,     0.0,     0.0,     1.0]
            ])

            # 末端->基座
            T_ee2base = current_T_ee2base.copy()

            # 目标在基座坐标系下的位置
            P_cam = np.array([[X], [Y], [Z], [1]])
            P_base = T_ee2base @ T_cam2ee @ P_cam

            msg = PosCmd()
            msg.x = float(P_base[0])
            msg.y = float(P_base[1])
            msg.z = float(P_base[2]) + 0.3
            msg.roll = -1.57
            msg.pitch = 1.57
            msg.yaw = 0.0
            msg.gripper = 250
            msg.mode1 = 1
            msg.mode2 = 0
            pub.publish(msg)
            time.sleep(2)
            '''
            msg = PosCmd()
            msg.x = float(P_base[0])
            msg.y = float(P_base[1])
            msg.z = float(P_base[2]) + 0.2
            msg.roll = 0
            msg.pitch = 0
            msg.yaw = 0.0
            msg.gripper = 0
            msg.mode1 = 1
            msg.mode2 = 0
            pub.publish(msg)
            time.sleep(2)

            msg = PosCmd()
            msg.x = float(P_base[0])
            msg.y = float(P_base[1])
            msg.z = float(P_base[2]) + 0.3
            msg.roll = 0
            msg.pitch = 0
            msg.yaw = 0.0
            msg.gripper = 250
            msg.mode1 = 1
            msg.mode2 = 0
            pub.publish(msg)
            time.sleep(2)
            '''
            '''
            msg = PosCmd()
            msg.x = tx_init
            msg.y = ty_init
            msg.z = tz_init
            msg.roll = 0
            msg.pitch = 3.14
            msg.yaw = 0.0
            msg.gripper = 0
            msg.mode1 = 1
            msg.mode2 = 0
            pub.publish(msg)
            time.sleep(2)
            '''
            '''
            deta1 = 0
            for step in range(20):
                msg = PosCmd()
                msg.x = float(P_base[0])
                msg.y = float(P_base[1])
                msg.z = float(P_base[2]) + 0.2 + deta1
                msg.roll = -1.57
                msg.pitch = 1.57
                msg.yaw = 0.0
                msg.gripper = 250
                msg.mode1 = 1
                msg.mode2 = 0
                pub.publish(msg)
                deta1 = deta1 +0.01
                time.sleep(0.1)
            '''
            '''
            deta2 = 0
            for step in range(10):
                msg = PosCmd()
                msg.x = float(P_base[0]) + deta2
                msg.y = float(P_base[1]) - deta2
                msg.z = float(P_base[2]) + 0.2 + deta1
                msg.roll = -1.57
                msg.pitch = 1.57
                msg.yaw = 0.0
                msg.gripper = 0.0
                msg.mode1 = 1
                msg.mode2 = 0
                pub.publish(msg)
                deta2 = deta2 +0.01
                time.sleep(0.1)
            '''
            found = True

        pipeline.stop()
        return found


# ============ 主函数 ============
def detect():
    # 检测初始化
    set_logging()
    device = select_device('')
    weights = opt.weights
    imgsz = opt.img_size
    model = attempt_load(weights, map_location=device)
    imgsz = check_img_size(imgsz, s=model.stride.max())
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # ROS 初始化
    if not rospy.core.is_initialized():
        rospy.init_node('realsense_detect_once', anonymous=True)
    pub = rospy.Publisher('/pin_pos_cmd', PosCmd, queue_size=1)
    rospy.Subscriber('/end_pose', PoseStamped, end_pose_callback)
    rospy.sleep(0.5)

    # RealSense 对齐对象
    align_to_color = rs.align(rs.stream.color)

    # 第一次检测
    rospy.loginfo("开始第一次检测...")
    success = detect_once(pub, model, device, imgsz, names, colors, align_to_color)

    if not success:
        rospy.logwarn("第一次未检测到目标，重试一次...")
        success = detect_once(pub, model, device, imgsz, names, colors, align_to_color)

    if success:
        rospy.loginfo("检测并发布完成，程序退出。")
    else:
        rospy.logwarn("两次检测均未找到目标，程序退出。")


# ============ 启动入口 ============
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    opt = parser.parse_args()

    with torch.no_grad():
        detect()
