#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import tf.transformations as tf_trans
from ultralytics import YOLO

from piper_msgs.msg import PosCmd
from geometry_msgs.msg import PoseStamped

# ===============================
# 1. 基础配置
# ===============================
MODEL_PATH = "/home/hank/piper_ws/src/piper_ros/src/bottle_grab/bottle.pt"
TARGET_CLASS_KEYWORD = "bottle"
CONF_THRES = 0.45
IMG_SIZE = 640
DISPARITY_SHIFT = 58 

# ===============================
# 2. 坐标变换矩阵
# ===============================
T_EE_TO_CAM = np.array([
    [-0.0482,  0.9987,  0.0147, -0.0699],
    [-0.9979, -0.0488,  0.0416,  0.0301],
    [ 0.0423, -0.0127,  0.9990,  0.0675],
    [ 0.0,     0.0,     0.0,     1.0]
])

T_EE_TO_TOOL = np.eye(4)
T_EE_TO_TOOL[0:3, 3] = [0.0, 0.0, -0.14] 
T_TOOL_TO_EE = np.linalg.inv(T_EE_TO_TOOL)

def get_horizontal_grasp_matrix():
    # 绕Y轴转90度使夹爪指向正前方
    R_y = tf_trans.rotation_matrix(np.pi/2, [0, 1, 0])
    # 如果夹爪方向需要旋转，调整此处 R_z
    R_z = tf_trans.rotation_matrix(0, [0, 0, 1])
    return tf_trans.concatenate_matrices(R_y, R_z)

T_OBJ_TO_GRASP = get_horizontal_grasp_matrix()

# ===============================
# 3. 核心工具函数
# ===============================
def get_robust_depth(depth_frame, x1, y1, x2, y2):
    w, h = x2 - x1, y2 - y1
    roi_x1, roi_y1 = int(x1 + w*0.3), int(y1 + h*0.3)
    roi_x2, roi_y2 = int(x2 - w*0.3), int(y2 - h*0.3)
    depth_values = []
    for v in range(roi_y1, roi_y2, 4):
        for u in range(roi_x1, roi_x2, 4):
            d = depth_frame.get_distance(u, v)
            if 0.15 < d < 1.0:
                depth_values.append(d)
    return np.median(depth_values) if depth_values else 0.0

# ===============================
# 4. 逻辑控制类
# ===============================
class PiperHorizontalGrabber:
    def __init__(self):
        rospy.init_node("piper_final_grab_node")
        self.pub = rospy.Publisher("/pin_pos_cmd", PosCmd, queue_size=1)
        self.current_ee_pose_mat = np.eye(4)
        self.pose_received = False
        
        rospy.Subscriber("/end_pose", PoseStamped, self.pose_callback)
        
        rospy.loginfo(">>> 正在初始化 YOLO 与 RealSense...")
        self.model = YOLO(MODEL_PATH)
        self.init_camera()
        
        while not self.pose_received and not rospy.is_shutdown():
            rospy.sleep(0.1)

    def pose_callback(self, msg):
        q = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        T = tf_trans.quaternion_matrix(q)
        T[0:3, 3] = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.current_ee_pose_mat = T
        self.pose_received = True

    def init_camera(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        
        adv = rs.rs400_advanced_mode(profile.get_device())
        dt = adv.get_depth_table()
        dt.disparityShift = DISPARITY_SHIFT
        adv.set_depth_table(dt)

    def send_pose(self, T_matrix, gripper=0, label=""):
        x, y, z = T_matrix[0:3, 3]
        roll, pitch, yaw = tf_trans.euler_from_matrix(T_matrix, axes='sxyz')
        
        # 打印即将发送的最终位姿
        if label:
            rospy.loginfo(f"[指令发送 - {label}] X:{x:.4f} Y:{y:.4f} Z:{z:.4f} | R:{roll:.3f} P:{pitch:.3f} Y:{yaw:.3f} | Gripper:{gripper}")
        
        cmd = PosCmd()
        cmd.x, cmd.y, cmd.z = x, y, z
        cmd.roll, cmd.pitch, cmd.yaw = roll, pitch, yaw
        cmd.gripper, cmd.mode1 = gripper, 1
        self.pub.publish(cmd)

    def get_target_from_yolo(self):
        for i in range(50):
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            color_f, depth_f = aligned.get_color_frame(), aligned.get_depth_frame()
            if not color_f or not depth_f: continue

            img = np.asanyarray(color_f.get_data())
            results = self.model.predict(img, conf=CONF_THRES, verbose=False)

            if results and len(results[0].boxes) > 0:
                best_box = None
                max_conf = -1
                for box in results[0].boxes:
                    name = self.model.names[int(box.cls)]
                    if TARGET_CLASS_KEYWORD.lower() in name.lower() and box.conf > max_conf:
                        max_conf = box.conf
                        best_box = box
                
                if best_box is not None:
                    xyxy = best_box.xyxy[0].cpu().numpy()
                    dist = get_robust_depth(depth_f, xyxy[0], xyxy[1], xyxy[2], xyxy[3])
                    
                    if dist > 0:
                        intr = depth_f.profile.as_video_stream_profile().intrinsics
                        cx, cy = int((xyxy[0]+xyxy[2])/2), int((xyxy[1]+xyxy[3])/2)
                        pt_c = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], dist)
                        
                        # 核心坐标计算
                        T_cam_to_base = self.current_ee_pose_mat @ T_EE_TO_CAM
                        P_base = T_cam_to_base @ np.array([pt_c[0], pt_c[1], pt_c[2], 1.0])
                        
                        # 打印瓶子相对于基座的最终坐标
                        rospy.loginfo("--------------------------------------------------")
                        rospy.loginfo(f"锁定瓶子: {self.model.names[int(best_box.cls)]}")
                        rospy.loginfo(f"瓶子基座坐标 (P_base) -> X: {P_base[0]:.4f}, Y: {P_base[1]:.4f}, Z: {P_base[2]:.4f}")
                        rospy.loginfo("--------------------------------------------------")
                        
                        T_obj = np.eye(4)
                        T_obj[0:3, 3] = P_base[0:3]
                        return T_obj
            rospy.sleep(0.05)
        return None

    def execute_sequence(self):
        # 1. 观察位
        T_view = tf_trans.euler_matrix(0, 1.57, 0, 'sxyz')
        T_view[0:3, 3] = [0.26, 0.0, 0.18] 
        self.send_pose(T_view, 0, label="观察位")
        time.sleep(7)

        # 2. 识别
        T_obj = self.get_target_from_yolo()
        if T_obj is None:
            rospy.logerr("识别超时，未发现有效目标。")
            return

        # 3. 计算机械臂最终抓取位姿 (EE)
        T_grasp_ee = T_obj @ T_OBJ_TO_GRASP @ T_TOOL_TO_EE

        # A. 预备点
        T_pre = T_grasp_ee @ tf_trans.translation_matrix([0, 0, 0])
        self.send_pose(T_pre, 80, label="水平预备点")
        time.sleep(3.0)
        '''
        # B. 最终抓取推进
        self.send_pose(T_grasp_ee, 80, label="最终抓取推进")
        time.sleep(2.0)

        # C. 闭合
        self.send_pose(T_grasp_ee, 0, label="闭合夹爪")
        time.sleep(1.5)

        # D. 抬起
        T_lift = T_grasp_ee.copy()
        T_lift[2, 3] += 0.1
        self.send_pose(T_lift, 0, label="抬起目标")
        rospy.loginfo(">>> 抓取序列执行完毕。")
        '''
    def run(self):
        try:
            self.execute_sequence()
        finally:
            self.pipeline.stop()

if __name__ == "__main__":
    PiperHorizontalGrabber().run()
