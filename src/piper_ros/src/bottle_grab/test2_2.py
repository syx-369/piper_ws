#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import tf.transformations as tf_trans

from piper_msgs.msg import PosCmd
from geometry_msgs.msg import PoseStamped

# ===============================
# 基本参数与平滑控制参数
# ===============================
ARUCO_REAL_SIZE_M = 0.03    # 二维码实际边长(米)
TARGET_ARUCO_ID = 6         # 目标二维码ID

RATE_HZ = 200               # 控制频率 (Hz)
V_TARGET = 0.08             # 期望最大速度 (m/s)
A_MAX = 0.2                 # 最大加速度 (m/s^2)
ROT_SPEED = 0.5             # 旋转角速度 (rad/s)

# ===============================
# 位姿矩阵
# ===============================
# 1. 相机相对于机械臂末端的位姿
T_EE_TO_CAM = np.array([
    [-0.0482,  0.9987,  0.0147, -0.0699],
    [-0.9979, -0.0488,  0.0416,  0.0301],
    [ 0.0423, -0.0127,  0.9990,  0.0675],
    [ 0.0,     0.0,     0.0,     1.0]
])

# 2. 工具(夹爪)相对于机械臂末端的位姿 (平移14厘米)
T_EE_TO_TOOL = np.eye(4)
# T_EE_TO_TOOL[2, 3] = -0.14
T_TOOL_TO_EE = np.linalg.inv(T_EE_TO_TOOL)

# 3. 抓取姿态相对于二维码的基准旋转 (Z同向, X向下, Y向右)
R1 = tf_trans.rotation_matrix(-np.pi/2, [0, 0, 1])
R2 = tf_trans.rotation_matrix(np.pi, [1, 0, 0])
T_ARUCO_TO_GRASP = tf_trans.concatenate_matrices(R1, R2)

class PiperVisionController:
    def __init__(self):
        rospy.init_node("piper_smooth_vision_mission", anonymous=True)
        
        # 发布者
        self.pub = rospy.Publisher("/pin_pos_cmd", PosCmd, queue_size=1)
        
        # 实时位姿反馈
        self.current_ee_pose_mat = np.eye(4)
        self.pose_received = False
        rospy.Subscriber("/end_pose", PoseStamped, self.pose_callback)
        
        rospy.loginfo("等待获取机械臂实时位姿...")
        while not self.pose_received and not rospy.is_shutdown():
            rospy.sleep(0.1)
        
        self.init_camera()

    def pose_callback(self, msg):
        p = msg.pose.position
        o = msg.pose.orientation
        T_mat = tf_trans.quaternion_matrix([o.x, o.y, o.z, o.w])
        T_mat[0:3, 3] = [p.x, p.y, p.z]
        self.current_ee_pose_mat = T_mat
        self.pose_received = True

    def init_camera(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.camera_matrix = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
        self.dist_coeffs = np.zeros((4, 1))
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.aruco_params = cv2.aruco.DetectorParameters()

    def get_aruco_pose(self, marker_id):
        """ 检测二维码（已移除弹窗显示） """
        for _ in range(30): self.pipeline.wait_for_frames() # 丢弃旧帧
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame: return None
        
        img = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is not None and marker_id in ids:
            idx = np.where(ids == marker_id)[0][0]
            obj_pts = np.array([[-ARUCO_REAL_SIZE_M/2, ARUCO_REAL_SIZE_M/2, 0],
                                [ ARUCO_REAL_SIZE_M/2, ARUCO_REAL_SIZE_M/2, 0],
                                [ ARUCO_REAL_SIZE_M/2,-ARUCO_REAL_SIZE_M/2, 0],
                                [-ARUCO_REAL_SIZE_M/2,-ARUCO_REAL_SIZE_M/2, 0]], dtype=np.float32)
            _, rvec, tvec = cv2.solvePnP(obj_pts, corners[idx][0], self.camera_matrix, self.dist_coeffs)
            
            # --- 视觉弹窗代码已移除 ---

            T_cam_aruco = np.eye(4)
            R, _ = cv2.Rodrigues(rvec)
            T_cam_aruco[0:3, 0:3], T_cam_aruco[0:3, 3] = R, tvec.flatten()
            return self.current_ee_pose_mat @ T_EE_TO_CAM @ T_cam_aruco
        return None

    def smooth_move_to_matrix(self, T_target, gripper_val=None):
        """ 融合了限加速度滤波的平滑移动函数 """
        rate = rospy.Rate(RATE_HZ)
        dt = 1.0 / RATE_HZ
        
        # 起始状态 (从实时反馈获取)
        T_start = self.current_ee_pose_mat.copy()
        start_xyz = T_start[0:3, 3]
        start_rpy = np.array(tf_trans.euler_from_matrix(T_start, 'sxyz'))
        
        # 目标状态
        target_xyz = T_target[0:3, 3]
        target_rpy = np.array(tf_trans.euler_from_matrix(T_target, 'sxyz'))
        
        # 规划变量
        current_xyz = start_xyz.copy()
        current_rpy = start_rpy.copy()
        v = 0.0  # 当前线速度
        
        rospy.loginfo("开始平滑移动...")
        
        while not rospy.is_shutdown():
            # 1. 计算当前距离目标的欧氏距离
            diff_xyz = target_xyz - current_xyz
            dist = np.linalg.norm(diff_xyz)
            
            # 2. 到达判定
            if dist < 0.001 and np.linalg.norm(target_rpy - current_rpy) < 0.01:
                break
                
            # 3. 速度滤波 (加速度限制逻辑)
            v_dir = diff_xyz / dist if dist > 0 else np.zeros(3)
            
            dv = V_TARGET - v
            dv_max = A_MAX * dt
            dv = max(min(dv, dv_max), -dv_max)
            v += dv
            
            # 比例减速
            v_exec = min(v, dist * 5.0) 
            
            # 4. 更新位置积分
            current_xyz += v_dir * v_exec * dt
            
            # 5. 姿态线性插值
            step_rot = ROT_SPEED * dt
            diff_rot = target_rpy - current_rpy
            for i in range(3):
                if abs(diff_rot[i]) > step_rot:
                    current_rpy[i] += np.sign(diff_rot[i]) * step_rot
                else:
                    current_rpy[i] = target_rpy[i]

            # 6. 发布指令
            cmd = PosCmd()
            cmd.x, cmd.y, cmd.z = current_xyz
            cmd.roll, cmd.pitch, cmd.yaw = current_rpy
            cmd.gripper = gripper_val if gripper_val is not None else 100
            cmd.mode1, cmd.mode2 = 1, 0
            self.pub.publish(cmd)
            
            rate.sleep()

    def execute_vision_grasp(self):
        rospy.loginfo(">>> 开始平滑视觉抓取任务")

        # 1. 移动到拍照位置 (平滑移动)
        T_init = tf_trans.euler_matrix(0, 1.57, 0, 'sxyz')
        T_init[0:3, 3] = [0.26, 0.0, 0.17]
        self.smooth_move_to_matrix(T_init, gripper_val=0)
        time.sleep(7.0)

        # 2. 识别二维码
        T_base_aruco_init = self.get_aruco_pose(TARGET_ARUCO_ID)
        if T_base_aruco_init is None: return

        # 3. 移动到二次识别位置
        T_pre = T_ARUCO_TO_GRASP.copy()
        T_pre[0:3, 3] = [0.0, -0.02, 0.16]
        self.smooth_move_to_matrix(T_base_aruco_init @ T_pre @ T_TOOL_TO_EE, gripper_val=100)
        time.sleep(5.0)
        
        # 4. 二次识别
        T_base_aruco = self.get_aruco_pose(TARGET_ARUCO_ID)
        if T_base_aruco is None:
            rospy.logerr("近距离丢失视野，使用第一次的数据继续...")
            T_base_aruco = T_base_aruco_init
        
        T_pre = T_ARUCO_TO_GRASP.copy()
        T_pre[0:3, 3] = [0.0, 0.0, 0.12]
        self.smooth_move_to_matrix(T_base_aruco @ T_pre @ T_TOOL_TO_EE, gripper_val=100)
        time.sleep(5.0)
        
        # 任务完成
        rospy.loginfo(">>> 视觉引导阶段结束")

    def run(self):
        try:
            self.execute_vision_grasp()
        finally:
            self.pipeline.stop()

if __name__ == "__main__":
    node = PiperVisionController()
    node.run()
