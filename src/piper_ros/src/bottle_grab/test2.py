#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import tf.transformations as tf_trans

from piper_msgs.msg import PosCmd

# ===============================
# 基本参数
# ===============================
ARUCO_REAL_SIZE_M = 0.03  # 二维码实际边长(米)
TARGET_ARUCO_ID = 6       # 目标二维码ID

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

# 2. 工具(夹爪)相对于机械臂末端的位姿：无旋转，沿Z轴平移14厘米
T_EE_TO_TOOL = np.eye(4)  # np.eye(4) 生成一个4x4的单位矩阵
# T_EE_TO_TOOL[0:3, 3] = [0.0, 0.0, 0.10]  # 分别对应 X, Y, Z 轴的平移，Z轴设置为 0.14 米
T_TOOL_TO_EE = np.linalg.inv(T_EE_TO_TOOL)  # 取逆

# 3. 抓取姿态相对于二维码的基准旋转
# R1: 绕 Z 轴旋转 -90 度 (-pi/2) -> 使 X, Y 轴映射对齐
R1 = tf_trans.rotation_matrix(-np.pi/2, [0, 0, 1])
# R2: 绕新的 X 轴旋转 180 度 (pi) -> 使 Z 轴反向对准二维码
R2 = tf_trans.rotation_matrix(np.pi, [1, 0, 0])
# 组合旋转 (注意顺序：先执行的变换写在右边，或者使用 concatenate 叠加)
T_ARUCO_TO_GRASP = tf_trans.concatenate_matrices(R1, R2)


class PiperVisionController:
    def __init__(self):
        rospy.init_node("piper_vision_mission", anonymous=True)
        
        # 发布者
        self.pub = rospy.Publisher("/pin_pos_cmd", PosCmd, queue_size=1)
        rospy.sleep(0.5)

        # 记录当前机械臂的理论位姿
        self.current_ee_pose_mat = np.eye(4)
        
        # 初始化 RealSense 相机
        self.init_camera()

    def init_camera(self):
        rospy.loginfo("正在初始化 RealSense 相机...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        profile = self.pipeline.start(config)
        
        # 获取相机内参
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.camera_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.zeros((4, 1))
        
        # ArUco 字典
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        rospy.loginfo("相机初始化完成！")

    def send_pose_matrix(self, T_matrix, gripper_val=0):
        """
        将 4x4 齐次变换矩阵转换为 PosCmd 并发布
        """
        # 1. 提取 XYZ 平移
        x, y, z = T_matrix[0:3, 3]
        
        # 2. 提取 Roll, Pitch, Yaw
        roll, pitch, yaw = tf_trans.euler_from_matrix(T_matrix, axes='sxyz')
        
        cmd = PosCmd()
        cmd.x, cmd.y, cmd.z = x, y, z
        cmd.roll, cmd.pitch, cmd.yaw = roll, pitch, yaw
        cmd.gripper = gripper_val
        cmd.mode1 = 1
        cmd.mode2 = 0
        
        self.pub.publish(cmd)
        
        # 更新当前位姿记录
        self.current_ee_pose_mat = T_matrix.copy()
        
        rospy.loginfo(f"发送位姿 -> X:{x:.3f} Y:{y:.3f} Z:{z:.3f} | R:{roll:.2f} P:{pitch:.2f} Y:{yaw:.2f}")

    def get_aruco_pose(self, marker_id):
        rospy.loginfo("正在检测 ArUco 二维码...")
        for _ in range(30):
            frames = self.pipeline.wait_for_frames()
            
        color_frame = frames.get_color_frame()
        if not color_frame:
            rospy.logerr("未获取到图像帧！")
            return None
            
        image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, rejected = detector.detectMarkers(gray)
        
        if ids is not None and marker_id in ids:
            idx = np.where(ids == marker_id)[0][0]
            
            marker_points = np.array([[-ARUCO_REAL_SIZE_M / 2,  ARUCO_REAL_SIZE_M / 2, 0],
                                      [ ARUCO_REAL_SIZE_M / 2,  ARUCO_REAL_SIZE_M / 2, 0],
                                      [ ARUCO_REAL_SIZE_M / 2, -ARUCO_REAL_SIZE_M / 2, 0],
                                      [-ARUCO_REAL_SIZE_M / 2, -ARUCO_REAL_SIZE_M / 2, 0]], dtype=np.float32)
            trash, rvec, tvec = cv2.solvePnP(marker_points, corners[idx][0], self.camera_matrix, self.dist_coeffs)
            
            T_cam_aruco = np.eye(4)
            R, _ = cv2.Rodrigues(rvec)
            T_cam_aruco[0:3, 0:3] = R
            T_cam_aruco[0:3, 3] = tvec[0][0]
            
            T_base_ee = self.current_ee_pose_mat
            T_base_aruco = T_base_ee @ T_EE_TO_CAM @ T_cam_aruco
            
            rospy.loginfo(f"检测到 ID:{marker_id}，计算基坐标位姿完成。")
            return T_base_aruco
        else:
            rospy.logwarn("视野内未发现目标二维码！")
            return None

    def execute_vision_grasp(self):
        """
        近距离二次识别
        """
        rospy.loginfo(">>> 开始视觉引导抓取任务")

        # 1. 移动到预备拍照位置 (固定位姿，类似你给的 baseline 代码)
        T_init = tf_trans.euler_matrix(0, 1.57, 0, axes='sxyz')
        T_init[0:3, 3] = [0.26, 0.0, 0.20]  # Z 抬高一点用于宏观拍照
        self.send_pose_matrix(T_init, gripper_val=0)
        time.sleep(6.0)

        # 2. 第一次远距离识别
        T_base_aruco_init = self.get_aruco_pose(TARGET_ARUCO_ID)
        if T_base_aruco_init is None:
            return
        
        rospy.loginfo("移动到二维码的近距离预备位 (准备二次识别)")
        T_pre = T_ARUCO_TO_GRASP.copy()
        T_pre[0:3, 3] = [0.0, -0.05, 0]
        # 计算机械臂末端目标位姿并发送
        target_ee = T_base_aruco_init @ T_pre @ T_TOOL_TO_EE
        self.send_pose_matrix(target_ee, gripper_val=100)
        time.sleep(5.0)
        
        # 3. 第二次近距离精准识别
        T_base_aruco = self.get_aruco_pose(TARGET_ARUCO_ID)
        if T_base_aruco is None:
            rospy.logerr("近距离丢失视野，使用第一次的数据继续...")
            T_base_aruco = T_base_aruco_init

        # T_base_aruco = T_base_aruco_init
        
        # 4. 执行抓取序列
        # a. 移动到抓取前方
        T_t1 = T_ARUCO_TO_GRASP.copy()
        T_t1[0:3, 3] = [0.0, 0.0, 0.05]
        self.send_pose_matrix(T_base_aruco @ T_t1 @ T_TOOL_TO_EE, gripper_val=100)
        time.sleep(2.5)
        
        # b. 准备抓取
        T_t2 = T_ARUCO_TO_GRASP.copy()
        T_t2[0:3, 3] = [0, 0, 0.0] 
        self.send_pose_matrix(T_base_aruco @ T_t2 @ T_TOOL_TO_EE, gripper_val=100)
        time.sleep(2.5)
        
        # c. 闭合夹爪
        rospy.loginfo("闭合夹爪")
        self.send_pose_matrix(T_base_aruco @ T_t2 @ T_TOOL_TO_EE, gripper_val=0)
        time.sleep(2.0)
        '''
        # d. 上抬
        rospy.loginfo("将目标上抬")
        T_up = T_ARUCO_TO_GRASP.copy()
        T_up[0:3, 3] = [0, 0.15, -0.05]
        self.send_pose_matrix(T_base_aruco @ T_up @ T_TOOL_TO_EE, gripper_val=0)
        time.sleep(3.0)'''
        '''
        rospy.loginfo(">>> 视觉引导抓取任务完成！")
        '''

    def run(self):
        try:
            self.execute_vision_grasp()
        finally:
            self.pipeline.stop()

if __name__ == "__main__":
    try:
        node = PiperVisionController()
        node.run()
    except rospy.ROSInterruptException:
        pass
