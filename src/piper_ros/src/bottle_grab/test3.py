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
# 1. 相机相对于机械臂末端的位姿 (手眼标定矩阵)
T_EE_TO_CAM = np.array([
    [-0.0482,  0.9987,  0.0147, -0.0699],
    [-0.9979, -0.0488,  0.0416,  0.0301],
    [ 0.0423, -0.0127,  0.9990,  0.0675],
    [ 0.0,     0.0,     0.0,     1.0]
])

# 2. 工具(夹爪)相对于机械臂末端的位姿：沿Z轴平移14厘米
T_EE_TO_TOOL = np.eye(4)
T_EE_TO_TOOL[2, 3] = 0.14  # 14cm
T_TOOL_TO_EE = np.linalg.inv(T_EE_TO_TOOL)

# 3. 抓取姿态相对于二维码的基准旋转
# 逻辑：绕 Z 轴旋转 -90 度 (-pi/2) -> 使 X 向下, Y 向右
R1 = tf_trans.rotation_matrix(-np.pi/2, [0, 0, 1])
# R2: 绕新的 X 轴旋转 180 度 (pi) -> 使夹爪 Z 轴反向对准二维码
R2 = tf_trans.rotation_matrix(np.pi, [1, 0, 0])
T_ARUCO_TO_GRASP = tf_trans.concatenate_matrices(R1, R2)


class PiperVisionController:
    def __init__(self):
        rospy.init_node("piper_vision_mission", anonymous=True)
        
        # 发布者
        self.pub = rospy.Publisher("/pin_pos_cmd", PosCmd, queue_size=1)
        rospy.sleep(0.5)

        # 记录当前机械臂的理论位姿 (初始默认单位阵，实际运行中会随发送更新)
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
        
        # ArUco 字典 (适配新版 API)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        rospy.loginfo("相机初始化完成！")

    def send_pose_matrix(self, T_matrix, gripper_val=0):
        x, y, z = T_matrix[0:3, 3]
        roll, pitch, yaw = tf_trans.euler_from_matrix(T_matrix, axes='sxyz')
        
        cmd = PosCmd()
        cmd.x, cmd.y, cmd.z = x, y, z
        cmd.roll, cmd.pitch, cmd.yaw = roll, pitch, yaw
        cmd.gripper = gripper_val
        cmd.mode1 = 1
        cmd.mode2 = 0
        
        self.pub.publish(cmd)
        self.current_ee_pose_mat = T_matrix.copy()
        rospy.loginfo(f"发送位姿 -> X:{x:.3f} Y:{y:.3f} Z:{z:.3f} | R:{roll:.2f} P:{pitch:.2f} Y:{yaw:.2f}")

    def get_aruco_pose(self, marker_id):
        rospy.loginfo("正在检测 ArUco 二维码...")
        # 尝试读取多帧确保图像稳定
        for _ in range(10):
            frames = self.pipeline.wait_for_frames()
            
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
            
        image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 检测
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, rejected = detector.detectMarkers(gray)
        
        if ids is not None and marker_id in ids:
            idx = np.where(ids == marker_id)[0][0]
            
            # 位姿估计 (SolvePnP)
            marker_points = np.array([[-ARUCO_REAL_SIZE_M / 2,  ARUCO_REAL_SIZE_M / 2, 0],
                                      [ ARUCO_REAL_SIZE_M / 2,  ARUCO_REAL_SIZE_M / 2, 0],
                                      [ ARUCO_REAL_SIZE_M / 2, -ARUCO_REAL_SIZE_M / 2, 0],
                                      [-ARUCO_REAL_SIZE_M / 2, -ARUCO_REAL_SIZE_M / 2, 0]], dtype=np.float32)
            _, rvec, tvec = cv2.solvePnP(marker_points, corners[idx][0], self.camera_matrix, self.dist_coeffs)
            
            # --- 绘图与弹窗逻辑 ---
            # 1. 框出二维码
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
            # 2. 标注坐标信息
            tx, ty, tz = tvec.flatten()
            info_str = f"ArUco ID {marker_id}: x={tx:.3f}, y={ty:.3f}, z={tz:.3f} (meters)"
            cv2.putText(image, info_str, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # 3. 显示弹窗 (保持 1 秒或直到按键)
            cv2.imshow("ArUco Detection", image)
            cv2.waitKey(1000) 
            # ---------------------

            # 构造变换矩阵
            T_cam_aruco = np.eye(4)
            R, _ = cv2.Rodrigues(rvec)
            T_cam_aruco[0:3, 0:3] = R
            T_cam_aruco[0:3, 3] = tvec.flatten()
            
            T_base_aruco = self.current_ee_pose_mat @ T_EE_TO_CAM @ T_cam_aruco
            return T_base_aruco
        else:
            cv2.imshow("ArUco Detection", image)
            cv2.waitKey(1)
            rospy.logwarn("视野内未发现目标二维码！")
            return None

    def execute_vision_grasp(self):
        rospy.loginfo(">>> 开始视觉引导抓取任务")

        # 1. 移动到预备拍照位置
        T_init = tf_trans.euler_matrix(0, 1.57, 0, axes='sxyz')
        T_init[0:3, 3] = [0.36, 0.0, 0.20]
        self.send_pose_matrix(T_init, gripper_val=0)
        time.sleep(4.0)

        # 2. 第一次远距离识别
        T_base_aruco_init = self.get_aruco_pose(TARGET_ARUCO_ID)
        if T_base_aruco_init is None: return

        # 3. 移动到近距离预备位 (距离二维码 15cm)
        T_pre = T_ARUCO_TO_GRASP.copy()
        T_pre[2, 3] = 0.15  # 在二维码正前方 15cm
        target_ee = T_base_aruco_init @ T_pre @ T_TOOL_TO_EE
        self.send_pose_matrix(target_ee, gripper_val=100)
        time.sleep(4.0)

        # 4. 第二次精准识别
        T_base_aruco = self.get_aruco_pose(TARGET_ARUCO_ID)
        if T_base_aruco is None: T_base_aruco = T_base_aruco_init

        # 5. 执行抓取序列
        # a. 接近点 (前方 5cm)
        T_t1 = T_ARUCO_TO_GRASP.copy()
        T_t1[2, 3] = 0.05
        self.send_pose_matrix(T_base_aruco @ T_t1 @ T_TOOL_TO_EE, gripper_val=100)
        time.sleep(2.0)

        # b. 抓取点 (深入 2.6cm)
        T_t2 = T_ARUCO_TO_GRASP.copy()
        T_t2[2, 3] = -0.026 
        self.send_pose_matrix(T_base_aruco @ T_t2 @ T_TOOL_TO_EE, gripper_val=100)
        time.sleep(2.0)

        # c. 闭合夹爪
        self.send_pose_matrix(self.current_ee_pose_mat, gripper_val=0)
        time.sleep(2.0)

        # d. 抬起 (向上 10cm)
        # 注意：这里的向上是相对于二维码坐标系的，如果二维码是竖直的，请根据需要调整
        T_up = T_t2.copy()
        T_up[2, 3] = 0.10 
        self.send_pose_matrix(T_base_aruco @ T_up @ T_TOOL_TO_EE, gripper_val=0)
        rospy.loginfo(">>> 任务完成！")

    def run(self):
        try:
            self.execute_vision_grasp()
        finally:
            cv2.destroyAllWindows()
            self.pipeline.stop()

if __name__ == "__main__":
    try:
        node = PiperVisionController()
        node.run()
    except rospy.ROSInterruptException:
        pass
