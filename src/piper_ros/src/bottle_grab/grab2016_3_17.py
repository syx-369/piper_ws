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

# 2. 工具(夹爪)相对于机械臂末端的位姿
T_EE_TO_TOOL = np.eye(4)
T_EE_TO_TOOL[0:3, 3] = [0.0, 0.0, 0.14]

# 3. 抓取姿态相对于二维码的基准旋转
R1 = tf_trans.rotation_matrix(-np.pi/2, [0, 0, 1])
R2 = tf_trans.rotation_matrix(np.pi, [1, 0, 0])
T_ARUCO_TO_GRASP = tf_trans.concatenate_matrices(R1, R2)

class PiperVisionController:
    def __init__(self):
        rospy.init_node("piper_vision_smooth_mission", anonymous=True)
        
        # 发布者
        self.pub = rospy.Publisher("/pin_pos_cmd", PosCmd, queue_size=1)
        
        # --- 实时位姿反馈变量初始化 ---
        self.current_ee_pose_mat = np.eye(4)
        self.pose_received = False
        
        # 订阅末端实时位姿
        rospy.Subscriber("/end_pose", PoseStamped, self.pose_callback)
        
        rospy.loginfo("等待获取机械臂实时位姿 (/end_pose)...")
        while not self.pose_received and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("成功接入实时位姿反馈！")
        
        # 初始化相机
        self.init_camera()

    def pose_callback(self, msg):
        px = msg.pose.position.x
        py = msg.pose.position.y
        pz = msg.pose.position.z
        q = [msg.pose.orientation.x, msg.pose.orientation.y, 
             msg.pose.orientation.z, msg.pose.orientation.w]
        T_mat = tf_trans.quaternion_matrix(q)
        T_mat[0:3, 3] = [px, py, pz]
        self.current_ee_pose_mat = T_mat
        self.pose_received = True

    def init_camera(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.camera_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.zeros((4, 1))
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.aruco_params = cv2.aruco.DetectorParameters()

    def move_to_target_smooth(self, T_target, v=0.05, rate_hz=100, gripper_val=0):
        """
        平滑移动核心：从当前位姿线性插值到目标位姿
        v: 移动速度 (m/s)
        rate_hz: 发布频率
        """
        rate = rospy.Rate(rate_hz)
        dt = 1.0 / rate_hz

        # 1. 获取起始位姿
        T_start = self.current_ee_pose_mat.copy()
        start_pos = T_start[0:3, 3]
        target_pos = T_target[0:3, 3]
        
        # 2. 计算距离和所需时间
        dist = np.linalg.norm(target_pos - start_pos)
        if dist < 0.001: 
            return # 距离太近无需平滑移动

        duration = dist / v
        steps = int(duration * rate_hz)
        
        # 3. 姿态线性插值准备 (四元数插值)
        q_start = tf_trans.quaternion_from_matrix(T_start)
        q_target = tf_trans.quaternion_from_matrix(T_target)

        rospy.loginfo(f"开始平滑移动：距离 {dist:.3f}m, 预计耗时 {duration:.2f}s")

        for i in range(1, steps + 1):
            if rospy.is_shutdown(): break
            
            alpha = i / float(steps) # 进度系数 0~1
            
            # 位置线性插值
            curr_pos = start_pos + (target_pos - start_pos) * alpha
            # 姿态球面线性插值 (Slerp)
            curr_q = tf_trans.quaternion_slerp(q_start, q_target, alpha)
            
            # 转换为 RPY 发送
            T_interp = tf_trans.quaternion_matrix(curr_q)
            roll, pitch, yaw = tf_trans.euler_from_matrix(T_interp, axes='sxyz')

            cmd = PosCmd()
            cmd.x, cmd.y, cmd.z = curr_pos
            cmd.roll, cmd.pitch, cmd.yaw = roll, pitch, yaw
            cmd.gripper = gripper_val
            cmd.mode1 = 1
            cmd.mode2 = 0
            
            self.pub.publish(cmd)
            rate.sleep()
            
        rospy.loginfo("到达目标点")

    def get_aruco_pose(self, marker_id):
        # ... (此处逻辑保持不变)
        for _ in range(30):
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: continue
            image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            corners, ids, _ = detector.detectMarkers(gray)
            
            if ids is not None and marker_id in ids:
                idx = np.where(ids == marker_id)[0][0]
                marker_points = np.array([[-ARUCO_REAL_SIZE_M/2,  ARUCO_REAL_SIZE_M/2, 0],
                                          [ ARUCO_REAL_SIZE_M/2,  ARUCO_REAL_SIZE_M/2, 0],
                                          [ ARUCO_REAL_SIZE_M/2, -ARUCO_REAL_SIZE_M/2, 0],
                                          [-ARUCO_REAL_SIZE_M/2, -ARUCO_REAL_SIZE_M/2, 0]], dtype=np.float32)
                _, rvec, tvec = cv2.solvePnP(marker_points, corners[idx][0], self.camera_matrix, self.dist_coeffs)
                T_cam_aruco = np.eye(4)
                R, _ = cv2.Rodrigues(rvec)
                T_cam_aruco[0:3, 0:3] = R
                T_cam_aruco[0:3, 3] = tvec[0][0]
                return self.current_ee_pose_mat @ T_EE_TO_CAM @ T_cam_aruco
        return None

    def execute_vision_grasp(self):
        rospy.loginfo(">>> 开始平滑视觉抓取任务")

        # 1. 移动到拍照位置 (平滑移动)
        T_init = tf_trans.euler_matrix(0, 1.57, 0, axes='sxyz')
        T_init[0:3, 3] = [0.26, 0.0, 0.23]
        self.move_to_target_smooth(T_init, v=0.08, gripper_val=0)
        time.sleep(5.0)

        # 2. 识别二维码
        T_base_aruco = self.get_aruco_pose(TARGET_ARUCO_ID)
        if T_base_aruco is None:
            rospy.logerr("未发现二维码")
            return

        # 3. 移动到预抓取位 (平滑移动)
        T_pre = T_ARUCO_TO_GRASP.copy()
        T_pre[0:3, 3] = [0.0, -0.06, 0.18]
        target_ee_pre = T_base_aruco @ T_pre @ T_EE_TO_TOOL
        self.move_to_target_smooth(target_ee_pre, v=0.06, gripper_val=100)
        time.sleep(3.5)

        # 4. 再次识别以修正误差 (可选)
        T_base_aruco_fine = self.get_aruco_pose(TARGET_ARUCO_ID)
        if T_base_aruco_fine is not None:
            T_base_aruco = T_base_aruco_fine

        # 5. 执行抓取序列
        # a. 移动到抓取点上方
        T_t1 = T_ARUCO_TO_GRASP.copy()
        T_t1[0:3, 3] = [0.0, 0.06, 0.12]
        self.move_to_target_smooth(T_base_aruco @ T_t1 @ T_EE_TO_TOOL, v=0.05, gripper_val=100)
        time.sleep(3.5)

        # b. 触碰抓取点 (慢速下降)
        T_t2 = T_ARUCO_TO_GRASP.copy()
        T_t2[0:3, 3] = [0, 0.06, 0.07] 
        self.move_to_target_smooth(T_base_aruco @ T_t2 @ T_EE_TO_TOOL, v=0.02, gripper_val=100)
        time.sleep(3.5)
        
        # c. 闭合夹爪
        rospy.loginfo("正在抓取...")
        cmd_close = PosCmd()
        # 保持当前位置不变，只改夹爪
        curr_pose = self.current_ee_pose_mat
        x, y, z = curr_pose[0:3, 3]
        r, p, yaw = tf_trans.euler_from_matrix(curr_pose, axes='sxyz')
        cmd_close.x, cmd_close.y, cmd_close.z = x, y, z
        cmd_close.roll, cmd_close.pitch, cmd_close.yaw = r, p, yaw
        cmd_close.gripper = 0 # 闭合
        cmd_close.mode1 = 1
        self.pub.publish(cmd_close)
        time.sleep(2.0)

        # d. 提升
        T_up = T_ARUCO_TO_GRASP.copy()
        T_up[0:3, 3] = [0, 0.15, 0.07]
        self.move_to_target_smooth(T_base_aruco @ T_up @ T_EE_TO_TOOL, v=0.05, gripper_val=0)

        rospy.loginfo(">>> 任务完成！")

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
