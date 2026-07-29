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
from sensor_msgs.msg import JointState  # 新增：导入 JointState 消息类型

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
T_EE_TO_TOOL[0:3, 3] = [0.0, 0.0, -0.14]

# 3. 抓取姿态相对于二维码的基准旋转
R1 = tf_trans.rotation_matrix(-np.pi/2, [0, 0, 1])
R2 = tf_trans.rotation_matrix(np.pi, [1, 0, 0])
T_ARUCO_TO_GRASP = tf_trans.concatenate_matrices(R1, R2)

class PiperVisionController:
    def __init__(self):
        rospy.init_node("piper_vision_smooth_mission", anonymous=True)
        
        # 发布者
        self.pub = rospy.Publisher("/pin_pos_cmd", PosCmd, queue_size=1)
        # 新增：用于发布关节状态的 Publisher
        self.joint_pub = rospy.Publisher("/joint_states", JointState, queue_size=1) 
        
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
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

    def move_to_target_smooth(self, T_target, v=0.05, rate_hz=100, gripper_val=0):
        """ 平滑移动：线性插值 (Llerp + Slerp) """
        rate = rospy.Rate(rate_hz)
        T_start = self.current_ee_pose_mat.copy()
        start_pos = T_start[0:3, 3]
        target_pos = T_target[0:3, 3]
        
        dist = np.linalg.norm(target_pos - start_pos)
        if dist < 0.001: return

        duration = dist / v
        steps = int(duration * rate_hz)
        
        q_start = tf_trans.quaternion_from_matrix(T_start)
        q_target = tf_trans.quaternion_from_matrix(T_target)

        for i in range(1, steps + 1):
            if rospy.is_shutdown(): break
            alpha = i / float(steps)
            curr_pos = start_pos + (target_pos - start_pos) * alpha
            curr_q = tf_trans.quaternion_slerp(q_start, q_target, alpha)
            
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

    def get_aruco_pose(self, marker_id):
        """ 检测二维码并打印其在基座坐标系下的 XYZ """
        rospy.loginfo(f"正在尝试检测 ID 为 {marker_id} 的二维码...")
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
                
                # 相机系下的位姿
                T_cam_aruco = np.eye(4)
                R, _ = cv2.Rodrigues(rvec)
                T_cam_aruco[0:3, 0:3] = R
                T_cam_aruco[0:3,3] = tvec.flatten()
                
                # 转换到基座坐标系
                T_base_aruco = self.current_ee_pose_mat @ T_EE_TO_CAM @ T_cam_aruco
                
                # --- 新增打印部分 ---
                aruco_cam_x = T_cam_aruco[0, 3]
                aruco_cam_y = T_cam_aruco[1, 3]
                aruco_cam_z = T_cam_aruco[2, 3]
                rospy.loginfo("========================================")
                rospy.loginfo(f"相机坐标系下的位置 (XYZ):")
                rospy.loginfo(f"  X: {aruco_cam_x:.4f} m")
                rospy.loginfo(f"  Y: {aruco_cam_y:.4f} m")
                rospy.loginfo(f"  Z: {aruco_cam_z:.4f} m")
                rospy.loginfo("========================================")

                # ------------------
                aruco_base_x = T_base_aruco[0, 3]
                aruco_base_y = T_base_aruco[1, 3]
                aruco_base_z = T_base_aruco[2, 3]
                rospy.loginfo("========================================")
                rospy.loginfo(f"检测成功！二维码 ID: {marker_id}")
                rospy.loginfo(f"基座坐标系下的位置 (XYZ):")
                rospy.loginfo(f"  X: {aruco_base_x:.4f} m")
                rospy.loginfo(f"  Y: {aruco_base_y:.4f} m")
                rospy.loginfo(f"  Z: {aruco_base_z:.4f} m")
                rospy.loginfo("========================================")
                
                return T_base_aruco
        
        rospy.logwarn(f"超时：未检测到 ID 为 {marker_id} 的二维码")
        return None

    def execute_vision_grasp(self):
        rospy.loginfo(">>> 开始平滑视觉抓取任务")

        # 1. 移动到拍照位置 (修改为发布 JointState)
        joint_msg = JointState()
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.name = ['']
        joint_msg.position = [0.2, 0.2, -0.2, 0.3, -0.2, 0.5, 0.0]
        joint_msg.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0]
        joint_msg.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
        
        self.joint_pub.publish(joint_msg)
        time.sleep(6.0) # 短暂稳定图像

        # 2. 第一次识别并打印位置
        T_base_aruco = self.get_aruco_pose(TARGET_ARUCO_ID)
        if T_base_aruco is None: return
        
        # 3. 移动到预抓取位
        T_pre = T_ARUCO_TO_GRASP.copy()
        T_pre[0:3, 3] = [0.0, -0.03, 0.15]
        target_ee_pre = T_base_aruco @ T_pre @ T_EE_TO_TOOL
        self.move_to_target_smooth(target_ee_pre, v=0.06, gripper_val=100)
        time.sleep(3.5)

        # 4. 第二次识别（近距离更精准）并打印位置
        T_base_aruco_fine = self.get_aruco_pose(TARGET_ARUCO_ID)
        if T_base_aruco_fine is not None:
            T_base_aruco = T_base_aruco_fine

        # 5. 执行抓取序列
        # a. 移动到抓取点上方
        T_t1 = T_ARUCO_TO_GRASP.copy()
        T_t1[0:3, 3] = [0.0, 0.0, 0.07]
        self.move_to_target_smooth(T_base_aruco @ T_t1 @ T_EE_TO_TOOL, v=0.05, gripper_val=100)
        time.sleep(2.5)

        # b. 触碰抓取点 (慢速下降)
        T_t2 = T_ARUCO_TO_GRASP.copy()
        T_t2[0:3, 3] = [0, 0.0, -0.04] 
        self.move_to_target_smooth(T_base_aruco @ T_t2 @ T_EE_TO_TOOL, v=0.02, gripper_val=100)
        time.sleep(2.5)
        
        # c. 闭合夹爪
        rospy.loginfo("正在闭合夹爪...")
        curr_pose = self.current_ee_pose_mat
        r, p, yaw = tf_trans.euler_from_matrix(curr_pose, axes='sxyz')
        cmd_close = PosCmd()
        cmd_close.x, cmd_close.y, cmd_close.z = curr_pose[0:3, 3]
        cmd_close.roll, cmd_close.pitch, cmd_close.yaw = r, p, yaw
        cmd_close.gripper = 0 
        cmd_close.mode1 = 1
        self.pub.publish(cmd_close)
        time.sleep(2.0)

        # d. 提升
        T_up = T_ARUCO_TO_GRASP.copy()
        T_up[0:3, 3] = [0, 0.15, 0.07]
        self.move_to_target_smooth(T_base_aruco @ T_up @ T_EE_TO_TOOL, v=0.05, gripper_val=0)

        rospy.loginfo(">>> 视觉抓取任务结束")
        
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
