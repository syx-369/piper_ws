#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
import pyrealsense2 as rs
import tf.transformations as tf_trans
from geometry_msgs.msg import PoseStamped

# ===============================
# 参数配置 (需与主程序一致)
# ===============================
ARUCO_REAL_SIZE_M = 0.03
TARGET_ARUCO_ID = 6

T_EE_TO_CAM = np.array([
    [-0.0482,  0.9987,  0.0147, -0.0699],
    [-0.9979, -0.0488,  0.0416,  0.0301],
    [ 0.0423, -0.0127,  0.9990,  0.0675],
    [ 0.0,     0.0,     0.0,     1.0]
])

class ConsistencyChecker:
    def __init__(self):
        rospy.init_node("vision_consistency_checker")
        
        self.current_ee_pose = np.eye(4)
        self.ref_aruco_pose = None
        
        # 订阅位姿
        rospy.Subscriber("/end_pose", PoseStamped, self.pose_callback)
        
        # 初始化相机
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

    def pose_callback(self, msg):
        q = [msg.pose.orientation.x, msg.pose.orientation.y, 
             msg.pose.orientation.z, msg.pose.orientation.w]
        self.current_ee_pose = tf_trans.quaternion_matrix(q)
        self.current_ee_pose[0:3, 3] = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]

    def get_aruco_in_base(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame: return None
        
        img = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and TARGET_ARUCO_ID in ids:
            idx = np.where(ids == TARGET_ARUCO_ID)[0][0]
            m_pts = np.array([[-ARUCO_REAL_SIZE_M/2,  ARUCO_REAL_SIZE_M/2, 0],
                              [ ARUCO_REAL_SIZE_M/2,  ARUCO_REAL_SIZE_M/2, 0],
                              [ ARUCO_REAL_SIZE_M/2, -ARUCO_REAL_SIZE_M/2, 0],
                              [-ARUCO_REAL_SIZE_M/2, -ARUCO_REAL_SIZE_M/2, 0]], dtype=np.float32)
            _, rvec, tvec = cv2.solvePnP(m_pts, corners[idx][0], self.camera_matrix, self.dist_coeffs)
            
            T_c_a = np.eye(4)
            R, _ = cv2.Rodrigues(rvec)
            T_c_a[0:3, 0:3] = R
            T_c_a[0:3, 3] = tvec[0][0]
            
            # 返回基座下的位姿
            return self.current_ee_pose @ T_EE_TO_CAM @ T_c_a
        return None

    def run(self):
        print("\n=== 位移一致性检查工具 ===")
        print("1. 请将二维码固定在桌面。")
        print("2. 移动机械臂使相机对准二维码。")
        print("3. 按 's' 键设置当前位置为基准点。")
        print("4. 手动微调机械臂位置，观察『视觉测量位移』。")
        print("5. 按 'q' 退出。\n")

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            t_base_aruco = self.get_aruco_in_base()
            
            if t_base_aruco is not None:
                cur_pos = t_base_aruco[0:3, 3]
                
                if self.ref_aruco_pose is None:
                    print(f"\r当前检测到二维码 XYZ: {cur_pos} | 等待按 's' 锁定基准...", end="")
                else:
                    # 计算相对于基准点的位移
                    diff = cur_pos - self.ref_aruco_pose
                    print(f"\r实时位移(mm) -> ΔX: {diff[0]*1000:6.1f}, ΔY: {diff[1]*1000:6.1f}, ΔZ: {diff[2]*1000:6.1f}", end="")
            else:
                print("\r[!] 未检测到二维码...                           ", end="")

            # 键盘交互
            key = cv2.waitKey(1) & 0xFF
            # 注意：在没有OpenCV窗口时，可以使用Python的input或者检测终端按键，这里为了简便建议运行后焦点在控制台按s/q
            # 如果是在终端运行，可以改用下面这个简单的逻辑：
            import sys, select
            if select.select([sys.stdin], [], [], 0)[0] == [sys.stdin]:
                cmd = sys.stdin.readline().strip()
                if cmd == 's':
                    if t_base_aruco is not None:
                        self.ref_aruco_pose = t_base_aruco[0:3, 3].copy()
                        print(f"\n[OK] 基准点已锁定: {self.ref_aruco_pose}")
                elif cmd == 'q':
                    break
            
            rate.sleep()

if __name__ == "__main__":
    try:
        checker = ConsistencyChecker()
        checker.run()
    except Exception as e:
        print(f"\n程序结束: {e}")
