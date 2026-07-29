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
ARUCO_REAL_SIZE_M = 0.03
TARGET_ARUCO_ID = 6

# ===============================
# 位姿矩阵
# ===============================

T_EE_TO_CAM = np.array([
    [-0.0482,  0.9987,  0.0147, -0.0699],
    [-0.9979, -0.0488,  0.0416,  0.0301],
    [ 0.0423, -0.0127,  0.9990,  0.0675],
    [ 0.0,     0.0,     0.0,     1.0]
])

T_EE_TO_TOOL = np.eye(4)
T_EE_TO_TOOL[0:3,3] = [0,0,0.10]
T_TOOL_TO_EE = np.linalg.inv(T_EE_TO_TOOL)

R1 = tf_trans.rotation_matrix(-np.pi/2,[0,0,1])
R2 = tf_trans.rotation_matrix(np.pi,[1,0,0])
T_ARUCO_TO_GRASP = tf_trans.concatenate_matrices(R1,R2)


class PiperVisionController:

    def __init__(self):

        rospy.init_node("piper_vision_mission")

        self.pub = rospy.Publisher("/pin_pos_cmd",PosCmd,queue_size=1)

        self.current_ee_pose_mat = np.eye(4)

        self.init_camera()


    # ===============================
    # 初始化相机
    # ===============================
    def init_camera(self):

        self.pipeline = rs.pipeline()

        config = rs.config()

        config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)

        profile = self.pipeline.start(config)

        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        self.camera_matrix = np.array([
            [intrinsics.fx,0,intrinsics.ppx],
            [0,intrinsics.fy,intrinsics.ppy],
            [0,0,1]
        ])

        self.dist_coeffs = np.zeros((4,1))

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

        self.aruco_params = cv2.aruco.DetectorParameters()


    # ===============================
    # 发送位姿
    # ===============================
    def send_pose_matrix(self,T_matrix,gripper_val=0):

        x,y,z = T_matrix[0:3,3]

        roll,pitch,yaw = tf_trans.euler_from_matrix(T_matrix)

        cmd = PosCmd()

        cmd.x = x
        cmd.y = y
        cmd.z = z

        cmd.roll = roll
        cmd.pitch = pitch
        cmd.yaw = yaw

        cmd.gripper = gripper_val

        cmd.mode1 = 1
        cmd.mode2 = 0

        self.pub.publish(cmd)

        self.current_ee_pose_mat = T_matrix.copy()


    # ===============================
    # 颜色识别
    # ===============================
    def detect_bottle_color(self,image):

        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

        # 橙色
        lower_orange = np.array([5,120,120])
        upper_orange = np.array([20,255,255])

        # 绿色
        lower_green = np.array([40,80,80])
        upper_green = np.array([85,255,255])

        # 黑色
        lower_black = np.array([0,0,0])
        upper_black = np.array([180,255,60])

        mask_orange = cv2.inRange(hsv,lower_orange,upper_orange)
        mask_green = cv2.inRange(hsv,lower_green,upper_green)
        mask_black = cv2.inRange(hsv,lower_black,upper_black)

        masks = {
            "orange":mask_orange,
            "green":mask_green,
            "black":mask_black
        }

        best_color = None
        best_area = 0
        best_bbox = None

        for color,mask in masks.items():

            contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:

                area = cv2.contourArea(cnt)

                if area > best_area and area > 2000:

                    x,y,w,h = cv2.boundingRect(cnt)

                    best_area = area
                    best_color = color
                    best_bbox = (x,y,w,h)

        return best_color,best_bbox


    # ===============================
    # ROI二维码识别
    # ===============================
    def detect_aruco_in_roi(self,image,bbox):

        x,y,w,h = bbox

        roi = image[y:y+h,x:x+w]

        gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)

        detector = cv2.aruco.ArucoDetector(self.aruco_dict,self.aruco_params)

        corners,ids,_ = detector.detectMarkers(gray)

        if ids is None:
            return None

        idx = 0

        marker_points = np.array([
            [-ARUCO_REAL_SIZE_M/2, ARUCO_REAL_SIZE_M/2,0],
            [ ARUCO_REAL_SIZE_M/2, ARUCO_REAL_SIZE_M/2,0],
            [ ARUCO_REAL_SIZE_M/2,-ARUCO_REAL_SIZE_M/2,0],
            [-ARUCO_REAL_SIZE_M/2,-ARUCO_REAL_SIZE_M/2,0]
        ],dtype=np.float32)

        ok,rvec,tvec = cv2.solvePnP(marker_points,corners[idx][0],self.camera_matrix,self.dist_coeffs)

        T_cam_aruco = np.eye(4)

        R,_ = cv2.Rodrigues(rvec)

        T_cam_aruco[0:3,0:3] = R
        T_cam_aruco[0:3,3] = tvec[:,0]

        T_base_ee = self.current_ee_pose_mat

        T_base_aruco = T_base_ee @ T_EE_TO_CAM @ T_cam_aruco

        return T_base_aruco


    # ===============================
    # 视觉识别
    # ===============================
    def get_target_pose(self):

        for _ in range(30):

            frames = self.pipeline.wait_for_frames()

        frame = frames.get_color_frame()

        if not frame:
            return None

        image = np.asanyarray(frame.get_data())

        color,bbox = self.detect_bottle_color(image)

        if color is None:

            rospy.logwarn("未检测到瓶子颜色")

            return None

        rospy.loginfo(f"检测到颜色: {color}")

        T_base_aruco = self.detect_aruco_in_roi(image,bbox)

        return T_base_aruco


    # ===============================
    # 执行抓取
    # ===============================
    def execute_vision_grasp(self):

        T_init = tf_trans.euler_matrix(0,1.57,0)

        T_init[0:3,3] = [0.26,0,0.20]

        self.send_pose_matrix(T_init,0)

        time.sleep(5)

        T_base_aruco = self.get_target_pose()

        if T_base_aruco is None:

            rospy.logerr("未识别到目标")

            return


        T_t1 = T_ARUCO_TO_GRASP.copy()

        T_t1[0:3,3] = [0,0,0]

        self.send_pose_matrix(T_base_aruco @ T_t1 @ T_TOOL_TO_EE,100)

        time.sleep(3)


        T_t2 = T_ARUCO_TO_GRASP.copy()

        T_t2[0:3,3] = [0,0,-0.20]

        self.send_pose_matrix(T_base_aruco @ T_t2 @ T_TOOL_TO_EE,100)

        time.sleep(3)


        self.send_pose_matrix(T_base_aruco @ T_t2 @ T_TOOL_TO_EE,0)

        time.sleep(2)


        T_up = T_ARUCO_TO_GRASP.copy()

        T_up[0:3,3] = [0,0.15,-0.05]

        self.send_pose_matrix(T_base_aruco @ T_up @ T_TOOL_TO_EE,0)

        time.sleep(3)


    def run(self):

        try:

            self.execute_vision_grasp()

        finally:

            self.pipeline.stop()


if __name__ == "__main__":

    node = PiperVisionController()

    node.run()
