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
# 参数
# ===============================

ARUCO_REAL_SIZE_M = 0.03
TARGET_ID = 9


# ===============================
# 手眼标定矩阵
# ===============================

T_EE_TO_CAM = np.array([
    [-0.0482,  0.9987,  0.0147, -0.0699],
    [-0.9979, -0.0488,  0.0416,  0.0301],
    [ 0.0423, -0.0127,  0.9990,  0.0675],
    [0.0, 0.0, 0.0, 1.0]
])


# ===============================
# ArUco -> 抓取姿态
# ===============================

T_ARUCO_TO_GRASP = tf_trans.rotation_matrix(np.pi,[1,0,0])


class VisionPiperNode():

    def __init__(self):

        rospy.init_node("vision_piper_node")

        # 发布机械臂控制
        self.pub = rospy.Publisher("/pin_pos_cmd",PosCmd,queue_size=1)

        # 订阅末端位姿
        rospy.Subscriber("/end_pose", PoseStamped, self.pose_callback)

        self.current_ee_pose = None

        # ===============================
        # RealSense初始化
        # ===============================

        self.pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)

        profile = self.pipeline.start(config)

        intr = profile.get_stream(rs.stream.color)\
            .as_video_stream_profile().get_intrinsics()

        self.cam_mtx = np.array([
            [intr.fx,0,intr.ppx],
            [0,intr.fy,intr.ppy],
            [0,0,1]
        ])

        # ===============================
        # ArUco
        # ===============================

        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        )

        s = ARUCO_REAL_SIZE_M

        self.obj_pts = np.array([
            [-s/2, s/2, 0],
            [ s/2, s/2, 0],
            [ s/2,-s/2, 0],
            [-s/2,-s/2, 0]
        ],dtype=np.float64)

        rospy.loginfo("视觉系统初始化完成")


    # ===============================
    # 末端位姿回调
    # ===============================

    def pose_callback(self,msg):

        self.current_ee_pose = msg.pose


    # ===============================
    # 识别二维码
    # ===============================

    def get_aruco_pose(self):

        while not rospy.is_shutdown():

            frames = self.pipeline.wait_for_frames()

            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            corners,ids,_ = self.detector.detectMarkers(gray)

            if ids is not None and TARGET_ID in ids and self.current_ee_pose:

                idx = np.where(ids==TARGET_ID)[0][0]

                ok,rvec,tvec = cv2.solvePnP(
                    self.obj_pts,
                    corners[idx][0].astype(np.float64),
                    self.cam_mtx,
                    np.zeros(5)
                )

                if ok:

                    R_cam,_ = cv2.Rodrigues(rvec)

                    T_cam_aruco = np.eye(4)
                    T_cam_aruco[:3,:3] = R_cam
                    T_cam_aruco[:3,3] = tvec.flatten()

                    # ===============================
                    # 末端位姿 -> 齐次矩阵
                    # ===============================

                    q = [
                        self.current_ee_pose.orientation.x,
                        self.current_ee_pose.orientation.y,
                        self.current_ee_pose.orientation.z,
                        self.current_ee_pose.orientation.w
                    ]

                    T_base_ee = tf_trans.quaternion_matrix(q)

                    T_base_ee[0:3,3] = [
                        self.current_ee_pose.position.x,
                        self.current_ee_pose.position.y,
                        self.current_ee_pose.position.z
                    ]

                    # ===============================
                    # 计算二维码世界坐标
                    # ===============================

                    T_base_aruco = T_base_ee @ T_EE_TO_CAM @ T_cam_aruco

                    return T_base_aruco,img,corners,ids

            return None,img,corners,ids


    # ===============================
    # 发送机械臂指令
    # ===============================

    def send_arm_cmd(self,T):

        cmd = PosCmd()

        cmd.x = T[0,3]
        cmd.y = T[1,3]
        cmd.z = T[2,3]

        roll,pitch,yaw = tf_trans.euler_from_matrix(T)

        cmd.roll = roll
        cmd.pitch = pitch
        cmd.yaw = yaw

        cmd.gripper = 0
        cmd.mode1 = 1
        cmd.mode2 = 0

        self.pub.publish(cmd)


    # ===============================
    # 主循环
    # ===============================

    def run(self):

        cv2.namedWindow("camera")

        while not rospy.is_shutdown():

            T_base_aruco,img,corners,ids = self.get_aruco_pose()

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(img,corners,ids)

            if T_base_aruco is not None:

                # 抓取姿态
                T_target = T_base_aruco @ T_ARUCO_TO_GRASP

                rospy.loginfo(
                    f"目标位置: {T_target[0,3]:.3f} "
                    f"{T_target[1,3]:.3f} "
                    f"{T_target[2,3]:.3f}"
                )

                self.send_arm_cmd(T_target)

                rospy.sleep(2)

            cv2.imshow("camera",img)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    node = VisionPiperNode()

    rospy.sleep(1)

    node.run()
