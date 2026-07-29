#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import socket
import threading
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import tf.transformations as tf_trans
import serial
import struct

from rm_msgs.msg import MoveJ, MoveJ_P
from geometry_msgs.msg import Pose

# ===============================
# 基本参数
# ===============================
ARUCO_REAL_SIZE_M = 0.03

HANDER_IP   = "192.168.144.11"
HANDER_PORT = 10030

DOG_IP   = "192.168.144.100"
DOG_PORT = 10020

ARM_SRC_ID = 0x01

# ===============================
# 位姿矩阵
# ===============================
T_EE_TO_CAM = np.array([
    [0.88111357,  0.3229869,  -0.34542487, -0.06411812],
    [-0.46729541, 0.48248076, -0.7408423,  -0.04256368],
    [-0.0726215,  0.81418166, 0.57605064, -0.01766401],
    [0.0, 0.0, 0.0, 1.0]
])

T_EE_TO_TOOL = tf_trans.euler_matrix(-1.513, -0.034, 2.597, axes='sxyz')
T_EE_TO_TOOL[0:3, 3] = [-0.061459, -0.099893, 0.028677]
T_TOOL_TO_EE = np.linalg.inv(T_EE_TO_TOOL)

T_ARUCO_TO_GRASP = tf_trans.rotation_matrix(np.pi, [1, 0, 0])

# ===============================


# ===============================
# 主任务节点
# ===============================
class RobotMissionNode:
    def __init__(self):
        rospy.init_node("mission_control_node")

        self.listener = DogSignalListener()

        self.movej_pub  = rospy.Publisher("/rm_driver/MoveJ_Cmd", MoveJ, queue_size=10)
        self.movejp_pub = rospy.Publisher("/rm_driver/MoveJ_P_Cmd", MoveJ_P, queue_size=10)

        rospy.Subscriber("/rm_driver/Pose_State", Pose, self._pose_cb)
        self.current_ee_pose = None

        self.current_point_idx = None

        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def _pose_cb(self, msg):
        self.current_ee_pose = msg

    # ===============================


    # ===============================
    # 第一阶段
    # ===============================
    def action_phase_1(self):
        rospy.loginfo(">>> 第一阶段开始")
        time.sleep(1.0)

        # 预识别一次获取初始位姿
        self.gripper.set_gripper_pos(0); rospy.sleep(2.0)
        T_base_aruco_init = self.get_aruco_pose(9)
        rospy.loginfo("移动到二维码的预备位")
        T_pre = T_ARUCO_TO_GRASP.copy(); T_pre[0:3, 3] = [0.0, -0.05, 0.15]  # 离二维码近一点，减小误差
        self.send_movejp_cmd(T_base_aruco_init @ T_pre @ T_TOOL_TO_EE, speed=0.3); rospy.sleep(6.0)

        # 二次刷新位姿
        T_base_aruco = self.get_aruco_pose(9)
        # 抓取序列
        T_t1 = T_ARUCO_TO_GRASP.copy(); T_t1[2, 3] = 0.03  # 距离二维码3cm
        self.send_movejp_cmd(T_base_aruco @ T_t1 @ T_TOOL_TO_EE, speed=0.3); rospy.sleep(3.5)
        T_t2 = T_ARUCO_TO_GRASP.copy(); T_t2[0:3, 3] = [-0.024, -0.018, 0.03]  # 相对于二维码的偏移,-x向右，-y向上，-z向里
        self.send_movejp_cmd(T_base_aruco @ T_t2 @ T_TOOL_TO_EE, speed=0.2); rospy.sleep(3.5)
        T_t3 = T_ARUCO_TO_GRASP.copy(); T_t3[0:3, 3] = [-0.024, -0.018, -0.026] # 抓卡的预备位置
        self.send_movejp_cmd(T_base_aruco @ T_t3 @ T_TOOL_TO_EE, speed=0.1); rospy.sleep(4.0)
        self.gripper.set_gripper_force(100); rospy.sleep(0.1); self.gripper.set_gripper_pos(100); rospy.sleep(3.0)
        rospy.loginfo("--- Action 5: 将盒子上抬---")
        T_up = T_ARUCO_TO_GRASP.copy(); T_up[0:3, 3] = [-0.031, -0.25, -0.023] # 将盒子上抬，防止盒子碰到其它物体
        self.send_movejp_cmd(T_base_aruco @ T_up @ T_TOOL_TO_EE, speed=0.3); rospy.sleep(4.0)

        rospy.loginfo(">>> 第一阶段动作完成")
        self.send_done(0x23)

    # ===============================
    # 第二阶段
    # ===============================
    def action_phase_2(self):
        rospy.loginfo(">>> 第二阶段开始")
        time.sleep(1.0)
        
        self.send_movej_cmd([-63.053, 59.23, 101.93, 26.71, 30.61, -27.03], 0.3); rospy.sleep(3.5) # 到固定点位识别二维码准备夹卡片
        T_base_aruco_new = self.get_aruco_pose(9) # 检测二维码
        T_steps = [[0,0,0.05], [-0.021, -0.032, 0.03], [-0.021, -0.033, -0.009]]
        # 抓卡序列，分别是：对准二维码的中心距二维码5cm、去找卡片、然后到达夹取卡片的预备位
        for off in T_steps:
            T_n = T_ARUCO_TO_GRASP.copy(); T_n[0:3, 3] = off
            self.send_movejp_cmd(T_base_aruco_new @ T_n @ T_TOOL_TO_EE, 0.15 if off[2]<0.05 else 0.5); rospy.sleep(2.5)
        
        self.gripper.set_gripper_force(100);rospy.sleep(0.1);  self.gripper.set_gripper_pos(100); rospy.sleep(1.0)# 闭合夹爪，夹取卡片
        T_n3 = T_ARUCO_TO_GRASP.copy(); T_n3[0:3, 3] = [-0.021, -0.05, -0.009]  # 将卡片上抬一点，直接拿走的话会撞到盒子
        self.send_movejp_cmd(T_base_aruco_new @ T_n3 @ T_TOOL_TO_EE, 0.15); rospy.sleep(2.5)

        # 插卡槽、按键、读卡
        self.send_movej_cmd([-44.476, 72.622, 84.144, -5.121, 25.305, -102.02], 0.3); rospy.sleep(3.0)  # 找黑色的卡槽，也就是检测的预定点位
        rospy.loginfo(">>> 第二阶段动作完成")
        self.send_done(0x22)

    # ===============================
    # 主循环
    # ===============================
    def run(self):
        rospy.loginfo("等待机械狗指令...")
        while not rospy.is_shutdown():
            cmd = self.listener.wait_cmd()

            if 0x31 <= cmd <= 0x3C:
                self.current_point_idx = cmd - 0x30
                rospy.loginfo(">>> 设置起始点位：第 %d 组", self.current_point_idx)
                continue

            if cmd == 0x23:
                self.action_phase_1()

            elif cmd == 0x22:
                self.action_phase_2()

# ===============================
# main
# ===============================
if __name__ == "__main__":
    node = RobotMissionNode()
    rospy.sleep(1.0)
    node.run()

