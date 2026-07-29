#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import csv
import numpy as np
import math
import tf.transformations as tf_trans
from piper_msgs.msg import PosCmd
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

class PiperGraspController:
    def __init__(self):
        rospy.init_node('vision_grasp_node')
        
        # 发布者
        self.pos_pub = rospy.Publisher('/pin_pos_cmd', PosCmd, queue_size=1)
        self.joint_pub = rospy.Publisher('/piper_driver/set_joint_cmd', JointState, queue_size=1)
        
        # 实时位姿订阅
        self.current_ee_pose_mat = np.eye(4)
        self.pose_received = False
        rospy.Subscriber("/end_pose", PoseStamped, self.pose_callback)

        # 路径指向 realsensedetect.py 生成的 TCP 坐标文件
        self.csv_path = '/home/hank/下载/realsense-D455-YOLOV5/tcp_coordinates.csv'

    def pose_callback(self, msg):
        px, py, pz = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        q = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        T_mat = tf_trans.quaternion_matrix(q)
        T_mat[0:3, 3] = [px, py, pz]
        self.current_ee_pose_mat = T_mat
        self.pose_received = True

    def read_target_from_csv(self):
        """读取最新的瓶子在末端坐标系下的坐标"""
        try:
            with open(self.csv_path, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
                if len(rows) <= 1: return None # 跳过表头
                last_row = rows[-1]
                # 返回 [x, y, z] 相对末端的偏移
                return np.array([float(last_row[1]), float(last_row[2]), float(last_row[3])])
        except Exception as e:
            rospy.logwarn(f"CSV读取失败: {e}")
            return None

    def get_target_in_base(self, rel_coords):
        """将末端系下的相对坐标转换到基座系"""
        T_rel = np.eye(4)
        T_rel[0:3, 3] = rel_coords
        # Base_T_Target = Base_T_EE * EE_T_Target
        T_target_base = np.dot(self.current_ee_pose_mat, T_rel)
        return T_target_base[0:3, 3]

    def send_pos_cmd(self, pos, r_mat, gripper_val=0):
        """发送控制指令"""
        roll, pitch, yaw = tf_trans.euler_from_matrix(r_mat, axes='sxyz')
        cmd = PosCmd()
        cmd.x, cmd.y, cmd.z = pos
        cmd.roll, cmd.pitch, cmd.yaw = roll, pitch, yaw
        cmd.gripper = gripper_val
        cmd.mode1 = 1 
        self.pos_pub.publish(cmd)

    def execute_grasp(self):
        # 1. 初始化检查
        while not self.pose_received and not rospy.is_shutdown():
            rospy.sleep(0.5)
        
        # 2. 移动到初始观察位
        rospy.loginfo("移动到观察位...")
        joint_msg = JointState()
        joint_msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        joint_msg.position = [0, 0, 0, 0, 0.5, 0] # 调整为适合相机观察的角度
        self.joint_pub.publish(joint_msg)
        rospy.sleep(6)

        # 3. 获取目标并执行
        rel_pos = self.read_target_from_csv()
        if rel_pos is not None:
            target_base = self.get_target_in_base(rel_pos)
            rospy.loginfo(f"目标基座坐标: {target_base}")

            # 保持末端垂直向下或当前姿态
            rot_mat = self.current_ee_pose_mat.copy()
            rot_mat[0:3, 3] = 0

            # 动作序列
            # A. 预抓取 (目标上方 10cm)
            pre_grasp = [target_base[0], target_base[1], target_base[2] + 0.1]
            self.send_pos_cmd(pre_grasp, rot_mat, gripper_val=80)
            rospy.sleep(6)

            # B. 下降到抓取位
            self.send_pos_cmd(target_base, rot_mat, gripper_val=80)
            rospy.sleep(5)

            # C. 闭合夹爪
            self.send_pos_cmd(target_base, rot_mat, gripper_val=0)
            rospy.sleep(5)

            # D. 抬起
            self.send_pos_cmd(pre_grasp, rot_mat, gripper_val=0)
            rospy.loginfo("抓取任务完成")
        else:
            rospy.logerr("未在CSV中发现有效目标")

if __name__ == '__main__':
    try:
        controller = PiperGraspController()
        controller.execute_grasp()
    except rospy.ROSInterruptException:
        pass
