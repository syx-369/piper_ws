#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import tf.transformations as tf_trans

from ultralytics import YOLO
from piper_msgs.msg import PosCmd
from geometry_msgs.msg import PoseStamped


# ===============================
# 参数
# ===============================

MODEL_PATH = "/home/hank/piper_ws/src/piper_ros/src/bottle_grab/bottle.pt"
CONF_THRES = 0.5

COLOR_WIDTH = 640
COLOR_HEIGHT = 480
FPS = 30

# ===============================
# 位姿矩阵
# ===============================

T_EE_TO_CAM = np.array([
    [-0.0482,  0.9987,  0.0147, -0.0699],
    [-0.9979, -0.0488,  0.0416,  0.0301],
    [ 0.0423, -0.0127,  0.9990,  0.0675],
    [ 0,0,0,1]
])

T_EE_TO_TOOL = np.eye(4)
T_EE_TO_TOOL[0:3,3] = [0,0,-0.14]
T_TOOL_TO_EE = np.linalg.inv(T_EE_TO_TOOL)

# 抓取姿态
R1 = tf_trans.rotation_matrix(-np.pi/2,[0,0,1])
R2 = tf_trans.rotation_matrix(np.pi,[1,0,0])
T_GRASP = tf_trans.concatenate_matrices(R1,R2)


# ===============================
# 控制类
# ===============================

class PiperVisionController:

    def __init__(self):

        rospy.init_node("piper_yolo_grasp")

        self.pub = rospy.Publisher("/pin_pos_cmd",PosCmd,queue_size=1)

        self.current_ee_pose_mat = np.eye(4)
        self.pose_received=False

        rospy.Subscriber("/end_pose",PoseStamped,self.pose_callback)

        rospy.loginfo("等待机械臂位姿...")

        while not self.pose_received:
            rospy.sleep(0.1)

        rospy.loginfo("位姿获取成功")

        self.init_camera()

        self.model = YOLO(MODEL_PATH)


    def pose_callback(self,msg):

        px=msg.pose.position.x
        py=msg.pose.position.y
        pz=msg.pose.position.z

        q=[msg.pose.orientation.x,
           msg.pose.orientation.y,
           msg.pose.orientation.z,
           msg.pose.orientation.w]

        T=tf_trans.quaternion_matrix(q)

        T[0:3,3]=[px,py,pz]

        self.current_ee_pose_mat=T

        self.pose_received=True


    def init_camera(self):

        self.pipeline=rs.pipeline()

        config=rs.config()

        config.enable_stream(rs.stream.depth,640,480,rs.format.z16,FPS)
        config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,FPS)

        profile=self.pipeline.start(config)

        self.align=rs.align(rs.stream.color)

        intrinsics=profile.get_stream(rs.stream.color)\
            .as_video_stream_profile()\
            .get_intrinsics()

        self.camera_matrix=np.array([
            [intrinsics.fx,0,intrinsics.ppx],
            [0,intrinsics.fy,intrinsics.ppy],
            [0,0,1]
        ])

        self.dist_coeffs=np.zeros((4,1))


    # ===============================
    # 发送位姿
    # ===============================

    def send_pose_matrix(self,T,gripper=0):

        x,y,z=T[0:3,3]

        roll,pitch,yaw=tf_trans.euler_from_matrix(T,'sxyz')

        cmd=PosCmd()

        cmd.x=x
        cmd.y=y
        cmd.z=z

        cmd.roll=roll
        cmd.pitch=pitch
        cmd.yaw=yaw

        cmd.gripper=gripper

        cmd.mode1=1
        cmd.mode2=0

        self.pub.publish(cmd)

        rospy.loginfo(f"发送位姿: {x:.3f} {y:.3f} {z:.3f}")


    # ===============================
    # 检测瓶子
    # ===============================

    def detect_bottle(self):

        rospy.loginfo("开始检测瓶子")

        for _ in range(30):
            frames=self.pipeline.wait_for_frames()

        frames=self.align.process(frames)

        depth_frame=frames.get_depth_frame()
        color_frame=frames.get_color_frame()

        color_image=np.asanyarray(color_frame.get_data())

        depth_intrinsics=depth_frame.profile\
            .as_video_stream_profile().intrinsics

        results=self.model.predict(color_image,conf=CONF_THRES,verbose=False)

        if len(results)==0:
            return None

        r=results[0]

        if r.boxes is None or len(r.boxes) == 0:
            rospy.logerr("YOLO未检测到目标")
            return None

        boxes=r.boxes.xyxy.cpu().numpy()

        box=boxes[0].astype(int)

        x1,y1,x2,y2=box

        cx=(x1+x2)//2
        cy=(y1+y2)//2

        dist=depth_frame.get_distance(cx,cy)

        if dist<=0:
            rospy.logerr("深度无效")
            return None

        point=rs.rs2_deproject_pixel_to_point(
            depth_intrinsics,
            [cx,cy],
            dist
        )

        X,Y,Z=point

        rospy.loginfo(f"检测到瓶子 camera坐标: {X:.3f} {Y:.3f} {Z:.3f}")

        T_cam_obj=np.eye(4)
        T_cam_obj[0:3,3]=[X,Y,Z]

        return T_cam_obj

    # ===============================
    # 抓取任务
    # ===============================

    def execute_grasp(self):

        rospy.loginfo("移动到观察位")

        T_init=tf_trans.euler_matrix(0,1.57,0,'sxyz')

        T_init[0:3,3]=[0.26,0,0.17]

        self.send_pose_matrix(T_init,0)

        time.sleep(6)

        T_cam_obj=self.detect_bottle()

        if T_cam_obj is None:
            rospy.logerr("未检测到瓶子")
            return

        T_base_ee=self.current_ee_pose_mat

        T_base_obj=T_base_ee@T_EE_TO_CAM@T_cam_obj

        # 抓取上方
        T_pre=T_GRASP.copy()
        T_pre[0:3,3]=[0,0,0.15]

        target=T_base_obj@T_pre@T_TOOL_TO_EE

        self.send_pose_matrix(target,100)

        time.sleep(4)

        # 下压
        T_down=T_GRASP.copy()
        T_down[0:3,3]=[0,0,0.08]

        target=T_base_obj@T_down@T_TOOL_TO_EE

        self.send_pose_matrix(target,100)

        time.sleep(3)

        # 闭合夹爪
        self.send_pose_matrix(target,0)

        time.sleep(2)

        # 抬起
        T_up=T_GRASP.copy()
        T_up[0:3,3]=[0,0.15,-0.05]

        target=T_base_obj@T_up@T_TOOL_TO_EE

        self.send_pose_matrix(target,0)

        time.sleep(3)

        rospy.loginfo("抓取完成")


    def run(self):

        try:

            self.execute_grasp()

        finally:

            self.pipeline.stop()



if __name__=="__main__":

    try:

        node=PiperVisionController()

        node.run()

    except rospy.ROSInterruptException:

        pass
