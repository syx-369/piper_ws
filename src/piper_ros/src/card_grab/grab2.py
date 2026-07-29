#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import time
from piper_msgs.msg import PosCmd

def descend_motion():
    rospy.init_node("smooth_integral_move", anonymous=True)
    pub = rospy.Publisher("/pin_pos_cmd", PosCmd, queue_size=1)
    rospy.sleep(0.5)
    
    # position1_1
    # 固定参数
    y = 0.0
    z = 0.08

    x_start = 0.36
    x_end   = 0.36 + 0.05

    rate_hz = 100              
    v = 0.01                   
    dt = 1.0 / rate_hz

    roll, pitch, yaw = 0.0, 1.57, 0.0

    rate = rospy.Rate(rate_hz)
    x = x_start
    
    # 先到指定位置前0.05m
    cmd = PosCmd()
    cmd.x = x
    cmd.y = y
    cmd.z = z
    cmd.roll  = roll
    cmd.pitch = pitch
    cmd.yaw   = yaw
    cmd.gripper = 0
    cmd.mode1 = 1
    cmd.mode2 = 0
    pub.publish(cmd)
    time.sleep(4)

    # 夹爪张开
    cmd = PosCmd()
    cmd.x = x
    cmd.y = y
    cmd.z = z
    cmd.roll  = roll
    cmd.pitch = pitch
    cmd.yaw   = yaw
    cmd.gripper = 250
    cmd.mode1 = 1
    cmd.mode2 = 0
    pub.publish(cmd)
    time.sleep(0.5)

    while x < x_end and not rospy.is_shutdown():
        x += v * dt           

        cmd = PosCmd()
        cmd.x = x
        cmd.y = y
        cmd.z = z
        cmd.roll  = roll
        cmd.pitch = pitch
        cmd.yaw   = yaw
        cmd.gripper = 250
        cmd.mode1 = 1
        cmd.mode2 = 0

        pub.publish(cmd)
        rate.sleep()
    time.sleep(0.5)
    
    # 夹爪闭合
    cmd = PosCmd()
    cmd.x = x
    cmd.y = y
    cmd.z = z
    cmd.roll  = roll
    cmd.pitch = pitch
    cmd.yaw   = yaw
    cmd.gripper = 0
    cmd.mode1 = 1
    cmd.mode2 = 0
    pub.publish(cmd)
    time.sleep(0.5)
    
    # position1_2
    # 固定参数
    y = 0.0
    z = 0.08

    x_start = 0.36 + 0.05
    x_end   = 0.36

    rate_hz = 100              
    v = 0.01                   
    dt = 1.0 / rate_hz

    roll, pitch, yaw = 0.0, 1.57, 0.0

    rate = rospy.Rate(rate_hz)
    x = x_start

    while x > x_end and not rospy.is_shutdown():
        x -= v * dt           

        cmd = PosCmd()
        cmd.x = x
        cmd.y = y
        cmd.z = z
        cmd.roll  = roll
        cmd.pitch = pitch
        cmd.yaw   = yaw
        cmd.gripper = 0
        cmd.mode1 = 1
        cmd.mode2 = 0

        pub.publish(cmd)
        rate.sleep()
        
    time.sleep(1)
    
    # position2
    # 固定参数
    x = 0.36
    z = 0.08
    
    y_start = 0
    y_end   = 0 + 0.1

    rate_hz = 100              
    v = 0.01                   
    dt = 1.0 / rate_hz

    roll, pitch, yaw = 0.0, 1.57, 0.0

    rate = rospy.Rate(rate_hz)
    y = y_start
    

    while y < y_end and not rospy.is_shutdown():
        y += v * dt           

        cmd = PosCmd()
        cmd.x = x
        cmd.y = y
        cmd.z = z
        cmd.roll  = roll
        cmd.pitch = pitch
        cmd.yaw   = yaw
        cmd.gripper = 0
        cmd.mode1 = 1
        cmd.mode2 = 0

        pub.publish(cmd)
        rate.sleep()
        
    time.sleep(2)
    
    # position3_1
    # 固定参数
    x = 0.36
    z = 0.08
    
    y_start = 0 + 0.1
    y_end   = 0 + 0.1 + 0.1

    rate_hz = 100              
    v = 0.01                   
    dt = 1.0 / rate_hz

    roll, pitch, yaw = 0.0, 1.57, 0.0

    rate = rospy.Rate(rate_hz)
    y = y_start
    

    while y < y_end and not rospy.is_shutdown():
        y += v * dt           

        cmd = PosCmd()
        cmd.x = x
        cmd.y = y
        cmd.z = z
        cmd.roll  = roll
        cmd.pitch = pitch
        cmd.yaw   = yaw
        cmd.gripper = 0
        cmd.mode1 = 1
        cmd.mode2 = 0

        pub.publish(cmd)
        rate.sleep()
        
    time.sleep(2)
    
    # position3_2
    # 固定参数
    y = 0 + 0.1 + 0.1
    z = 0.08

    x_start = 0.36
    x_end   = 0.36 + 0.05

    rate_hz = 100              
    v = 0.01                   
    dt = 1.0 / rate_hz

    roll, pitch, yaw = 0.0, 1.57, 0.0

    rate = rospy.Rate(rate_hz)
    x = x_start
    

    while x < x_end and not rospy.is_shutdown():
        x += v * dt           

        cmd = PosCmd()
        cmd.x = x
        cmd.y = y
        cmd.z = z
        cmd.roll  = roll
        cmd.pitch = pitch
        cmd.yaw   = yaw
        cmd.gripper = 0
        cmd.mode1 = 1
        cmd.mode2 = 0

        pub.publish(cmd)
        rate.sleep()
        
    # 夹爪张开
    cmd = PosCmd()
    cmd.x = x
    cmd.y = y
    cmd.z = z
    cmd.roll  = roll
    cmd.pitch = pitch
    cmd.yaw   = yaw
    cmd.gripper = 250
    cmd.mode1 = 1
    cmd.mode2 = 0
    pub.publish(cmd)
    time.sleep(0.5)
    
    # position3_3
    # 固定参数
    y = 0 + 0.1 + 0.1
    z = 0.08

    x_start = 0.36 + 0.05
    x_end   = 0.36

    rate_hz = 100              
    v = 0.01                   
    dt = 1.0 / rate_hz

    roll, pitch, yaw = 0.0, 1.57, 0.0

    rate = rospy.Rate(rate_hz)
    x = x_start

    while x > x_end and not rospy.is_shutdown():
        x -= v * dt           

        cmd = PosCmd()
        cmd.x = x
        cmd.y = y
        cmd.z = z
        cmd.roll  = roll
        cmd.pitch = pitch
        cmd.yaw   = yaw
        cmd.gripper = 250
        cmd.mode1 = 1
        cmd.mode2 = 0

        pub.publish(cmd)
        rate.sleep()
    
    # 夹爪闭合
    cmd = PosCmd()
    cmd.x = x
    cmd.y = y
    cmd.z = z
    cmd.roll  = roll
    cmd.pitch = pitch
    cmd.yaw   = yaw
    cmd.gripper = 0
    cmd.mode1 = 1
    cmd.mode2 = 0
    pub.publish(cmd)
    time.sleep(0.5)

if __name__ == "__main__":
    descend_motion()
