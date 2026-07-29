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
    y = -0.2
    z = 0.145

    x_start = 0.36
    x_end   = 0.36 + 0.08

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
    cmd.gripper = 0.03
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
        cmd.gripper = 0.03
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
    y = -0.2
    z = 0.145

    x_start = 0.36 + 0.08
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
    
    '''# position2
    # 固定参数
    x = 0.36
    z = 0.12
    
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

    
    x = 0.36
    y = -0.2 + 0.1
    z = 0.12    
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
    
    # <<<<<<<<<<<<<<<<<<<<<高度
    z_start = 0.12
    z_end   = 0.12 - 0.06
    z = z_start
    while z > z_end and not rospy.is_shutdown():
        z -= v * dt           

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
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    # position3_1
    # 固定参数<<<<<<<<<<<<<<<<<高度
    x = 0.36
    y = -0.2 + 0.1
    
    z_start = 0.12 - 0.06
    z_end   = 0.12
    z = z_start
    while z < z_end and not rospy.is_shutdown():
        z += v * dt           

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
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # 固定参数
    x = 0.36
    z = 0.12
    
    y_start = -0.2 + 0.1
    y_end   = -0.2 + 0.1 + 0.1

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
    y = -0.2 + 0.1 + 0.1
    z = 0.12

    x_start = 0.36
    x_end   = 0.36 + 0.08

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
    cmd.gripper = 0.03
    cmd.mode1 = 1
    cmd.mode2 = 0
    pub.publish(cmd)
    time.sleep(0.5)
    
    # position3_3
    # 固定参数
    y = -0.2 + 0.1 + 0.1
    z = 0.12

    x_start = 0.36 + 0.08
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
        cmd.gripper = 0.03
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
    '''
    
    # position2_1
    # 固定参数
    x = 0.36
    y = -0.2 + 0.4
    z = 0.145 + 0.06
    
    roll, pitch, yaw = 0.0, 1.57, 0.0
    
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
    
    
    # position2_2
    # 固定参数
    y = -0.2 + 0.4
    z = 0.145 + 0.06
    
    x_start = 0.36
    x_end   = 0.36 + 0.13

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
        
    time.sleep(0.5)
        
    # position3_1
    # 固定参数
    y = -0.2 + 0.4
    z = 0.145 + 0.06
    
    x_start = 0.36 + 0.13
    x_end   = 0.36 + 0.06

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
        
    time.sleep(0.5)
    
    # position3_2
    # 固定参数
    y = -0.2 + 0.4
    x = 0.36 + 0.06
    
    z_start = 0.145 + 0.06
    z_end   = 0.145 + 0.06 - 0.09

    rate_hz = 100              
    v = 0.01                   
    dt = 1.0 / rate_hz

    roll, pitch, yaw = 0.0, 1.57, 0.0

    rate = rospy.Rate(rate_hz)
    z = z_start
    
    while z > z_end and not rospy.is_shutdown():
        z -= v * dt           

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
        
    time.sleep(0.5)
    
        
    # position3_2
    # 固定参数
    x = 0.36 + 0.06
    z = 0.145 + 0.06 - 0.09
    
    y_start = -0.2 + 0.4
    y_end   = -0.2 + 0.4 + 0.085

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
        
    time.sleep(0.5)
    
    # position3_3
    # 固定参数
    y = -0.2 + 0.4 + 0.085
    z = 0.145 + 0.06 - 0.09
    
    x_start = 0.36 + 0.06
    x_end   = 0.36 + 0.06 + 0.12

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
    
    time.sleep(0.5)
    
    # 夹爪张开
    cmd = PosCmd()
    cmd.x = x
    cmd.y = y
    cmd.z = z
    cmd.roll  = roll
    cmd.pitch = pitch
    cmd.yaw   = yaw
    cmd.gripper = 0.03
    cmd.mode1 = 1
    cmd.mode2 = 0
    pub.publish(cmd)
    time.sleep(0.5)
    
    
    # position3_4
    # 固定参数
    y = -0.2 + 0.4 + 0.085
    z = 0.145 + 0.06 - 0.09
    
    x_start = 0.36 + 0.06 + 0.12
    x_end   = 0.36 + 0.06

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
        cmd.gripper = 0.03
        cmd.mode1 = 1
        cmd.mode2 = 0

        pub.publish(cmd)
        rate.sleep()


if __name__ == "__main__":
    descend_motion()
