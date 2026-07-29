#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import time
from piper_msgs.msg import PosCmd

def descend_motion():
    rospy.init_node("fixed_xy_descend", anonymous=True)

    pub = rospy.Publisher("/pin_pos_cmd", PosCmd, queue_size=1)
    rospy.sleep(0.5)

    # 固定参数
    
    x_fixed = 0.36 + 0.05
    y_fixed = 0
    z_fixed = 0.08
    '''
    x_fixed = 0.34 + 0.13
    y_fixed = 0.0
    z_fixed = 0.12
    '''
    # 姿态：末端水平
    roll  = 0
    pitch = 1.57
    yaw   = 0



    cmd = PosCmd()
    cmd.x = x_fixed
    cmd.y = y_fixed
    cmd.z = z_fixed

    cmd.roll  = roll
    cmd.pitch = pitch
    cmd.yaw   = yaw

    cmd.gripper = 0
    cmd.mode1 = 1
    cmd.mode2 = 0

    pub.publish(cmd)
    time.sleep(3)



if __name__ == "__main__":
    descend_motion()
