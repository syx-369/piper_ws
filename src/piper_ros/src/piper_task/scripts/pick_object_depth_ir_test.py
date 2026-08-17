#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""抓取区候选点机械臂观察位与全类别视觉测试。

机械臂使用正式比赛主控的 PICK_SCAN_JOINTS，随后持续识别模型支持的
全部方块和瓶子。本节点不控制车辆、不执行抓取，并可独立于航迹录制节点
启动和关闭；退出后机械臂默认返回主控运输零位。

运行：
  rosrun piper_task pick_object_depth_ir_test.py

仅移动到抓取观察位、不开相机：
  rosrun piper_task pick_object_depth_ir_test.py _enable_vision:=false
"""

import importlib.util
import os

import rospy


# catkin_install_python 在 devel/lib 下生成的是 exec 转发脚本；把该转发脚本
# 当普通模块 import 时，源文件中的类不会出现在转发模块的 globals 中。
# 因此按真实源文件路径加载，共享放置脚本中的观察/相机实现。
IMPLEMENTATION_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "place_bottle_depth_ir_test.py",
)
IMPLEMENTATION_SPEC = importlib.util.spec_from_file_location(
    "piper_task_observation_depth_ir_impl", IMPLEMENTATION_PATH
)
if IMPLEMENTATION_SPEC is None or IMPLEMENTATION_SPEC.loader is None:
    raise ImportError("无法加载候选点观察实现: %s" % IMPLEMENTATION_PATH)
IMPLEMENTATION_MODULE = importlib.util.module_from_spec(IMPLEMENTATION_SPEC)
IMPLEMENTATION_SPEC.loader.exec_module(IMPLEMENTATION_MODULE)
PlaceBottleDepthIrTestNode = IMPLEMENTATION_MODULE.PlaceBottleDepthIrTestNode


def main():
    rospy.init_node("pick_object_depth_ir_test", anonymous=False)
    if not rospy.has_param("~observation_mode"):
        rospy.set_param("~observation_mode", "pick")
    node = PlaceBottleDepthIrTestNode()
    node.run()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except Exception as exc:
        try:
            rospy.logfatal("抓取区候选点观察启动/运行失败: %s", exc)
        except Exception:
            pass
        raise
