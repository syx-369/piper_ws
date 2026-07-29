cmake_minimum_required(VERSION 3.0.2)
project(piper_task)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  piper_msgs
  rospy
  sensor_msgs
  std_msgs
)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  scripts/arm_function_test_follower.py
  scripts/piper_task_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
