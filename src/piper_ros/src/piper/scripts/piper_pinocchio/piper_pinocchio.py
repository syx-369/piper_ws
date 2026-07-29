import casadi
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin
import time
try:
    import termios
    import tty
except ImportError:
    import msvcrt
import rospy
from pinocchio import casadi as cpin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import os
import sys
import threading
from piper_control import PIPER
from piper_msgs.msg import PosCmd

piper_control = PIPER()

exit_flag = False

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


class Arm_IK:
    def __init__(self):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)
        cur_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        last_path = os.path.dirname(current_dir)
        last_path = os.path.dirname(last_path)
        last_path = os.path.dirname(last_path)
        urdf_dir = os.path.join(last_path, 'piper_description/urdf/piper_description.urdf')
        # urdf_path = '/home/agilex/piper_ws/src/piper_description/urdf/piper_description.urdf'
        urdf_path = urdf_dir
    
        # 从 URDF 生成 robot
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path)

        # 把 gripper 两个关节锁定，让 IK 只求 6 自由度
        self.mixed_jointsToLockIDs = ["joint7",
                                      "joint8"
                                      ]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0] * self.robot.model.nq),
        )

        # 在 joint6 处创建了一个末端坐标系
        # q = quaternion_from_euler(0, -1.57, -1.57)
        q = quaternion_from_euler(0, 0, 0)
        self.reduced_robot.model.addFrame(
            pin.Frame('ee',                                           # 坐标系名
                      self.reduced_robot.model.getJointId('joint6'),  # 把末端坐标系关联在 joint6 的末端
                      pin.SE3(                                        # 代表坐标系相对 joint6 的位姿
                          # pin.Quaternion(1, 0, 0, 0),
                          pin.Quaternion(q[3], q[0], q[1], q[2]),     # 旋转部分
                          np.array([0.0, 0.0, 0.0]),                  # 平移部分
                      ),
                      pin.FrameType.OP_FRAME)                         # 操作空间坐标系
        )

        # 从 URDF 文件中 构建几何模型,得到每个 link 对应的碰撞体
        self.geom_model = pin.buildGeomFromUrdf(self.robot.model, urdf_path, pin.GeometryType.COLLISION)
        
        # 手动添加“需要检测碰撞的 link 对”，即后半段（关节 4~关节 9）避免与前半段（关节 0~2）自碰撞
        for i in range(4, 10):
            for j in range(0, 3):
                self.geom_model.addCollisionPair(pin.CollisionPair(i, j))
        # 为几何模型创建对应的数据缓存
        self.geometry_data = pin.GeometryData(self.geom_model)
        # 初始化 IK 的“初值”和“历史解”
        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.history_data = np.zeros(self.reduced_robot.model.nq)

        # 初始化 Meshcat 可视化器
        self.vis = MeshcatVisualizer(self.reduced_robot.model,            # 运动学模型（关节与连杆）
                                     self.reduced_robot.collision_model,  # 用于碰撞检测的几何体（可视化透明）
                                     self.reduced_robot.visual_model)     # 用于显示的 3D 网格模型（视觉效果好）
        self.vis.initViewer(open=True)                                    # 启动 Meshcat viewer
        self.vis.loadViewerModel("pinocchio")                             # 把 robot 的 3D 模型加载进 Meshcat，并给模型取一个前缀 "pinocchio"
        # 显示两个关键坐标系
        self.vis.displayFrames(True,                                      # 开启显示坐标轴
                               frame_ids=[113, 114], 
                               axis_length=0.15,                          # 长度
                               axis_width=5)                              # 线宽
        self.vis.display(pin.neutral(self.reduced_robot.model))           # 把机器人显示到 默认姿态（初始姿态）

        # 手动绘制坐标轴
        frame_viz_names = ['ee_target']
        FRAME_AXIS_POSITIONS = (
        np.array([
		[0, 0, 0], [1, 0, 0],  # X 轴 从 (0,0,0) 到 (1,0,0)
		[0, 0, 0], [0, 1, 0],  # Y 轴 从 (0,0,0) 到 (0,1,0)
		[0, 0, 0], [0, 0, 1]   # Z 轴 从 (0,0,0) 到 (0,0,1)
	    ]).astype(np.float32).T
	)
        FRAME_AXIS_COLORS = (
        np.array([
		[1, 0, 0], [1, 0.6, 0],  # X 轴渐变颜色（深红 → 橙红）
		[0, 1, 0], [0.6, 1, 0],  # Y 轴（深绿 → 黄绿）
		[0, 0, 1], [0, 0.6, 1],  # Z 轴（深蓝 → 青蓝）
	    ]).astype(np.float32).T
	)
        axis_length = 0.1
        axis_width = 10
        # 把画好的坐标轴放到 Meshcat 中
        for frame_viz_name in frame_viz_names:
            self.vis.viewer[frame_viz_name].set_object(
                mg.LineSegments(
                    mg.PointsGeometry(
                        position=axis_length * FRAME_AXIS_POSITIONS,
                        color=FRAME_AXIS_COLORS,
                    ),
                    mg.LineBasicMaterial(
                        linewidth=axis_width,
                        vertexColors=True,
                    ),
                )
            )

        '''创建一个符号化（symbolic）的误差函数 error(q, Tf)表示当前机械臂末端执行器姿态与目标姿态之间的 SE(3) 误差，用于优化IK'''
        # 创建 CasADi 符号化模型
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()
        # 创建符号变量
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        # 调用符号化的前向运动学,输出到 self.cdata.oMf,目的是得到指定关节变量后得到的位姿，与目标位姿进行比较
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)
        # 取末端执行器（ee）坐标系 ID，要计算误差，必须知道哪个 frame 是末端执行器
        self.gripper_id = self.reduced_robot.model.getFrameId("ee")
        # 构造误差函数 error(q, Tf)
        self.error = casadi.Function(
            "error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log6(
                        self.cdata.oMf[self.gripper_id].inverse() * cpin.SE3(self.cTf)
                    ).vector,
                )
            ],
        )
        '''
        这段代码详解看笔记
        '''

        # 创建优化器
        self.opti = casadi.Opti()
        # 定义优化变量：关节角。要让 q 满足末端去到目标位置，因此 q 是优化的主要对象
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        # self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # for smooth
        # 这是输入到 IK 的目标位姿
        self.param_tf = self.opti.parameter(4, 4)
        # self.totalcost = casadi.sumsqr(self.error(self.var_q, self.param_tf))
        # 给误差函数赋值
        error_vec = self.error(self.var_q, self.param_tf)
        pos_error = error_vec[:3]  # 取前3个值为位置误差
        ori_error = error_vec[3:]  # 取后3个值为姿态误差
        # 设置位置和姿态的权重
        weight_position = 1.0      # 位置权重
        weight_orientation = 0.1   # 姿态权重
        # 定义总成本函数
        self.totalcost = casadi.sumsqr(weight_position * pos_error) + casadi.sumsqr(weight_orientation * ori_error)
        # 正则化项（避免解跳动，不对称）
        self.regularization = casadi.sumsqr(self.var_q)
        # self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last) # for smooth

        # 关节角上下界约束
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        # print("self.reduced_robot.model.lowerPositionLimit:", self.reduced_robot.model.lowerPositionLimit)
        # print("self.reduced_robot.model.upperPositionLimit:", self.reduced_robot.model.upperPositionLimit)
        # 组合最终的代价函数（该函数值要越小越好）
        self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization)
        # self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization + 0.1 * self.smooth_cost) # for smooth
        # 给优化器传参
        opts = {                         # 外层的参数（如 print_time）是 CasADi 自己的配置
            'ipopt': {                   # 里面是传给 IPOPT 优化器本体的参数
                'print_level': 0,        # 设置为 0，是为了 IK 每次求解很干净，不往 Terminal 打垃圾（完全不输出迭代过程）
                'max_iter': 50,          # 设置最多迭代次数
                'tol': 1e-4              # 设置收敛判定的容忍度（越小越严格）
            },
            'print_time': False          # 不打印任何信息
        }
        self.opti.solver("ipopt", opts)  # 指定求解器为IPOPT，并用刚配置好的设置（opts）

    def ik_fun(self, target_pose, gripper=0, motorstate=None, motorV=None):
        gripper = np.array([gripper/2.0, -gripper/2.0])
        # 设置优化初值
        if motorstate is not None:
            self.init_data = motorstate
        self.opti.set_initial(self.var_q, self.init_data)
        # 用 Meshcat 把目标位姿画出来
        self.vis.viewer['ee_target'].set_transform(target_pose)     # for visualization
        # 设置优化器的目标位姿（把 target_pose 赋给 self.param_tf）
        self.opti.set_value(self.param_tf, target_pose)
        # self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            # sol = self.opti.solve()
            sol = self.opti.solve_limited()         # 启动求解器求解
            sol_q = self.opti.value(self.var_q)     # 把求解后 Opti 中变量 self.var_q 的数值解提取出来

            if self.init_data is not None:
                max_diff = max(abs(self.history_data - sol_q))               # history_data 保存的是上一帧的解
                # print("max_diff:", max_diff)
                self.init_data = sol_q                                       # 将这次的求解结果设置为下次求解的初值
                if max_diff > 30.0/180.0*3.1415:                             # 如果本次解跳变太大（>30°），说明求解不可信
                    # print("Excessive changes in joint angle:", max_diff)
                    self.init_data = np.zeros(self.reduced_robot.model.nq)   # 则把初值重置为 0，让下一次重新求
            else:
                self.init_data = sol_q
            self.history_data = sol_q

            # 把 IK 结果显示出来
            self.vis.display(sol_q)  # for visualization

            if motorV is not None:
                v = motorV * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            # 计算前馈力矩
            tau_ff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v,
                              np.zeros(self.reduced_robot.model.nv))
            # 自碰撞检测
            is_collision = self.check_self_collision(sol_q, gripper)

            return sol_q, tau_ff, not is_collision

        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")
            # sol_q = self.opti.debug.value(self.var_q)   # return original value
            return None, '', False

    def check_self_collision(self, q, gripper=np.array([0, 0])):
        pin.forwardKinematics(self.robot.model, self.robot.data, np.concatenate([q, gripper], axis=0))
        pin.updateGeometryPlacements(self.robot.model, self.robot.data, self.geom_model, self.geometry_data)
        collision = pin.computeCollisions(self.geom_model, self.geometry_data, False)
        # print("collision:", collision)
        return collision

    def get_ik_solution(self, x,y,z,roll,pitch,yaw,gripper):
        # 转换欧拉角为四元数（避免万向节锁）
        q = quaternion_from_euler(roll, pitch, yaw)
        
        '''
        目标位姿 target =
        R  （旋转矩阵，由四元数构建）
        p  （位置向量，由 np.array([x,y,z]) 构建）

        '''
        target = pin.SE3(
            pin.Quaternion(q[3], q[0], q[1], q[2]),        # 重新排列q = [qx, qy, qz, qw] →  Pinocchio( qw, qx, qy, qz )
            np.array([x, y, z]),
        )
        print(target)
        # target.homogeneous 就是把 Pinocchio 的 SE3 变成 CasADi 计算使用的 4×4 齐次变换矩阵
        sol_q, tau_ff, get_result = self.ik_fun(target.homogeneous,0)
        print("result:", sol_q)
        
        if get_result :
            piper_control.joint_control_piper(sol_q[0],sol_q[1],sol_q[2],sol_q[3],sol_q[4],sol_q[5],gripper)
        else :
            print("collision!!!")
    
class C_PiperIK():
    def __init__(self):
        rospy.init_node('inverse_solution_node', anonymous=True)
        # 创建Arm_IK实例
        self.arm_ik = Arm_IK()
        
        # 启动订阅线程
        sub_pos_th = threading.Thread(target=self.SubPosThread, daemon=True)
        sub_pos_th.daemon = True
        sub_pos_th.start()
    
    def SubPosThread(self):
        # 创建订阅者，监听PosCmd类型的消息
        rospy.Subscriber('pin_pos_cmd', PosCmd, self.pos_cmd_callback)
        rospy.spin()

    def pos_cmd_callback(self, msg):
        # 获取PosCmd类型消息中的数据
        x = msg.x
        y = msg.y
        z = msg.z
        roll = msg.roll
        pitch = msg.pitch
        yaw = msg.yaw
        gripper = msg.gripper
        # 调用Arm_IK类的逆解函数
        self.arm_ik.get_ik_solution(x, y, z, roll, pitch, yaw, gripper)

def key_listener():
    global exit_flag
    if os.name == 'nt':
        while True:
            if msvcrt.kbhit():
                if msvcrt.getch().lower() == b'q':
                    exit_flag = True
                    print("exit...")
                    break
    else:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                if sys.stdin.read(1).lower() == 'q':
                    exit_flag = True
                    print("exit...")
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def clear_terminal():
    os.system("cls" if os.name == "nt" else "clear")

if __name__ == "__main__":
    piper_ik = C_PiperIK()
    print("Press 'q' to quit")
    listener_thread = threading.Thread(target=key_listener, daemon=True)
    listener_thread.start()
    while not exit_flag:
        time.sleep(0.1)

