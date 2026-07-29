# import pyrealsense2 as rs
# import numpy as np
# import cv2
#
# # ===============================
# # 1. RealSense 初始化
# # ===============================
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# pipeline.start(config)
#
# # ===============================
# # 2. ★就在这里改 ArUco 字典★
# # ===============================
# aruco_dict = cv2.aruco.getPredefinedDictionary(
#     cv2.aruco.DICT_6X6_250   # ← 你用的是什么字典，就改这里
# )
#
# # ArUco 参数（放宽，提升检测率）
# aruco_params = cv2.aruco.DetectorParameters()
# aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
# aruco_params.adaptiveThreshWinSizeMin = 3
# aruco_params.adaptiveThreshWinSizeMax = 53
# aruco_params.adaptiveThreshWinSizeStep = 4
#
# # 新版 OpenCV 的 Detector
# detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
#
# print("开始检测 ArUco，按 q 退出")
#
# # ===============================
# # 3. 主循环
# # ===============================
# try:
#     while True:
#         frames = pipeline.wait_for_frames()
#         color_frame = frames.get_color_frame()
#         if not color_frame:
#             continue
#
#         img = np.asanyarray(color_frame.get_data())
#
#         # 灰度 + 对比度增强（非常关键）
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         gray = cv2.equalizeHist(gray)
#
#         # ArUco 检测
#         corners, ids, _ = detector.detectMarkers(gray)
#
#         if ids is not None:
#             cv2.aruco.drawDetectedMarkers(img, corners, ids)
#             print("检测到 ArUco ID:", ids.flatten())
#
#         cv2.imshow("ArUco Detection", img)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
# finally:
#     pipeline.stop()
#     cv2.destroyAllWindows()
import pyrealsense2 as rs
import numpy as np
import cv2
import math

# ===============================
# ArUco 实际边长（米）
# ===============================
ARUCO_REAL_SIZE_M = 0.03  # 16 cm

# ===============================
# RealSense 初始化
# ===============================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

align = rs.align(rs.stream.color)

# ===============================
# 相机内参
# ===============================
color_profile = profile.get_stream(rs.stream.color)
intr = color_profile.as_video_stream_profile().get_intrinsics()

WIDTH = intr.width
HEIGHT = intr.height
fx = intr.fx
fy = intr.fy
ppx = intr.ppx
ppy = intr.ppy

H_FOV = 2 * math.atan((WIDTH / 2) / fx) * 180 / math.pi
V_FOV = 2 * math.atan((HEIGHT / 2) / fy) * 180 / math.pi

print(f"H_FOV={H_FOV:.2f}°, V_FOV={V_FOV:.2f}°")

# ===============================
# ArUco 初始化（★就在这里改字典）
# ===============================
aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_6X6_250
)
# cv2.aruco.DICT_4X4_50
# # 或
# cv2.aruco.DICT_5X5_100

aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

print("开始检测 ArUco，按 q 退出")

# ===============================
# 主循环
# ===============================
try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)

        # 左上角显示控制
        base_x = 10
        base_y = 25
        line_h = 22
        line_idx = 0

        if ids is not None:
            for i in range(len(ids)):
                c = corners[i][0]  # 4x2

                # ---------- 中心点 ----------
                cx = int(np.mean(c[:, 0]))
                cy = int(np.mean(c[:, 1]))

                # ---------- 深度 ----------
                z = depth_frame.get_distance(cx, cy)
                if z <= 0:
                    continue

                # ---------- 像素 → 物理 ----------
                width_m = 2 * z * math.tan(math.radians(H_FOV / 2))
                height_m = 2 * z * math.tan(math.radians(V_FOV / 2))

                meter_per_pixel_x = width_m / WIDTH
                meter_per_pixel_y = height_m / HEIGHT

                x_dist = cx * meter_per_pixel_x
                y_dist = cy * meter_per_pixel_y

                # ---------- ★ 角度计算（关键） ----------
                # ArUco 顺序：TL, TR, BR, BL
                tl = c[0]
                tr = c[1]

                dx = tr[0] - tl[0]
                dy = tr[1] - tl[1]

                angle = math.degrees(math.atan2(dy, dx))
                if angle < 0:
                    angle += 360.0

                # ---------- 画框 ----------
                cv2.polylines(
                    img,
                    [c.astype(np.int32)],
                    True,
                    (0, 255, 0),
                    2
                )

                # ---------- 左上角分行显示 ----------
                lines = [
                    f"ID: {ids[i][0]}",
                    f"X: {x_dist:.3f} m",
                    f"Y: {y_dist:.3f} m",
                    f"Z: {z:.3f} m",
                    f"Angle: {angle:.1f} deg"
                ]

                for line in lines:
                    cv2.putText(
                        img,
                        line,
                        (base_x, base_y + line_idx * line_h),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2
                    )
                    line_idx += 1

                print(" | ".join(lines))

        cv2.imshow("ArUco Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
