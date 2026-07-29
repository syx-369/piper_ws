#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyrealsense2 as rs
import numpy as np
import cv2
import math

# ===============================
# ArUco 实际边长（米）
# ===============================
ARUCO_REAL_SIZE_M = 0.03  # 3 cm

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
# ArUco 初始化
# ===============================
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# ===============================
# 蓝色盒子 HSV 阈值
# ===============================
lower_blue = np.array([100, 120, 70])
upper_blue = np.array([140, 255, 255])
kernel = np.ones((5, 5), np.uint8)

print("开始检测 ArUco + 蓝色盒子，按 q 退出")

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

        # -----------------------------
        # ArUco 检测
        # -----------------------------
        corners, ids, _ = detector.detectMarkers(gray)

        base_x = 10
        base_y = 25
        line_h = 22
        line_idx = 0

        if ids is not None:
            for i in range(len(ids)):
                c = corners[i][0]

                cx = int(np.mean(c[:, 0]))
                cy = int(np.mean(c[:, 1]))

                z = depth_frame.get_distance(cx, cy)
                if z == 0:
                    continue

                width_m = 2 * z * math.tan(math.radians(H_FOV / 2))
                height_m = 2 * z * math.tan(math.radians(V_FOV / 2))
                meter_per_pixel_x = width_m / WIDTH
                meter_per_pixel_y = height_m / HEIGHT

                x_dist = cx * meter_per_pixel_x
                y_dist = cy * meter_per_pixel_y

                tl = c[0]
                tr = c[1]
                dx = tr[0] - tl[0]
                dy = tr[1] - tl[1]
                angle = math.degrees(math.atan2(dy, dx))
                if angle < 0:
                    angle += 360.0

                cv2.polylines(img, [c.astype(np.int32)], True, (0, 255, 0), 2)

                lines = [
                    f"ID {ids[i][0]}",
                    f"X {x_dist:.3f} m",
                    f"Y {y_dist:.3f} m",
                    f"Z {z:.3f} m",
                    f"Angle {angle:.1f} deg"
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

        # -----------------------------
        # 蓝色盒子检测（粗定位）
        # -----------------------------
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(cnt) > 1500:
                x, y, w, h = cv2.boundingRect(cnt)
                cx_box = int(x + w / 2)
                cy_box = int(y + h / 2)

                depth_box = depth_frame.get_distance(cx_box, cy_box)
                if depth_box > 0:
                    X, Y, Z = rs.rs2_deproject_pixel_to_point(
                        intr, [cx_box, cy_box], depth_box
                    )

                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.circle(img, (cx_box, cy_box), 5, (0, 0, 255), -1)
                    cv2.putText(
                        img,
                        f"Box X{X:.2f} Y{Y:.2f} Z{Z:.2f}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2
                    )

        cv2.imshow("ArUco + Blue Box", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

