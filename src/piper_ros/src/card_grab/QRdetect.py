import pyrealsense2 as rs
import numpy as np
import cv2
from pyzbar import pyzbar
import math

# 说明：
# 1) 使用 RealSense 同步获取深度+彩色帧，对齐后在彩色画面上检测二维码。
# 2) 优先用深度相机给出的 Z；若深度无效，用二维码在图像中的像素边长结合真实尺寸估算距离（针孔模型）。
# 3) 额外输出二维码倾斜角、预测框占全帧面积比例以及 XY 物理位置。

# 已知二维码实际边长（米），用于在深度失败时用成像尺寸估算距离；请按实际尺寸修改
QR_REAL_SIZE_M = 0.04  # 4 cm

# 初始化 RealSense 管道并开启深度/彩色流
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# 将深度帧对齐到彩色帧，保证二者坐标系一致
align_to = rs.stream.color
align = rs.align(align_to)

# 相机内参，用于像素->物理尺寸与针孔模型
color_profile = profile.get_stream(rs.stream.color)
color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
WIDTH = color_intrinsics.width  # 640
HEIGHT = color_intrinsics.height  # 480
fx = color_intrinsics.fx  # 水平焦距
fy = color_intrinsics.fy  # 垂直焦距
ppx = color_intrinsics.ppx  # 主点x坐标
ppy = color_intrinsics.ppy  # 主点y坐标

# 由内参推算水平/垂直视场角（FOV），用于把像素尺寸换算成物理尺寸
H_FOV = 2 * math.atan((WIDTH / 2) / fx) * (180 / math.pi)
V_FOV = 2 * math.atan((HEIGHT / 2) / fy) * (180 / math.pi)

print(f"计算出的水平FOV：{H_FOV:.2f}°，垂直FOV：{V_FOV:.2f}°")
print("开始检测二维码，按 'q' 退出...")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # 检测二维码
        qrcodes = pyzbar.decode(gray)

        for qr in qrcodes:
            (x, y, w, h) = qr.rect
            pts = qr.polygon
            if len(pts) > 0:
                pts_np = np.array([[p.x, p.y] for p in pts], dtype=np.int32)

                # 倾斜角度：最小外接矩形返回的旋转角，保证输出更直观
                rect = cv2.minAreaRect(pts_np)
                angle = rect[-1]
                if rect[1][0] < rect[1][1]:
                    angle += 90

                # 二维码中心的像素坐标（用于深度采样与物理距离计算）
                pixel_x = int(sum([p.x for p in pts]) / len(pts))
                pixel_y = int(sum([p.y for p in pts]) / len(pts))

                # 优先使用深度相机；若无效则用像素尺寸+真实尺寸估算 Z
                z = depth_frame.get_distance(pixel_x, pixel_y)
                depth_source = "depth"
                if z <= 0:
                    # 使用最小外接矩形的较大边作为二维码像素边长
                    qr_pixel_size = max(rect[1])
                    if qr_pixel_size <= 0:
                        continue
                    # 针孔模型：Z = (f * 实际尺寸) / 像素尺寸
                    z = (fx * QR_REAL_SIZE_M) / qr_pixel_size
                    depth_source = "size"

                # 将像素换算成物理尺度（米/像素）
                h_physical_width = 2 * z * math.tan(math.pi * H_FOV / 360)
                meter_per_pixel_x = h_physical_width / WIDTH
                v_physical_height = 2 * z * math.tan(math.pi * V_FOV / 360)
                meter_per_pixel_y = v_physical_height / HEIGHT

                # 相对于左上角的物理距离
                x_distance = pixel_x * meter_per_pixel_x
                y_distance = pixel_y * meter_per_pixel_y

                # 预测框占整帧面积比例，用于指示目标大小/距离粗感
                area_ratio = (w * h) / (WIDTH * HEIGHT)

                cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = (
                    f"{qr.data.decode('utf-8')} "
                    f"(X:{x_distance:.3f}m, Y:{y_distance:.3f}m, Z:{z:.3f}m[{depth_source}], "
                    f"Angle:{angle % 90:.1f}deg, Box:{area_ratio:.4f})"
                )
                cv2.putText(color_image, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                print(text)

        cv2.imshow("QR Detection", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
