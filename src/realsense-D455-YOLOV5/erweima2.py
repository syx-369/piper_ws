import pyrealsense2 as rs
import numpy as np
import cv2
from pyzbar import pyzbar
import csv
from datetime import datetime

# -------------------------- 配置参数 --------------------------
VIDEO_SAVE_PATH = "qr_detection_video1.avi"  # 视频保存路径
DATA_SAVE_PATH = "qrcode_data.csv"          # 二维码数据保存路径
FRAME_WIDTH = 640                           # 帧宽度
FRAME_HEIGHT = 480                          # 帧高度
FPS = 30                                    # 视频帧率

# -------------------------- 初始化RealSense --------------------------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, FPS)
config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FPS)
profile = pipeline.start(config)

# 对齐深度帧到彩色帧
align_to = rs.stream.color
align = rs.align(align_to)

# -------------------------- 初始化视频写入器 --------------------------
# 编码格式：Windows推荐DIVX，Linux/Mac推荐XVID
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(
    VIDEO_SAVE_PATH,
    fourcc,
    FPS,
    (FRAME_WIDTH, FRAME_HEIGHT)  # 必须和帧的分辨率一致
)

# -------------------------- 初始化CSV数据文件 --------------------------
with open(DATA_SAVE_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["QR Content", "X Center", "Y Center", "Depth (m)", "Detection Time"])

print("开始检测二维码，按 'q' 退出...")
print(f"视频将保存到: {VIDEO_SAVE_PATH}")
print(f"数据将保存到: {DATA_SAVE_PATH}")

try:
    while True:
        # 获取对齐后的帧
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # 转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # 检测二维码
        qrcodes = pyzbar.decode(gray)

        for qr in qrcodes:
            (x, y, w, h) = qr.rect
            pts = qr.polygon
            if len(pts) > 0:
                # 计算二维码中心坐标
                x_center = int(sum([p.x for p in pts]) / len(pts))
                y_center = int(sum([p.y for p in pts]) / len(pts))
                # 获取中心位置的深度
                depth = depth_frame.get_distance(x_center, y_center)
                # 解码二维码内容
                qr_content = qr.data.decode('utf-8')
                # 获取检测时间
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # 绘制检测框和信息
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = f"{qr_content} ({x_center},{y_center},{depth:.3f}m)"
                cv2.putText(
                    color_image, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
                )

                # 保存数据到CSV
                with open(DATA_SAVE_PATH, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([qr_content, x_center, y_center, round(depth, 3), current_time])

                print(text)

        # 将当前帧写入视频
        video_writer.write(color_image)
        # 显示实时画面
        cv2.imshow("QR Detection", color_image)

        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 必须释放所有资源
    pipeline.stop()
    video_writer.release()  # 关键：释放视频写入器，否则视频无法播放
    cv2.destroyAllWindows()
    print("程序结束，资源已释放！")