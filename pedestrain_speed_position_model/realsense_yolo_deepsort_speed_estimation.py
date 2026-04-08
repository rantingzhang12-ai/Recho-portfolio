# ============================================
# 1. 导入库
# ============================================

import pyrealsense2 as rs
import numpy as np
import cv2
import time
import csv
from collections import deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# ============================================
# 2. 初始化 RealSense
# ============================================

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

# ⭐ 新增部分
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()  # 获取深度比例
print("Depth Scale:", depth_scale)

depth_stream = profile.get_stream(rs.stream.depth)
depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()


# ============================================
# 3. 加载模型
# ============================================

model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)
# max_age=30 表示：
# 如果一个人连续30帧没被检测到，才删除这个ID
# 假设 30FPS： 30帧 ≈ 1秒
# 也就是说：人被短暂遮挡 1 秒以内  ID 不会丢失
# 1️⃣ 运动预测（卡尔曼滤波）
# 2️⃣ 外观特征提取（深度特征）
# 3️⃣ 最优匹配（匈牙利算法）


# ============================================
# 4. 初始化数据结构
# ============================================

track_history = {}           # 存储三维历史位置
speed_history = {}           # 存储速度历史用于滑动平均
kalman_filters = {}          # 每个ID一个卡尔曼滤波器

previous_time = time.time()

# 创建CSV文件
csv_file = open("pedestrian_data.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)

# csv_writer.writerow(["Time", "ID", "X", "Y", "Z", "Speed"])
csv_writer.writerow([
    "Timestamp",
    "ID",
    "X_raw", "Y_raw", "Z_raw",
    "X_kf", "Y_kf", "Z_kf",
    "Speed_raw",
    "Speed_filtered"
])


# ============================================
# 5. 创建卡尔曼滤波器函数
# ============================================

def create_kalman():
    kf = cv2.KalmanFilter(6, 3)

    # 状态：[x,y,z,vx,vy,vz]
    # 测量：[x,y,z]

    # kf.transitionMatrix = np.array([
    #     [1,0,0,1,0,0],
    #     [0,1,0,0,1,0],
    #     [0,0,1,0,0,1],
    #     [0,0,0,1,0,0],
    #     [0,0,0,0,1,0],
    #     [0,0,0,0,0,1]
    # ], np.float32)          放入主循环，根据实时帧率更新▲t

    kf.measurementMatrix = np.array([
        [1,0,0,0,0,0],
        [0,1,0,0,0,0],
        [0,0,1,0,0,0]
    ], np.float32)

    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.5

    return kf


# ============================================
# 6. 主循环
# 采集数据 → 检测 → 跟踪 → 三维重建 → 滤波 → 速度计算 → 记录 → 显示
# ============================================

while True:

    # 获取相机数据
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()  # 获取深色图像 用来算真实距离
    color_frame = frames.get_color_frame()  # 获取彩色图像 用来做YOLO检测

    if not depth_frame or not color_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())  # 把相机数据转换为 numpy 数组

    current_time = time.time()
    delta_time = current_time - previous_time   # Δt=当前帧时间−上一帧时间
    previous_time = current_time

    # ============================================
    # YOLO 检测
    # ============================================

    results = model(color_image)
    detections = []  # 准备给 DeepSort 的数据格式

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])   # 获取该检测框置信度最高的类别ID：0 = person

            if cls == 0:  # person
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 检测框坐标
                conf = float(box.conf[0])  # 置信度
                detections.append(([x1, y1, x2-x1, y2-y1], conf, 'person'))
                # 转成 DeepSort 需要的格式： ([左上x, 左上y, 宽, 高], 置信度, 类别)

    tracks = tracker.update_tracks(detections, frame=color_image)   # tracker 是一个“有记忆的对象”，它内部保存了30帧历史状态。
    # 1️⃣ 匹配上一帧ID
    # 2️⃣ 分配当前帧ID
    # 3️⃣ 返回所有跟踪目标

    for track in tracks:

        if not track.is_confirmed(): # 过滤掉不稳定目标。
            continue

        track_id = track.track_id
        l, t, w, h = map(int, track.to_ltrb())

        # 计算目标中心点
        center_x = int(l + w/2)
        center_y = int(t + h/2)

        # depth = depth_frame.get_distance(center_x, center_y)  # 获取这个像素点（人的中心点）的深度：也就是距离（米）

        # depth_region = depth_frame.get_data()
        # depth_array = np.asanyarray(depth_region)
        #
        # roi = depth_array[t:t+h, l:l+w]
        # valid_depth = roi[roi > 0]
        #
        # if len(valid_depth) > 0:
        #     depth = np.median(valid_depth) * depth_scale
        # else:
        #     continue
        # 取检测框区域的“中值深度”
        # 优点：
        # 抗噪声 抗异常值 不容易跳变 ；比单点稳定很多。

        # 优化后的工业场景测距算法
        depth_image = np.asanyarray(depth_frame.get_data())

        # 取下半部分
        roi = depth_image[t + int(h * 0.5):t + h, l:l + w]

        valid_depth = roi[(roi > 0) & (roi < 10000)]  # 去异常值

        if len(valid_depth) > 0:
            depth = np.median(valid_depth) * depth_scale
        else:
            continue

        if depth == 0:   # 深度偶尔为0，则跳过
            continue

        # 计算三维坐标： 三维空间重建
        X = (center_x - depth_intrinsics.ppx) * depth / depth_intrinsics.fx
        Y = (center_y - depth_intrinsics.ppy) * depth / depth_intrinsics.fy
        Z = depth


        # ============================================
        # 卡尔曼滤波
        # ============================================

        measurement = np.array([[np.float32(X)],
                                [np.float32(Y)],
                                [np.float32(Z)]])  # 构造测量值

        if track_id not in kalman_filters:
            kalman_filters[track_id] = create_kalman()

        kf = kalman_filters[track_id]

        dt = max(0.001, min(delta_time, 0.1))

        kf.transitionMatrix = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], np.float32)

        prediction = kf.predict() # 预测当前状态
        kf.correct(measurement)   # 用测量值修正


        Xf, Yf, Zf = prediction[0][0], prediction[1][0], prediction[2][0] # 得到滤波后位置

        # ============================================
        # 速度计算
        # ============================================

        if track_id in track_history:  # 如果之前存在该ID。

            prev = track_history[track_id]
            dx = Xf - prev[0]
            dy = Yf - prev[1]
            dz = Zf - prev[2]  # 计算三维位移

            distance = np.sqrt(dx*dx + dy*dy + dz*dz)  # 勾股定理计算距离
            speed = distance / delta_time

        else:
            speed = 0

        track_history[track_id] = (Xf, Yf, Zf)

        # ============================================
        # 滑动平均滤波
        # ============================================

        if track_id not in speed_history:
            speed_history[track_id] = deque(maxlen=5)    # 保存最近5帧速度，每个ID创建一个长度为5的队列

        speed_history[track_id].append(speed)
        speed_filtered = np.mean(speed_history[track_id])

        # ============================================
        # 写入CSV
        # ============================================

        # csv_writer.writerow([current_time, track_id,
        #                      Xf, Yf, Zf, speed_filtered])

        # 优化后csv论文级数据记录
        csv_writer.writerow([
            current_time,
            track_id,
            X, Y, Z,  # 原始坐标
            Xf, Yf, Zf,  # 卡尔曼坐标
            speed,  # 原始速度
            speed_filtered  # 滑动平均速度
        ])

        # ============================================
        # 显示
        # ============================================

        cv2.rectangle(color_image, (l, t), (l+w, t+h), (0,255,0), 2)

        text = f"ID:{track_id}"
        pos_text = f"X:{Xf:.2f} Y:{Yf:.2f} Z:{Zf:.2f}"
        speed_text = f"V:{speed_filtered:.2f}m/s"

        cv2.putText(color_image, text, (l, t-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.putText(color_image, pos_text, (l, t-25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.putText(color_image, speed_text, (l, t-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imshow("Enhanced Pedestrian Detection", color_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ============================================
# 7. 释放资源
# ============================================

pipeline.stop()
csv_file.close()
cv2.destroyAllWindows()
