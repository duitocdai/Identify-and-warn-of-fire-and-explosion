import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape
import telepot
from threading import Timer

def is_bbox_abnormal(bbox, max_area):
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    return area > max_area

SCALE_THRESHOLD = 1

def is_bbox_too_large(bbox, previous_bbox):
    x1, y1, x2, y2 = bbox
    prev_x1, prev_y1, prev_x2, prev_y2 = previous_bbox
    current_area = (x2 - x1) * (y2 - y1)
    previous_area = (prev_x2 - prev_x1) * (prev_y2 - prev_y1)
    area_change = current_area / previous_area

    if area_change > (1 + SCALE_THRESHOLD * 2) or area_change < (1 - SCALE_THRESHOLD * 2):
        return False  # Vẫn tiếp tục theo dõi đối tượng thay vì bỏ qua
    return False

# Giá trị cấu hình
video_path = "C:/streamlit/yolov9/data.ext/VideoNightMixt.mp4"
conf_threshold = 0.5
tracking_class = [0, 1, 2]  # IDs của các lớp cần theo dõi

# Màu sắc cho các lớp
colors = {
    'smoke': (254, 78, 240),  # Màu sắc cho smoke (FE4EF0)
    'fire': (72, 146, 234),   # Màu sắc cho fire (4892EA)
    'other': (0, 238, 195)    # Màu sắc cho other (00EEC3)
}

# Khởi tạo DeepSort
tracker = DeepSort(max_age=30)

# Khởi tạo YOLOv9
device = "cpu"  # "cuda": GPU, "cpu": CPU, "mps:0"
model = DetectMultiBackend(weights="C:/streamlit/yolov9/runs/train/exp5/weights/last.pt", device=device, fuse=True)
model = AutoShape(model)

# Tải tên class từ file classes.names
with open("C:/streamlit/yolov9/data.ext/classes.names") as f:
    class_names = f.read().strip().split('\n')

tracks = []

# Khởi tạo VideoCapture để đọc từ file video
cap = cv2.VideoCapture(video_path)

# Tiến hành đọc từng frame từ video
while True:
    # Đọc
    ret, frame = cap.read()
    if not ret:
        break  # Thoát vòng lặp nếu không đọc được frame

    # Chuyển đổi frame sang định dạng RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Đưa qua model để detect
    results = model(rgb_frame)

    detect = []
    for detect_object in results.pred[0]:
        label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
        x1, y1, x2, y2 = map(int, bbox)
        class_id = int(label)

        if class_id in tracking_class and confidence >= conf_threshold:
            detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

    # Cập nhật và gán ID bằng DeepSort
    tracks = tracker.update_tracks(detect, frame=rgb_frame)

    # Vẽ các khung chữ nhật kèm ID lên màn hình
    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id

            # Lấy tọa độ, class_id để vẽ lên hình ảnh
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)

            # Sử dụng màu sắc cho từng lớp
            class_name = class_names[class_id]
            color = colors.get(class_name, (255, 255, 255))  # Màu sắc mặc định nếu lớp không có trong colors
            
            B, G, R = color
            label = "{}-{}".format(class_name, track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Hiển thị hình ảnh lên màn hình
    cv2.imshow("OT", frame)
    
    # Bấm Q để thoát
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
