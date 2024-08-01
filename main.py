import cv2
import numpy as np

from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape

def is_bbox_abnormal(bbox, max_area):
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    return area > max_area

SCALE_THRESHOLD = 0.2

def is_bbox_too_large(bbox, previous_bbox):
    x1, y1, x2, y2 = bbox
    prev_x1, prev_y1, prev_x2, prev_y2 = previous_bbox
    current_area = (x2 - x1) * (y2 - y1)
    previous_area = (prev_x2 - prev_x1) * (prev_y2 - prev_y1)
    area_change = current_area / previous_area
    return area_change > (1 + SCALE_THRESHOLD) or area_change < (1 - SCALE_THRESHOLD)

def counter_sheep(video_path, tracking_classes, conf_threshold=0.5, line_x=50):
    tracker = DeepSort(max_age=75, nms_max_overlap=0.2)
    device = "cpu"
    model = DetectMultiBackend(weights="C:/streamlit/yolov9/runs/train/exp5/weights/best.pt", device=device, fuse=True)
    model = AutoShape(model)

    # Tải tên các lớp
    with open("C:\streamlit\yolov9\data.ext\classes.names") as f:
        class_names = f.read().strip().split('\n')
    colors = np.random.randint(0, 255, size=(len(class_names), 3))
    previous_bboxes = []
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_bbox_area = frame_width * frame_height * 0.25
    # Đọc từng khung hình từ video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Phát hiện các đối tượng sử dụng YOLOv9
        results = model(frame)
        detections = []

        for detection in results.pred[0]:
            label, confidence, bbox = detection[5], detection[4], detection[:4]
            x1, y1, x2, y2 = map(int, bbox)
            class_id = int(label)

            # Kiểm tra xem đối tượng có phải là một trong các lớp cần theo dõi và vượt qua ngưỡng tin cậy không
            if class_id in tracking_classes and confidence >= conf_threshold:
                detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

        # Cập nhật các đối tượng sử dụng DeepSort
        tracks = tracker.update_tracks(detections, frame=frame)

        # Vẽ các khung hình và id
        for track in tracks:
            if track.is_confirmed() and track.time_since_update < 4:
                track_id = track.track_id

                # Lấy toạ độ, class_id để vẽ lên hình ảnh
                ltrb = track.to_ltrb()
                if is_bbox_abnormal(ltrb, max_bbox_area):  # Loại bỏ các bbox to đột ngột
                    continue
                class_id = track.get_det_class()
                x1, y1, x2, y2 = map(int, ltrb)
                if track_id in previous_bboxes and is_bbox_too_large((x1, y1, x2, y2), previous_bboxes[track_id]):
                    continue
                color = colors[class_id]
                B, G, R = map(int, color)

                label = "{}-{}".format(class_names[class_id], track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color.tolist(), 2)
                cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), color.tolist(), -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Hiển thị video
        cv2.imshow("VIDEO", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Gọi hàm counter_sheep để chạy đoạn mã với các lớp theo dõi là fire, smoke, other
tracking_classes = [0, 1, 2]  # Giả sử class_id cho fire, smoke, other lần lượt là 0, 1, 2
counter_sheep("C:\streamlit\yolov9\data.ext\VideoNightMixt.mp4", tracking_classes=tracking_classes, conf_threshold=0.5, line_x=50)
