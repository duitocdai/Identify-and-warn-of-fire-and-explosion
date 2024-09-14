import tempfile
import time
import cv2
import numpy as np
import streamlit as st
import io
from pathlib import Path
import telepot
import pygame
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape
import threading

token = '6385683960:AAEBx33B1sP5S8owuQOak8_b1Gx6NB574OI'
receiver_id =1388289037
bot = telepot.Bot(token)


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

    if area_change > (1 + SCALE_THRESHOLD * 2) or area_change < (1 - SCALE_THRESHOLD * 2):
        return False  
    return False

def initialize_heatmap(frame_shape):
    height, width = frame_shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)
    bbox_ages = []
    return heatmap, bbox_ages 

def update_heatmap(heatmap, bboxes, bbox_ages, max_age=10):
    new_bbox_ages = []
    for bbox, age in bbox_ages:
        if age < max_age:
            x1, y1, x2, y2 = bbox
            heatmap[y1:y2, x1:x2] += 1
            new_bbox_ages.append((bbox, age + 1))
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        heatmap[y1:y2, x1:x2] += 1
        new_bbox_ages.append((bbox, 0))
    return heatmap, new_bbox_ages

def apply_heatmap_overlay(frame, heatmap):
    heatmap_normalized = cv2.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)

def fire_smoke(video_source, tracking_classes, conf_threshold=0.5, use_webcam=False, use_heatmap=False):
    tracker = DeepSort(max_age=50, nms_max_overlap=0.45)  
    device = "cpu"
    model = DetectMultiBackend(weights="runs/train/exp5/weights/best.pt", device=device, fuse=True)
    model = AutoShape(model)
    max_bbox_area = 50000
    with open("data.ext/classes.names") as f:
        class_names = f.read().strip().split('\n')
    colors = { 
        '0': (72, 146, 234),   
        '1': (0, 238, 195),
        '2': (254, 78, 240)  
    }
    previous_bboxes = {}
    cap = cv2.VideoCapture(video_source)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_bbox_area = frame_width * frame_height * 0.25

    start_time = time.time()
    frame_count = 0

    heatmap, bbox_ages = initialize_heatmap((frame_height, frame_width))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        fps = frame_count / (current_time - start_time)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        detections = []
        bboxes = []

        fire_count = 0

        for detection in results.pred[0]:
            label, confidence, bbox = detection[5], detection[4], detection[:4]
            x1, y1, x2, y2 = map(int, bbox)
            class_id = int(label)

            if class_id in tracking_classes and confidence >= conf_threshold:
                detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])
                if class_id == 0:
                    fire_count += 1
                    bboxes.append((x1, y1, x2, y2))
                
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if track.is_confirmed() and track.time_since_update < 4:
                track_id = track.track_id
                ltrb = track.to_ltrb()
                if is_bbox_abnormal(ltrb, max_bbox_area):
                    continue
                class_id = track.get_det_class()
                x1, y1, x2, y2 = map(int, ltrb)
                if track_id in previous_bboxes and is_bbox_too_large((x1, y1, x2, y2), previous_bboxes[track_id]):
                    continue
                color = colors.get(str(class_id), (255, 0, 0))  
                B, G, R = color


                label = "{}-{:.2f}".format(class_names[class_id], confidence)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
                cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                previous_bboxes[track_id] = (x1, y1, x2, y2)  

        if use_heatmap:
            heatmap, bbox_ages = update_heatmap(heatmap, bboxes, bbox_ages)
            frame = apply_heatmap_overlay(frame, heatmap)

        end_time = time.time()
        fps_text = f"FPS: {1.0 / (end_time - start_time):.2f}"
        cv2.putText(frame, fps_text, (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv2.LINE_AA)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame, fps, fire_count

    cap.release()
def send_telegram_message_async(text, image=None):
    def send_message():
        bot.sendMessage(receiver_id, text)
        if image is not None:
            image_file = io.BytesIO(image)
            bot.sendPhoto(receiver_id, photo=image_file)

    threading.Thread(target=send_message).start()
def play_sound(file_path):
    pygame.mixer.init()
    sound = pygame.mixer.Sound(file_path)
    sound.play()

def main():
    st.title('Nhận dạng khói, lửa và cảnh báo cháy nổ')

    st.sidebar.title('Setting')

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    use_webcam = st.sidebar.checkbox('Use Webcam')
    use_heatmap = st.sidebar.checkbox('Use heatmap')
    st.sidebar.markdown('---')
    confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.25)
    st.sidebar.markdown('---')
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi", "asf", "m4v"])

    DEMO_VIDEO = 'video.mp4'

    if not video_file_buffer and not use_webcam:
        vid = cv2.VideoCapture(DEMO_VIDEO)
        tfflie_name = DEMO_VIDEO
        with open(tfflie_name, 'rb') as dem_vid:
            demo_bytes = dem_vid.read()
    
        st.sidebar.text('Input Video')
        st.sidebar.video(demo_bytes)
    elif not use_webcam:
        tfflie = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        tfflie.write(video_file_buffer.read())
        tfflie.seek(0)
        demo_bytes = tfflie.read()
    
        st.sidebar.text('Input Video')
        st.sidebar.video(demo_bytes)
        tfflie.close()
        tfflie_name = tfflie.name
    else:
        tfflie_name = 0  # Use webcam

    stframe = st.empty()
    
    st.markdown("<hr/>", unsafe_allow_html=True)
    kpi1, kpi2 = st.columns(2)

    with kpi1:
        st.markdown("**Tốc độ khung hình**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Có bao nhiêu đám cháy**")
        kpi2_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.text('Video is Processed')

    tracking_classes = [0, 1, 2]  # Thay đổi danh sách các lớp theo yêu cầu

    is_fire_alert_played = False
    last_fire_time = None

    for frame, fps, fire_count in fire_smoke(tfflie_name, tracking_classes=tracking_classes, conf_threshold=confidence, use_webcam=use_webcam, use_heatmap=use_heatmap):
        stframe.image(frame, channels="RGB")
        if fire_count > 0:
            kpi2_text.write(f"<h1 style='color: red;'>{fire_count}</h1>", unsafe_allow_html=True)
            kpi1_text.write(f"<h1 style='color: red;'>{int(fps)}</h1>", unsafe_allow_html=True)
            if not is_fire_alert_played:
                play_sound('data.ext/fire.wav')
                is_fire_alert_played = True
                last_fire_time = time.time()

                _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                image_bytes = img_encoded.tobytes()
                send_telegram_message_async("🔥 Phát hiện lửa!", image=image_bytes)
            else:
                time_since_last_fire = time.time() - last_fire_time
                if time_since_last_fire >= 5:
                    last_fire_time = time.time()

                    _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    image_bytes = img_encoded.tobytes()
                    send_telegram_message_async("🔥 Phát hiện lửa!", image=image_bytes)
        else:
            kpi2_text.write(f"<h1 style='color: green;'>{fire_count}</h1>", unsafe_allow_html=True)
            kpi1_text.write(f"<h1 style='color: green;'>{int(fps)}</h1>", unsafe_allow_html=True)

            if is_fire_alert_played:
                time_since_last_fire = time.time() - last_fire_time
                if time_since_last_fire >= 60:
                    play_sound('data.ext/fire.wav')
                    is_fire_alert_played = False
                    last_fire_end_time = time.time()

                    send_telegram_message_async("🚒 Lửa đã được dập tắt.")
    
if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
