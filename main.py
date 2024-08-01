import tempfile
import os
import time
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape
import pygame
from threading import Timer

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

def fire_smoke(video_source, tracking_classes, conf_threshold=0.5, line_x=50, use_webcam=False):
    tracker = DeepSort(max_age=75, nms_max_overlap=0.2)
    device = "cpu"
    model = DetectMultiBackend(weights="C:/streamlit/yolov9/runs/train/exp5/weights/best.pt", device=device, fuse=True)
    model = AutoShape(model)

    with open("C:/streamlit/yolov9/data.ext/classes.names") as f:
        class_names = f.read().strip().split('\n')
    colors = np.random.randint(0, 255, size=(len(class_names), 3))
    previous_bboxes = []
    cap = cv2.VideoCapture(video_source)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_bbox_area = frame_width * frame_height * 0.25

    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = time.time()
        fps = frame_count / (current_time - start_time)

        results = model(frame)
        detections = []

        fire_count = 0

        for detection in results.pred[0]:
            label, confidence, bbox = detection[5], detection[4], detection[:4]
            x1, y1, x2, y2 = map(int, bbox)
            class_id = int(label)

            if class_id in tracking_classes and confidence >= conf_threshold:
                detections.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])
                if class_id == 1:  # Assuming class_id 1 is for fire
                    fire_count += 1

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
                color = colors[class_id]
                B, G, R = map(int, color)

                label = "{}-{}".format(class_names[class_id], track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color.tolist(), 2)
                cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), color.tolist(), -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame, fps, fire_count

    cap.release()

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

    tracking_classes = [0, 1, 2]  # Giả sử class_id cho fire, smoke, other lần lượt là 0, 1, 2

    fire_detected = False
    fire_alert_played = False

    for frame, fps, fire_count in fire_smoke(tfflie_name, tracking_classes=tracking_classes, conf_threshold=confidence, line_x=50, use_webcam=use_webcam):
        stframe.image(frame, channels="RGB")
        kpi1_text.markdown(f"**{fps:.2f} FPS**")
        kpi2_text.markdown(f"**{fire_count} Đám cháy**")

        if fire_count > 0:
            if not fire_detected:
                fire_detected = True
                fire_alert_played = True
                # Play the alert sound
                Timer(5.0, lambda: play_sound("C:\\streamlit\\yolov9\\data.ext\\fire.wav")).start()
        else:
            if fire_detected and fire_alert_played:
                fire_detected = False
                fire_alert_played = False
                # Play the end fire sound
                play_sound("C:\\streamlit\\yolov9\\data.ext\\endfire.wav")

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
