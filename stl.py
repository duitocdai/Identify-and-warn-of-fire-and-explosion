import tempfile
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import streamlit as st

def main():
    # Title
    st.title('Nhận dạng khói , lửa và cảnh báo cháy nổ')

    # Sidebar title
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

    save_img = st.sidebar.checkbox('Save Video')
    enable_GPU = st.sidebar.checkbox('Enable GPU')

    custom_classes = st.sidebar.checkbox('Use Custom Classes')
    assigned_class_id = []
    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi", "asf", "m4v"])

    DEMO_VIDEO = 'video.mp4'

    # We get our input video here
    if not video_file_buffer:
        if use_webcam:
            vid = cv2.VideoCapture(0)
            tfflie_name = 0
        else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie_name = DEMO_VIDEO
            with open(tfflie_name, 'rb') as dem_vid:
                demo_bytes = dem_vid.read()
    
            st.sidebar.text('Input Video')
            st.sidebar.video(demo_bytes)
    else:
        tfflie = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        tfflie.write(video_file_buffer.read())
        tfflie.seek(0)
        demo_bytes = tfflie.read()
    
        st.sidebar.text('Input Video')
        st.sidebar.video(demo_bytes)
        tfflie.close()

    stframe = st.empty()
    
    st.markdown("<hr/>", unsafe_allow_html=True)
    kpi1, kpi2, kpi3 = st.columns(3)  # Updated to use st.columns

    with kpi1:
        st.markdown("**Frame Rate**")
        kpi1_text = st.markdown("0")

    with kpi2:
        st.markdown("**Tracked Objects**")
        kpi2_text = st.markdown("0")

    with kpi3:
        st.markdown("**Total Count**")
        kpi3_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.text('Video is Processed')

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
