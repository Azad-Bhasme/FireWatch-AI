import streamlit as st
import cv2
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# Page setup
st.set_page_config(page_title="FireWatch AI", page_icon="🔥")
st.markdown("<h1 style='text-align: center;'>🔥 FireWatch AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Smart detection, safer protection. 🔥🧠</h4>", unsafe_allow_html=True)

# Load YOLOv8 model
model = YOLO("best.pt")

# Input mode selection
input_mode = st.radio("Choose Input Mode:", ["📤 Upload file", "📸 Use webcam"], horizontal=True)

# --- Upload Mode ---
if input_mode == "📤 Upload file":
    uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mpeg4"])

    if uploaded_file is not None:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        st.success("✅ File uploaded. Detecting fire...")

        try:
            results = model.predict(source=temp_path, save=True, conf=0.5)
            save_dir = results[0].save_dir

            video_shown = False
            for f in os.listdir(save_dir):
                file_path = os.path.join(save_dir, f)
                if f.endswith((".jpg", ".jpeg", ".png")):
                    st.image(file_path, caption="Fire Detected 🔥", use_column_width=True)
                elif f.endswith((".mp4", ".avi")):
                    st.video(file_path)
                    video_shown = True

            if not video_shown:
                st.warning("✅ Detection completed, but no video was generated.")
        except Exception as e:
            st.error(f"❌ Detection failed: {e}")

# --- Webcam Mode (Local only) ---
elif input_mode == "📸 Use webcam":
    st.warning("⚠️ Webcam only works in **local environments**. Streamlit Cloud does not support real-time webcam access.")

    if st.button("Start Live Detection (Local Only)"):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to capture image from webcam.")
                break

            results = model.predict(frame, conf=0.5)
            res_plotted = results[0].plot()
            stframe.image(res_plotted, channels="BGR", use_column_width=True)

            if st.button("Stop"):
                break

        cap.release()

# --- Blinking Footer Tagline ---
st.markdown("""
    <style>
    .blinking {
        animation: blinker 1s linear infinite;
        color: #ff4b4b;
        font-weight: bold;
        text-align: center;
        font-size: 14px;
    }
    @keyframes blinker {
        50% { opacity: 0; }
    }
    </style>
    <p class="blinking">made with ❤️ by Azad Bhasme</p>
""", unsafe_allow_html=True)
