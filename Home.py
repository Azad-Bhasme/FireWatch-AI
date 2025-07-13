import streamlit as st
import cv2
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import uuid
import shutil
import time

# Clean up old results
try:
    shutil.rmtree("runs")
except Exception:
    pass

# Load model
model = YOLO("best.pt")

# Page setup
st.set_page_config(page_title="FireWatch AI", page_icon="🔥")
st.markdown("<h1 style='text-align: center;'>🔥 FireWatch AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Smart detection, safer protection. 🔥🧠</h4>", unsafe_allow_html=True)

# Select mode
mode = st.radio("Choose Input Mode:", ["📤 Upload file", "📷 Use Webcam"], horizontal=True)

# 🔍 Fire Detection Function
def detect_fire(path, is_video=False):
    results = model.predict(source=path, save=True, conf=0.75)
    return results[0].save_dir


# 📤 Upload Mode
if mode == "📤 Upload file":
    uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi"])

    if uploaded_file is not None:
        suffix = uploaded_file.name.split('.')[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.success("✅ File uploaded. Running detection ⭕")

        result_dir = detect_fire(tmp_path, is_video=suffix in ["mp4", "avi"])
        st.session_state["result_dir"] = result_dir

        if st.button("👀 View Your Result"):
            found = False
            for file in os.listdir(result_dir):
                file_path = os.path.join(result_dir, file)
                if file.endswith((".jpg", ".jpeg", ".png")):
                    st.image(file_path, caption="Detected Fire 🔥", use_container_width=True)
                    found = True
                elif file.endswith((".mp4", ".avi")):
                    st.video(file_path)
                    found = True

            if not found:
                st.warning("✅ Detection done, but no output file found.")

        # Download Button(s)
        for file in os.listdir(result_dir):
            file_path = os.path.join(result_dir, file)
            with open(file_path, "rb") as f:
                st.download_button("📥 Download Result", data=f, file_name=file, mime="application/octet-stream")


# 📷 Webcam Mode
elif mode == "📷 Use Webcam":
    st.warning("⚠️ Webcam access works only on your **local machine**.")

    if st.button("📸 Start Camera Detection"):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        progress = st.progress(0)
        fire_message = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to access webcam.")
                break

            results = model.predict(source=frame, conf=0.6)
            annotated_frame = results[0].plot()

            # Detect fire (assuming class 0 is 'fire')
            fire_detected = any(cls == 0 for cls in results[0].boxes.cls)

            # Update UI
            progress.progress(100 if fire_detected else 0)
            fire_message.info("🔥 Fire Detected!" if fire_detected else "No Fire Detected")
            stframe.image(annotated_frame, channels="BGR")

            # Break loop on user interrupt
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.03)

        cap.release()


# Add this to the bottom of your app.py file
# ✨ Stylish Footer with blinking ❤️ and centered text
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Quicksand:wght@500&display=swap" rel="stylesheet">
    <style>
        .footer {
            position: fixed;
            bottom: 20px;
            left: 0;
            width: 100%;
            text-align: center;
            font-family: 'Quicksand', sans-serif;
            font-size: 15px;
            color: #fff;
        }
        .blink-heart {
            animation: blink 1s infinite;
            color: #ff4b4b;
        }
        @keyframes blink {
            50% { opacity: 0; }
        }
    </style>
    <div class="footer">made with <span class="blink-heart">❤️</span> by Azad Bhasme</div>
""", unsafe_allow_html=True)
