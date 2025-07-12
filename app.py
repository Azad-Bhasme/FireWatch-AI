import streamlit as st
import cv2
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import uuid
import shutil

# Clean up old results on each run
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
    results = model.predict(source=path, save=True, conf=0.5)
    output_dir = results[0].save_dir
    return output_dir


# 📤 Upload File
if mode == "📤 Upload file":
    uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi"])

    if uploaded_file is not None:
        suffix = uploaded_file.name.split('.')[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.success("✅ File uploaded. Running detection..⭕")

        # Run detection
        result_dir = detect_fire(tmp_path, is_video=suffix in ["mp4", "avi"])
        st.session_state["result_dir"] = result_dir  # Store for view/download buttons

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
                st.warning("✅ Detection done, but no output found!")

        # Download Button
        for file in os.listdir(result_dir):
            file_path = os.path.join(result_dir, file)
            with open(file_path, "rb") as f:
                st.download_button("📥 Download Result", data=f, file_name=file, mime="application/octet-stream")


# 📷 Webcam (Local Only)
elif mode == "📷 Use Webcam":
    st.warning("⚠️ Webcam works only in **local environments** (not Streamlit Cloud).")

    if st.button("🎥 Start Camera Detection"):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        stop_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to read from webcam.")
                break

            # YOLO inference
            results = model.predict(frame, conf=0.5, stream=True)
            for r in results:
                annotated = r.plot()
                stframe.image(annotated, caption="Live Fire Detection", use_container_width=True)

            # Stop button rendered once
            if stop_placeholder.button("🛑 Stop Camera"):
                break

        cap.release()
        stop_placeholder.empty()

# ✨ Footer with blinking ❤️ only
st.markdown("""
<style>
.footer {
    text-align: center;
    font-size: 13px;
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
