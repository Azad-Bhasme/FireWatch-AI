import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import cv2
import time
import shutil

# App Config
st.set_page_config(page_title="FireWatch AI", page_icon="🔥")
st.title("🔥 FireWatch AI")
st.markdown("### Smart detection, safer protection. 🔥🧠")

# Load Model
model = YOLO("best.pt")

# Input Mode Selection
mode = st.radio("Choose Input Mode:", ["📤 Upload file", "📷 Use webcam"])

# 📤 File Upload Mode
if mode == "📤 Upload file":
    uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mpeg4"])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        st.success("✅ File uploaded. Running detection...")

        try:
            results = model.predict(source=temp_file_path, save=True, conf=0.6)
            output_path = results[0].save_dir

            if st.button("🎥 View Your Result"):
                output_files = [f for f in os.listdir(output_path) if f.lower().endswith((".jpg", ".png", ".mp4", ".avi"))]

                if not output_files:
                    st.warning("⚠️ No result files found.")
                else:
                    for file in output_files:
                        full_path = os.path.join(output_path, file)
                        temp_output = os.path.join(tempfile.gettempdir(), file)
                        shutil.copy(full_path, temp_output)

                        if file.lower().endswith((".jpg", ".png")):
                            st.image(temp_output, caption="🖼️ Detected Image")
                        elif file.lower().endswith((".mp4", ".avi")):
                            st.video(temp_output)

        except Exception as e:
            st.error(f"❌ Detection failed: {e}")

# 📷 Webcam Mode
elif mode == "📷 Use webcam":
    start = st.button("🎬 Start Live Detection")
    stop = st.button("🛑 Stop Detection", key="stop_live")

    if start:
        stframe = st.empty()
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Unable to access the webcam.")
        else:
            st.success("✅ Webcam is live. Detecting fire...")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Failed to read from webcam.")
                    break

                results = model.predict(source=frame, imgsz=640, conf=0.6, verbose=False)
                annotated_frame = results[0].plot()

                stframe.image(annotated_frame, channels="BGR")

                if stop:
                    cap.release()
                    st.success("🛑 Detection stopped.")
                    break

                time.sleep(0.03)

# Footer
st.markdown("---")
st.markdown("<center><sub>made with ❤️ by Azad Bhasme</sub></center>", unsafe_allow_html=True)


st.markdown("---")
st.markdown("<center><sub>made with ❤️ by Azad Bhasme</sub></center>", unsafe_allow_html=True)

