import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import cv2
import time

st.set_page_config(page_title="FireWatch AI", page_icon="🔥")
st.title("🔥 FireWatch AI")
st.markdown("### Smart detection, safer protection. 🔥🧠")

model = YOLO("best.pt")

mode = st.radio("Select Detection Mode:", ["Upload Image/Video", "Use Webcam 🔴"])

if mode == "Upload Image/Video":
    uploaded_file = st.file_uploader("📤 Upload image or video", type=["jpg", "jpeg", "png", "mp4", "avi"])

    if uploaded_file is not None:
        ext = uploaded_file.name.split('.')[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        st.success("✅ File uploaded. Detecting fire...")
        results = model.predict(source=temp_path, save=True, conf=0.6)
        output_path = results[0].save_dir

        for file in os.listdir(output_path):
            if file.lower().endswith((".jpg", ".png")):
                st.image(os.path.join(output_path, file))
            elif file.lower().endswith((".mp4", ".avi")):
                st.video(os.path.join(output_path, file))

        if st.button("📂 View Detection Folder"):
            st.markdown(f"📁 Results saved to: `{output_path}`")

elif mode == "Use Webcam 🔴":
    st.warning("Click below to start your **live fire detection** via webcam.")
    start_cam = st.button("🎥 Start Camera")

    if start_cam:
        stframe = st.empty()
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Webcam not detected. Please connect a camera.")
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from webcam.")
                    break

                results = model.predict(source=frame, conf=0.6, verbose=False)
                frame_with_boxes = results[0].plot()
                stframe.image(frame_with_boxes, channels="BGR")

                time.sleep(0.1)

            cap.release()

st.markdown("---")
st.markdown("<center><sub>made with ❤️ by Azad Bhasme</sub></center>", unsafe_allow_html=True)
