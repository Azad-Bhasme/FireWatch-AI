import streamlit as st
import cv2
from PIL import Image
import ultralytics
from ultralytics import YOLO
import tempfile
import os
import uuid


st.set_page_config("Fire Detection")
st.title("🔥 FireWatch AI")

# Load model
model = YOLO("best.pt")

# Initialize session state to store result path
if 'result_path' not in st.session_state:
    st.session_state.result_path = None

uploaded_file = st.file_uploader("Upload video/image", type=["mp4", "avi", "jpg", "jpeg", "png"])

if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    st.success("✅ File uploaded. Detecting fire...")
    results = model.predict(source=temp_path, save=True, conf=0.5)
    st.session_state.result_path = results[0].save_dir  # Save result folder

# 👇 "View Your Result" only appears after detection
if st.session_state.result_path:
    if st.button("👀 View Your Result"):
        for file in os.listdir(st.session_state.result_path):
            f_path = os.path.join(st.session_state.result_path, file)
            if file.endswith((".jpg", ".jpeg", ".png")):
                st.image(f_path, caption="🔥 Detected Image", use_container_width=True)
            elif file.endswith((".mp4", ".avi")):
                st.video(f_path, format="video/mp4")

# Webcam logic (local only)
elif mode == "📷 Use Webcam":
    st.warning("⚠️ This works only in local environment (not Streamlit Cloud)")
    if st.button("📷 Start Camera", key="start_cam"):
        cam = cv2.VideoCapture(0)
        stframe = st.empty()

        stop = False
        stop_key = str(uuid.uuid4())

        while cam.isOpened() and not stop:
            ret, frame = cam.read()
            if not ret:
                st.error("❌ Failed to access camera.")
                break

            cv2.imwrite("frame.jpg", frame)
            results = model.predict("frame.jpg", conf=0.5)
            img = results[0].plot()
            stframe.image(img, channels="BGR", use_container_width=True)

            stop = st.button("🛑 Stop Camera", key=stop_key)

        cam.release()

# Blinking Tagline
st.markdown("""
<style>
.blink {
  animation: blink-animation 10s steps(2, start) infinite;
  color: #fa5252;
  text-align: center;
  font-weight: bold;
  font-size: 14px;
}
@keyframes blink-animation {
  to {
    visibility: hidden;
  }
}
</style>
<p class="blink">made with ❤️ by Azad Bhasme</p>
""", unsafe_allow_html=True)
