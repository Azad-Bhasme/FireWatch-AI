import streamlit as st
import cv2
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import time

# Page setup
st.set_page_config(page_title="FireWatch AI", page_icon="🔥")
st.title("🔥 FireWatch AI")
st.markdown("### Smart detection, safer protection. 🔥🧠")

# Load YOLOv8 model
model = YOLO("best.pt")

# Choose input mode
mode = st.radio("Choose Input Mode:", ["📤 Upload file", "📷 Use Webcam"])

# Detection function
def detect_fire(source_path, is_video):
    results = model.predict(source=source_path, save=True, conf=0.4)
    output_path = results[0].save_dir

    # Display results
    for file in os.listdir(output_path):
        if file.endswith((".jpg", ".png")):
            st.image(os.path.join(output_path, file), caption="Detected Image")
        elif file.endswith((".mp4", ".avi")):
            st.video(os.path.join(output_path, file))

    return output_path

# Upload file section
if mode == "📤 Upload file":
    uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi"])
    if uploaded_file:
        suffix = uploaded_file.name.split('.')[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        st.success("✅ File uploaded. Detecting fire...")
        output_dir = detect_fire(temp_path, is_video=suffix in ["mp4", "avi"])

# Webcam section (LOCAL ONLY)
elif mode == "📷 Use Webcam":
    st.warning("⚠️ Webcam access only works on local PC (not Streamlit Cloud).")

    if st.button("Start Camera Detection"):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to capture image from webcam.")
                break

            img_path = "frame.jpg"
            cv2.imwrite(img_path, frame)

            results = model.predict(source=img_path, conf=0.4)
            annotated = results[0].plot()
            annotated_img = Image.fromarray(annotated)

            stframe.image(annotated_img, caption="Live Fire Detection", use_column_width=True)
            if st.button("Stop"):
                break

        cap.release()

# Footer with blink effect
st.markdown("""
<style>
.blink-footer {
  text-align: center;
  animation: blink 1s step-start 0s infinite;
  color: #fa5252;
  font-size: 13px;
}
@keyframes blink {
  50% { opacity: 0; }
}
</style>
<div class='blink-footer'>made with ❤️ by Azad Bhasme</div>
""", unsafe_allow_html=True)
