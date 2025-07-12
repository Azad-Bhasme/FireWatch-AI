import streamlit as st
import cv2
import os
import uuid
import tempfile
import shutil
from PIL import Image
from ultralytics import YOLO

# Clean previous runs
if os.path.exists("runs"):
    shutil.rmtree("runs")

# Page configuration
st.set_page_config(page_title="FireWatch AI", page_icon="🔥")
st.markdown("<h1 style='text-align: center;'>🔥 FireWatch AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Smart detection, safer protection. 🔥🧠</h4>", unsafe_allow_html=True)

# Load model
model = YOLO("best.pt")

# Function to perform detection and return result file path
def detect_and_save(source_path, is_video=False):
    results = model.predict(source=source_path, conf=0.5, save=True)
    output_dir = results[0].save_dir

    result_file = None
    for file in os.listdir(output_dir):
        if file.endswith((".jpg", ".jpeg", ".png", ".mp4", ".avi")):
            result_file = os.path.join(output_dir, file)
            break
    return result_file

# Input mode
input_mode = st.radio("Choose Input Mode:", ["📤 Upload File"], horizontal=True)

if input_mode == "📤 Upload File":
    uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi"])

    if uploaded_file:
        ext = uploaded_file.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        st.success("✅ File uploaded. Running detection.. ⭕")
        result_path = detect_and_save(temp_path, is_video=ext in ["mp4", "avi"])

        if result_path:
            # Show result
            if st.button("👀 View Your Result", key="view_btn"):
                if result_path.endswith((".jpg", ".jpeg", ".png")):
                    st.image(result_path, caption="Fire Detected", use_container_width=True)
                elif result_path.endswith((".mp4", ".avi")):
                    st.video(result_path)

            # Offer download
            with open(result_path, "rb") as f:
                result_data = f.read()
            st.download_button("📥 Download Result", result_data, file_name=f"result_{ext}", mime="video/mp4" if "mp4" in ext else "image/jpeg")

        else:
            st.error("❌ No result was generated.")

# Blinking ❤️ footer (only heart blinks)
st.markdown("""
    <style>
    .footer-blink {
        text-align: center;
        font-family: 'Segoe UI', sans-serif;
        font-size: 14px;
        font-weight: 500;
        color: #333;
    }
    .footer-heart {
        color: red;
        animation: blink 1s linear infinite;
    }
    @keyframes blink {
        50% { opacity: 0; }
    }
    </style>

    <div class="footer-blink">made with <span class="footer-heart">❤️</span> by Azad Bhasme</div>
""", unsafe_allow_html=True)
