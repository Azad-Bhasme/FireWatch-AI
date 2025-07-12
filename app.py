import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

st.set_page_config(page_title="FireWatch AI", page_icon="🔥")
st.title("🔥 FireWatch AI")
st.markdown("### Smart detection, safer protection. 🔥🧠")

model = YOLO("best.pt")

uploaded_file = st.file_uploader("📤 Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()


    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    st.success("✅ File uploaded. Running detection...")

    try:
        results = model.predict(source=temp_file_path, save=True, conf=0.6)

        output_path = results[0].save_dir
        for file in os.listdir(output_path):
            if file.lower().endswith((".jpg", ".png")):
                st.image(os.path.join(output_path, file))
            elif file.lower().endswith((".mp4", ".avi")):
                st.video(os.path.join(output_path, file))
    except Exception as e:
        st.error(f"❌ Detection failed: {e}")

st.markdown("---")
st.markdown("<center><sub>made with ❤️ by Azad Bhasme</sub></center>", unsafe_allow_html=True)
