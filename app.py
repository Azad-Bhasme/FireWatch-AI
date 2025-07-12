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

# Option to choose between upload or webcam
option = st.radio("Choose Input Mode:", ["📤 Upload file", "📷 Use webcam"])

# -------------------- Upload Mode --------------------
if option == "📤 Upload file":
    uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi"])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        st.success("✅ File uploaded. Running detection...")

        try:
            results = model.predict(source=temp_file_path, save=True, conf=0.6)
            output_path = results[0].save_dir

            # Show output
            output_files = [f for f in os.listdir(output_path) if f.lower().endswith((".jpg", ".png", ".mp4", ".avi"))]
            if not output_files:
                st.warning("⚠️ No output files found.")
            else:
                for file in output_files:
                    full_path = os.path.join(output_path, file)
                    if file.lower().endswith((".jpg", ".png")):
                        st.image(full_path, caption="🖼️ Detected Image")
                    elif file.lower().endswith((".mp4", ".avi")):
                        st.video(full_path, format="video/mp4")

        except Exception as e:
            st.error(f"❌ Detection failed: {e}")

# -------------------- Webcam Mode --------------------
elif option == "📷 Use webcam":
    run = st.checkbox("🎥 Start Webcam Fire Detection")
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to access the webcam.")
            break

        results = model.predict(source=frame, imgsz=640, conf=0.6, verbose=False)
        annotated_frame = results[0].plot()

        FRAME_WINDOW.image(annotated_frame, channels="BGR")

        # Add a break condition for Streamlit
        if not run:
            break
        time.sleep(0.03)

    cap.release()

st.markdown("---")
st.markdown("<center><sub>made with ❤️ by Azad Bhasme</sub></center>", unsafe_allow_html=True)

