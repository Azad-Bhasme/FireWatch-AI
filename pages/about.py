import streamlit as st

st.set_page_config(page_title="About FireWatch AI", page_icon="🔥")

st.markdown("<h1 style='text-align: center;'>📘 About FireWatch AI</h1>", unsafe_allow_html=True)

st.markdown("""
### 🔥 What is FireWatch AI?
**FireWatch AI** is a smart, real-time fire detection system built using **YOLOv8** and **Streamlit**. It allows users to:

- Upload images/videos or use a webcam 📸
- Get instant fire detection results 🚒
- View and download detection output 📥
- Run directly in browser (no coding needed)

### 🧠 Why FireWatch AI?
This project aims to provide early fire warnings using AI and computer vision to prevent fire-related hazards in homes, industries, and forests.

### ⚙️ Technologies Used:
- YOLOv8 for real-time fire detection
- OpenCV for image/video processing
- Streamlit for the web interface
- Python for backend logic

### 👨‍💻 Developer:
Developed with ❤️ by **Azad Bhasme**

### 📫 Contact:
- GitHub: [Azad-Bhasme](https://github.com/Azad-Bhasme)
- Email: azadbhasme@gmail.com
""")

# Optional Footer
st.markdown("""
<div style="text-align: center; font-size: 13px; padding-top: 2rem;">
    <span style="font-family: 'Poppins', sans-serif;">made with <span style="color: #e63946;">❤️</span> by Azad Bhasme</span>
</div>
""", unsafe_allow_html=True)
