import streamlit as st
import cv2
import numpy as np
from decode import recognize_faces

COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 255)]

st.set_page_config(page_title="🎓 Face Recognition Attendance", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; color: #3b82f6;'>Smart Attendance System</h1>
    <p style='text-align: center;'><h3 >Upload a image — We'll recognize the faces and display roll numbers!</h3></p>
    <hr>
    """, unsafe_allow_html=True
)

st.markdown("### Upload Group Photo")
uploaded_file = st.file_uploader("Upload an image file (JPG, PNG)", type=["jpg", "jpeg", "png"])


if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # BGR format

    with st.spinner("🔍 Recognizing faces..."):
        processed_image, names = recognize_faces(image.copy())
        unique_names = sorted(set(names))

    rgb_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    st.image(rgb_image, caption="🖼️ Processed Image", use_container_width=True, output_format = "PNG")

    st.markdown("---")
    st.markdown(f"""
        <div style="background-color:#d1fae5;padding:20px;border-radius:10px;">
        <h3 style="color:#065f46;">Total Students Recognized: {len(unique_names)}</h3>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("### Recognized Roll Numbers:")
    for name in unique_names:
        st.markdown(f"* {name}")

    st.markdown("---")
    st.success("📚 Thank You")