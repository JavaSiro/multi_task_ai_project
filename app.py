import streamlit as st
from PIL import Image
from ultralytics import YOLO
import io

model = YOLO("yolov8n.pt")  # or yolov8s.pt etc.

st.title("YOLOv8 Object Detection")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run YOLO inference
    results = model(image)[0]

    # Draw boxes on image
    res_plotted = results.plot()
    st.image(res_plotted, caption="Detection Results", use_column_width=True)