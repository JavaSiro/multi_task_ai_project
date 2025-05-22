# yolo_model.py

from ultralytics import YOLO
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

# Load model once
model = YOLO("yolov8m.pt")

def run_inference(image_bytes):
    # Load image from bytes
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    frame = np.array(image)

    # Run inference
    results = model(frame)[0]

    # Parse results into JSON format
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append({
            "label": label,
            "confidence": round(conf, 2),
            "box": [x1, y1, x2, y2]
        })

    return detections