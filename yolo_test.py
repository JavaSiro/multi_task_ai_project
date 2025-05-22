# yolo_test.py

import cv2
from ultralytics import YOLO

# Load a pretrained YOLOv8 model (COCO dataset)
model = YOLO("yolov8m.pt")  # you can try yolov8n.pt if it's slow

# Open webcam (or replace 0 with a video file path)
cap = cv2.VideoCapture("1.mp4")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO detection
    results = model(frame)[0]

    # Draw bounding boxes and labels on the frame
    annotated_frame = results.plot()

    # Display the annotated frame
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()