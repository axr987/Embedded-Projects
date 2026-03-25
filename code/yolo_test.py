import cv2
from ultralytics import YOLO
import numpy as np

# Load YOLO model (nano for speed)
model = YOLO("yolov8n.pt")

# Open webcam (0 usually default)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

print("YOLO Person Detection Test. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Run YOLO on frame, detect only persons (class 0), conf > 0.5
    results = model.predict(frame, classes=39, conf=0.5, verbose=False)
    
    # Draw boxes if detections
    if results[0].boxes is not None:
        for box in results[0].boxes:
            # Extract coords
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            # Draw rectangle and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'Bottle: {conf:.2f}', (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display
    cv2.imshow('YOLO Webcam Test - Person Detection', frame)
    
    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Verification complete.")
