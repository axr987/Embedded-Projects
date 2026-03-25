# YOLO version - detects 'person' class (class 0 in COCO) instead of cascade fullbody
# Retains multiprocessing: detection in separate process
# Install: pip install ultralytics picamera2 opencv-python
# Uses YOLOv8n (nano, fast for Pi); change to yolov8s.pt etc. for accuracy

import cv2 as cv
from picamera2 import Picamera2, Preview
import time
import os
from datetime import datetime
import argparse
import multiprocessing as mp
import subprocess
from ultralytics import YOLO  # YOLO import [web:33]

# Global variables
state = 0
frame_count = 0
output_dir = "captures"
os.makedirs(output_dir, exist_ok=True)
stream_dir = "temp_stream"
os.makedirs(stream_dir, exist_ok=True)
scale_factor = 1.3  # Not used now, kept for ref
video_writer = None
last_save_time = time.time()
frame_queue = mp.Queue(maxsize=1)
box_queue = mp.Queue(maxsize=1)  # Now queues (x1,y1,x2,y2) tuples
full_bodies_scaled = []  # Scaled boxes for drawing

configs = {
    0: {"res": (640, 480), "hz": 3, "mode": "img"},
    1: {"res": (1280, 720), "hz": 6, "mode": "img"}, 
    2: {"res": (1920, 1080), "hz": 30, "mode": "video"}
}

# Detection function with YOLO - focuses on person class [web:39]
def detectFullBody(frame_queue, box_queue):
    model = YOLO("yolov8n.pt")  # Load nano model (downloads if needed) [web:33]
    while True:
        if not frame_queue.empty():  # Fixed: check empty, not full for get()
            frame = frame_queue.get()
            # Run YOLO predict, classes=0 for person only, conf=0.25, verbose=False
            results = model.predict(frame, classes=0, conf=0.25, verbose=False, device='cpu')  # CPU for Pi [web:33][web:39]
            boxes = []
            if results and len(results[0].boxes) > 0:
                for box in results[0].boxes.xyxy:  # xyxy format
                    x1, y1, x2, y2 = box.cpu().numpy()
                    boxes.append((int(x1), int(y1), int(x2-x1), int(y2-y1)))  # To (x,y,w,h)
            box_queue.put(boxes)
        else:
            time.sleep(0.1)

def generate_frames():
    while True:
        frame = picam2.capture_array()
        if frame is not None:
            yield frame
        else:
            print("Failed to capture frame")

def config_state(state):
    picam2.stop()
    if video_writer:
        video_writer.release()
    config = configs[state]
    width, height = config["res"]
    mode, hz = config["mode"], config["hz"]
    picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (width, height)}))
    picam2.start()
    print(f"State {state}: {width}x{height} {mode} @ {hz}Hz")
    return width, height, mode, hz

# Start detection worker
worker = mp.Process(target=detectFullBody, args=(frame_queue, box_queue), daemon=True)
worker.start()

# Camera setup
picam2 = Picamera2()
width, height, mode, hz = config_state(state)

print("Recording... Press 'q' to quit")
subprocess.run(['sudo', 'bash', 'send_image_geeqie.sh'], check=True)

for frame in generate_frames():
    now = time.time()
    
    resized = cv.resize(frame, (640, 480))
    if frame_queue.empty():
        frame_queue.put(resized.copy())

    # Get YOLO boxes (already scaled to input size)
    if not box_queue.empty():
        full_bodies_scaled = box_queue.get()  # List of (x,y,w,h) from 640x480
        # Scale to full frame size
        scaled_boxes = []
        for (x, y, w, h) in full_bodies_scaled:
            sx = int(x * width / 640)
            sy = int(y * height / 480)
            sw = int(w * width / 640)
            sh = int(h * height / 480)
            scaled_boxes.append((sx, sy, sw, sh))
        full_bodies_scaled = scaled_boxes  # Update global-like list

    # Draw rectangles if boxes present
    if len(full_bodies_scaled) > 0:
        for (x, y, w, h) in full_bodies_scaled:
            top_left = (x, y)
            bottom_right = (x + w, y + h)
            frame = cv.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)  # Green for YOLO
        cv.imshow('YOLO Person Detection', frame)
        cv.imwrite(os.path.join(stream_dir, "frame.jpg"), frame, [cv.IMWRITE_JPEG_QUALITY, 90])
        subprocess.run(['sudo', 'bash', 'send_image_geeqie.sh'], check=True)
    else:
        cv.imshow('YOLO Person Detection', frame)
    
    buttonpress = cv.waitKey(10) & 0xFF
    if buttonpress == ord('q'):
        break
    elif buttonpress == ord('0'):
        width, height, mode, hz = config_state(0)
    elif buttonpress == ord('1'):
        width, height, mode, hz = config_state(1)
    elif buttonpress == ord('2'):
        width, height, mode, hz = config_state(2)

    # Video save
    if mode == "video":
        if video_writer is None:
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            filename = os.path.join(output_dir, f"s{state}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            video_writer = cv.VideoWriter(filename, fourcc, hz, (width, height))
        if video_writer:
            video_writer.write(frame)
    
    # Image save
    if mode == "img":
        if now - last_save_time >= 1.0 / hz:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = os.path.join(output_dir, f"s{state}_{timestamp}.jpg")
            cv.imwrite(filename, frame, [cv.IMWRITE_JPEG_QUALITY, 90])
            last_save_time = now
    
    frame_count += 1
    if frame_count > 300:
        print("Reached 300 frames, exiting.")
        break

# Cleanup
picam2.stop()
if video_writer:
    video_writer.release()
cv.destroyAllWindows()
worker.terminate()  # Clean up worker
print("Done.")
