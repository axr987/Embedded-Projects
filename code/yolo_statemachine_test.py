# Multiprocessing script with YOLO detector (updated from cascade)

import os
os.environ["QT_LOGGING_RULES"] = "*.warning=false"  # Suppress warning about missing fonts

import cv2 as cv
from picamera2 import Picamera2
import time
from datetime import datetime
import argparse
import multiprocessing as mp
import subprocess
from ultralytics import YOLO  # pip install ultralytics

# Global variables
state = 0
frame_count = 0

# Main output folder
output_dir = "captures"
os.makedirs(output_dir, exist_ok=True)

# Temporary streaming folder
stream_dir = "temp_stream"
os.makedirs(stream_dir, exist_ok=True)

# Event to stop the worker process
stop_event = mp.Event()

# More global variables
scale_factor = 1.0  # YOLO uses normalized coords, no scaling needed
video_writer1 = None
video_writer2 = None
last_save_time = time.time()
frame_queue1 = mp.Queue(maxsize=2)
frame_queue2 = mp.Queue(maxsize=2)
box_queue1 = mp.Queue(maxsize=2)
box_queue2 = mp.Queue(maxsize=2)
full_bodies1 = []
full_bodies2 = []
frame_max = 10000
send_over_network = True
last_send_time = time.time()
approach_scale = 2
past_target_area1 = 0  # Fixed: Initialize per-camera areas
past_target_area2 = 0

# Load YOLO model (person class: 0)
model = YOLO('yolov8n.pt')  # Or yolov8s.pt for better accuracy

configs = {
    0: {"res": (640, 480), "hz": 3, "mode": "preview"},
    1: {"res": (640, 480), "hz": 3, "mode": "img"},
    2: {"res": (1280, 720), "hz": 6, "mode": "img"}, 
    3: {"res": (1920, 1080), "hz": 30, "mode": "video"}
}

picam2 = Picamera2(0)
picam3 = Picamera2(1)

# YOLO detection function (class 0 = person)
def detectFullBody(frame_queue, box_queue, stop_event):
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.5)
        except:
            continue
        # Run YOLO inference
        results = model(frame, verbose=False, conf=0.5, classes=[0])  # Person only, conf>0.5
        boxes = []
        if results[0].boxes is not None:
            for box in results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1
                boxes.append((x1, y1, w, h))
        box_queue.put(boxes)

def generate_frames(picam):
    while True:
        frame = picam.capture_array()
        if frame is not None:
            yield frame
        else:
            print("Failed to capture frame")

def config_state(state, picam):
    picam.stop()
    if video_writer1:
        video_writer1.release()
    if video_writer2:
        video_writer2.release()
    config = configs[state]
    width, height = config["res"]
    mode, hz = config["mode"], config["hz"]
    if state == 2:
        alarm_timer = time.time()
    else:
        alarm_timer = 0
    picam.configure(picam.create_preview_configuration(main={"format": "RGB888", "size": (width, height)}))
    picam.start()
    print(f"State {state}: {width}x{height} {mode} @ {hz}Hz")
    return width, height, mode, hz, alarm_timer

# Start workers
worker1 = mp.Process(target=detectFullBody, args=(frame_queue1, box_queue1, stop_event), daemon=True)
worker1.start()
worker2 = mp.Process(target=detectFullBody, args=(frame_queue2, box_queue2, stop_event), daemon=True)
worker2.start()

width, height, mode, hz, alarm_timer = config_state(state, picam2)
_, _, _, _, _ = config_state(state, picam3)

print("Recording... Press 'q' to quit, 0-3 for states")

if send_over_network:
    subprocess.Popen(['sudo', 'bash', 'send_image_geeqie.sh'])

cv.namedWindow('Capture - Full body detection 1')
cv.namedWindow('Capture - Full body detection 2')
cv.moveWindow('Capture - Full body detection 1', 40, 50)
cv.moveWindow('Capture - Full body detection 2', 960, 50)

alarming_timer = 0
alarm_states = 0

# Main loop
for frame1, frame2 in zip(generate_frames(picam2), generate_frames(picam3)):
    buttonpress = cv.waitKey(10) & 0xFF
    if buttonpress == ord('q'):
        break
    elif buttonpress in [ord(str(i)) for i in range(4)]:
        state = int(chr(buttonpress))
        width, height, mode, hz, alarm_timer = config_state(state, picam2)
        _, _, _, _, _ = config_state(state, picam3)

    resized1 = cv.resize(frame1, (640, 480))
    resized2 = cv.resize(frame2, (640, 480))
    
    try: frame_queue1.put_nowait(resized1.copy())
    except: pass
    try: frame_queue2.put_nowait(resized2.copy())
    except: pass

    # Get detections
    try:
        while True: full_bodies1 = box_queue1.get_nowait()
    except: pass
    try:
        while True: full_bodies2 = box_queue2.get_nowait()
    except: pass

    detections = [(full_bodies1, frame1, 1), (full_bodies2, frame2, 2)]
    state_change = False

    for full_bodies, frame, cam_id in detections:
        if len(full_bodies) > 0:
            if state == 0:
                state = 1
                width, height, mode, hz, alarm_timer = config_state(state, picam2)
                _, _, _, _, _ = config_state(state, picam3)
                state_change = True

            # Scale boxes to full res (YOLO already pixel coords)
            full_bodies_scaled = [(int(x), int(y), int(w), int(h)) for x, y, w, h in full_bodies]
            
            for x, y, w, h in full_bodies_scaled:
                current_target_area = w * h
                past_target_area = past_target_area1 if cam_id == 1 else past_target_area2
                
                # Approach detection (fixed logic)
                if state == 1 and current_target_area > past_target_area * approach_scale:
                    state = 2
                    globals()['past_target_area%d' % cam_id] = past_target_area  # Save past
                    width, height, mode, hz, alarm_timer = config_state(state, picam2)
                    _, _, _, _, _ = config_state(state, picam3)
                    state_change = True
                
                # Draw box
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for YOLO
                
                # Update past area
                if cam_id == 1: past_target_area1 = current_target_area
                else: past_target_area2 = current_target_area

            # Stream/save preview
            stream_fn = os.path.join(stream_dir, f"frame{cam_id}.jpg")
            cv.imwrite(stream_fn, frame, [cv.IMWRITE_JPEG_QUALITY, 90])
            if send_over_network and time.time() - last_send_time > 1.0:
                subprocess.Popen(['sudo', 'bash', 'send_image_geeqie.sh'])
                last_send_time = time.time()

            cv.imshow(f'Capture - Full body detection {cam_id}', cv.resize(frame, (800, 600)))

    # State transitions (fixed)
    if state == 2 and time.time() - alarm_timer > 10:
        state = 3
        alarming_timer = time.time()
        width, height, mode, hz, alarm_timer = config_state(state, picam2)
        _, _, _, _, _ = config_state(state, picam3)
    elif state == 2 and min(past_target_area1, past_target_area2) < (globals().get('saved_past_target_area1', 0) or globals().get('saved_past_target_area2', 0)) and time.time() - alarm_timer > 1:
        state = 1
    elif state == 3 and time.time() - alarming_timer > 5:
        state = 1

    if not any(len(b) > 0 for b in [full_bodies1, full_bodies2]) and time.time() - last_send_time > 5:
        state = 0

    # No detections: show plain frames
    if len(full_bodies1) == 0: cv.imshow('Capture - Full body detection 1', cv.resize(frame1, (800, 600)))
    if len(full_bodies2) == 0: cv.imshow('Capture - Full body detection 2', cv.resize(frame2, (800, 600)))

    # Video mode
    if mode == "video":
        if video_writer1 is None:
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            video_writer1 = cv.VideoWriter(os.path.join(output_dir, f"cam1_s{state}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"), fourcc, hz, (width, height))
        video_writer1.write(frame1)
        if video_writer2 is None:
            video_writer2 = cv.VideoWriter(os.path.join(output_dir, f"cam2_s{state}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"), fourcc, hz, (width, height))
        video_writer2.write(frame2)

    # Image mode
    elif mode == "img":
        now = time.time()
        if now - last_save_time >= 1.0 / hz:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            cv.imwrite(os.path.join(output_dir, f"cam1_s{state}_{timestamp}.jpg"), frame1, [cv.IMWRITE_JPEG_QUALITY, 90])
            cv.imwrite(os.path.join(output_dir, f"cam2_s{state}_{timestamp}.jpg"), frame2, [cv.IMWRITE_JPEG_QUALITY, 90])
            last_save_time = now

    frame_count += 1
    if frame_count > frame_max:
        break

# Cleanup
stop_event.set()
worker1.join()
worker2.join()
for q in [frame_queue1, frame_queue2, box_queue1, box_queue2]:
    q.cancel_join_thread()
    q.close()
picam2.stop()
picam3.stop()
if video_writer1: video_writer1.release()
if video_writer2: video_writer2.release()
cv.destroyAllWindows()
print("Done.")