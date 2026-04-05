# Manual picture and video capture with YOLOv5 human detection.
# Very slow, but would work for birds.
# Maybe it would be alright for birds when they're sitting mostly still.

import os
from pyexpat import model
os.environ["QT_LOGGING_RULES"] = "*.warning=false" # Suppress warning about missing fonts that aren't really missing

import cv2 as cv
from picamera2 import Picamera2
import time
from datetime import datetime
import multiprocessing as mp
import subprocess
import torch

# Global variables
state = 0 # change to real state select code later
frame_count = 0 # Replace with threading later

# Main output folder
output_dir = "captures"
os.makedirs(output_dir, exist_ok=True)

# Temporary streaming folder
stream_dir = "temp_stream"
os.makedirs(stream_dir, exist_ok=True)

# Event to stop the worker process
stop_event = mp.Event()

# More global variables
scale_factor = 1.3 # For rescaling the image for bounding box drawing
video_writer = None # Required for video writing
last_save_time = time.time() # For saving images at a regular interval
frame_queue = mp.Queue(maxsize=2) # Queue for sharing frames between processes
box_queue = mp.Queue(maxsize=2) # Queue for sharing bounding boxes between processes
full_bodies = [] # List to hold detected bounding boxes
frame_max = 10000 # Just a safety to prevent infinite loops during testing
send_over_network = True # Set to True to enable sending images over the network to geeqie
conf_thresh = 0.35 # Confidence threshold for YOLOv5
nms_thresh = 0.45 # NMS threshold for YOLOv5
last_send_time = time.time() # For sending images at a regular interval

# Modify resolution below
# picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
configs = {
    0: {"res": (640, 480), "hz": 3, "mode": "img"},
    1: {"res": (1280, 720), "hz": 6, "mode": "img"}, 
    2: {"res": (1920, 1080), "hz": 30, "mode": "video"}
}

# detection function
def detectFullBody(frame_queue, box_queue, stop_event):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True) # Load YOLOv5 model
    model.conf = conf_thresh # Set confidence threshold
    model.iou = nms_thresh # Set NMS threshold
    model.classes = [0] # Only detect person class (class 0 in COCO dataset)
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.5)
        except:
            continue
        print("Getting frame from queue")
        frame = frame_queue.get()
        with torch.no_grad():
            results = model(frame)
        print("Did the thing")
        boxes = results.xyxy[0].cpu().numpy()
        filtered_boxes = [
            [int(x1), int(y1), int(x2), int(y2)]
            for x1, y1, x2, y2, conf, cls in boxes
            if conf >= conf_thresh
        ]

        print(f"Putting full bodies in queue: {filtered_boxes}")
        box_queue.put(filtered_boxes) # Put bounding boxes in queue

# frame generation function
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
    # set config based on state
    config = configs[state]
    width, height = config["res"]
    mode, hz = config["mode"], config["hz"]

    picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (width, height)}))
    picam2.start()
    print(f"State {state}: {width}x{height} {mode} @ {hz}Hz")
    return width, height, mode, hz

# Create worker function for box drawing
worker = mp.Process(target=detectFullBody,
                    args=(frame_queue, box_queue, stop_event),
                    daemon=True) # Set as daemon so it will exit when the main process exits
worker.start()

# turn camera on
picam2 = Picamera2()

width, height, mode, hz = config_state(state)

print("Recording... Press 'q' to quit")

#subprocess.run(['sudo', 'bash', 'send_image_geeqie.sh'], check=True)
subprocess.Popen(['sudo', 'bash', 'send_image_geeqie.sh'])

# loop time
for frame in generate_frames():
    
    resized = cv.resize(frame, (640, 480))
    try:
        frame_queue.put_nowait(resized.copy())
        print("Put frame")
    except:
        pass

    # drawing the boxes when available
    try:
        while True:
            full_bodies = box_queue.get_nowait()
            print(f"Detected full bodies: {full_bodies}")
    except:
        pass

    # shows the boxes if drawn
    if len(full_bodies) > 0:
        for (x1,y1,x2,y2) in full_bodies:
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            # rectangle uses top left corner and bottom right corner
            top_left = (x1, y1)
            bottom_right = (x2, y2)
            frame = cv.rectangle(frame, top_left, bottom_right, (255,0,0), 2)
        cv.imshow('Capture - Full body detection', frame)
        cv.imwrite(os.path.join(stream_dir, "frame.jpg"), frame, [cv.IMWRITE_JPEG_QUALITY, 90])
        if send_over_network and time.time() - last_send_time > 1.0:
            subprocess.run(['sudo', 'bash', 'send_image_geeqie.sh'], check=True)
            last_send_time = time.time()
    # shows just the frame if no boxes are drawn
    else:
        cv.imshow('Capture - Full body detection', frame)
    
    # delay
    buttonpress = cv.waitKey(10) & 0xFF
    if buttonpress == ord('q'):
            stop_event.set()
            break
    elif buttonpress == ord('0'):
            width, height, mode, hz = config_state(0)
            print(f"State {state}: {width}x{height} {mode} @ {hz}Hz")
    elif buttonpress == ord('1'):
            width, height, mode, hz = config_state(1)
            print(f"State {state}: {width}x{height} {mode} @ {hz}Hz")
    elif buttonpress == ord('2'):
            width, height, mode, hz = config_state(2)
            print(f"State {state}: {width}x{height} {mode} @ {hz}Hz")

    # video mode code
    if mode == "video":
        if video_writer is None:
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            filename = os.path.join(output_dir, f"s{state}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            video_writer = cv.VideoWriter(filename, fourcc, hz, (width, height))
            # print(f"VIDEO: {os.path.basename(filename)}")
        video_writer.write(frame) if video_writer else None
    
    # image mode code
    if mode == "img":
        now = time.time()
        if now - last_save_time >= 1.0 / hz:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = os.path.join(output_dir, f"s{state}_{timestamp}.jpg")
            
            last_save_time = now

            # JPEG compress
            cv.imwrite(filename, frame, [cv.IMWRITE_JPEG_QUALITY, 90])
            # print(f"IMG: {os.path.basename(filename)}")
            
    frame_count += hz / 30
    if frame_count > frame_max: # Just a safety to prevent infinite loops during testing
        print("Reached 300 frames, exiting loop.")
        break

# cleanup 
picam2.stop()
if video_writer:
    video_writer.release()
cv.destroyAllWindows()
print("Done.")
