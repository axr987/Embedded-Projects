# This is just a reference
# I think what needs to be done is we need to separate detection in a thread separate from the rest of the code.
# We might not even need to time how many frames are between drawing bounding boxes.
# The only shared objects should be the current frame and the detected bounding boxes.
# Whenever the detection thread is ready, it can update the bounding boxes and sample a new frame from what the main thread has collected.
# I'm getting tired, so I might do this later.

import cv2 as cv
from picamera2 import Picamera2, Preview
import time
import os
from datetime import datetime
import argparse
import multiprocessing as mp
import subprocess

# Global variables
state = 0 # change to real state select code later
frame_count = 0 # Replace with threading later
# Main output folder
output_dir = "captures"
os.makedirs(output_dir, exist_ok=True)
# Temporary streaming folder
stream_dir = "temp_stream"
os.makedirs(stream_dir, exist_ok=True)
scale_factor = 1.3 # For rescaling the image for bounding box drawing
video_writer = None # Required for video writing
last_save_time = time.time() # For saving images at a regular interval
frame_queue = mp.Queue(maxsize=1) # Queue for sharing frames between processes
box_queue = mp.Queue(maxsize=1) # Queue for sharing bounding boxes between processes
full_bodies = [] # List to hold detected bounding boxes
full_bodies_scaled = [] # List to hold scaled bounding boxes for drawing

# Create argument parser for cascade and camera 
parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--fullbody_cascade', help='Path to face cascade.', default='../data/haarcascade_fullbody.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()

fullbody_cascade_name = args.fullbody_cascade
#-- load cascade
fullbody_cascade = cv.CascadeClassifier()
if not fullbody_cascade.load(os.path.join("code", fullbody_cascade_name)):
    print('--(!)Error loading full body cascade')
    exit(0)

# Modify resolution below
# picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
configs = {
    0: {"res": (640, 480), "hz": 3, "mode": "img"},
    1: {"res": (1280, 720), "hz": 6, "mode": "img"}, 
    2: {"res": (1920, 1080), "hz": 30, "mode": "video"}
}

# detection function
def detectFullBody(frame_queue, box_queue):
    while True:
        if frame_queue.full():
            # print("Getting frame from queue")
            frame = frame_queue.get()
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame_gray = cv.equalizeHist(frame_gray)
            full_bodies = fullbody_cascade.detectMultiScale(frame_gray, scaleFactor=scale_factor, minNeighbors=3)
            # print(f"Putting full bodies in queue: {full_bodies}")
            box_queue.put(full_bodies)
        else:
            time.sleep(0.1) # Sleep briefly to avoid unnecessary CPU usage

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
                    args=(frame_queue, box_queue),
                    daemon=True) # Set as daemon so it will exit when the main process exits
worker.start()

# turn camera on
picam2 = Picamera2()

width, height, mode, hz = config_state(state)

print("Recording... Press 'q' to quit")

subprocess.run(['sudo', 'bash', 'send_image_geeqie.sh'], check=True)

# loop time
for frame in generate_frames():

    now = time.time()
    
    resized = cv.resize(frame, (640, 480))
    if frame_queue.empty():
        # print("Putting frame in queue")
        frame_queue.put(resized.copy())

    # drawing the boxes when available
    if box_queue.full():
        full_bodies_scaled = [] # Clear the scaled boxes list before adding new ones
        full_bodies = box_queue.get() 
        # print(f"Detected full bodies: {full_bodies}")
        if len(full_bodies) > 0:
            for (x,y,w,h) in full_bodies:
                x = int(x * width / 640)
                y = int(y * height / 480)
                w = int(w * width / 640)
                h = int(h * height / 480)
                # Tuples are immutable, so you have to make a new one to scale it
                full_bodies_scaled.append((x, y, w, h))
    # shows the boxes if drawn
    if len(full_bodies_scaled) > 0:
        for (x,y,w,h) in full_bodies_scaled:
            # rectangle uses top left corner and bottom right corner
            top_left = (x, y)
            bottom_right = (x + w, y + h)
            frame = cv.rectangle(frame, top_left, bottom_right, (255,0,0), 2)
        cv.imshow('Capture - Full body detection', frame)
        cv.imwrite(os.path.join(stream_dir, "frame.jpg"), frame, [cv.IMWRITE_JPEG_QUALITY, 90])
        subprocess.run(['sudo', 'bash', 'send_image_geeqie.sh'], check=True)
    # shows just the frame if no boxes are drawn
    else:
        cv.imshow('Capture - Full body detection', frame)
    
    # delay
    buttonpress = cv.waitKey(10) & 0xFF
    if buttonpress == ord('q'):
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
        if now - last_save_time >= 1.0 / hz:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = os.path.join(output_dir, f"s{state}_{timestamp}.jpg")
            
            # JPEG compress
            cv.imwrite(filename, frame, [cv.IMWRITE_JPEG_QUALITY, 90])
            # print(f"IMG: {os.path.basename(filename)}")
            
            last_save_time = now
  
    frame_count += 1
    if frame_count > 300: # Just a safety to prevent infinite loops during testing
        print("Reached 300 frames, exiting loop.")
        break

# cleanup 
picam2.stop()
if video_writer:
    video_writer.release()
cv.destroyAllWindows()
print("Done.")
