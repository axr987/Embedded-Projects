# Multiprocessing script with haar cascade classifier

import os
os.environ["QT_LOGGING_RULES"] = "*.warning=false" # Suppress warning about missing fonts that aren't really missing

import cv2 as cv
from picamera2 import Picamera2
import time
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

# Event to stop the worker process
stop_event = mp.Event()

# More global variables
scale_factor = 1.3 # For rescaling the image for bounding box drawing
video_writer1 = None # Required for video writing
video_writer2 = None # Required for video writing
last_save_time = time.time() # For saving images at a regular interval
frame_queue1 = mp.Queue(maxsize=2) # Queue for sharing frames between processes
frame_queue2 = mp.Queue(maxsize=2) # Queue for sharing frames between processes
box_queue1 = mp.Queue(maxsize=2) # Queue for sharing bounding boxes between processes
box_queue2 = mp.Queue(maxsize=2) # Queue for sharing bounding boxes between processes
full_bodies1 = [] # List to hold detected bounding boxes
full_bodies2 = [] # List to hold detected bounding boxes
full_bodies_scaled1 = [] # List to hold scaled bounding boxes for drawing
full_bodies_scaled2 = [] # List to hold scaled bounding boxes for drawing
frame_max = 10000 # Just a safety to prevent infinite loops during testing
send_over_network = True # Set to True to enable sending images over the network to geeqie
last_send_time = time.time() # For sending images at a regular interval
sequencing_timer = 0
alarm_timer = 0

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
    # State 0 does not record anything, just shows the preview
    0: {"res": (640, 480), "hz": 3, "mode": "preview"},
    1: {"res": (640, 480), "hz": 3, "mode": "img"},
    2: {"res": (1280, 720), "hz": 6, "mode": "img"}, 
    3: {"res": (1920, 1080), "hz": 30, "mode": "video"}
}

picam2 = Picamera2(0)
picam3 = Picamera2(1)

# detection function
def detectFullBody(frame_queue, box_queue, stop_event):
    while not stop_event.is_set():
        try:
            # print("Getting frame from queue")
            frame = frame_queue.get(timeout=0.5)
        except:
            continue
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)
        full_bodies = fullbody_cascade.detectMultiScale(frame_gray, scaleFactor=scale_factor, minNeighbors=3)
        # print(f"Putting full bodies in queue: {full_bodies}")
        box_queue.put(full_bodies)

# frame generation function
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
    # set config based on state
    config = configs[state]
    width, height = config["res"]
    mode, hz = config["mode"], config["hz"]
    if state == 2:
        alarm_timer = time.time()
    else:
        alarm_timer = 0
    if state == 3:
        # Do whatever is done to make the buzzer go off here
        pass
    picam.configure(picam.create_preview_configuration(main={"format": "RGB888", "size": (width, height)}))
    picam.start()
    print(f"State {state}: {width}x{height} {mode} @ {hz}Hz")
    return width, height, mode, hz, alarm_timer

# Create worker function for box drawing
worker1 = mp.Process(target=detectFullBody,
                    args=(frame_queue1, box_queue1, stop_event),
                    daemon=True) # Set as daemon so it will exit when the main process exits
worker1.start()

worker2 = mp.Process(target=detectFullBody,
                    args=(frame_queue2, box_queue2, stop_event),
                    daemon=True) # Set as daemon so it will exit when the main process exits
worker2.start()

width, height, mode, hz, alarm_timer = config_state(state, picam2)
_, _, _, _, _ = config_state(state, picam3)

print("Recording... Press 'q' to quit")

if send_over_network:
    #subprocess.run(['sudo', 'bash', 'send_image_geeqie.sh'], check=True)
    subprocess.Popen(['sudo', 'bash', 'send_image_geeqie.sh'])

cv.namedWindow('Capture - Full body detection 1')
cv.namedWindow('Capture - Full body detection 2')
cv.moveWindow('Capture - Full body detection 1', 40, 40)
cv.moveWindow('Capture - Full body detection 2', 960, 40)

# loop time
for frame1, frame2 in zip(generate_frames(picam2), generate_frames(picam3)):
    
    # determine state (get rid of this in final version)
    buttonpress = cv.waitKey(10) & 0xFF
    if buttonpress == ord('q'):
            break
    elif buttonpress == ord('0'):
            state = 0
            width, height, mode, hz, alarm_timer = config_state(state, picam2)
            _, _, _, _, _ = config_state(state, picam3)
    elif buttonpress == ord('1'):
            state = 1
            width, height, mode, hz, alarm_timer = config_state(state, picam2)
            _, _, _, _, _ = config_state(state, picam3)
    elif buttonpress == ord('2'):
            state = 2
            width, height, mode, hz, alarm_timer = config_state(state, picam2)
            _, _, _, _, _ = config_state(state, picam3)
    elif buttonpress == ord('3'):
            state = 3
            width, height, mode, hz, alarm_timer = config_state(state, picam2)
            _, _, _, _, _ = config_state(state, picam3)

    resized1 = cv.resize(frame1, (640, 480))
    resized2 = cv.resize(frame2, (640, 480))
    try:
        frame_queue1.put_nowait(resized1.copy())
        #print("Put frame")
    except:
        pass

    try:
        frame_queue2.put_nowait(resized2.copy())
        #print("Put frame")
    except:
        pass

    try:
        while True:
            full_bodies1 = box_queue1.get_nowait()
            print(f"Detected full bodies: {full_bodies1}")
    except:
        pass

    try:
        while True:
            full_bodies2 = box_queue2.get_nowait()
            print(f"Detected full bodies: {full_bodies2}")
    except:
        pass

    #full_bodies_scaled1 = []
    #full_bodies_scaled2 = []

    # drawing the boxes when available
    if len(full_bodies1) > 0 or len(full_bodies2) > 0:
        if state == 0:
            state = 1
            width, height, mode, hz, alarm_timer = config_state(state, picam2)
            _, _, _, _, _ = config_state(state, picam3)
        else:
            for targets in [full_bodies1, full_bodies2]:
                full_bodies_scaled = []
                for (x,y,w,h) in targets:
                    x = int(x * width / 640)
                    y = int(y * height / 480)
                    w = int(w * width / 640)
                    h = int(h * height / 480)
                    # Tuples are immutable, so you have to make a new one to scale it
                    full_bodies_scaled.append((x, y, w, h))
                for (x,y,w,h) in full_bodies_scaled:
                    # rectangle uses top left corner and bottom right corner
                    top_left = (x, y)
                    bottom_right = (x + w, y + h)
                    if (targets == full_bodies1).any():
                        frame1 = cv.rectangle(frame1, top_left, bottom_right, (255,0,0), 2)
                        cv.imshow('Capture - Full body detection 1', cv.resize(frame1, (800, 600)))
                        cv.imwrite(os.path.join(stream_dir, "frame1.jpg"), frame1, [cv.IMWRITE_JPEG_QUALITY, 90])
                        if send_over_network and time.time() - last_send_time > 1.0:
                            #subprocess.run(['sudo', 'bash', 'send_image_geeqie.sh'], check=True)
                            subprocess.Popen(['sudo', 'bash', 'send_image_geeqie.sh'])
                            last_send_time = time.time()
                    else:
                        frame2 = cv.rectangle(frame2, top_left, bottom_right, (255,0,0), 2)
                        cv.imshow('Capture - Full body detection 2', cv.resize(frame2, (800, 600)))
                        cv.imwrite(os.path.join(stream_dir, "frame2.jpg"), frame2, [cv.IMWRITE_JPEG_QUALITY, 90])
                        #if send_over_network and time.time() - last_send_time > 1.0:
                            #subprocess.run(['sudo', 'bash', 'send_image_geeqie.sh'], check=True)
                            #subprocess.Popen(['sudo', 'bash', 'send_image_geeqie.sh'])
                            #last_send_time = time.time()
                
        if state == 2:
            if time.time() - alarm_timer > 10:
                state = 3
                width, height, mode, hz = config_state(state, picam2)
                _, _, _, _ = config_state(state, picam3)
    # shows just the frame if no boxes are drawn
    else:
        cv.imshow('Capture - Full body detection 1', cv.resize(frame1, (800, 600)))
        cv.imshow('Capture - Full body detection 2', cv.resize(frame2, (800, 600)))

    # video mode code
    if mode == "video":
        if video_writer1 is None:
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            filename = os.path.join(output_dir, f"cam1_s{state}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            video_writer1 = cv.VideoWriter(filename, fourcc, hz, (width, height))
            # print(f"VIDEO: {os.path.basename(filename)}")
        video_writer1.write(frame1) if video_writer1 else None
        if video_writer2 is None:
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            filename = os.path.join(output_dir, f"cam2_s{state}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            video_writer2 = cv.VideoWriter(filename, fourcc, hz, (width, height))
            # print(f"VIDEO: {os.path.basename(filename)}")
        video_writer2.write(frame2) if video_writer2 else None
    
    # image mode code
    if mode == "img":
        now = time.time()
        if now - last_save_time >= 1.0 / hz:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = os.path.join(output_dir, f"cam1_s{state}_{timestamp}.jpg")
            
            last_save_time = now

            # JPEG compress
            cv.imwrite(filename, frame1, [cv.IMWRITE_JPEG_QUALITY, 90])
            # print(f"IMG: {os.path.basename(filename)}")

            filename = os.path.join(output_dir, f"cam2_s{state}_{timestamp}.jpg")

            # JPEG compress
            cv.imwrite(filename, frame2, [cv.IMWRITE_JPEG_QUALITY, 90])
            
    frame_count += hz / 30
    if frame_count > frame_max: # Just a safety to prevent infinite loops during testing
        print("Reached 300 frames, exiting loop.")
        break

# cleanup 
stop_event.set()
worker1.join()
worker2.join()

frame_queue1.cancel_join_thread()
frame_queue1.close()

frame_queue2.cancel_join_thread()
frame_queue2.close()

box_queue1.cancel_join_thread()
box_queue1.close()

box_queue2.cancel_join_thread()
box_queue2.close()

picam2.stop()
if video_writer1:
    video_writer1.release()
if video_writer2:
    video_writer2.release()
cv.destroyAllWindows()
cv.waitKey(1)
print("Done.")
