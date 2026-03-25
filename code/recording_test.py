import cv2 as cv
from picamera2 import Picamera2, Preview
import time
import os
from datetime import datetime
import argparse

# Global variables
state = 0 # change to real state select code later
frame_count = 0 # Replace with threading later
output_dir = "captures"
os.makedirs(output_dir, exist_ok=True)
scale_factor = 1.1 # For rescaling the image for bounding box drawing
video_writer = None # Required for video writing
last_save_time = time.time() # For saving images at a regular interval

# Create argument parser for cascade and camera 
parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--fullbody_cascade', help='Path to face cascade.', default='../data/haarcascade_fullbody.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()

fullbody_cascade_name = args.fullbody_cascade

# Modify resolution below
# picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
configs = {
    0: {"res": (640, 480), "hz": 3, "mode": "img"},
    1: {"res": (1280, 720), "hz": 6, "mode": "img"}, 
    2: {"res": (1920, 1080), "hz": 30, "mode": "video"}
}

# detection function
def detectFullBody(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    #-- detect full bodies
    full_bodies = fullbody_cascade.detectMultiScale(frame_gray, scaleFactor=scale_factor, minNeighbors=3)
    return full_bodies

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

#-- load cascade
fullbody_cascade = cv.CascadeClassifier()
if not fullbody_cascade.load(os.path.join("code", fullbody_cascade_name)):
    print('--(!)Error loading full body cascade')
    exit(0)

# turn camera on
picam2 = Picamera2()

width, height, mode, hz = config_state(state)

print("Recording... Press 'q' to quit")

# loop time
for frame in generate_frames():

    now = time.time()
    
    resized = cv.resize(frame, (640, 480))

    # drawing the boxes after a set of frames
    if frame_count % 10 == 0:
        full_bodies = detectFullBody(resized)
        for (x,y,w,h) in full_bodies:
            x = int(x * width / 640)
            y = int(y * height / 480)
            w = int(w * width / 640)
            h = int(h * height / 480)
    # shows the boxes if drawn
    print(width, height)
    if len(full_bodies) > 0:
            # rectangle uses top left corner and bottom right corner
            top_left = (x, y)
            bottom_right = (x + w, y + h)
            frame = cv.rectangle(frame, top_left, bottom_right, (255,0,0), 2)
            cv.imshow('Capture - Full body detection', frame)
    # shows just the frame if no boxes are drawn
    else:
        cv.imshow('Capture - Full body detection', frame)
    
    # delay
    buttonpress = cv.waitKey(10) & 0xFF
    if buttonpress == ord('q'):
            break
    elif buttonpress == ord('0'):
            width, height, mode, hz = config_state(0)
    elif buttonpress == ord('1'):
            width, height, mode, hz = config_state(1)
    elif buttonpress == ord('2'):
            width, height, mode, hz = config_state(2)
        
    # video mode code
    if mode == "video":
        if video_writer is None:
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            filename = os.path.join(output_dir, f"s{state}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            video_writer = cv.VideoWriter(filename, fourcc, hz, (width, height))
            print(f"VIDEO: {os.path.basename(filename)}")
        video_writer.write(frame) if video_writer else None
    
    # image mode code
    if mode == "img":
        if now - last_save_time >= 1.0 / hz:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = os.path.join(output_dir, f"s{state}_{timestamp}.jpg")
            
            # JPEG compress
            cv.imwrite(filename, frame, [cv.IMWRITE_JPEG_QUALITY, 90])
            print(f"IMG: {os.path.basename(filename)}")
            
            last_save_time = now
  
    frame_count += 1

# cleanup 
picam2.stop()
if video_writer:
    video_writer.release()
cv.destroyAllWindows()
print("Done.")
