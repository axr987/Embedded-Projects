import cv2 as cv
from picamera2 import Picamera2, Preview
import time
import os
from datetime import datetime
import argparse

state = 2  # change to real state select code later
frame_count = 0
output_dir = "captures"
os.makedirs(output_dir, exist_ok=True)

def detectFullBody(frame):
    print("Start frame")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S_%f"))
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    #-- detect full bodies
    full_bodies = fullbody_cascade.detectMultiScale(frame_gray,scaleFactor = 1.01, minNeighbors=5)
    return full_bodies

def generate_frames():
    while True:
        frame = picam2.capture_array()
        if frame is not None:
            yield frame
        else:
            print("Failed to capture frame")

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--fullbody_cascade', help='Path to face cascade.', default='../data/haarcascade_fullbody.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()

fullbody_cascade_name = args.fullbody_cascade

fullbody_cascade = cv.CascadeClassifier()

#-- load cascade
if not fullbody_cascade.load(os.path.join("code", fullbody_cascade_name)):
    print('--(!)Error loading full body cascade')
    exit(0)

picam2 = Picamera2()
# Modify resolution below
# picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))

configs = {
    0: {"res": (640, 480), "hz": 3, "mode": "img"},
    1: {"res": (1280, 720), "hz": 6, "mode": "img"}, 
    2: {"res": (1920, 1080), "hz": 30, "mode": "video"}
}

# set config based on state
config = configs[state]
width, height = config["res"]
mode, hz = config["mode"], config["hz"]

picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (1920, 1080)}))
picam2.start()
print(f"State {state}: {width}x{height} {mode} @ {hz}Hz")

video_writer = None
last_save_time = time.time()

print("Recording... Press 'q' to quit")

for frame in generate_frames():

    now = time.time()
    
    if mode == "img":
        if now - last_save_time >= 1.0 / hz:
            resized = cv.resize(frame, (width, height))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = os.path.join(output_dir, f"s{state}_{timestamp}.jpg")
            
            # JPEG compress
            cv.imwrite(filename, resized, [cv.IMWRITE_JPEG_QUALITY, 90])
            print(f"IMG: {os.path.basename(filename)}")
            
            last_save_time = now
    
    elif mode == "video":
        if video_writer is None:
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            filename = os.path.join(output_dir, f"s{state}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            video_writer = cv.VideoWriter(filename, fourcc, hz, (width, height))
            print(f"VIDEO: {os.path.basename(filename)}")
    
    resized = cv.resize(frame, (640, 480))
    if frame_count % 10 == 0:
        full_bodies = detectFullBody(resized)
    if len(full_bodies) > 0:
        for (x,y,w,h) in full_bodies:
            x = int(x * width / 640)
            y = int(y * height / 480)
            w = int(w * width / 640)
            h = int(h * height / 480)
            center = (x + w//2, y + h//2)
            frame = cv.rectangle(frame, center,(x + w, y + h),(255,0,0),2)
            cv.imshow('Capture - Full body detection', frame)
    else:
        cv.imshow('Capture - Full body detection', frame)
    cv.waitKey(10)
    video_writer.write(frame) if video_writer else None
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
    frame_count += 1

picam2.stop()
if video_writer:
    video_writer.release()
cv.destroyAllWindows()
print("Done.")
