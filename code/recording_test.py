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

# detection function
def detectFullBody(frame):
    print("Start frame")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S_%f"))
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    #-- detect full bodies
    full_bodies = fullbody_cascade.detectMultiScale(frame_gray,scaleFactor = 1.01, minNeighbors=5)
    return full_bodies

# frame generation function
def generate_frames():
    while True:
        frame = picam2.capture_array()
        if frame is not None:
            yield frame
        else:
            print("Failed to capture frame")

# Create argument parser for cascade and camera 
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

# turn camera on
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

# loop time
for frame in generate_frames():

    now = time.time()


    # image mode code
    if mode == "img":
        if now - last_save_time >= 1.0 / hz:
            resized = cv.resize(frame, (width, height))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = os.path.join(output_dir, f"s{state}_{timestamp}.jpg")
            
            # JPEG compress
            cv.imwrite(filename, resized, [cv.IMWRITE_JPEG_QUALITY, 90])
            print(f"IMG: {os.path.basename(filename)}")
            
            last_save_time = now

    # video mode code
    elif mode == "video":
        if video_writer is None:
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            filename = os.path.join(output_dir, f"s{state}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            video_writer = cv.VideoWriter(filename, fourcc, hz, (width, height))
            print(f"VIDEO: {os.path.basename(filename)}")
    resized = cv.resize(frame, (640, 480))

    # drawing the boxes after a set of frames
    if frame_count % 10 == 0:
        full_bodies = detectFullBody(resized)
        for (x,y,w,h) in full_bodies:
            x = int(x * width / 640)
            y = int(y * height / 480)
            w = int(w * width / 640)
            h = int(h * height / 480)
            center = (x + w//2, y + h//2)
    # shows the boxes if drawn
    if len(full_bodies) > 0: 
            frame = cv.rectangle(frame, center,(x + w, y + h),(255,0,0),2)
            cv.imshow('Capture - Full body detection', frame)
    # shows the frame if not drawn
    else:
        cv.imshow('Capture - Full body detection', frame)
    
    # delay
    cv.waitKey(10)
    
    # save the video
    video_writer.write(frame) if video_writer else None
    
    # another delay
    if cv.waitKey(10) & 0xFF == ord('q'):
        break
    frame_count += 1

# cleanup 
picam2.stop()
if video_writer:
    video_writer.release()
cv.destroyAllWindows()
print("Done.")
