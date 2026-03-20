import cv2 as cv
from picamera2 import Picamera2, Preview
import time
import os
from datetime import datetime
import argparse
from picamera2.encoders import H264Encoder

state = 2  # change to real state select code later
recording = False
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
    0: {"size": (640, 480), "FrameDurationLimits": (1e6/3, 1e6/3), "mode": "img"},
    1: {"size": (1280, 720), "FrameDurationLimits": (1e6/6, 1e6/6), "mode": "img"}, 
    2: {"size": (1920, 1080), "FrameDurationLimits": (1e6/20, 1e6/20), "mode": "video"}
}

# set config based on state
config = configs[state]
width, height = config["size"]
mode, hz = config["mode"], config["FrameDurationLimits"][0] if config["FrameDurationLimits"][0] != 0 else float('inf')

print(f"State {state}: {width}x{height} {mode} @ {hz}Hz")

# set video res
video_config = picam2.create_video_configuration(main={"format": "RGB888", "size": (1920, 1080)})
picam2.configure(video_config)
picam2.start()
# ------------------------------------------------------------------
#cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
#cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
#cap.set(cv.CAP_PROP_FPS, 60)

video_writer = None
last_save_time = time.time()

print("Recording... Press 'q' to quit")

while True:
    frame = picam2.capture_array() # Captures as OpenCV-compatible array
    if frame is None:
        print("Failed to capture frame")
        break
    #ret, frame = cap.read()
    #if not ret:
    #    break
    
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
        #if video_writer is None:
        #    fourcc = cv.VideoWriter_fourcc(*'h264')
        #    filename = os.path.join(output_dir, f"s{state}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h264")
        #    video_writer = cv.VideoWriter(filename, fourcc, hz, (width, height))
        #    print(f"VIDEO: {os.path.basename(filename)}")
        if recording == False:
            encoder = H264Encoder(bitrate=10000000)
            output = os.path.join(output_dir, f"s{state}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h264")
            picam2.start_recording(encoder, output)
            recording = True
        #resized = cv.resize(frame, (width, height))
        #video_writer.write(frame)

        resized = cv.resize(frame, (640, 480))
        if frame_count % 30 == 0:
            full_bodies = detectFullBody(resized)
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S_%f"))
        if len(full_bodies) > 0:
            for (x,y,w,h) in full_bodies:
                center = (x + w//2, y + h//2)
                frame = cv.rectangle(resized, center,(x + w, y + h),(255,0,0),2)
                cv.imshow('Capture - Full body detection', frame)
        else:
            cv.imshow('Capture - Full body detection', frame)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

        frame_count += 1

if recording:
    picam2.stop_recording()
picam2.stop()
cv.destroyAllWindows()
print("Done.")
