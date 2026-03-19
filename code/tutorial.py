from picamera2 import Picamera2, MappedArray, Preview
from picamera2.encoders import H264Encoder
import cv2
import time
import numpy as np
from datetime import datetime
from os import path
import asyncio

#cap = cv2.VideoCapture(0)

#while (cap.isOpened()):
#	ret, frame = cap.read()
#	cv2.imshow('Frame Window', frame)
#	cv2.waitKey(60)

#cap.release()
#cv2.destroyAllWindows()

# Because we're using RPi cams, we waste some time and power converting to numpy arrays
framecount = 100
picam2 = Picamera2()
# Modify resolution below
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.start()
while (framecount > 0):
    frame = picam2.capture_array() # Captures as OpenCV-compatible array
    cv2.imshow("Libcamera Feed", frame)
    cv2.waitKey(60) # Delay between frames in ms
    framecount = framecount - 1
cv2.imwrite("testphoto1.jpg", frame)
picam2.stop()

# This is a callback executed each frame before the frame is saved, I think.
def apply_timestamp(request):
    global nowold
    global nownew
    nownew = datetime.now()
    now = nownew - nowold
    nowold = nownew
    timestamp = f"{now.microseconds} us"
    # Put timestamp on the frame at specific coordinates with specific font, color, and thickness
    with MappedArray(request, "main") as m:
        cv2.putText(m.array, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


nowold = datetime.now()
nownew = datetime.now()

# Record video for 10 s
video_config = picam2.create_video_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(video_config)

# target_fps:
fps = 60
frame_ns = int(1e6 / fps)               # microseconds per frame
picam2.set_controls({"FrameDurationLimits": (frame_ns, frame_ns)})

# Encode video
encoder = H264Encoder(bitrate=10000000)
output = "test.h264"

# This callback is to test the frame rate
picam2.pre_callback = apply_timestamp

# Interestingly, there is no need to convert to numpy arrays. I don't know why you need to do that with individual frames.

picam2.start_recording(encoder, output)
t_range = np.linspace(0,10,num=600)
overlay = np.zeros((480, 640, 4), np.uint8)
for t in t_range:
    frame = picam2.capture_array()
    cv2.imshow("Libcamera Feed", frame)
    cv2.waitKey(10)
picam2.stop_recording()