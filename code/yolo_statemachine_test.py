# Multiprocessing script with YOLO detector (updated from cascade)

import os
os.environ["QT_LOGGING_RULES"] = "*.warning=false"  # Suppress warning about missing fonts

import cv2 as cv
from picamera2 import Picamera2
import time
from datetime import datetime
import multiprocessing as mp
import subprocess
from ultralytics import YOLO
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)  # Alarm output pin
buzzer_pwm = GPIO.PWM(18, 523)  # Buzzer on GPIO 18 at 1kHz
buzzer_pwm.start(0)  # Start with buzzer off

# Global variables
class_num = 0 # 0 for people, 14 for birds with YOLOv8.
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
scale_factor = 1.0
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
past_target_area1 = 0
past_target_area2 = 0
note_interval = 2 ** (1/12)
note_start = 523
alarm_last_step = time.time()
alarm_step = 0
alarm_active = False

# Load YOLO model (person class: 0)
model = YOLO('yolov8n.pt')

configs = {
    0: {"res": (640, 480), "hz": 3, "mode": "preview"},
    1: {"res": (640, 480), "hz": 3, "mode": "img"},
    2: {"res": (1280, 720), "hz": 6, "mode": "img"}, 
    3: {"res": (1920, 1080), "hz": 30, "mode": "video"}
}

picam2 = Picamera2(0)
picam3 = Picamera2(1)

def cleanup_GPIO():
    try:
        buzzer_pwm.ChangeDutyCycle(0)
        buzzer_pwm.stop()
    except:
        pass
    GPIO.cleanup()

# YOLO detection function (class 0 = person)
def detectFullBody(frame_queue, box_queue, stop_event):
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.5)
        except:
            continue
        # Run YOLO inference
        results = model(frame, verbose=False, conf=0.5, classes=[class_num])  # Person only, conf>0.5
        boxes = []
        if results[0].boxes is not None:
            for box in results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                boxes.append((x1, y1, x2, y2))
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

def update_alarm():
    global alarm_last_step, alarm_step

    now = time.time()

    # timing for each step
    intervals = [0.05, 0.05, 0.10, 0.05]
    freqs = [
        note_start,
        note_interval * note_start,
        note_interval ** 11 * note_start,
        None  # silence
    ]

    if now - alarm_last_step >= intervals[alarm_step]:
        alarm_last_step = now

        if freqs[alarm_step] is not None:
            buzzer_pwm.ChangeDutyCycle(50)
            buzzer_pwm.ChangeFrequency(freqs[alarm_step])
        else:
            buzzer_pwm.ChangeDutyCycle(0)

        alarm_step = (alarm_step + 1) % len(intervals)

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

alarm_on_timer = 0
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

    for full_bodies, frame, cam_id in detections:
        if len(full_bodies) > 0:
            if state == 0:
                state = 1
                width, height, mode, hz, alarm_timer = config_state(state, picam2)
                _, _, _, _, _ = config_state(state, picam3)

            full_bodies_scaled = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in full_bodies]
            
            for x1, y1, x2, y2 in full_bodies_scaled:
                x1 = int(x1) * configs[state]["res"][0] // 640
                y1 = int(y1) * configs[state]["res"][1] // 480
                x2 = int(x2) * configs[state]["res"][0] // 640
                y2 = int(y2) * configs[state]["res"][1] // 480
                current_target_area = (x2 - x1) * (y2 - y1)
                past_target_area = past_target_area1 if cam_id == 1 else past_target_area2
                
                # Approach detection
                if state == 1 and current_target_area > past_target_area * approach_scale:
                    state = 2
                    globals()['past_target_area%d' % cam_id] = past_target_area  # Save past
                    width, height, mode, hz, alarm_timer = config_state(state, picam2)
                    _, _, _, _, _ = config_state(state, picam3)
                
                # Draw box
                cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Update past area
                if cam_id == 1: past_target_area1 = current_target_area
                else: past_target_area2 = current_target_area

            # Stream/save preview
            stream_fn = os.path.join(stream_dir, f"frame{cam_id}.jpg")
            cv.imwrite(stream_fn, frame, [cv.IMWRITE_JPEG_QUALITY, 90])
            if send_over_network and time.time() - last_send_time > 1.0:
                subprocess.Popen(['sudo', 'bash', 'send_image_geeqie.sh'])
                last_send_time = time.time()

            #cv.imshow(f'Capture - Full body detection {cam_id}', cv.resize(frame, (800, 600)))
            cv.imshow(f'Capture - Full body detection {cam_id}', frame)

    # State transitions
    if state == 2 and time.time() - alarm_timer > 10:
        state = 3
        alarm_on_timer = time.time()
        width, height, mode, hz, alarm_timer = config_state(state, picam2)
        _, _, _, _, _ = config_state(state, picam3)
    elif state == 2 and min(past_target_area1, past_target_area2) < (globals().get('saved_past_target_area1', 0) or globals().get('saved_past_target_area2', 0)) and time.time() - alarm_timer > 3:
        state = 1
    elif state == 3 and time.time() - alarm_on_timer > 5:
        state = 1

    if not any(len(b) > 0 for b in [full_bodies1, full_bodies2]) and time.time() - last_send_time > 5:
        state = 0

    # state 3 things 
    if state == 3:
        alarm_active = True
    else:
        alarm_active = False
        buzzer_pwm.ChangeDutyCycle(0)  # ensure OFF

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

    if alarm_active:
        update_alarm()

    frame_count += 1
    if frame_count > frame_max:
        break

# Cleanup
stop_event.set()
worker1.join()
worker2.join()
time.sleep(1)
cleanup_GPIO()
time.sleep(1)
for q in [frame_queue1, frame_queue2, box_queue1, box_queue2]:
    q.cancel_join_thread()
    q.close()
picam2.stop()
picam3.stop()
if video_writer1: video_writer1.release()
if video_writer2: video_writer2.release()
cv.destroyAllWindows()
print("Done.")