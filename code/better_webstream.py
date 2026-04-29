import os
os.environ["QT_LOGGING_RULES"] = "*.warning=false"

import sys
import cv2 as cv
import time
import multiprocessing as mp
from ultralytics import YOLO
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import threading
import numpy as np
from multiprocessing import shared_memory
from multiprocessing import Value
from flask import Flask, Response, jsonify, request

# ---------------- GPIO SETUP ----------------
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)
buzzer_pwm = GPIO.PWM(18, 523)
buzzer_pwm.start(0)

# ---------------- GLOBALS ----------------
class_num = 0 # 0 or 14 for person or bird respectively
state = 0
last_state = -1
alarm_active = False
note_interval = 2 ** (1/12)
note_start = 523
alarm_last_step = time.time()
alarm_step = 0
curent_target_area = 0
past_target_area1 = 0
past_target_area2 = 0
approach_scale = 2
deapproach_scale = 0.5
state2start = time.time()
video_writer1 = None
video_writer2 = None
output_dir = "captures"
os.makedirs(output_dir, exist_ok=True)
last_save_time = time.time()
shutdown_requested = False

# Sunfounder resolution 2592x1944
# RPi cam resolution 4608x2592

configs = {
    0: {"res": (640, 480), "hz": 3, "mode": "preview"},
    1: {"res": (640, 480), "hz": 3, "mode": "img"},
    2: {"res": (1280, 720), "hz": 6, "mode": "img"}, 
    3: {"res": (1920, 1080), "hz": 30, "mode": "video"}
}

box_queue1 = mp.Queue(maxsize=2)
box_queue2 = mp.Queue(maxsize=2)
stop_event = mp.Event()

# ---------------- CAMERA ----------------
picam2 = Picamera2(0)
picam3 = Picamera2(1)

full_res1 = (picam2.sensor_resolution[0] // 2, picam2.sensor_resolution[1] // 2)
full_res2 = (picam3.sensor_resolution[0] // 2, picam3.sensor_resolution[1] // 2)
#full_res2 = (1920, 1080) # RPi cam has issues with higher res, so just use 640x480

picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": full_res1}
))
picam3.configure(picam3.create_preview_configuration(
    main={"format": "RGB888", "size": full_res2}
))

picam2.start()
picam3.start()

# --------- SETUP SHARED MEMORY ------------
frame_shape = (480, 640, 3)
frame_size = np.prod(frame_shape)

shm1 = shared_memory.SharedMemory(create=True, size=frame_size)
shm2 = shared_memory.SharedMemory(create=True, size=frame_size)

shared_frame1 = np.ndarray(frame_shape, dtype=np.uint8, buffer=shm1.buf)
shared_frame2 = np.ndarray(frame_shape, dtype=np.uint8, buffer=shm2.buf)

frame_id1 = Value('i', 0)
frame_id2 = Value('i', 0)

# ---------------- WEB GLOBALS ----------------
app = Flask(__name__)

latest_frame1 = None
latest_frame2 = None
latest_boxes1 = []
latest_boxes2 = []
latest_state = 0
latest_alarm = False

# ---------------- STATE ----------------
def set_state(new_state):
    global state, last_state, video_writer1, video_writer2

    if new_state == last_state:
        return

    state = new_state
    last_state = new_state

    if video_writer1:
        video_writer1.release()
        video_writer1 = None
    if video_writer2:
        video_writer2.release()
        video_writer2 = None

# ---------------- ALARM ----------------
def alarm_worker():
    global alarm_active, alarm_step

    intervals = [0.05, 0.05, 0.10, 0.05]
    freqs = [
        note_start,
        note_interval * note_start,
        note_interval ** 11 * note_start,
        None
    ]

    while not stop_event.is_set():
        if not alarm_active:
            buzzer_pwm.ChangeDutyCycle(0)
            time.sleep(0.01)
            continue

        freq = freqs[alarm_step]

        if freq is not None:
            buzzer_pwm.ChangeDutyCycle(50)
            buzzer_pwm.ChangeFrequency(freq)
        else:
            buzzer_pwm.ChangeDutyCycle(0)

        time.sleep(intervals[alarm_step]) 

        alarm_step = (alarm_step + 1) % len(intervals)

# ---------------- DETECTION PROCESS ----------------
def detectFullBody(shm_name, frame_id, box_queue, stop_event):
    shm = shared_memory.SharedMemory(name=shm_name)
    frame = np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf)

    model = YOLO("yolov8n.pt")
    last_id = -1

    while not stop_event.is_set():
        if frame_id.value == last_id:
            time.sleep(0.005)
            continue

        last_id = frame_id.value

        results = model(frame, verbose=False, imgsz=640, conf=0.5, classes=[class_num])

        boxes = []
        if results[0].boxes is not None:
            for box in results[0].boxes.xyxy.cpu().numpy():
                boxes.append(tuple(map(int, box)))

        try:
            box_queue.put_nowait(boxes)
        except:
            pass

# ---------------- STREAM ----------------
def generate(cam_id):
    global latest_frame1, latest_frame2

    while True:
        frame = latest_frame1 if cam_id == 1 else latest_frame2

        if frame is None:
            time.sleep(0.01)
            continue

        _, jpeg = cv.imencode('.jpg', frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               jpeg.tobytes() + b'\r\n')

@app.route('/cam1')
def cam1():
    return Response(generate(1),
        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cam2')
def cam2():
    return Response(generate(2),
        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/boxes')
def boxes():
    return jsonify({
        "cam1": latest_boxes1,
        "cam2": latest_boxes2,
        "state": latest_state,
        "alarm": latest_alarm
    })

def shutdown_server():
    with app.test_request_context():
        func = request.environ.get('werkzeug.server.shutdown')
        if func:
            func()

def delayed_shutdown():
    time.sleep(0.5)
    shutdown_server()

@app.route('/shutdown', methods=['POST'])
def shutdown():
    global shutdown_requested
    shutdown_requested = True

    threading.Thread(target=delayed_shutdown).start()

    return "Shutting down..."

@app.route('/')
def index():
    return """
<html>
<body style="margin:0;background:black;color:white;">
<h2 id="status"></h2>

<div style="position:relative;display:inline-block;">
<img src="/cam1">
<canvas id="c1" width="640" height="480"
style="position:absolute;top:0;left:0;"></canvas>
</div>

<div style="position:relative;display:inline-block;">
<img src="/cam2">
<canvas id="c2" width="640" height="480"
style="position:absolute;top:0;left:0;"></canvas>
</div>

<script>
const c1 = document.getElementById("c1").getContext("2d");
const c2 = document.getElementById("c2").getContext("2d");

async function update(){
    const res = await fetch('/boxes');
    const data = await res.json();

    document.getElementById("status").innerText =
        `State: ${data.state} | Alarm: ${data.alarm ? "ON":"OFF"}`;

    draw(c1, data.cam1);
    draw(c2, data.cam2);
}

function draw(ctx, boxes){
    ctx.clearRect(0,0,640,480);

    for (let b of boxes){
        let [x1,y1,x2,y2] = b;
        ctx.strokeStyle = "blue";
        ctx.strokeRect(x1,y1,x2-x1,y2-y1);
    }
}

setInterval(update, 100);
</script>

<button onclick="shutdown()" 
        style="position:fixed; top:10px; right:10px; padding:10px; font-size:16px;">
    STOP SYSTEM
</button>

<script>
function shutdown() {
    fetch('/shutdown', { method: 'POST' });
}
</script>

</body>
</html>
"""

# ---------------- MAIN APP ----------------
class App:
    def __init__(self):
        self.worker1 = mp.Process(target=detectFullBody,
            args=(shm1.name, frame_id1, box_queue1, stop_event), daemon=True)
        self.worker2 = mp.Process(target=detectFullBody,
            args=(shm2.name, frame_id2, box_queue2, stop_event), daemon=True)

        self.worker1.start()
        self.worker2.start()

        threading.Thread(target=alarm_worker, daemon=True).start()

        threading.Thread(target=self.run_server, daemon=True).start()

        self.full_bodies1 = []
        self.full_bodies2 = []
        self.cleaned_up = False

        self.loop()

    def run_server(self):
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    
    def loop(self):
        global latest_frame1, latest_frame2
        global latest_boxes1, latest_boxes2
        global latest_state, latest_alarm
        global state, past_target_area1, past_target_area2, state2start
        global alarm_active, last_save_time, video_writer1, video_writer2
        global shutdown_requested

        while True:
            if shutdown_requested:
                self.exit_app()
                return
            framefull1 = picam2.capture_array()
            framefull2 = picam3.capture_array()
            frame1 = cv.resize(framefull1, configs[state]["res"])
            frame2 = cv.resize(framefull2, configs[state]["res"])
            small1 = cv.resize(framefull1, (640, 480))
            small2 = cv.resize(framefull2, (640, 480))

            shared_frame1[:] = small1
            with frame_id1.get_lock():
                frame_id1.value += 1

            shared_frame2[:] = small2
            with frame_id2.get_lock():
                frame_id2.value += 1

            try: self.full_bodies1 = box_queue1.get_nowait()
            except: pass
            try: self.full_bodies2 = box_queue2.get_nowait()
            except: pass

            # State 0 to 1 transition and any to 0 transition
            if any(len(b) > 0 for b in [self.full_bodies1, self.full_bodies2]):
                if state == 0:
                    state = 1
                    past_target_area1 = max([(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in self.full_bodies1], default=0)
                    past_target_area2 = max([(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in self.full_bodies2], default=0)
                    set_state(state)
            else:
                state = 0
                past_target_area1 = 0
                past_target_area2 = 0
                set_state(state)

            # State 2 to 3 transition

            if state == 2 and time.time() - state2start > 10:
                state = 3
                set_state(state)

            # Draw boxes
            for (boxes, frame, cam_id) in [(self.full_bodies1, frame1, 1), (self.full_bodies2, frame2, 2)]:
                for (x1, y1, x2, y2) in boxes:
                    x1 = int(x1 * configs[state]["res"][0] / 640)
                    y1 = int(y1 * configs[state]["res"][1] / 480)
                    x2 = int(x2 * configs[state]["res"][0] / 640)
                    y2 = int(y2 * configs[state]["res"][1] / 480)
                    cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    current_target_area = (x2 - x1) * (y2 - y1)
                    if cam_id == 1:
                        if past_target_area1 == 0:
                            past_target_area1 = current_target_area
                        past_target_area = past_target_area1
                    else:
                        if past_target_area2 == 0:
                            past_target_area2 = current_target_area
                        past_target_area = past_target_area2
                    # State 1 to 2 transition
                    if state == 1 and current_target_area > past_target_area * approach_scale:
                        state = 2
                        state2start = time.time()
                        globals()['past_target_area%d' % cam_id] = current_target_area  # Save past
                        set_state(state)
                    # State 2 or 3 to 1 transition
                    if state in [2, 3]:
                        if current_target_area < past_target_area * deapproach_scale:
                            state = 1
                            globals()['past_target_area%d' % cam_id] = current_target_area  # Save past
                            set_state(state)

            alarm_active = (state == 3)

            latest_frame1 = small1.copy()
            latest_frame2 = small2.copy()
            latest_boxes1 = self.full_bodies1
            latest_boxes2 = self.full_bodies2
            latest_state = state
            latest_alarm = alarm_active

    def cleanup(self):
        if self.cleaned_up:
            return

        self.cleaned_up = True

        stop_event.set()

        # Give workers a moment to exit
        self.worker1.join(timeout=1)
        self.worker2.join(timeout=1)

        try:
            shm1.close()
            shm1.unlink()
            shm2.close()
            shm2.unlink()
        except:
            pass

        for q in [box_queue1, box_queue2]:
            q.cancel_join_thread()
            q.close()

        try:
            picam2.stop()
            picam3.stop()
        except:
            pass

        try:
            if buzzer_pwm is not None:
                buzzer_pwm.stop()
                del buzzer_pwm
        except:
            pass

        try:
            GPIO.cleanup()
        except:
            pass

    def exit_app(self):
        self.cleanup()

# ---------------- RUN ----------------
if __name__ == "__main__":
    App()