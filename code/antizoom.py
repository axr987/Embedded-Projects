import os
os.environ["QT_LOGGING_RULES"] = "*.warning=false"

import sys
import cv2 as cv
import time
import multiprocessing as mp
from ultralytics import YOLO
from picamera2 import Picamera2
import RPi.GPIO as GPIO
from datetime import datetime
import threading

from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

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
stream_dir = "temp_stream"
os.makedirs(stream_dir, exist_ok=True)
last_save_time = time.time()
frameset1 = []
frameset2 = []

#Sunfounder resolution 2592x1944
#RPi cam resolution 4608x2592

configs = {
    0: {"res": (640, 480), "hz": 3, "mode": "preview"},
    1: {"res": (640, 480), "hz": 3, "mode": "img"},
    2: {"res": (1280, 720), "hz": 6, "mode": "img"}, 
    3: {"res": (1920, 1080), "hz": 30, "mode": "video"}
}

frame_queue1 = mp.Queue(maxsize=2)
frame_queue2 = mp.Queue(maxsize=2)
box_queue1 = mp.Queue(maxsize=2)
box_queue2 = mp.Queue(maxsize=2)
stop_event = mp.Event()

# ---------------- CAMERA ----------------
picam2 = Picamera2(0)
picam3 = Picamera2(1)

full_res1 = (picam2.sensor_resolution[0] // 2, picam2.sensor_resolution[1] // 2)
full_res2 = (picam3.sensor_resolution[0] // 2, picam3.sensor_resolution[1] // 2)

picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": full_res1}
))
picam3.configure(picam3.create_preview_configuration(
    main={"format": "RGB888", "size": full_res2}
))

picam2.start()
picam3.start()

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

def alarm_worker():
    global alarm_active, alarm_step, alarm_last_step

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
def detectFullBody(frame_queue, box_queue, stop_event):
    # YOLO
    model = YOLO("yolov8n.pt")
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except:
            continue

        results = model(frame, verbose=False, imgsz=640, conf=0.5, classes=[class_num])
        boxes = []

        if results[0].boxes is not None:
            for box in results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                boxes.append((x1, y1, x2, y2))

        try:
            box_queue.put_nowait(boxes)
        except:
            pass

# ---------------- GUI ----------------
class CameraGUI(QWidget):
    def __init__(self, on_close=None):
        super().__init__()
        self.on_close = on_close

        self.setWindowTitle("Dual Camera Monitor")

        self.cam1 = QLabel()
        self.cam2 = QLabel()

        self.state_label = QLabel("State: 0")
        self.state_label.setStyleSheet("font-size: 18px; font-weight: bold;")

        layout_h = QHBoxLayout()
        layout_h.addWidget(self.cam1)
        layout_h.addWidget(self.cam2)

        layout_v = QVBoxLayout()
        layout_v.addLayout(layout_h)
        layout_v.addWidget(self.state_label)

        self.setLayout(layout_v)

    def update_ui(self, frame1, frame2, state, alarm):
        self.cam1.setPixmap(self.to_qt(frame1))
        self.cam2.setPixmap(self.to_qt(frame2))

        color = ["gray", "blue", "orange", "red"][state]
        self.state_label.setStyleSheet(f"font-size: 18px; color: {color};")

        self.state_label.setText(
            f"State: {state} | Alarm: {'ON' if alarm else 'OFF'} | Resolution1: {frame1.shape[1]}x{frame1.shape[0]} | Resolution2: {frame2.shape[1]}x{frame2.shape[0]}"
        )

    def to_qt(self, frame):
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(img).scaled(640, 480)
    
    def closeEvent(self, event):
        if self.on_close:
            self.on_close()
        event.accept()

# ---------------- MAIN APP ----------------
class App:
    def __init__(self):
        global state

        self.gui = CameraGUI(on_close=self.exit_app)
        self.gui.show()

        # Start cameras
        set_state(state)

        # Start workers
        self.worker1 = mp.Process(target=detectFullBody, args=(frame_queue1, box_queue1, stop_event), daemon=True)
        self.worker2 = mp.Process(target=detectFullBody, args=(frame_queue2, box_queue2, stop_event), daemon=True)
        self.worker1.start()
        self.worker2.start()

        # Get alarm thread running
        self.alarm_thread = threading.Thread(target=alarm_worker, daemon=True)
        self.alarm_thread.start()

        self.full_bodies1 = []
        self.full_bodies2 = []

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(30)

        self.cleaned_up = False

    def update(self):
        global state, alarm_active, past_target_area1, past_target_area2, state2start, configs, output_dir, video_writer1, video_writer2, last_save_time, frameset1, frameset2

        framefull1 = picam2.capture_array()
        framefull2 = picam3.capture_array()
        frame1 = cv.resize(framefull1, configs[state]["res"])
        frame2 = cv.resize(framefull2, configs[state]["res"])

        #frameset1.append(frame1)
        #frameset2.append(frame2)

        try: frame_queue1.put_nowait(frame1)
        except: pass
        try: frame_queue2.put_nowait(frame2)
        except: pass

        # Get detections
        try:
            self.full_bodies1 = box_queue1.get_nowait()
            self.just_finished1 = True
        except:
            self.just_finished1 = False

        try:
            self.full_bodies2 = box_queue2.get_nowait()
            self.just_finished2 = True
        except:
            self.just_finished2 = False

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
            #h, w = frame.shape[:2]
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

        if state == 3:
            alarm_active = True
        else:
            alarm_active = False

        # Update GUI
        self.gui.update_ui(frame1, frame2, state, alarm_active)

        # Video mode
        if configs[state]["mode"] == "video":
            if video_writer1 is None:
                fourcc = cv.VideoWriter_fourcc(*'mp4v')
                video_writer1 = cv.VideoWriter(os.path.join(output_dir, f"cam1_s{state}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"), fourcc, configs[state]["hz"], configs[state]["res"])
            video_writer1.write(frame1)
            if video_writer2 is None:
                video_writer2 = cv.VideoWriter(os.path.join(output_dir, f"cam2_s{state}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"), fourcc, configs[state]["hz"], configs[state]["res"])
            video_writer2.write(frame2)

        # Image mode
        elif configs[state]["mode"] == "img":
            now = time.time()
            if now - last_save_time >= 5.0 / configs[state]["hz"]:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                cv.imwrite(os.path.join(output_dir, f"cam1_s{state}_{timestamp}.jpg"), frame1, [cv.IMWRITE_JPEG_QUALITY, 90])
                cv.imwrite(os.path.join(output_dir, f"cam2_s{state}_{timestamp}.jpg"), frame2, [cv.IMWRITE_JPEG_QUALITY, 90])
                last_save_time = now

    def cleanup(self):
        if self.cleaned_up:
            return

        self.cleaned_up = True

        stop_event.set()

        # Give workers a moment to exit
        self.worker1.join(timeout=1)
        self.worker2.join(timeout=1)

        for q in [frame_queue1, frame_queue2, box_queue1, box_queue2]:
            q.cancel_join_thread()
            q.close()

        try:
            picam2.stop()
            picam3.stop()
        except:
            pass

        try:
            buzzer_pwm.ChangeDutyCycle(0)
            buzzer_pwm.stop()
        except:
            pass

        try:
            GPIO.cleanup()
        except:
            pass

    def exit_app(self):
        self.cleanup()
        QApplication.quit()

# ---------------- RUN ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_app = App()

    sys.exit(app.exec_())