# This is just a reference
# I think what needs to be done is we need to separate detection in a thread separate from the rest of the code.
# We might not even need to time how many frames are between drawing bounding boxes.
# The only shared objects should be the current frame and the detected bounding boxes.
# Whenever the detection thread is ready, it can update the bounding boxes and sample a new frame from what the main thread has collected.
# I'm getting tired, so I might do this later.

import cv2
import threading
import time

# Load the cascade
haar_cascade = cv2.CascadeClassifier('path/to/haarcascade.xml')

# Initialize video capture
cap = cv2.VideoCapture(0)  # Change to your video source

# Shared data structures
frame_lock = threading.Lock()
detected_objects = []
drawing_frame = None

# Detection thread function
def detect_objects():
    global detected_objects, drawing_frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Object detection
        objects = haar_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=3)

        with frame_lock:
            detected_objects.clear()  # Clear previous detections
            for (x, y, w, h) in objects:
                detected_objects.append((x, y, w, h))
            drawing_frame = frame.copy()  # Make a separate copy of the frame for drawing

# Drawing thread function
def draw_rectangles():
    global drawing_frame, detected_objects

    while True:
        with frame_lock:
            if drawing_frame is not None:
                for (x, y, w, h) in detected_objects:
                    cv2.rectangle(drawing_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Show the frame with rectangles
                cv2.imshow('Detected Objects', drawing_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Create threads
detection_thread = threading.Thread(target=detect_objects)
drawing_thread = threading.Thread(target=draw_rectangles)

# Start threads
detection_thread.start()
drawing_thread.start()

# Wait for threads to finish
detection_thread.join()
drawing_thread.join()

# Clean up
cap.release()
cv2.destroyAllWindows()
