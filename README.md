# Embedded-Projects
## Projects for ECSE 488 Embedded Systems and Thinkbox Embedded AI Microgrant

### Thanks to the Sears think\[box\] at CWRU for funding this project!

### This repository contains the code for Flappydex, our autonomous birdwatcher. Currently, it is able to distinguish birds from the surrounding environment at a close range.

### This repository contains a captures folder, a code folder, a data folder, a temp_stream folder, a CAD folder, and some other files.

## captures
The captures folder contains images and video that are collected with the code.

## code
The code folder, of course, contains our code.
- antizoom.py was a test script to attempt to prevent the zoom of the cameras from changing when the device changed states.
- better_webstream.py is a script for testing the Flask server.
- buzzertest.py tests the buzzer connected to the Pi with a breadboard circuit.
- cascade_class_multiprocess.py tests human detection with Haar cascade classifiers and tests using multiprocessing to speed up bounding box drawing alongside collecting and displaying frames.
- cascade_statemachine.py is a version of the human detection code using Haar cascade classifiers that also includes a state machine to change the capture frame rate depending on the situation.
- flappydex_birder.py is the final version of the code for our automatic bird tracking system.
- fullbody_tracker_test.py is a semi-pseudocode version of what we wanted for the Haar cascade classifier code.
- motion_tracker.py experiments with frame differencing to detect motion.
- recording_test.py was our first attempt using the Haar cascade classifiers to save images and video of bounding boxes around people.
- ***security_system_final.py is the final version of our security system code.***
- shared_memory.py tests the use of shared memory instead of queues to reduce memory load.
- tutorial.py tests the functionality of one of the cameras connected to the Pi.
- webstream.py was our first attempt of using a Flask server.
- yolo_multiprocess.py tests using YOLO for detection of people and multiprocessing to speed up bounding box drawing with YOLO alongside frame capture and display.
- yolo_statemachine_gui_wip.py was our most up-to-date code before transitioning to using a Flask server for displaying frames. It used a GUI displayed to the user's monitor.
- yolo_statemachine_test.py is an earlier iteration of yolo_statemachine_gui_wip.py

## data
The data folder includes two .xml files, bird-cascade.xml for Haar cascade classifier testing with bird subjects, and haarcascade_fullbody.xml for Haar cascade classifier testing with human subjects.

## temp_stream
The temp_stream folder contains a few images that are relayed over to other devices connected to the Pi with Windows Remote Desktop.

## CAD
The CAD folder contains the CAD used to make the PLA housing for the device.

## Remaining Files
The remaining files include:

- birdphotoex.jpg to demonstrate that YOLOv8 can draw bounding boxes around birds.
- midterm_report_pic.jpg to demonstrate that YOLO can draw bounding boxes around people.
- This README.
- send_image_geeqie.sh, which is currently how we send images over to other devices connected to the Pi with Windows Remote Desktop.
- yolov5s.pt, which we had used previously for just human detection.
- yolov8n.pt, which we are using for bird detection.
