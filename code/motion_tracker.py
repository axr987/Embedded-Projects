import cv2
import numpy as np

def detect_motion(prev_frame, curr_frame, threshold=25):
    """
    Detect motion via frame differencing.
    
    Args:
        prev_frame: Previous grayscale or color frame (H, W) or (H, W, 3)
        curr_frame: Current grayscale or color frame
        threshold: Intensity diff threshold for motion (0-255)
    
    Returns:
        motion_mask: Binary mask (255 where motion detected)
        motion_count: Number of motion pixels
    """
    # Convert to grayscale if needed
    if len(prev_frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray, curr_gray = prev_frame, curr_frame
    
    # Blur to reduce noise
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (21, 21), 0)
    
    # Frame differencing
    frame_diff = cv2.absdiff(prev_gray, curr_gray)
    motion_mask = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)[1]
    motion_mask = cv2.dilate(motion_mask, None, iterations=2)
    
    motion_count = np.sum(motion_mask > 0)
    
    return motion_mask, motion_count

# 
cap = cv2.VideoCapture(0)
ret, prev_frame = cap.read()
while True:
    ret, curr_frame = cap.read()
    if not ret: break
    
    motion_mask, motion_count = detect_motion(prev_frame, curr_frame, threshold=25)
    
    # Use motion_mask (e.g., overlay on curr_frame)
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(curr_frame, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Motion Overlay', curr_frame)
    cv2.imshow('Motion Mask', motion_mask)
    
    prev_frame = curr_frame.copy()
    if cv2.waitKey(1) & 0xFF == ord('q'): break







