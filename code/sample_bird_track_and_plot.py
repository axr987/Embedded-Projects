import argparse
import os
import csv
from datetime import datetime

import cv2
import pandas as pd
import torch
import matplotlib.pyplot as plt

CONFIDENCE_THRESHOLD = 0.35
NMS_IOU_THRESHOLD = 0.45
CSV_FILE = "bird_counts.csv"

def run(video_path):
   model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
   model.conf = CONFIDENCE_THRESHOLD
   model.iou = NMS_IOU_THRESHOLD

   cap = cv2.VideoCapture(video_path)
   if not cap.isOpened():
       raise RuntimeError(f"Could not open video file: {video_path}")

   fps = cap.get(cv2.CAP_PROP_FPS)
   frame_count = 0

   with open(CSV_FILE, mode="w", newline="") as f:
       writer = csv.writer(f)
       writer.writerow(["time_seconds", "bird_count"])

       while True:
           ok, frame = cap.read()
           if not ok:
               break

           results = model(frame, size=640)
           df = results.pandas().xyxy[0]
           bird_df = df[(df["name"] == "bird") & (df["confidence"] >= CONFIDENCE_THRESHOLD)]
           bird_count = len(bird_df)

           time_sec = frame_count / fps
           writer.writerow([round(time_sec, 2), bird_count])

           display_frame = frame.copy()
           for _, row in bird_df.iterrows():
               x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
               conf = float(row["confidence"])
               label = f"bird {conf:.2f}"
               cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
               cv2.putText(display_frame, label, (x1, max(0, y1 - 6)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

           cv2.imshow("Bird Tracking", display_frame)
           if cv2.waitKey(1) & 0xFF == ord("q"):
               break

           frame_count += 1

   cap.release()
   cv2.destroyAllWindows()

   data = pd.read_csv(CSV_FILE)
   plt.figure(figsize=(10, 5))
   plt.plot(data["time_seconds"], data["bird_count"], marker="o")
   plt.xlabel("Time (seconds)")
   plt.ylabel("Number of birds")
   plt.title("Birds detected over time")
   plt.grid(True)
   plt.savefig("bird_graph.png")
   plt.show()

def parse_args():
   parser = argparse.ArgumentParser()
   parser.add_argument("--video", type=str, default="input.mp4")
   return parser.parse_args().video
