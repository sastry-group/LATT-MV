import cv2
import numpy as np
import torch
import csv
import matplotlib.pyplot as plt
from ultralytics import YOLO

def process_frame(frame, model):
    model_output = model(frame, verbose=False)[0]
    detections = []
    for box in model_output.boxes:
        c = box.conf[0]
        if c > 0.75 and len(detections) < 2:
            x, y, w, h = box.xywh.cpu()[0]
            detections.append([x, y])
    return np.array(detections).tostring()
            
def process_video(path, output_csv_path):
    model = YOLO("models/yolov8s-oiv7-finetuned.pt").cuda()

    print(f"{path}")
    cap = cv2.VideoCapture(path)
    
    data = []
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        paddle_detections = process_frame(frame, model)
        data.append([
            frame_number, 
            paddle_detections
        ])
        frame_number += 1
        
    cap.release()
    cv2.destroyAllWindows()
    with open(f"{output_csv_path}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "paddle_detections"])
        for row in data:
            writer.writerow(row)
    return data

if __name__ == "__main__":
    data = process_video("match38_23.mp4", "temp")

    
    

    
