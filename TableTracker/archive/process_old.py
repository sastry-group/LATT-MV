from ultralytics import YOLO

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import csv


def extend_mask(mask, k_x, k_y, flag=False):
    extended_mask = mask.copy()
    
    m, n = mask.shape
    for i in range(m):
        for j in range(n):
            if mask[i][j] == 1:
                a, b, c, d = max(0, i-k_y), min(i+k_y+1, m), max(0, j-k_x), min(j+k_x+1, n)
                if flag: b = min(i+2, m)
                extended_mask[a:b, c:d] = 1
    return extended_mask

def get_detections(img_path, model):
    model_output = model(img_path)[0]
    orig_image = model_output.orig_img
    classes = model_output.boxes.cls
    
    table_detections = torch.where(classes == 0)
    if len(table_detections) == 1:
        table_mask = model_output.masks.data[table_detections[0]].squeeze().cpu().detach().numpy()
        table_mask = extend_mask(table_mask, 4, 8)
        fy = orig_image.shape[0] / table_mask.shape[0]
        fx = orig_image.shape[1] / table_mask.shape[1]
        table_mask = cv2.resize(table_mask, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
        table_mask = np.expand_dims(table_mask, axis=-1)
    else:
        table_mask = None
    
    base_detections = torch.where(classes == 1)
    if len(base_detections) == 1:
        base_mask = model_output.masks.data[base_detections[0]].squeeze().cpu().detach().numpy()
        base_mask = extend_mask(base_mask, 3, 5)
        fy = orig_image.shape[0] / base_mask.shape[0]
        fx = orig_image.shape[1] / base_mask.shape[1]
        base_mask = cv2.resize(base_mask, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
        base_mask = np.expand_dims(base_mask, axis=-1)
    else:
        base_mask = None
        
    return orig_image, table_mask, base_mask

def segment_to_line(x1, y1, x2, y2):
    """Convert a line segment to a line in the form ax + by = c"""
    a = y2 - y1
    b = x1 - x2
    c = a * x1 + b * y1
    return a, b, c

def intersection(line1, line2):
    """Find the intersection point of two lines given in the form ax + by = c"""
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    determinant = a1 * b2 - a2 * b1
    if determinant == 0:
        return None  # Lines are parallel
    else:
        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
        return int(x), int(y)

def find_table_corners(t, b, l, r):
    """Find the four corners of the table"""
    # Convert segments to lines
    t = None if t is None else segment_to_line(*t)
    b = None if b is None else segment_to_line(*b)
    l = None if l is None else segment_to_line(*l)
    r = None if r is None else segment_to_line(*r)
    
    # Find intersection points
    tl = (0, 0) if t is None or l is None else intersection(t, l)
    tr = (0, 0) if t is None or r is None else intersection(t, r)
    bl = (0, 0) if b is None or l is None else intersection(b, l)
    br = (0, 0) if b is None or r is None else intersection(b, r)
    
    return np.array((tl, tr, bl, br))

def get_table_corners(orig_image, table_mask):
    image = orig_image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 25, 150)
    edges = np.where(table_mask[:, :, 0] == 1, edges, 0)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=25, minLineLength=15, maxLineGap=5)
    
    # Initialize variables to store the top, bottom, left, and right lines
    t = b = l = r = None

    # Classify and find the extreme lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < abs(x2 - x1):  # Horizontal line
                if t is None or min(y1, y2) < min(t[1], t[3]): t = line[0]
                if b is None or max(y1, y2) > max(b[1], b[3]): b = line[0]
            else:  # Vertical line
                if l is None or min(x1, x2) < min(l[0], l[2]): l = line[0]
                if r is None or max(x1, x2) > max(r[0], r[2]): r = line[0]
        
        if t is not None and b is not None and abs(t[1] - b[1]) < 15: t = b = None
        if l is not None and r is not None and abs(l[0] - r[0]) < 30: l = r = None
        
    corners = find_table_corners(t, b, l, r)
    return corners

def get_base(orig_image, base_mask):
    image = np.where(base_mask == 1, orig_image, 255)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=25, minLineLength=15, maxLineGap=5)

    min_line = None
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if min_line is None or min(y1, y2) > min(min_line[1], min_line[3]): 
                min_line = line[0]
    return min_line

def plot_extended_line(image, pt1, pt2, color=(0, 255, 0), thickness=2):
    """Plots a line passing through pt1 and pt2 extending to the borders of the image."""
    height, width, _ = image.shape
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Calculate the slope (m) and intercept (c) of the line
    if x1 == x2:  # Vertical line
        cv2.line(image, (x1, 0), (x1, height), color, thickness)
    elif y1 == y2:
        cv2.line(image, (0, y1), (width, y1), color, thickness)
    else:
        m = (y2 - y1) / (x2 - x1)
        c = y1 - m * x1
        
        # Calculate intersection points with the image borders
        points = []
        
        # Intersect with left border (x = 0)
        y = int(c)
        if 0 <= y <= height:
            points.append((0, y))
        
        # Intersect with right border (x = width)
        y = int(m * width + c)
        if 0 <= y <= height:
            points.append((width, y))
        
        # Intersect with top border (y = 0)
        x = int(-c / m)
        if 0 <= x <= width:
            points.append((x, 0))
        
        # Intersect with bottom border (y = height)
        x = int((height - c) / m)
        if 0 <= x <= width:
            points.append((x, height))
        
        if len(points) >= 2:
            cv2.line(image, points[0], points[1], color, thickness)

def process_frame(frame, model):
    global table_mask
    
    orig_image, table_mask, base_mask = get_detections(frame, model)
    corners = get_table_corners(orig_image, table_mask)
    base = get_base(orig_image, base_mask)
    return base, corners

def process_video(input_video_path, output_csv_name, indent=0):
    model = YOLO("models/yolov8n-seg-finetuned1.pt").cuda()
    
    print(f"{input_video_path}")
    cap = cv2.VideoCapture(input_video_path)

    data = []
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        base, corners = process_frame(frame, model)
        for i in range(len(corners)):
            x, y = corners[i]
            if x == 0 and y == 0 and "prev_corners" in locals():
                corners[i] = prev_corners[i]
        prev_corners = corners 
        base_height = (base[1] + base[3]) // 2
        
        data.append([
            frame_number,
            *corners.flatten(),
            base_height,
            indent
        ])
    
        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()
    
    with open(f"{output_csv_name}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["frame","back_left_x","back_left_y","back_right_x","back_right_y","front_left_x","front_left_y","front_right_x","front_right_y","base_y","indent"])
        for row in data:
            writer.writerow(row)
    
    return data

    
