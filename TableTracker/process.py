import cv2
import numpy as np
import torch
import csv
import matplotlib.pyplot as plt
from refine_corners import *
from ultralytics import YOLO
from scipy.ndimage import binary_dilation, median_filter
from unet import UNet  # Ensure unet.py is in the same directory

# ----------------------------
# Helper Functions
# ----------------------------

def median_smoothing(array, window_size):
    """
    Apply median smoothing to an array.

    Args:
        array (np.ndarray): Input array.
        window_size (int): Smoothing parameter.

    Returns:
        np.ndarray: Smoothed array.
    """
    smoothed_array = median_filter(array, size=(window_size, 1), mode='reflect')
    return smoothed_array.astype(int)

def smooth_data(data, window_size=30):
    data = np.array(data)
    
    for col in range(1, data.shape[1]):
        non_zero_indices = np.nonzero(data[:, col])[0]
        if non_zero_indices.size > 0:
            first_non_zero = non_zero_indices[0]
            if first_non_zero > 0:
                data[:first_non_zero, col] = data[first_non_zero, col]
    
    # Apply median smoothing to all columns starting from index 1
    data[:, 1:] = median_smoothing(data[:, 1:], window_size)
    
    return data

def extend_mask(mask, k_x, k_y):
    kernel = np.ones((2 * k_y + 1, 2 * k_x + 1), dtype=int)
    extended_mask = binary_dilation(mask, structure=kernel).astype(mask.dtype)
    return extended_mask

# ----------------------------
# Detection Functions
# ----------------------------

def get_detections(img_path, model):
    model_output = model(img_path, verbose=False)[0]
    
    orig_image = model_output.orig_img
    classes = model_output.boxes.cls
    
    table_detections = torch.where(classes == 0)
    table_bbox = None  # Initialize table_bbox
    if len(table_detections[0]) == 1:        
        x, y, w, h = model_output.boxes.xywh[table_detections[0]].squeeze().cpu().detach().numpy()
        w, h = 1.2 * w, 1.6 * h
        table_bbox = np.array([x - w / 2, x + w / 2, y - h / 2, y + h / 2])
        table_bbox = table_bbox.round().astype(int)
        
        table_mask = model_output.masks.data[table_detections[0]].squeeze().cpu().detach().numpy()
        fy = orig_image.shape[0] / table_mask.shape[0]
        fx = orig_image.shape[1] / table_mask.shape[1]
        table_mask = extend_mask(table_mask, 0, int(table_mask.sum() / 4500))
        table_mask = cv2.resize(table_mask, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    else:
        table_mask = None  # Ensure table_mask is defined
    
    base_detections = torch.where(classes == 1)
    if len(base_detections[0]) == 1:
        base_mask = model_output.masks.data[base_detections[0]].squeeze().cpu().detach().numpy()
        fy = orig_image.shape[0] / base_mask.shape[0]
        fx = orig_image.shape[1] / base_mask.shape[1]
        base_mask = extend_mask(base_mask, 0, int(base_mask.sum() / 900))
        base_mask = cv2.resize(base_mask, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    else:
        base_mask = None
        
    return orig_image, table_mask, base_mask, table_bbox

def get_corners(table_mask):
    if table_mask is None:
        return np.zeros((4, 2), dtype=int)
    y, x = np.where(table_mask == 1)
    if len(x) == 0 or len(y) == 0:
        return np.zeros((4, 2), dtype=int)
    pts = np.array([x, y]).T
    tl = pts[np.argmax(pts @ [-1, -1])]
    bl = pts[np.argmax(pts @ [-1, +1])]
    tr = pts[np.argmax(pts @ [+1, -1])]
    br = pts[np.argmax(pts @ [+1, +1])]
    return np.array((tl, tr, bl, br))

def get_base(base_mask):
    if base_mask is None:
        return np.zeros((2, 2), dtype=int)
    y, x = np.where(base_mask == 1)
    if len(x) == 0 or len(y) == 0:
        return np.zeros((2, 2), dtype=int)
    pts = np.array([x, y]).T
    bl = pts[np.argmax(pts @ [-1, +1])]
    br = pts[np.argmax(pts @ [+1, +1])]
    return np.array((bl, br))

def process_frame(frame, model):    
    orig_image, table_mask, base_mask, table_bbox = get_detections(frame, model)
    corners = get_corners(table_mask)
    base = get_base(base_mask)
    if base is not None and len(base) == 2:
        base_height = (base[0][1] + base[1][1]) // 2
    else:
        base_height = 0
    return corners, base_height, table_bbox

# ----------------------------
# Main Processing Function
# ----------------------------

def process_video(input_video_path, output_csv_name, indent=0):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the UNet and YOLO models
    unet_model = load_model('models_corners/best_model.pth', device)
    model = YOLO("models_yolo/yolov8n-seg-finetuned1.pt").to(device)
    
    print(f"Processing video: {input_video_path}")
    cap = cv2.VideoCapture(input_video_path)

    data = []
    table_bboxes = []
    frame_number = 0
    corners = np.zeros((6, 2), dtype=int)
    base_height = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            corners, base_height, table_bbox = process_frame(frame, model)
            table_bboxes.append(table_bbox)
        except Exception as e:
            print(f"Error processing frame {frame_number}: {e}")
            corners = np.zeros((4, 2), dtype=int)
            base_height = 0
            table_bboxes.append(None)
        
        # Fill missing corners with previous frame's corners
        for i in range(len(corners)):
            x, y = corners[i]
            if x == 0 and y == 0 and 'prev_corners' in locals():
                corners[i] = prev_corners[i]
        prev_corners = corners.copy()
        
        # Append data
        data.append([
            frame_number,
            *corners.flatten(),
            base_height,
            indent
        ])
    
        frame_number += 1

    cap.release()
    cv2.destroyAllWindows()

    # Convert data to NumPy array for processing
    data = np.array(data)
    
    # Median smoothing
    data_smoothed = smooth_data(data, window_size=30)
    
    # After smoothing, refine the corners using refine_corners
    cap = cv2.VideoCapture(input_video_path)
    refined_data = []
    frame_number = 0
    prev_refined_corners = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            table_bbox = table_bboxes[frame_number]

            # Get the smoothed corners as prior
            prior_corners = data_smoothed[frame_number][1:-2]  # Exclude frame_number, base_height, indent
            prior_corners_array = prior_corners.reshape(-1, 2)
            # Map prior_corners_array to labels
            prior_points = {
                'top_left': prior_corners_array[0],
                'top_right': prior_corners_array[1],
                'bottom_left': prior_corners_array[2],
                'bottom_right': prior_corners_array[3],
                'top_mid': (prior_corners_array[0] + prior_corners_array[1]) / 2,
                'bottom_mid': (prior_corners_array[2] + prior_corners_array[3]) / 2,
            }
            refined_corners = refine_corners(unet_model, device, frame, table_bbox, prior_points)
        except Exception as e:
            refined_corners = data_smoothed[frame_number][1:-2]
            refined_corners = refined_corners.reshape(-1, 2)
            refined_corners = np.concatenate((refined_corners, np.zeros((2, 2))), 0)
        
        # Fill missing corners with previous frame's corners
        if prev_refined_corners is not None:
            for i in range(len(refined_corners)):
                x, y = refined_corners[i]
                if x == 0 and y == 0:
                    refined_corners[i] = prev_refined_corners[i]
                    
        prev_refined_corners = refined_corners.copy()
        
        refined_data.append([
            frame_number,
            *refined_corners.flatten(),
            data_smoothed[frame_number][-2],  # base_height
            data_smoothed[frame_number][-1]   # indent
        ])
        
        frame_number += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Convert data to NumPy array for processing
    refined_data = np.array(refined_data)
    
    # Median smoothing
    refined_data_smoothed = smooth_data(refined_data, window_size=15)
    
    # Write refined data to CSV
    with open(f"{output_csv_name}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "frame",
            "back_left_x", "back_left_y",
            "back_right_x", "back_right_y",
            "front_left_x", "front_left_y",
            "front_right_x", "front_right_y",
            "back_mid_x", "back_mid_y",
            "front_mid_x", "front_mid_y",
            "base_y", "indent"
        ])
        for row in refined_data_smoothed:
            writer.writerow(row)
    
    return refined_data

# ----------------------------
# Entry Point
# ----------------------------

if __name__ == "__main__":
    process_video("../matches/match1/match1_108.mp4", "temp")



        