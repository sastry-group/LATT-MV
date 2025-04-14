import csv
import cv2
import numpy as np
import codecs
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from .table import get_homography_matrix

def last_none_start(arr):
    """
    Find the starting index of the last sequence of None values in the array.

    Args:
        arr (list): Input list.

    Returns:
        int: The starting index of the last sequence of None values.
    """
    n = len(arr)
    
    # Start from the end of the array and move backwards
    for i in range(n-1, -1, -1):
        if arr[i] is not None:
            # If we encounter a non-None value, return the next index (i + 1) if it is within bounds
            return i + 1 if i + 1 < n else -1
    
    # If the entire array is None
    return 0 if n > 0 else -1

def median_smoothing(array, window_size):
    """
    Apply median smoothing to an array.

    Args:
        array (np.ndarray): Input array.
        q (int): Smoothing parameter.

    Returns:
        np.ndarray: Smoothed array.
    """
    smoothed_array = median_filter(array, size=(window_size, 1), mode='reflect')
    return smoothed_array.astype(int)

def load_frames(video_path):
    """
    Load frames from a video file.

    Args:
        video_path (str): Path to the video file.

    Returns:
        np.ndarray: Array of video frames.
        int: FPS of the video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) 
    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        frames.append(frame)
    cap.release()
    
    return np.array(frames), fps, frame_size

def load_table_data(video_name, table_dir, frames, verbose=False):
    """
    Load table data from CSV files.

    Args:
        video_name (str): Name of the video.
        table_dir (str): Directory containing table data CSV files.
        verbose (bool, optional): Whether to print detailed information. Defaults to False.
        orig_frames (np.ndarray, optional): Original video frames. Defaults to None.

    Returns:
        tuple: Two lists containing homography matrices.
    """
    table_path = f"{table_dir}/{video_name}_table.csv"
    homs_table, homs = [], []
    
    data = []
    with open(table_path, "r") as table_file:
        csv_reader = csv.reader(table_file)
        header = csv_reader.__next__()
        for row in csv_reader:
            row = list(map(float, row))
            data.append(row)
    
    data = np.array(data)    
    table_boundaries, base_boundaries = [], []
    n = len(frames)
    for i in range(n):
        row = data[i]
        boundary = np.array(row[1:9]).reshape(4, 2)
        table_boundaries.append(np.array(row[1:13]).reshape(6, 2))
        H1, H2, base_boundary = get_homography_matrix(boundary, row[13], row[14], verbose=(verbose and i in [0, n-1]), orig_frame=frames[i])
        base_boundaries.append(base_boundary)
        homs_table.append(H1)
        homs.append(H2)
    return homs_table, homs, table_boundaries, base_boundaries

def load_paddle_data(video_name, paddle_dir):
    """
    Load paddle data from CSV files.

    Args:
        video_name (str): Name of the video.
        paddle_dir (str): Directory containing the paddle data CSV file.
    """
    paddle_path = f"{paddle_dir}/{video_name}_paddle.csv"
    
    data = []
    with open(paddle_path, "r") as table_file:
        csv_reader = csv.reader(table_file)
        header = csv_reader.__next__()
        for row in csv_reader:
            arr = codecs.escape_decode(row[1][2:-1])[0]
            arr = np.fromstring(arr, dtype="float32")
            if len(arr) > 0:
                arr = arr.reshape((-1, 2))
            data.append(arr)
    return data
    
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def dist(a, b):
    return np.linalg.norm(a-b)