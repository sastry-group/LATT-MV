import torch
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import os
import multiprocessing
from functools import partial

from cnn import CNNVideoFrameClassifier
from constants import *

def preprocess_frame(frame, width, height):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize
    resized = cv2.resize(gray, (width, height))
    # Normalize to [0, 1] range
    normalized = resized.astype(float) / 255.0
    # Convert to tensor and add batch and channel dimensions
    tensor = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0)
    return tensor

def correct_predictions(df, window_size=10):
    df = df.copy()
    for i in range(len(df)):
        left_window = df.loc[max(0, i-window_size):i-1, 'InSegment']
        right_window = df.loc[i+1:min(len(df)-1, i+window_size), 'InSegment']
        
        left_majority = left_window.mean() > 0.5
        right_majority = right_window.mean() > 0.5
        
        if df.loc[i, 'InSegment'] == 0 and left_majority and right_majority:
            df.loc[i, 'InSegment'] = 1
        elif df.loc[i, 'InSegment'] == 1 and not left_majority and not right_majority:
            df.loc[i, 'InSegment'] = 0
    
    return df

def find_segments(sequence):
    segments = []
    start = None
    for i, value in enumerate(sequence):
        if value == 1:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - 1 - start > 30:
                    segments.append((start, i - 1))
                start = None
    # Catch any segment that ends at the last index
    if start is not None and len(sequence) - 1 - start > 30:
        segments.append((start, len(sequence) - 1))
    return segments

def save_segment(video_path, output_folder, segment, segment_number):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    start_frame, end_frame = segment
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_folder, f"{video_name}_{segment_number}.mp4")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for _ in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    cap.release()
    out.release()

def run_inference(model, video_path, output_folder, width, height, batch_size=32, gpu_id=0):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    model = model.to(device)
    model.eval()
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    results = []
    batch_frames = []
    batch_frame_numbers = []

    with torch.no_grad():
        for frame_number in tqdm(range(total_frames), desc=f"Processing {os.path.basename(video_path)} on GPU {gpu_id}"):
            ret, frame = cap.read()
            if not ret:
                break
            
            preprocessed_frame = preprocess_frame(frame, width, height)
            batch_frames.append(preprocessed_frame)
            batch_frame_numbers.append(frame_number)
            
            if len(batch_frames) == batch_size or frame_number == total_frames - 1:
                batch_tensor = torch.cat(batch_frames, dim=0).to(device)
                outputs = model(batch_tensor)
                probabilities = torch.softmax(outputs, dim=1)[:, 1]
                predictions = (probabilities > 0.75).int()
                
                for i, pred in enumerate(predictions):
                    results.append({
                        'Frame': batch_frame_numbers[i],
                        'InSegment': pred.item(),
                        'Probability': probabilities[i].item()
                    })
                
                batch_frames = []
                batch_frame_numbers = []

    cap.release()

    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Apply correction
    print(f"Applying correction to predictions for {os.path.basename(video_path)}...")
    df_corrected = correct_predictions(df)
    
    # Find segments
    segments = find_segments(list(df_corrected['InSegment']))
    
    # Save segments
    print(f"Saving segments for {os.path.basename(video_path)}...")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_folder = os.path.join(output_folder, video_name)
    os.makedirs(video_output_folder, exist_ok=True)
    
    for i, segment in enumerate(segments):
        save_segment(video_path, video_output_folder, segment, i+1)
    
    print(f"Saved {len(segments)} segments for {os.path.basename(video_path)} to {video_output_folder}")

def process_video(video_file, args, gpu_id):
    video_path = os.path.join(args.video_folder, video_file)
    
    # Load the model for each process
    model = CNNVideoFrameClassifier(args.width, args.height)
    model.load_state_dict(torch.load('best_model.pth', map_location=f'cuda:{gpu_id}'))
    
    run_inference(model, video_path, args.output_folder, args.width, args.height, args.batch_size, gpu_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on videos in a folder using a trained model.")
    parser.add_argument("--video_folder", type=str, required=True, help="Path to the folder containing input video files")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save the output video segments")
    parser.add_argument("--width", type=int, default=WIDTH, help="Frame width for model input")
    parser.add_argument("--height", type=int, default=HEIGHT, help="Frame height for model input")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    args = parser.parse_args()

    # Ensure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    # Get list of video files in the input folder
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']  # Add or remove extensions as needed
    video_files = [f for f in os.listdir(args.video_folder) if os.path.splitext(f)[1].lower() in video_extensions]

    if not video_files:
        print(f"No video files found in {args.video_folder}")
    else:
        print(f"Found {len(video_files)} video files.")

        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")

        if num_gpus == 0:
            print("No GPUs available. Running on CPU.")
            num_gpus = 1  # Use CPU

        # Create a pool of workers
        pool = multiprocessing.Pool(processes=3*num_gpus)

        # Process videos in parallel
        for i, video_file in enumerate(video_files):
            gpu_id = i % num_gpus
            pool.apply_async(process_video, args=(video_file, args, gpu_id))
            print(pool)

        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()

    print("All videos processed.")