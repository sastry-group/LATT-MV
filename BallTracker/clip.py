import pandas as pd
import numpy as np
import os
import subprocess
import argparse
from pathlib import Path

def process_ball_data(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, 
                     dtype={'Frame': int, 'Visibility': int, 'X': int, 'Y': int})
    
    # Remove rows where Visibility is 0 (ball is missing)
    df = df[df['Visibility'] != 0].reset_index(drop=True)
    
    # Function to check if all values in a window are the same
    def all_same(x):
        return pd.Series(x == x.iloc[0]).all()
    
    # Create masks for X and Y separately
    window_size = 7  # 3 above + current + 3 below
    x_constant = df['X'].rolling(window=window_size, center=True, min_periods=1).apply(all_same).astype(bool)
    y_constant = df['Y'].rolling(window=window_size, center=True, min_periods=1).apply(all_same).astype(bool)
    
    # Combine masks
    mask = ~(x_constant & y_constant)
    
    # Apply the mask and reset the index
    df = df[mask].reset_index(drop=True)
    
    # Check if frame numbers are consecutive
    frame_diff = df['Frame'].diff()
    if len(df) == 0 or (frame_diff[1:] != 1).any():
        return False, (None, None)
    
    # Reset the Frame column to be like an index
    df['Frame'] = range(len(df))
    
    # Save the processed data to a new CSV file
    df.to_csv(file_path, index=False)
            
    return True, (df['Frame'][0], df['Frame'][len(df)-1])

def process_matches(root_dir, matches_dir):
    for match_folder in os.listdir(root_dir):
        match_path = os.path.join(root_dir, match_folder)
        if os.path.isdir(match_path):
            process_match(match_path, os.path.join(matches_dir, match_folder))

def process_match(csv_folder, video_folder):
    for csv_file in os.listdir(csv_folder):
        if csv_file.endswith('.csv'):
            csv_path = os.path.join(csv_folder, csv_file)
            video_name = csv_file.replace('_ball.csv', '.mp4')
            video_path = os.path.join(video_folder, video_name)
            
            if not os.path.exists(video_path):
                print(f"Warning: No corresponding video found for {csv_file}")
                continue
            
            flg, (a, b) = process_ball_data(csv_path)
            
            if not flg:
                delete_video(video_path)
            else:
                trim_video(video_path, a, b)

def delete_video(video_path):
    os.remove(video_path)
    print(f"Deleted: {video_path}")

def trim_video(video_path, start_frame, end_frame):
    output_path = video_path.replace('.mp4', '_trimmed.mp4')
    
    # Construct ffmpeg command to trim video
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f"select='between(n\,{start_frame}\,{end_frame})'",
        '-vsync', 'vfr',
        '-frame_pts', 'true',
        output_path
    ]
    
    subprocess.run(cmd, check=True)
    
    # Replace original file with trimmed version
    os.remove(video_path)
    os.rename(output_path, video_path)
    
    print(f"Trimmed: {video_path} (frames {start_frame} to {end_frame})")

def main():
    parser = argparse.ArgumentParser(description="Process CSV files and trim or delete corresponding MP4 videos.")
    parser.add_argument("matches_dir", help="Path to the matches directory containing MP4 files")
    parser.add_argument("root_dir", help="Path to the root directory containing CSV files")
    args = parser.parse_args()

    root_dir = args.root_dir
    matches_dir = args.matches_dir

    if not os.path.isdir(root_dir):
        print(f"Error: {root_dir} is not a valid directory")
        return

    if not os.path.isdir(matches_dir):
        print(f"Error: {matches_dir} is not a valid directory")
        return

    process_matches(root_dir, matches_dir)

if __name__ == "__main__":
    main()