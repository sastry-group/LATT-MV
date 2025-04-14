import os
import random
import csv
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.ndimage import rotate, zoom

from constants import *

class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, n_samples, frame_width, frame_height, augment_prob=0.25):
        self.root_dir = root_dir
        self.n_samples = n_samples # number of sample per video
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.augment_prob = augment_prob
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        video_dir = os.path.join(self.root_dir, 'video')
        csv_dir = os.path.join(self.root_dir, 'csv')

        for video_file in os.listdir(video_dir):
            video_path = os.path.join(video_dir, video_file)
            csv_path = os.path.join(csv_dir, video_file.replace('.mp4', '.csv'))

            if not os.path.exists(csv_path):
                continue

            # Read all frames from the video
            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert to grayscale and resize
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                # Normalize to [0, 1] range
                frame = frame.astype(float) / 255.0
                frames.append(frame)
            cap.release()
            total_frames = len(frames)

            # Read CSV and filter out frames beyond video length
            df = pd.read_csv(csv_path)
            df = df[df['Frame'] < total_frames]

            category_1_frames = df[df['InSegment'] == 1]['Frame'].tolist()
            category_2_frames = df[df['InSegment'] == 0]['Frame'].tolist()

            n_per_category = self.n_samples // 2
            category_1_samples = random.sample(category_1_frames, min(n_per_category, len(category_1_frames)))
            category_2_samples = random.sample(category_2_frames, min(n_per_category, len(category_2_frames)))

            for frame in category_1_samples + category_2_samples:
                samples.append((frames[frame], df.loc[df['Frame'] == frame, 'InSegment'].iloc[0]))

        return samples

    def __len__(self):
        return len(self.samples)

    def _augment_frame(self, frame):
        # Rotate
        angle = random.uniform(-20, 20)
        frame = rotate(frame, angle, reshape=False)

        # Zoom
        zoom_factor = random.uniform(1.0, 1.2)  # 1.0 to 1.2 represents up to 20% zoom
        zoomed_frame = zoom(frame, zoom_factor)

        # Crop to original size
        start_y = (zoomed_frame.shape[0] - frame.shape[0]) // 2
        start_x = (zoomed_frame.shape[1] - frame.shape[1]) // 2
        frame = zoomed_frame[start_y:start_y+frame.shape[0], start_x:start_x+frame.shape[1]]

        return frame

    def __getitem__(self, idx):
        frame, label = self.samples[idx]
        
        # Apply augmentations with probability self.augment_prob
        if random.random() < self.augment_prob:
            frame = self._augment_frame(frame)

        # Convert to PyTorch tensor
        frame_tensor = torch.from_numpy(frame).float().unsqueeze(0)  # Add channel dimension
        label_tensor = torch.tensor(label, dtype=torch.long)

        return frame_tensor, label_tensor
    
if __name__ == "__main__":
    dataset = VideoFrameDataset(root_dir='train', n_samples=1000, frame_width=WIDTH, frame_height=HEIGHT)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)