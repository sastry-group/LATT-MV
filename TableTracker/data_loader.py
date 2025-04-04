import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import cv2

class TableTennisDataset(Dataset):
    def __init__(self, data_root, transform=None):
        """
        Args:
            data_root (string): Root directory of the dataset.
                                Should contain folders like match1, match2, etc.
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
        """
        self.data_root = data_root
        self.transform = transform
        self.samples = []

        # Collect all frame and label paths
        for match_folder in os.listdir(self.data_root):
            match_path = os.path.join(self.data_root, match_folder)
            if os.path.isdir(match_path):
                match_number = match_folder.replace('match', '')
                for item in os.listdir(match_path):
                    item_path = os.path.join(match_path, item)
                    if os.path.isdir(item_path) and item.endswith('_frames'):
                        # Extract rally number
                        rally_number = item.replace(f'match{match_number}_', '').replace('_frames', '')
                        frame_folder = item_path
                        label_file = os.path.join(match_path, f'match{match_number}_{rally_number}_points.csv')
                        if os.path.exists(label_file):
                            # Collect frame paths
                            frame_files = sorted(os.listdir(frame_folder), key=lambda x: int(x.split('_')[1].split('.')[0]))
                            for frame_file in frame_files:
                                if frame_file.endswith('.npy'):
                                    frame_path = os.path.join(frame_folder, frame_file)
                                    frame_number = int(frame_file.split('_')[1].split('.')[0])
                                    self.samples.append({
                                        'frame_path': frame_path,
                                        'label_path': label_file,
                                        'frame_number': frame_number
                                    })
                        else:
                            print(f"Warning: Label file not found for {item_path}")
        
        # Total number of frames
        self.total_frames = len(self.samples)
    
    def __len__(self):
        return self.total_frames
    
    def __getitem__(self, idx):
        # Get sample information
        sample_info = self.samples[idx]
        frame_path = sample_info['frame_path']
        label_path = sample_info['label_path']
        frame_number = sample_info['frame_number']

        # Load frame
        frame = np.load(frame_path)  # Shape: (H, W, C) or (H, W)
        
        # Convert to grayscale if necessary
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Assuming frames are in BGR format
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame  # Already grayscale
        
        # Load labels
        df = pd.read_csv(label_path)
        
        # Create label mask
        label_mask = np.zeros_like(frame_gray, dtype=np.uint8)
        
        # Get points for the current frame
        frame_points = df[df['frame'] == frame_number]
        for _, row in frame_points.iterrows():
            point_str = row['point']
            x_str, y_str = point_str.split(',')
            x, y = int(float(x_str)), int(float(y_str))
            # Create a 4x4 grid around the point
            x_min = max(0, x - 2)
            x_max = min(label_mask.shape[1] - 1, x + 2)
            y_min = max(0, y - 2)
            y_max = min(label_mask.shape[0] - 1, y + 2)
            label_mask[y_min:y_max+1, x_min:x_max+1] = 1  # Set to 1 in label mask

        # Resize frame_gray to (256, 128) (width x height)
        frame_gray = cv2.resize(frame_gray, (256, 128), interpolation=cv2.INTER_AREA)

        # Resize label_mask to (256, 128) using nearest neighbor to avoid interpolation
        label_mask = cv2.resize(label_mask, (256, 128), interpolation=cv2.INTER_NEAREST)

        # Optionally apply transformations
        if self.transform:
            # Combine frame and label into a single sample
            sample = {'image': frame_gray, 'label': label_mask}
            sample = self.transform(sample)
            frame_gray = sample['image']
            label_mask = sample['label']
        else:
            # Convert to tensors
            frame_gray = torch.from_numpy(frame_gray).float().unsqueeze(0)  # Shape: (1, H, W)
            label_mask = torch.from_numpy(label_mask).float().unsqueeze(0)  # Shape: (1, H, W)
        
        return frame_gray, label_mask
    
if __name__ == "__main__":
    # Define the dataset
    dataset = TableTennisDataset(data_root='data/')

    # Get a sample
    frame_gray, label_mask = dataset[0]
    frame_gray_np = frame_gray.squeeze(0).numpy()
    label_mask_np = label_mask.squeeze(0).numpy()

    # Display the image and label
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(frame_gray_np, cmap='gray')
    plt.title('Grayscale Frame')

    plt.subplot(1, 2, 2)
    plt.imshow(label_mask_np, cmap='gray')
    plt.title('Label Mask')

    plt.show()
