import os
import torch
import numpy as np
import itertools
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from . import youtube_loader
# import youtube_loader

PLAYER_MASK_INDICES = np.array([30, 31, 32, 33, 34, 35, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74])

class LabDataset(Dataset):
    
    def __init__(self, root_dir, constant_fps=False):
        self.root_dir = root_dir  
        self.constant_fps = constant_fps  
        self._load_data()
        self.lab_mask = np.ones(self.data[0].shape[-1])
        if "p1" in youtube_loader.FORMAT:
            self.lab_mask[PLAYER_MASK_INDICES+youtube_loader.FORMAT_RANGES["p1"][0]] = 0.0
        if "p2" in youtube_loader.FORMAT:
            self.lab_mask[PLAYER_MASK_INDICES+youtube_loader.FORMAT_RANGES["p2"][0]] = 0.0
        
    def _load_data(self):
        self.paths = []
        for file in os.listdir(self.root_dir):
            if file.endswith('.npy'):
                self.paths.append(os.path.join(self.root_dir, file))
        
        self.data = []
        self.masks = []
        self.hit_times = []
        for path, mirrored in itertools.product(self.paths, [False, True]):
            segment, segment_mask, hitter, hit_time = load_data(path, mirrored=mirrored)
            if hitter == 1:
                self.data.append(segment)
                self.masks.append(segment_mask)
                self.hit_times.append(hit_time)
            
        # Compute statistics for normalization.
        data_concated = np.concatenate(self.data, 0)
        self.mean, self.std = np.mean(data_concated, 0), np.std(data_concated, 0)
        self.token_dim = self.data[0].shape[-1]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        mask = self.masks[index] * self.lab_mask
        data = (data - self.mean) / (self.std + 1e-8)
        data = data + np.random.randn(*data.shape) / 100
        if self.constant_fps:
            freq = 3
        else:
            freq = np.random.choice([2, 3, 4, 5])
        data[:, youtube_loader.FORMAT_RANGES["fps"][0]] = (100.0/freq) / 100.0
        data_and_mask = np.concatenate((data, mask), -1)
        data_and_mask = data_and_mask[::freq]
        
        return torch.from_numpy(data_and_mask).float()
    
    def return_to_raw(self, data, fps=100):
        data = data * (self.std + 1e-8) + self.mean
        data_dict = {}
        for key in youtube_loader.FORMAT_LENGTHS:
            if key in youtube_loader.FORMAT:
                data_dict[key] = data[:, youtube_loader.FORMAT_RANGES[key][0]:youtube_loader.FORMAT_RANGES[key][1]].numpy()
            else:
                data_dict[key] = np.zeros((data.shape[0], youtube_loader.FORMAT_LENGTHS[key]))
            
        data_dict["p1"] = data_dict["p1"].reshape(data.shape[0], -1, 3)
        data_dict["p2"] = data_dict["p2"].reshape(data.shape[0], -1, 3)
        data_dict["b"]  = data_dict["b"].reshape(data.shape[0], -1, 3)
        data_dict["p1"] = data_dict["p1"] + data_dict["p1_root"].reshape(data.shape[0], -1, 3)
        data_dict["p2"] = data_dict["p2"] + data_dict["p2_root"].reshape(data.shape[0], -1, 3)
        data_dict["p1_pad"][:, :3] = (data_dict["p1_pad"][:, :3] + data_dict["p1_root"]) / 0.003048
        data_dict["p2_pad"][:, :3] = (data_dict["p2_pad"][:, :3] + data_dict["p2_root"]) / 0.003048
        
        raw_data = np.concatenate((np.zeros((data.shape[0], 2, 3)), data_dict["p1"], data_dict["p2"], data_dict["b"]), axis=1) / 0.003048
        raw_data[0, 0, 0] = fps
        raw_data[0, 0, 1] = raw_data[0, 0, 2] = len(raw_data)
        return raw_data, data_dict["p1_pad"], data_dict["p2_pad"]

def swap_xy1(seq):
    seq_copy = seq.copy()
    seq_copy[:, 0] = seq[:, 1]
    seq_copy[:, 1] = seq[:, 0]
    return seq_copy

def swap_xy2(seq):
    seq_copy = seq.copy()
    seq_copy[:, :, 0] = seq[:, :, 1]
    seq_copy[:, :, 1] = seq[:, :, 0]
    return seq_copy

def load_hand(p, pad):
    d1 = np.median(np.linalg.norm(p[:, 4] - pad[:, :3]))
    d2 = np.median(np.linalg.norm(p[:, 7] - pad[:, :3]))
    if abs(d1 - d2) < 0.1:
        return 0
    if d1 < d2:
        return 4
    else:
        return 7

def load_data(path, mirrored=False, mask_non_hitter=False):
    """Loads a reconstructed table tennis rally into a dictionary."""
    raw_data = np.load(path)
    
    # Add missing frames.
    timestamps = raw_data[:, 167] - raw_data[0, 167]
    sequence_length = int(max(timestamps)+1)
    data = np.zeros((sequence_length, raw_data.shape[1]))
    for i in range(len(timestamps)):
        data[int(timestamps[i])] = raw_data[i]
    token_masks = (data != 0.0)
    
    # Form the data dictionary.
    root_joint = youtube_loader.ROOT_JOINT
    data = {
        "sequence_length": sequence_length,
        "p1_hits": data[:, 168],
        "p2_hits": data[:, 169],
        "p1":      data[:,    0:75].reshape(data.shape[0], -1, 3) / 100,  
        "p2":      data[:,  75:150].reshape(data.shape[0], -1, 3) / 100,
        "b":       data[:, 164:167].reshape(data.shape[0], -1, 3) / 100,
        "p1_pad":  data[:, 150:157],
        "p2_pad":  data[:, 157:164],
        "fps":     np.ones((sequence_length, 1))
    }
    
    data_mask = {
        "sequence_length": sequence_length,
        "p1":      token_masks[:,    0:75].reshape(token_masks.shape[0], -1, 3),  
        "p2":      token_masks[:,  75:150].reshape(token_masks.shape[0], -1, 3),
        "b":       token_masks[:, 164:167].reshape(token_masks.shape[0], -1, 3),
        "p1_pad":  token_masks[:, 150:157],
        "p2_pad":  token_masks[:, 157:164],
        "fps":     np.ones((sequence_length, 1))
    }
    
    # Swap xy positions.
    data["b"]  = swap_xy2(data["b"])
    data["p1"] = swap_xy2(data["p1"])
    data["p2"] = swap_xy2(data["p2"])
    data["p1_pad"][:, :3] = swap_xy1(data["p1_pad"][:, :3]) / 100
    data["p2_pad"][:, :3] = swap_xy1(data["p2_pad"][:, :3]) / 100
    data["p1_pad"][:, 4:] = swap_xy1(data["p1_pad"][:, 4:])
    data["p2_pad"][:, 4:] = swap_xy1(data["p2_pad"][:, 4:])
    
    # Adjust z positions.
    data["b"][:, :, 2]   +=  0.762
    data["p1"][:, :, 2]  +=  0.762
    data["p2"][:, :, 2]  +=  0.762
    data["p1_pad"][:, 2] +=  0.762
    data["p2_pad"][:, 2] +=  0.762
    
    # Mirroring
    if mirrored:
        data["b"][:, :, :2] *= -1
        data["p1"][:, :, :2] *= -1
        data["p2"][:, :, :2] *= -1
        data["p1_pad"][:, :2] *= -1
        data["p2_pad"][:, :2] *= -1
        data["p1_pad"][:, 3:] = data["p1_pad"][:, 3:][:, ::-1]
        data["p2_pad"][:, 3:] = data["p2_pad"][:, 3:][:, ::-1]
        data["p1_pad"][:, 3:5] *= -1
        data["p2_pad"][:, 3:5] *= -1
        data["p1"], data["p2"] = data["p2"], data["p1"]
        data["p1_hits"], data["p2_hits"] = data["p2_hits"], data["p1_hits"]
        data["p1_pad"],  data["p2_pad"]  = data["p2_pad"],  data["p1_pad"]
        data_mask["p1"], data_mask["p2"] = data_mask["p2"], data_mask["p1"]
        data_mask["p1_pad"],  data_mask["p2_pad"]  = data_mask["p2_pad"],  data_mask["p1_pad"]
        
    # Center joint and paddle positions around player roots.
    data_mask["p1_root"] = data_mask["p1"][:, root_joint:root_joint+1]
    data_mask["p2_root"] = data_mask["p2"][:, root_joint:root_joint+1]
    data["p1_root"] = data["p1"][:, root_joint:root_joint+1]
    data["p2_root"] = data["p2"][:, root_joint:root_joint+1]
    data["p1"] = data["p1"] - data["p1_root"]
    data["p2"] = data["p2"] - data["p2_root"]
    data["p1_pad"][:, :3] = data["p1_pad"][:, :3] - data["p1_root"][:, 0, :]
    data["p2_pad"][:, :3] = data["p2_pad"][:, :3] - data["p2_root"][:, 0, :]
    
    # Paddle hands
    p1_hand = load_hand(data["p1"], data["p1_pad"])
    p2_hand = load_hand(data["p2"], data["p2_pad"]) 
    if not p1_hand:
        data["p1_pad_hand"] = np.zeros((sequence_length, 3))
        data_mask["p1_pad_hand"] = np.zeros((sequence_length, 3))
    else:
        data["p1_pad_hand"] = data["p1"][:, p1_hand]
        data_mask["p1_pad_hand"] = data_mask["p1"][:, p1_hand]
    if not p2_hand:
        data["p2_pad_hand"] = np.zeros((sequence_length, 3))
        data_mask["p2_pad_hand"] = np.zeros((sequence_length, 3))
    else:
        data["p2_pad_hand"] = data["p2"][:, p2_hand]
        data_mask["p2_pad_hand"] = data_mask["p2"][:, p2_hand]
        
    if 1 not in data["p1_hits"][[0, -1]]:
        data["hitter"] = 1
        data["hit_time"] = np.where(data["p1_hits"] == 1)[0].flatten()[0]
        if mask_non_hitter: 
            data_mask["p2"] = np.zeros(data["p2"].shape)
            data_mask["p2_pad_hand"] = np.zeros(data["p2_pad_hand"].shape)
            data_mask["p2_pad"] = np.zeros(data["p2_pad"].shape)
    else:
        data["hitter"] = 2
        data["hit_time"] = np.where(data["p2_hits"] == 1)[0].flatten()[0]
        if mask_non_hitter: 
            data_mask["p1"] = np.zeros(data["p1"].shape)
            data_mask["p1_pad_hand"] = np.zeros(data["p1_pad_hand"].shape)
            data_mask["p1_pad"] = np.zeros(data["p1_pad"].shape)

    return youtube_loader.format_data(data), youtube_loader.format_data(data_mask), data["hitter"], data["hit_time"]

if __name__ == "__main__":
    ds = LabDataset("../data/recons_lab")