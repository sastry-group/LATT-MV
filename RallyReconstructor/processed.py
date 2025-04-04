import numpy as np
import matplotlib.pyplot as plt
import os

from utils.general import *
from utils.render import render
from utils.table import load_table_data, transform 
from utils.camera import calibrate_camera_from_bounds
from utils.ball import load_ball_data, process_ball_data, interpolate_parabolic_trajectory, adjust_trajectory
from utils.optimize_ball import reconstruct_ball_trajectories
from utils.pose import load_pose_data, get_player_ids, get_player_data, get_player_rs, get_rotation_matrix
# from utils.audio import create_audio_file, get_times

class ProcessedVideo:
    """
    A class for processing and analyzing a ping pong video.
    """

    def __init__(self, video_path, video_dir, verbose=False, rescale_factors=None):
        """
        Initialize the ProcessedVideo object.

        Args:
            video_path (str): Path to the video file.
            video_dir (str): Directory containing video data.
            verbose (bool, optional): Whether to print detailed information. Defaults to False.
        """
        self.path = video_path
        self.name = os.path.splitext(os.path.basename(video_path))[0]
        self.dir_name = os.path.dirname(video_path)
        self.frames, self.fps, self.frame_size = load_frames(video_path)
        self.num_frames = len(self.frames)
        self.rescale_factors = rescale_factors
        
        # Table Tracking
        self.homs_table, self.homs, self.table_bounds, self.base_bounds = load_table_data(self.name, video_dir, self.frames, verbose=verbose)  # homography matrices
        self.camera_matrix, self.cam_rmats, self.cam_tvecs = calibrate_camera_from_bounds(self.table_bounds, self.base_bounds, self.frame_size)

        # Paddle Tracking
        self.paddle_positions = load_paddle_data(self.name, video_dir)
        
        # Human Pose Estimation
        self.pose_metadata, self.pose_keypoints_2d, self.pose_keypoints_3d = load_pose_data(self.name, video_dir)
        self.pose_rs = self.frames[0].shape[0] / self.pose_metadata[0]["frame_size"][1]  # pose rescale factor
        self.players = get_player_ids(self.pose_metadata, self.paddle_positions, self.pose_keypoints_2d, self.pose_rs)
        self.player1_keypoints_2d, self.player1_keypoints_3d = get_player_data(self.players[0], self.pose_metadata, self.pose_keypoints_2d, self.pose_keypoints_3d, self.pose_rs, self.num_frames)
        self.player2_keypoints_2d, self.player2_keypoints_3d = get_player_data(self.players[1], self.pose_metadata, self.pose_keypoints_2d, self.pose_keypoints_3d, self.pose_rs, self.num_frames)        
        self.set_players()
        
        # Ball Tracking
        self.ball_positions_2d = load_ball_data(self.name, video_dir)[:self.num_frames]
        self.bounces, self.player1_hits, self.player2_hits = process_ball_data(
            self.ball_positions_2d,  
            self.player1_keypoints_2d[:, self.player1_hand, :], 
            self.player2_keypoints_2d[:, self.player2_hand, :], 
            self.homs_table, 
            verbose=verbose
        )
        self.set_ball()
        reconstruct_ball_trajectories(self)
        
        self.num_frames_usable = last_none_start(self.ball)
        if self.num_frames_usable == -1: 
            self.num_frames_usable = self.num_frames
    
    def set_players(self):
        """
        Process pose data for both players, including feet positions and rotation matrices.
        """
        
        # Some basic processing        
        self.player1_feet_2d = self.player1_keypoints_2d[:, [20, 23]]
        self.player2_feet_2d = self.player2_keypoints_2d[:, [20, 23]]   
        self.player1_feet_3d = self.player1_keypoints_3d[:, [20, 23]]
        self.player2_feet_3d = self.player2_keypoints_3d[:, [20, 23]] 
        self.player1_rs_3d = get_player_rs(self.player1_feet_2d, self.player1_feet_3d, self.homs, self.homs_table)
        self.player2_rs_3d = get_player_rs(self.player2_feet_2d, self.player2_feet_3d, self.homs, self.homs_table)
        if self.rescale_factors is not None:
            if self.player1_rs_3d < self.player2_rs_3d:
                self.player1_rs_3d, self.player2_rs_3d = min(self.rescale_factors), max(self.rescale_factors)
            else:
                self.player1_rs_3d, self.player2_rs_3d = max(self.rescale_factors), min(self.rescale_factors)
        self.player1_hands_2d = self.player1_keypoints_2d[:, [4, 7]]
        self.player2_hands_2d = self.player2_keypoints_2d[:, [4, 7]]
        
        # Transform the players feet positions based on the homography matrix
        for i in range(self.num_frames):
            H = self.homs[i]
            self.player1_feet_2d[i] = transform(self.player1_feet_2d[i], H)
            self.player1_feet_2d[i][:, 1] *= -1
            self.player2_feet_2d[i] = transform(self.player2_feet_2d[i], H)
            self.player2_feet_2d[i][:, 1] *= -1
        
        # Compute each players rotation matrix
        self.player1_R = self.get_rotation_matrices(1)
        self.player2_R = self.get_rotation_matrices(2)
        
        # Finalize each players global position
        self.player1 = self.get_player_positions(1)
        self.player2 = self.get_player_positions(2)
        
        # Set the each players paddle hand
        self.player1_hand = self.get_player_paddle_hand(self.player1_hands_2d)
        self.player2_hand = self.get_player_paddle_hand(self.player2_hands_2d)
    
    def get_rotation_matrices(self, player):
        """
        Calculates rotation matrices for the players' keypoints.

        Returns:
            tuple: Rotation matrices for player 1 and player 2.
        """
        if player == 1:
            player_feet_2d, player_feet_3d = self.player1_feet_2d, self.player1_feet_3d
        if player == 2:
            player_feet_2d, player_feet_3d = self.player2_feet_2d, self.player2_feet_3d
        
        rotation_matrices = []
        for i in range(self.num_frames):
            feet_2d = player_feet_2d[i]
            feet_3d = player_feet_3d[i]
            
            # Determine midpoints
            midpoint_2d = (feet_2d[1] + feet_2d[0]) / 2
            midpoint_3d = (feet_3d[1] + feet_3d[0]) / 2
            
            # Find the rotation matrix
            v = feet_3d[1] - midpoint_3d
            u = feet_2d[1] - midpoint_2d
            u = np.array([u[0], u[1], 0])
            R = get_rotation_matrix(v, u)
            rotation_matrices.append(R)
            
        window_size = 25
        rotation_matrices_smoothed = []
        for i in range(self.num_frames):
            l, r = max(0, i-window_size//2), min(self.num_frames, i+1+window_size//2)
            rotation_matrices_smoothed.append(np.nanmedian(rotation_matrices[l:r], axis=0))
        
        return rotation_matrices_smoothed # np.median(rotation_matrices, axis=0)
                
    def get_player_positions(self, player):
        if player == 1:
            player_feet_2d, player_feet_3d, player_joints, pose_rs_3d, R = self.player1_feet_2d, self.player1_feet_3d, self.player1_keypoints_3d, self.player1_rs_3d, self.player1_R
        if player == 2:
            player_feet_2d, player_feet_3d, player_joints, pose_rs_3d, R = self.player2_feet_2d, self.player2_feet_3d, self.player2_keypoints_3d, self.player2_rs_3d, self.player2_R
        
        player_positions = []
        for i in range(self.num_frames):
            # Load the 2D and 3D feet positions
            feet_2d = player_feet_2d[i]
            feet_3d = player_feet_3d[i]
                        
            # Determine the player's feet midpoint
            midpoint_2d = (feet_2d[1] + feet_2d[0]) / 2
            midpoint_2d = np.array([midpoint_2d[0], midpoint_2d[1], 0])
            midpoint_3d = (feet_3d[1] + feet_3d[0]) / 2
            
            # Rotate the positions         
            positions = player_joints[i]
            positions = positions - midpoint_3d
            positions = positions @ R[i].T
            # P = np.array([[-1, 0, 0], 
            #               [ 0, 0, 1], 
            #               [ 0, 1, 0]])
            # E = np.array([[-1, 0, 0], 
            #               [ 0,-1, 0], 
            #               [ 0, 0, 1]]) 
            # positions = positions @ P @ E @ self.cam_rmats[i] @ P
            positions += np.array([0, 0, -np.min(positions[:, 2])])
            
            # Rescale and translate the positions
            positions = positions * pose_rs_3d + midpoint_2d
            player_positions.append(positions)
        
        return np.array(player_positions)

    def get_player_paddle_hand(self, player_hands_2d):
        hand1_dist, hand2_dist = [], []
        for frame_number, paddles in enumerate(self.paddle_positions):
            hand1, hand2 = player_hands_2d[frame_number]
            for paddle in paddles:
                d1, d2 = dist(hand1, paddle), dist(hand2, paddle)
                if d1 < 50 or d2 < 50:
                    hand1_dist.append(d1)
                    hand2_dist.append(d2)
        m1 = np.median(hand1_dist)
        m2 = np.median(hand2_dist)        
        if abs(m1 - m2) < 0.25:
            raise ValueError("Unable to determine Player's paddle hand.")
        return 4 if m1 < m2 else 7
        
    def set_ball(self):
        ball_positions = []
        for i in range(self.num_frames):
            if i in self.player1_hits:
                ball_pos = self.player1[i, self.player1_hand, :]
            elif i in self.player2_hits:
                ball_pos = self.player2[i, self.player2_hand, :]
            elif i in self.bounces:
                H = self.homs_table[i]
                pos = transform(self.ball_positions_2d[i], H) + np.array([-450, -250])
                ball_pos = np.array([pos[0], -pos[1], 250])
            else:
                ball_pos = None
            ball_positions.append(ball_pos) 
        ball_positions = adjust_trajectory(ball_positions, self.player1_hits, self.player2_hits, self.bounces, alpha=0.0)    
        ball_positions = interpolate_parabolic_trajectory(ball_positions, self.fps)
        self.ball = ball_positions
        
    def save(self, save_path):
        metadata = np.zeros((self.num_frames, 1, 3))
        metadata[0, 0, 0] = self.fps
        metadata[0, 0, 1] = self.num_frames
        metadata[0, 0, 2] = self.num_frames_usable
        metadata[1, 0, 0] = self.player1_rs_3d
        metadata[1, 0, 1] = self.player2_rs_3d
        metadata[1, 0, 2] = self.player1_hand
        metadata[2, 0, 0] = self.player2_hand
        
        player1_hits = np.zeros(self.num_frames)
        player2_hits = np.zeros(self.num_frames)
        bounces      = np.zeros(self.num_frames)
        player1_hits[self.player1_hits] = 1.0
        player2_hits[self.player2_hits] = 1.0
        bounces[self.bounces]           = 1.0
        temp = np.vstack((player1_hits, player2_hits, bounces)).T
        temp = temp[:, np.newaxis, :]
        
        ball = np.array([ b if b is not None else np.array([np.nan, np.nan, np.nan]) for b in self.ball ])
        ball = ball[:, np.newaxis, :]
        
        output = np.concatenate((
                metadata,
                temp,
                self.player1, 
                self.player2, 
                ball
        ), 1)
        np.save(save_path, output)
        
    def __getitem__(self, i):
        return self.player1[i], self.player2[i], self.ball[i]
    
    def __len__(self):
        return self.num_frames_usable
    
    def render(self, fps=None, show_feet=False, show_extended=False):
        if fps is None:
            fps = self.fps
        render(self, fps, show_feet=show_feet, show_extended=show_extended)


class ProcessedVideoLite:
    
    def __init__(self, save_path):
        self.load(save_path)
        
    def load(self, save_path):
        data = np.load(save_path)
        
        # Load metadata
        self.fps = data[0, 0, 0]
        self.num_frames = int(data[0, 0, 1])
        self.num_frames_usable = int(data[0, 0, 2])
        self.player1_rs_3d = data[1, 0, 0]
        self.player2_rs_3d = data[1, 0, 1]
        self.player1_hand  = data[1, 0, 2]
        self.player2_hand  = data[2, 0, 0]
        
        # Load event data
        self.player1_hits = data[:, 1, 0] > 0.5
        self.player2_hits = data[:, 1, 1] > 0.5
        self.bounces = data[:, 1, 2] > 0.5
        
        # Load player and ball data
        self.player1 = data[:,  2:46, :]  # Assuming 44 keypoints for each player
        self.player2 = data[:, 46:90, :]
        self.ball    = data[:, 90, :]
        
        # Replace NaN values with None for ball positions
        self.ball = [b if not np.isnan(b).any() else None for b in self.ball]
        
    def __getitem__(self, i):
        return self.player1[i], self.player2[i], self.ball[i]
        
    def __len__(self):
        return self.num_frames_usable
    
    def render(self, fps=None, show_extended=False):
        if fps is None:
            fps = self.fps
        render(self, fps, show_feet=False, show_extended=show_extended)
        
        
class ProcessedVideoPartial:
    """
    A class for processing and analyzing a ping pong video.
    """

    def __init__(self, video_path, video_dir):
        """
        Initialize the ProcessedVideo object.

        Args:
            video_path (str): Path to the video file.
            video_dir (str): Directory containing video data.
            verbose (bool, optional): Whether to print detailed information. Defaults to False.
        """
        self.path = video_path
        self.name = os.path.splitext(os.path.basename(video_path))[0]
        self.dir_name = os.path.dirname(video_path)
        self.frames, self.fps, self.frame_size = load_frames(video_path)
        self.num_frames = len(self.frames)
        
        # Table Tracking
        self.homs_table, self.homs, self.table_bounds, self.base_bounds = load_table_data(self.name, video_dir, self.frames)  # homography matrices
        
        # Paddle Tracking
        self.paddle_positions = load_paddle_data(self.name, video_dir)
        
        # Human Pose Estimation
        self.pose_metadata, self.pose_keypoints_2d, self.pose_keypoints_3d = load_pose_data(self.name, video_dir)
        self.pose_rs = self.frames[0].shape[0] / self.pose_metadata[0]["frame_size"][1]  # pose rescale factor
        self.players = get_player_ids(self.pose_metadata, self.paddle_positions, self.pose_keypoints_2d, self.pose_rs)  
        self.player1_keypoints_2d, self.player1_keypoints_3d = get_player_data(self.players[0], self.pose_metadata, self.pose_keypoints_2d, self.pose_keypoints_3d, self.pose_rs, self.num_frames)
        self.player2_keypoints_2d, self.player2_keypoints_3d = get_player_data(self.players[1], self.pose_metadata, self.pose_keypoints_2d, self.pose_keypoints_3d, self.pose_rs, self.num_frames)        
        self.player1_feet_2d = self.player1_keypoints_2d[:, [20, 23]]
        self.player2_feet_2d = self.player2_keypoints_2d[:, [20, 23]]   
        self.player1_feet_3d = self.player1_keypoints_3d[:, [20, 23]]
        self.player2_feet_3d = self.player2_keypoints_3d[:, [20, 23]] 
        self.player1_rs_3d = get_player_rs(self.player1_feet_2d, self.player1_feet_3d, self.homs, self.homs_table)
        self.player2_rs_3d = get_player_rs(self.player2_feet_2d, self.player2_feet_3d, self.homs, self.homs_table)
 
"""
# Audio Analysis
self.audio_path = f"{self.dir_name}/{self.name}.wav"
if not os.path.exists(self.audio_path):
    create_audio_file(self.path, self.audio_path)
self.important_frames = get_times(self.audio_path) * self.fps
"""
