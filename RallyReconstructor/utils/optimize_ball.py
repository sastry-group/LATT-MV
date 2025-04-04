import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import pickle
import cv2
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def project_points(points_3d, K, Rs, t_cams):
    """
    Projects 3D points onto the 2D image plane using per-frame camera parameters.
    """
    points_2d = []
    for i in range(len(points_3d)):
        R = Rs[i]
        t_cam = t_cams[i]
        point_3d = points_3d[i] / 0.003048  # Convert units if necessary
        # Transform point to camera coordinate system
        point_cam = R @ point_3d + t_cam
        # Project onto image plane
        point_proj = K @ point_cam
        # Normalize homogeneous coordinates
        u, v = point_proj[:2] / point_proj[2]
        points_2d.append([u, v])
    return np.array(points_2d)

def simulate_trajectory(t, initial_conditions, air_resistance):
    """
    Simulates the ball's trajectory given initial conditions and air resistance.
    """
    x0, y0, z0, xf, yf, zf, tf = initial_conditions
    g = 9.81  # Acceleration due to gravity (m/s^2)
    k = air_resistance  # Air resistance coefficient (1/s)

    # Precompute exponentials
    one_minus_exp_neg_kt = 1 - np.exp(-k * t)
    K = (1 - np.exp(-k * tf)) / k
    
    # Initial velocities
    vx0 = (xf - x0) / K
    vy0 = (yf - y0) / K
    vz0 = ((zf - z0 + g*tf/k) / K) - g/k

    # Positions
    x_t = x0 + (vx0 / k) * one_minus_exp_neg_kt
    y_t = y0 + (vy0 / k) * one_minus_exp_neg_kt
    z_t = z0 + ((vz0 + g / k) / k) * one_minus_exp_neg_kt - (g / k) * t

    positions = np.vstack((x_t, z_t, y_t)).T
    return positions

def residuals(params, t_obs, observed_2d_points, K, Rs, t_cams, known_3d_points):
    """
    Computes the residuals for the optimization.
    """
    x0, y0, z0 = known_3d_points[0]
    xf, yf, zf = known_3d_points[-1]
    initial_conditions = x0, z0, y0, xf, zf, yf, t_obs[-1]
    air_resistance = params[0]

    # Enforce non-negative air resistance
    air_resistance = max(air_resistance, 0)

    # Simulate trajectory
    positions_3d = simulate_trajectory(t_obs, initial_conditions, air_resistance)

    # Project to 2D
    projected_2d = project_points(positions_3d, K, Rs, t_cams)

    # Residuals between observed and projected 2D points
    res_2d = (observed_2d_points - projected_2d).ravel()

    # Combine residuals
    return res_2d

def reconstruct_ball_trajectories(vid):
    """
    Processes the vid object and outputs all the ball trajectories over all its segments.
    """
    # Collect all keypoints
    ball_traj_keypoints = np.concatenate((vid.player1_hits, vid.bounces, vid.player2_hits))
    ball_traj_keypoints.sort()

    # Initialize list to collect trajectories
    all_positions_3d = []

    # Camera intrinsic matrix
    K = vid.camera_matrix
    P = np.array([[-1, 0, 0],
                  [ 0, 0, 1],
                  [ 0, 1, 0]])

    # Loop over segments between keypoints
    for i in range(len(ball_traj_keypoints) - 1):
        t1 = ball_traj_keypoints[i]
        t2 = ball_traj_keypoints[i+1]  # Include endpoint

        # Get observed 2D points and times for this segment
        observed_2d_points = vid.ball_positions_2d[t1:t2+1]
        t_obs = np.arange(0, t2+1-t1) / vid.fps

        # Get known 3D points and their indices
        known_3d_points = np.array([
            vid.ball[t1],  # Starting point
            vid.ball[t2],  # Ending point
        ]) @ P * 0.003048

        # Get per-frame camera parameters
        Rs = [vid.cam_rmats[frame_idx] for frame_idx in range(t1, t2+1)]
        t_cams = [vid.cam_tvecs[frame_idx][:, 0] for frame_idx in range(t1, t2+1)]

        # Initial guess for parameters
        rng = np.arange(0.0001, 10.0, 0.05)
        res = [np.mean(abs(residuals([a], t_obs, observed_2d_points, K, Rs, t_cams, known_3d_points))) for a in rng]
        air_resistance_opt = rng[np.argmin(res)]

        # Simulate optimized trajectory
        x0, y0, z0 = known_3d_points[0]
        xf, yf, zf = known_3d_points[-1]
        initial_conditions_opt = x0, z0, y0, xf, zf, yf, t_obs[-1]
        positions_3d_opt = simulate_trajectory(t_obs, initial_conditions_opt, air_resistance_opt)
            
        # Collect positions (convert back to original units)
        all_positions_3d.append(positions_3d_opt[:-1] @ P / 0.003048)

    # Concatenate all positions into a single array
    all_positions_3d.append(positions_3d_opt[-1:] @ P / 0.003048)
    all_positions_3d = np.concatenate(all_positions_3d, axis=0)
    prev_positions_3d = vid.ball[ball_traj_keypoints[0]:ball_traj_keypoints[-1]+1]
    
    Rs = [vid.cam_rmats[frame_idx] for frame_idx in range(ball_traj_keypoints[0], ball_traj_keypoints[-1]+1)]
    t_cams = [vid.cam_tvecs[frame_idx][:, 0] for frame_idx in range(ball_traj_keypoints[0], ball_traj_keypoints[-1]+1)]
    projected_points = project_points(all_positions_3d @ P * 0.003048, K, Rs, t_cams)
    prev_projected_points = project_points(prev_positions_3d @ P * 0.003048, K, Rs, t_cams)
    
    reprojection_error = np.mean(np.linalg.norm(projected_points - vid.ball_positions_2d[ball_traj_keypoints[0]:ball_traj_keypoints[-1]+1], axis=1))
    prev_reprojection_error = np.mean(np.linalg.norm(prev_projected_points - vid.ball_positions_2d[ball_traj_keypoints[0]:ball_traj_keypoints[-1]+1], axis=1))
    
    # print(f"Ball Trajectory Reprojection Error: {reprojection_error}")
    # print(f"Prev Ball Trajectory Reprojection Error: {prev_reprojection_error}")
    
    vid.ball[ball_traj_keypoints[0]:ball_traj_keypoints[-1]+1] = all_positions_3d


