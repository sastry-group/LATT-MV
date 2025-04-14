#!/usr/bin/env python3
"""
This script reprojects processed information onto a video and saves the result.
It loads processed video data (from a hard-coded "data" folder) based on a match and rally number,
computes selected projection overlays, renders these onto the video frames, and saves the output as a video.

Command-line Flags:
  --match:         Match number (used for folder naming as "match{n}")
  --rally:         Rally number (used for naming the file as "match{match}_{rally}")
  --projections:   Comma-separated list of projection keys to compute.
                   Available keys: b_orig, b_reconstructed, racket, table,
                   table_reconstructed, players, grid_world.
                   If not provided, no projections are computed.
  --output-file:   Output video file name (default: projection_video.mp4)
  --fps:           Frames per second for the output video. If not provided, uses the original video's FPS.
"""

import pickle
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2

from utils.general import load_frames
from processed import ProcessedVideoLite, ProcessedVideoPartial, ProcessedVideo

# Global resize factor and skeleton definition
PARENT_DIR = "../release_data"
RESIZE = 2
SKELETON = np.array([
    [17, 15],
    [15, 0],
    [18, 16],
    [16, 0],
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [1, 5],
    [5, 6],
    [6, 7],
    [1, 8],
    [8, 9],
    [8, 12],
    [9, 10],
    [10, 11],
    [12, 13],
    [13, 14],
    [11, 24],
    [11, 22],
    [22, 23],
    [14, 21],
    [14, 19],
    [19, 20]
])

def reconstruct(root_dir, match_folder, name, render=False, verbose=False, rescale_factors=None):
    """
    Reconstructs the processed video using the provided folder and file names.
    """
    dirname = f"{root_dir}/{match_folder}/{name}"
    vid = ProcessedVideo(
        dirname + f"/{name}.mp4",
        dirname,
        verbose=verbose,
        rescale_factors=rescale_factors
    )
    if render: 
        vid.render()
    return vid

def augment_rotation(timestep, R, t) :
    """
    Augments the rotation matrix at the given timestep with a rotation around the z-axis.
    """
    angle = -timestep*0.5*np.pi/180.
    c, s = np.cos(angle), np.sin(angle)
    R = R @ np.array([[c, -s, 0],
                    [s,  c, 0],
                    [0,  0, 1]])
    
    
    return R, t

def project_point(p, v, timestep):
    """
    Projects a 3D point 'p' using the camera matrix, rotation, and translation data from video 'v' at timestep.
    """
    P = np.array([[-1, 0, 0],
                  [ 0, 0, 1],
                  [ 0, 1, 0]]) 
    K = v.camera_matrix
    R = v.cam_rmats[timestep]
    t = v.cam_tvecs[timestep][:, 0]
    R,t = augment_rotation(timestep, R @ P, t)
    p_cam = (R @ p) + t
    p_proj = K @ p_cam
    return RESIZE * p_proj[:2] / p_proj[2]

def draw_scale_indicators(frame, position='bottom_right'):
    """
    Draws two lines of lengths 6px and 18px with labels on the frame.
    
    Parameters:
    - frame: The image on which to draw.
    - position: Where to place the scale indicators ('bottom_right', 'bottom_left', 'top_right', 'top_left').
    
    Returns:
    - frame: The image with scale indicators drawn.
    """
    height, width = frame.shape[:2]
    margin = 0  # Margin from the edges
    line_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_color = (255, 255, 255)  # White color for text

    if position == 'bottom_right':
        x_start = width // 2 - margin
        y_start = height // 2 - margin
        x_direction = -1
        y_direction = -1
        text_offset_x = -40
        text_offset_y = -5
    elif position == 'bottom_left':
        x_start = margin
        y_start = height - margin
        x_direction = 1
        y_direction = -1
        text_offset_x = 5
        text_offset_y = -5
    elif position == 'top_right':
        x_start = width - margin
        y_start = margin
        x_direction = -1
        y_direction = 1
        text_offset_x = -40
        text_offset_y = 15
    elif position == 'top_left':
        x_start = margin
        y_start = margin
        x_direction = 1
        y_direction = 1
        text_offset_x = 5
        text_offset_y = 15
    else:
        raise ValueError("Invalid position argument. Choose from 'bottom_right', 'bottom_left', 'top_right', 'top_left'.")

    pt1_6px = (x_start, y_start)
    pt2_6px = (x_start + x_direction * 6, y_start)
    pt1_18px = (x_start, y_start + y_direction * 20)  # Slight offset for separation
    pt2_18px = (x_start + x_direction * 18, y_start + y_direction * 20)

    cv2.line(frame, pt1_6px, pt2_6px, (255, 255, 255), thickness=line_thickness)
    cv2.putText(frame, '6px', (pt1_6px[0] + text_offset_x, pt1_6px[1] + text_offset_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    cv2.line(frame, pt1_18px, pt2_18px, (255, 255, 255), thickness=line_thickness)
    cv2.putText(frame, '18px', (pt1_18px[0] + text_offset_x, pt1_18px[1] + text_offset_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return frame

def render_projections_on_frames(v, frame_projections, streak_length=5):
    """
    Renders the computed projection overlays on each frame.
    """
    rendered_frames = []
    b_orig_trail = []
    b_reconstructed_trail = []
    past_frames = [] 
    alpha = 0.95
    
    frames = v.frames # load_frames("../../human_pose_estimates.mp4")
    frames = np.array([cv2.resize(frame, v.frame_size) for frame in frames])
    
    for t, frame in enumerate(frames):
        frame_with_projections = frame.copy()
        
        if "b_orig" in frame_projections[t]:
            b_orig_trail.append(frame_projections[t]["b_orig"])
            if len(b_orig_trail) > streak_length:
                b_orig_trail.pop(0)
        else:
            b_orig_trail = []

        

        if "grid_world" in frame_projections[t]:
            for p1, p2 in frame_projections[t]["grid_world"]:
                cv2.line(frame_with_projections, tuple(map(int, p1)), tuple(map(int, p2)), (155, 155, 155), 1 * RESIZE)

        if "table" in frame_projections[t]:
            for corner in frame_projections[t]["table"]:
                cv2.circle(frame_with_projections, tuple(map(int, corner)), 4, (255, 0, 0), -1)
        
        

        if "table_reconstructed" in frame_projections[t]:
            points = np.array(frame_projections[t]["table_reconstructed"], dtype=np.int32)
            cv2.polylines(frame_with_projections, [points], isClosed=True, color=(255, 60, 60), thickness=2 * RESIZE)
            overlay = frame_with_projections.copy()
            cv2.fillPoly(overlay, [points], color=(255, 60, 60))

            beta = 0.5
            # Blend the overlay with the original frame
            cv2.addWeighted(overlay, beta, frame_with_projections, 1 - beta, 0, frame_with_projections)

            points = np.array(frame_projections[t]["table_net"], dtype=np.int32)
            # cv2.polylines(frame_with_projections, [points], isClosed=True, color=(255, 60, 60), thickness=2 * RESIZE)
            
            overlay = frame_with_projections.copy()
            cv2.fillPoly(overlay, [points], color=(255, 60, 60))

            beta = 0.5
            # Blend the overlay with the original frame
            cv2.fillPoly(overlay, [points], color=(255, 255, 255))
            cv2.addWeighted(overlay, beta, frame_with_projections, 1 - beta, 0, frame_with_projections)


            for p1, p2 in frame_projections[t]["table_legs"]:
                cv2.line(frame_with_projections, tuple(map(int, p1)), tuple(map(int, p2)), (255, 60, 60), 2 * RESIZE)   

        if "b_reconstructed" in frame_projections[t]:
            b_reconstructed_trail.append(frame_projections[t]["b_reconstructed"])
            if len(b_reconstructed_trail) > streak_length:
                b_reconstructed_trail.pop(0)
        else:
            b_reconstructed_trail = []

        if "players" in frame_projections[t]:
            player1 = frame_projections[t]["players"][:25]
            player2 = frame_projections[t]["players"][44:25+44]
            for pt1, pt2 in SKELETON:
                cv2.line(frame_with_projections, tuple(map(int, player2[pt1])), tuple(map(int, player2[pt2])), (224, 48, 224), 2 * RESIZE)
                cv2.line(frame_with_projections, tuple(map(int, player1[pt1])), tuple(map(int, player1[pt2])), (0, 0, 255), 2 * RESIZE)
            for pt in player2:
                cv2.circle(frame_with_projections, tuple(map(int, pt)), 2 * RESIZE, (224, 48, 224), -1)
            for pt in player1:
                cv2.circle(frame_with_projections, tuple(map(int, pt)), 2 * RESIZE, (0, 0, 255), -1)
        
        if len(b_orig_trail) > 1:
            points = np.array(b_orig_trail, dtype=np.int32)
            cv2.polylines(frame_with_projections, [points], False, (0, 0, 255), thickness=1 * RESIZE)

        if len(b_reconstructed_trail) > 1:
            points = np.array(b_reconstructed_trail, dtype=np.int32)
            cv2.polylines(frame_with_projections, [points], False, (255, 255, 255), thickness=1 * RESIZE)
            
        if "b_orig" in frame_projections[t]:
            b_orig_current = frame_projections[t]["b_orig"]
            cv2.circle(frame_with_projections, tuple(map(int, b_orig_current)), 4 * RESIZE, (0, 0, 255), -1)

        if "b_reconstructed" in frame_projections[t]:
            b_reconstructed_current = frame_projections[t]["b_reconstructed"]
            cv2.circle(frame_with_projections, tuple(map(int, b_reconstructed_current)), 2 * RESIZE, (255, 255, 255), -1)

        if "racket" in frame_projections[t]:
            rackets = frame_projections[t]["racket"]
            for racket in rackets:
                pt1 = tuple((racket - np.array([7, 7])).astype(int))
                pt2 = tuple((racket + np.array([7, 7])).astype(int))
                cv2.rectangle(frame_with_projections, pt1, pt2, (255, 0, 0), 5)
        
        if past_frames:
            for past_frame in past_frames:
                frame_with_projections = cv2.addWeighted(frame_with_projections, alpha, past_frame, 1 - alpha, 0)

        past_frames.append(frame_with_projections.copy())
        if len(past_frames) > streak_length:
            past_frames.pop(0)

        rendered_frames.append(frame_with_projections)

    return rendered_frames

def save_frames_as_video(frames, output_file="output.mp4", fps=30):
    """
    Saves a list of frames as a video file.
    """
    height, width, layers = frames[0].shape
    video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame in frames:
        video_writer.write(frame)
    
    video_writer.release()

def main():
    parser = argparse.ArgumentParser(
        description="Reprojects processed video information onto a video and saves the output.\n\n"
                    "Flags:\n"
                    "  --match:       Match number (used in folder naming as 'match{n}')\n"
                    "  --rally:       Rally number (used in naming file as 'match{match}_{rally}')\n"
                    "  --projections: Comma-separated list of projection keys to compute.\n"
                    "                 Available keys: b_orig (ground truth ball position), b_reconstructed (reconstructed ball position), racket, table, players, grid_world.\n"
                    "                 If not provided, no projections are computed.\n"
                    "  --output-file: Output video file name (default: projection_video.mp4)\n"
                    "  --fps:         Frames per second for the output video. If not provided, uses the original video's FPS.\n",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--match', type=int, default=501,
                        help="Match number (used in folder naming as 'match{n}')")
    parser.add_argument('--rally', type=int, default=35,
                        help="Rally number (used in naming file as 'match{match}_{rally}')")
    parser.add_argument('--projections', type=str, default="b_reconstructed,table_reconstructed,players,grid_world",
                        help="Comma-separated list of projection keys to compute. Available keys: b_orig, b_reconstructed, racket, table, table_reconstructed, players, grid_world. If not provided, no projections are computed.")
    parser.add_argument('--output-file', type=str, default="lengthy_matches/match501_35.mp4",
                        help="Output video file name (default: projection_video.mp4)")
    parser.add_argument('--fps', type=float, default=None,
                        help="Frames per second for the output video. If not provided, uses the original video's FPS.")

    args = parser.parse_args()

    # Process the projections flag
    if args.projections:
        to_proj = [proj.strip() for proj in args.projections.split(',') if proj.strip()]
    else:
        to_proj = []

    # a corresponds to the match number, b to the rally number
    a = args.match
    b = args.rally
    # print("Start")
    # Reconstruct the video from processed data (root directory is hard-coded as "data")
    v = reconstruct(PARENT_DIR, f"match{a}", f"match{a}_{b}", verbose=False, render=False)
    # print("Here 0")

    T = range(len(v.frames))
    frame_projections = [dict() for _ in T]

    # Setup grid lines, table lines, and table legs for projections
    grid_lines = (
        [[[ -15, n, 0], [ 15, n, 0]] for n in range(-15, 16, 1)] +
        [[[ n, -15, 0], [ n,  15, 0]] for n in range(-15, 16, 1)]
    )
    grid_lines = np.array(grid_lines) * 100

    table_lines = [
        [-450, -250, 250],
        [-450,  250, 250],
        [ 450,  250, 250],
        [ 450, -250, 250]
    ]

    H = 50
    net_lines = [
        [0, 300, 300],
        [0, 300, 300 + H],
        [0, -300, 300 + H],
        [0, -300, 300],
    ]

    table_legs = [
        [[-450, -250, 250], [-450, -250, 0]],
        [[-450,  250, 250], [-450,  250, 0]],
        [[ 450,  250, 250], [ 450,  250, 0]],
        [[ 450, -250, 250], [ 450, -250, 0]],
    ]
    ball_positions = []
    player_joints = []
    for t in T:
        if "b_orig" in to_proj:
            frame_projections[t]["b_orig"] = v.ball_positions_2d[t] * RESIZE
        if "b_reconstructed" in to_proj:
            b_val = v.ball[t]
            # print(b_val)
            if b_val is not None:
                ball_positions.append(b_val*0.003048)
            else :
                ball_positions.append(np.array([0,0,0]))
            if b_val is not None:
                frame_projections[t]["b_reconstructed"] = project_point(b_val, v, t)
        if "racket" in to_proj:
            frame_projections[t]["racket"] = v.paddle_positions[t] * RESIZE
        if "table" in to_proj:
            frame_projections[t]["table"] = v.table_bounds[t] * RESIZE
        if "table_reconstructed" in to_proj:
            frame_projections[t]["table_reconstructed"] = [tuple(map(int, project_point(p, v, t))) for p in table_lines]
            frame_projections[t]["table_legs"] = [[project_point(p1, v, t), project_point(p2, v, t)] for p1, p2 in table_legs]
            frame_projections[t]["table_net"] = [tuple(map(int, project_point(p, v, t))) for p in net_lines]
        if "players" in to_proj:
            joint_positions = np.concatenate((v.player1[t], v.player2[t]), 0)
            player_joints.append(joint_positions*0.003048)
            frame_projections[t]["players"] = [project_point(j, v, t) for j in joint_positions]
        if "grid_world" in to_proj:
            frame_projections[t]["grid_world"] = [[project_point(p1, v, t), project_point(p2, v, t)] for p1, p2 in grid_lines]

    ball_positions = np.array(ball_positions)
    player_joints = np.array(player_joints)
    # Save as npy file
    # np.save(f"saved_data/ball_positions_{a}_{b}.npy", ball_positions)
    # np.save(f"saved_data/player_joints_{a}_{b}.npy", player_joints)
    # print("Here 1")
    # Prepare frames for rendering by resizing
    # v.frames *= 0
    # print(len(v.frames))
    # if len(v.frames) <= 100:
    #     print("Len:",len(v.frames))
    #     exit(0)
    print("Saved ", len(v.frames))
    new_frames = [cv2.resize(v.frames[i], (v.frame_size[0]*RESIZE, v.frame_size[1]*RESIZE)) for i in range(len(v.frames))]
    v.frames = np.array(new_frames, dtype=np.uint8)*0
    v.frame_size = (v.frame_size[0]*RESIZE, v.frame_size[1]*RESIZE)

    # Render projections on each frame
    rendered_frames = render_projections_on_frames(v, frame_projections)
    # print("Here 2")

    # Save frames as video (use provided fps if available; otherwise, default to v.fps)
    fps = args.fps if args.fps is not None else v.fps
    save_frames_as_video(rendered_frames, output_file=args.output_file, fps=fps)
    # print("Here 3")

if __name__ == "__main__":
    main()
