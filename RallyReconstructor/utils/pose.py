import numpy as np
import json
import matplotlib.pyplot as plt

from .table import transform
from .general import dist

def load_pose_data(video_name, pose_dir):
    metadata_path     = f"{pose_dir}/{video_name}_metadata.json"
    keypoints_2d_path = f"{pose_dir}/{video_name}_keypoints_2d.npy"
    keypoints_3d_path = f"{pose_dir}/{video_name}_keypoints_3d.npy"
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    keypoints_2d = np.load(keypoints_2d_path)
    keypoints_3d = np.load(keypoints_3d_path)
    keypoints_3d[..., [1, 2]] = keypoints_3d[..., [2, 1]]
    keypoints_3d[..., [2]] *= -1
    return metadata, keypoints_2d, keypoints_3d

def get_player_ids(metadata, paddle_data, keypoints_2d, rs):
    person_ids = set(md["object_id"] for md in metadata)
    if len(person_ids) < 2:
        raise ValueError("Less than two people detected in the video.")
    
    person_votes = { person_id: 0 for person_id in person_ids }
    person_hands = { person_id: [] for person_id in person_ids }
    
    for i in range(len(metadata)):
        person_id = metadata[i]["object_id"]
        frame = metadata[i]["frame"]
        hand1, hand2 = keypoints_2d[i][4] * rs, keypoints_2d[i][7] * rs
        person_hands[person_id].append(hand1[0])
        person_hands[person_id].append(hand2[0])
        for paddle in paddle_data[frame]:
            d = min(dist(hand1, paddle), dist(hand2, paddle))
            if d < 50:
                person_votes[person_id] += 1
    person_ids = sorted(person_ids, key=lambda person_id: person_votes[person_id])
    player_ids = sorted(person_ids[-2:], key=lambda person_id: np.mean(person_hands[person_id]))
    
    return player_ids
        
def get_player_data(object_id, metadata, keypoints_2d, keypoints_3d, rs, num_frames):
    object_keypoints_2d = [None] * num_frames
    object_keypoints_3d = [None] * num_frames
    
    for i in range(len(metadata)):
        if metadata[i]["object_id"] == object_id:
            frame = metadata[i]['frame']
            object_keypoints_2d[frame] = keypoints_2d[i]
            object_keypoints_3d[frame] = keypoints_3d[i]
    
    missing_frames = 0
    while object_keypoints_2d[missing_frames] is None:
        missing_frames += 1
    object_keypoints_2d[:missing_frames] = [object_keypoints_2d[missing_frames]]*missing_frames
    object_keypoints_3d[:missing_frames] = [object_keypoints_3d[missing_frames]]*missing_frames
    for i in range(num_frames):
        if object_keypoints_2d[i] is None:
            object_keypoints_2d[i] = object_keypoints_2d[i-1]
            object_keypoints_3d[i] = object_keypoints_3d[i-1]
            missing_frames += 1
    if missing_frames > 5:
        raise ValueError(f"Player pose detection failed on {missing_frames} frames.")
    
    object_keypoints_2d = np.array(object_keypoints_2d) * rs
    object_keypoints_3d = np.array(object_keypoints_3d)

    return object_keypoints_2d, object_keypoints_3d

def get_player_rs(player_feet_2d, player_feet_3d, homs, homs_table):
    rs_factors = []
    eps = 50
    table_contains = lambda p: (-eps <= p[0] <= 900 + eps) and (-eps <= p[1] <= 500 + eps)
    for i in range(len(player_feet_2d)):
        f1, f2 = transform(player_feet_2d[i], homs_table[i])
        if not (table_contains(f1) or table_contains(f2)):
            f1, f2 = transform(player_feet_2d[i], homs[i])
            a = np.linalg.norm(f1 - f2)
            b = np.linalg.norm(player_feet_3d[i, 0] - player_feet_3d[i, 1])
            rs_factors.append(a/b) 
    return np.median(rs_factors)

def get_rotation_matrix(v, u):
    """
    Computes the rotation matrix that rotates vector v onto vector u in 3D space.

    Parameters:
    v (numpy.ndarray): A 3-element array representing the initial vector.
    u (numpy.ndarray): A 3-element array representing the target vector.

    Returns:
    numpy.ndarray: A 3x3 rotation matrix that rotates vector v to align with vector u.
    
    Note:
    Both input vectors should be unit vectors. If they are not, they will be normalized 
    within the function.
    """
    # Ensure the input vectors are unit vectors
    v = v / np.linalg.norm(v)
    u = u / np.linalg.norm(u)
    
    # Compute the cross product and dot product
    v_cross_u = np.cross(v, u)
    v_dot_u = np.dot(v, u)
    
    # Compute the skew-symmetric cross-product matrix of v_cross_u
    K = np.array([
        [0, -v_cross_u[2], v_cross_u[1]],
        [v_cross_u[2], 0, -v_cross_u[0]],
        [-v_cross_u[1], v_cross_u[0], 0]
    ])
    
    # Compute the rotation matrix using Rodrigues' formula
    I = np.identity(3)
    R = I + K + K @ K * ((1 - v_dot_u) / (np.linalg.norm(v_cross_u) ** 2))
    
    return R

def cross(v1, v2) -> np.ndarray:
    return np.cross(v1, v2)
