import csv
import numpy as np
import matplotlib.pyplot as plt     
from .parabola import fit_parabola
from .table import transform

def load_ball_data(video_name, ball_dir):
    ball_data_path = f"{ball_dir}/{video_name}_ball.csv"
    data = []
    with open(ball_data_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)  
        header = csv_reader.__next__()
        for row in csv_reader:
            row = list(map(float, row))
            if row[1] or len(row) == 0:
                data.append(row)
            else:
                data.append(data[-1])
        data = np.array(data)
    return data[:, 2:4]

def average(arr, window_size=3):
    window_size = window_size // 2
    n = len(arr)
    output = np.zeros(arr.shape)
    for i in range(0, n):
        l = max(0, i-window_size)
        r = min(i+1+window_size, n)
        output[i] = np.mean(arr[l:r], axis=0)
    return output

def rotate_about(idx, theta, sequence):
    if theta == 0:
        return sequence
    points = np.array(list(enumerate(sequence)))    
    base = points[idx]
    points = points - base        
    R = np.array([
            [np.cos(theta), -np.sin(theta)], 
            [np.sin(theta),  np.cos(theta)]
        ])
    points = points @ R.T    
    points = points + base
    return points[:, 1]

def reparam_theta(t):
    if np.pi/2 <= t and t <= 3*np.pi/2:
        return t - np.pi/2
    if 0 <= t <= np.pi/2:
        return np.pi/2 - t
    return 5*np.pi/2 - t

def find_bounce_points(sequence, window_size):
    n = len(sequence)
    sequence = normalize(sequence) * n
    bounce_points = []
    points = np.array(list(enumerate(sequence)))
    for i in range(n):
        l = max(0, i-(window_size//2))
        r = min(i+1+(window_size//2), n)
        theta1 = np.arctan2(points[l:i, 1]   - points[i, 1], points[l:i, 0]   - points[i, 0]) + np.pi
        theta1 = [reparam_theta(t) for t in theta1]
        theta1 = 0 if len(theta1) == 0 else max(theta1)
        theta2 = np.arctan2(points[i+1:r, 1] - points[i, 1], points[i+1:r, 0] - points[i, 0]) + np.pi
        theta2 = [reparam_theta(t) for t in theta2]
        theta2 = 0 if len(theta2) == 0 else max(theta2)
        if theta1 + theta2 < 3.08:
            bounce_points.append(i)
    return bounce_points

def find_peaks(arr, window_size):
    window_size = window_size // 2
    n = len(arr)
    peaks = []
    for i in range(0, n-1):
        l = max(0, i-window_size)
        r = min(i+1+window_size, n)
        if np.all(arr[l:i] <= arr[i]) and np.all(arr[i+1:r] <= arr[i]):
            peaks.append(i) 
    return peaks

def distance(arr1, arr2):
    return np.linalg.norm(arr1-arr2, axis=1)

def find_hit_points_from_hand_dists(hand_dists):
    hand_hits = find_peaks(-hand_dists, window_size=7)
    if hand_hits[0] == 0 and hand_dists[0] > 200:
        hand_hits = hand_hits[1:]
    return np.intersect1d(hand_hits, np.where(hand_dists < 400))

def normalize(sequence):
    mean = np.mean(sequence)
    std  = np.std(sequence)
    z    = (sequence - mean) / std
    return z

def discrete_projection(a, domain, thresh=5):
    n = len(a)
    output = []
    for i in range(n):
        distances = abs(domain - a[i])
        idx = np.argmin(distances)
        if distances[idx] <= thresh:
            output.append(domain[idx])
    return np.array(output)

def adjust_hits(player1_hits, player2_hits, player1_hand_dist, player2_hand_dist):
    pruned_player1_hits, pruned_player2_hits = [], []
    hits = np.concatenate((player1_hits, player2_hits))
    hits.sort()    
    curr_player = -1
    for hit in hits:
        if hit in player1_hits:
            if curr_player == 1:
                if player1_hand_dist[pruned_player1_hits[-1]] > player1_hand_dist[hit]:
                    pruned_player1_hits[-1] = hit
            else:
                curr_player = 1
                pruned_player1_hits.append(hit)
        else:
            if curr_player == 2:
                if player2_hand_dist[pruned_player2_hits[-1]] > player2_hand_dist[hit]: 
                    pruned_player2_hits[-1] = hit
            else:
                curr_player = 2
                pruned_player2_hits.append(hit)
    return np.array(pruned_player1_hits), np.array(pruned_player2_hits)

def adjust_bounces(player1_hits, player2_hits, bounces, y):
    hits = np.concatenate((player1_hits, player2_hits))
    hits.sort()
    
    res = []
    h1, h2 = hits[0:2]
    res = np.concatenate((res, fit_parabola(y[h1:h2+1], bounces[(h1<bounces) & (bounces<h2)] - h1, num_segments=2) + h1))
    for h in hits[2:]:
        h1, h2 = h2, h
        try:
            res = np.concatenate((res, fit_parabola(y[h1:h2+1], bounces[(h1<bounces) & (bounces<h2)] - h1, num_segments=1) + h1))
        except Exception as e:
            return res.astype(int), player1_hits[player1_hits <= h1], player2_hits[player2_hits <= h1]
    return res.astype(int), player1_hits, player2_hits

def differentiate(points, order=1):
    n = len(points)
    gradients = np.zeros(points.shape)
    for i in range(n):
        l, r = max(0, i-1), min(n-1, i+1)
        if order == 1:
            gradients[i] = (points[r] - points[l]) / (r - l)
        if order == 2:
            gradients[i] = (points[r] - 2*points[i] + points[l]) / (r - l)**2
    return gradients

def prune_hits(hits, not_pot_hits):
    for k in range(1, len(hits)):
        if not any(hits[k]-i in not_pot_hits for i in range(1, 5)):
            return hits[:k]
    return hits 
            
def process_ball_data(points, p1_hand, p2_hand, homs_table, verbose=False):    
    # Load and transform the data
    points = points.copy()
    for i in range(len(points)):
        points[i]  = transform(points[i],  homs_table[i])
        p1_hand[i] = transform(p1_hand[i], homs_table[i])
        p2_hand[i] = transform(p2_hand[i], homs_table[i])
    y = points[:, 1]
    
    # Smooth the data
    points_smoothed = average(points, window_size=3)
    x_smoothed, y_smoothed = points_smoothed[:, 0], points_smoothed[:, 1]

    # Find potential bounce points 
    e = 200
    table_contains = lambda p: (-e <= p[0] <= 900 + e) and (-e <= p[1] <= 500 + e)
    bounce_idxs = find_bounce_points(y_smoothed, window_size=7)
    bounce_idxs = np.intersect1d(bounce_idxs, np.where([table_contains(p) for p in points]))
    
    # Find potential hit points (based on ball's horizontal position)
    v_x = average(differentiate(x_smoothed), window_size=5)
    p1_pot_hit_idxs = np.where(v_x >= 0)[0]
    p2_pot_hit_idxs = np.where(v_x <= 0)[0]
    
    # Find potential hit points (based on the balls distance to each player)
    p1_hand_dists = average(distance(p1_hand, points), window_size=3)
    p2_hand_dists = average(distance(p2_hand, points), window_size=3)
    p1_hit_idxs = find_hit_points_from_hand_dists(p1_hand_dists)
    p2_hit_idxs = find_hit_points_from_hand_dists(p2_hand_dists)

    # Merge both sets of potential hit points
    p1_hit_idxs = prune_hits(discrete_projection(p1_hit_idxs, p1_pot_hit_idxs), p2_pot_hit_idxs)
    p2_hit_idxs = prune_hits(discrete_projection(p2_hit_idxs, p2_pot_hit_idxs), p1_pot_hit_idxs)
    
    # Finalize the hit/bounce points
    p1_hit_idxs, p2_hit_idxs = adjust_hits(p1_hit_idxs, p2_hit_idxs, p1_hand_dists, p2_hand_dists)
    bounce_idxs, p1_hit_idxs, p2_hit_idxs = adjust_bounces(p1_hit_idxs, p2_hit_idxs, bounce_idxs, y_smoothed)

    if verbose:        
        y = y_smoothed
        plt.plot(y, color="black")
        plt.scatter(p1_pot_hit_idxs, y[p1_pot_hit_idxs], marker="o", color="blue", alpha=0.25)
        plt.scatter(p2_pot_hit_idxs, y[p2_pot_hit_idxs], marker="o", color="red", alpha=0.25)
        plt.scatter(bounce_idxs,  y[bounce_idxs], marker="x", color="green", label="bounces")
        plt.scatter(p1_hit_idxs,  y[p1_hit_idxs], marker="x", color="blue", label="player1_hits")
        plt.scatter(p2_hit_idxs,  y[p2_hit_idxs], marker="x", color="red",  label="player2_hits")
        plt.gca().invert_yaxis()
        plt.legend()
        plt.show()
        
        
    if len(p1_hit_idxs) + len(p2_hit_idxs) < 3:
        raise ValueError("Not enough of hit points detected.")
    
    x1 = points[bounce_idxs[0]][0]
    x2 = points[bounce_idxs[1]][0]
    if not ((x1 < 500 and x2 > 400) or (x1 > 400 and x2 < 500)):
        raise ValueError("Serve detection failed.")
    
    return bounce_idxs, p1_hit_idxs, p2_hit_idxs

def interpolate_parabolic_trajectory(trajectory, fps):
    def parabolic_interpolation(p1, p2, num_points):
        # Create a parabolic path between p1 and p2
        t = np.linspace(0, 1, num_points + 2)
        t = t[:, np.newaxis]  # Make t a column vector for broadcasting
        
        g = num_points**2 * 32.2 * 100 / fps**2
        a, b, c = -g, g + p2[2]-p1[2], p1[2]
        parabolic_path_z = a*t**2 + b*t + c
        a, b = p2[0] - p1[0], p1[0]
        path_x = a*t + b
        a, b = p2[1] - p1[1], p1[1]
        path_y = a*t + b
          
        return np.hstack((path_x, path_y, parabolic_path_z))[1:-1]  # Exclude the end points p1 and p2
    
    n = len(trajectory)
    result = trajectory.copy()
    
    i = 0
    while i < n:
        if result[i] is None:
            # Find the start and end of the None segment
            start = i - 1
            while i < n and result[i] is None:
                i += 1
            end = i
            
            # Interpolate the segment
            if start >= 0 and end < n:
                num_missing = end - start - 1
                interpolated_points = parabolic_interpolation(result[start], result[end], num_missing)
                result[start+1:end] = interpolated_points
        else:
            i += 1
    
    return result

def adjust_trajectory(ball_positions, player1_hits, player2_hits, bounces, alpha=0.5):
    hit_times = np.concatenate((player1_hits, player2_hits))
    hit_times.sort()
    j = 0
    
    for i in range(len(bounces)):
        t = bounces[i]
        if t > hit_times[j+1]:
           j += 1 
        
        # Interpolate the bounce position between the two hit positions
        bounce = ball_positions[t]
        hit1   = ball_positions[hit_times[j]]
        hit2   = ball_positions[hit_times[j+1]]
        x, x1, x2 = bounce[0], hit1[0], hit2[0]
        m = (x - x2) / (x1 - x2)
        bounce_interp = m * hit1 + (1 - m) * hit2
        bounce_interp[2] = bounce[2]
        ball_positions[t] = alpha * bounce_interp + (1 - alpha) * bounce
        
        # Clip the ball to hit the table
        ball_positions[t][0] = np.clip(ball_positions[t][0], -450, 450)
        ball_positions[t][1] = np.clip(ball_positions[t][1], -250, 250)
        
    return ball_positions