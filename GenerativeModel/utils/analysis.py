import numpy as np

def generate_quad_matrix(n):
    rows = np.arange(1, n+1)
    matrix = np.column_stack((rows**2, rows, np.ones_like(rows)))
    return matrix

def generate_linear_matrix(n):
    rows = np.arange(1, n+1)
    matrix = np.column_stack((rows, np.ones_like(rows)))
    return matrix

M2 = np.linalg.pinv(generate_quad_matrix(4))
M1 = np.linalg.pinv(generate_linear_matrix(4))
def correct_pred(pred_ball):
    v = pred_ball[1:] - pred_ball[:-1]
    v_neg = np.where(v[:, 0] < 0)[0] + 1
    v_neg_diff = np.where(v_neg[1:] - v_neg[:-1] > 5)[0] + 1
    if len(v_neg_diff) == 0:
        return pred_ball
    
    i = v_neg[v_neg_diff[0]]
    n = len(pred_ball)-i+4
    segment = pred_ball[i-4:i]
    correction_x, correction_y, correction_z = extend_traj(segment, n)
    pred_ball[i:, 0] = correction_x
    pred_ball[i:, 1] = correction_y
    pred_ball[i:, 2] = correction_z
    return pred_ball
    
def extend_traj(segment, n):
    correction_x = generate_linear_matrix(n)[4:] @ M1 @ segment[:, 0]
    correction_y = generate_linear_matrix(n)[4:] @ M1 @ segment[:, 1]
    correction_z = generate_quad_matrix(n)[4:] @ M2 @ segment[:, 2]
    return correction_x, correction_y, correction_z

def generate_reachable_box(t, c=[1.77, 0, 0.9], v=[2.5, 1.5, 1.5], bounds=[1.3, 5, -1.25, 1.25, 0.0, 2]):
    x, y, z = c
    v_x, v_y, v_z = v
    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    return [
        [max(x-v_x*t, min_x), min(x+v_x*t, max_x)],
        [max(y-v_y*t, min_y), min(y+v_y*t, max_y)],
        [max(z-v_z*t, min_z), min(z+v_z*t, max_z)],
    ]    
    
def bound_regions(confidence_regions, bounds=[-1.25, 1.25]):
    l, r = bounds
    for i in range(len(confidence_regions)):
        confidence_regions[i][1][0] = min(max(confidence_regions[i][1][0], l), r)
        confidence_regions[i][1][1] = max(min(confidence_regions[i][1][1], r), l)
    return confidence_regions
    
def contains(box1, box2):
    for d in range(len(box1)):
        a1, b1 = box1[d]
        a2, b2 = box2[d]
        if a2 < a1 or b2 > b1:
            return False
    return True
    
def project(p, box):
    new_p = []
    for d in range(len(box)):
        a, b = box[d]
        x = p[d]
        new_p.append(min(max(a, x), b))
    return np.array(new_p)

def shrink(p, x0=[1.77, 0, 0.9], lmbda=0.1):
    x0 = np.array(x0)
    p  = np.array(p)
    return lmbda*x0 + (1-lmbda)*p
    
    
    