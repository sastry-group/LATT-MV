import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

def least_squares(X, Y):
    if len(X) == 0:
        return 0, []
    
    B = np.linalg.pinv(X.T @ X) @ X.T @ Y
    mse = np.linalg.norm(Y - X @ B) ** 2
    return mse, B

def fit_parabola(y, bounce_points, num_segments, extend_bounce_points=False):
    n = len(y)

    x = np.arange(n)
    x = np.vstack([x**2, x**1, x**0]).transpose()

    best_mse, best_sub_range, best_coeffs = float("inf"), None, None
    if len(bounce_points) < num_segments:
        raise ValueError("Not enough of bounce points detected.")
    else:
        rng = bounce_points
        
    # Compute the LS solution.
    for sub_range in combinations(rng, num_segments):
        sub_range = [0] + list(sub_range) + [n]
        
        total_mse, coeffs = 0, []
        for i in range(len(sub_range)-1):
            a, b = sub_range[i], sub_range[i+1]
            X, Y = x[a:b+1], y[a:b+1]
            mse, coeff = least_squares(X, Y)
            total_mse += mse
            coeffs.append(coeff)
        
        if total_mse < best_mse:
            best_mse = total_mse
            best_sub_range = sub_range
            best_coeffs = coeffs

    # Plot the results.
    # plt.plot(np.arange(n), y)
    # for i in range(len(best_sub_range)-1):
    #     l, u = best_sub_range[i], best_sub_range[i+1]
    #     if l != u:
    #         a, b, c = best_coeffs[i]
    #         x = np.arange(l, u, 0.1)
    #         y = a * x**2 + b * x + c
    #         plt.plot(x, y)
    # plt.gca().invert_yaxis()
    # plt.show()
    
    if best_mse > 25000:
        raise ValueError("Parabola Fitting Failed.")

    return np.array(best_sub_range[1:-1])