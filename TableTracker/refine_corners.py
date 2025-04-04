# corners.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from unet import UNet

def visualize_results(frame_gray_np, output_np, assigned_points):
    """Visualize the input image, ground truth, model output, and detected points."""
    # Prepare to plot the centroids on the image
    frame_with_points = cv2.cvtColor(frame_gray_np, cv2.COLOR_GRAY2BGR)
    point_colors = {
        'top_left': (255, 0, 0),      # Blue
        'top_mid': (0, 255, 0),       # Green
        'top_right': (0, 0, 255),     # Red
        'bottom_left': (255, 255, 0), # Cyan
        'bottom_mid': (255, 0, 255),  # Magenta
        'bottom_right': (0, 255, 255) # Yellow
    }
    for label, centroid in assigned_points.items():
        if centroid is not None:
            x, y = int(centroid[0]), int(centroid[1])
            color = point_colors.get(label, (255, 255, 255))  # Default to white
            cv2.circle(frame_with_points, (x, y), 5, color, -1)
            cv2.putText(frame_with_points, label, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Convert BGR to RGB for displaying with Matplotlib
    frame_with_points_rgb = cv2.cvtColor(frame_with_points, cv2.COLOR_BGR2RGB)
    
    # Visualize
    plt.figure(figsize=(20, 5))
    
    # Input Image
    plt.subplot(1, 4, 1)
    plt.imshow(frame_gray_np, cmap='gray')
    plt.title('Input Image')
    plt.axis('off')

    # Model Output
    plt.subplot(1, 4, 3)
    plt.imshow(output_np, cmap='gray')
    plt.title('Model Output')
    plt.axis('off')
    
    # Detected Points
    plt.subplot(1, 4, 4)
    plt.imshow(frame_with_points_rgb)
    plt.title('Detected Points')
    plt.axis('off')
    
    plt.savefig("temp_vis.png")
    plt.clf()

def load_model(model_path, device):
    """Load the trained model."""
    model = UNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

def process_image(frame_gray, model, device, threshold, prior_points):
    """Process a single image and detect points."""
    frame_gray = frame_gray.to(device).unsqueeze(0)  # Shape: (1, 1, H, W)
    with torch.no_grad():
        # Perform inference
        output = model(frame_gray)  # Output shape: (1, 1, H, W)
        output_np = torch.sigmoid(output).squeeze().cpu().numpy()  # Shape: (H, W)
    
    # Apply thresholding to get a binary mask
    binary_output = (output_np > threshold).astype(np.uint8)
    
    # Detect points with prior information
    detected_points = detect_points(binary_output, prior_points)
    
    return output_np, detected_points


def detect_points(binary_mask, prior_points):
    """Detect points from the binary mask and assign them to table parts."""
    # Find connected components
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
    
    # Exclude background label (0)
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    component_centroids = centroids[1:]
    
    # Assign points using prior information
    assigned_points = assign_points_to_table(component_centroids, prior_points)
    
    return assigned_points

def assign_points_to_table(centroids, prior_points):
    """
    Assign each centroid to a specific part of the table based on proximity to prior points.
    
    centroids: array of detected centroids (N x 2)
    prior_points: dictionary with labels as keys and (x, y) coordinates as values
    """
    if prior_points is None:
        raise ValueError("Prior points must be provided for assigning centroids.")
    
    assigned_points = {}
    centroids_list = centroids.tolist()
    
    for label, prior in prior_points.items():
        if len(centroids_list) == 0:
            assigned_points[label] = [np.nan, np.nan]
            continue  # No more centroids to assign
    
        # Compute distances from prior to all remaining centroids
        distances = [np.linalg.norm(np.array(prior) - np.array(c)) for c in centroids_list]
    
        # Find the centroid with minimum distance
        min_index = np.argmin(distances)
        
        if distances[min_index] < 50:
            assigned_centroid = centroids_list[min_index]
        
            # Assign the centroid to the label
            assigned_points[label] = assigned_centroid
        
            # Remove the assigned centroid from the list
            del centroids_list[min_index]
        else:
            assigned_points[label] = [np.nan, np.nan]

    return assigned_points

def refine_corners(model, device, frame, bbox, prior_points):
    frame = frame[bbox[2]:bbox[3], bbox[0]:bbox[1]]
    frame_gray_np = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray_np = cv2.resize(frame_gray_np, (256, 128), interpolation=cv2.INTER_AREA)
    image_height, image_width = frame_gray_np.shape
    
    # Adjust prior_points to match resized frame
    adjusted_prior_points = {}
    for key, point in prior_points.items():
        x, y = point
        x = (x - bbox[0]) * (256 / (bbox[1] - bbox[0]))
        y = (y - bbox[2]) * (128 / (bbox[3] - bbox[2]))
        x = np.clip(x, 0, image_width - 1)
        y = np.clip(y, 0, image_height - 1)
        adjusted_prior_points[key] = (x, y)
    
    frame_gray = torch.from_numpy(frame_gray_np).float().to(device).unsqueeze(0)  # Shape: (1, 1, H, W)
    output_np, assigned_points = process_image(frame_gray, model, device, 0.5, adjusted_prior_points)
    
    # visualize_results(frame_gray_np, output_np, assigned_points)
    # input("")
    
    corners = np.array([
        assigned_points.get("top_left"),
        assigned_points.get("top_right"),
        assigned_points.get("bottom_left"),
        assigned_points.get("bottom_right"),
        assigned_points.get("top_mid"),
        assigned_points.get("bottom_mid")
    ])
    
    # Map points back to original image coordinates
    corners[:, 0] = corners[:, 0] * (bbox[1] - bbox[0]) / 256 + bbox[0]
    corners[:, 1] = corners[:, 1] * (bbox[3] - bbox[2]) / 128 + bbox[2]
    corners = np.nan_to_num(corners, 0.0)
    
    # print(corners.round().astype(int).flatten())
    return corners.round().astype(int)