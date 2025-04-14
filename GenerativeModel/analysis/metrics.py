import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from model import TransformerModel
from loaders.loader import CombinedDataset
from loaders.youtube_loader import FORMAT_RANGES, FORMAT_SIZE
from inference import generate, load_model
from collections import defaultdict

# Hyperparameters (make sure these match your training settings)
TOKEN_DIM = FORMAT_SIZE
MODEL_PARAMS = (TOKEN_DIM, 256, 16, 4, 1024)  # d_input, d_model, nhead, num_layers, dim_feedforward
CONTEXT_WINDOW = 16
device = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

def basic_loss_metrics(obj_name, loss_name, obj, loss_fn, num_samples, dataset_name, model_name, loss_unit="meters", ymax=2.0, add_pad_r=False, make_plot=True):    
    dataset = loader.youtube_test_dataset
    mask = loader.youtube_mask
    context_size = 8
    
    plt.figure(figsize=(10, 6))  # Increase figure size for better readability
    
    losses_dict = defaultdict(list)
    for i in range(num_samples):
        predicted_sequences, fps, updated_context_size = generate(
            model, 
            dataset, 
            mask, 
            context_size, 
            device,
            1, 
            use_mask_on_generation=True
        )
        predicted_sequences = predicted_sequences[:, updated_context_size:]
        
        predicted_sequences = predicted_sequences * (dataset.std + 1e-8) + dataset.mean
        ground_truth = predicted_sequences[0][:, FORMAT_RANGES[obj][0]:FORMAT_RANGES[obj][1]]
        pred = predicted_sequences[1][:, FORMAT_RANGES[obj][0]:FORMAT_RANGES[obj][1]]
        ground_truth = ground_truth.numpy()
        pred = pred.numpy()
        
        loss = loss_fn(ground_truth, pred)
        timesteps = np.arange(len(ground_truth)) / fps
        
        pruned_timesteps, pruned_loss = [], []
        for i in range(len(ground_truth)):
            if loss[i] > 0:
                pruned_timesteps.append(timesteps[i])
                pruned_loss.append(loss[i])
        loss = np.array(pruned_loss)
        timesteps = np.array(pruned_timesteps)
        for i in range(len(loss)):
            if loss[i] > -1:
                losses_dict[timesteps[i]].append(loss[i])
        plt.plot(timesteps, loss, color="grey", alpha=0.1)
    
    timesteps = sorted(list(losses_dict.keys()))
    losses = []
    for t in timesteps:
        losses.append(np.nanmedian(losses_dict[t]))
    
    if make_plot:   
        # Add vertical bar at the point where the opponent hits the ball
        opponent_hit_time = (abs(context_size) / 100) - 0.03
        plt.axvline(x=opponent_hit_time, color='blue', linestyle='--', label='Opponent hits ball')
        if add_pad_r:
            plt.axhline(y=0.16, color="red", label="Paddle diameter", alpha=0.5)
            
        plt.plot(timesteps, losses, color="black", label="Median loss")
        plt.xlabel("Time (seconds)", fontsize=12)
        plt.ylabel(f"Loss ({loss_unit})", fontsize=12)
        plt.xlim(xmin=0, xmax=0.4)
        plt.ylim(ymin=0, ymax=ymax)
        plt.title(f"{loss_name} loss of {model_name} on {obj_name}", fontsize=14)

        plt.legend()
        
        # Adjust layout and save
        plt.tight_layout()
        if not os.path.exists(f"plots/{model_name.replace(' ', '')}/"):
            os.makedirs(f"plots/{model_name.replace(' ', '')}/")
        plt.savefig(f"plots/{model_name.replace(' ', '')}/metric_{model_name}_{obj_name}_{dataset_name} Data_{time.time()}.png", dpi=300)
        plt.clf()
    
    return timesteps, losses
        
def dist_loss(a, b):
    loss = np.sqrt(np.sum((a - b)**2, axis=-1))
    return loss

def many_dist_loss(a, b):
    a = a.reshape(a.shape[0], -1, 3)
    b = b.reshape(b.shape[0], -1, 3)
    return np.mean(np.sqrt(np.sum((a - b)**2, axis=-1)), axis=-1)

def pad_pos_loss(a, b):
    a = a[:, :3]
    b = b[:, :3]
    return dist_loss(a, b)
        
def depth_loss(a, b):
    a = a[:, 0:1]
    b = b[:, 0:1]
    return dist_loss(a, b)

def speed_loss(a, b):
    a_speeds = (a[2:]-a[:-2])/2
    b_speeds = (b[2:]-b[:-2])/2
    a_speeds = np.linalg.norm(a_speeds, axis=-1)
    b_speeds = np.linalg.norm(b_speeds, axis=-1)
    a_speeds = np.concatenate(([np.linalg.norm(a[1] - a[0])], a_speeds, [np.linalg.norm(a[-1] - a[-2])]), 0)
    b_speeds = np.concatenate(([np.linalg.norm(b[1] - b[0])], b_speeds, [np.linalg.norm(b[-1] - b[-2])]), 0)
    return abs(a_speeds - b_speeds)

def pad_orientation_loss(q1, q2):
    q1 = q1[:, 3:]
    q2 = q2[:, 3:]
    dot_products = np.einsum('ij,ij->i', q1, q2)
    thetas = np.arccos(2 * dot_products**2 - 1)
    return thetas

def crosses_hitting_plane(seq):
    i = np.where(seq[:, 0] > 1.3)[0]
    if len(i) == 0:
        return -1
    else:
        return i[0]

def hitting_plane_metrics(num_samples, dataset_name, model_name):    
    dataset = loader.youtube_test_dataset
    mask = loader.youtube_mask
    context_size = 24
    
    plt.figure(figsize=(10, 6))  # Increase figure size for better readability
    
    losses = []
    for i in range(num_samples):
        predicted_sequences, fps, updated_context_size = generate(
            model, 
            dataset, 
            mask, 
            context_size, 
            device,
            1, 
            use_mask_on_generation=True
        )
        predicted_sequences = predicted_sequences[:, updated_context_size:]
        
        predicted_sequences = predicted_sequences * (dataset.std + 1e-8) + dataset.mean
        ground_truth = predicted_sequences[0][:, FORMAT_RANGES["b"][0]:FORMAT_RANGES["b"][1]]
        pred = predicted_sequences[1][:, FORMAT_RANGES["b"][0]:FORMAT_RANGES["b"][1]]
        ground_truth = ground_truth.numpy()
        pred = pred.numpy()
        
        a = crosses_hitting_plane(ground_truth) 
        b = crosses_hitting_plane(pred)
        if a >= 0 and b >= 0:
            loss = np.linalg.norm(ground_truth[a][1:] - pred[b][1:])
            losses.append(loss)

    # Calculate mean and standard deviation
    median_loss = np.median(losses)
    std_loss = np.std(losses)

    # Create the histogram
    plt.hist(losses, bins=50, edgecolor='black')
    plt.title(f"Absolute Loss of {model_name} on Hit Plane Cross Point Prediction", fontsize=14)
    plt.xlabel("Loss (meters)")
    plt.ylabel("Freq")

    # Add mean and standard deviation lines
    plt.axvline(median_loss, color='r', linestyle='dashed', linewidth=2, label=f'Median: {median_loss:.2f}m')

    # Adjust layout and save
    plt.legend()
    plt.tight_layout()
    if not os.path.exists(f"plots/{model_name.replace(' ', '')}/"):
        os.makedirs(f"plots/{model_name.replace(' ', '')}/")
    plt.savefig(f"plots/{model_name.replace(' ', '')}/metric_{model_name}_Hit Plane_{dataset_name} Data_{time.time()}.png", dpi=300)
    plt.clf()

def overlay(data, labels, model1_name, model2_name, obj_name, loss_unit="meters", ymax=2.0, add_pad_r=False):
    assert len(data) == len(labels)
    
    plt.figure(figsize=(10, 6))  # Increase figure size for better readability
    
    opponent_hit_time = (abs(8) / 100) - 0.03
    plt.axvline(x=opponent_hit_time, color='blue', linestyle='--', label='Opponent hits ball')
    if add_pad_r:
        plt.axhline(y=0.16, color="red", label="Paddle diameter", alpha=0.5)
        
    for i in range(len(data)):
        timesteps, losses = data[i]
        label = labels[i]
        plt.plot(timesteps, losses, label=label)
 
    plt.ylabel(f"Loss ({loss_unit})", fontsize=12)
    plt.xlim(xmin=0, xmax=0.4)
    plt.ylim(ymin=0, ymax=ymax)
    plt.title(f"Comparison of {model1_name} Model and {model2_name} Model on {obj_name} Prediciton", fontsize=14)
    plt.legend()
        
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"plots/comparisons/{model1_name}_{model2_name}_{obj_name}.png", dpi=300)
    plt.clf()
    
# Load data
loader = CombinedDataset("data/recons", "data/recons_lab", "data/recons_test", "data/recons_lab_test", val_split=0.05, constant_fps=True)

# MODEL_NAME = "Masked Ball Model"
# DATASET_NAME = "Lab"
# model = load_model('models/masked_ball_model.pth', device)

MODEL_NAME = "Full Model"
DATASET_NAME = "Youtube"
model = load_model('models/full_model.pth', device)

# MODEL_NAME = "Lab Model"
# DATASET_NAME = "Lab"
# model = load_model('models/lab_model.pth', device)

# MODEL_NAME = "Finetuned Model"
# DATASET_NAME = "Lab"
# model = load_model('models/finetuned_model.pth', device)

basic_loss_metrics("Ball Speed", "Absolute", "b", speed_loss, 100, DATASET_NAME, MODEL_NAME, ymax=0.25, loss_unit="meters/sec")
basic_loss_metrics("Ball X Position", "Absolute", "b", depth_loss, 100, DATASET_NAME, MODEL_NAME, ymax=1.0)
basic_loss_metrics("Ball Position", "Absolute", "b", dist_loss, 100, DATASET_NAME, MODEL_NAME, ymax=1.5, add_pad_r=True)
basic_loss_metrics("Rel. Player Joints", "Absolute", "p1", many_dist_loss, 100, DATASET_NAME, MODEL_NAME, ymax=0.2)
basic_loss_metrics("Player Root", "Absolute", "p1_root", dist_loss, 100, DATASET_NAME, MODEL_NAME, ymax=1.0)
# basic_loss_metrics("Paddle Position", "Absolute", "p1_pad", pad_pos_loss, 100, DATASET_NAME, MODEL_NAME, ymax=1.0, add_pad_r=True)
# basic_loss_metrics("Paddle Orientation", "Absolute", "p1_pad", pad_orientation_loss, 100, DATASET_NAME, MODEL_NAME, ymax=6, loss_unit="radians")
hitting_plane_metrics(1000, DATASET_NAME, MODEL_NAME)


# Comparison
# MODEL_NAME = "Finetuned Model"
# DATASET_NAME = "Lab"
# model = load_model('models/finetuned_model.pth', device)
# finetuned_model_bp_data = basic_loss_metrics("Ball Position", "Absolute", "b", dist_loss, 100, DATASET_NAME, MODEL_NAME, ymax=1.5, add_pad_r=True, make_plot=False)
# finetuned_model_pp_data = basic_loss_metrics("Paddle Position", "Absolute", "p1_pad", pad_pos_loss, 100, DATASET_NAME, MODEL_NAME, ymax=1.0, add_pad_r=True, make_plot=False)

# MODEL_NAME = "Lab Model"
# DATASET_NAME = "Lab"
# model = load_model('models/lab_model.pth', device)
# lab_model_bp_data = basic_loss_metrics("Ball Position", "Absolute", "b", dist_loss, 100, DATASET_NAME, MODEL_NAME, ymax=1.5, add_pad_r=True, make_plot=False)
# lab_model_pp_data = basic_loss_metrics("Paddle Position", "Absolute", "p1_pad", pad_pos_loss, 100, DATASET_NAME, MODEL_NAME, ymax=1.0, add_pad_r=True, make_plot=False)

# MODEL_NAME = "Full Model"
# DATASET_NAME = "Lab"
# model = load_model('models/full_model.pth', device)
# full_model_bp_data = basic_loss_metrics("Ball Position", "Absolute", "b", dist_loss, 100, DATASET_NAME, MODEL_NAME, ymax=1.5, add_pad_r=True, make_plot=False)
# full_model_pp_data = basic_loss_metrics("Paddle Position", "Absolute", "p1_pad", pad_pos_loss, 100, DATASET_NAME, MODEL_NAME, ymax=1.0, add_pad_r=True, make_plot=False)

# MODEL_NAME = "Masked Ball Model"
# DATASET_NAME = "Lab"
# model = load_model('models/masked_ball_model.pth', device)
# masked_model_bp_data = basic_loss_metrics("Ball Position", "Absolute", "b", dist_loss, 100, DATASET_NAME, MODEL_NAME, ymax=1.5, add_pad_r=True, make_plot=False)
# masked_model_pp_data = basic_loss_metrics("Paddle Position", "Absolute", "p1_pad", pad_pos_loss, 100, DATASET_NAME, MODEL_NAME, ymax=1.0, add_pad_r=True, make_plot=False)

# overlay([full_model_bp_data, lab_model_bp_data], ["Full model performance", "Lab model performance"], "Full", "Lab", "Ball Position", loss_unit="meters", ymax=1.5)
# overlay([finetuned_model_bp_data, lab_model_bp_data], ["Finetuned model performance", "Lab model performance"], "Finetuned", "Lab", "Ball Position", loss_unit="meters", ymax=1.5)
# overlay([finetuned_model_pp_data, lab_model_pp_data], ["Finetuned model performance", "Lab model performance"], "Finetuned", "Lab", "Paddle Position", loss_unit="meters", ymax=1.5, add_pad_r=False)
# overlay([masked_model_bp_data, lab_model_bp_data], ["Masked model performance", "Lab model performance"], "Masked", "Lab", "Ball Position", loss_unit="meters", ymax=1.5)
