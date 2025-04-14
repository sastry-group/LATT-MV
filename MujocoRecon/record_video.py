from mujoco_env_just_vis import KukaTennisEnv
# from stable_baselines3.common.env_util import SubprocVecEnv
from stable_baselines3 import PPO
import time
import numpy as np
import pickle
from scipy.optimize import curve_fit
from scipy.spatial.transform import Rotation as R
import tqdm
import argparse
import random
import torch
import imageio
import cv2

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)
time_sleep = 0.
PARENT_DIR = '../release_data'

def parabola(x, a, b, c):
    return a * x**2 + b * x + c

successful = 0
unsuccesful = 0

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='oracle', help='Name of experiment')
parser.add_argument('--ind', default=0, help='indice to generate videos for')
parser.add_argument('--pre', action='store_true', help='Enable pre-positioning')
parser.add_argument('--gt', action='store_true', help='Enable GT pre-positioning')
parser.add_argument('--init', action='store_true', help='Enable initializing to average hit position')
parser.add_argument('--lambd', default=0.1, help='Value of lambda')

parser.add_argument('--render', action='store_true', help='Enable rendering')

args = parser.parse_args()

with open("targets.pkl", "rb") as f:
    targets = pickle.load(f)
# print(targets[0])
# exit(0)
env = KukaTennisEnv(proc_id=1)
# env.seed(0)
model = PPO.load("logs/best_model2/best_model")
obs, _ = env.reset()
where_landed = []
target_poses = []
poses_achieved = []
lambda_ = float(args.lambd)
hit_positions = []
for target in tqdm.tqdm(targets):
    ball_trajectory, switch_timestep, anticipatory_target_position, filename = target
    ball_trajectory = np.array(ball_trajectory)
    ball_trajectory[:,0] = -ball_trajectory[:,0]
    ball_trajectory[:,0] += 1.5
    ball_trajectory[:,1] *= -1
    ball_trajectory[:,2] -= 0.1
    hit_positions.append(ball_trajectory[-1])
hit_positions = np.array(hit_positions)
avg_hit_position = np.mean(hit_positions, axis=0)
# print("Average hit position:", avg_hit_position)
# exit(0)

target_id = 321
target = targets[target_id]
ball_trajectory, switch_timestep, anticipatory_target_position, filename = target
print(filename, switch_timestep)
# exit(0)
match_no = filename.split("h")[-1].split("_")[0]
rally_no = filename.split("_")[-1].split(".")[0]
scene_info = np.load(f"{PARENT_DIR}/match{match_no}/match{match_no}_{rally_no}/match{match_no}_{rally_no}_recon.npy")
ball_positions = scene_info[:, -1]
player_joints = scene_info[:, -89:-1]
# print(ball_positions)

side = 0
i = 0
while ball_positions[i,0] == 0.:
    i += 1
if ball_positions[i,1] < 0:
    side = 1
else:
    side = -1

print("Side:", side)

ball_positions[:,0] *= side
ball_positions[:,1] *= side
player_joints[:,:,0] *= side
player_joints[:,:,1] *= side

ball_trajectory[:,1] *= -1
ball_trajectory[:,0] *= -1

diffs = []
for i in range(0,len(ball_positions)-len(ball_trajectory)):
    diff = np.sum(np.linalg.norm(ball_trajectory - ball_positions[i:i+len(ball_trajectory)],axis=1))
    # print(diff)
    diffs.append(diff)
mini = np.argmin(diffs)

ball_positions[:,0] += 1.5
ball_positions[:,2] -= 0.1
ball_trajectory = np.array(ball_trajectory)
ball_trajectory[:,0] += 1.5
ball_trajectory[:,2] -= 0.1
player_joints[:,:,0] += 1.5
player_joints[:,:,2] -= 0.1
env.render()
for step_t in range(len(ball_positions)) :
    action = np.zeros(9)
    env.reset_ball_throw_(ball_positions[step_t],np.zeros(3))
    env.draw_human_pose(player_joints[step_t,:25],color=(1.,0.,0.,1.))
    env.draw_human_pose(player_joints[step_t,44:44+25],color=(1.,.1,.78,1.),init_i=24)

    obs, reward, done, _, info = env.step(action)
    env.render()
    time.sleep(time_sleep)

# Save frames to video
fps = 30  # Adjust the frames per second as needed
filename = args.exp_name+'.mp4'
writer = imageio.get_writer(filename, fps=fps)
for frame in env.frames:
    frame_new = frame.copy()[288:, 256:-256]
    # rescale image frame_new to 2560x1440
    frame_new = np.array(frame_new)
    frame_new = np.clip(frame_new, 0, 255).astype(np.uint8)
    frame_new = cv2.resize(frame_new, (2560, 1440))
    # print(frame_new.shape)

    writer.append_data(frame_new)
writer.close()

env.close()
