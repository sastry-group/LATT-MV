from mujoco_env_skeleton import KukaTennisEnv
from mujoco_env_only_kuka_ik import KukaTennisEnv as KukaTennisEnvIK
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
import os
import cv2

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)
time_sleep = 0.
FPS = 30
PARENT_DIR = '../release_data'

def parabola(x, a, b, c):
    return a * x**2 + b * x + c

successful = 0
unsuccesful = 0

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='oracle', help='Name of experiment')
parser.add_argument('--ind', default=321, help='indice to generate videos for')
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
print("Average hit position:", avg_hit_position)
# exit(0)
target_id = int(args.ind)
target = targets[target_id]

where_landed.append(None)
target_poses.append(None)
poses_achieved.append(None)
ball_trajectory, switch_timestep, anticipatory_target_position, filename = target
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
if ball_positions[i,0] < 0:
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
anticipatory_target_position[0] = -anticipatory_target_position[0]
anticipatory_target_position[0] += 1.5
anticipatory_target_position[2] -= 0.1

bounce_points = []
for i in range(switch_timestep+1,len(ball_trajectory)-1):
    if (ball_trajectory[i,2] - ball_trajectory[i-1,2]) < 0 and (ball_trajectory[i+1,2] - ball_trajectory[i,2]) > 0:
        bounce_points.append(i)
if len(bounce_points) != 1:
    print("Too many bounce points")
else:
    bounce_point = bounce_points[0]
    if ball_trajectory[bounce_point][0] < 1.5-1.365:
        print("Table edge")

table_z = 0.66
for i in range(1,len(ball_trajectory)-1):
    if (ball_trajectory[i,2] - ball_trajectory[i-1,2])*(ball_trajectory[i+1,2] - ball_trajectory[i,2]) < 0:
        table_z = min(table_z, ball_trajectory[i,2])
ball_trajectory[:,2] += 0.66 - table_z  
anticipatory_target_position[2] += 0.66 - table_z
# print(anticipatory_target_position)
# exit(0)
parts = []
curr_part = [ball_trajectory[0]]
prev_pos = ball_trajectory[0]
curr_x_dir = (ball_trajectory[1,0] - ball_trajectory[0,0]) > 0
curr_z_dir = (ball_trajectory[1,2] - ball_trajectory[0,2]) > 0
for pos in ball_trajectory[1:]:
    if (prev_pos[0] > pos[0] and curr_x_dir) or (prev_pos[0] < pos[0] and not curr_x_dir):
        parts.append(curr_part)
        curr_x_dir = not curr_x_dir
        curr_part = []
    if (prev_pos[2] > pos[2] and curr_z_dir) or (prev_pos[2] < pos[2] and not curr_z_dir):
        curr_z_dir = not curr_z_dir
        if curr_z_dir:
            parts.append(curr_part)
            curr_part = []
    curr_part.append(pos)
    prev_pos = pos
parts.append(curr_part)
vs = []
positions = []
end_positions = []
if len(parts) <= 2:
    print("Not enough parts to get a segment!")
for i in range(len(parts)):
    # print("Part ", i, ":-")
    # for j in range(len(parts[i])):
    #     print(parts[i][j])
    if len(parts[i]) < 3:
        if i==1 and len(parts[i]) == 2:
            traj = np.array([parts[i-1][-1]] + parts[i])
        elif i==0 and len(parts[i]) == 2:
            traj = np.array(parts[i] + [parts[i+1][0]])
        else :
            print("Part is too small to fit a parabola!")
            continue
    if len(parts[i])>=3 :
        traj = np.array(parts[i])
    params, _ = curve_fit(parabola, traj[:,0], traj[:,2])
    if params[0] > 0:
        print("Parabola is opening onwards and hence invalid!")
    vx = np.sqrt(-9.8/(2*params[0]))*np.sign(parts[i][1][0] - parts[i][0][0])
    vz0 = vx*params[1]
    vz = vz0 - 9.8*parts[i][0][0]/vx 
    params, _ = curve_fit(parabola, traj[:,1], traj[:,2])
    if params[0] > 0:
        print("Parabola is opening onwards and hence invalid!")
        break
    vy = np.sqrt(-9.8/(2*params[0]))*np.sign(parts[i][1][1] - parts[i][0][1])
    vz0 = vy*params[1]
    vz_ = vz0 - 9.8*parts[i][0][1]/vy
    # print("Velocities:",vx,vy,vz,vz_)
    vs.append([vx,vy,vz])
    positions.append(parts[i][0])
    end_positions.append(parts[i][-1])
if len(vs) <= 2:
    print("Not enough segments to simulate!")
vs = np.array(vs)
positions = np.array(positions)
end_positions = np.array(end_positions)

obs, _ = env.reset()
curr_vx = vs[0,0]
curr_i = 0
t = 0.
while curr_vx > 0:
    t += (end_positions[curr_i,0] - positions[curr_i,0])/(curr_vx)
    curr_i += 1
    curr_vx = vs[curr_i,0]
curr_i -= 1
if curr_i > 0:
    env.set_next_bounce_vel(vs[curr_i])
    env.set_next_bounce_pos(positions[curr_i])
else :
    env.next_bounce_vel = None
# env.reset_ball_throw_(positions[curr_i], vs[curr_i])

# Pre-positioning phase
env.reset_ball_throw_([10.0,0.,0.1], vs[0]*0)
z_axis = np.array([1.,0.,0.])
x_axis = np.array([0.,0.,1.])
y_axis = np.cross(z_axis, x_axis)
q = R.from_matrix(np.array([x_axis,y_axis,z_axis]).T).as_quat()
pose = np.zeros(7)
pose[0] = -0

pose[2] = 0.8
pose[3:] = q
if args.init:
    # pose[0] = avg_hit_position[0]
    pose[1] = avg_hit_position[1]
    pose[2] = avg_hit_position[2]
env.set_target_pose(pose)    
for step_t in range(200):
    action,_ = model.predict(obs, deterministic=True)
    curr_t = 0
    env.reset_ball_throw_(positions[0], vs[0])
    if mini+int(curr_t) < len(player_joints):
        human_pose = player_joints[int(curr_t)+mini]
        if side == 1:
            env.draw_human_pose(human_pose[44:44+25],color=(1.,.1,.78,1.),init_i=0)
        else :
            env.draw_human_pose(human_pose[:25],color=(1.,.1,.78,1.),init_i=0)
    obs, reward, done, _, info = env.step(action)
    if args.render:
        env.render()
    time.sleep(time_sleep)

if args.pre:
    if args.gt:
        env.calc_target_racket_pose(positions[-1], vs[-1])
    else :
        z_axis = np.array([1.,0.,0.])
        x_axis = np.array([0.,0.,1.])
        y_axis = np.cross(z_axis, x_axis)
        q = R.from_matrix(np.array([x_axis,y_axis,z_axis]).T).as_quat()
        pose = np.zeros(7)
        C = np.array([0.,0.,0.8])
        pose[0] = -0.35#anticipatory_target_position[0]
            
        pose[1] = -anticipatory_target_position[1]
        # if target_id == 321 :
        #     print("Offsetted position")
        #     pose[1] -= 0.2
        pose[2] = 0.8 + 0.5*(anticipatory_target_position[2]-0.8)
        pose[:3] = C + (1-lambda_)*(pose[:3] - C)/0.9
        print("Anticipatory target position:", pose[:3])
        pose[3:] = q
        env.set_target_pose(pose)  
env.reset_ball_throw_(positions[0], vs[0])    
curr_t = 0
for step_t in range(int(t/0.01)) :
    curr_t += 0.01*FPS
    if mini+int(curr_t) < len(player_joints):
        human_pose = player_joints[int(curr_t)+mini]
        if side == 1:
            env.draw_human_pose(human_pose[44:44+25],color=(1.,.1,.78,1.),init_i=0)
        else :
            env.draw_human_pose(human_pose[:25],color=(1.,.1,.78,1.),init_i=0)
    action,_ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    if args.render:
        env.render()
    time.sleep(time_sleep)
        # print(step_t)
    
curr_i += 1
if curr_i +1>= len(vs):
    print("Not enough segments to simulate!")
# t /= 2.
# positions[curr_i][0] -= vs[curr_i][0]*t*0.5
# positions[curr_i][1] -= vs[curr_i][1]*t*0.5
# positions[curr_i][2] -= vs[curr_i][2]*t*0.5 + 0.5*9.8*(0.5*t)**2
# print(positions[curr_i])
# vs[curr_i][2] += 9.8*0.5*t
env.reset_ball_throw_(positions[curr_i], vs[curr_i])
if args.pre and not args.gt:
    env.calc_target_racket_pose(positions[-1], vs[-1],x_gantry=-0.35+(1-lambda_)*(pose[0]+0.35)/0.9)
else :
    env.calc_target_racket_pose(positions[-1], vs[-1])
env.set_next_bounce_vel(vs[curr_i+1])
env.set_next_bounce_pos(positions[curr_i+1])
env.success = False
env.achieved_target_pose = None
env.bounce_loc = None
target_poses[-1] = np.array(env.curr_target)
for step_t in range(400) :
    curr_t += 0.01*FPS
    action,_ = model.predict(obs, deterministic=True)
    if mini+int(curr_t) < len(player_joints):
        human_pose = player_joints[int(curr_t)+mini]
        if side == 1:
            env.draw_human_pose(human_pose[44:44+25],color=(1.,.1,.78,1.),init_i=0)
        else :
            env.draw_human_pose(human_pose[:25],color=(1.,.1,.78,1.),init_i=0)

    obs, reward, done, _, info = env.step(action)
    if args.render:
        env.render()
    time.sleep(time_sleep)


if env.success:
    print("Success!")
    successful += 1
else :
    print("Failure!")
    unsuccesful += 1
poses_achieved[-1] = np.array(env.achieved_target_pose)
where_landed[-1] = np.array(env.bounce_loc)
print("Success rate:", successful/(successful+unsuccesful))


if args.pre and not args.gt:
    suffix = 'pre'
elif args.gt:
    suffix = 'gt'
else :
    suffix = 'without'
# suffix = 'transition'
fps = 30  # Adjust the frames per second as needed
filename = 'videos1/mujoco_{}_{}/{}.mp4'.format(match_no, rally_no, suffix)
os.makedirs('videos1/mujoco_{}_{}'.format(match_no, rally_no), exist_ok=True)
writer = imageio.get_writer(filename, fps=fps)
for frame in env.frames:
    # print("frame")
    # print(frame.shape)
    frame_new = frame.copy()[288:, 256:-256]
    # rescale image frame_new to 2560x1440
    frame_new = np.array(frame_new)
    frame_new = np.clip(frame_new, 0, 255).astype(np.uint8)
    frame_new = cv2.resize(frame_new, (2560, 1440))
    # print(frame_new.shape)

    writer.append_data(frame_new)
writer.close()


env.close()
