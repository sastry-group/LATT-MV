from mujoco_env import KukaTennisEnv
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

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)
time_sleep = 0.

def parabola(x, a, b, c):
    return a * x**2 + b * x + c

successful = 0
unsuccesful = 0

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='oracle', help='Name of experiment')
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
print(len(targets))
# exit(0)

for target in tqdm.tqdm(targets[1047:1048]):
    try:
        where_landed.append(None)
        target_poses.append(None)
        poses_achieved.append(None)
        ball_trajectory, switch_timestep, anticipatory_target_position, filename = target
        print(filename)
        ball_trajectory = np.array(ball_trajectory)
        ball_trajectory[:,0] = -ball_trajectory[:,0]
        ball_trajectory[:,0] += 1.5
        ball_trajectory[:,1] *= -1
        ball_trajectory[:,2] -= 0.1
        anticipatory_target_position[0] = -anticipatory_target_position[0]
        anticipatory_target_position[0] += 1.5
        anticipatory_target_position[2] -= 0.1
        print(ball_trajectory)
        bounce_points = []
        for i in range(switch_timestep+1,len(ball_trajectory)-1):
            if (ball_trajectory[i,2] - ball_trajectory[i-1,2]) < 0 and (ball_trajectory[i+1,2] - ball_trajectory[i,2]) > 0:
                bounce_points.append(i)
        if len(bounce_points) != 1:
            print("Too many bounce points")
            continue
        else:
            bounce_point = bounce_points[0]
            if ball_trajectory[bounce_point][0] < 1.5-1.365:
                print("Table edge")
                continue
        
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
        # print(parts)
        vs = []
        positions = []
        end_positions = []
        if len(parts) <= 2:
            print("Not enough parts to get a segment!")
            continue
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
            continue
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
                pose[2] = 0.8 + 0.5*(anticipatory_target_position[2] - 0.8)
                pose[:3] = C + (1-lambda_)*(pose[:3] - C)/0.9
                print("Anticipatory target position:", pose[:3])
                pose[3:] = q
                env.set_target_pose(pose)  
        env.reset_ball_throw_(positions[0], vs[0])    
        
        for step_t in range(int(t/0.01)) :
            action,_ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            if args.render:
                env.render()
            time.sleep(time_sleep)
                # print(step_t)
            
        curr_i += 1
        if curr_i +1>= len(vs):
            print("Not enough segments to simulate!")
            continue
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
            action,_ = model.predict(obs, deterministic=True)
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
        poses_achieved[-1] = np.array(env.achieved_target_pose).copy()
        where_landed[-1] = np.array(env.bounce_loc).copy()
        print(np.array(env.bounce_loc))
        print("Success rate:", successful/(successful+unsuccesful))
        
    except Exception as e:
        print("Exception:",e)
        continue
print("Success rate:", successful/(successful+unsuccesful))
print(where_landed[1111:1114])
#Save the results (Successful and unsuccessful) in a text file
with open(args.exp_name + ".txt", "w") as f:
    f.write("Successful: " + str(successful) + "\n")
    f.write("Unsuccessful: " + str(unsuccesful) + "\n")
    f.write("Success rate: " + str(successful/(successful+unsuccesful)) + "\n")
    f.write("Invalid: " + str(len(targets) - successful - unsuccesful) + "\n")

# Save the target poses and achieved poses in a pickle file
with open(args.exp_name + ".pkl", "wb") as f:
    pickle.dump((target_poses, poses_achieved, where_landed), f)

env.close()
