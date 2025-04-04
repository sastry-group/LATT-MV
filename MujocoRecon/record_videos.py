import time
import numpy as np
import pickle
import os

# make directory output
os.makedirs("videos1", exist_ok=True)

# load 0111.txt file for indices
# indices = np.loadtxt("is_011.txt")
indices = np.loadtxt("rerun_inds.txt")
with open("targets.pkl", "rb") as f:
    targets = pickle.load(f)

mr_indices = []
for ind in indices:
    i = int(ind)
    target = targets[i]
    ball_trajectory, switch_timestep, anticipatory_target_position, filename = target
    match_no = int(filename.split("h")[-1].split("_")[0])
    rally_no = int(filename.split("_")[-1].split(".")[0])
    # mr_indices.append([match_no, rally_no])
    os.system("python eval_generate_videos.py --ind {} --render".format(i))
    os.system("python eval_generate_videos.py --pre --ind {} --render".format(i))
    os.system("python eval_generate_videos.py --pre --gt --ind {} --render".format(i))
    # exit(0)

np.savetxt("mr_indices.txt", mr_indices)
print("Saved mr_indices.txt")