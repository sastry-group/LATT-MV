import pickle
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time

from concurrent.futures import ProcessPoolExecutor, as_completed
from processed import ProcessedVideoLite, ProcessedVideoPartial, ProcessedVideo
from utils.rescale import gaussian_mixture_analysis


def get_rescale_factors(root_dir, match_folder, name):
    dirname = f"{root_dir}/{match_folder}/{name}"
    vid = ProcessedVideoPartial(
        dirname + f"/{name}.mp4",
        dirname,
    )
    return vid.player1_rs_3d, vid.player2_rs_3d

def save_reconstruction(root_dir, match_folder, name, render=False, verbose=False, rescale_factors=None):
    dirname = f"{root_dir}/{match_folder}/{name}"
    vid = ProcessedVideo(
        dirname + f"/{name}.mp4",
        dirname,
        verbose=verbose,
        rescale_factors=rescale_factors
    )
    if render: 
        vid.render()
    vid.save(f"recons/{match_folder}/{name}.npy")
    return vid

def process_match(match_folder, root_dir):
    full_match_path = os.path.join(root_dir, match_folder)
    print(f"Processing {match_folder}")
    
    os.makedirs(f"recons/{match_folder}", exist_ok=True)  
    
    r = []
    for video_folder in os.listdir(full_match_path):
        try:
            r1, r2 = get_rescale_factors(root_dir, match_folder, video_folder)
            print(f"Finished processing {video_folder} from {match_folder}")
            r.append(r1)
            r.append(r2)
        except Exception as e:
            print(f"Failed to process {video_folder} from {match_folder}")
            print(e)

    mu1, mu2 = gaussian_mixture_analysis(np.array(r).reshape(-1, 2))
    for video_folder in os.listdir(full_match_path):
        try:            
            save_reconstruction(root_dir, match_folder, video_folder, rescale_factors=[mu1, mu2])
            print(f"Finished processing {video_folder} from {match_folder}")
        except Exception as e:
            print(f"Failed to process {video_folder} from {match_folder}")
            print(e)

def main(root_dir):
    match_folders = [f for f in os.listdir(root_dir) if f.startswith('match')]
    match_folders.sort(key=lambda x: int(x.split('match')[1]))

    # Determine the number of CPU cores available
    max_workers = os.cpu_count()
    if max_workers is None:
        max_workers = 4  # Default to 4 if unable to determine CPU count
    
    os.makedirs("recons", exist_ok=True)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for match_folder in match_folders:
            future = executor.submit(process_match, match_folder, root_dir)
            futures.append(future)

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process match folders in parallel using CPU")
    parser.add_argument("root_dir", help="Root directory containing match folders")
    args = parser.parse_args()
    main(args.root_dir)
    
    # a, b = 26, 2
    # v = save_reconstruction("data", f"match{a}", f"match{a}_{b}", verbose=False, render=False)
    
"""
conda deactivate
conda activate ppr

find recons/match* -type f | wc -l
find recons/match* -type f | sed 's|/[^/]*$||' | sort | uniq -c
"""
