import os
import subprocess
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    
    # Parse the output and convert to integers
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    return gpu_memory

def process_match(match_folder, root_dir, gpu_id):
    match_number = match_folder.split('match')[1]

    # Set the CUDA_VISIBLE_DEVICES environment variable
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["EGL_DEVICE_ID"] = str(gpu_id)
    env["PYOPENGL_PLATFORM"] = "egl"

    command = [
        "python", "track.py",
        f"video.source={os.path.join(root_dir, match_folder)}",
        "device=cuda:0",  # Always use cuda:0 as it's the only visible device for this process
        "video.start_frame=-1",
        "video.end_frame=-1",
        f"video.output_dir={match_folder}",
        "base_tracker=pose",
        "phalp.low_th_c=0.8",
        "phalp.small_w=25",
        "phalp.small_h=50",
    ]

    print(f"Processing {match_folder} on GPU {gpu_id}")
    subprocess.run(command, check=True, env=env)
    print(f"Finished processing {match_folder}")

def main(root_dir, gpu_ids, workers_per_gpu):
    match_folders = [f for f in os.listdir(root_dir) if f.startswith('match')]
    match_folders.sort(key=lambda x: int(x.split('match')[1]))

    total_workers = len(gpu_ids) * workers_per_gpu
    
    with ProcessPoolExecutor(max_workers=total_workers) as executor:
        futures = []
        for i, match_folder in enumerate(match_folders):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            future = executor.submit(process_match, match_folder, root_dir, gpu_id)
            futures.append(future)
            time.sleep(10)

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process match folders in parallel using specified GPUs")
    parser.add_argument("root_dir", help="Root directory containing match folders")
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[0, 1, 2, 3], help="List of GPU IDs to use")
    parser.add_argument("--workers_per_gpu", type=int, default=2, help="Number of workers per GPU")
    args = parser.parse_args()

    main(args.root_dir, args.gpu_ids, args.workers_per_gpu)