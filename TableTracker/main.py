import os
import argparse
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from process import process_video  # Import the process_video function

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
    input_dir = os.path.join(root_dir, match_folder)
    output_dir = os.path.join('outputs', match_folder)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set the CUDA_VISIBLE_DEVICES environment variable
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Process all videos in the match folder
    for video_file in os.listdir(input_dir):
        if video_file.endswith(('.mp4', '.avi', '.mov')): # and f"{video_file[:-4]}_table.csv" not in os.listdir(output_dir):  # Add or remove video extensions as needed
            video_path = os.path.join(input_dir, video_file)
            output_path = os.path.join(output_dir, os.path.splitext(video_file)[0] + '_table')
            print(f"Processing {video_file} from {match_folder} on GPU {gpu_id}")
            process_video(video_path, output_path)
            print(f"Finished processing {video_file} from {match_folder}")

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
    parser = argparse.ArgumentParser(description="Process videos in match folders in parallel using specified GPUs")
    parser.add_argument("root_dir", help="Root directory containing match folders")
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[0, 1, 2, 3], help="List of GPU IDs to use")
    parser.add_argument("--workers_per_gpu", type=int, default=40, help="Number of workers per GPU")
    args = parser.parse_args()

    main(args.root_dir, args.gpu_ids, args.workers_per_gpu)