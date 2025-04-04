import os
import subprocess
import argparse
import time

def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=index,memory.free',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    
    # Parse the output and convert to dict
    gpu_memory = {}
    for line in result.strip().split('\n'):
        index, mem_free = line.strip().split(',')
        gpu_memory[int(index)] = int(mem_free)
    return gpu_memory

def start_process(match_folder, root_dir, gpu_id):
    match_number = match_folder.split('match')[1]

    # Set the CUDA_VISIBLE_DEVICES environment variable
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    command = [
        "python", "predict.py",
        f"--video_file={os.path.join(root_dir, match_folder)}",
        "--tracknet_file=finetune/TrackNet_best.pt",
        "--inpaintnet_file=finetune/InpaintNet_best.pt",
        f"--save_dir=outputs/{match_folder}",
        "--cuda=0"  # Always use cuda:0 as it's the only visible device for this process
    ]

    if "test" in root_dir:
        command.append("--output_video")

    print(f"Processing {match_folder} on GPU {gpu_id}")
    p = subprocess.Popen(command, env=env)
    return p

def main(root_dir, gpu_ids, workers_per_gpu):
    match_folders = [f for f in os.listdir(root_dir) if f.startswith('match')]
    match_folders.sort(key=lambda x: int(x.split('match')[1]))
    max_processes_per_gpu = workers_per_gpu

    # Initialize a dict to keep track of processes per GPU
    gpu_processes = {gpu_id: [] for gpu_id in gpu_ids}
    match_folder_iter = iter(match_folders)

    while True:
        # Remove completed processes and check for errors
        for gpu_id in gpu_ids:
            new_process_list = []
            for p in gpu_processes[gpu_id]:
                if p.poll() is None:
                    new_process_list.append(p)
                else:
                    if p.returncode != 0:
                        print(f"Process on GPU {gpu_id} exited with error code {p.returncode}")
            gpu_processes[gpu_id] = new_process_list

        # Check if all processes are finished and no more match folders
        all_done = True
        next_match_folder = None
        for gpu_id in gpu_ids:
            if gpu_processes[gpu_id]:
                all_done = False
                break
        if all_done:
            try:
                next_match_folder = next(match_folder_iter)
                all_done = False
            except StopIteration:
                pass
            if all_done:
                print("All processes completed.")
                break

        # Assign new tasks
        assigned = False
        for gpu_id in gpu_ids:
            if len(gpu_processes[gpu_id]) < max_processes_per_gpu:
                # Check GPU memory
                match_folder, next_match_folder = next_match_folder, None
                if match_folder is None:
                    try:
                        match_folder = next(match_folder_iter)
                    except StopIteration:
                        break
                # Start the process
                p = start_process(match_folder, root_dir, gpu_id)
                gpu_processes[gpu_id].append(p)
                assigned = True
        if not assigned:
            # If no processes were assigned, sleep for a while
            time.sleep(10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process match folders in parallel using specified GPUs")
    parser.add_argument("root_dir", help="Root directory containing match folders")
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[0, 1, 2, 3], help="List of GPU IDs to use")
    parser.add_argument("--workers_per_gpu", type=int, default=2, help="Number of workers per GPU")
    args = parser.parse_args()
    main(args.root_dir, args.gpu_ids, args.workers_per_gpu)
