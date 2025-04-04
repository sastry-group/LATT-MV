import os
import argparse
import time
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from inference import *
import torch

def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    
    # Parse the output and convert to integers
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    return gpu_memory

def process_video(video_file, args, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Processing {video_file} on GPU {gpu_id}")
    video_path = os.path.join(args.root_dir, video_file)
    model = CNNVideoFrameClassifier(args.width, args.height)
    model.load_state_dict(torch.load('best_model.pth', map_location=f'cuda', weights_only=True))
    run_inference(model, video_path, args.output_dir, args.width, args.height, args.batch_size, 0)
    print(f"Finished processing {video_file}")

def main(args):
    os.makedirs(args.output_dir, exist_ok=True) # Ensure output folder exists
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']  # Add or remove extensions as needed
    video_files = [f for f in os.listdir(args.root_dir) if os.path.splitext(f)[1].lower() in video_extensions]
    if not video_files:
        raise ValueError(f"No video files found in {args.root_dir}")
    else:
        print(f"Found {len(video_files)} video files.")

    total_workers = len(args.gpu_ids) * args.workers_per_gpu

    with ProcessPoolExecutor(max_workers=total_workers) as executor:
        futures = []
        for i, video_file in enumerate(video_files):
            gpu_id = args.gpu_ids[i % len(args.gpu_ids)]
            future = executor.submit(process_video, video_file, args, gpu_id)
            futures.append(future)
            time.sleep(10)

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Run inference on videos in a folder using a trained model.")
    parser.add_argument("root_dir", type=str, help="Path to the folder containing input video files")
    parser.add_argument("output_dir", type=str, help="Folder to save the output video segments")
    parser.add_argument("--width", type=int, default=WIDTH, help="Frame width for model input")
    parser.add_argument("--height", type=int, default=HEIGHT, help="Frame height for model input")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--gpu_ids", nargs='+', type=int, default=[0, 1, 2, 3], help="List of GPU IDs to use")
    parser.add_argument("--workers_per_gpu", type=int, default=1, help="Number of workers per GPU")
    args = parser.parse_args()
    main(args)