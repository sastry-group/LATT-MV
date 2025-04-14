# LATTE-MV: Learning to Anticipate Table Tennis Hits from Monocular Videos

<div align="center">

[[Website]](https://sastry-group.github.io/LATTE-MV/)        [[PDF]](https://arxiv.org/pdf/2503.20936)        [[Dataset]](https://huggingface.co/datasets/ember-lab-berkeley/LATTE-MV/tree/main)

<img src="teaser.gif" width="600px"/>

</div>

This repository contains the code (dataset setup, training and MuJoCo setup) for the paper "LATTE-MV: Learning to Anticipate Table Tennis Hits from Monocular Videos".

LATTE-MV is a ***scalable system*** for reconstructing monocular videos of table tennis matches in 3D. This data is used to train a large transformer capable of anticipating opponent actions using ***conformal prediction*** for uncertainty estimation. Reconstructed trajectories are simulated in MuJoCo with a robotic system on receiving end capable of returning balls with 59.0% accuracy as compared to 49.9% with no anticipation

## Table of Contents

1. [Setup](#setup)
2. [Download dataset](#download-dataset)
3. [Phase 1: Reconstructing gameplay in 3D](#phase-1-reconstructing-gameplay-in-3d)
4. [Phase 2: Train transformer to anticipate](#phase-2-train-transformer-to-anticipate)
5. [Phase 3: Validate in MuJoCo](#phase-3-validate-in-mujoco)
6. [Visualization](#visualization)
7. [Running on your own video](#running-on-your-own-video)
8. [Citing this Work](#bibtex)

## Setup

Make sure you have [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed. All the conda environments specific to running each step of the pipeling are given in their respective directories. Run the following to install them all:

```bash
conda env create -f RallyClipper/env.yml
conda env create -f BallTracker/env.yml
conda env create -f TableTracker/env.yml
conda env create -f GenerativeModel/env.yml
conda env create -f Visualize/env.yml
```

Install mujoco for phase 3 with
```bash
pip install mujoco
```



## Download dataset
Download the dataset from [here](https://huggingface.co/datasets/ember-lab-berkeley/LATTE-MV/resolve/main/release_data.zip?download=true) and unzip it. The dataset should be in the following format:

```
release_data/
├── match1/
│   ├── match1_3/
│   │   ├── match1_3.mp4               # Original clipped video
│   │   ├── match1_3_ball.csv          # Ball tracking results
│   │   ├── match1_3_keypoints_2d.npy  # 2D keypoints for detected humans
│   │   ├── match1_3_keypoints_3d.npy  # 3D keypoints for detected humans
│   │   ├── match1_3_metadata.json     # Metadata for detected humans
│   │   ├── match1_3_paddle.csv        # Paddle detection results
│   │   ├── match1_3_table.npy         # Detected table corners
│   │   ├── match1_3_recon.npy         # Reconstructed scene information
│   ├── ...
└── ...
```

NOTE: match{match_id}_{rally_id}_recon.npy consists of a 3d numpy array of shape (no of frames, 91, 3) with each row containing scene information for a frame in that timestep. The columns are as follows:-
- metadata ((fps, no of frames, no of frames usable) in first row, (dist from player 1 hand, dist from player 2 hand, player 1 hand id) in second row, (player 2 hand id, 0, 0) in third row and (0,0,0) in all other rows), hit info ((is contact with player 1 racket?, is contact with player 2 racket?, is contact with table?)), next 44 columns contain 3d feature points of player 1, next 44 columns contain 3d feature points of player 2, next columns contain ball position in 3d 

## Phase 1: Reconstructing gameplay in 3D

Follow these instructions to setup and run the full pipeline. 

> [!IMPORTANT]
> Please note that the labels generated from all of these steps are already in the downloaded dataset so you can skip this and need not run all of these again.

### Setting up the Human Pose Tracker

For the following steps navigate to the HumanPoseTracker folder in this repo.
```bash
cd HumanPoseTracker
```

### Human Pose Tracker environment setup:

```bash
conda env create --file env.yml
conda activate pose
conda remove cuda-version
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install -n pose gxx_linux-64
```

### PHALP setup:

Run:
```bash
pip install git+https://github.com/brjathu/PHALP.git
```
Then replace phalp/trackers/PHALP.py with the local version in this repo. To help find the file, open a python shell and run:
```bash
import phalp
phalp.__file__
```

### SMPL Mesh

Make sure to find you have the `phalp/3D/models/smpl/SMPL_NEUTRAL.pkl` files


### Downloading Ball Tracking Model
For the following steps navigate to the HumanPoseTracker folder in this repo.
```bash
cd BallTracker
```

Once there run the following commands.
```bash
conda deactivate
conda activate ball
pip install gdown
rm -r finetune
rm -r ckpts
gdown 1b7esQo0NNkFutR5ScC1KKWW0zyUGjZ1E
gdown 1sK9H5_5kbHegb-_b-5PuDeifXNQQeMHv
unzip finetune.zip
unzip ckpts.zip
rm finetune.zip
rm ckpts.zip
conda deactivate
```


### Run the pipeline 

To run the pipeline, place all videos in RallyClipper/matches (follow the match{i}.mp4 notation for consistency) and run the following. 

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh --gpu_ids "0 1 2 3 4 5" all
```

You can replace "0 1 2 3 4 5" with the ids of the gpus on your machine. This will generate outputs.zip file . Unzip this file to get the outputs in the same format as the downloaded dataset. You can copy this folder back to release_data/ directory to append to the dataset 



## Phase 2: Train transformer to anticipate

For the following steps navigate to the GenerativeModel directory in this repo.

```bash
cd GenerativeModel
```

### Download additional test dataset

### Setting up the environment

Setup and activate the conda environment

```bash
conda env create -f env.yml
conda activate gen
```

### Download additional test data

The test data is not in the original dataset. You can download it through
```bash
wget https://huggingface.co/datasets/ember-lab-berkeley/LATTE-MV/resolve/main/test_data.zip
unzip test_data.zip
rm test_data.zip
```

### Download pretrained models
You can download the pretrained models through
```bash
wget https://huggingface.co/datasets/ember-lab-berkeley/LATTE-MV/resolve/main/models.zip
unzip models.zip
rm models.zip
```

### (optional) Training the model

You can train the model(s) for conformal prediction with
```bash
python3 train.py 
```

This will train transformer model(s) for anticipatory ball trajectory prediction with default settings. You can change setting in the header of the script. The model will be saved in the `models` directory. Please note that the models are already trained and available in the `models` directory. You can skip this step if you want to use the pretrained models.

### Using the trained model(s) for inference

You can run the inference script on a specific match, rally and a given hit segment. The script will generate a video with the predicted trajectory and the ground truth trajectory in rec.gif. You can run the script with

```bash
python3 inference.py --match <match_id> --rally <rally_id> --hit_id <hit_id>
```

where `<match_id>` is the match id, `<rally_id>` is the rally id, `<hit_id>` is the hit segment id where hit_id of 0 means the transformer will output the future predicted trajectory of the player when he return the 1st shot after serving, hit_id of 1 means the predicted trajectory is of the 2nd shot and so on. This will also save the predicted trajectories from individual transformers in release_data/match{match_id}/match{match_id}_{rally_id}/match{match_id}_{rally_id}_pred.npy. You can change the video name in the script to run inference on other videos. These predicted trajectories can be better visualized with scripts in visualize/ directory explained later.

## Phase 3: Validate in MuJoCo
For the following steps navigate to the MujocoRecon directory in this repo.

```bash
cd MujocoRecon
```

You can run the evaluation scipt with
```bash
python3 eval.py [--pre] [--gt] --exp_name <exp_name>
```

Use the --pre flag to indicate that you want to use the predicted trajectories from the previous step to pre-position the robot. Use the --gt flag to indicate that you want to use the ground truth trajectories. The script will generate a pkl file with the results of the evaluation. Change the pkl file name under eval_stats.py and you can get the statistics with

```bash
python3 eval_stats.py
```

The statistics will be saved under the same file name as pkl with txt extension. You can also visualize the results for a specific example

```bash
python3 eval_generate_videos.py --idx <idx>
```

This will create and save videos for the 3 cases:- without anticipation, with anticipation and oracle under mujoco_{idx} directory. You can change the idx to visualize other examples. 

## Visualization
For the following steps navigate to the visualize directory in this repo.

```bash
cd Visualize
```

Setup and activate the conda environment

```bash
conda env create -f env.yml
conda activate visualize
```

### Visualizing the reconstructed gameplay and predictions (phase 1 and phase 2)
You can visualize the reconstructed gameplay with the following command. You can add comma separated options to this with --visualizations option:- Comma-separated list of projection keys to compute b_orig (ground truth ball position), b_reconstructed (reconstructed ball position), racket, table, players, grid_world.

```bash
python3 visualize_recon.py --match <match_id> --rally <rally_id> --projections <projections list>
```

You can visualize the predicted trajectory from phase 2 with the following command. Before running this command, make sure that you have run anticipatory_pred.py on that rally from previous step. 

```bash
python3 visualize_pred.py --match <match_id> --rally <rally_id> --projections <projections list>
```

## Running on your own video

To run the pipeline on your own video, you can follow the same steps as above. Make sure to place your video in the RallyClipper/matches directory and follow the same naming convention. You can then run the pipeline with the same command as above. The output will be saved in the same format as the downloaded dataset. You can then use this output to train the transformer model and validate if anticipation helps in MuJoCo.

## Bibtex

```bibtex
@article{etaat2025lattemv,
        title={LATTE-MV: Learning to Anticipate Table Tennis hits from Monocular Videos},
        author={Etaat, Daniel and Kalaria, Dvij and Rahmanian, Nima and Sastry, Shankar},
        booktitle={2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2025},
        organization={IEEE}
      }
```
