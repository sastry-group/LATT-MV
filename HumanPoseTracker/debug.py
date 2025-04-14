from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hmr2.utils.renderer import Renderer, cam_crop_to_full
from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

parser = argparse.ArgumentParser(description='HMR2 demo code')
parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
parser.add_argument('--img_folder', type=str, default='example_data/images', help='Folder with input images')
parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder to save rendered results')
parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
parser.add_argument('--top_view', dest='top_view', action='store_true', default=False, help='If set, render top view also')
parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=False, help='If set, render all people together also')
parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
parser.add_argument('--detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
parser.add_argument('--device', type=str, default='cuda:0', choices=['cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cpu'])

args = parser.parse_args()

# Download and load checkpoints
download_models(CACHE_DIR_4DHUMANS)
model, model_cfg = load_hmr2(args.checkpoint)

# Setup HMR2.0 model
device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)
model.eval()

# Load detector
if args.detector == 'vitdet':
    from detectron2.config import LazyConfig
    import hmr2
    cfg_path = Path(hmr2.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg, device=args.device)
elif args.detector == 'regnety':
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
    detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
    detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
    detector       = DefaultPredictor_Lazy(detectron2_cfg, device=args.device)

# Setup the renderer
renderer = Renderer(model_cfg, faces=model.smpl.faces)

# Make output directory if it does not exist
os.makedirs(args.out_folder, exist_ok=True)

# Load image
img_path = Path(args.img_folder).glob('*.png').__next__()
img_cv2 = cv2.imread(str(img_path))

# Detect humans in image
det_out = detector(img_cv2)
det_instances = det_out['instances']
valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
boxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

# Run HMR2.0 on all detected humans
dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

all_verts = []
all_cam_t = []

# Run the model each image
positions = []
batches = list(dataloader)
for k in range(len(batches)):
    batch = batches[k]
    batch = recursive_to(batch, device)
    with torch.no_grad():
        out = model(batch)
    pred_cam = out['pred_cam']
    box_center = batch["box_center"].float()
    box_size = batch["box_size"].float()
    img_size = batch["img_size"].float()
    scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
    pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

    # Get filename from path img_path
    img_fn, _ = os.path.splitext(os.path.basename(img_path))
    
    extra = False
    if extra:
        n = 0
        person_id = int(batch['personid'][n])
        white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
        input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
        input_patch = input_patch.permute(1,2,0).numpy()

        # Render the result
        regression_img, scene, mesh, pyrenderer, camera_pose = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                out['pred_cam_t'][n].detach().cpu().numpy(),
                                batch['img'][n],
                                mesh_base_color=LIGHT_BLUE,
                                scene_bg_color=(1, 1, 1),
        )

        final_img = np.concatenate([input_patch, regression_img], axis=1)
        cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

    # Plotting
    c, s = box_center.detach().cpu().numpy(), box_size.detach().cpu().numpy()[0]
    points = out['pred_keypoints_2d'].detach().cpu().numpy()[0]
    points = points*s + c
    positions.append(points[20])
    positions.append(points[23])

positions = np.array(positions)    
plt.figure()
plt.imshow(img_cv2)
plt.scatter(positions[:, 0], positions[:, 1], color='red', marker='x')
plt.savefig(os.path.join(args.out_folder, f'{img_fn}.png'))
    
# python -i debug.py --img_folder example_data/ppr --out_folder ppr_out --batch_size=48 --save_mesh --full_frame --device=cuda