"""
NOTE: Some things to keep in mind:
- Assumes depth and RGB are spatially /and/ temporally aligned
- Assumes the visibility of *one* right hand
- If a hand is not detected, just stores 778x3 zeroes
- Uses detectron2 + ViTPose for more robust hand detection
"""

from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import json
from typing import Dict, Optional

from wilor.models import WiLoR, load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from wilor.utils.renderer import Renderer, cam_crop_to_full, old_cam_crop_to_full
from tqdm import tqdm

import open3d as o3d
import matplotlib.pyplot as plt
import imageio.v3 as iio
import av

import sys
sys.path.append('./third_party/Grounded-SAM-2')
from gsam_wrapper import GSAM2

# Detectron2 imports
from detectron2.config import LazyConfig
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

LIGHT_PURPLE=(0.25098039, 0.274117647, 0.65882353)

def load_detectron2_predictor():
    """Load Detectron2 human detector"""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # Lower threshold for better detection
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)
    return predictor

def load_vitpose_model():
    """Load ViTPose model for keypoint detection"""
    from vitpose_model import ViTPoseModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vitpose = ViTPoseModel(device)
    return vitpose

def detect_hands_detectron2(img, predictor, vitpose, kpt2d_th=0.5):
    """
    Detect hands using Detectron2 + ViTPose pipeline
    Returns: list of hand bboxes and whether they're right hands
    """
    # Convert RGB to BGR for OpenCV/Detectron2
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Detect humans
    outputs = predictor(img_bgr)
    instances = outputs["instances"]
    
    # Filter for person class (class 0 in COCO) with good confidence
    person_mask = (instances.pred_classes == 0) & (instances.scores > 0.1)
    if not person_mask.any():
        return [], []
    
    person_boxes = instances.pred_boxes[person_mask].tensor.cpu().numpy()
    person_scores = instances.scores[person_mask].cpu().numpy()
    
    # Use ViTPose for keypoint detection
    vitpose_input = [np.concatenate([person_boxes, person_scores[:, None]], axis=1)]
    vitposes_out = vitpose.predict_pose(img, vitpose_input)

    if len(vitposes_out) == 0:
        return [], []

    hand_bboxes = []
    is_right_list = []

    # Process first person only for now
    for vitposes in vitposes_out[:1]:
        left_hand_keyp = vitposes['keypoints'][-42:-21]
        right_hand_keyp = vitposes['keypoints'][-21:]

        # Process left hand
        valid = left_hand_keyp[:,2] > kpt2d_th
        if sum(valid) > 3:
            bbox = [left_hand_keyp[valid,0].min(), left_hand_keyp[valid,1].min(),
                    left_hand_keyp[valid,0].max(), left_hand_keyp[valid,1].max()]
            hand_bboxes.append(bbox)
            is_right_list.append(0)  # Left hand

        # Process right hand
        valid = right_hand_keyp[:,2] > kpt2d_th
        if sum(valid) > 3:
            bbox = [right_hand_keyp[valid,0].min(), right_hand_keyp[valid,1].min(),
                    right_hand_keyp[valid,0].max(), right_hand_keyp[valid,1].max()]
            hand_bboxes.append(bbox)
            is_right_list.append(1)  # Right hand

    return hand_bboxes, is_right_list

def project_full_img(points, cam_trans, K):
    points = points + cam_trans
    points = points / points[..., -1:]

    V_2d = (K @ points.T).T
    return V_2d[..., :-1]

def infill_hand_verts(seq):
    """
    Interpolate hole frames in a sequence of point clouds.
    """
    T = seq.shape[0]
    valid_mask = np.array([np.mean(np.abs(frame)) != 0 for frame in seq])
    hole_idxs = np.where(~valid_mask)[0]

    if hole_idxs.size == 0:
        return seq

    groups = []
    group = [hole_idxs[0]]
    for idx in hole_idxs[1:]:
        if idx == group[-1] + 1:
            group.append(idx)
        else:
            groups.append(group)
            group = [idx]
    groups.append(group)

    for group in groups:
        start_hole = group[0]
        end_hole = group[-1]
        if start_hole - 1 >= 0 and end_hole + 1 < T:
            if valid_mask[start_hole - 1] and valid_mask[end_hole + 1]:
                frame_before = seq[start_hole - 1]
                frame_after  = seq[end_hole + 1]
                num_steps = len(group) + 1
                for i, idx in enumerate(range(start_hole, end_hole + 1)):
                    alpha = (i + 1) / num_steps
                    seq[idx] = (1 - alpha) * frame_before + alpha * frame_after
    return seq

def read_depth_video(video_path):
    container = av.open(str(video_path))
    video_stream = container.streams.video[0]

    frames = []
    for i, frame in enumerate(container.decode(video_stream)):
        frames.append(frame.to_ndarray(format='gray16le'))
    frames = np.array(frames)
    container.close()
    return frames

def main():
    parser = argparse.ArgumentParser(description='WiLoR demo code with Detectron2')
    parser.add_argument('--input_folder', type=str, default="/home/sriram/.cache/huggingface/lerobot/sriramsk/human_mug_0718/videos/chunk-000/")
    parser.add_argument('--output_folder', type=str, default="/data/sriram/lerobot_extradata/sriramsk/human_mug_0718/wilor_hand_pose")
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--no_gsam2', action='store_true', help='Disable GSAM2 hand masking')
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--intrinsics_txt', default="kinect_intrinsics.txt")

    args = parser.parse_args()

    # Download and load checkpoints
    model, model_cfg = load_wilor(checkpoint_path = './pretrained_models/wilor_final.ckpt' , cfg_path= './pretrained_models/model_config.yaml')
    
    # Load Detectron2 predictor instead of YOLO
    detector = load_detectron2_predictor()
    vitpose = load_vitpose_model()
    
    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    renderer_side = Renderer(model_cfg, faces=model.mano.faces)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Initialize GSAM2 by default unless disabled
    gsam2 = None
    if not args.no_gsam2:
        print("Will use GSAM2 for hand masking")
        gsam2 = GSAM2(device=device, output_dir=Path('.'), debug=False)

    rgb_path = f"{args.input_folder}/observation.images.cam_azure_kinect.color/"
    depth_path = f"{args.input_folder}/observation.images.cam_azure_kinect.transformed_depth/"

    videos = sorted(os.listdir(rgb_path))
    visualize = args.visualize

    os.makedirs(args.output_folder, exist_ok=True)

    for vid_name in tqdm(videos):
        if visualize:
            os.makedirs(f"scaled_hand_viz/{vid_name}", exist_ok=True)

        rgb_images = iio.imread(f"{rgb_path}/{vid_name}")
        depth_images = read_depth_video(f"{depth_path}/{vid_name}".replace(".mp4", ".mkv"))
        K = np.loadtxt(args.intrinsics_txt)
        demo_verts = []

        # Same height and width
        assert rgb_images.shape[1:3] == depth_images.shape[1:3]

        for idx in tqdm(range(rgb_images.shape[0])):
            img = rgb_images[idx]
            depth = depth_images[idx].squeeze() / 1000.

            # Use Detectron2 + ViTPose for hand detection
            bboxes, is_right = detect_hands_detectron2(img, detector, vitpose)

            # Filter for right hands only (keep original logic)
            right_hand_bboxes = []
            for i, (bbox, is_r) in enumerate(zip(bboxes, is_right)):
                if is_r:  # Right hand
                    right_hand_bboxes.append(bbox)

            # We only continue if we have *one* *right* hand detected.
            if len(right_hand_bboxes) != 1:
                print(f"SKIPPING. number of right hand bboxes: {len(right_hand_bboxes)}. Total bboxes: {len(bboxes)}")
                demo_verts.append(np.zeros((778, 3)))
                if visualize:
                    plt.imsave(f"scaled_hand_viz/{vid_name}/{str(idx).zfill(5)}.png", img)
                continue

            boxes = np.array([right_hand_bboxes[0]])  # Take the first (and only) right hand
            right = np.array([1])  # Right hand
            dataset = ViTDetDataset(model_cfg, img, boxes, right, rescale_factor=args.rescale_factor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

            batch = next(iter(dataloader))
            batch = recursive_to(batch, device)

            with torch.no_grad():
                out = model(batch)

            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = K[0, 0]
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, K).detach().cpu().numpy()

            # Render the result
            verts = out['pred_vertices'][0].detach().cpu().numpy()
            joints = out['pred_keypoints_3d'][0].detach().cpu().numpy()

            is_right_batch = batch['right'][0].cpu().numpy()
            verts[:,0] = (2*is_right_batch-1)*verts[:,0]
            joints[:,0] = (2*is_right_batch-1)*joints[:,0]
            cam_t = pred_cam_t_full[0]
            kpts_2d = project_full_img(verts, cam_t, K)

            camera_translation = cam_t.copy()

            # Get hand mask if GSAM2 is enabled
            hand_mask = np.zeros((img.shape[0],img.shape[1]))
            if gsam2 is not None:
                # Use "hand" as the object to detect
                masks, scores, _, _, _, _ = gsam2.get_masks_image("hand", img)
                if masks is not None and len(masks) > 0:
                    # Take the first mask with highest confidence
                    hand_mask = masks[0][0]  # Shape: (H, W)

            tmesh = renderer.vertices_to_trimesh_using_depth(
                verts,
                camera_translation,
                depth,
                scaled_focal_length,
                img_size[0],
                mesh_base_color=LIGHT_PURPLE,
                is_right=is_right_batch,
                K=K,
                hand_mask=hand_mask,
            )

            # Could be 0, could be 777 for some reason on one instance?
            if tmesh.vertices.shape[0] != 778:
                tmesh.vertices = np.zeros((778, 3))

            if visualize:
                hand_pcd = o3d.geometry.PointCloud()
                hand_pcd.points = o3d.utility.Vector3dVector(tmesh.vertices)

                rgb_o3d = o3d.geometry.Image(img.astype(np.uint8))
                depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    rgb_o3d, depth_o3d, depth_scale=1,
                    depth_trunc=2, convert_rgb_to_intensity=False)
                intrinsic = o3d.camera.PinholeCameraIntrinsic()
                intrinsic.intrinsic_matrix = K
                scene_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

                # Draw detected bbox for visualization
                if len(right_hand_bboxes) > 0:
                    x1, y1, x2, y2 = right_hand_bboxes[0]
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                key_y = np.clip(kpts_2d[:,1].astype(int), 0, img.shape[0]-1)
                key_x = np.clip(kpts_2d[:,0].astype(int), 0, img.shape[1]-1)
                img[key_y, key_x] = (0, 0, 255)
                plt.imsave(f"scaled_hand_viz/{vid_name}/{str(idx).zfill(5)}.png", img)

            demo_verts.append(tmesh.vertices)

        demo_verts = np.array(demo_verts)
        demo_verts = infill_hand_verts(demo_verts)

        np.save(f"{args.output_folder}/{vid_name}.npy", demo_verts)

if __name__ == '__main__':
    main()
