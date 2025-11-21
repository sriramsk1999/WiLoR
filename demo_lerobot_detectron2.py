"""
NOTE: Some things to keep in mind:
- Assumes depth and RGB are spatially /and/ temporally aligned
- Assumes the visibility of *one* right hand
- If a hand is not detected, just stores 778x3 zeroes
- Uses detectron2 + ViTPose for more robust hand detection
"""

from pathlib import Path
import torch
import torch.nn as nn
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

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

LIGHT_PURPLE=(0.25098039, 0.274117647, 0.65882353)

def transform_to_world(points_cam, T_world_from_cam):
    """Transform points from camera frame to world frame.

    Args:
        points_cam: (B, N, 3) - points in camera frame
        T_world_from_cam: (B, 4, 4) - transformation matrix

    Returns:
        points_world: (B, N, 3) - points in world frame
    """
    B, N, _ = points_cam.shape
    # Convert to homogeneous coordinates
    ones = np.ones((B, N, 1))
    points_hom = np.concatenate([points_cam, ones], axis=-1)  # (B, N, 4)

    # Apply transformation: (B, 4, 4) @ (B, N, 4) -> (B, N, 4)
    points_world_hom = np.einsum("bij,bnj->bni", T_world_from_cam, points_hom)

    # Convert back to 3D
    points_world = points_world_hom[:, :, :3]  # (B, N, 3)

    return points_world

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

@torch.enable_grad()
def smooth_bbox_sequence(bboxes, device='cuda', T=1000, w=0.05, w_c=0.02, w_s=10):
    """
    Temporally smooth bounding box sequences using gradient descent optimization.
    Minimizes acceleration (jerkiness) while staying close to original detections.
    From HAPTIC

    Args:
        bboxes: List of bbox arrays (T, 4) where each bbox is [x1, y1, x2, y2]
        device: Device for computation
        T: Number of optimization iterations
        w: Regularization weight
        w_c: Center acceleration weight
        w_s: Scale acceleration weight

    Returns:
        Smoothed bboxes
    """
    if len(bboxes) < 3:  # Need at least 3 frames for acceleration
        return bboxes

    # Convert bboxes to center and scale format
    bboxes_array = np.array(bboxes)  # (T, 4)
    x1, y1, x2, y2 = bboxes_array.T

    # Convert to center and scale
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    scale_w = x2 - x1
    scale_h = y2 - y1

    centers = np.stack([center_x, center_y], axis=1)  # (T, 2)
    scales = np.stack([scale_w, scale_h], axis=1)     # (T, 2)

    # Convert to tensors
    center = torch.FloatTensor(centers).to(device)
    scale = torch.FloatTensor(scales).to(device)
    dcenter = nn.Parameter(torch.zeros_like(center))
    dscale = nn.Parameter(torch.zeros_like(scale))

    optimizer = torch.optim.AdamW([dcenter, dscale], lr=1e-3)

    for t in range(T):
        cur_center = center + dcenter
        cur_scale = scale + dscale

        # acceleration
        center_diff = cur_center[1:] - cur_center[:-1]
        scale_diff = cur_scale[1:] - cur_scale[:-1]
        center_acc = center_diff[1:] - center_diff[:-1]
        scale_acc = scale_diff[1:] - scale_diff[:-1]  # (T-2, 2)

        loss_acc = w_c * center_acc.norm() + w_s * scale_acc.norm()
        loss_reg = w * (w_c * dcenter.norm() + w_s * dscale.norm())
        loss = loss_acc + loss_reg

        if t % 100 == 0:
            print(f"Smoothing bbox t={t}, loss={loss.item():.6f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Convert back to bbox format
    smoothed_centers = (center + dcenter).cpu().detach().numpy()
    smoothed_scales = (scale + dscale).cpu().detach().numpy()

    center_x, center_y = smoothed_centers.T
    scale_w, scale_h = smoothed_scales.T

    # Convert back to x1, y1, x2, y2
    x1_smooth = center_x - scale_w / 2
    y1_smooth = center_y - scale_h / 2
    x2_smooth = center_x + scale_w / 2
    y2_smooth = center_y + scale_h / 2

    smoothed_bboxes = []
    for i in range(len(bboxes)):
        smoothed_bboxes.append([x1_smooth[i], y1_smooth[i], x2_smooth[i], y2_smooth[i]])

    return smoothed_bboxes

def detect_and_smooth_hands_sequence(rgb_images, detector, vitpose, device='cuda'):
    """
    Detect hands across entire video sequence and apply temporal smoothing
    """
    print("Detecting hands across video sequence...")
    all_detections = []

    # First pass: detect hands in all frames
    for idx in tqdm(range(len(rgb_images)), desc="Detecting hands"):
        img = rgb_images[idx]
        bboxes, is_right = detect_hands_detectron2(img, detector, vitpose)

        # Filter for right hands only
        right_hand_bboxes = []
        for bbox, is_r in zip(bboxes, is_right):
            if is_r:  # Right hand
                right_hand_bboxes.append(bbox)

        all_detections.append(right_hand_bboxes)

    # Extract sequences where we have exactly one right hand
    valid_frames = []
    valid_bboxes = []

    for idx, detections in enumerate(all_detections):
        if len(detections) == 1:
            valid_frames.append(idx)
            valid_bboxes.append(detections[0])

    if len(valid_bboxes) < 3:
        print("Not enough valid detections for smoothing")
        return all_detections

    print(f"Smoothing {len(valid_bboxes)} valid detections...")

    # Apply temporal smoothing to valid detections
    smoothed_bboxes = smooth_bbox_sequence(valid_bboxes, device=device)

    # Create final detection list with smoothed bboxes
    final_detections = []
    smooth_idx = 0

    for idx in range(len(rgb_images)):
        if idx in valid_frames:
            final_detections.append([smoothed_bboxes[smooth_idx]])
            smooth_idx += 1
        else:
            final_detections.append([])  # No detection

    return final_detections

def main():
    parser = argparse.ArgumentParser(description='WiLoR demo code with Detectron2')
    parser.add_argument('--input_folder', type=str, default="/home/sriram/.cache/huggingface/lerobot/sriramsk/human_mug_0718/videos/chunk-000/")
    parser.add_argument('--output_folder', type=str, default="/data/sriram/lerobot_extradata/sriramsk/human_mug_0718/wilor_hand_pose")
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--no_gsam2', action='store_true', help='Disable GSAM2 hand masking')
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--calibration_json', default="aloha_calibration/calibration_multiview.json")
    parser.add_argument('--cam_names', nargs='+', default=["cam_azure_kinect_front", "cam_azure_kinect_back"], help='List of camera names (tries in order, falls back if no hand detected)')

    args = parser.parse_args()

    with open(args.calibration_json) as f:
        all_calibration_data = json.load(f)
        # Load calibration for all cameras
        calibration_data = {cam_name: all_calibration_data[cam_name] for cam_name in args.cam_names}

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

    # Load paths and calibration for all cameras
    camera_data = {}
    for cam_name in args.cam_names:
        rgb_path = f"{args.input_folder}/observation.images.{cam_name}.color/"
        depth_path = f"{args.input_folder}/observation.images.{cam_name}.transformed_depth/"
        K = np.loadtxt(calibration_data[cam_name]["intrinsics"])
        T_world_from_cam = np.loadtxt(calibration_data[cam_name]["extrinsics"])
        camera_data[cam_name] = {
            'rgb_path': rgb_path,
            'depth_path': depth_path,
            'K': K,
            'T_world_from_cam': T_world_from_cam
        }

    # Use first camera to get video list (assuming all cameras have same videos)
    videos = sorted(os.listdir(camera_data[args.cam_names[0]]['rgb_path']))
    visualize = args.visualize

    os.makedirs(args.output_folder, exist_ok=True)

    for vid_name in tqdm(videos):
        if os.path.exists(f"{args.output_folder}/{vid_name}.npy"): continue
        if visualize:
            os.makedirs(f"scaled_hand_viz/{vid_name}", exist_ok=True)

        # Load videos from all cameras
        camera_videos = {}
        for cam_name in args.cam_names:
            rgb_images = iio.imread(f"{camera_data[cam_name]['rgb_path']}/{vid_name}")
            depth_images = read_depth_video(f"{camera_data[cam_name]['depth_path']}/{vid_name}".replace(".mp4", ".mkv"))
            assert rgb_images.shape[1:3] == depth_images.shape[1:3]
            camera_videos[cam_name] = {
                'rgb': rgb_images,
                'depth': depth_images
            }

        # Get number of frames (assume all cameras have same number)
        num_frames = camera_videos[args.cam_names[0]]['rgb'].shape[0]
        demo_verts = []
        frame_cameras = []  # Track which camera was used for each frame

        # Apply temporal smoothing to hand detections for all cameras
        print("Applying temporal smoothing to hand detections for all cameras...")
        all_smoothed_detections = {}
        for cam_name in args.cam_names:
            print(f"Processing camera: {cam_name}")
            smoothed_detections = detect_and_smooth_hands_sequence(
                camera_videos[cam_name]['rgb'], detector, vitpose, device
            )
            all_smoothed_detections[cam_name] = smoothed_detections

        for idx in tqdm(range(num_frames)):
            # Try cameras in order until we find a detection
            selected_cam = None
            right_hand_bboxes = None

            for cam_name in args.cam_names:
                detections = all_smoothed_detections[cam_name][idx]
                if len(detections) == 1:
                    selected_cam = cam_name
                    right_hand_bboxes = detections
                    break

            # If no camera detected a hand, skip this frame
            if selected_cam is None:
                demo_verts.append(np.zeros((778, 3)))
                frame_cameras.append(None)
                if visualize:
                    # Visualize the primary camera
                    img = camera_videos[args.cam_names[0]]['rgb'][idx]
                    plt.imsave(f"scaled_hand_viz/{vid_name}/{str(idx).zfill(5)}.png", img)
                continue

            # Use the selected camera's data
            img = camera_videos[selected_cam]['rgb'][idx]
            depth = camera_videos[selected_cam]['depth'][idx].squeeze() / 1000.
            K = camera_data[selected_cam]['K']
            T_world_from_cam = camera_data[selected_cam]['T_world_from_cam']

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
                selected_cam = None

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
            frame_cameras.append(selected_cam)

        demo_verts = np.array(demo_verts)

        # NOTE: Transform to world frame BEFORE infilling
        # This ensures interpolation happens in a consistent coordinate frame
        demo_verts_world = np.zeros_like(demo_verts)
        for idx in range(len(demo_verts)):
            if frame_cameras[idx] is None:
                # No detection - keep as zeros
                demo_verts_world[idx] = demo_verts[idx]
            else:
                # Transform using the camera that was used for this frame
                T_world_from_cam = camera_data[frame_cameras[idx]]['T_world_from_cam']
                demo_verts_world[idx] = transform_to_world(demo_verts[idx:idx+1], T_world_from_cam[None])[0]

        # Now infill in world coordinates
        demo_verts_world = infill_hand_verts(demo_verts_world)

        # Handle edge cases: first N frames and last N frames that are still zeros
        valid_mask = np.array([np.mean(np.abs(frame)) != 0 for frame in demo_verts_world])

        # Handle leading zeros: copy from first non-zero frame
        if not valid_mask[0] and valid_mask.any():
            first_valid_idx = np.where(valid_mask)[0][0]
            demo_verts_world[:first_valid_idx] = demo_verts_world[first_valid_idx]

        # Handle trailing zeros: copy from last non-zero frame
        if not valid_mask[-1] and valid_mask.any():
            last_valid_idx = np.where(valid_mask)[0][-1]
            demo_verts_world[last_valid_idx+1:] = demo_verts_world[last_valid_idx]

        if (demo_verts_world.mean(1).mean(1) == 0).any():
            raise Exception("Some elements where hand pose could not be estimated or interpolated, please check.")
        np.save(f"{args.output_folder}/{vid_name}.npy", demo_verts_world)

if __name__ == '__main__':
    main()
