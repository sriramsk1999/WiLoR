"""
- Process data in zarr to get hand detection and pose estimation
- The zarr data is generated after running the code in lfd3d-system on foxglove

NOTE: Some things to keep in mind:
- Assumes depth and RGB are spatially /and/ temporally aligned
- Assumes the visibility of *one* right hand
- If a hand is not detected, just stores 778x3 zeroes
- Definitely needs to be refactored into a more readable and robust script
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
from ultralytics import YOLO
import zarr
from tqdm import tqdm

import open3d as o3d
import matplotlib.pyplot as plt

import sys
sys.path.append('./third_party/Grounded-SAM-2')
from gsam_wrapper import GSAM2

LIGHT_PURPLE=(0.25098039, 0.274117647, 0.65882353)


def infill_hand_verts(demo_name, seq):
    """
    Interpolate hole frames in a sequence of point clouds.
    """
    T = seq.shape[0]
    valid_mask = np.array([np.mean(np.abs(frame)) != 0 for frame in seq])

    hole_idxs = np.where(~valid_mask)[0]
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

def main():
    parser = argparse.ArgumentParser(description='WiLoR demo code')
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--no_gsam2', action='store_true', help='Disable GSAM2 hand masking')
    parser.add_argument('--visualize', action='store_true', default=False)

    args = parser.parse_args()

    # Download and load checkpoints
    model, model_cfg = load_wilor(checkpoint_path = './pretrained_models/wilor_final.ckpt' , cfg_path= './pretrained_models/model_config.yaml')
    detector = YOLO('./pretrained_models/detector.pt')
    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    renderer_side = Renderer(model_cfg, faces=model.mano.faces)
    
    device   = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model    = model.to(device)
    detector = detector.to(device)
    model.eval()

    # Initialize GSAM2 by default unless disabled
    gsam2 = None
    if not args.no_gsam2:
        print("Will use GSAM2 for hand masking")
        gsam2 = GSAM2(device=device, output_dir=Path('.'), debug=False)

    root = zarr.group(args.input_folder)
    RGB_KEY, DEPTH_KEY, CAM_KEY = "_rgb_image_rect", "_depth_registered_image_rect", "_rgb_camera_info"
    visualize = args.visualize

    for demo_name in tqdm(root.keys()):
        demo = root[demo_name]
        if "_puppet_right_joint_states" in demo.keys() or "_follower_right_joint_states" in demo.keys():
            continue # robot demo

        if "_rgb_image_rect" not in demo.keys():
            continue # old demo, not being used anymore.


        if "gripper_pos" in demo.keys():
            # del demo["gripper_pos"]
            # Already generated
            continue

        if visualize:
            os.makedirs(f"scaled_hand_viz/{demo_name}", exist_ok=True)

        rgb_images = np.asarray(demo[RGB_KEY]["img"])
        depth_images = np.asarray(demo[DEPTH_KEY]["img"])
        K = np.asarray(demo[CAM_KEY]["k"])[0]

        # Same height and width
        assert rgb_images.shape[1:3] == depth_images.shape[1:3]

        num_images = min(rgb_images.shape[0], depth_images.shape[0])
        demo_verts = []

        for idx in tqdm(range(num_images)):
            img = rgb_images[idx]
            depth = depth_images[idx].squeeze() / 1000.

            detections = detector(img, conf = 0.3, verbose=False)[0]
            bboxes    = []
            is_right  = []
            for det in detections:
                Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
                is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
                bboxes.append(Bbox[:4].tolist())

            # We only continue if we have *one* *right* hand detected.
            if len(bboxes) != 1 or not is_right[0]:
                demo_verts.append(np.zeros((778, 3)))
                if visualize:
                    plt.imsave(f"scaled_hand_viz/{demo_name}/{str(idx).zfill(5)}.png", img)
                    plt.imsave(f"scaled_hand_viz/{demo_name}/mask_{str(idx).zfill(5)}.png", np.zeros_like(img))
                continue

            boxes = np.stack(bboxes)
            right = np.stack(is_right)
            dataset = ViTDetDataset(model_cfg, img, boxes, right, rescale_factor=args.rescale_factor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

            batch = next(iter(dataloader))
            batch = recursive_to(batch, device)

            with torch.no_grad():
                out = model(batch)

            multiplier    = (2*batch['right']-1)
            pred_cam      = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center    = batch["box_center"].float()
            box_size      = batch["box_size"].float()
            img_size      = batch["img_size"].float()
            scaled_focal_length = K[0, 0]
            pred_cam_t_full     = cam_crop_to_full(pred_cam, box_center, box_size, K).detach().cpu().numpy()

            # Render the result
            verts  = out['pred_vertices'][0].detach().cpu().numpy()
            joints = out['pred_keypoints_3d'][0].detach().cpu().numpy()

            is_right    = batch['right'][0].cpu().numpy()
            verts[:,0]  = (2*is_right-1)*verts[:,0]
            joints[:,0] = (2*is_right-1)*joints[:,0]
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
                is_right=is_right,
                K=K,
                hand_mask=hand_mask,
            )

            if tmesh.vertices.shape[0] == 0:
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
                o3d.io.write_point_cloud(f"scaled_hand_viz/{demo_name}/{idx}_hand_pcd.ply", hand_pcd)
                o3d.io.write_point_cloud(f"scaled_hand_viz/{demo_name}/{idx}_scene_pcd.ply", scene_pcd)

                key_y = np.clip(kpts_2d[:,1].astype(int), 0, img.shape[0]-1)
                key_x = np.clip(kpts_2d[:,0].astype(int), 0, img.shape[1]-1)
                img[key_y, key_x] = (0, 0, 255)
                plt.imsave(f"scaled_hand_viz/{demo_name}/{str(idx).zfill(5)}.png", img)
                plt.imsave(f"scaled_hand_viz/{demo_name}/mask_{str(idx).zfill(5)}.png", hand_mask)

            demo_verts.append(tmesh.vertices)

        demo_verts = np.array(demo_verts)
        demo_verts = infill_hand_verts(demo_name, demo_verts)

        # Store as gripper_pos in the zarr
        demo.create_dataset('gripper_pos', data=demo_verts)

def project_full_img(points, cam_trans, K):
    points = points + cam_trans
    points = points / points[..., -1:]
    
    V_2d = (K @ points.T).T 
    return V_2d[..., :-1]


if __name__ == '__main__':
    main()
