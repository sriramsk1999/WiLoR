"""Example usage:
    # Process all RGB-D + K images wtihin a folder to get hand detection and pose estimation
    python demo_rgbdk.py --npy_folder demo_rgbdk --out_folder demo_out --save_mesh
    
    # This will process demo_rgbdk/cup.npy and save:
    # - Rendered visualization as out_demo/cup.jpg
    # - 3D hand mesh (if --save_mesh is used) as out_demo/cup_*.obj
    
    # To use GSAM2 for hand masking (enabled by default):
    python demo_rgbdk.py --npy_folder demo_rgbdk --out_folder demo_out --save_mesh
    # To disable GSAM2 hand masking:
    python demo_rgbdk.py --npy_folder demo_rgbdk --out_folder demo_out --save_mesh --no_gsam2
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

import sys
sys.path.append('./third_party/Grounded-SAM-2')
from gsam_wrapper import GSAM2

LIGHT_PURPLE=(0.25098039, 0.274117647, 0.65882353)

def main():
    parser = argparse.ArgumentParser(description='WiLoR demo code')
    parser.add_argument('--npy_folder', type=str, default='npy_files', help='Folder with input npy files')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--no_gsam2', action='store_true', help='Disable GSAM2 hand masking')

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
        gsam2 = GSAM2(device=device, output_dir=Path(args.out_folder), debug=False)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Get all npy files in the folder
    npy_paths = list(Path(args.npy_folder).glob('*.npy'))
    # Iterate over all npy files in folder
    for npy_path in npy_paths:
        data = np.load(npy_path, allow_pickle=True)
        data = data.item()
        img_cv2 = cv2.cvtColor(data['rgb'], cv2.COLOR_BGR2RGB)
        K = data['K']
        detections = detector(img_cv2, conf = 0.3, verbose=False)[0]
        bboxes    = []
        is_right  = []
        for det in detections: 
            Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
            bboxes.append(Bbox[:4].tolist())
        
        if len(bboxes) == 0:
            continue
        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        all_joints= []
        all_kpts  = []
        
        for batch in dataloader: 
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
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Get filename from path npy_path
                img_fn, _ = os.path.splitext(os.path.basename(npy_path))
                
                verts  = out['pred_vertices'][n].detach().cpu().numpy()
                joints = out['pred_keypoints_3d'][n].detach().cpu().numpy()
                
                is_right    = batch['right'][n].cpu().numpy()
                verts[:,0]  = (2*is_right-1)*verts[:,0]
                joints[:,0] = (2*is_right-1)*joints[:,0]
                cam_t = pred_cam_t_full[n]
                kpts_2d = project_full_img(verts, cam_t, K)
                
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)
                all_joints.append(joints)
                all_kpts.append(kpts_2d)
                
                # Save all meshes to disk
                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    
                    # Get hand mask if GSAM2 is enabled
                    hand_mask = None
                    if gsam2 is not None:
                        # Use "hand" as the object to detect
                        masks, scores, _, _, _, _ = gsam2.get_masks_image("hand", img_cv2)
                        if masks is not None and len(masks) > 0:
                            # Take the first mask with highest confidence
                            hand_mask = masks[0][0]  # Shape: (H, W)
                    
                    tmesh = renderer.vertices_to_trimesh_using_depth(
                        verts, 
                        camera_translation, 
                        data['depth'], 
                        scaled_focal_length, 
                        img_size[n], 
                        mesh_base_color=LIGHT_PURPLE, 
                        is_right=is_right, 
                        K=K,
                        hand_mask=hand_mask,
                    )
                    tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{n}.obj'))

        # Render front view
        if len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_PURPLE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}.jpg'), 255*input_img_overlay[:, :, ::-1])
        
        # Create mask overlay visualization if GSAM2 was used
        if gsam2 is not None:
            if masks is not None and len(masks) > 0:
                # Create visualization with mask overlay
                hand_mask = masks[0][0].astype(bool)  # Explicitly convert to boolean
                mask_overlay = img_cv2.copy()
                mask_overlay[hand_mask] = mask_overlay[hand_mask] * 0.6 + np.array(LIGHT_PURPLE) * 0.4
                # Save visualization
                cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_gsam2_mask.jpg'), mask_overlay)
        

def project_full_img(points, cam_trans, K):
    points = points + cam_trans
    points = points / points[..., -1:]
    
    V_2d = (K @ points.T).T 
    return V_2d[..., :-1]


if __name__ == '__main__':
    main()
