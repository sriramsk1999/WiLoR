""" 
    This will process each frame in the video and save outputs in the same folder as the input video:
    - Main visualization as {video_folder}/rendered/frame_{idx}.jpg
    - Hand mask visualization (if --save_mask) as {video_folder}/rendered/frame_{idx}_gsam2_mask.jpg
    - 2D keypoints visualization (if --save_2d) as {video_folder}/rendered/frame_{idx}_keypoints.jpg
    - 3D hand mesh (if --save_mesh) as {video_folder}/meshes/frame_{idx}_{hand_idx}.obj
    
    To use GSAM2 for hand masking (enabled by default):
    python demo_mkv.py --video_path path/to/video.mkv --save_mesh --save_mask --save_2d
    
    To disable GSAM2 hand masking:
    python demo_mkv.py --video_path path/to/video.mkv --save_mesh --no_gsam2
"""

from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

from wilor.models import WiLoR, load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils.renderer import Renderer, cam_crop_to_full
from ultralytics import YOLO
from pyk4a import PyK4APlayback
from pyk4a.calibration import CalibrationType

import sys
sys.path.append('./third_party/Grounded-SAM-2')
from gsam_wrapper import GSAM2

LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)

def process_frame(frame_rgb, frame_depth, K, model, model_cfg, detector, renderer, device, 
                 rescale_factor=2.0, gsam2=None, save_path=None, frame_idx=None,
                 save_mask=False, save_2d=False):
    """Process a single frame and return/save the results"""
    
    # Convert BGR to RGB
    img_cv2 = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
    
    # Run hand detection
    detections = detector(img_cv2, conf=0.3, verbose=False)[0]
    bboxes = []
    is_right = []
    for det in detections:
        bbox = det.boxes.data.cpu().detach().squeeze().numpy()
        is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
        bboxes.append(bbox[:4].tolist())
    
    if len(bboxes) == 0:
        return None
    
    boxes = np.stack(bboxes)
    right = np.stack(is_right)
    dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=rescale_factor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    all_verts = []
    all_cam_t = []
    all_right = []
    all_joints = []
    all_kpts = []
    
    for batch in dataloader:
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
        
        batch_size = batch['img'].shape[0]
        for n in range(batch_size):
            verts = out['pred_vertices'][n].detach().cpu().numpy()
            joints = out['pred_keypoints_3d'][n].detach().cpu().numpy()
            
            is_right = batch['right'][n].cpu().numpy()
            verts[:,0] = (2*is_right-1)*verts[:,0]
            joints[:,0] = (2*is_right-1)*joints[:,0]
            cam_t = pred_cam_t_full[n]
            kpts_2d = project_full_img(verts, cam_t, K)
            
            # Get hand mask if GSAM2 is enabled
            hand_mask = None
            if gsam2 is not None and save_mask:
                masks, scores, _, _, _, _ = gsam2.get_masks_image("hand", img_cv2)
                if masks is not None and len(masks) > 0:
                    # Take the first mask with highest confidence
                    hand_mask = masks[0][0].astype(bool)  # Explicitly convert to boolean
                    
                    # Save mask visualization if requested
                    if save_path is not None:
                        mask_overlay = img_cv2.copy()
                        mask_overlay[hand_mask] = mask_overlay[hand_mask] * 0.6 + np.array(LIGHT_PURPLE) * 255 * 0.4
                        cv2.imwrite(
                            os.path.join(save_path, 'rendered', f'frame_{frame_idx}_gsam2_mask.jpg'),
                            mask_overlay
                        )
            
            # Save 2D keypoint visualization if requested
            if save_path is not None and save_2d:
                vis_img = img_cv2.copy()
                # Draw 2D keypoints
                for kpt in kpts_2d:
                    x, y = int(kpt[0]), int(kpt[1])
                    if 0 <= x < vis_img.shape[1] and 0 <= y < vis_img.shape[0]:
                        cv2.circle(vis_img, (x, y), 2, (0, 255, 0), -1)
                cv2.imwrite(
                    os.path.join(save_path, 'rendered', f'frame_{frame_idx}_keypoints.jpg'),
                    vis_img[:, :, ::-1]  # Convert RGB to BGR for OpenCV
                )
            
            # Save mesh if requested
            if save_path is not None:
                mesh_path = os.path.join(save_path, 'meshes', f'frame_{frame_idx}_{n}.obj')
                tmesh = renderer.vertices_to_trimesh_using_depth(
                    verts,
                    cam_t,
                    frame_depth,
                    scaled_focal_length,
                    img_size[n],
                    mesh_base_color=LIGHT_PURPLE,
                    is_right=is_right,
                    K=K,
                    hand_mask=hand_mask,
                )
                tmesh.export(mesh_path)
            
            all_verts.append(verts)
            all_cam_t.append(cam_t)
            all_right.append(is_right)
            all_joints.append(joints)
            all_kpts.append(kpts_2d)
    
    return all_verts, all_cam_t, all_right, all_joints, all_kpts, img_size[0]

def project_full_img(points, cam_trans, K):
    points = points + cam_trans
    points = points / points[..., -1:]
    
    V_2d = (K @ points.T).T 
    return V_2d[..., :-1]

def main():
    parser = argparse.ArgumentParser(description='Process MKV video for hand pose estimation')
    parser.add_argument('--video_path', type=str, required=True, help='Path to input MKV video')
    parser.add_argument('--save_mesh', action='store_true', default=False, help='Save hand meshes')
    parser.add_argument('--save_mask', action='store_true', default=False, help='Save hand mask visualization')
    parser.add_argument('--save_2d', action='store_true', default=False, help='Save 2D keypoint visualization')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--no_gsam2', action='store_true', help='Disable GSAM2 hand masking')
    args = parser.parse_args()

    # Setup output directories
    video_dir = str(Path(args.video_path).parent)
    os.makedirs(os.path.join(video_dir, 'rendered'), exist_ok=True)
    if args.save_mesh:
        os.makedirs(os.path.join(video_dir, 'meshes'), exist_ok=True)

    # Load models
    model, model_cfg = load_wilor(
        checkpoint_path='./pretrained_models/wilor_final.ckpt',
        cfg_path='./pretrained_models/model_config.yaml'
    )
    detector = YOLO('./pretrained_models/detector.pt')
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    detector = detector.to(device)
    model.eval()

    # Initialize GSAM2
    gsam2 = None if args.no_gsam2 else GSAM2(device=device, output_dir=Path(video_dir), debug=False)

    # Process video
    playback = PyK4APlayback(args.video_path)
    playback.open()  # Open the playback before using it
    
    # Get calibration
    calibration = playback.calibration.get_camera_matrix(CalibrationType.COLOR)
    
    frame_idx = 0
    pbar = tqdm(desc="Processing frames")
    
    while True:
        try:
            capture = playback.get_next_capture()
            
            # Get RGB and depth frames
            rgb_frame = capture.color[:, :, :3][:, :, ::-1]  # BGR format
            depth_frame = capture.transformed_depth
            
            if rgb_frame is None or depth_frame is None:
                continue
            
            # Process frame
            results = process_frame(
                rgb_frame, depth_frame, calibration,
                model, model_cfg, detector, renderer, device,
                rescale_factor=args.rescale_factor,
                gsam2=gsam2,
                save_path=video_dir if (args.save_mesh or args.save_mask or args.save_2d) else None,
                frame_idx=frame_idx,
                save_mask=args.save_mask,
                save_2d=args.save_2d
            )
            
            if results is None:
                frame_idx += 1
                pbar.update(1)
                continue
                
            all_verts, all_cam_t, all_right, all_joints, all_kpts, img_size = results
            
            # Render visualization
            misc_args = dict(
                mesh_base_color=LIGHT_PURPLE,
                scene_bg_color=(1, 1, 1),
                focal_length=calibration[0, 0],
            )
            cam_view = renderer.render_rgba_multiple(
                all_verts, cam_t=all_cam_t, 
                render_res=img_size, 
                is_right=all_right, 
                **misc_args
            )

            # Create overlay
            input_img = rgb_frame.astype(np.float32)/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2)
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
            
            # Save visualization
            cv2.imwrite(
                os.path.join(video_dir, 'rendered', f'frame_{frame_idx}.jpg'),
                255*input_img_overlay[:, :, ::-1]
            )
            
            frame_idx += 1
            pbar.update(1)
            
        except EOFError:
            break  # End of video file reached
            
    pbar.close()
    playback.close()  # Close the playback when done

if __name__ == '__main__':
    main() 