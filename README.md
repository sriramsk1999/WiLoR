<div align="center">

# Scaling for better 3D Alignment

</div>

## What we do
Outputs of WiLoR:
- 3D hand and its 2D projection onto the image. <br>
What we infer:
- we modify (or "scale") the coordinates of the 3D hand so they align (in 3D) with the real 3D hand. We leverage their alignment in the image. <br>

Pseudocode: <br>
1) Identify which of the 3D wilor hand points are visible from the camera point of view.
2) Average their depth (obtaining avg_3d_wilor_hand)
3) Identify which of the pixels in the image correspond to these 3D wilor hand points
    3.1) Project the 3D wilor hand points onto the image
    3.2) Create a mask from the projection
4) Average the depth of the pixels from step 3, obtaining avg_3d_real_hand.
    4.1) Note: we don’t consider the real point clouds that are further away than 1.6m (distance to the end of the experiments table in this case)
6) Compute the ratio of both averages: depth_ratio = avg_3d_real_hand / avg_3d_wilor_hand.
7) Scale the 3D wilor hand points using the depth_ratio
    6.1) Save their depths into an array old_depths
    6.1) Project the 3D wilor hand points onto the image
    6.2) Turn those projected points back to 3D considering their new depths = old_depths * depth_ratio, and considering the real camera intrinsics.


### Before:
<img src="https://github.com/user-attachments/assets/c34ad910-0fbe-4356-aded-486195a97485" width="300">

### After:
<img src="https://github.com/user-attachments/assets/4a54cdb3-648e-4a45-9d4d-afb4a2cc14ae" width="300">

### 2D projection for reference:
<img src="https://github.com/user-attachments/assets/46bf054b-a092-4b77-8b1a-b2a519e7f9f3" width="300">


## Further improvements using GSAM2 segmentation
In step 3.2), the WiLoR hand 2D projection might match pixels that do not match with the real hand but with an object (i.e. when the real hand is occluded by that object). That would mean in steps 3.2) and 4) we include the depths of the object rather than the ones from the real hand, making the scaling a bit inaccurate. We solve this issue using a hand segmentation from GSAM2.
This improvement is set by default.
<br>
Here's an example taken from the file "demo_rgbk/mug_lots_of_occlusion.npy":
### Before
<img src="https://github.com/user-attachments/assets/a50043ed-e2b3-43a7-a870-0deac4371477" width="300">

### After
<img src="https://github.com/user-attachments/assets/3865c951-5fc7-4883-a1a0-56671b6ce167" width="300">

## Model weights, submodules and mano hand model
```bash
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P ./pretrained_models/
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt -P ./pretrained_models/
```

```bash
git submodule update --init --recursive
```

It is also required to download MANO model from [MANO website](https://mano.is.tue.mpg.de). 
Create an account by clicking Sign Up and download the models (mano_v*_*.zip). Unzip and place the right hand model `MANO_RIGHT.pkl` under the `mano_data/` folder. 
Note that MANO model falls under the [MANO license](https://mano.is.tue.mpg.de/license.html).

## Our installation (Cuda 12.1 / 12.4)
```bash
conda create --name wilor python=3.10
conda activate wilor
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install nvidia/label/cuda-12.1.0::cuda-toolkit -c nvidia/label/cuda-12.1.0 -y
pip install -r requirements.txt
pip install -v -e third_party/ViTPose
```

Install GSAM2 within the same conda environment as WiLoR.
Additional requirements for GSAM2 and visualizations:
```bash
pip install -r third_party/Grounded-SAM-2/grounding_dino/requirements.txt
# Additional requirements from the fork
pip install open3d
pip install pyk4a
```

## Further installation instructions
Download GSAM2 model weights:
```bash
cd third_party/Grounded-SAM-2/checkpoints
bash download_ckpts.sh
cd ../gdino_checkpoints
bash download_ckpts.sh
```

## Detectron2 and VitPose

Borrowed from HAPTIC

``` bash
mkdir -p output/vitpose_ckpts/vitpose+_huge

# Download the wholebody model (133 keypoints including hands)
# From: https://1drv.ms/u/s!AimBgYV7JjTlgccoXv8rCUgVe7oD9Q?e=ZBw6gR
# You'll need to download this manually and place it at:
# output/vitpose_ckpts/vitpose+_huge/wholebody.pth
```

## Running it
```bash
python demo_rgbdk.py --npy_folder demo_rgbdk --out_folder demo_out --save_mesh
python demo_rgbdk.py --npy_folder demo_rgbdk --out_folder demo_out --save_mesh --no_gsam2


python demo_lerobot_detectron2.py --input_folder "/home/sriram/.cache/huggingface/lerobot/sriramsk/mug_on_platform_20250910_human/videos/chunk-000/" --output_folder "/data/sriram/lerobot_extradata/sriramsk/mug_on_platform_20250910_human/wilor_hand_pose" --visualize
```

## Visualizations
Python scripts have been added to the folder [utils](https://github.com/jacintosuner/WiLoR/tree/main/utils) to:
- capture an rbgdk frame (rgb + depth + camera intrinsics): [capture_rgbdk.py](https://github.com/jacintosuner/WiLoR/blob/main/utils/capture_rgbdk.py)
- visualize an rgbdk frame together with / or an obj file together with / or a pcd file (npy file with point cloud info): [visualize_rgbdk_or_obj_or_pcd.py ](https://github.com/jacintosuner/WiLoR/blob/main/utils/visualize_rgbdk_or_obj_or_pcd.py)

## Further comments:
I'm not sure if the MANO model also considers the size of the hand. That means that if we scale the hand, it might not correspond anymore to a MANO model hand.

<div align="center">

# WiLoR: End-to-end 3D hand localization and reconstruction in-the-wild

[Rolandos Alexandros Potamias](https://rolpotamias.github.io)<sup>1</sup> &emsp; [Jinglei Zhang]()<sup>2</sup> &emsp; [Jiankang Deng](https://jiankangdeng.github.io/)<sup>1</sup> &emsp; [Stefanos Zafeiriou](https://www.imperial.ac.uk/people/s.zafeiriou)<sup>1</sup>  

<sup>1</sup>Imperial College London, UK <br>
<sup>2</sup>Shanghai Jiao Tong University, China

<font color="blue"><strong>CVPR 2025</strong></font> 

<a href='https://rolpotamias.github.io/WiLoR/'><img src='https://img.shields.io/badge/Project-Page-blue'></a>
<a href='https://arxiv.org/abs/2409.12259'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
<a href='https://huggingface.co/spaces/rolpotamias/WiLoR'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-green'></a>
<a href='https://colab.research.google.com/drive/1bNnYFECmJbbvCNZAKtQcxJGxf0DZppsB?usp=sharing'><img src='https://colab.research.google.com/assets/colab-badge.svg'></a>
</div>

<div align="center">

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wilor-end-to-end-3d-hand-localization-and/3d-hand-pose-estimation-on-freihand)](https://paperswithcode.com/sota/3d-hand-pose-estimation-on-freihand?p=wilor-end-to-end-3d-hand-localization-and)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wilor-end-to-end-3d-hand-localization-and/3d-hand-pose-estimation-on-ho-3d)](https://paperswithcode.com/sota/3d-hand-pose-estimation-on-ho-3d?p=wilor-end-to-end-3d-hand-localization-and)

</div>

This is the official implementation of **[WiLoR](https://rolpotamias.github.io/WiLoR/)**, an state-of-the-art hand localization and reconstruction model:

![teaser](assets/teaser.png)

## Installation
### [Update] Quick Installation
Thanks to [@warmshao](https://github.com/warmshao) WiLoR can now be installed using a single pip command:  
```
pip install git+https://github.com/warmshao/WiLoR-mini
```
Please head to [WiLoR-mini](https://github.com/warmshao/WiLoR-mini) for additional details. 

**Note:** the above code is a simplified version of WiLoR and can be used for demo only. 
If you wish to use WiLoR for other tasks it is suggested to follow the original installation instructued bellow: 
### Original Installation
```
git clone --recursive https://github.com/rolpotamias/WiLoR.git
cd WiLoR
```

The code has been tested with PyTorch 2.0.0 and CUDA 11.7. It is suggested to use an anaconda environment to install the the required dependencies:
```bash
conda create --name wilor python=3.10
conda activate wilor

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

Download the pretrained models using: 
```bash
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P ./pretrained_models/
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt -P ./pretrained_models/
```
It is also required to download MANO model from [MANO website](https://mano.is.tue.mpg.de). 
Create an account by clicking Sign Up and download the models (mano_v*_*.zip). Unzip and place the right hand model `MANO_RIGHT.pkl` under the `mano_data/` folder. 
Note that MANO model falls under the [MANO license](https://mano.is.tue.mpg.de/license.html).
## Demo
```bash
python demo.py --img_folder demo_img --out_folder demo_out --save_mesh 
```
## Start a local gradio demo
You can start a local demo for inference by running:
```bash
python gradio_demo.py
```
## WHIM Dataset
The dataset will be released soon. 

## Acknowledgements
Parts of the code are taken or adapted from the following repos:
- [HaMeR](https://github.com/geopavlakos/hamer/)
- [Ultralytics](https://github.com/ultralytics/ultralytics)

## License 
WiLoR models fall under the [CC-BY-NC--ND License](./license.txt). This repository depends also on [Ultralytics library](https://github.com/ultralytics/ultralytics) and [MANO Model](https://mano.is.tue.mpg.de/license.html), which are fall under their own licenses. By using this repository, you must also comply with the terms of these external licenses.
## Citing
If you find WiLoR useful for your research, please consider citing our paper:

```bibtex
@misc{potamias2024wilor,
    title={WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild},
    author={Rolandos Alexandros Potamias and Jinglei Zhang and Jiankang Deng and Stefanos Zafeiriou},
    year={2024},
    eprint={2409.12259},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
