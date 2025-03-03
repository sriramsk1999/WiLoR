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


## Before:
<img src="https://github.com/user-attachments/assets/c34ad910-0fbe-4356-aded-486195a97485" width="300">

## After:
<img src="https://github.com/user-attachments/assets/4a54cdb3-648e-4a45-9d4d-afb4a2cc14ae" width="300">

## 2D projection for reference:
<img src="https://github.com/user-attachments/assets/46bf054b-a092-4b77-8b1a-b2a519e7f9f3" width="300">


## Areas for improvement:
The WiLoR hand 2D projection might match pixels that do not match with the real hand but with an object (i.e. when the real hand is occluded by that object). That would mean in steps 3.2) and 4) we include the depths
of the object rather than the ones from the real hand, making the scaling a bit inaccurate.

## Further comments:
I'm not sure if the MANO model also considers the size of the hand. That means that if we scale the hand, it might not correspond anymore to a MANO model hand.

<div align="center">

# WiLoR: End-to-end 3D hand localization and reconstruction in-the-wild

[Rolandos Alexandros Potamias](https://rolpotamias.github.io)<sup>1</sup> &emsp; [Jinglei Zhang]()<sup>2</sup> &emsp; [Jiankang Deng](https://jiankangdeng.github.io/)<sup>1</sup> &emsp; [Stefanos Zafeiriou](https://www.imperial.ac.uk/people/s.zafeiriou)<sup>1</sup>  

<sup>1</sup>Imperial College London, UK <br>
<sup>2</sup>Shanghai Jiao Tong University, China

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
# Install requirements
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
