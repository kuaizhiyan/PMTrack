# Positional-Aware Multi-Scale Tracking with Fine-Grained Appearance Modeling for Multi-Object Tracking

Here is the repository for PMTrack, and the project is built based on [MMDet](https://github.com/open-mmlab/mmdetection).

The core code of this project is in `./PMTrack/`.


# Abstract
Multi-Object Tracking (MOT) is a fundamental task in computer vision, where robust appearance modeling is crucial for accurate target association. Tracking-by-detection (TbD) has emerged as a dominant paradigm due to its lightweight architecture and efficiency. However, the quality of appearance feature extraction significantly impacts tracking performance, highlighting the need for fine-grained and scalable feature representations.
In this work, we propose the Positional-Aware Multi-Scale Tracker (PM-Track) to enhance appearance modeling in MOT. PM-Track incorporates a Multi-Scale Encoder (MSE), which leverages a Pyramid Sampler and Multi-Scale Position Encoding to fully exploit contextual information at multiple scales. Additionally, we design a Global-Local Decoder (GLD), which uses learnable part queries constrained by positional priors to extract local features, complemented by a Global Query for aggregating all local information. To further improve robustness under occlusions and deformations, we introduce Gaussian Erasing Augmentation (GEA), a novel data augmentation method that uses anisotropic Gaussian kernels to generate erasing masks with richer variations.
We evaluate PM-Track on standard MOT benchmarks and related Re-ID tasks. Experimental results demonstrate that PM-Track achieves state-of-the-art performance, with 64.6 HOTA on MOT17, 63.2 HOTA on MOT20, and 53.1 HOTA on DanceTrack, while delivering competitive results in Re-ID tasks.


# Installation
For detailed installation instructions, please see [Installation Instructions](./docs/installation.md).


## Data 

Please follow the guide in [MMDectection-tracking](https://github.com/open-mmlab/mmdetection/blob/main/docs/en/user_guides/tracking_dataset_prepare.md) to prepare the datasets as structure as :
```
PDTrack
├── data
│   ├── coco
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   │   ├── annotations
│   │
|   ├── MOT15/MOT16/MOT17/MOT20
|   |   ├── train
|   |   |   ├── MOT17-02-DPM
|   |   |   |   ├── det
|   │   │   │   ├── gt
|   │   │   │   ├── img1
|   │   │   │   ├── seqinfo.ini
│   │   │   ├── ......
|   |   ├── test
|   |   |   ├── MOT17-01-DPM
|   |   |   |   ├── det
|   │   │   │   ├── img1
|   │   │   │   ├── seqinfo.ini
│   │   │   ├── ......
│   │
│   ├── crowdhuman
│   │   ├── annotation_train.odgt
│   │   ├── annotation_val.odgt
│   │   ├── train
│   │   │   ├── Images
│   │   │   ├── CrowdHuman_train01.zip
│   │   │   ├── CrowdHuman_train02.zip
│   │   │   ├── CrowdHuman_train03.zip
│   │   ├── val
│   │   │   ├── Images
│   │   │   ├── CrowdHuman_val.zip
│   │
```

# Usage
For detailed instructions on training and testing the model, please refer to [Usage Instructions](./docs/usage.md).

# Models
For a detailed list of models and their corresponding weights, please refer to [modelzoo.md](./docs/modelzoo.md).


# Cite
```bibtex
@article{kuai2025pmtrack,
  title={Positional-Aware Multi-Scale Tracking with Fine-Grained Appearance Modeling for Multi-Object Tracking},
  author={Kuai, Zhiyan and Song, YuZe and Liu, Tao and Gu, Yanzhen and Xu, Gang and He, Shuangyan and Yefei,Bai and Li, Peiliang and Huang, Hui},
  journal={Applied Intelligence},
  year={2025},
  pages={1--27},
}

```

# Acknowledgement
This project is inspired by excellent works such as [StrongSORT](https://github.com/dyhBUPT/StrongSORT), [MMDetection](https://github.com/open-mmlab/mmdetection), [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [ConditionalDETR](https://github.com/Atten4Vis/ConditionalDETR), [DINO](https://github.com/IDEA-Research/DINO),etc. Many thanks for their wonderful works.