# PM-Track: A Positional-Aware Multi-Scale Tracker for Multi-Object Tracking


Here is the repository for PMTrack, and the project is built based on [MMDet](https://github.com/open-mmlab/mmdetection).

The core code of this project is in `./PMTrack/`.


# Abstract
Multi-Object Tracking (MOT) is a fundamental task in computer vision, where robust appearance modeling is crucial for accurate target association. Tracking-by-detection (TbD) has emerged as a dominant paradigm due to its lightweight architecture and efficiency. However, the quality of appearance feature extraction significantly impacts tracking performance, highlighting the need for fine-grained and scalable feature representations.
In this work, we propose the Positional-Aware Multi-Scale Tracker (PM-Track) to enhance appearance modeling in MOT. PM-Track incorporates a Multi-Scale Encoder (MSE), which leverages a Pyramid Sampler and Multi-Scale Position Encoding to fully exploit contextual information at multiple scales. Additionally, we design a Global-Local Decoder (GLD), which uses learnable part queries constrained by positional priors to extract local features, complemented by a Global Query for aggregating all local information. To further improve robustness under occlusions and deformations, we introduce Gaussian Erasing Augmentation (GEA), a novel data augmentation method that uses anisotropic Gaussian kernels to generate erasing masks with richer variations.
We evaluate PM-Track on standard MOT benchmarks and related Re-ID tasks. Experimental results demonstrate that PM-Track achieves state-of-the-art performance, with 64.6 HOTA on MOT17, 63.2 HOTA on MOT20, and 53.1 HOTA on DanceTrack, while delivering competitive results in Re-ID tasks.


# Models

## Detector

|  Backbone  | size | Mem (GB) | box AP |                  Config                  |                                                                                                                                         Download                                                                                                                                         |
| :--------: | :--: | :------: | :----: | :--------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  YOLOX-l   | 640  |   19.9   |  49.4  |  [config](./yolox_l_8xb8-300e_coco.py)   |       [model](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth)      |
|  YOLOX-x   | 640  |   28.1   |  50.9  |  [config](./yolox_x_8xb8-300e_coco.py)   |       [model](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth)      |

**Note:**
 In this project we use the [YOLOX](https://arxiv.org/abs/2107.08430) pretrained models from [MMDetection](https://github.com/open-mmlab/mmdetection/tree/main/configs/yolox) without additional traning.



# Running 
## Install 

We implement PDTrack based on [MMDectection](https://github.com/open-mmlab/mmdetection) and [MMCV](https://github.com/open-mmlab/mmcv). 

**Step 0.** Install [MMEngie](https://github.com/open-mmlab/mmengine) and [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim):
```
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

**Step 1.** Install PMTrack.

Install the project to local environment.

```bash
git clone https://github.com/kuaizhiyan/PMTrack.git
cd PMTrack
pip install -v -e . -r requirements/tracking.txt
```
**Step 2.** Install TrackEval.
```
pip install git+https://github.com/JonathonLuiten/TrackEval.git
```

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

## Training 


Train FARE with 1 GPU on MOT17:
```
python tools/train.py ./configs/reid_pmnet_4xb32_14000iter_mot17train80_test-mot17val20.py --work-dir ./experiments
```


Train FARE with 4 GPUs on MOT17:
```
sh tools/dist_train.sh ./configs/reid_pmnet_4xb32_14000iter_mot17train80_test-mot17val20.py 4 --work-dir ./experiments
```

Train FARE with 4 GPUs on MOT20:
```
sh tools/dist_train.sh ./configs/reid_pmnet_4xb32_14000iter_mot20train80_test-mot20val20.py 4 --work-dir ./experiments
```

Train extral detector with 8 GPUs on crowdhuman:
```
sh tools/dist_train.sh configs/strongsort/yolox_x_8xb4-80e_crowdhuman-mot17halftrain_test-mot17halfval.py 8 --work-dir ./experiments
```

<!-- ## Testing
Test on MOT17-half:
```
sh tools/dist_test.sh  projects/configs/xxxxxx 8 --eval bbox
``` -->

# Cite

# Acknowledgement
This project is inspired by excellent works such as [StrongSORT](https://github.com/dyhBUPT/StrongSORT), [MMDetection](https://github.com/open-mmlab/mmdetection), [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), [ConditionalDETR](https://github.com/Atten4Vis/ConditionalDETR), [DINO](https://github.com/IDEA-Research/DINO),etc. Many thanks for their wonderful works.