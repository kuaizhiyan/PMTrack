# Models

## Detector

|  Backbone  | size | Mem (GB) | box AP |                  Config                  |                                                                                                                                         Download                                                                                                                                         |
| :--------: | :--: | :------: | :----: | :--------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  YOLOX-l   | 640  |   19.9   |  49.4  |  [config](./yolox_l_8xb8-300e_coco.py)   |       [model](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth)      |
|  YOLOX-x   | 640  |   28.1   |  50.9  |  [config](./yolox_x_8xb8-300e_coco.py)   |       [model](https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth)      |

**Note:**
 In this project we use the [YOLOX](https://arxiv.org/abs/2107.08430) pretrained models from [MMDetection](https://github.com/open-mmlab/mmdetection/tree/main/configs/yolox) without additional traning.

## PMTrack
| Model | Config | Download|
| :-: |:-: |:-: |
|PMTrack-Market1501 pretrained | [config](configs/reid/pmnet_base_market1501-cls.py)|[model](https://drive.google.com/file/d/1qwN9oAYwqdesEp7qSlzEm3idnrOpthkd/view?usp=drive_link) |
|PMTrack-MOT17 | [config]()|[model](https://drive.google.com/file/d/1y_xIHgiho2j9WxesqVrCRLovog33yEL9/view?usp=drive_link) |
|PMTrack-MOT20 | [config](configs/reid/pmnet_base_market1501-cls.py)|[model](https://drive.google.com/file/d/1RgQ1reYqhIIl8Ol0bqQW6SMjv_3Zh_dF/view?usp=drive_link) |