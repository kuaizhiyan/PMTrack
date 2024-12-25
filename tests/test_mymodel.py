import torch
import numpy as np
from mmdet.registry import MODELS
from mmengine.config import Config
from mmdet.models.necks import ChannelMapper
from torchsummary import summary
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

if __name__ == '__main__':
    
    x = torch.randn(1,3,128,256)
    cfg = Config.fromfile('/home/kzy/project/PartDecoder/mmdetection/configs/reid/reid_pdnet.py')
    model = MODELS.build(cfg.model)
    out = model(x)
    print(out[0].shape)