# -*- coding: utf-8 -*-
"""
Created on Sat May 20 01:22:17 2023
@title: training loop
"""

import torch
from PIL import Image
from utils.voc import Vocabulary
from utils.config import MyConfig
from utils.dataset import DataHandler
from utils.transform import train_transform, test_transform
from model.efficientnet_ec import EfficientNetEc
from model.bleu import Bleu

# load config file
cfg = MyConfig()

# vocabulary object
voc = Vocabulary(cfg)
voc.load()

# load dataset
data_handler = DataHandler(cfg, voc)
train_dataset, val_dataset, test_dataset = data_handler.getDatasets()
train_dataloader, val_dataloader, test_dataloader = data_handler.getDataloader(train_dataset, val_dataset, test_dataset)

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# This parts of model need to pack into a Model Class
model_cnn = EfficientNetEc().to(device)     # output size: torch.Size([1, 1280])
model_ec = None
model_dc = None

criterion = Bleu()
optimizer = None

# training loop
for epoch in range(cfg.epoch):   
    # iteration of training data
    for vid, label in train_dataloader:
        hs = 0
        # iteration of video
        for frame_idx in range(len(vid)-cfg.chunk_size-1):
            frame_chunk = vid[frame_idx:frame_idx+cfg.chunk_size].asnumpy()
            # iteration of frame chunk
            for frame in frame_chunk:
                frame = train_transform(Image.fromarray(frame)).to(device)
                feature_ec = model_cnn(frame[None, :, :, :])
                
                
                
            break
        break
