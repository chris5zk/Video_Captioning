# -*- coding: utf-8 -*-
"""
Created on Fri June 2 16:59:17 2023
@title: inference - output video and caption
"""

import os
import json
import torch
from decord import VideoReader
from utils.voc import Vocabulary
from utils.config import MyConfig
from utils.transform import test_transform
from model.efficientnet_ec import EfficientNetEc
from model.s2vt_module_lstm import S2VT


def print_in_english(caption, cfg):
    id2word = json.load(open(cfg.i2w))
    sentence = ''
    for cap in caption:
        sentence += id2word[str(cap)]
        sentence += ' '
        if cap == cfg.EOS_token:
            print(sentence.replace('EOS', '.'))
            break


# load config file
cfg = MyConfig()

# vocabulary object
voc = Vocabulary(cfg)
voc.load()
voc.trim()

# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_encoder = EfficientNetEc().to(device)  # output size: torch.Size([1, 1280])
model = S2VT(cfg, len(voc)).to(device)
model.load_state_dict(torch.load(cfg.weight_root + cfg.weight_file))
model.eval()

# input data
file = os.listdir(cfg.input)
print('Start inference...')
for video in file:
    vid = VideoReader(cfg.input + video)
    frame_chunk = vid[:cfg.chunk_size].asnumpy()
    lstm_input = torch.zeros((1, cfg.chunk_size, 1280)).to(device)
    # cnn encode
    for idx, frame in enumerate(frame_chunk):
        frame = test_transform(frame).to(device)
        feature_ec = cnn_encoder(frame[None, :, :, :])  # torch.Size([1, 1280])
        lstm_input[0, idx] = feature_ec

    cap_out = model(lstm_input)
    caption = []
    for tensor in cap_out:
        caption.append(tensor.item())
    print(caption)
    print('............................\nCaption:')
    print_in_english(caption, cfg)
