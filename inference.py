# -*- coding: utf-8 -*-
"""
Created on Fri June 2 16:59:17 2023
@title: inference - output video and caption
"""

import os
import json
import torch
import numpy as np

from tqdm import tqdm
from utils.voc import Vocabulary
from utils.config import MyConfig
from utils.dataset import DataHandler
from utils.transform import test_transform
from model.cnn_encoder import VGG16
from model.s2vt_module_lstm import S2VT

# load config file
cfg = MyConfig()

# vocabulary object
voc = Vocabulary(cfg)
voc.load()
voc.trim()

# input data
data_handler = DataHandler(cfg, voc)
test_dataset = data_handler.test_dataset()
test_dataloader = data_handler.test_loader(test_dataset)

# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_encoder = VGG16().to(device)             # output size: torch.Size([1, 4096])
model = S2VT(cfg, len(voc)).to(device)
model.load_state_dict(torch.load(cfg.weight_root + cfg.weight_file))
model.train()

i = 0
gts = {}
res = {}
pred = {}
for vid, (video_id, sen_id, label, mask) in tqdm(test_dataloader):
    frame_chunk = vid.get_batch(np.linspace(1, len(vid)-1, cfg.chunk_size).astype('int')).asnumpy()
    lstm_input = torch.zeros((1, cfg.chunk_size, cfg.frame_dim)).to(device)
    # cnn encode
    for idx, frame in enumerate(frame_chunk):
        frame = test_transform(frame).to(device)
        feature_ec = cnn_encoder(frame[None, :, :, :])
        lstm_input[0, idx] = feature_ec

    # lstm
    caption = torch.tensor(label).view(-1, len(label)).to(device)
    mask = torch.tensor(mask).to(device)
    cap_out = model(lstm_input, caption)

    check = cap_out.cpu().detach().numpy()
    label = ''
    sen = ''

    pred[sen_id] = {}
    pred[sen_id]['video_id'] = video_id
    # pred
    for w in check:
        word = voc.index2word[w.argmax()]
        if word == 'EOS':
            break
        else:
            sen += word
            sen += ' '
    pred[sen_id]['pred'] = sen[:-1]

    # gt
    for l in caption.cpu().detach().numpy()[0]:
        word = voc.index2word[l]
        if word == 'EOS':
            break
        elif word == 'SOS':
            continue
        else:
            label += voc.index2word[l]
            label += ' '
    pred[sen_id]['gt'] = label


# save predict
pred_file = os.path.join(cfg.output + 'test_pred.json')
with open(pred_file, 'w') as fp:
    json.dump(pred, fp, indent=4)
fp.close()
