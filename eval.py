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
val_dataloader = data_handler.val_loader(val_dataset)

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
for vid, (video_id, sen_id, label, mask) in tqdm(val_dataloader):
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
pred_file = os.path.join(cfg.output + 'val_pred.json')
with open(pred_file, 'w') as fp:
    json.dump(pred, fp, indent=4)
fp.close()

# evaluate metrics
print('start eval...')
from utils.metrics import EvalMetrics

gt_path = cfg.dataset_root + 'msrvtt_label_val.json'
eval_path = cfg.output +  'val_pred.json'

with open(gt_path, 'r') as gt_file, open(eval_path, 'r') as eval_file:
    gt_data = json.load(gt_file)
    eval_data = json.load(eval_file)

length_list = [5, 10, 15, 20]

for length in length_list:
    for vid_id, captions in gt_data.items():
        cap = []
        for _, caption in captions.items():
            sen = caption.replace('SOS ', '').replace(' EOS', '')
            if len(sen.split()) <= length:
                cap.append(sen)
        gts[vid_id] = cap

    for k, v in list(gts.items()):
        if len(v) == 0:
            del gts[k]

    for _, pred in eval_data.items():
        cap = []
        sen = ''
        for word in pred['pred'].split():
            if word != 'EOS':
                sen += word
            else:
                sen = sen[:-1]
                break
            sen += ' '
        cap.append(sen)

        if pred['video_id'] in gts.keys():
            res[pred['video_id']] = cap

    metrics = EvalMetrics()
    msg = metrics.compute_scores(gts, res)
    print(f'Sentence length: {length}')
    print(msg)
