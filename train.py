# -*- coding: utf-8 -*-
"""
Created on Sat May 20 01:22:17 2023
@title: training loop
"""

import os
import torch
from datetime import datetime
from torch import nn
from tqdm import tqdm
from utils.voc import Vocabulary
from utils.config import MyConfig
from utils.dataset import DataHandler
from utils.transform import train_transform, test_transform
from model.efficientnet_ec import EfficientNetEc
from model.s2vt_module_lstm import S2VT

# from model.bleu import Bleu

# load config file
cfg = MyConfig()
tfm = "%Y%m%d-%H%M%S"
os.makedirs(cfg.log_root, exist_ok=True)
os.makedirs(cfg.weight_root, exist_ok=True)
os.makedirs(cfg.ckpt_root, exist_ok=True)

# vocabulary object
voc = Vocabulary(cfg)
voc.load()
voc.trim()

# load dataset
print('Loading Dateset...')
data_handler = DataHandler(cfg, voc)
train_dataset, val_dataset, test_dataset = data_handler.getDatasets()
train_dataloader, val_dataloader, test_dataloader = data_handler.getDataloader(train_dataset, val_dataset, test_dataset)

# model
print('Loading Model...')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_encoder = EfficientNetEc().to(device)  # output size: torch.Size([1, 1280])
model = S2VT(cfg, len(voc)).to(device)

# loss
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

# checkpoint
loss, epoch = None, None
if cfg.use_ckpt:
    print(f"Using checkpoint file for training: '{cfg.ckpt_file}'")
    try:
        checkpoint = torch.load(cfg.ckpt_root + cfg.ckpt_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    except Exception as e:
        print(f"Root '{cfg.ckpt_root + cfg.ckpt_file}' not found...")

# epoch range
loop = range(cfg.epoch) if epoch is None else range(epoch, cfg.epoch)

# training loop
print('Start training...')
for epoch in loop:
    # iteration of training data
    i = 0
    for vid, (label, mask) in train_dataloader:
        # iteration of video
        # for frame_idx in range(len(vid)-cfg.chunk_size-1):
        # frame_chunk = vid[frame_idx:frame_idx+cfg.chunk_size].asnumpy()
        frame_chunk = vid[:cfg.chunk_size].asnumpy()
        # if len(frame_chunk) != cfg.chunk_size:
        #     continue

        lstm_input = torch.zeros((1, cfg.chunk_size, 1280)).to(device)
        # cnn encode
        for idx, frame in enumerate(frame_chunk):
            frame = train_transform(frame).to(device)
            feature_ec = cnn_encoder(frame[None, :, :, :])  # torch.Size([1, 1280])
            lstm_input[0, idx] = feature_ec

        # lstm
        caption = torch.tensor(label).view(-1, len(label)).to(device)
        mask = torch.tensor(mask).to(device)
        cap_out = model(lstm_input, caption)
        cap_labels = caption[:, 1:].contiguous().view(-1)
        cap_mask = mask[:, 1:].contiguous().view(-1)

        logit_loss = loss_func(cap_out, cap_labels)
        masked_loss = logit_loss * mask
        loss = torch.sum(masked_loss) / torch.sum(mask)

        optimizer.zero_grad()
        logit_loss.backward()
        optimizer.step()

        if i % 60 == 0:
            print(f"Epoch: {epoch + 1}/{cfg.epoch} iteration:{i}/{len(train_dataset)}, loss: {loss}")
        if i != 0 and i % 300 == 0:
            print(f"Save checkpoint at '{cfg.ckpt_root}'")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, cfg.ckpt_root + datetime.now().strftime(tfm) + f'_epoch_{epoch + 1}_iter_{i}.ckpt')
            print('Done.')
        i += 1
    print(f"Save weight for epoch{epoch+1} at '{cfg.weight_root}'")
    torch.save(model.state_dict(), cfg.weight_root + datetime.now().strftime(tfm) + f'_epoch_{epoch + 1}.pt')
    # if loss > best_mIoU:
    #         best_mIoU = mean_IoU
    #         torch.save(model.module.state_dict(),
    #                 os.path.join(final_output_dir, 'best.pt'))
