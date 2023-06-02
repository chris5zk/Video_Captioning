# -*- coding: utf-8 -*-
"""
Created on Sat May 20 01:22:17 2023
@title: training loop
"""

import os
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from torch import nn
from utils.voc import Vocabulary
from utils.config import MyConfig
from utils.dataset import DataHandler
from utils.logging import create_logger
from utils.transform import train_transform, test_transform
from model.efficientnet_ec import EfficientNetEc
from model.s2vt_module_lstm import S2VT


# load config file
cfg = MyConfig()
tfm = "%y%m%d-%H%M%S"
logger, log_path, timestamp = create_logger(cfg)
os.makedirs(cfg.log_root, exist_ok=True)
os.makedirs(cfg.weight_root, exist_ok=True)
os.makedirs(cfg.ckpt_root, exist_ok=True)

# vocabulary object
voc = Vocabulary(cfg)
voc.load()
voc.trim()
logging.info(f'Voc Length after trim: {len(voc)}')

# load dataset
logging.info('Loading Dateset...')
data_handler = DataHandler(cfg, voc)
train_dataset, val_dataset, test_dataset = data_handler.getDatasets()
train_dataloader, val_dataloader, test_dataloader = data_handler.getDataloader(train_dataset, val_dataset, test_dataset)

# model
logging.info('Loading Model...')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_encoder = EfficientNetEc().to(device)  # output size: torch.Size([1, 1280])
model = S2VT(cfg, len(voc)).to(device)

# loss
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

# initialize
loss_history = []
loop = range(cfg.epoch)

# checkpoint
if cfg.use_ckpt:
    logging.info(f"Using checkpoint file for training: '{cfg.ckpt_file}'")
    try:
        checkpoint = torch.load(cfg.ckpt_root + cfg.ckpt_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loop = range(epoch, cfg.epoch)
        loss_history = checkpoint['loss_history']
    except Exception as e:
        logging.info(f"Root '{cfg.ckpt_root + cfg.ckpt_file}' not found...")

# training loop
logging.info('Start training...')
for epoch in loop:
    # iteration of training data
    i = 0
    for vid, (label, mask) in train_dataloader:
        frame_chunk = vid[:cfg.chunk_size].asnumpy()
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

        if i % cfg.iter_plt == 0:
            logging.info(f"Epoch: {epoch + 1}/{cfg.epoch} iteration:{i}/{len(train_dataset)}, Total loss: {loss}, logit_loss: {logit_loss}, sentence length: {torch.sum(mask)}")
            loss_history.append(loss.detach().cpu().numpy())
            plt.plot(np.squeeze(loss_history))
            plt.ylabel('CE loss')
            plt.xlabel(f'iterations (per {cfg.iter_plt})')
            plt.title(f"Learning rate = {cfg.lr}")
            plt.savefig(log_path + 'loss.png')

        if i != 0 and i % cfg.ckpt_save == 0:
            logging.info(f"Save checkpoint at '{cfg.ckpt_root}'")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_history': loss_history,
            },  cfg.ckpt_root + timestamp + '_checkpoint.pth.tar')
        i += 1

    logging.info(f"Save weight for epoch{epoch+1} at '{cfg.weight_root}'")
    torch.save(model.state_dict(), cfg.weight_root + timestamp + f'_epoch_{epoch + 1}.pt')
