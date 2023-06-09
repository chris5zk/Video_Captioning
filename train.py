# -*- coding: utf-8 -*-
"""
Created on Sat May 20 01:22:17 2023
@title: training loop
"""

import os
import json
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from tqdm import tqdm
from utils.voc import Vocabulary
from utils.config import MyConfig
from utils.metrics import EvalMetrics
from utils.dataset import DataHandler
from utils.log import create_logger
from utils.transform import train_transform
from model.cnn_encoder import EfficientNetEc, VGG16
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

train_dataset = data_handler.train_dataset()
train_dataloader = data_handler.train_loader(train_dataset)

val_dataset = data_handler.val_dataset()
val_dataloader = data_handler.val_loader(val_dataset)

# model
logging.info('Loading Model...')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_encoder = VGG16().to(device)             # output size: torch.Size([1, 4096])
model = S2VT(cfg, len(voc)).to(device)

# loss
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

# initialize
loss_history = []
val_loss_history = []
loop = range(cfg.epoch)

# checkpoint
if cfg.use_ckpt:
    logging.info(f"Using checkpoint file for training: '{cfg.ckpt_file}'")
    try:
        checkpoint = torch.load(cfg.ckpt_root + cfg.ckpt_file, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1
        loop = range(epoch, cfg.epoch)
        loss_history = checkpoint['loss_history']
        val_masked_loss = checkpoint['val_loss_history']
    except Exception as e:
        print(e)
        logging.info(f"Root '{cfg.ckpt_root + cfg.ckpt_file}' not found...")

# training loop
logging.info('Start training...')
for epoch in loop:
    # iteration of training data
    i = 0
    loss_acc = 0
    val_loss_acc = 0
    res = {}
    val_res = {}
    pred_dict = {}
    val_pred_dict = {}
    metrics = EvalMetrics()

    for vid, (video_id, sen_id, label, mask) in train_dataloader:
        frame_chunk = vid.get_batch(np.linspace(1, len(vid)-1, cfg.chunk_size).astype('int')).asnumpy()
        lstm_input = torch.zeros((1, cfg.chunk_size, cfg.frame_dim)).to(device)
        # cnn encode
        for idx, frame in enumerate(frame_chunk):
            frame = train_transform(frame).to(device)
            feature_ec = cnn_encoder(frame[None, :, :, :])  # torch.Size([1, 25088])
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
        loss.backward()
        optimizer.step()

        if i != 0 and i % cfg.train_info == 0:
            logging.info(f"Epoch: {epoch + 1}/{cfg.epoch} iteration:{i}/{len(train_dataset)}, Total loss: {loss}, logit_loss: {logit_loss}, sentence length: {torch.sum(mask)}")

        if i != 0 and i % cfg.loss_interval == 0:
            loss_acc += loss.detach().cpu().numpy()

        if i != 0 and i % cfg.mean_loss == 0:
            mean_loss = loss_acc / (cfg.mean_loss/cfg.loss_interval)
            loss_history.append(mean_loss)
            logging.info(f"mean loss: {mean_loss} (mean of {int(cfg.mean_loss/cfg.loss_interval)} loss, and one loss per {cfg.loss_interval} iter)")
            plt.plot(np.squeeze(loss_history))
            plt.ylabel('CE loss')
            plt.xlabel(f'iterations (per {cfg.mean_loss})')
            plt.title(f"Learning rate = {cfg.lr}")
            plt.savefig(log_path + f'{timestamp.split("-")[1]}_loss.png')
            loss_acc = 0

        if i != 0 and i % cfg.ckpt_save == 0:
            logging.info(f"Save checkpoint at '{cfg.ckpt_root}'")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_history': loss_history,
                'val_loss_history': val_loss_history
            },  cfg.ckpt_root + timestamp + '_checkpoint.pth.tar')

        if i != 0 and i % cfg.weight_save == 0:
            logging.info(f"Save weight for epoch{epoch+1}_{i} at '{cfg.weight_root}'")
            torch.save(model.state_dict(), cfg.weight_root + timestamp + f'_epoch_{epoch + 1}_{i}.pt')

        i += 1

    logging.info(f"Save weight for epoch{epoch+1} at '{cfg.weight_root}'")
    torch.save(model.state_dict(), cfg.weight_root + timestamp + f'_epoch_{epoch + 1}.pt')

    if epoch >= cfg.start_val and cfg.validation:
        logging.info(f'Epoch {epoch+1} Validation...')
        j = 0
        model.eval()
        for vid, (video_id, sen_id, label, mask) in tqdm(val_dataloader):
            frame_chunk = vid.get_batch(np.linspace(1, len(vid)-1, cfg.chunk_size).astype('int')).asnumpy()
            lstm_input = torch.zeros((1, cfg.chunk_size, cfg.frame_dim)).to(device)
            # cnn encode
            for idx, frame in enumerate(frame_chunk):
                frame = train_transform(frame).to(device)
                feature_ec = cnn_encoder(frame[None, :, :, :])  # torch.Size([1, 25088])
                lstm_input[0, idx] = feature_ec

            # lstm
            caption = torch.tensor(label).view(-1, len(label)).to(device)
            mask = torch.tensor(mask).to(device)
            cap_out, probs = model(lstm_input)

            cap_labels = caption[:, 1:].contiguous().view(-1)
            cap_mask = mask[:, 1:].contiguous().view(-1)

            val_logit_loss = loss_func(probs, cap_labels)
            val_masked_loss = val_logit_loss * mask
            val_loss = torch.sum(val_masked_loss) / torch.sum(mask)

            if j != 0 and j % 5 == 0:
                val_loss_acc += val_loss.detach().cpu().numpy()
            if j != 0 and j % 99 == 0:
                val_mean_loss = val_loss_acc / 19
                val_loss_history.append(val_mean_loss)
                logging.info(f"mean val loss: {val_mean_loss} (mean of 19 loss, and one loss per 5 iter)")
                plt.plot(np.squeeze(val_loss_history))
                plt.ylabel('CE loss')
                plt.xlabel(f'iterations (per 5)')
                plt.title(f"Learning rate = {cfg.lr} (validation)")
                plt.savefig(log_path + f'{timestamp.split("-")[1]}_val_loss.png')
                val_loss_acc = 0
            j += 1
        model.train()
