# -*- coding: utf-8 -*-
"""
Created on Fri June 2 16:59:17 2023
@title: test -  generate BLEU score
"""

import torch
from utils.voc import Vocabulary
from utils.config import MyConfig
from model.efficientnet_ec import EfficientNetEc
from model.s2vt_module_lstm import S2VT

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


