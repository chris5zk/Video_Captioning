# -*- coding: utf-8 -*-
"""
Created on Mon May  8 17:30:22 2023

@author: chrischris
"""


class MyConfig:
    def __init__(self):
        # path
        self.voc = './vocabulary'
        self.dataset = 'msrvtt'
        self.vid_root = './dataset/msrvtt/videos/'
        self.train_val_annotation_file = './dataset/msrvtt/train_val_videodatainfo.json'
        self.test_annotation_file = './dataset/msrvtt/test_videodatainfo.json'  
        self.weight_root = './pretrained/weight/'
        self.weight_file = ''
        self.ckpt_root = './pretrained/checkpoint/'
        self.ckpt_file = '20230602-152635_epoch_1_iter_0.ckpt'
        self.log_root = './logs/'

        # vocabulary
        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2
        self.UNK_token = 3
        self.encode_num_start = 4
        
        # data
        self.input_size = None
        self.chunk_size = 40
        self.length = 40        # max 73
        self.min_count = 1
        
        # train
        self.use_ckpt = False
        self.num_workers = 0
        self.epoch = 5
        self.lr = 0.0001
        
        # S2VT model
        self.dropout = 0.5
        self.hidden_size = 128
        self.frame_dim = 1280
