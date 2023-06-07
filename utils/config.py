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
        self.dataset_root = './dataset/msrvtt/'
        self.vid_root = './dataset/msrvtt/videos/'
        self.train_val_annotation_file = './dataset/msrvtt/train_val_videodatainfo.json'
        self.test_annotation_file = './dataset/msrvtt/test_videodatainfo.json'  
        self.weight_root = './pretrained/weight/'
        self.ckpt_root = './pretrained/checkpoint/'
        self.weight_file = ''
        self.ckpt_file = ''
        self.log_root = './logs/'
        self.input = './input/'
        self.output = './output/'
        self.i2w = './vocabulary/msrvtt_index2word_dic.json'
        self.eval_file = './logs/230605/134546_pred_epoch_19.json'

        # vocabulary
        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2
        self.UNK_token = 3
        self.encode_num_start = 4
        
        # data
        self.input_size = None
        self.chunk_size = 30
        self.length = 30        # max 73
        self.min_count = 1
        
        # train
        self.use_ckpt = False
        self.train_info = 20        # print info
        self.loss_interval = 20     # acc loss
        self.mean_loss = 1000       # plot
        self.ckpt_save = 300
        self.weight_save = 2000
        self.pred_save = 4

        self.num_workers = 0
        self.epoch = 30
        self.lr = 0.0001
        
        # S2VT model
        self.dropout = 0.5
        self.hidden_size = 2000
        self.frame_dim = 4096

        # validation
        self.validation = True
        self.start_val = 24

