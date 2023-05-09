# -*- coding: utf-8 -*-
"""
Created on Mon May  8 17:30:22 2023

@author: chrischris
"""

class MyConfig:
    def __init__(self):
        # path
        self.dataset = 'msrvtt'
        self.vid_root = './dataset/msrvtt/videos/'
        self.train_val_annotation_file = './dataset/msrvtt/train_val_videodatainfo.json'
        self.test_annotation_file = './dataset/msrvtt/test_videodatainfo.json'  
        
        # vocabulary
        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2
        self.UNK_token = 3
        self.encode_num_start = 4
        
        # data
        self.batch_size = 4
        self.num_workers = 0
        
        