# -*- coding: utf-8 -*-
"""
Created on Mon May  8 16:20:57 2023
@title:  VC dataset
@author: chrischris
Read the videos(?):
    - define a CustomDataset (__init__, __getitem__: random choose one caption from dict each iter only)
    - use DataLoader to load data into model
    - how to load video using dataloader??????
"""

import json
import random

class MyDataset:
    def __init__(self, cfg, annotation_dict, video_name_list, voc):
        self.cfg = cfg
        self.annotation_dict = annotation_dict
        self.video_name_list = video_name_list
        self.voc = voc
    
    def __len__(self):
        return len(self.video_name_list)
    
    def __getitem____(self, idx):
        anno = random.choice(self.annotation_dict[self.video_name_list[idx]])
        anno_idx = []
        for word in anno.split(' '):
            try:
                anno_idx.append(self.voc.word2index[word])
            except:
                pass
        anno_idx = anno_idx + [self.cfg.EOS_token]
    
        return anno_idx, self.video_name_list[idx]

class DataHandler:
    def __init__(self, cfg, voc):
        self.voc = voc
        self.cfg = cfg
        self.train_dict, self.val_dict, self.test_dict = self.msrvtt_create_dict()

        self.train_name_list = list(self.train_dict.keys())
        self.val_name_list = list(self.val_dict.keys())
        self.test_name_list = list(self.test_dict.keys())

    def msrvtt_create_dict(self):
        train_val_file = json.load(open(self.cfg.train_val_annotation_file))
        test_file = json.load(open(self.cfg.test_annotation_file))
        train_dict, val_dict, test_dict = {}, {}, {}
        train_id_list = list(range(0, 6513))
        val_id_list = list(range(6513, 7010))
        test_id_list = list(range(7010, 10000))

        for data in train_val_file['sentences']:
            # training data caption
            if int(data['video_id'][5:]) in train_id_list:
                if data['video_id'] in list(train_dict.keys()):
                    train_dict[data['video_id']] += [data['caption']]
                else:
                    train_dict[data['video_id']] = [data['caption']]
            # validation data caption
            if int(data['video_id'][5:]) in val_id_list:
                if data['video_id'] in list(val_dict.keys()):
                    val_dict[data['video_id']] += [data['caption']]
                else:
                    val_dict[data['video_id']] = [data['caption']]

        for data in test_file['sentences']:
            # testing data caption
            if int(data['video_id'][5:]) in test_id_list:
                if data['video_id'] in list(test_dict.keys()):
                    test_dict[data['video_id']] += [data['caption']]
                else:
                    test_dict[data['video_id']] = [data['caption']]

        return train_dict, val_dict, test_dict

    def getDatasets(self):
        train_dataset = MyDataset(self.cfg, self.train_dict, self.train_name_list, self.voc)
        val_dataset = MyDataset(self.cfg, self.val_dict, self.val_name_list, self.voc)
        test_dataset = MyDataset(self.cfg, self.test_dict, self.test_name_list, self.voc)
        
        return train_dataset, val_dataset, test_dataset
    
    
    def getDataloader(self):
        pass


if __name__ == '__main__':
    
    from config import MyConfig
    from voc import Vocabulary
    
    cfg = MyConfig()
    voc = Vocabulary(cfg)
    data_handler = DataHandler(cfg, voc)

