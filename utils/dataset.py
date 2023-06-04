# -*- coding: utf-8 -*-
"""
Created on Mon May  8 16:20:57 2023
@title:  Video Captioning dataset
"""

import json
import random

from decord import VideoReader
from torch.utils.data import DataLoader, Dataset


def my_collate_fn(batch):
    return batch[0][0], batch[0][1]


class MyDataset(Dataset):
    def __init__(self, cfg, annotation_dict, video_name_list, voc):
        self.cfg = cfg
        self.annotation_dict = annotation_dict
        self.video_name_list = video_name_list
        self.voc = voc

    def __len__(self):
        return len(self.video_name_list)

    def __getitem__(self, idx):
        sen_id, caption = random.choice(list(self.annotation_dict[self.video_name_list[idx]].items()))

        anno_idx = []
        for word in caption.split(' '):
            try:
                anno_idx.append(self.voc.word2index[word])
            except:
                anno_idx.append(self.cfg.UNK_token)

        nWord = len(anno_idx) - 2
        cap_mask = [[1.0] * nWord + [0.0] * (self.cfg.length - nWord)]

        for i in range(len(anno_idx), self.cfg.length):
            anno_idx.append(self.cfg.EOS_token)

        vid = VideoReader(self.cfg.vid_root + self.video_name_list[idx] + '.mp4')

        return vid, (sen_id, anno_idx, cap_mask)


class DataHandler:
    def __init__(self, cfg, voc):
        self.voc = voc
        self.cfg = cfg

    def train_dataset(self):
        train_dict = json.load(open(self.cfg.dataset_root + 'msrvtt_label_train.json'))
        train_name_list = list(train_dict.keys())
        train_dataset = MyDataset(self.cfg, train_dict, train_name_list, self.voc)
        return train_dataset

    def val_dataset(self):
        val_dict = json.load(open(self.cfg.dataset_root + 'msrvtt_label_val.json'))
        val_name_list = list(val_dict.keys())
        val_dataset = MyDataset(self.cfg, val_dict, val_name_list, self.voc)
        return val_dataset

    def test_dataset(self):
        test_dict = json.load(open(self.cfg.dataset_root + 'msrvtt_label_test.json'))
        test_name_list = list(test_dict.keys())
        test_dataset = MyDataset(self.cfg, test_dict, test_name_list, self.voc)
        return test_dataset

    def train_loader(self, train_dataset):
        return DataLoader(train_dataset, batch_size=1, num_workers=self.cfg.num_workers, shuffle=True, collate_fn=my_collate_fn)

    def val_loader(self, val_dataset):
        return DataLoader(val_dataset, batch_size=1, num_workers=self.cfg.num_workers, shuffle=False, collate_fn=my_collate_fn)
    
    def test_loader(self, test_dataset):
        return DataLoader(test_dataset, batch_size=1, num_workers=self.cfg.num_workers, shuffle=False, collate_fn=my_collate_fn)


if __name__ == '__main__':

    from config import MyConfig
    from voc import Vocabulary

    cfg = MyConfig()
    cfg.train_val_annotation_file = '../dataset/msrvtt/train_val_videodatainfo.json'
    cfg.test_annotation_file = '../dataset/msrvtt/test_videodatainfo.json'
    cfg.vid_root = '../dataset/msrvtt/videos/'
    cfg.dataset_root = '../dataset/msrvtt/'
    cfg.voc = '../vocabulary'

    voc = Vocabulary(cfg)
    voc.load()
    data_handler = DataHandler(cfg, voc)

    train_dataset = data_handler.train_dataset()
    train_loader = data_handler.train_loader(train_dataset)

    for vid, (sen_id, anno_idx, cap_mask) in train_loader:
        print(sen_id, anno_idx, cap_mask)
