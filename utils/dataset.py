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
        anno = random.choice(self.annotation_dict[self.video_name_list[idx]])
        anno_idx = []
        for word in anno.split(' '):
            try:
                anno_idx.append(self.voc.word2index[word])
            except:
                anno_idx.append(self.cfg.UNK_token)

        nWord = len(anno_idx) + 1
        anno_idx = [self.cfg.SOS_token] + anno_idx + [self.cfg.EOS_token]
        cap_mask = [[1.0] * nWord + [0.0] * (self.cfg.length - nWord)]

        for i in range(len(anno_idx), self.cfg.length):
            anno_idx.append(self.cfg.EOS_token)

        vid = VideoReader(self.cfg.vid_root + self.video_name_list[idx] + '.mp4')

        return vid, (anno_idx, cap_mask)


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
            if int(data['video_id'][5:]) in train_id_list and len(data['caption'].split(' ')) <= self.cfg.length - 2:
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
            if int(data['video_id'][5:]) in test_id_list and len(data['caption'].split(' ')) <= self.cfg.length - 2:
                if data['video_id'] in list(test_dict.keys()):
                    test_dict[data['video_id']] += [{data['sen_id']: data['caption']}]
                else:
                    test_dict[data['video_id']] = [{data['sen_id']: data['caption']}]

        return train_dict, val_dict, test_dict

    def getDatasets(self, mode='train'):
        if mode == 'train':
            train_dataset = MyDataset(self.cfg, self.train_dict, self.train_name_list, self.voc)
            val_dataset = MyDataset(self.cfg, self.val_dict, self.val_name_list, self.voc)
            return train_dataset, val_dataset
        elif mode == 'test':
            test_dataset = MyDataset(self.cfg, self.test_dict, self.test_name_list, self.voc)
            return test_dataset

    def getDataloader(self, mode='train', train=None, val=None, test=None):
        if mode == 'train':
            train_loader = DataLoader(train, batch_size=1, num_workers=self.cfg.num_workers, shuffle=True, collate_fn=my_collate_fn)
            val_loader = DataLoader(val, batch_size=1, num_workers=self.cfg.num_workers, shuffle=False, collate_fn=my_collate_fn)
            return train_loader, val_loader
        elif mode == 'test':
            test_loader = DataLoader(test, batch_size=1, num_workers=self.cfg.num_workers, shuffle=False, collate_fn=my_collate_fn)
            return test_loader


if __name__ == '__main__':

    from tqdm import tqdm
    from config import MyConfig
    from voc import Vocabulary

    cfg = MyConfig()
    cfg.train_val_annotation_file = '../dataset/msrvtt/train_val_videodatainfo.json'
    cfg.test_annotation_file = '../dataset/msrvtt/test_videodatainfo.json'
    cfg.vid_root = '../dataset/msrvtt/videos/'
    cfg.voc = '../vocabulary'

    voc = Vocabulary(cfg)
    voc.load()
    data_handler = DataHandler(cfg, voc)
    test_dst = data_handler.getDatasets(mode='test')
    test_loader = data_handler.getDataloader(mode='test', test=test_dst)

    for batch in tqdm(test_loader):
        vid, label = batch
