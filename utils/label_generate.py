# -*- coding: utf-8 -*-
"""
Created on Mon June 4 15:20:57 2023
@title:  dataset label with embedding
"""

import os
import json


class label_generator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_dict, self.val_dict, self.test_dict = self.msrvtt_create_dict()

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
                label = 'SOS ' + data['caption'] + ' EOS'
                if data['video_id'] in list(train_dict.keys()):
                    train_dict[data['video_id']][data['sen_id']] = label
                else:
                    train_dict[data['video_id']] = {}
                    train_dict[data['video_id']][data['sen_id']] = label

            # validation data caption
            if int(data['video_id'][5:]) in val_id_list and len(data['caption'].split(' ')) <= self.cfg.length - 2:
                label = 'SOS ' + data['caption'] + ' EOS'
                if data['video_id'] in list(val_dict.keys()):
                    val_dict[data['video_id']][data['sen_id']] = label
                else:
                    val_dict[data['video_id']] = {}
                    val_dict[data['video_id']][data['sen_id']] = label

        for data in test_file['sentences']:
            # test data caption
            if int(data['video_id'][5:]) in test_id_list and len(data['caption'].split(' ')) <= self.cfg.length - 2:
                label = 'SOS ' + data['caption'] + ' EOS'
                if data['video_id'] in list(test_dict.keys()):
                    test_dict[data['video_id']][data['sen_id']] = label
                else:
                    test_dict[data['video_id']] = {}
                    test_dict[data['video_id']][data['sen_id']] = label

        return train_dict, val_dict, test_dict

    def save(self):
        label_train = os.path.join(self.cfg.dataset_root, self.cfg.dataset + '_label_train.json')
        label_val = os.path.join(self.cfg.dataset_root, self.cfg.dataset + '_label_val.json')
        label_test = os.path.join(self.cfg.dataset_root, self.cfg.dataset + '_label_test.json')
        try:
            with open(label_train, 'w') as fp:
                json.dump(self.train_dict, fp, indent=4)
            with open(label_val, 'w') as fp:
                json.dump(self.val_dict, fp, indent=4)
            with open(label_test, 'w') as fp:
                json.dump(self.test_dict, fp, indent=4)
        except Exception as e:
            print(f'Path Error, Verify the path of the filename is correct: {e}')


if __name__ == '__main__':

    from config import MyConfig


    cfg = MyConfig()
    cfg.train_val_annotation_file = './dataset/msrvtt/train_val_videodatainfo.json'
    cfg.test_annotation_file = './dataset/msrvtt/test_videodatainfo.json'
    cfg.vid_root = './dataset/msrvtt/videos/'

    lb_gen = label_generator(cfg)
    lb_gen.save()
