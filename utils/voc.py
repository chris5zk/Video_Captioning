# -*- coding: utf-8 -*-
"""
Created on Mon May  8 16:20:57 2023
@title : Vocabulary build
"""

import os
import json


class Vocabulary:
    def __init__(self, cfg):
        self.cfg = cfg
        self.name = cfg.dataset
        self.trimmed = False
        self.word2index = {"PAD":cfg.PAD_token, "SOS":cfg.SOS_token, "EOS":cfg.EOS_token, "UNK":cfg.UNK_token}
        self.word2count = {}
        self.index2word = {cfg.PAD_token:"PAD", cfg.EOS_token:"EOS", cfg.SOS_token:"SOS", cfg.UNK_token:"UNK"}
        self.encode = cfg.encode_num_start
    
    def __len__(self):
        return len(self.word2index)
    
    def addSentence(self, sentence): 
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.encode
            if not self.trimmed:
                self.word2count[word] = 1
            self.index2word[self.encode] = word
            self.encode += 1
        else:
            if not self.trimmed:
                self.word2count[word] += 1
                
    def save(self, w2i_file='word2index_dic.json', i2w_file='index2word_dic.json', w2c_file='word2count_dic.json'):
        w2i = os.path.join(self.cfg.voc, self.name + '_' + w2i_file)
        i2w = os.path.join(self.cfg.voc, self.name + '_' + i2w_file)
        w2c = os.path.join(self.cfg.voc, self.name + '_' + w2c_file)       
        try:
            with open(w2i, 'w') as fp:
                json.dump(self.word2index, fp, indent=4)

            with open(i2w, 'w') as fp:
                json.dump(self.index2word, fp, indent=4)

            with open(w2c, 'w') as fp:
                json.dump(self.word2count, fp, indent=4)
        except Exception as e:
            print(f'Path Error, Verify the path of the filename is correct: {e}')

    def load(self, w2i_file='word2index_dic.json', i2w_file='index2word_dic.json', w2c_file='word2count_dic.json'):
        w2i = os.path.join(self.cfg.voc, self.name + '_' + w2i_file)
        i2w = os.path.join(self.cfg.voc, self.name + '_' + i2w_file)
        w2c = os.path.join(self.cfg.voc, self.name + '_' + w2c_file)
        try:        
            with open(w2i, 'r') as fp:
                self.word2index = json.load(fp)
            with open(i2w, 'r') as fp:
                self.index2word = json.load(fp)
            with open(w2c, 'r') as fp:
                self.word2count = json.load(fp)
            self.encode = len(self.word2index)
        except:
            print('File loading error.. check the path or filename is correct')

    def trim(self):
        if self.trimmed:
            print('Already trimmed before.')
            return 0
        self.trimmed = True
        
        keep_words = []
        for k, v in self.word2count.items():
            if v > self.cfg.min_count:
                keep_words.append(k)
        
        print('keep_words {} / {} = {:.4f}'.format(len(keep_words), len(self.word2index)-4, len(keep_words) / (len(self.word2index)-4)))
        
        # Reinitialize dictionaries
        self.word2index = {"PAD":self.cfg.PAD_token, "SOS":self.cfg.SOS_token, "EOS":self.cfg.EOS_token, "UNK":self.cfg.UNK_token}
        self.index2word = {self.cfg.PAD_token:"PAD", self.cfg.EOS_token:"EOS", self.cfg.SOS_token:"SOS", self.cfg.UNK_token:"UNK"}
        self.encode = self.cfg.encode_num_start
        
        new_count = {}
        for word in keep_words:
            self.addWord(word)
            new_count[word] = self.word2count[word]        
        self.word2count = new_count

if __name__ == '__main__':
    
    # from dataset import DataHandler
    from config import MyConfig

    cfg = MyConfig()
    cfg.train_val_annotation_file = '../dataset/msrvtt/train_val_videodatainfo.json'
    cfg.test_annotation_file = '../dataset/msrvtt/test_videodatainfo.json'
    cfg.vid_root = '../dataset/msrvtt/videos/'
    cfg.voc = '../vocabulary'
    
    voc = Vocabulary(cfg)
    voc.load()
    # text_dict = {}
    # data_handler = DataHandler(cfg, voc)
    
    # text_dict.update(data_handler.train_dict)
    # text_dict.update(data_handler.val_dict)
    # text_dict.update(data_handler.test_dict)
    
    # for k,v in text_dict.items():
    #     for anno in v:
    #         voc.addSentence(anno)
    # voc.save()
    voc.trim(cfg.min_count)
