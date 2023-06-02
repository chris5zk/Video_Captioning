# -*- coding: utf-8 -*-
"""S2VT Module LSTM.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17xudoJI6JM_0Gut6j0Gfq0a3iTXQmnEM
"""

import torch
import torch.nn as nn

vocab_size = 1000  # Example vocabulary size
hidden_size = 256  # Example hidden size
batch_size = 1  # Example batch size

# Creating an instance of the embedding layer
embedding = nn.Embedding(vocab_size, hidden_size)

# Generating random bos_id tensor
bos_id = torch.randint(0, vocab_size, (batch_size,))

# Embedding the bos_id tensor
embedded_bos = embedding(bos_id)  # shape: (batch_size, hidden_size)

# print(embedded_bos)  # Output: torch.Size([1, 256])
# print(embedded_bos.shape)  # Output: torch.Size([1, 256])

# def build_vocab(word_count_threshold, unk_requried=False):
#     sentance_train = open(TXT_DIR+'sents_train_lc_nopunc.txt', 'r').read().splitlines()
#     sentance_val = open(TXT_DIR+'sents_val_lc_nopunc.txt', 'r').read().splitlines()
#     sentance_test = open(TXT_DIR+'sents_test_lc_nopunc.txt', 'r').read().splitlines()
#     all_captions = []
#     word_counts = {}
#     for cap in sentance_test+sentance_train+sentance_val:
#         caption = cap.split('\t')[-1]
#         caption = '<BOS> ' + caption + ' <EOS>'
#         all_captions.append(caption)
#         for word in caption.split(' '):
#             if word in word_counts:
#                 word_counts[word] += 1
#             else:
#                 word_counts[word] = 1
#         for word in word_counts:
#             if word_counts[word] < word_count_threshold:
#                 word_counts.pop(word)
#                 unk_requried = True
#     return word_counts, unk_requried


# def word_to_ids(word_counts, unk_requried):
#     word_to_id = {}
#     id_to_word = {}
#     count = 0
#     if unk_requried:
#         word_to_id['<UNK>'] = count
#         id_to_word[count] = '<UNK>'
#         count += 1
#         print("<UNK> True")
#     for word in word_counts:
#         word_to_id[word] = count
#         id_to_word[count] = word
#         count += 1
#     return word_to_id, id_to_word

# word_counts, unk_required = build_vocab(word_count_threshold=0)
# word2id, id2word = word_to_ids(word_counts, unk_requried=unk_required)

import torch
from torch import nn


class S2VT(nn.Module):
    def __init__(self, cfg, vocab_size):
        super(S2VT, self).__init__()
        self.batch_size = 1
        self.frame_dim = cfg.frame_dim
        self.hidden_size = cfg.hidden_size
        self.n_steps = cfg.length    # This will depend on caption sequence length
    
        self.drop = nn.Dropout(p=cfg.dropout)
        self.linear1 = nn.Linear(self.frame_dim, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, vocab_size)
        
        # At 2nd stage LSTM will concat output from 1st LSTM and the Padding
        self.lstm1 = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True, dropout=cfg.dropout)
        self.lstm2 = nn.LSTM(self.hidden_size*2, self.hidden_size, batch_first=True, dropout=cfg.dropout)     

        self.embedding = nn.Embedding(vocab_size, self.hidden_size)

    def forward(self, video, caption=None):
        # Video shape is (Batch, Frames, Extracted_features)
        video = video.contiguous().view(-1, self.frame_dim)
        video = self.drop(video)
        video = self.linear1(video)     # Video Embed to lower dimension(hidden size) output
        video = video.view(-1, self.n_steps, self.hidden_size)  # Video shape now = (Batch, n_steps, hidden_size)
        padding = torch.zeros([self.batch_size, self.n_steps-1, self.hidden_size]).cuda()
 
        # Video Input
        video = torch.cat((video, padding), 1)  # Video input shape = (Batch, 2*n_steps-1, hidden_size)
        vid_out, state_vid = self.lstm1(video)  # Vid_out = (B, n_steps, hidden) State_vid = (hidden_out, Cell_out) = (num_layers, Batch, hidden_size)

        if self.training:   # Default True when model.train()
            caption = self.embedding(caption[:, 0:self.n_steps-1])  # After embedding, caption shape = (Batch, n_steps-1, hidden_size)
            padding = torch.zeros([self.batch_size, self.n_steps, self.hidden_size]).cuda()
            # Caption Padding
            caption = torch.cat((caption, padding), 1)
            # Caption Input 2*hidden_size
            caption = torch.cat((caption, vid_out), 2) 
    
            cap_out, state_cap = self.lstm2(caption)    # Cap_out shape = (Batch, 2*n_steps-1, hidden_size)
            cap_out = cap_out[:, self.n_steps:, :] 
            cap_out = cap_out.contiguous().view(-1, self.hidden_size)    # Caption shape (Batch * n_time_steps-1, hidden_size)
            cap_out = self.drop(cap_out) 
            cap_out = self.linear2(cap_out)     # Linear transform mapping to the vocabulary space (Batch * n_time_steps-1, vocab_size)
            # Return Cap_out for loss computation in training
            return cap_out 
    
        else: 
            # padding input of the second layer of LSTM, 40 time steps
            padding = torch.zeros([self.batch_size, self.n_step, self.hidden]).cuda()
            cap_input = torch.cat((padding, vid_out[:, 0:self.n_step, :]), 2)
            cap_out, state_cap = self.lstm2(cap_input)
            
            bos_id = word2id['<BOS>']*torch.ones(self.batch_size, dtype=torch.long)
            bos_id = bos_id.cuda()
            cap_input = self.embedding(bos_id)
            cap_input = torch.cat((cap_input, vid_out[:, self.n_step, :]), 1)
            cap_input = cap_input.view(self.batch_size, 1, 2*self.hidden)
      
            # input ["<BOS>"] to let the generate start
            cap_out, state_cap = self.lstm2(cap_input, state_cap)
            cap_out = cap_out.contiguous().view(-1, self.hidden)
            cap_out = self.drop(cap_out)
            cap_out = self.linear2(cap_out)
            cap_out = torch.argmax(cap_out, 1)
      
            # put the generate word index in caption list 
            caption = []
            caption.append(cap_out)
      
            # generate one word at one time step for each batch
            for i in range(self.n_step-2): # output n_steps-2 words
                cap_input = self.embedding(cap_out)
                cap_input = torch.cat((cap_input, vid_out[:, self.n_step+1+i, :]), 1)
                cap_input = cap_input.view(self.batch_size, 1, 2 * self.hidden)
        
                cap_out, state_cap = self.lstm2(cap_input, state_cap)
                cap_out = cap_out.contiguous().view(-1, self.hidden)
                cap_out = self.drop(cap_out)
                cap_out = self.linear2(cap_out)
                cap_out = torch.argmax(cap_out, 1)
                
                # get the index of each word in vocabulary
                caption.append(cap_out)
      
            return caption
      # size of caption is [79, batch_size]
