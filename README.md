Video_Captioning
===

### Folder Structure
```
Program/
├─  dataset/
│      └─  msrvtt/
│            └─  videos/
│                   │  video1.mp4
│                   │      ⋮
│                   └─ video9999.mp4
├─  input/
├─  logs/
├─  model/
│       ├─  bleu.py                 # metric
│       ├─  bleu_scorer.py          # metric utils
│       ├─  efficientnet_ec.py      # CNN encoder
│       └─  s2vt_module_lstm.py     # stacked LSTM
├─  output/
├─  pretrained/
│       ├─  checkpoint/     # training state
│       │      │   date-time_epoch_n_iter_i.ckpt
│       │      │                ⋮
│       │      └─               ⋮
│       └─  weight/         # pretrained model
│              │       date-time_epoch_n.pt
│              │                ⋮
│              └─               ⋮
├─  utils/
│       ├─  config.py       # setting of the whole project
│       ├─  dataset.py      # split and preprocess the dataset
│       ├─  transform.py    # data augmentation
│       └─  voc.py          # generate the vocabulary of raw dataset
├─  vocabulary/
│       ├─  msrvtt_index2word_dic.json
│       ├─  msrvtt_word2count_dic.json
│       └─  msrvtt_word2index_dic.json
│   .gitignore
│   eval.py         # use metrice and confusion matrix to evaluate the model
│   inference.py    # generate the output
│   LICENSE
│   README.md
└─   train.py        # train the model
```
