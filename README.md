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
├─  output/
├─  utils/
├─  vocabulary/
│       ├─  msrvtt_index2word_dic.json
│       ├─  msrvtt_word2count_dic.json
│       └─  msrvtt_word2index_dic.json
│   .gitignore
│   config.py       # setting of the whole project
│   dataset.py      # split and preprocess the dataset
│   eval.py         # use metrice and confusion matrix to evaluate the model
│   inference.py    # generate the output
│   LICENSE
│   README.md
│   train.py        # train the model
└─  voc.py          # generate the vocabulary of raw dataset
```
