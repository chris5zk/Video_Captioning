Video_Captioning
===

### Folder Structure
```
Program/
├─  dataset/
│      └─  msrvtt/
│            └─  videos/
│            │      │  video1.mp4
│            │      │      ⋮
│            │      └─ video9999.mp4
│            ├─  categoty.txt
│            ├─  msrvtt_label_test.json       # generate by ./utils/label_generate.py
│            ├─  msrvtt_label_test.json       # generate by ./utils/label_generate.py
│            ├─  msrvtt_label_test.json       # generate by ./utils/label_generate.py
│            ├─  readme.txt
│            ├─  test_videodatainfo.json      # msrvtt raw data
│            └─  train_val_videodatainfo.json # msrvtt raw data
├─  input/                  # input video
├─  logs/
├─  model/
│       ├─  efficientnet_ec.py      # CNN encoder
│       └─  s2vt_module_lstm.py     # stacked LSTM
├─  output/                 # output caption
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
│       ├─  config.py               # setting of the whole project
│       ├─  dataset.py              # split and preprocess the dataset
│       ├─  label_generate.py       # reconstruct raw data to another format
│       ├─  log.py                  # create logger
│       ├─  metrics.py              # evaluate metrics
│       ├─  transform.py            # data augmentation
│       └─  voc.py                  # generate the vocabulary of raw dataset
├─  vocabulary/
│       ├─  msrvtt_index2word_dic.json
│       ├─  msrvtt_word2count_dic.json
│       └─  msrvtt_word2index_dic.json
│   .gitignore
│   eval.py           # use metrice and confusion matrix to evaluate the model
│   inference.py      # generate the output
│   LICENSE
│   README.md
│   requirement.txt   # requirement package
└─   train.py         # train the model
```

Environment
---
Run `pip install -r requirements.txt` to install usage package.
```
Package List
- PyTorch
- Numpy
- tqdm
- decord
- matplotlib
- json
```

How to run?
---
### Training
1. Download the MSR-VTT videos dataset for training. (https://www.mediafire.com/folder/h14iarbs62e7p/shared)
2. Ensure your path setting and folder structure are proper, please change the path and root in `./utils/config.py` .
3. Run `./utils/label_generate.py` to reconstruct the data dictionary.
4. Run `train.py`.

### Evaluate
1. Run `eval.py`
2. Evaluate metrics will generate at the command window and predict captions will save as `./output/val_pred.json`

### Inference
1. Run `eval.py`
2. Predict captions will save as `./output/test_pred.json`

Acknowledgement
---
- Dataset process: https://github.com/nasib-ullah/video-captioning-models-in-Pytorch/tree/main/MSRVTT/captions
- Model reference: https://github.com/YiyongHuang/S2VT
- Metrics: https://github.com/wangleihitcs/CaptionMetrics
