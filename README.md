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
│            ├─  msrvtt_label_train.json      # generate by ./utils/label_generate.py
│            ├─  msrvtt_label_val.json        # generate by ./utils/label_generate.py
│            ├─  msrvtt_label_test.json       # generate by ./utils/label_generate.py
│            ├─  readme.txt
│            ├─  test_videodatainfo.json      # msrvtt raw data
│            └─  train_val_videodatainfo.json # msrvtt raw data
├─  input/     # input video (useless now)
├─  logs/      # log file of training stage
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
│   eval.py           # use metrice and confusion matrix to evaluate the model (evaluate for validation set)
│   inference.py      # generate the output (test for testing set)
│   LICENSE
│   README.md
│   requirement.txt   # requirement package
└─  train.py          # train  model
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

Completion Progress: 80%
---
- [X] Pre-process MSRVTT dataset
- [X] Building model: VGG16 and stacked LSTM
- [X] Proper caption in training stage
- [ ] Proper caption in testing stage

### Unsolved Problem
At testing and validation stage, decoder without ground true can't caption the video properly. So at `inference.py` and `eval.py`, we use `model.train()` instead of `model.eval()` to generate proper caption. **Please forgive us for this lack of competence, and we hope to be able to successfully solve this problem in the future.**

How to run?
---
### Training
1. Download the MSR-VTT videos dataset for training. (https://www.mediafire.com/folder/h14iarbs62e7p/shared)
2. Ensure your path setting and folder structure are proper, please change the path and root in `./utils/config.py` .
3. Run `./utils/label_generate.py` to reconstruct the raw data dictionary if `msrvtt_label_train.json`, `msrvtt_label_val.json`, and `msrvtt_label_test.json` do not exist.
4. Run `train.py`.

### Evaluate
1. Run `eval.py`
2. `eval.py` will read the validation set from `./dataset/msrvtt/msrvtt_label_val.json` and corresponding videos.
3. Evaluate metrics will generate at the command window and predict captions will save as `./output/val_pred.json`

### Inference
1. Run `eval.py`
2. `inference.py` will **read the testing set from `./dataset/msrvtt/msrvtt_label_test.json` and corresponding videos instead of `./input` videos now.**
3. Predict captions will save as `./output/test_pred.json`

Acknowledgement
---
- Paper: https://openaccess.thecvf.com/content_iccv_2015/html/Venugopalan_Sequence_to_Sequence_ICCV_2015_paper.html
- Dataset process: https://github.com/nasib-ullah/video-captioning-models-in-Pytorch/tree/main/MSRVTT/captions
- Model reference: https://github.com/YiyongHuang/S2VT
- Metrics: https://github.com/wangleihitcs/CaptionMetrics
