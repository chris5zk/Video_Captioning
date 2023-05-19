import torch
from PIL import Image
from torchvision.transforms import (Compose, RandomRotation, Resize, RandomCrop,
                                    ColorJitter, ToTensor, Normalize)
from voc import Vocabulary
from config import MyConfig
from dataset import DataHandler
from model.efficientnet_ec import EfficientNetEc

# load config file
cfg = MyConfig()

# vocabulary object
voc = Vocabulary(cfg)
voc.load()

# load dataset
data_handler = DataHandler(cfg, voc)
train_dataset, val_dataset, _ = data_handler.getDatasets()
train_dataloader, val_dataloader, _ = data_handler.getDataloader(train_dataset, val_dataset, _)

# data augmentation
train_transform = Compose([
        RandomRotation(30),
        Resize(256),
        RandomCrop((224, 224)),
        ColorJitter(0.5, 0.5, 0.5, 0.25),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# load model
model_ec = EfficientNetEc() # output size: torch.Size([1, 1280])



# training loop
for epoch in range(cfg.epoch):
    # iteration of training data
    for vid, label in train_dataloader:
        # iteration of video
        for frame_idx in range(len(vid)-cfg.chunk_size-1):
            frame_chunk = vid[frame_idx:frame_idx+cfg.chunk_size].asnumpy()
            # iteration of frame chunk
            for frame in frame_chunk:
                frame = train_transform(Image.fromarray(frame))
                feature_ec = model_ec(frame[None, :, :, :])
            break
        break
