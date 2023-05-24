from torchvideotransforms.video_transforms import Compose, Resize, RandomCrop, RandomRotation, ColorJitter, Normalize, RandomHorizontalFlip
from torchvideotransforms.volume_transforms import ClipToTensor
import os
import cv2
from decord import VideoReader
import numpy as np

from torchvideotransforms import video_transforms, volume_transforms
from torchvideotransforms.volume_transforms import ClipToTensor
import os
import cv2
from decord import VideoReader
from torchvideotransforms import video_transforms, volume_transforms
from torchvideotransforms.video_transforms import (
    Compose,
    Resize,
    RandomCrop,
    RandomRotation,
    ColorJitter,
    Normalize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomGrayscale,
    RandomResize,
    RandomResizedCrop,
    CenterCrop



    )
from PIL import Image


# 設定augmentation
video_transform_list = [
       Resize((224, 224)),
       RandomResize((8,8)), #放大
       RandomVerticalFlip(1),
       RandomHorizontalFlip(1),      
       RandomCrop(224),
       RandomResizedCrop(224),
       CenterCrop(128),
      # RandomRotation(30), #輸出黑色
      # RandomGrayscale(),#無法使用
      # ColorJitter(), #無法使用
  ]
video_transform = Compose(video_transform_list)