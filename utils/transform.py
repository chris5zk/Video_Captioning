# -*- coding: utf-8 -*-
"""
Created on Sat May 20 01:25:07 2023
@title: Data Augmentation
"""

from torchvision.transforms import (Compose, RandomRotation, Resize, RandomCrop,
                                    ColorJitter, ToTensor, Normalize, ToPILImage)

train_transform = Compose([
        ToPILImage(),
        RandomRotation(30),
        Resize(256),
        RandomCrop((224, 224)),
        ColorJitter(0.5, 0.5, 0.5, 0.25),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

test_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])