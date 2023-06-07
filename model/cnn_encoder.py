# -*- coding: utf-8 -*-
"""
Created on Sat May 20 01:22:17 2023
@title: model - EfficientNet-b0
"""

import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNetEc(nn.Module):
    def __init__(self):
        super(EfficientNetEc, self).__init__()
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        self.efficientnet.classifier = nn.Identity()

    def forward(self, x):
        output = self.efficientnet(x)   # torch.Size([1, 1280])
        return output


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.VGG16 = models.vgg16(pretrained=True)
        self.VGG16.classifier[6] = nn.Identity()
        
    def forward(self, x):
        output = self.VGG16(x)   # torch.Size([1, 1280])
        return output


if __name__ == '__main__':

    input_size = 224
    input_image = torch.randn(30, 3, input_size, input_size).cuda()

    # model = EfficientNetEc().cuda()
    # features = model(input_image)
    # print("Features shape:", features.shape)
    
    model = VGG16().cuda()
    features = model(input_image)
    print(model)
    print("Features shape:", features.shape)
