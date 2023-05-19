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


if __name__ == '__main__':

    input_size = 360
    input_image = torch.randn(1, 3, input_size, input_size)

    model = EfficientNetEc()
    features = model(input_image)
    print("Features shape:", features.shape)
