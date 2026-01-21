import torch
import torch.nn as nn
from torchvision import models

class Resnet18:
    def __init__(self):
        pass

    def get_custom_resnet(self,input_channels=1,num_classes=7,use_pretrained=1):
        if use_pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            # with torch.no_grad():
            #     old_weights = model.conv1.weight.clone()
            #     model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            #     model.conv1.weight = nn.Parameter(old_weights.mean(dim=1, keepdim=True))
        else:
            model = models.resnet18(weights=None)
            model.conv1 = nn.Conv2d(input_channels,64,kernel_size=7,stride=2,padding=3,bias=False)

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        return model