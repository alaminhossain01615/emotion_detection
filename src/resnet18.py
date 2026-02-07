import torch
import torch.nn as nn
from torchvision import models

class Resnet18:
    def __init__(self):
        pass

    def get_custom_resnet(self,input_channels=1,num_classes=7,use_pretrained=1):
        if use_pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            model = models.resnet18(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        return model