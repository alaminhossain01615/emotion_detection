import torch
from torch import nn

class MLP_network(nn.Module):
    def __init__(self,input_size=48*48,num_classes=7):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.relu1=nn.ReLU()
        self.pool1=nn.MaxPool2d(2,2) #24*24 after maxpool

        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.relu2=nn.ReLU()
        self.pool2=nn.MaxPool2d(2,2) #12*12 after maxpool

        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.relu3=nn.ReLU()
        self.pool3=nn.MaxPool2d(2,2) #6*6 after maxpool

        self.drop1=nn.Dropout(0.25)
        self.fc1=nn.Linear(6*6*128,512)
        self.relu4=nn.ReLU()
        self.drop2=nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, 7)

    def forward(self, x):
        x=self.pool1(self.relu1(self.conv1(x)))
        x=self.pool2(self.relu2(self.conv2(x)))
        x=self.pool3(self.relu3(self.conv3(x)))

        x=torch.flatten(self.drop1(x),1)
        x=self.drop2(self.relu4(self.fc1(x)))

        x=self.fc2(x)
        return x