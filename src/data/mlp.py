import torch
from torch import nn

class MLP_network(nn.Module):
    def __init__(self,input_size=48*48,num_classes=7):
        super().__init__()
        self.flatten=nn.Flatten()

        self.fc1=nn.Linear(input_size,512)
        self.relu1=nn.ReLU()
        self.dropout1=nn.Dropout(0.5)

        self.fc2=nn.Linear(512,256)
        self.relu2=nn.ReLU()
        self.dropout2=nn.Dropout(0.5)

        self.fc3=nn.Linear(256,num_classes)

    def forward(self, x):
        x=self.flatten(x) #(batch_size,1,48,48)

        #now shape will be (batch_size, 48*48)

        x=self.dropout1(self.relu1(self.fc1(x)))
        x=self.dropout2(self.relu2(self.fc2(x)))

        x=self.fc3(x)
        return x