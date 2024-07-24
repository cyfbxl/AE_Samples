
import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1,16,3,padding=1)
        self.conv2 = nn.Conv1d(16,8,3,padding=1)
        
        self.mp = nn.MaxPool1d(2,2)
        
        self.t_conv1 = nn.ConvTranspose1d(8,16,2,stride=2)
        self.t_conv2 = nn.ConvTranspose1d(16,1,2,stride=2)
        
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x = self.mp(self.relu(self.conv1(x)))
        x = self.mp(self.relu(self.conv2(x)))
        
        x = self.relu(self.t_conv1(x))
        x = (self.t_conv2(x))
        
        return x
