
import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.dataset_name =='cifar':
            self.conv1 = nn.Conv2d(3,16,3,padding=1)
            
        if args.dataset_name =='minst':
            self.conv1 = nn.Conv2d(1,16,3,padding=1)
            
        self.conv2 = nn.Conv2d(16,8,3,padding=1)
        
        self.mp = nn.MaxPool2d(2,2)
        
        self.t_conv1 = nn.ConvTranspose2d(8,16,2,stride=2)
        
        if args.dataset_name =='cifar':
            self.t_conv2 = nn.ConvTranspose2d(16,3,2,stride=2)
            
        if args.dataset_name =='minst':
            self.t_conv2 = nn.ConvTranspose2d(16,1,2,stride=2)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x = self.mp(self.relu(self.conv1(x)))
        x = self.mp(self.relu(self.conv2(x)))
        
        x = self.relu(self.t_conv1(x))
        x = self.sigmoid(self.t_conv2(x))
        
        return x
