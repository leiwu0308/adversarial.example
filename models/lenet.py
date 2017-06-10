import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self,num_classes=10):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(1,20,5,stride=1) # 28-5+1=24
        self.conv2 = nn.Conv2d(20,50,5,stride=1) # 12-5+1=8
        self.fc1 = nn.Linear(50*4*4,500)
        self.fc2 = nn.Linear(500,num_classes)

    def forward(self,x):
        o = self.conv1(x)
        o = F.relu(o)
        o = F.avg_pool2d(o,2,2)
        o = self.conv2(o)
        o = F.relu(o)
        o = F.avg_pool2d(o,2,2)
        o = o.view(-1,4*4*50)
        o = self.fc1(o)
        o = F.relu(o)
        o = self.fc2(o)
        return o
    def name(self):
        return 'LeNet:(1x28x28->20x24x24->20x12x12->50x8x8->50x4x4->500->10)'

def lenet(num_classes=10):
    model = LeNet(num_classes)
    return model
