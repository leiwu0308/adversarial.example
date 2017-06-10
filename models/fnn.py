import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math

class FNN(nn.Module):
    def __init__(self,arch):
        super(FNN,self).__init__()
        self.fc = []
        self.depth = len(arch)-1
        for i in range(self.depth):
            self.fc.append(nn.Linear(arch[i],arch[i+1]))
        self.fc = nn.ModuleList(self.fc)

        for m in self.modules():
            if isinstance(m,nn.Linear):
                fin = m.weight.size(1)
                m.weight.data.normal_(0,math.sqrt(2.0/fin))
                m.bias.data.zero_()

    def forward(self,x):
        o = x.view(x.size(0),-1)
        for i in range(self.depth):
            o = self.fc[i](o)
            if i < self.depth-1:
                o = F.relu(o)
        return o

    def name(self):
        return 'fully connected nerual networks'


def fnnMNIST_shallow(num_classes=10):
    model = FNN([784,500,num_classes])
    return model

def fnnMNIST_wide(num_classes=10):
    model = FNN([784,2000,num_classes])
    return model

def fnnMNIST_shallow1(num_classes=10):
    model = FNN([784,500,300,200,200,num_classes])
    return model

def fnnMNIST_deep(num_classes=10):
    model = FNN([784,500,300,200,200,100,num_classes])
    return model

def fnnMNIST_deepwide(num_classes=10):
    model = FNN([784,500,500,500,500,100,num_classes])
    return model

def fnnMNIST_wider(num_classes=10):
    model = FNN([784,4000,num_classes])
    return model

def fnnMNIST_deeper(num_classes=10):
    model = FNN([784,500,300,200,200,200,200,200,200,200,200,100,num_classes])
    return model

def fnnMNIST_deeperwide(num_classes=10):
    model = FNN([784,500,500,500,500,500,500,500,500,500,500,100,num_classes])
    return model

def fnnMNIST_deepless(num_classes=10):
    model = FNN([784,500,300,200,200,200,200,100,num_classes])
    return model

# ====================================================
def fnnCIFAR_shallow(num_classes=10):
    model = FNN([32*32*3,1000,500,num_classes])
    return model

def fnnCIFAR_deep(num_classes=10):
    model = FNN([32*32*3,1000,1000,500,500,500,500,100,num_classes])
    return model

def linear_classifier(num_classes=10):
    model = FNN([28*28,num_classes])
    return model
