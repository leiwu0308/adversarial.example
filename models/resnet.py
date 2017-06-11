from __future__ import division

import math
import torch
import torch.nn as nn
from torch.nn import init
import torchvision as tv
from torch.autograd import Variable

# resnet
def conv3x3(in_channels,out_channels,stride=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=3,
                    stride=stride,padding=1,bias=True)


class short_cut(nn.Module):
    def __init__(self,in_channels,out_channels,type='A'):
        super(short_cut,self).__init__()
        self.type = 'D' if in_channels == out_channels else type
        if self.type == 'C':
            self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0,stride=2,bias=False)
            self.bn   = nn.BatchNorm2d(out_channels)
        elif self.type == 'A':
            self.avg  = nn.AvgPool2d(kernel_size=1,stride=2)

    def forward(self,x):
        if self.type == 'A':
            x = self.avg(x)
            return torch.cat((x,x.mul(0)),1)
        elif self.type == 'C':
            x = self.conv(x)
            x = self.bn(x)
            return x
        elif self.type == 'D':
            return x

class residual_block(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,shortcutType='D'):
        super(residual_block,self).__init__()
        self.conv1 = conv3x3(in_channels,out_channels,stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels,out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = short_cut(in_channels,out_channels,type=shortcutType)

    def forward(self,x):
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.relu(o)
        o = self.conv2(o)
        o = self.bn2(o)
        o += self.shortcut(x)

        o = self.relu(o)
        return o

class ResNet(nn.Module):
    def __init__(self,block,depth,num_classes=10,shortcutType='A'):
        super(ResNet,self).__init__()
        assert (depth-2) %6 == 0 , 'depth should be 6*m + 2, like 20 32 44 56 110'
        num_blocks = (depth-2)//6
        print('resnet: depth: %d, # of blocks at each stage: %d'%(depth,num_blocks))

        self.in_channels = 16
        self.conv = conv3x3(3,16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.inplane = 16
        self.stage1 = self._make_layer(block,16,num_blocks,1) # 32x32x16
        self.stage2 = self._make_layer(block,32,num_blocks,2) # 16x16x32
        self.stage3 = self._make_layer(block,64,num_blocks,2) # 8x8x64
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64,num_classes)

        # initialization by Kaiming strategy
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                fin = m.kernel_size[0]*m.kernel_size[1]*m.out_channels #??????
                m.weight.data.normal_(0,math.sqrt(2.0/fin))
                m.bias.data.zero_()
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()


    def forward(self,x):
        o = self.conv(x)
        o = self.bn(o)
        o = self.relu(o)

        o = self.stage1(o)
        o = self.stage2(o)
        o = self.stage3(o)

        o = self.avg_pool(o)
        o = o.view(o.size(0),-1)
        o = self.fc(o)
        return o

    def _make_layer(self,block,out_channels,num_blocks,stride=1,shortcutType='A'):
        layers = []
        layers.append(block(self.in_channels,out_channels,stride,shortcutType=shortcutType))
        self.in_channels = out_channels
        for i in range(1,num_blocks):
            layers.append(block(out_channels,out_channels))
        return nn.Sequential(*layers)

def resnet8(num_classes=10):
    model = ResNet(residual_block,8,num_classes)
    return model

def resnet14(num_classes=10):
    model = ResNet(residual_block,14,num_classes)
    return model

def resnet20(num_classes=10):
    model = ResNet(residual_block,20,num_classes)
    return model

def resnet26(num_classes=10):
    model = ResNet(residual_block,26,num_classes)
    return model

def resnet32(num_classes=10):
    model = ResNet(residual_block,32,num_classes)
    return model

def resnet38(num_classes=10):
    model = ResNet(residual_block,38,num_classes)
    return model

def resnet44(num_classes=10):
    model = ResNet(residual_block,44,num_classes)
    return model

def resnet56(num_classes=10):
    model = ResNet(residual_block,56,num_classes)
    return model

def resnet110(num_classes=10):
    model = ResNet(residual_block,110,num_classes)
    return model
