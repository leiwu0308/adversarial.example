import torch
import torch.nn as nn
from torch.autograd import Variable

class LinearClassifier(nn.Module):
    def __init__(self,input_size,output_size):
        super(LinearClassifier,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Linear(input_size,output_size)

    def forward(self,x):
        o = x.view(-1,self.input_size)
        o = self.fc(o)
        return o




def linear_classifier(num_classes=10):
    model = LinearClassifier(28*28,num_classes)
    return model
