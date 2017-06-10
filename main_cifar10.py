import torch
import torch.nn as nn
import torchvision.transforms as trans
import torchvision.datasets as dsets
from torch.autograd import Variable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import os,math,random,argparse

import utils
import models
import attacks

# -[Option]-
parser = argparse.ArgumentParser()
parser.add_argument('--arch',       default='resnet8',                                help='model')
parser.add_argument('--modelpath',  default='pretrains/store/resnet8-cifar10-87.25.pkl',    help='path to load model')
parser.add_argument('--gpuid',      default='1',                                      help='GPU ID')
opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=opt.gpuid


# -[Hyper Parameters]-
batch_size  = 200
model_path  = opt.modelpath
lr          = 5/255.0
eps         = 5/255.0
niter       = 1


# -[DATA]-
train_set = dsets.CIFAR10(root='/home/leiwu/data/cifar10/',train=True, transform=trans.ToTensor(),download=True)
test_set  = dsets.CIFAR10(root='/home/leiwu/data/cifar10/',train=False,transform=trans.ToTensor(),download=False)

train_img = train_set.train_data.copy()
test_img  = test_set.test_data.copy()

train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size,shuffle=False)
test_loader  = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size,shuffle=False)

model       = models.__dict__[opt.arch]().cuda()
model_state = torch.load(model_path)
model.load_state_dict(model_state)
ct    = nn.CrossEntropyLoss().cuda()
#print(model)

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)),interpolation='nearest')


def batch_attack(model,ct,lr,eps,niter=1):
    adv_img = torch.zeros(10000,3,32,32)

    for i,(x,y) in enumerate(test_loader):
        if niter==1:
            lr = eps
        adv_x = attacks.gsm(model,ct,x.cuda(),y.cuda(),lr=lr,eps=eps,niter=niter)
        adv_img[i*batch_size:(i+1)*batch_size] = adv_x

    adv_img = np.round(adv_img.mul_(255).numpy()).astype(np.uint8)
    #adv_img = np.transpose(adv_img,(0,2,3,1))
    print(adv_img.shape)
    print(adv_img.dtype)

    return adv_img

# -[Generating]
eps_range = [1,2,3,4,5]
for eps in eps_range:
    adv_img = batch_attack(model,ct,lr,eps/255.0,niter)

    test_set.test_data = adv_img
    _,acc,_ = utils.eval(model,ct,test_loader)
    test_set.test_data = test_img
    print('perturbation: %d, accuracy: %.2f\n'%(eps,acc))

