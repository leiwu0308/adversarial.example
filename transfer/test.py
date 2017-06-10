import sys
sys.path.insert(0,'..')

import torch
import torch.nn as nn
import torchvision.transforms as trans
import torchvision.datasets as dsets
from torch.autograd import Variable

import matplotlib
import matplotlib.pyplot as plt

import os,math,random,argparse
os.environ['CUDA_VISIBLE_DEVICES']='0'

import utils
import models
import attacks

def batch_attack(model,ct,data_loader,lr,eps,niter):
    adv_img = torch.Tensor(test_img.size())
    batch_size = data_loader.batch_size
    for i,(x,y) in enumerate(test_loader):
        if niter == 1:
            lr = eps
        adv_x = attacks.gsm(model,ct,x.cuda(),y.cuda(),lr=lr,eps=eps,niter=niter)
        adv_img[i*batch_size:(i+1)*batch_size] = adv_x

    adv_img = (adv_img*255).byte()
    return adv_img


parser = argparse.ArgumentParser()
parser.add_argument('--arch1',  default='linear_classifier',  help='network1')
parser.add_argument('--arch2',  default='linear_classifier', help='network2')
parser.add_argument('--path1',  default='../pretrains/store/linear_classifier-mnist-92.43.pkl', help='parameters of network1')
parser.add_argument('--path2',  default='../pretrains/store/linear_classifier-mnist-92.43.', help='parameters of network2')
parser.add_argument('--lr', type=float, default=10,     help='step size of adversarial example generating method')
parser.add_argument('--eps',type=float, default=10,     help='maximal allowed average perturbation')
parser.add_argument('--niter',type=int, default=1,      help='number of iteration')
parser.add_argument('--batch_size', type=int, default=200, help='batch size')
opt = parser.parse_args()
print(opt)


# -[DATA]-
train_set = dsets.MNIST(root='/home/leiwu/data/mnist',train=True, transform=trans.ToTensor(),download=True)
test_set  = dsets.MNIST(root='/home/leiwu/data/mnist',train=False,transform=trans.ToTensor(),download=False)
train_img, test_img = train_set.train_data.clone(), test_set.test_data.clone()
train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=opt.batch_size,shuffle=False)
test_loader  = torch.utils.data.DataLoader(dataset=test_set, batch_size=opt.batch_size,shuffle=False)

model1 = models.__dict__[opt.arch1]().cuda()
model1.load_state_dict(torch.load(opt.path1))
model2 = models.__dict__[opt.arch2]().cuda()
model2.load_state_dict(torch.load(opt.path2))
ct    = nn.CrossEntropyLoss().cuda()

################################################
adv_img = batch_attack(model1,ct,test_loader,opt.lr/255.0,opt.eps/255.0,opt.niter)

_,acc0,_ = utils.eval(model1,ct,test_loader)
test_set.test_data.copy_(adv_img)
_,acc1,_ = utils.eval(model1,ct,test_loader)
_,acc2,_ = utils.eval(model2,ct,test_loader)
test_set.test_data.copy_(test_img)
print('net2 accuracy: %.2f'%(acc0))
print('(1-->2) perturbation: %d, acc_net1: %.2f, acc_net2: %.2f\n'%(opt.eps,acc1,acc2))



adv_img = batch_attack(model2,ct,test_loader,opt.lr/255.0,opt.eps/255.0,opt.niter)

_,acc0,_ = utils.eval(model2,ct,test_loader)
test_set.test_data.copy_(adv_img)
_,acc2,_ = utils.eval(model2,ct,test_loader)
_,acc1,_ = utils.eval(model1,ct,test_loader)
test_set.test_data.copy_(test_img)
print('net2 accuracy: %.2f'%(acc0))
print('(2-->1) perturbation: %d, acc_net1: %.2f, acc_net2: %.2f\n'%(opt.eps,acc2,acc1))
