from __future__ import print_function


import torch
import torch.nn as nn
import torchvision.transforms as trans
import torchvision.datasets as dsets
from torch.autograd import Variable


import os,math,random,argparse
os.environ['CUDA_VISIBLE_DEVICES']='1'

import utils
import models
import attacks

# -[Option]-
parser = argparse.ArgumentParser()
parser.add_argument('--arch',       default='fnnMNIST_deep',                              help='model')
parser.add_argument('--modelpath',  default='pretrains/store/fnnMNIST_deep-mnist-98.51.pkl',        help='path to load model')
parser.add_argument('--lr', type=int, default=40, help='stepsize')
parser.add_argument('--niter',type=int, default=1, help='number of iteration')
opt = parser.parse_args()
print(opt)


# -[Hyper Parameters]-
batch_size  = 200
model_path  = opt.modelpath
lr          = opt.lr/255.0
niter       = opt.niter

# -[DATA]-
train_set = dsets.MNIST(root='/home/leiwu/data/mnist',train=True, transform=trans.ToTensor(),download=True)
test_set  = dsets.MNIST(root='/home/leiwu/data/mnist',train=False,transform=trans.ToTensor(),download=False)
train_img, test_img = train_set.train_data.clone(), test_set.test_data.clone()
train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=batch_size,shuffle=False)
test_loader  = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size,shuffle=False)


arch, depth, width = utils.pathextract(opt.modelpath)
model,ct = utils.load_model(arch,depth,width)
model.load_state_dict(torch.load(opt.modelpath))
print(model)


#_,acc,_ = utils.eval(model,ct,test_loader)
#print('clear data',acc)


# -[Generate]
eps_range = [0,5,10,15,20,25,30,35,40]
acc_list  = []
for eps in reversed(eps_range):
    adv_img = torch.Tensor(test_img.size())
    for i,(x,y) in enumerate(test_loader):
        if niter == 1:
            lr = eps/255.0
        adv_x = attacks.gsm(model,ct,x.cuda(),y.cuda(),lr=lr,eps=eps/255.0,niter=niter)
        adv_img[i*batch_size:(i+1)*batch_size] = adv_x
    adv_img = (adv_img*255).byte()

    test_set.test_data.copy_(adv_img)
    _,acc,_ = utils.eval(model,ct,test_loader)
    acc_list.append(acc)
    test_set.test_data.copy_(test_img)
    print('perturbation: %d, accuracy: %.2f'%(eps,acc))


# print a line for Latex recording
for acc in reversed(acc_list):
    print('%.2f|'%(acc),end='')

print('')
