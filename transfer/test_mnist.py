import sys
sys.path.insert(0,'..')

import torch
import torch.nn as nn
import torchvision.transforms as trans
import torchvision.datasets as dsets
from torch.autograd import Variable


import os,math,random,argparse,glob
os.environ['CUDA_VISIBLE_DEVICES']='1'

import utils
import models
import attacks



parser = argparse.ArgumentParser()
parser.add_argument('--path1',  default='../pretrains/store/fnn_depth_1_width_500-mnist-98.41.pkl', help='parameters of network1')
parser.add_argument('--path2',  default=None, help='parameters of network1')
parser.add_argument('--lr', type=float, default=10,     help='step size of adversarial example generating method')
parser.add_argument('--eps',type=float, default=10,     help='maximal allowed average perturbation')
parser.add_argument('--niter',type=int, default=1,      help='number of iteration')
parser.add_argument('--batch_size', type=int, default=200, help='batch size')
parser.add_argument('--data',   default='test',         help='train | test [default]')
opt = parser.parse_args()
print(opt)


# -[DATA]-
train_set = dsets.MNIST(root='/home/leiwu/data/mnist',train=True, transform=trans.ToTensor(),download=True)
test_set  = dsets.MNIST(root='/home/leiwu/data/mnist',train=False,transform=trans.ToTensor(),download=False)
train_img, test_img = train_set.train_data.clone(), test_set.test_data.clone()
train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=opt.batch_size,shuffle=False)
test_loader  = torch.utils.data.DataLoader(dataset=test_set, batch_size=opt.batch_size,shuffle=False)

arch1, depth1, width1 = utils.pathextract(opt.path1)
model1 = utils.load_model(arch1,depth1,width1)
model1.load_state_dict(torch.load(opt.path1))

if opt.path2 is not None:
    arch2, depth2, width2 = utils.pathextract(opt.path2)
    model2 = utils.load_model(arch2,depth2,width2)
    model2.load_state_dict(torch.load(opt.path2))
ct    = nn.CrossEntropyLoss().cuda()

if opt.data == 'test':
    data_loader = test_loader
    clean_img = test_img
    img_pool = test_set.test_data
else:
    data_loader = train_loader
    clean_img = train_img
    img_pool = train_set.train_data

def batch_attack(model,ct,lr,eps,niter):
    adv_img = torch.Tensor(clean_img.size())
    batch_size = data_loader.batch_size
    for i,(x,y) in enumerate(data_loader):
        if niter == 1:
            lr = eps
        adv_x = attacks.gsm(model,ct,x.cuda(),y.cuda(),lr=lr,eps=eps,niter=niter)
        adv_img[i*batch_size:(i+1)*batch_size] = adv_x

    adv_img = (adv_img*255).byte()
    return adv_img


def attack_all_model(model_path):
    acc = {}
    for filename in glob.iglob(model_path+'/*mnist*'):
        arch,depth,width = utils.pathextract(filename)
        model = utils.load_model(arch,depth,width)
        model.load_state_dict(torch.load(filename))
        key = '%s_depth_%d_width_%d'%(arch,depth,width)
        _,acc[key],_ = utils.eval(model,ct,data_loader)
        print('%s \t    %.2f '%(key,acc[key]))

################################################
_,acc0,_ = utils.eval(model1,ct,data_loader)
adv_img = batch_attack(model1,ct,opt.lr/255.0,opt.eps/255.0,opt.niter)
img_pool.copy_(adv_img)

print('%s_depth_%d_width_%d, accuracy on clean data %.2f'%(arch1,depth1,width1,acc0))
if opt.path2 is None:
    attack_all_model('../pretrains/store/')
else:
    _,acc,_ = utils.eval(model2,ct,data_loader)
    print('attack accuracy: %.2f'%(acc))



