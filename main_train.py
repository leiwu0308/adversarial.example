
# coding: utf-8

# In[ ]:

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision as tv
from torch.autograd import Variable

import time
import random
import argparse
import os,sys
import models
from utils import *


# [1]: OPTIONS and SET RANDOM SEED

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate',type=float,default=1e-1,help='initial learning rate')
parser.add_argument('--drop',type=int,nargs='+',default=[81,95],help='Decrease the learning rate')
parser.add_argument('--decay',type=float,default=1e-4,help='Weight decay')
parser.add_argument('--nepochs',type=int,default=100,help='training epoch')
parser.add_argument('--batch_size',type=int,default=128,help='batch size of training')
parser.add_argument('--shortcutType',default='A',help='set the shortcut type')
parser.add_argument('--seed',type=int, default=123,help='manual seed')
parser.add_argument('--workers',type=int,default=1,help='# of worker used to load data')
parser.add_argument('--ngpu',type=int, default=1,       help='number of gpu')
parser.add_argument('--gpuid',         default='0,2',     help='device id used to train [default 1]')

# -- dataset --
parser.add_argument('--dataset', default='mnist',          help='dataset: | cifar10 | cifar100 | mnist [default] | imagenet')
parser.add_argument('--dataroot',default='/home/leiwu/data/',help='root path that stores data')
parser.add_argument('--arch',    default='lenet',         help='model used to classify the data set')
parser.add_argument('--resume',  default='None',             help='path to load checkpoint model')
parser.add_argument('--depth',   type=int, default=1)
parser.add_argument('--width',   type=int, default=500)
opt = parser.parse_args()
opt.savepath = 'checkpoints/%s-%s.pkl'%(opt.arch,opt.dataset)
print(opt)
os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpuid
print('--> GPU %s are selected <--'%(opt.gpuid))
torch.manual_seed(opt.seed)

random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
cudnn.benchmark = True

# [2]: DATA and TRANSFORM

train_loader, test_loader,num_classes = build_data_loader(opt)

# [3] MODEL AND LOSS
model = load_model(opt.arch,depth=opt.depth,width=opt.width,num_classes=num_classes)

print(model.name)
if opt.ngpu > 1:
    model = nn.DataParallel(model,device_ids=[0,1])
model = model.cuda()
if opt.resume != 'None':
    model_state = torch.load(opt.resume)
    model.load_state_dict(model_state)
print(model)

# loss and optimizer
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(),lr=opt.learning_rate,momentum=0.9,weight_decay=opt.decay,nesterov=True)

#for para in model.parameters():
#    para.data.zero_()



 #[5] TRAINING
best_acc1,best_acc2 = 0,0
bestmodel = None
for epoch in range(1,opt.nepochs+1):
    current_learning_rate=adjust_learning_rate(optimizer,epoch,drop=10,step=opt.drop)

    time_st = time.time()
    trL,trA1,trA5 = train(model,criterion,optimizer,train_loader)
    teL,teA1,teA5 = eval(model,criterion,test_loader)
    time_ed = time.time()

    print('[%3d/%d, %.0f seconds] lr:%.2e | train, %.2e, %.2f, %.2f |  test, %.1e, %.2f, %.2f'%(
            epoch,opt.nepochs,time_ed - time_st,current_learning_rate,
            trL,trA1,trA5,teL,teA1,teA5))

    #update checkpoints
    torch.save(model.state_dict(),'pretrains/tmp.pkl')
    if best_acc1 < teA1:
        best_acc1,best_acc5 = teA1,teA5
        torch.save(model.state_dict(),'pretrains/best.pkl')
    print('Best Model: %.2f%%\t %.2f%%'%(best_acc1,best_acc5))

torch.save(model.state_dict(),'pretrains/store/%s-%s-%.2f.pkl'%(model.name,opt.dataset,best_acc1))


