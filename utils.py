import torch
from torch.autograd import Variable
import torchvision as tv
import os,math

# loader MNIST

# [ DATA LOADER BUILD]
def build_data_loader(opt,train_transform=None,test_transform=None):
    if opt.dataset == 'cifar10':
        dset = tv.datasets.CIFAR10
        #mean = [x/255 for x in [125.3,123.0, 113.9]]
        #std  = [x/255 for x in [63.0, 62.1, 66.7]]
        num_classes = 10

    elif opt.dataset == 'cifar100':
        dset = tv.datasets.CIFAR100
        #mean = [x/255 for x in [129.3,124.1,112.4]]
        #std  = [x/255 for x in [68.2, 65.4, 70.4]]
        num_classes = 100

    elif opt.dataset == 'mnist':
        dset = tv.datasets.MNIST
        num_classes = 10
    else:
        assert False, 'Do not support this dataset, %s'%(opt.dataset)

    if train_transform is None:
        if opt.dataset=='mnist':
            train_transform = tv.transforms.ToTensor()
        elif opt.dataset=='cifar10' or opt.dataset == 'cifar100':
            train_transform = tv.transforms.Compose([
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.RandomCrop(32,padding=4),
                tv.transforms.ToTensor()
                #tv.transforms.Normalize(mean,std)
            ])

    if test_transform is None:
        if opt.dataset == 'mnist':
            test_transform = tv.transforms.ToTensor()
        elif opt.dataset == 'cifar10' or opt.dataset == 'cifar100':
            test_transform = tv.transforms.Compose([
                    tv.transforms.ToTensor()
                    #tv.transforms.Normalize(mean,std)
            ])

    datapath = os.path.join(opt.dataroot,opt.dataset)
    train_set = dset(root=datapath,train=True,transform=train_transform,download=True)
    test_set  = dset(root=datapath,train=False,transform=test_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=opt.batch_size,
                                    shuffle=True,num_workers=opt.workers,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=opt.batch_size,
                                    shuffle=False,num_workers=opt.workers,pin_memory=True)

    return train_loader, test_loader, num_classes

# [ evaluate accuracy given prediction ]
def accuracy(output,target,topk=(1,)):
    maxk = max(topk)
    batchsize = target.size(0)
    _,pred = output.topk(maxk,1)
    pred = pred.t()
    #print(target.size())
    correct = pred.eq(target.view(1,-1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum()
        res.append(correct_k*(100.0)/batchsize)
    return res

def predictor(model,x):
    batchsize = 100
    nsample   = x.size(0)
    pred = torch.LongTensor(nsample)
    for i in range(0,nsample,batchsize):
        j = min(nsample,i+batchsize)
        batch_x = Variable(x[i:j])
        outputs = model(batch_x).data
        _,pred[i:j] = outputs.topk(1,1)
    return pred


def adjust_learning_rate(optimizer,epoch,drop=None,step=None):
    de,lr = 1,None
    if drop!=None and step!=None:
        for s in step:
            if epoch == s:
                de *= drop
    for param_group in optimizer.param_groups:
        param_group['lr'] /= de
        lr = param_group['lr']
    return lr


#[] train

def train(model,criterion,optimizer,dataloader):
    model.train()
    loss,acc1,acc5 = 0,0,0
    for images,labels in dataloader:
        labels = Variable(labels.cuda(async=True))
        images = Variable(images.cuda())

        optimizer.zero_grad()
        outputs = model(images)
        loss_ = criterion(outputs,labels)
        loss_.backward()
        optimizer.step()

        loss += loss_.data[0]
        acc1_,acc5_ = accuracy(outputs.data,labels.data,topk=(1,5))
        acc1 += acc1_
        acc5 += acc5_
    batch_num = len(dataloader)
    return loss/batch_num,acc1/batch_num, acc5/batch_num

#[] test performance
def eval(model,criterion,dataloader):
    model.eval()
    loss,acc1,acc5 = 0,0,0
    for images,labels in dataloader:
        labels = Variable(labels.cuda(async=True),volatile=True)
        images = Variable(images.cuda(),volatile=True)

        #optimizer.zero_grad()
        outputs = model(images)
        loss += criterion(outputs,labels).data[0]
        acc1_,acc5_ = accuracy(outputs.data,labels.data,topk=(1,5))
        acc1 += acc1_
        acc5 += acc5_

    batch_num = len(dataloader)
    model.train()
    return loss/batch_num, acc1 / batch_num, acc5 / batch_num

#[] Quantilization
def quantify(img,level=2):
    step_v = range(0,level+1)
    step_v = map(lambda x: math.floor(255*float(x)/level), step_v)
    #print(step_v)
    x = img.clone().float()

    mask = torch.zeros(x.size()).byte()
    for l in range(level-1,0,-1):
        mask0 = ((x>step_v[l]) +  mask) %2
        x[mask0] = step_v[l+1]
        mask = mask0 + mask
    x[x<step_v[1]] = 0
    return x.byte()


if __name__ == '__main__':
    x = (torch.rand(4,4)*255).byte()
    y = quantify(x,level=4)
    print(x)
    print(y)
