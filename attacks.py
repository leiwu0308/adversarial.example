import torch
from torch.autograd import Variable
# All images in this package are assumed to represented by [0,1] float number


def gsm(model,ct,x,y,lr=0.1,eps=0.1,niter=1):
    x0 = x.clone()
    x,y = Variable(x.clone(),requires_grad=True),Variable(y.clone())
    model.eval()
    for i in range(niter):
        if x.grad is not None:
            x.grad.data.zero_()
        y_ = model(x)
        loss = ct(y_,y)
        loss.backward()

        dx = x.grad.data
        x.data.add_(lr,dx.sign())
        x.data = image_clip(x.data,x0,eps)

    model.train()
    return x.data


def batch_one_step_attack(model,ct,x,y,x0,lr=0.1,eps=0.1,batch_size=200):
    # Input should be torch Tensor
    model.eval()
    nsample = x.size(0)
    x_adv = x.clone().float()/255.0
    for i in range(0,nsample,batch_size):
        j = min(i+batch_size,nsample)
        bx, by = Variable(x_adv[i:j].cuda(),requires_grad=True), Variable(y[i:j].cuda())
        by_    = model(bx)
        loss   = ct(by_,by)
        loss.backward()

        dx = bx.grad.data
        bx.data.add_(lr,dx.sign())
        x_adv[i:j] = bx.data.cpu()

    x_adv = image_clip(x_adv,x0,eps)
    x.copy_(x_adv.mul_(255).byte())
    dx = x.float() - x0 * 255
    print(dx.max(),dx.min())

    #[3]
    model.train()
    return x

#-----------------------------------------
# Auxiliary routine
#-----------------------------------------
def image_clip(x,x0,eps):
    dx = x - x0
    dx.clamp_(-eps,eps)
    x.copy_(x0).add_(1,dx).clamp_(0,1)
    return x


