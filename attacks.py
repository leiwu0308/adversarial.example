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
        ImgProj(x.data,x0,eps)

    return x.data


#-----------------------------------------
# Auxiliary routine
#-----------------------------------------
def ImgProj(x,x0,eps):
    dx = x - x0
    dx.clamp_(-eps,eps)
    x.copy_(x0).add_(1,dx).clamp_(0,1)
    return x
