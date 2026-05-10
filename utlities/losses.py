# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 20:14:33 2022

@author: sadif
"""

import numpy as np
import torch

from utlities.kernels import mmd2,gaussiankernel,energykernel,distancekernel



def const_calculator(y,h,method=gaussiankernel):
    return method(y, y,kernel_h=h).mean().detach().clone()


def losses(y,x,x_prev,tau,h,method=gaussiankernel,const=False,mode=1):
    if mode==1:
        loss1=(1/(2*tau))*torch.mean(torch.sum((x - x_prev)**2, dim = 1, keepdim = True))
        loss2=mmd2(y,x,const=const,mmd_h=h,method=method)
        return loss1,loss2
    
    if mode==2:
        loss1=(1/(2*tau))*torch.mean(torch.sum((x - x_prev)**2, dim = 1, keepdim = True))
        loss2=mmd2(y,x,const=const,mmd_h=h,method=method).sqrt()
        return loss1,loss2
    
    if mode==3:
        loss1=(1/(2*tau))*mmd2(x_prev,x,const=True,mmd_h=h,method=method)
        loss2=mmd2(y,x,const=const,mmd_h=h,method=method)
        return loss1,loss2
    
    if mode==4:
        loss1=(1/(2*tau))*mmd2(x_prev,x,const=True,mmd_h=h).sqrt()
        loss2=mmd2(y,x,const=const,mmd_h=h,method=method).sqrt()
        return loss1,loss2
    
    if mode==5:
        loss1=(1/(2*tau)) *mmd2(x_prev,x,const=True,mmd_h=h,method=method)
        loss2=1*mmd2(y,x,const=const,mmd_h=h,method=method)+1*mmd2(y,x,const=const,mmd_h=h/20,method=method)
        return loss1,loss2


def losses_grad(x,y,x_prev,tau,h):
    x=x.detach().numpy()
    y=y.detach().numpy()
    
    Ny=y.shape[0]
    n_particles=x.shape[0]

    diff = x[:, None, :] - x[None, :, :]
    diff_xy = x[:, None, :] - y[None, :, :]
    kxy = np.exp(-np.sum(diff ** 2, axis=-1) / (2 * h ** 2)) 
    k_xiyj = np.exp(-np.sum(diff_xy ** 2, axis=-1) / (2 * h ** 2)) 
    
    gradK = -diff * kxy[:, :, None] / h ** 2
    grad_K_xiyj = - diff_xy * k_xiyj[:, :, None] / h ** 2
    
    dxkxy = np.sum(gradK, axis=1)
    dxk_xiyj = np.sum(grad_K_xiyj, axis=1)
    
    cross=2/Ny * dxk_xiyj 
    square=2 / n_particles* dxkxy
    
    loss2_grad=(square-cross)/n_particles
     
    y=x_prev.detach().numpy()
    
    Ny=y.shape[0]
    n_particles=x.shape[0]

    diff = x[:, None, :] - x[None, :, :]
    diff_xy = x[:, None, :] - y[None, :, :]
    kxy = np.exp(-np.sum(diff ** 2, axis=-1) / (2 * h ** 2)) 
    k_xiyj = np.exp(-np.sum(diff_xy ** 2, axis=-1) / (2 * h ** 2)) 
    
    gradK = -diff * kxy[:, :, None] / h ** 2
    grad_K_xiyj = - diff_xy * k_xiyj[:, :, None] / h ** 2
    
    dxkxy = np.sum(gradK, axis=1)
    dxk_xiyj = np.sum(grad_K_xiyj, axis=1)
    
    cross=2/Ny * dxk_xiyj 
    square=2 / n_particles* dxkxy
    
    loss1_grad=(1/(2*tau))*(square-cross)/n_particles
    
    #loss1_grad=(1/(2*tau))*2*(x-y)
    
    return loss1_grad+loss2_grad


def losses_distribution(model, t, x, h = 0.2, x_prev = 0, tau = 0):
    
    n_particles,d = x.shape
    
    n_t = t.shape[0]
    
    diff = x[:, None, :] - x[None, :, :]
    
    tt = t[None, :, :] + x[:, None, :]
    
    exp_pow =-diff.pow(2).sum(dim=2)/(2*h**2)
    
    kxy = exp_pow.exp()
    
    square = kxy.mean()

    f = model.logp_tensor(tt.reshape(n_t*n_particles,d))

    cross = ((np.sqrt(2*np.pi)*h)**d)*f.mean()
    
    #cross = ((2*np.pi*(h))**(d/2))*f.mean()
    
    
    # exp_pow2 =-diff.pow(2).sum(dim=2)/(0.2*h**2)
    
    # kxy2 = exp_pow2.exp()
    
    # square2 = kxy2.mean()
    
    # tt2 = 0.1*t[None, :, :] + x[:, None, :]

    # f2 = model.logp_tensor(tt2.reshape(n_t*n_particles,2))

    # cross2 = ((np.sqrt(2*np.pi)*0.1*h)**d)*f2.mean()

    loss1 = (1/(2*tau))*torch.mean(torch.sum((x - x_prev)**2, dim = 1, keepdim = True))
    loss2 = square - 2*cross
    
    return loss1, loss2