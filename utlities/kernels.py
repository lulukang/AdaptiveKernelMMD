 # -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 16:22:28 2022

@author: sadif
"""
import numpy as np
import torch

def rationalkernel(y,x,kernel_h=1):
    diff = y[:,None,:] - x[None,:,:]
    rat = diff.pow(2).sum(dim=2) / (2*kernel_h)
    del diff
    return (rat+1).pow(-kernel_h)

def gaussiankernel(y,x,kernel_h=1):
    diff=y[:,None,:]-x[None,:,:]
    exp_pow=-diff.pow(2).sum(dim=2)/(2*kernel_h**2)
    result=exp_pow.exp()
    del diff,exp_pow
    return result

def nonisogauskernel(y,x,kernel_h=[1,2]):
    d = np.size(kernel_h)
    diff = y[:,None,:]-x[None,:,:]
    for i in range(d):
        diff[:,:,i] = diff[:,:,i] / (np.sqrt(2)*max(kernel_h[i],1e-3))
    exp_pow=-diff.pow(2).sum(dim=2)
    result=exp_pow.exp()
    del diff,exp_pow
    return result
        

def energykernel(y,x,kernel_h=1):
    diff=y[:,None,:]-x[None,:,:]
    return -(diff.norm(dim=2).pow(kernel_h))
    # return -(diff.pow(2).sum(dim=2).pow(kernel_h/2))

def distancekernel(y,x,kernel_h=1):
    diff=y[:,None,:]-x[None,:,:]
    kh=float(kernel_h)
    return y[:,None,:].norm(dim=2).pow(kh)+x[None,:,:].norm(dim=2).pow(kh)-diff.norm(dim=2).pow(kh)

def mmd2(y,x,const=False,method=gaussiankernel,mmd_h=1,unbias=True):
    cross=method(y,x,kernel_h=mmd_h)
    square=method(x,x,kernel_h=mmd_h)
    
    if const==True:
        constant=method(y,y,kernel_h=mmd_h)
        return constant.mean()-2*cross.mean()+square.mean()
    
    if const==False:
        return -2*cross.mean()+square.mean()
    
    if type(const) == int or float:
        return const-2*cross.mean()+square.mean()
    else:
        return
    