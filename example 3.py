# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 17:08:10 2022

@author: sadif
"""

import argparse

import numpy as np
import torch

from utlities.importer import fashion,mnist,cifar10,cifar100
from utlities.plots import generative_plot
from utlities.losses import const_calculator,losses
from utlities.kernels import mmd2

from utlities.kernels import mmd2,gaussiankernel,energykernel,distancekernel,rationalkernel

parser = argparse.ArgumentParser('generative')

parser.add_argument('--dataset', type=str, default='mnist',
                    choices=['mnist', 'fashion', 'cifar10','cifar100'])
parser.add_argument('--titleaddon',type=str,default='_1')
parser.add_argument('--n_particles',type=int,default=100)
parser.add_argument('--h',type=float,default=10)
parser.add_argument('--tau',type=float,default=784)
parser.add_argument('--batch_size',type=int,default=50)
parser.add_argument('--tol',type=float,default=0.0005)
parser.add_argument('--max_iter',type=int,default=500)
parser.add_argument('--mode',type=int,default=1,choices=[1,2,3,4,5])
parser.add_argument('--noise',type=float,default=1)
parser.add_argument('--method',type=str,default='lbfgs',choices=['lbfgs','adagrad'])
parser.add_argument('--resample',type=bool,default=True)
parser.add_argument('--save_every',type=int,default=50)
parser.add_argument('--a',type=float,default=10)
parser.add_argument('--kernel',type=str,default='gaussiankernel',
                    choices=['gaussiankernel','energykernel','distancekernel','rationalkernel'])

b = 1
c = -0.5

args=parser.parse_args()
dataset=args.dataset
title=dataset+args.titleaddon
data=locals()[dataset]("./data")
#h=args.h
tau=args.tau
n_particles=args.n_particles
batch_size=args.batch_size
tol=args.tol
max_iter=args.max_iter
mode=args.mode
noise_propotion=args.noise
resample=args.resample
save_every=args.save_every
kernel=locals()[args.kernel]
a = args.a
h = args.h

d=np.prod(list(data.shape))

#uncomment this line if want to customize tau
tau = d

#initialize x
x_1=torch.tensor(np.random.uniform(0,1,[n_particles,d]),requires_grad=True)
y=data.sample(n_particles)
x=noise_propotion*x_1.detach().clone()+(1-noise_propotion)*y

y = data.sample(batch_size)

#comment this line if want to customize the value of a
a = (x[:,None,:] - x[None,:,:]).norm(dim = 2).median().detach()

const=const_calculator(y, h)



x.requires_grad=True

#store historical datas
x_hist=[(x.detach().clone(),'init')]
mmd_hist=[(mmd2(y,x,const=True,mmd_h=h,method=kernel).detach().numpy(),'init')]

x_prev=x.detach().clone()

#trainning
if args.method=='lbfgs':
    for i in range(max_iter):
        
        h = a*(1 + i)**c + 1

        
        optimizer = torch.optim.LBFGS([x],
                                    history_size=30,
                                    max_iter =30,
                                    line_search_fn= 'strong_wolfe')
    
        
        if resample:
            y=data.sample(batch_size)
            const=const_calculator(y, h,method=kernel)
            
        if args.h==-1:
            idx1 = np.random.choice(np.arange(n_particles), min(n_particles, batch_size), replace=False)
            idx2 = np.random.choice(np.arange(batch_size), min(n_particles, batch_size), replace=False)
            h = ((x[idx1]-y[idx2]).norm(dim=1).mean()/np.sqrt(2)).detach().numpy()
        
        def closure():
            
            optimizer.zero_grad()
            loss_1,loss_2=losses(y,x,x_prev,tau,h,const=const,mode=mode,method=kernel)
            loss=loss_1+loss_2
            loss.backward()
            return loss
        optimizer.step(closure)
        
        if ((i+1)%save_every==0) or (i<=100):
            x_hist.append((x.detach().clone(),i))

        mmd_hist.append((mmd2(y,x,const=True,mmd_h=h,method=kernel).detach().numpy(),str(i)))
        

        print(mmd2(y,x,const=True,mmd_h=h,method=kernel),h)
        
        x_prev=x.detach().clone()
        



if args.method=='adagrad':
    for i in range(max_iter):
        if resample:
            y=data.sample(batch_size)
            const=const_calculator(y, h,method=kernel)
            
            if args.h==-1:
                idx1 = np.random.choice(np.arange(n_particles), min(n_particles, batch_size), replace=False)
                idx2 = np.random.choice(np.arange(batch_size), min(n_particles, batch_size), replace=False)
                h = ((x[idx1]-y[idx2]).norm(dim=1).mean()/np.sqrt(2)).detach().numpy()
        for j in range(500):
            optimizer = torch.optim.Adagrad([x], lr=0.01)
            
            optimizer.zero_grad()
            
            loss_1,loss_2=losses(y,x,x_prev,tau,h,const=const,mode=mode,method=kernel)
            loss=loss_1+loss_2
            loss.backward()
            
            optimizer.step()
        
        
        if (i+1)%save_every==0:
            x_hist.append((x.detach().clone(),i))

        mmd_hist.append((mmd2(y,x,const=True,mmd_h=h,method=kernel).detach().numpy(),str(i)))
        

        print(mmd2(y,x,const=True,mmd_h=h,method=kernel),h)
        
        x_prev=x.detach().clone()  

#uncomment this line if want to monitor final result
generative_plot(x,data.shape,title=title)
torch.save(x_hist,'output/'+title+'.pt')
torch.save(mmd_hist,'output/'+title+'_mmd.pt')
