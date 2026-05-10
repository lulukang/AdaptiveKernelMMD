# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 16:21:00 2022

@author: sadif
"""

import argparse

import numpy as np
import torch
import time
import pandas as pd


from utlities.distributions import eightgaussian,twogaussian,onegaussian,multigaussian,multiexp,funnel
from utlities.plots import toy_plot,toy_anime
from utlities.losses import const_calculator,losses,losses_grad
from utlities.kernels import mmd2,gaussiankernel,energykernel,distancekernel,nonisogauskernel,rationalkernel


import matplotlib.pyplot as plt



np.random.seed(42)
 
parser = argparse.ArgumentParser('generative')

parser.add_argument('--dataset', type=str, default='multigaussian',
                    choices=['eightgaussian','twogaussian','onegaussian',
                             'multiexp','multigaussian_multidimension','funnel'])
parser.add_argument('--titleaddon',type=str,default='gaussian')
parser.add_argument('--n_particles',type=int,default=500)
parser.add_argument('--h',type=float,default=1)
parser.add_argument('--tau',type=float,default=2)
parser.add_argument('--batch_size',type=int,default=50000)
parser.add_argument('--tol',type=float,default=1e-6)
parser.add_argument('--max_iter',type=int,default = 500)
parser.add_argument('--mode',type=int,default=1,choices=[1,2,3,4,5])
parser.add_argument('--noise',type=float,default=1)
parser.add_argument('--method',type=str,default='lbfgs',choices=['lbfgs','adam'])
parser.add_argument('--kernel',type=str,default='gaussiankernel',
                    choices=['gaussiankernel','energykernel','distancekernel','nonisogauskernel','rationalkernel'])
parser.add_argument('--save_every',type=int,default = 50)
parser.add_argument('--c', type = float, default = - 0.1)
parser.add_argument('--d', type = int, default = 20)

torch.set_default_dtype(torch.double)


args=parser.parse_args()
dataset=args.dataset
title=args.titleaddon
d = args.d
data=locals()[dataset](d) 
h=args.h
tau=args.tau
n_particles=args.n_particles
batch_size=args.batch_size
tol=args.tol 
max_iter=args.max_iter
mode=args.mode
noise_propotion=args.noise
kernel=locals()[args.kernel]
save_every=args.save_every

c = args.c

tau = d

x = torch.tensor(np.random.uniform(-2,2,[n_particles,d]),requires_grad=False)
x = x


# A = (torch.rand(d,d) - 0.5)*2
y=data.sample(batch_size)
# A = torch.sin(A * np.pi / 2)


# y = torch.matmul(y,A)
# yy,_ = y.max(dim = 0)
# y = 2*y/yy


# loc = torch.zeros(d)
# scale = torch.ones(d)
# laplace = torch.distributions.Laplace(loc=loc, scale=scale)

# y = laplace.sample([batch_size])


  # number of dimensions
# df = 3  # degrees of freedom

# t_dist = torch.distributions.StudentT(df, 0, 1)
# y = t_dist.sample((batch_size,d))


# alpha = 2*torch.ones(d)
# beta = 4*torch.ones(d)
# dist = torch.distributions.Beta(alpha, beta)

# y = dist.sample((batch_size,))



# concentration = 0.5*torch.ones(d)  # concentration parameters

# dirichlet_dist = torch.distributions.Dirichlet(concentration)

# y = dirichlet_dist.sample((batch_size,))


# concentration = torch.randn(2).abs()
# scale = 2*torch.randn(2).abs()

# # Create a 2-dimensional Weibull distribution
# weibull_dist = torch.distributions.weibull.Weibull(concentration, scale)

# # Generate a sample of size (n x 2) from the distribution

# y = weibull_dist.sample((batch_size,))

yy = y.numpy()
np.savetxt(r"C:\Users\sadif\Desktop\mac drop off\AMMD_v2\testfile_" + title + ".csv", yy, delimiter=',')


# base_data = pd.read_csv(r"C:\Users\sadif\Desktop\mac drop off\AMMD_v2\deca.csv", index_col = False).to_numpy()
# y = torch.tensor(base_data)


x.requires_grad=True


a = (x[:,None,:] - x[None,:,:]).norm(dim = 2).median().detach()


idx = np.random.choice(batch_size,10*n_particles,replace = False)
y_part=y[idx]

const=const_calculator(y_part, h, method=kernel)

#store historical datas
x_hist=[(x.detach().clone(),'init')]
mmd_hist=[mmd2(y_part,x,const=True,mmd_h=1,method=kernel).detach().numpy()]

grad_hist=[]
times=[0]

x_prev=x.detach().clone()



if args.method=='lbfgs':
    for i in range(max_iter):
        
        if c == 0:
            h = 1
            
        else:
        
            h = a*(1 + i)**c + 0.01

        
        start = time.time()
        optimizer = torch.optim.LBFGS([x],
                                    history_size=30,
                                    max_iter =30,
                                    line_search_fn= 'strong_wolfe')
        
        
        idx = np.random.choice(batch_size,10*n_particles, replace = False)
        y_part=y[idx]
        const=const_calculator(y_part, h)
        def closure():
            optimizer.zero_grad()
            loss1,loss2=losses(y_part,x,x_prev,tau,h,const=const,mode=mode,method=kernel)
            loss=loss1+loss2
            loss.backward()
            return loss
        optimizer.step(closure)

        end = time.time()
        
        times.append(times[-1]+(end-start))
        
        
        #loss2=mmd2(y_part,x,const=const,mmd_h=h,method=kernel).detach().numpy()

        
        if (i%save_every==0) or (i<=save_every):
            x_hist.append((x.detach().clone(),i))
            
        mmd_hist.append(mmd2(y_part,x,const=True,mmd_h=1,method=kernel).detach().numpy())
        
        if end-start<0:
            break
        print(i, mmd2(y_part,x,const=True,mmd_h=1,method=kernel).detach().numpy())

        # if loss2<=tol:
        #     break
        x_prev=x.detach().clone()
        
plt.scatter(y[:,0], y[:,1], s = 0.2, alpha = 0.5)
plt.scatter(x[:,0].detach(), x[:,1].detach(), s = 5)
# plt.xlim(-0.2, 1.2)
# plt.ylim(-.2, 1.2)

torch.save(y,'output/'+title+'_y.pt')
torch.save(times,'output/'+title+'_time.pt')
torch.save(x_hist,'output/'+title+'.pt')
torch.save(mmd_hist,'output/'+title+'_mmd.pt')

