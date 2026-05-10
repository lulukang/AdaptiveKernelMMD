# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 20:43:35 2022

@author: sadif
"""
import argparse

import numpy as np
import torch
import timeit


from utlities.distributions import eightgaussian,twogaussian,onegaussian,waveshape,star_gaussian
from utlities.distributions import multigaussian,gaus,multigaussian_multidimension,multiexp
from utlities.losses import const_calculator,losses,losses_grad, losses_distribution
from utlities.kernels import mmd2,gaussiankernel,energykernel,distancekernel,nonisogauskernel,rationalkernel
from utlities.methods import acceleration_distribution_step, trainer_distribution

import matplotlib.pyplot as plt

np.random.seed(42)
 
parser = argparse.ArgumentParser('generative')

parser.add_argument('--dataset', type=str, default='eightgaussian',
                    choices=['eightgaussian','twogaussian','onegaussian',
                             'multiexp','multigaussian_multidimension','funnel'
                             ,'star_gaussian','waveshape'])
parser.add_argument('--titleaddon',type=str,default='1_gaussian,h=median_logn')
parser.add_argument('--n_particles',type=int,default=200)
parser.add_argument('--h',type=float,default=10)
parser.add_argument('--tau',type=float,default=2)
parser.add_argument('--batch_size',type=int,default=50000)
parser.add_argument('--tol',type=float,default=1e-6)
parser.add_argument('--max_iter',type=int,default=5000)
parser.add_argument('--mode',type=int,default=1,choices=[1,2,3,4,5])
parser.add_argument('--noise',type=float,default=1)
parser.add_argument('--method',type=str,default='lbfgs',choices=['lbfgs','adam'])
parser.add_argument('--kernel',type=str,default='gaussiankernel',
                    choices=['gaussiankernel','energykernel','distancekernel','nonisogauskernel','rationalkernel'])
parser.add_argument('--save_every',type=int,default=1)
parser.add_argument('--c', type = float, default = -0.5)
parser.add_argument('--d', type = int, default = 2)
parser.add_argument('--a', type = float, default = 1)
parser.add_argument('--ini_multiplier', type = float, default = 4)

args = parser.parse_args()
dataset = args.dataset
d = args.d
model = locals()[dataset]()
n_particles=args.n_particles
max_iter = args.max_iter
tau = args.tau
c = args.c
save_every = args.save_every
title=args.titleaddon
ini_multiplier = args.ini_multiplier
a = args.a


#initialize x
x_init = torch.tensor(np.random.uniform(-1,1,[n_particles,d]),requires_grad=False)

x_init *= ini_multiplier

#comment this line if want to customize the value of a
a = (x_init[:,None,:] - x_init[None,:,:]).norm(dim = 2).median().detach()

x = x_init.detach().numpy()

# plt.scatter(x[:,0],x[:,1])
# plt.show()



x = torch.tensor(x,requires_grad=True)
x_prev = x.detach().clone()
x_hist = [(x.detach().clone(),'init')]

time_spent=[0]
start=timeit.default_timer()


#trainning
for i in range(max_iter):
    
    h = a*(1 + i)**c + 0.01

    optimizer = torch.optim.LBFGS([x],
                                history_size=30,
                                max_iter =30,
                                line_search_fn= 'strong_wolfe')
    
    distribution_t =  torch.distributions.MultivariateNormal(torch.zeros(d), torch.eye(d)*(h**2))
    t = distribution_t.sample([500])
    
    def closure():
        
        optimizer.zero_grad()
        loss1, loss2=losses_distribution(model, t, x, h = h , x_prev = x_prev , tau = tau)
        loss = loss1+loss2
        loss.backward()
        return loss
    
    optimizer.step(closure)
    
    stop=timeit.default_timer()
    time_spent.append(stop-start)
    
    if (i % save_every == 0) or (i <= save_every):
        #uncomment this line if want to monitor the evolvement of particles
        # plt.scatter(x[:,0].detach().numpy(),x[:,1].detach().numpy())
        # plt.show()
        x_hist.append((x.detach().clone(),i))
    
    x_prev = x.detach().clone()
    
    print(i,h)
    
torch.save(x_hist,'output/'+title+'.pt')

