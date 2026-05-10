# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 17:10:42 2023

@author: sadifenchen
"""

import torch
from torch import nn, Tensor
import numpy as np
from numpy.random import choice
import pandas as pd


_scaling_min = 0
meshXMin = -8
meshXMax = -meshXMin

torch.set_default_dtype(torch.float64)

dimx = 2


def V(phi: Tensor) -> Tensor:
    # determinant sigma = 1
    x = phi
    V=torch.zeros(x.shape[0])
    for i in range(dimx-1):
        V+=10* (x[:,i+1]-x[:,i]**2)**2 + (x[:,i]-1)**2
    return 3*V/2**dimx

def weightedSamplingFromTrue(sample_num):
    '''
    generate particles from true densities from mesh grid
    '''
    #sample_discrepancy = 1
    #sample_counter = 0
    #while sample_discrepancy > 1e-8 and sample_counter<5:
    #    coor_sobol_sampler = Sobol(dimx, seed=1227)
    #    coor_sobol = coor_sobol_sampler.random_base2(16)
    #    coor_discrepancy = qmc.discrepancy(coor_sobol)
    #    sample_counter+=1
    #l_bounds = np.repeat(meshXMin,dimx)
    #u_bounds = np.repeat(meshXMax,dimx)
    #coor_sobol = qmc.scale(coor_sobol, l_bounds, u_bounds)
    uniform_sampler = torch.distributions.uniform.Uniform(meshXMin,meshXMax)
    coor_uni = uniform_sampler.sample((200000,dimx))

    sample_coor_uni = coor_uni
    density_on_uni = torch.exp(-V(sample_coor_uni))
    density_on_uni = density_on_uni.cpu().numpy()
    
    samplePool=np.arange(sample_coor_uni.shape[0])
    sampledIndex = choice(samplePool, sample_num,
                          p=density_on_uni.ravel()/density_on_uni.sum(),replace=False)
    sampled = sample_coor_uni[sampledIndex,:]
    sampled_densities = density_on_uni[sampledIndex]
    return sampled.requires_grad_(False), torch.from_numpy(sampled_densities).requires_grad_(False)

base_data,_ = weightedSamplingFromTrue(50000)
base_numpy = base_data.numpy()

df = pd.DataFrame(base_numpy) #convert to a dataframe
df.to_csv("testfile.csv",index=False) #save to file
