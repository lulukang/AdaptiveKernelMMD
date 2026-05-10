# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 08:38:26 2022

@author: sadif
"""

import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics.pairwise as sk_metric

from geomloss import SamplesLoss
from utlities.plots import toy_plot

from utlities.distributions import eightgaussian,twogaussian,onegaussian,waveshape,star_gaussian
from utlities.distributions import multigaussian,gaus,multigaussian_multidimension,multiexp
from utlities.kernels import mmd2
from utlities.mmd import square_mmd_fine


model = eightgaussian()
noise = multigaussian(2)

y = model.sample(200)
x = noise.sample(200)

mmd = []

hs = [4, 2, 1, 0.5, 0.2, 0.1,0.05]
thetas = np.arange(0,2,0.01)

for h in hs:
    mmd_temp = []
    for theta in thetas:
        z = y + (2-theta)*x
        mmd_temp.append(mmd2(y, z, const = True, mmd_h = h).item())
    mmd.append(mmd_temp)

fig, axe = plt.subplots(1, 1)
for idx,h in enumerate(hs):
    axe.plot(thetas, mmd[idx], label = 'h = ' + str(h))
    axe.set_xlim(0,2)
    axe.set_ylim(0,0.05)
    axe.set_xlabel('theta')
    axe.set_ylabel('MMD_Gaussian')
plt.legend() 
plt.show()


ngrid = 500
xx = np.linspace(-6, 6, ngrid)
yy = np.linspace(-6, 6, ngrid)
X, Y = np.meshgrid(xx, yy)
Z = np.exp(model.logp(np.vstack((np.ndarray.flatten(X), np.ndarray.flatten(Y))).T)).reshape(ngrid, ngrid)

fig,axe = plt.subplots(2,3,figsize = (18,12))

hs = [0, 1, 1.5, 1.75, 1.9, 2]

for i in range(2):
    for j in range(3):
        theta = hs[i*3+j]
        axe[i,j].contourf(X,Y,Z,64)
        axe[i,j].scatter(y[:,0],y[:,1],c='white', marker='x', s=64, alpha=0.8, label = 'Y')
        z = y + (2-theta)*x
        axe[i,j].scatter(z[:,0], z[:,1], c = 'red', marker = '+', s = 32, alpha = 1, label = 'X')
        axe[i,j].axis('off')
        axe[i,j].set_xlim(-6,6)
        axe[i,j].set_ylim(-6,6)
plt.legend()
plt.subplots_adjust(wspace=0.02, hspace=0.02)
plt.savefig('evlove.png')