# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 14:12:59 2022

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

from utlities.kernels import gaussiankernel, energykernel

directory = './output/sensitivity/'

directory_plot = './plots/'

model = eightgaussian()

df=pd.read_csv(directory+'example.csv',header = None)

particles = []
for i in range(100):
#for i in []
    i=str(i)

    particles.append(torch.load(directory+i+'.pt'))
    
def h_to_iter(h, c, a):
    return ((h-0.1)/a)**(1/c)

def iter_to_h(i, c, a):
    return a*(i+1)**c + 0.1

def iter_to_iter(old_iter, old_c, old_a, new_c, new_a):
    return h_to_iter(iter_to_h(old_iter, old_c, old_a), new_c, new_a)

def plot_parameter(axe, a, c, tau, n = 5000):
    m = df[(df[1] == c) & (df[2] == tau) & (df[3] == a)][0].item()
    x = particles[m][n][0]
    
    ngrid = 500
    xx = np.linspace(-6, 6, ngrid)
    yy = np.linspace(-6, 6, ngrid)
    X, Y = np.meshgrid(xx, yy)
    Z = np.exp(model.logp(np.vstack((np.ndarray.flatten(X), np.ndarray.flatten(Y))).T)).reshape(ngrid, ngrid)


    axe.contourf(X,Y,Z,64)
    axe.scatter(x[:,0],x[:,1],c='red', marker='x', s=16, alpha=1.0)
    axe.set_aspect('equal', 'box')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    axe.set_ylim(-6,6)
    axe.set_xlim(-6,6)
    axe.set_xlabel('h = ' + str(round(iter_to_h(n, c, a),2)), fontsize = 40)

ass = [5, 4, 3, 2, 1]
cs = [-0.1, -0.2, -0.3, -0.4, -0.5]
taus = [0.1, 0.5, 1, 2]

fig, axe = plt.subplots(5, 4, figsize = (35, 40))
for idx1,i in enumerate(cs):
    for idx2,j in enumerate(taus):
        plot_parameter(axe[idx1, idx2], a = 4, c = i, tau = j)
        
for idx1, i in enumerate(cs):
    axe[idx1, 0].set_ylabel('c = ' + str(i), fontsize = 40)
for idx2, j in enumerate(taus):
    axe[0, idx2].set_title('tau = '+ str(j), fontsize = 40)
fig.savefig(directory_plot + 'tauvsc')

m = df[(df[1] == -0.3) & (df[2] == 2) & (df[3] == 5)][0].item()
n = df[(df[1] == -0.1) & (df[2] == 2) & (df[3] == 1)][0].item()
n2 = df[(df[1] == -0.5) & (df[2] == 2) & (df[3] == 4)][0].item()
n3 = df[(df[1] == -0.4) & (df[2] == 2) & (df[3] == 2)][0].item()


mmd_1 = []
mmd_2 = []
mmd_3 = []
mmd_4 = []
y = model.sample(5000)
con = gaussiankernel(y, y, 0.5).mean()
for i in range(0,5000,100):
    mmd_1.append(mmd2(y, particles[m][i][0], const = con, mmd_h = 0.5).numpy())
    mmd_2.append(mmd2(y, particles[n][i][0], const = con, mmd_h = 0.5).numpy())
    mmd_3.append(mmd2(y, particles[n2][i][0], const = con, mmd_h = 0.5).numpy())
    mmd_4.append(mmd2(y, particles[n3][i][0], const = con, mmd_h = 0.5).numpy())

fig2, axe2 = plt.subplots(1, 1, figsize = (6,6))

axe2.plot(mmd_1, label = 'a=5 c=-0.3')
axe2.plot(mmd_2, label = 'a=1 c=-0.1')
axe2.plot(mmd_3, label = 'a=4 c=-0.5')
axe2.plot(mmd_4, label = 'a=2 c=-0.4')
axe2.set_xlabel('Iterations*100')
axe2.set_ylabel('MMD_Gaussian^2')
plt.yscale('log')
plt.legend()
plt.show()