# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 16:30:02 2022

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

directory = './output/mmd/'
directory_compare = './output/toy compare/'
directory_plot = './plots/'

dataset = ['eightgaussian', 'star_gaussian', 'waveshape']
model1 = star_gaussian()
model0 = eightgaussian()
model2 = waveshape()
#models = [model2, model1, model3]

markerstyles = ['o', 'v', 's', 'p', 'D', '+', '2']



particles = []
for i in range(3):
    i=str(i)
    particles.append(torch.load(directory+'/output/'+i+'.pt'))

dtype = 'wave'
data_svgd = np.load(directory_compare+'svgd_'+dtype+'.npz')
data_lmc = np.load(directory_compare+'lmc_'+dtype+'.npz')
data_evi = np.load(directory_compare+'evi_'+dtype+'.npz')
evolve_svgd = data_svgd['evolves']
evolve_lmc = data_lmc['evolves']
evolve_evi = data_evi['evolves']



max_iter = 501

x = particles[2][:max_iter]
model = model2

index = []
mmd_mmd = []
ce_mmd = []
if dtype =='wave':
    data_true=np.load(directory_compare+'wave_samples.npz')
    y=np.zeros([5000,2])
    y[:,0]=data_true['X']
    y[:,1]=data_true['Y']
    k_yi_yj = sk_metric.rbf_kernel(y, y)
    const=np.sum(k_yi_yj)/5000/5000
else:
    y = model.sample(5000)
    k_yi_yj = sk_metric.rbf_kernel(y, y)
    const=np.sum(k_yi_yj)/5000/5000
    
ce_true = model.logp(y).mean()

for idx,i in enumerate(x):
    if idx == 0:
        index.append(0)
    else:
        index.append(i[1]+1)
    xx = i[0].numpy()
    
    mmd_mmd.append(square_mmd_fine(y,xx,'gaussian'))
    ce_mmd.append(model.logp(xx).mean())

ce_mmd = np.array(ce_mmd)
mmd_mmd = np.array(mmd_mmd)

index = np.arange(0, max_iter, 1)
index2 = index*100

mmd_evi=np.zeros(max_iter)
mmd_svgd=np.zeros(max_iter)
mmd_lmc=np.zeros(max_iter)
ce_evi=np.zeros(max_iter)
ce_svgd=np.zeros(max_iter)
ce_lmc=np.zeros(max_iter)
for idx,i in enumerate(index):
    x = evolve_evi[i,:,:]
    mmd_evi[idx]=square_mmd_fine(y,x,'gaussian')
    ce_evi[idx] = model.logp(x).mean()
    
for idx,i in enumerate(index2):
    x = evolve_svgd[i,:,:]
    mmd_svgd[idx]=square_mmd_fine(y,x,'gaussian')
    ce_svgd[idx] = model.logp(x).mean()
    
for idx,i in enumerate(index2):
    x = evolve_lmc[i,:,:]
    mmd_lmc[idx]=square_mmd_fine(y,x,'gaussian')
    ce_lmc[idx] = model.logp(x).mean()
    
m_mmd = mmd_mmd + const
m_evi = mmd_evi + const
m_svgd = mmd_svgd + const
m_lmc = mmd_lmc + const

me = 50

plt.plot(index, m_lmc, label = 'LMC', marker = 'p', markevery = me, mfc = 'white')
plt.plot(index, m_mmd, label = 'EVI-MMD', marker = 'o', markevery = me, mfc = 'white')
plt.plot(index, m_evi, label = 'EVI-Im', marker = 's', markevery = me, mfc = 'white')
plt.plot(index, m_svgd, label = 'SVGD' , marker = 'v', markevery = me, mfc = 'white')

plt.xlabel('Iterations')
plt.ylabel('Discrepancy^2')
plt.yscale('log')
plt.xlim(0,500)
plt.legend()
plt.savefig(dtype+'_mmd.png')
plt.show()


plt.plot(index,  -ce_lmc, label = 'LMC', marker = 'p', markevery = me, mfc = 'white')
plt.plot(index,  -ce_mmd, label = 'EVI-MMD', marker = 'o', markevery = me, mfc = 'white')
plt.plot(index,  -ce_evi, label = 'EVI-Im', marker = 's', markevery = me, mfc = 'white')
plt.plot(index,  -ce_svgd, label = 'SVGD', marker = 'v', markevery = me, mfc = 'white')
plt.axhline(-ce_true, linestyle = '--')
plt.xlabel('Iterations')
plt.ylabel('Cross Entropy')
#plt.xlim(0)
#plt.yscale('log')
plt.legend()
plt.savefig(directory_plot+dtype+'_crossentropy.png')
plt.show()

#x = evolve[-1,:,:]
dtype = 'wave'
model = model2
x = particles[2][5000][0]

ngrid = 500
xx = np.linspace(-6, 6, ngrid)
yy = np.linspace(-6, 6, ngrid)
X, Y = np.meshgrid(xx, yy)
Z = np.exp(model.logp(np.vstack((np.ndarray.flatten(X), np.ndarray.flatten(Y))).T)).reshape(ngrid, ngrid)

fig,axe = plt.subplots(1,1,figsize = (12,6))
axe.contourf(X,Y,Z,64)
axe.scatter(x[:,0],x[:,1],c='red', marker='x', s=16, alpha=1.0)
axe.set_aspect('equal', 'box')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
axe.set_ylim(-3,3)
axe.set_xlim(-6,6)
fig.savefig(directory_plot+dtype+'iter5000_evi')
plt.show()

