# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 13:33:36 2023

@author: sadifenchen
"""

import torch
import pandas as pd

import numpy as np

from utlities.kernels import mmd2,gaussiankernel,energykernel,distancekernel,nonisogauskernel,rationalkernel
from utlities.losses import const_calculator

import matplotlib.pyplot as plt

d = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
c = [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2]


mmd_mmd = []
mmd_energy = []
mmd_support = []

energy_mmd = []
energy_energy = []
energy_support = []
for i,j in zip(d,c):
    name = str(i)+ ',' + str(j)
    
    y = torch.load(r"C:\Users\sadif\Desktop\mac drop off\AMMD_v2\code_submit\server try\output/" + name + "_y.pt")
    x_hist = torch.load(r"C:\Users\sadif\Desktop\mac drop off\AMMD_v2\code_submit\server try\output/" + name + ".pt")
    x_hist1 = torch.load(r"C:\Users\sadif\Desktop\mac drop off\AMMD_v2\code_submit\server try\output/" + str(i) + "e.pt")

    df = pd.read_csv(r"C:\Users\sadif\Desktop\mac drop off\AMMD_v2\R\output_" + str(i) + ",-0.5.csv", index_col=0)
    df2 = torch.tensor(df.values)
    kernel = gaussiankernel
    batch_size = 50000
    n_particles = 5000
    
    idx = np.random.choice(batch_size,n_particles, replace = False)
    #df_part= df[idx]
    y_part = y[idx]
    
    const1 = const_calculator(y_part, 1, kernel)
    mmd_support.append(mmd2(y_part, df2, const=const1, mmd_h = 1, method = kernel).detach().numpy())
    mmd_mmd.append(mmd2(y_part, x_hist[-1][0], const = const1, mmd_h = 1, method = kernel).detach().numpy())
    mmd_energy.append(mmd2(y_part, x_hist1[-1][0], const = const1, mmd_h = 1, method = kernel).detach().numpy())


    const2 = const_calculator(y_part, 1, energykernel)
    energy_mmd.append(mmd2(y_part, df2, const=const2, mmd_h=1, method = energykernel).detach().numpy())
    energy_energy.append(mmd2(y_part, x_hist[-1][0], const = const2, mmd_h = 1, method = energykernel).detach().numpy())
    energy_support.append(mmd2(y_part, x_hist1[-1][0], const = const2, mmd_h = 1, method = energykernel).detach().numpy())

    
plt.plot(mmd_support, label = 'Support Point', marker = 'o',  mfc = 'white')
plt.plot(mmd_energy, label = 'EVI-Energy Distance', marker = 'v',  mfc = 'white')
plt.plot(mmd_mmd, label = 'EVI-MMD', marker = 's',  mfc = 'white')
plt.legend()
plt.ylabel('Discrepancy_Gaussian^2')
plt.xlabel('Dimensions')
plt.yscale('log')
plt.xticks(np.arange(10), d)
plt.savefig('mmd2.png')
plt.show()

plt.plot(energy_support, label = 'Support Point', marker = 'o',  mfc = 'white')
plt.plot(energy_energy, label = 'EVI-Energy Distance', marker = 'v',  mfc = 'white')
plt.plot(energy_mmd, label = 'EVI-MMD', marker = 's',  mfc = 'white')
plt.legend()
plt.ylabel('Discrepancy_energy distance^2')
plt.xlabel('Dimensions')
plt.xticks(np.arange(10), d)
plt.yscale('log')
plt.savefig('mmd2_energy.png')
plt.show()