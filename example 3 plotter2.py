# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 14:30:11 2022

@author: sadif
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from utlities.plots import generative_plot

from utlities.importer import fashion,mnist,cifar10,cifar100

from utlities.kernels import mmd2



directory = './output/8/output/'

names = 'fashion1'

data = torch.load(directory+names+'.pt')
    
idx = ['init', 5, 10, 15, 20, 25, 100, 199, 299, 399, 499]

loc = [1, 10, 20, 31, 45, 54, 62, 79, 81, 90]

y_scope=[28,28]
size=[10,10]

tionary = dict(data)

my_dict = {value: key for key, value in tionary.items()}

aspect = 1.
n = size[0]
m = size[1]
bottom = 0.1; left=0.05
top=1.-bottom; right = 1.-0.18
fisasp = (1-bottom-(1-top))/float( 1-left-(1-right) )
wspace=0
hspace=wspace/float(aspect)
figheight= 4
figwidth = (m + (m-1)*wspace)/float((n+(n-1)*hspace)*aspect)*figheight*fisasp
fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(figwidth, figheight))
plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, 
                    wspace=wspace, hspace=hspace)

index_of = 0

for i in range(10):
    x = my_dict[idx[i]]
    
    x = x.detach().numpy()
    
    x[x < 0] = 0
    x[x > 1] = 0.9999999
    for j in range(10):
        z = x[loc[j]].reshape(y_scope)
        axes[j][i].imshow(z , cmap = 'Greys_r')
        axes[j][i].axis('off')

fig.savefig('output/'+names+'.png',dpi=300)