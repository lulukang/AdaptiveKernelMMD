# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 18:02:29 2022

@author: sadif
"""

import itertools

import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

def generative_plot(y,y_scope=[28,28],size=[10,10],title='result',save=True):
    image=y.detach().numpy()
    
    image[image<0]=0
    image[image>1]=0.999999
    
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
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(image[i, :].reshape(y_scope), cmap = "Greys_r")
        ax.axis('off')
    if save:    
        fig.savefig('output/'+title+'.png',dpi=300)
    
    
def toy_plot(x,y,title='scatter.png'):
    y=y.detach().numpy()
    x=x.detach().numpy()
    
    plt.scatter(y[:,0],y[:,1],s=20,alpha=0.2)
    plt.scatter(x[:,0],x[:,1],s=10)
    plt.savefig('output/'+title+'.png',dpi = 300)
    plt.show()
    
    
    
def toy_anime(x_hist,title='result'):
    y=x_hist[0].detach().numpy()
    fig, ((ax)) = plt.subplots(ncols=1, nrows=1, gridspec_kw={'wspace': 0.2, 'hspace': 0.2},  figsize=(15, 15))
    scat=ax.scatter(y[:,0],y[:,1], c='yellow', marker='x', s=10, alpha=1.0)
    def animate(i):
        y=x_hist[i].detach().numpy()
        scat.set_offsets(np.c_[y[:,0],y[:,1]])
    
    anim = FuncAnimation(fig, animate, interval=100, frames=len(x_hist))
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=50, bitrate=1800)
    anim.save('output/'+title+'.mp4', writer=writer)
    
    return