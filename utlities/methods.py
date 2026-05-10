# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 12:58:03 2022

@author: sadif
"""

import numpy as np
import timeit
import matplotlib.pyplot as plt

def acceleration_sample_step(y, x_init, lr = 1, h = 0.2, a = 1, tol = 1e-3,
                        adagrad=True,replace=True):
    
    
    Ny,d = y.shape
    n_particles = x_init.shape[0]
    
    x = x_init
    
    adag = np.zeros([n_particles, d])
    

    #print(i)
    diff = x[:, None, :] - x[None, :, :]
    
    if h=='adap':
        diff_norm=np.linalg.norm(diff,axis=2)
        h=0.1*np.median(diff_norm)
    
    diff_xy = x[:, None, :] - y[None, :, :]
    kxy = np.exp(-np.sum(diff ** 2, axis=-1) / (2 * h ** 2)) 
    k_xiyj = np.exp(-np.sum(diff_xy ** 2, axis=-1) / (2 * h ** 2)) 
    
    gradK = -diff * kxy[:, :, None] / h ** 2
    grad_K_xiyj = - diff_xy * k_xiyj[:, :, None] / h ** 2
    
    dxkxy = np.sum(gradK, axis=1)
    dxk_xiyj = np.sum(grad_K_xiyj, axis=1)
    
    cross=2 / Ny * dxk_xiyj
    square=2 / n_particles* dxkxy
    

    #print(i)
    cross_norm=np.linalg.norm(cross,axis=1)
    outlier=cross_norm < tol
    cross[outlier]=cross[outlier]*a/(np.moveaxis(np.tile(cross_norm[outlier],[d,1]),0,1))

    v=square-cross
    
    adag += v ** 2
    if adagrad:
        x = x - lr * v / np.sqrt(adag + 1e-12)
    else:
        x = x - lr * v

        
    return x



def acceleration_distribution_step(model, x_init, normalizing = 1, Ny = 500, lr = 0.5, h = 0.2, 
                                    acceratepara=[0.1,1],
                                    adagrad=True,replace=False,accerate=False,adap=False):
    n_particles,d=x_init.shape
    x=x_init
    
    adag = np.zeros([n_particles, d])
    
    mean = np.zeros(d)
    
    cov = np.eye(d)*(h**2)
    
    t=np.random.multivariate_normal(mean,cov,Ny)
    
    diff = x[:, None, :] - x[None, :, :]
    
    if replace:
        t = np.random.multivariate_normal(np.array([0,0]),np.array([[h**2,0],[0,h**2]]),Ny)
        
    if adap:
        diff_norm=np.linalg.norm(diff,axis=2)
        h=np.median(diff_norm)**2/4
    
     
    tt=t[None, :, :] + x[:, None, :]
    kxy = np.exp(-np.sum(diff ** 2, axis=-1) / (2 * h ** 2)) / np.power(np.pi * 2.0 * h * h, d / 2)
    gradK = -diff * kxy[:, :, None]/h**2
    dxkxy = np.sum(gradK, axis=1)
    square=2 / n_particles  * dxkxy
    f=np.moveaxis(np.tile(np.exp(np.asarray(list(map(model.logp,tt)))),[2,1,1]),0,2)
    diff_xt=np.tile(t,[n_particles,1,1])/h**2
    gradf_xt=diff_xt*f
    cross=2*np.mean(gradf_xt,axis=1)/normalizing

    cross_norm=np.linalg.norm(cross,axis=1)
    outlier=cross_norm < acceratepara[0]
    cross[outlier]*=acceratepara[1]/np.moveaxis(np.tile(cross_norm[outlier],[d,1]),0,1)
            
    #update the location of each particle via forward euler 
    v=square-cross

    adag += v ** 2
    if adagrad:
        x = x - lr * v / np.sqrt(adag + 1e-12)
    else:
        x=x-lr*v
         
    return x
    
    

def trainer_distribution(model,x_init,outer_iter,normalizing=1,Ny=500,lr1=0.5,lr2=0.1,h=0.2,adagrad=True,replace=False,accerate=False,adap=False,acceratepara=[50,0.1,1],intspecial=False):
    
    #Set parameters and initialize particles
    n_particles,d=x_init.shape
    x=x_init
    
    #the evolution of particles is stored in EVOLVEX
    EVOLVEX = np.zeros([outer_iter+1, n_particles, d])
    EVOLVEX[0,:,:]=x_init
    adag = np.zeros([n_particles, d])
    
    #recore elapsed time
    time_spent=[0]
    start=timeit.default_timer()
    
    #generate a set of random variables to evaluate the cross term
    t=np.random.multivariate_normal(np.array([0,0]),np.array([[h**2,0],[0,h**2]]),Ny)
    
    #start the forward euler
    for i in range(outer_iter):
        diff = x[:, None, :] - x[None, :, :]
        
        #determing if we want to generate a new sample to calculate cross term for each iteration
        if replace:
            t = np.random.multivariate_normal(np.array([0,0]),np.array([[h**2,0],[0,h**2]]),Ny)
        
        #if h=adap we will adjust the value of h based on the distance between particles for current iteration
        if adap:
            diff_norm=np.linalg.norm(diff,axis=2)
            if intspecial:
                h=np.median(diff_norm)**2/4
            else:
                h=0.1*np.median(diff_norm)        
        
        #evaluate cross and square term
        tt=t[None, :, :] + x[:, None, :]
        kxy = np.exp(-np.sum(diff ** 2, axis=-1) / (2 * h ** 2)) / np.power(np.pi * 2.0 * h * h, d / 2)
        gradK = -diff * kxy[:, :, None]/h**2
        dxkxy = np.sum(gradK, axis=1)
        square=2 / n_particles  * dxkxy
        f=np.moveaxis(np.tile(np.exp(np.asarray(list(map(model.logp,tt)))),[20,1,1]),0,2)
        diff_xt=((np.sqrt(2*np.pi)*h)**d)*np.tile(t,[n_particles,1,1])/h**2
        gradf_xt=diff_xt*f
        cross=2*np.mean(gradf_xt,axis=1)/normalizing
        if accerate: #if accerate is true we will enlarge the value of cross term to accerate euler.
            if i<acceratepara[0]:
                cross_norm=np.linalg.norm(cross,axis=1)
                outlier=cross_norm<acceratepara[1]
                cross[outlier]*=acceratepara[2]/np.moveaxis(np.tile(cross_norm[outlier],[d,1]),0,1)
        
            if i==acceratepara[0]:
                adag=np.zeros([n_particles,d])
                
        #update the location of each particle via forward euler 
        v=square-cross
        print(i)
        adag += v ** 2
        if adagrad:
            if i>=acceratepara[0]:
                x = x - lr2 * v / np.sqrt(adag + 1e-12)
            else:
                x=x-lr1*v
        else:
            x = x - lr1 * v
        EVOLVEX[i+1,:,:]=x
        plt.scatter(x[:,0],x[:,1])
        plt.show()
        #record time elapsed for each iteration
        stop=timeit.default_timer()
        time_spent.append(stop-start)
        
    return x
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    