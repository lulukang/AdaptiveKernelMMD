# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 16:23:40 2022

@author: sadif
"""
import torch
import numpy as np


from torchvision.datasets import FashionMNIST,MNIST,CIFAR10,CIFAR100

class fashion():
    def __init__(self,directory="./"):
        self.data=FashionMNIST(root=directory,download=True).data

        self.N=self.data.size(0)
        self.shape=self.data.size()[1:4]
        
    def sample(self,n,normalize=True):
        idx = np.random.choice(np.arange(self.N), n, replace=False)        
        drawed_sample=self.data[idx]
        
        Ny,d1,d2=drawed_sample.size()
        d=d1*d2
        
        if normalize:
            return drawed_sample.view([Ny,d])/255
        else:
            return drawed_sample.view([Ny,d])
        
class mnist():
    def __init__(self,directory="./",normalize=True):
        self.data=MNIST(root=directory,download=True).data

        self.N=self.data.size(0)
        self.shape=self.data.size()[1:4]
        
    def sample(self,n,normalize=True):
        idx = np.random.choice(np.arange(self.N), n, replace=False)        
        drawed_sample=self.data[idx]
        
        Ny,d1,d2=drawed_sample.size()
        d=d1*d2
        
        if normalize:
            return drawed_sample.view([Ny,d])/255
        else:
            return drawed_sample.view([Ny,d])
    
class cifar10():
    def __init__(self,directory="./",normalize=True):
        self.data=torch.tensor(CIFAR10(root=directory,download=True).data)

        #self.N=self.data.size(0)
        self.N = 5000
        self.shape=self.data.size()[1:4]
        
    def sample(self,n,normalize=True):
        idx = np.random.choice(np.arange(self.N), n, replace=False)        
        drawed_sample=self.data.data[idx]
        
        Ny,d1,d2,d3=drawed_sample.size()
        d=d1*d2*d3
        
        if normalize:
            return drawed_sample.reshape([Ny,d])/255
        else:
            return drawed_sample.reshape([Ny,d])
        
class cifar100():
    def __init__(self,directory="./",normalize=True):

        self.data=torch.tensor(CIFAR10(root=directory,download=True).data)

        self.N=self.data.size(0)
        self.shape=self.data.size()[1:4]
        
    def sample(self,n,normalize=True):
        idx = np.random.choice(np.arange(self.N), n, replace=False)        
        drawed_sample=self.data.data[idx]
        
        Ny,d1,d2,d3=drawed_sample.size()
        d=d1*d2*d3
        
        if normalize:
            return drawed_sample.reshape([Ny,d])/255
        else:
            return drawed_sample.reshape([Ny,d])
        
