# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 16:21:41 2022

@author: sadif
"""

import torch
import numpy as np
from scipy.stats.qmc import Sobol
from scipy.stats import norm
from scipy.stats import multivariate_normal

from torch.distributions import MultivariateNormal, Exponential

class funnel():
    def __init__(self, d):
        self.d = d

    def sample(self, n_samples, key=None):
        a = torch.distributions.normal.Normal(0,1)
        y = (a.sample([n_samples])*3)[:,None]
        x = a.sample([n_samples,self.d-1])*torch.exp(y/2)
        return torch.cat([y,x],dim = 1)
    
    def logp_tensor(self,x):
        pass

class multiexp():
    def __init__(self,dim = 2):
        self.d = dim
        
    def sample(self,n=100):
        a = Exponential(torch.tensor([1.0]))
        return a.sample([n,self.d]).squeeze()
        
    def logp_tensor(self, x):
        #z = torch.heaviside(x,torch.tensor([1],dtype=torch.float64))
        xx = (-x).exp()
        #y = z*xx
        return xx.prod()

class multigaussian():
    def __init__(self,dim = 2):
        self.d = dim
        self.mean = np.zeros(dim)
        self.cov = np.eye(dim)
        self.a = MultivariateNormal(torch.tensor(self.mean),torch.tensor(self.cov))
        
    def sample(self,n=200):
        return self.a.sample([n])
        
    def logp(self,x):
        return multivariate_normal.pdf(x, mean=self.mean, cov=self.cov)
    
    def logp_tensor(self, x):
        pdfi = (self.a).log_prob(x)
        #pdfi = torch.exp(-(x**2).sum(dim=1)/(2)) / ((2*np.pi)**(self.d/2))
        return pdfi

class onegaussian():
    def __init__(self,scales=[1,1]):
        self.d=2
        self.s=scales
        
    def sample(self,n=4):
        sob=Sobol(2,scramble=True)
        x=sob.random_base2(n)
        for i in range(2):
            x[:,i]=norm.ppf(x[:,i])*self.s[i]
        return torch.tensor(x)
    


class twogaussian():
    def __init__(self,scales=[0.05,1]):
        self.d=2
        self.s=scales
        self.cov=np.array([[self.s[0],0],[0,self.s[1]]])
        
        
    def sample(self,n=4):
        sob=Sobol(2)
        x=sob.random_base2(n)
        for i in range(2):
            x[:,i]=norm.ppf(x[:,i])*self.s[i]
        x1=x+np.tile([1,0],(2**n,1))
        x2=x-np.tile([1,0],(2**n,1))
        x3=np.concatenate([x1,x2],axis=0)
        return torch.tensor(x3)

class multigaussian_multidimension():
    def __init__(self, d = 10, outer_radius = 4, inner_radius = 0.2, n = 8):
        self.d = d
        self.n = n
        self.cov = np.eye(d)*inner_radius
        self.mean = np.eye(d)[:self.n]
        
    def logp_tensor(self, x):
        n, d = x.shape
        Fx = torch.zeros([n])
        for k in range(self.n):
            a = MultivariateNormal(torch.tensor(self.mean[k]),torch.tensor(self.cov))
            pdfi = a.log_prob(x).exp()
            Fx += pdfi
        return Fx / self.n
            

class eightgaussian():     
    def __init__(self,outer_radius=4,inner_radius=0.02):
        self.d=2
        self.cov=np.array([[0.2,0],[0,0.2]])
        span=np.array([0,outer_radius])
        theta=np.pi/4
        
        rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        
        self.mean=[span]
        for i in range(7):
            span=np.matmul(rotation,span)
            self.mean.append(span)
            
            
    def sample(self,n_samples):
        n= int(np.ceil((n_samples/8)))
        x= np.zeros([8 * n, 2])
        for k in range(8):
            x[k*n:(k+1)*n, :] = np.random.multivariate_normal(self.mean[k], self.cov, n)
        np.random.shuffle(x)
        x=x[:n_samples,:]
        return torch.tensor(x)
    
    def logp(self,x):
        n, d = x.shape
        Fx = np.zeros(n)
        for k in range(8):
            pdfi = multivariate_normal.pdf(x, mean=self.mean[k], cov=self.cov)
            Fx += pdfi
        return np.log(Fx / 8)

    def logp_tensor(self,x):
        n, d = x.shape
        Fx = torch.zeros([n])
        for k in range(8):
            a = MultivariateNormal(torch.tensor(self.mean[k]),torch.tensor( self.cov))
            pdfi = a.log_prob(x).exp()
            Fx += pdfi
        return Fx / 8
    
    
class waveshape(object):
    dimension = 2
    def logp(self,x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        w1=np.sin(np.pi*x1)
        return -(0.1*x1**2+(x2-w1)**2)
    
    def logp_tensor(self,x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        w1=torch.sin(torch.pi*x1)
        return (-(0.1*x1**2+(x2-w1)**2)).exp()/9.93
        		#log_p = -0.5*(x2-np.sin(np.pi*x1/2))**2/0.16
    
    def grad_log_p(self,x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        grad1 = (-0.2*x1 + 2 *np.pi*np.cos(np.pi* x1)*(x2 - np.sin(np.pi*x1)))
        grad2 = -2*(x2-np.sin(np.pi*x1))
        return np.array([grad1, grad2]).T
    
    
class star_gaussian(object):
    def __init__(self, skewness = 100, n = 5):
        self.d = 2
        self.dimension = 2
        self.K = n
        theta = 2*np.pi/n
        U = np.array([[np.cos(theta), np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
        self.mu = np.zeros([self.K, self.d])
        self.sigma = np.zeros([self.K, self.d, self.d])
        self.inv_sigma = np.zeros_like(self.sigma)
              
        self.mu[0, :] = 1.5 * np.array([1., 0.])
        self.sigma[0, :, :] = np.diag([1., 1./skewness])
        self.inv_sigma[0, :, :] = np.diag([1., skewness])
              
        for i in range(1, n):
            self.mu[i, :] = np.matmul(U, self.mu[i-1, :])
            self.sigma[i, :, :] = np.matmul(U, np.matmul(self.sigma[i-1, :, :], U.T))
            self.inv_sigma[i, :, :] = np.matmul(U, np.matmul(self.inv_sigma[i-1, :, :], U.T))
              
        self.mean = np.mean(self.mu)
        self.x2 = self.mean * self.mean * self.K
        for i in range(self.K):
            self.x2 += np.diag(self.sigma[i, :, :])
        self.x2 /= self.K

    def sample(self, n_samples):
        n = int(n_samples/self.K)
        x = np.zeros([self.K * n, self.d])
        for k in range(self.K):
            x[k*n:(k+1)*n, :] = np.random.multivariate_normal(self.mu[k, :], self.sigma[k, :, :], n)
        np.random.shuffle(x)
        return x

    def logp(self, x):
        n, d = x.shape
        Fx = np.zeros(n)
        for k in range(self.K):
            pdfi = multivariate_normal.pdf(x, mean=self.mu[k, :], cov=self.sigma[k, :, :])
            Fx += pdfi
        return np.log(Fx / self.K)
    
    def logp_tensor(self,x):
        n, d = x.shape
        Fx = torch.zeros([n])
        for k in range(self.K):
            a = MultivariateNormal(torch.tensor(self.mu[k,:]),torch.tensor(self.sigma[k,:,:]))
            pdfi = a.log_prob(x).exp()
            Fx += pdfi
        return Fx / self.K
    
    
class gaus(object):
    def __init__(self,n =2):
        self.dimension=n
        
    def p(self,x):
        pdfi=multivariate_normal.pdf(x,mean=np.zeros([1,self.dimension]).flatten(),cov=np.eye(self.dimension))
        return pdfi
    
    def logp(self,x):
        pdfi=multivariate_normal.pdf(x,mean=np.zeros([1,self.dimension]).flatten(),cov=np.eye(self.dimension))
        
        return np.log(pdfi)
    
    def logp_tensor(self,x):
        a = MultivariateNormal(torch.tensor(np.zeros(self.dimension)),torch.tensor(np.eye(self.dimension)))
        pdfi = a.log_prob(x)
        
        return pdfi.exp()
    
    def grad_log_p(self,x):
        pdfi = multivariate_normal.pdf(x, np.zeros([1,self.dimension]).flatten(), cov=np.eye(self.dimension))
        Fx=pdfi
        Jx= pdfi[:, None] * np.matmul(np.zeros([1,self.dimension]).flatten() - x, np.eye(self.dimension))
        return Jx/Fx
    
