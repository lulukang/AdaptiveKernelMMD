import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from utlities.plots import generative_plot

from utlities.importer import fashion,mnist,cifar10,cifar100

from utlities.kernels import mmd2

import pandas as pd

from geomloss import SamplesLoss

df = pd.read_csv (r"C:\Users\sadif\Desktop\school folder\particle based evi\R\data3.csv")

support = torch.tensor(df.to_numpy()[:,1:])



directory = './output/8/output/'

# names = ['cifar103', 'cifar107', 'cifar108', 'fashion3', 'fashion4',
#         'fashion7', 'fashion8', 'mnist3', 'mnist7', 'mnist8']

names = ['cifar101']

#names = ['mnist1', 'mnist2','mnist3']

#names = ['cifar104', 'cifar105', 'cifar106']

data=cifar10('./data')

# calculate frechet inception distance


#loss = torch.nn.KLDivLoss(reduction="batchmean")
loss = SamplesLoss("energy", blur=0.5)

#support = torch.load(r'C:\Users\sadif\Desktop\school folder\particle based evi\R\data1.csv')

mmd = []
particles = []
for i in names:
    mmd.append(torch.load(directory+i+'_mmd.pt'))
    particles.append(torch.load(directory+i+'.pt'))
    
# for name,i in zip(names,mmd):
#    # i[0]=i[0][0].detach().numpy()
#     #print(name,min(i))
#     x=[j[0] for j in i]
    
#     plt.figure(1)
#     plt.plot(x,label=name)
#     #plt.yscale('log')
#     plt.legend()
#     # ihat=signal.savgol_filter(i,11,3)
#     # plt.figure(2)
#     # plt.plot(ihat,label=name)
#     # #plt.yscale('log')
#     # plt.legend()

for i in particles:
    generative_plot(i[-1][0],data.shape,save=False)

generative_plot(support,data.shape,save=False)

# l1 = []
# l2 = []
# for j in particles:
#     for i in range(20):
#         y=data.sample(100).double()
#         x=j[-1][0]

#         l1.append(loss(support,y).detach().numpy())
#         l2.append(loss(x,y).detach().numpy())

# l1 = np.array(l1)
# l2 = np.array(l2)
# plt.boxplot([l1,l2])

# x = particles[0][-1][0]
# image = x.detach().numpy()

# image[image<0]=0
# image[image>1]=0.999999

# for i in range(100):
#     plt.imshow(image[i, :].reshape([32,32,3]), cmap = "Greys_r")
#     plt.axis('off')
#     plt.savefig('cifar1/'+str(i)+'.png')
