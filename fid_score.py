 # -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 15:54:11 2022

@author: sadif
"""


import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
from keras.datasets import cifar10,fashion_mnist,mnist

import torch
import pandas as pd
import matplotlib.pyplot as plt
from utlities.importer import fashion,mnist,cifar10,cifar100

#df = pd.read_csv (r"C:\Users\sadif\Desktop\school folder\particle based evi\R\data1.csv")

#support = torch.tensor(df.to_numpy()[:,1:])

directory = './output/8/output/'

names = ['fashion1']

x = torch.load(directory+names[0]+'.pt')[-1][0]

support = torch.load('./output/9/output/'+names[0]+'.pt')[-1][0]

data=fashion('./data')

support_1 = pd.read_csv('./output/generative/data1.csv')
support_2 = pd.read_csv('./output/generative/data2.csv')
 
# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)
 
# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid
 
# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
# load cifar10 images
#(images1, _), (images2, _) = cifar10.load_data()
images2 = x.detach().numpy()
#images1 = data.sample(100).double().detach().numpy()
images3 = support.detach().numpy()
images4 = support_2.to_numpy()
# shuffle(images1)
# images1 = images1[:1000]
#print('Loaded', images1.shape, images2.shape)
# convert integer to floating point values
#images1 = images1.astype('float32')
images2 = images2.astype('float32')
images3 = images3.astype('float32')
images4 = images4.astype('float32')
#resize images
#images1 = scale_images(images1, (299,299,3))
images2 = scale_images(images2, (299,299,3))
images3 = scale_images(images3, (299,299,3))
images4 = scale_images(images4, (299,299,3))
#print('Scaled', images1.shape, images2.shape)
# pre-process images
#images1 = preprocess_input(images1)
images2 = preprocess_input(images2)
images3 = preprocess_input(images3)
images4 = preprocess_input(images4)
# calculate fid

f1 = []
f2 = []
f3 = []

for i in range(10):
    images1 = data.sample(500).detach().numpy()
    images1 = images1.astype('float32')
    images1 = scale_images(images1,(299,299,3))
    images1 = preprocess_input(images1)
    f1.append(calculate_fid(model, images1, images2))
    f2.append(calculate_fid(model, images1, images3))
    #f3.append(calculate_fid(model, images1, images4))
    print(i,f1[-1],f2[-1])
    # print('FID: %.3f' % fid)
    # print('FID2: %.3f' % fid2)

# images1 = data.sample(1000).detach().numpy()
# images1 = images1.astype('float32')
# images1 = scale_images(images1,(299,299,3))
# images1 = preprocess_input(images1)

# calculate_fid(model,images1,images2)
# calculate_fid(model,images1,images3)

fig, ax = plt.subplots()

ax.boxplot([f1,f2])
ax.set_xticklabels(['EVI-MMD-Gaussian', 'EVI-MMD-Energy distance'])
ax.set_ylabel('fid score')
fig.savefig('histogram_fashion')