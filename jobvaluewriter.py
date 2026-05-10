# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 18:05:04 2022

@author: sadif
"""

import numpy as np
import csv
import itertools


f = open('./toyjob.txt', 'a')

#ds = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

#models = ['eightgaussian', 'star_gaussian', 'waveshape']
#multiplier = [1, 2, 3, 4]
ass = [5, 4, 3, 2, 1]
cs = [-0.1, -0.2, -0.3, -0.4, -0.5]
taus = [0.1, 0.5, 1, 2]
#kernels = ['gaussiankernel','energykernel','distancekernel']
#modes = [1,2,3]



idx_table = []

for idx,permutation in enumerate(itertools.product(cs, taus, ass)):
    s = '--titleaddon=' + str(idx) + ' --c=' + str(permutation[0]) + ' --a=' + str(permutation[2]) +' --tau=' + str(permutation[1]) + ' --ini_multiplier=4'
    f.write(s+'\n')
    idx_table.append([idx,permutation[0],permutation[1],permutation[2]])

f.close()
        
with open ('Example.csv','w',newline = '') as csvfile:
    my_writer = csv.writer(csvfile, delimiter = ',')
    my_writer.writerows(idx_table)

