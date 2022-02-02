# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 10:07:50 2021

@author: Rylei
"""
import numpy as np
from time import time
D =2
N = 1000

times = []
time2s= []
from tqdm import trange
for j in trange(200):
    tensor = np.random.rand(N,D,D)
    mesh = np.random.rand(D,N)
    start = time()
    vv = np.dot(tensor,mesh)
    end = time()
    total2 = end-start
    time2s.append(total2)



    vals = []
    start = time()
    for i in range(N):
        cov = tensor[i,:,:]
        val = cov @ mesh
        vals.append(np.copy(val))
    end = time()
    total = end-start
    times.append(total)


print("tensor time:")
print(np.average(np.asarray(time2s)))
print("loop time:")
print(np.average(np.asarray(times)))



