# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 18:57:47 2020

@author: Rylei
"""
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm, trange
import random
from scipy.spatial import Delaunay
import pickle
import os
import datetime
import time
import pickle
import scipy as sp
from scipy.stats import multivariate_normal
from sympy import Matrix
from families import HermitePolynomials
import indexing
import LejaPoints as LP
import opolynd

H = HermitePolynomials(rho=0)
d=2
k = 60    
ab = H.recurrence(k+1)
lambdas = indexing.total_degree_indices(d, k)
H.lambdas = lambdas
Kmax = len(H.lambdas)-1
samples, two = LP.getLejaPoints(Kmax+1, np.asarray([[0,0]]).T,H, num_candidate_samples = 10000, candidateSampleMesh = [], returnIndices = False)

def getImplicitQuad(D):
    # plt.scatter(samples[:,0], samples[:,1])
    
    initSamples = samples[:D+1,:]
    otherSamples = np.ndarray.tolist(samples[D+1:,:])
    
    vmat = opolynd.opolynd_eval(initSamples, H.lambdas[:len(initSamples)+3,:], H.ab, H).T
    
    # weights = np.asarray([(1/(D+1))*np.ones(len(initSamples))]).T
    # weights = weights / np.sum(weights)
    rv = multivariate_normal([0, 0], [[1, 0], [0, 1]])
    weights = np.asarray([rv.pdf(initSamples)]).T
    
    nodes = np.copy(initSamples)
    for K in range(D, Kmax): # up to Kmax - 1
        # print(K)
        # Add Node
        nodes = np.vstack((nodes, otherSamples.pop()))
        # one = ((K+1)/(K+2))*weights
        # two = np.asarray([[1/(K+2)]])
        # weights = np.concatenate((one, two))
        weights = np.asarray([rv.pdf(nodes)]).T
        weights = weights / np.sum(weights)
    
    
        
        # Update weights
        vmat = opolynd.opolynd_eval(nodes, H.lambdas[:len(nodes)-1,:], H.ab, H).T
    
        nullspace = sp.linalg.null_space(vmat)
    
    
        c = np.asarray([nullspace[:,0]]).T
        
        a = weights/c
        aPos = np.ma.masked_where(c<0, a) # only values where c > 0 
        alpha1 = np.min(aPos.compressed())
        
        aNeg =  np.ma.masked_where(c>0, a) # only values where c < 0 
        alpha2 = np.max(aNeg.compressed())
        
        # Choose alpha1 or alpha2
        alpha = alpha2
        
        # Remove Node
        vals = weights <= alpha*c
        # print(np.min(weights - alpha1*c))
        assert np.isclose(np.min(weights - alpha1*c),0)
        # print(np.sum(vals))
        idx = np.argmax(vals)
        if (np.sum(vals)) !=1:
            idx = np.argmin(weights - alpha*c)
            # print("No w_k is less than  alpha_k*c_k", np.min(weights - alpha*c))
        # print(alpha1, alpha2)
        assert alpha2 < alpha1
        nodes = np.delete(nodes, idx, axis=0)
        
        weights = weights - alpha*c
        assert weights[idx] < 10**(-15)
        weights = np.delete(weights, idx, axis=0)
        # print(np.sum(weights))
        
    return weights, nodes
    
# weights, nodes = getImplicitQuad(30)

vals = [10,30,50,100,200,300,500,800]
# vals = [50]
values = []
for i in vals:
    print(i)
    weights, nodes = getImplicitQuad(i)
    tt = np.dot(nodes[:,0]**4, weights)
    values.append(tt)
    
plt.figure()
plt.loglog([10,30,50,100,200,300,500],np.abs(np.asarray(values)-3))

# plt.figure()
# plt.scatter(samples[:,0], samples[:,1], marker='*', label = 'Samples')
# plt.scatter(nodes[:,0], nodes[:,1], label='Chosen Mesh')
# plt.legend()
# plt.show()

sigma = 1
var = sigma**2

rv = multivariate_normal([0,0], [[var, 0], [0, var]])
vals1 = np.asarray([rv.pdf(nodes)])

print(np.dot(vals1, weights))
print(np.dot(nodes[:,0], weights))
print(np.dot(np.ones(len(nodes)), weights))


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(nodes[:,0], nodes[:,1], vals1, c='r', marker='.')

  

plt.figure()
plt.scatter(np.reshape(nodes[:,0],-1), np.reshape(nodes[:,1],-1), c=np.reshape(weights,-1), s=300, cmap="summer", edgecolor="k")
plt.colorbar(label="values")

plt.show()
    
    

    
    
    
    
    