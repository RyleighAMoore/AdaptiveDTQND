# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:01:58 2019

@author: Ryleigh
"""
import numpy as np
#mesh : points in Nx2 array 
def fillDistance(mesh):
    minVals = []
    for i in range(len(mesh)):
        vecX = mesh[i,0]*np.ones(len(mesh))  
        vecY = mesh[i,1]*np.ones(len(mesh))
        vals = np.sqrt((vecX - mesh[:,0])**2 +(vecY - mesh[:,1])**2)
        sortedVals = sorted(vals)
        minVals.append(np.copy(np.min(sortedVals[1]))) #Take 1 since 0 in at 0 index since same point
    return np.max(minVals)
        
def fillDistance2(mesh, mesh2):
    minVals = []
    for i in range(len(mesh)):
        vecX = mesh[i,0]*np.ones(len(mesh2))  
        vecY = mesh[i,1]*np.ones(len(mesh2))
        vals = np.sqrt((vecX - mesh2[:,0])**2 +(vecY - mesh2[:,1])**2)
        sortedVals = sorted(vals)
        minVals.append(np.copy(np.min(sortedVals[1]))) #Take 1 since 0 in at 0 index since same point
    return np.max(minVals)
# fillDist = fillDistance(Meshes[-1])

def separationDistance(mesh):
    minVals = []
    for i in range(len(mesh)):
        vecX = mesh[i,0]*np.ones(len(mesh))  
        vecY = mesh[i,1]*np.ones(len(mesh))
        vals = np.sqrt((vecX - mesh[:,0])**2 +(vecY - mesh[:,1])**2)
        sortedVals = sorted(vals)
        minVals.append(np.copy(np.min(sortedVals[1]))) #Take 1 since 0 in at 0 index since same point
    return np.min(minVals)

# separationDist = separationDistance(Meshes[-1])
def fillDistanceAvg(mesh):
    minVals = []
    for i in range(len(mesh)):
        vecX = mesh[i,0]*np.ones(len(mesh))  
        vecY = mesh[i,1]*np.ones(len(mesh))
        vals = np.sqrt((vecX - mesh[:,0])**2 +(vecY - mesh[:,1])**2)
        sortedVals = sorted(vals)
        minVals.append(np.copy(np.min(sortedVals[1]))) #Take 1 since 0 in at 0 index since same point
    return np.percentile(minVals,80)

# minmin = []
# maxmin = []
# for i in range(len(Meshes)):
#     x = Meshes[i][:,0]
#     y = Meshes[i][:,1]
#     m = separationDistance(Meshes[i])
#     M = fillDistance(Meshes[i])
#     minmin.append(m)
#     maxmin.append(M)

# plt.figure()
# plt.plot(range(len(Meshes)), minmin)
# plt.plot(range(len(Meshes)), maxmin)


    
    