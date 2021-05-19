# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:50:31 2020

@author: Ryleigh
"""
import numpy as np
import Functions as fun
from pyopoly1.Scaling import GaussScale

def MatrixMultiplyDTQ(NumSteps, kstep, h, drift, diff, xmin, xmax, ymin, ymax):
    mydrift = drift
    mydiff = diff    
    
    X, Y = np.mgrid[xmin:xmax+kstep/2:kstep, ymin:ymax+kstep/2:kstep]
    mesh = np.vstack([X.ravel(), Y.ravel()]).T
    
    
    scale = GaussScale(2)
    scale.setMu(h*mydrift(np.asarray([0,0])).T)
    scale.setCov((h*mydiff(np.asarray([0,0]))*mydiff(np.asarray([0,0])).T).T)
    pdf = fun.Gaussian(scale, mesh)

    '''Initialize Transition probabilities'''
    GMat = np.empty([len(mesh), len(mesh)])
    for i in range(len(mesh)):
        v = kstep**2*fun.G(i,mesh, h, mydrift, mydiff)
        GMat[i,:len(v)] = v
    
          
    surfaces = [] 
    surfaces.append(np.copy(pdf))
    t=0
    while t < NumSteps-1: # Since one time step happens before
        pdf = np.matmul(GMat, pdf)
        surfaces.append(np.copy(pdf))
        t=t+1
    return mesh, surfaces