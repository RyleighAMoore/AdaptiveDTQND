# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:50:31 2020

@author: Ryleigh
"""
import numpy as np
import Functions as fun
from pyopoly1.Scaling import GaussScale
import numpy as np
import Functions as fun
from scipy.spatial import Delaunay
import LejaQuadrature as LQ
from pyopoly1.families import HermitePolynomials
from pyopoly1 import indexing
import MeshUpdates2D as MeshUp
from pyopoly1.Scaling import GaussScale
import ICMeshGenerator as M
from pyopoly1.LejaPoints import getLejaSetFromPoints, getLejaPoints
import matplotlib.pyplot as plt


def MatrixMultiplyDTQ(NumSteps, kstep, h, drift, diff, meshRadius, TimeStepType, dimension, minDistanceBetweenPoints, numPointsForLejaCandidates):
    ''' Initializd orthonormal Polynomial family'''
    poly = HermitePolynomials(rho=0)
    d=dimension
    k = 40    
    lambdas = indexing.total_degree_indices(d, k)
    poly.lambdas = lambdas
    
    mydrift = drift
    mydiff = diff    
    
    #X, Y = np.mgrid[xmin:xmax+kstep/2:kstep, ymin:ymax+kstep/2:kstep]
    # mesh = np.vstack([X.ravel(), Y.ravel()]).T
    mesh = M.NDGridMesh(dimension, minDistanceBetweenPoints, meshRadius, UseNoise = False)

    
    scale = GaussScale(dimension)
    scale.setMu(h*drift(np.zeros(dimension)).T)
    scale.setCov((h*diff(np.zeros(dimension))*diff(np.zeros(dimension)).T).T)
    # scale = GaussScale(2)
    # scale.setMu(h*mydrift(np.asarray([0,0])).T)
    # scale.setCov((h*mydiff(np.asarray([0,0]))*mydiff(np.asarray([0,0])).T).T)
    
    pdf = fun.Gaussian(scale, mesh)
   
    maxDegFreedom = len(mesh)*8
    SpatialDiff = False

    if TimeStepType == "EM":
        GMat = fun.GenerateEulerMarMatrix(maxDegFreedom, mesh, h, drift, diff, SpatialDiff)
    elif TimeStepType == "AM":
        GMat = fun.GenerateAndersonMatMatrix(h, drift, diff, mesh, dimension, poly, numPointsForLejaCandidates, maxDegFreedom, minDistanceBetweenPoints)


    # '''Initialize Transition probabilities'''
    # GMat = np.empty([len(mesh), len(mesh)])
    # for i in range(len(mesh)):
    #     v = kstep**2*fun.G(i,mesh, h, mydrift, mydiff)
    #     GMat[i,:len(v)] = v
    
    GMat = GMat[:len(mesh), :len(mesh)]    
    surfaces = [] 
    surfaces.append(np.copy(pdf))
    t=0
    while t < NumSteps-1: # Since one time step happens before
        pdf = kstep*np.matmul(GMat, pdf)
        surfaces.append(np.copy(pdf))
        t=t+1
    return mesh, surfaces


