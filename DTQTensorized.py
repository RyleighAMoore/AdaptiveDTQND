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


def MatrixMultiplyDTQ(NumSteps, kstep, h, drift, diff, meshRadius, TimeStepType, dimension):
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
    mesh = M.NDGridMesh(dimension, kstep, meshRadius, UseNoise = False)

    
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
        GMat = fun.GenerateAndersonMatMatrix(h, drift, diff, mesh, dimension, maxDegFreedom, kstep)


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

import numpy as np
from scipy.interpolate import griddata, interp2d

def ApproxExactSoln(EndTime, drift, diff, TimeStepType, dimension, Meshes, PdfTraj, Times):
    kstep = 0.005
    h = 0.001
    NumSteps = int(np.ceil(EndTime/h))+1
    meshesm = abs(min(np.min(Meshes[-1]), np.min(Meshes[0])))+2
    meshesM = abs(max(np.max(Meshes[-1]), np.max(Meshes[0])))+2
    TimeStepType = "EM"
    
    
    meshRadius = np.ceil(max(meshesM, meshesm))

    mesh, surfaces = MatrixMultiplyDTQ(NumSteps, kstep, h, drift, diff, meshRadius, TimeStepType, dimension)
    
    solnIndices = Times/h -1
    print("We assume Times/h is an integer right now.")
    surfaces2 = []
    for i in solnIndices:
        surfaces2.append(surfaces[int(i)])
    solns = []
    for i in range(len(surfaces2)):
        gridSolnOnLejas = griddata(mesh, surfaces2[i], Meshes[i], method='cubic', fill_value=-1)
        solns.append(np.squeeze(gridSolnOnLejas))
    
    LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(Meshes, PdfTraj, solns, h, plot=False)
    
    return LinfErrors, L2Errors, L1Errors, L2wErrors, solns
from DTQAdaptive import DTQ
import numpy as np
from DriftDiffFunctionBank import FourHillDrift, DiagDiffptSevenFive
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import ParametersClass as Param
from Errors import ErrorValsExact
from exactSolutions import TwoDdiffusionEquation
from NDFunctionBank import SimpleDriftSDE
import time

EndTime = 1 
dimension = 1
sde = SimpleDriftSDE(0.5,0.5,dimension)
drift = sde.Drift
diff = sde.Diff

TimeStepType= "EM" 
    
    
    
    
    
    
    
    
    
    