# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 18:16:26 2021

@author: Rylei
"""
from DTQAdaptive import DTQ
import numpy as np
from DriftDiffFunctionBank import FourHillDrift, DiagDiffptSevenFive
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import ParametersClass as Param
from Errors import ErrorValsExact
from exactSolutions import TwoDdiffusionEquation
from NDFunctionBank import SimpleDriftSDE
from time import time
from DTQTensorized import ApproxExactSoln
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


dimension =1
sde = SimpleDriftSDE(1,0.5,dimension)


def driftfun(mesh):
      if mesh.ndim ==1:
        mesh = np.expand_dims(mesh, axis=0)
    # return 0*np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
    # return -1*mesh
      return 0.2*mesh*(4-mesh**2)
    
def difffun(mesh):
    return np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
    # return np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
    # return np.expand_dims(np.asarray(0.5*np.asarray(np.ones((np.size(mesh))))),1)

ApproxSoln = True
h = 0.01
EndTime = 2
minDistanceBetweenPoints = 0.06 # lambda
meshRadius = 3 # R
SpatialDiff = False
conditionNumForAltMethod = 10
NumLejas = 5
numPointsForLejaCandidates = 50
numQuadFit = 30
meshAM = M.NDGridMesh(dimension, minDistanceBetweenPoints, meshRadius, UseNoise = False)
mesh = meshAM[2:-2]

theta = 0.5
a1 = fun.alpha1(theta)
a2 = fun.alpha2(theta)

startTime = time()
fun.computeN2s(mesh, meshAM, h, driftfun, difffun, SpatialDiff, theta, a1, a2, dimension, minDistanceBetweenPoints)
print(f'baseline is {time()-startTime}')

    
    
    
    
    