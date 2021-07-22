# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 12:06:44 2021

@author: Rylei
"""

import ICMeshGenerator as M
import Functions as F

dimension = 1
minDistanceBetweenPoints = 0.03
meshRadius = 1
mesh = M.NDGridMesh(dimension, minDistanceBetweenPoints, meshRadius, UseNoise = False)

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
from DTQAdaptive import DTQ
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import ICMeshGenerator as MG
import QuadraticFit as QF
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
from pyopoly1 import variableTransformations as VT
from pyopoly1.QuadratureRules import QuadratureByInterpolationND_KnownLP
"""
Created on Fri Apr  3 12:44:33 2020
@author: Rylei
"""
from pyopoly1 import variableTransformations as VT
import numpy as np
import matplotlib.pyplot as plt
from pyopoly1 import opolynd
from mpl_toolkits.mplot3d import Axes3D
from Functions import *
from pyopoly1.Scaling import GaussScale
from pyopoly1.Plotting import productGaussians2D
import UnorderedMesh as UM
from pyopoly1.families import HermitePolynomials
import pyopoly1.indexing
import pyopoly1.LejaPoints as LP
from QuadraticFit import leastSquares, ComputeDividedOut
from scipy.interpolate import griddata
import math
np.seterr(divide='ignore', invalid='ignore')




from NDFunctionBank import SimpleDriftSDE
dimension = 1
sde = SimpleDriftSDE(1,1,dimension)
theta = 0.5
a1 = F.alpha1(theta)
a2 = F.alpha2(theta)
h=0.01

# indexOfMesh=int(len(mesh)/2)
# indexOfMesh2 = int(len(mesh)/2)

# val, scaleComb = F.AndersonMattingly(indexOfMesh, indexOfMesh2, mesh, h, sde.Drift, sde.Diff, False, theta, a1, a2, dimension)

# val = np.expand_dims(val,1)
# scale1, LSFit, Const, combinations = QF.leastSquares(mesh, val)

# vals = QF.ComputeDividedOut(mesh, LSFit, Const, scale1, combinations)

# final = val/vals.T


poly = HermitePolynomials(rho=0)
d=dimension
k = 40    
lambdas = indexing.total_degree_indices(d, k)
poly.lambdas = lambdas

'''Generate Alt Leja Points'''
# lejaPointsFinal, new = getLejaPoints(10, np.zeros((dimension,1)), poly, num_candidate_samples=5000, candidateSampleMesh = [], returnIndices = False)

# mesh = VT.map_from_canonical_space(lejaPointsFinal, scale1)
ii=2
ALp = np.zeros((len(mesh)-ii*2, len(mesh)-ii*2))
for i in range(ii,len(mesh)-ii):
    for j in range(ii,len(mesh)-ii):
        indexOfMesh= mesh[j]
        indexOfMesh2 =mesh[i]
        
        val, scaleComb = F.AndersonMattingly(indexOfMesh, indexOfMesh2, mesh, h, sde.Drift, sde.Diff, False, theta, a1, a2, dimension)
        val = np.expand_dims(val,1)
        
        scale1, LSFit, Const, combinations = QF.leastSquares(mesh, val)
        
        vals = QF.ComputeDividedOut(mesh, LSFit, Const, scale1, combinations)
        
        lejaPointsFinal, new = getLejaPoints(10, np.zeros((dimension,1)), poly, num_candidate_samples=5000, candidateSampleMesh = [], returnIndices = False)
        meshLP = VT.map_from_canonical_space(lejaPointsFinal, scale1)
        
        valLP, temp = F.AndersonMattingly(indexOfMesh, indexOfMesh2, meshLP, h, sde.Drift, sde.Diff, False, theta, a1, a2, dimension)
        valLP = np.expand_dims(valLP,1)
        
        valsLP = QF.ComputeDividedOut(meshLP, LSFit, Const, scale1, combinations)
        
        
        finalLP = valLP/valsLP.T
        
        # value, condNum = QuadratureByInterpolationND_KnownLP(poly, scale1, meshLP, pdf2, LejaIndices)
        V = opolynd.opolynd_eval(meshLP, poly.lambdas[:len(meshLP),:], poly.ab, poly)
        
        vinv = np.linalg.inv(V)
          
        c = np.matmul(vinv[0,:], finalLP)
        ALp[i-ii,j-ii] = c
        print(c)
        print(np.sum(np.abs(vinv[0,:])))










