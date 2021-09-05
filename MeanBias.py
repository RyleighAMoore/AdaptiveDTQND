import numpy as np
from scipy.stats import multivariate_normal
import ICMeshGenerator as M
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
import random

def Gaussian(scaling, mesh):
    rv = multivariate_normal(scaling.mu.T[0], scaling.cov)        
    soln_vals = np.asarray([rv.pdf(mesh)]).T
    soln = np.squeeze(soln_vals)
    return soln


sigma = 1
Delta = 0.1

dimension=1
minDistanceBetweenPoints=0.001
meshRadius=1
# mesh = M.NDGridMesh(dimension, minDistanceBetweenPoints, meshRadius, UseNoise = False)
numPoints = 10000
individualMethod =[]
groupMethod =[]
numGroupSamples = 10000

diffs = []
ApproxSpacing = 0.0001

meshMin = -10
xx=[0.5,1]
for sigma in xx:
    print(sigma)
    for i in range(5):
        xbar1 = random.gauss(0, sigma)
        numPoints =  int((xbar1+2*sigma - meshMin)/ApproxSpacing)
        mesh = np.linspace(meshMin, xbar1+2*sigma, num=numPoints)
        spacing  = mesh[1]-mesh[0]
        
        scale = GaussScale(dimension)
        scale.setMu(np.asarray([xbar1+Delta]))
        scale.setCov(np.asarray([[sigma**2]]))
        PDF = Gaussian(scale, mesh)
        assert np.isclose(PDF[0],0)
        # assert np.max(PDF)>1

        integral1 = np.sum(spacing*PDF)
        
        numPoints =  int((xbar1-2*sigma - meshMin)/ApproxSpacing)
        mesh = np.linspace(meshMin, xbar1-2*sigma, num=numPoints)
        spacing  = mesh[1]-mesh[0]
        PDF2 = Gaussian(scale, mesh)
        
        integral2 = np.sum(spacing*PDF2)
        
        total = 1-integral1+integral2
        individualMethod.append(np.copy(total))
    
        groupSamples =[]
        for k in range(numGroupSamples):
            xbar0 = random.gauss(0, sigma)
            groupSamples.append(xbar0)
            
        xbar0 = np.average(np.asarray(groupSamples))
        
        numPoints =  int((xbar0+2*sigma - meshMin)/ApproxSpacing)
        mesh = np.linspace(meshMin, xbar0+2*sigma, num=numPoints)
        spacing  = mesh[1]-mesh[0]
        
        scale = GaussScale(dimension)
        scale.setMu(np.asarray([xbar1+Delta]))
        scale.setCov(np.asarray([[sigma**2]]))
        PDF = Gaussian(scale, mesh)
        assert np.isclose(PDF[0],0)
        # assert np.max(PDF)>1
        
        integral1 = np.sum(spacing*PDF)
        
        numPoints =  int((xbar0-2*sigma - meshMin)/ApproxSpacing)
        mesh = np.linspace(meshMin, xbar0-2*sigma, num=numPoints)
        spacing  = mesh[1]-mesh[0]
        PDF2 = Gaussian(scale, mesh)

        
        integral2 = np.sum(spacing*PDF2)
        
        total = 1-integral1+integral2
        groupMethod.append(np.copy(total))
    

    diffs.append(np.max(abs(np.asarray(individualMethod)- np.asarray(groupMethod))))
 
plt.scatter(np.asarray(xx),diffs)


