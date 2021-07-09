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
import itertools


def NDGridMesh(dimension, stepsize, radius, UseNoise = True):
    subdivision = radius/stepsize+1
    step = radius/subdivision
    grid= np.mgrid[tuple(slice(step - radius, radius, step * 2) for _ in range(dimension))]
    mesh = []
    for i in range(grid.shape[0]):
        new = grid[i].ravel()
        if UseNoise:
            noise = np.random.normal(0,1, size = (len(grid),2))
            meshSpacing = stepsize
            noise = np.random.uniform(-meshSpacing, meshSpacing,size = (len(new)))
            
            shake = 0.2
            noise = -meshSpacing*shake +(meshSpacing*shake - - meshSpacing*shake)/(np.max(noise)-np.min(noise))*(noise-np.min(noise))
            new = new+noise
        mesh.append(new)
    grid = np.asarray(mesh).T
    return grid

grid = NDGridMesh(3, 0.028, .3, UseNoise = True)
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')
graph = ax.scatter3D(grid[:,0], grid[:,1],  grid[:,2], marker=".")



