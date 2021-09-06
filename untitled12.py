# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 13:42:12 2021

@author: Rylei
"""

from scipy.spatial import Delaunay
import numpy as np
from collections import defaultdict
import numpy as np
import Functions as fun
import UnorderedMesh as UM
from scipy.spatial import Delaunay
from itertools import chain
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyopoly1 import LejaPoints as LP
from pyopoly1 import LejaPoints as LP
from scipy.interpolate import griddata
import random
random.seed(10)
import ICMeshGenerator as M
import Circumsphere as CS
from itertools import combinations
from collections import defaultdict

def alpha_shape_3D(pos, alpha):
    """
    Compute the alpha shape (concave hull) of a set of 3D points.
    Parameters:
        pos - np.array of shape (n,3) points.
        alpha - alpha value.
    return
        outer surface vertex indices, edge indices, and triangle indices
    """

    tetra = Delaunay(pos)
    # Find radius of the circumsphere.
    # By definition, radius of the sphere fitting inside the tetrahedral needs 
    # to be smaller than alpha value
    # http://mathworld.wolfram.com/Circumsphere.html
    tetrapos = np.take(pos,tetra.vertices,axis=0)
    normsq = np.sum(tetrapos**2,axis=2)[:,:,None]
    ones = np.ones((tetrapos.shape[0],tetrapos.shape[1],1))
    a = np.linalg.det(np.concatenate((tetrapos,ones),axis=2))
    Dx = np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[1,2]],ones),axis=2))
    Dy = -np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[0,2]],ones),axis=2))
    Dz = np.linalg.det(np.concatenate((normsq,tetrapos[:,:,[0,1]],ones),axis=2))
    c = np.linalg.det(np.concatenate((normsq,tetrapos),axis=2))
    r = np.sqrt(Dx**2+Dy**2+Dz**2-4*a*c)/(2*np.abs(a))

    # Find tetrahedrals
    tetras = tetra.vertices[r<alpha,:]
    # triangles
    TriComb = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    Triangles = tetras[:,TriComb].reshape(-1,3)
    Triangles = np.sort(Triangles,axis=1)
    # Remove triangles that occurs twice, because they are within shapes
    TrianglesDict = defaultdict(int)
    for tri in Triangles:TrianglesDict[tuple(tri)] += 1
    Triangles=np.array([tri for tri in TrianglesDict if TrianglesDict[tri] ==1])
    #edges
    EdgeComb=np.array([(0, 1), (0, 2), (1, 2)])
    Edges=Triangles[:,EdgeComb].reshape(-1,2)
    Edges=np.sort(Edges,axis=1)
    Edges=np.unique(Edges,axis=0)

    Vertices = np.unique(Edges)
    return Vertices,Edges,Triangles

dimension = 3
minDistanceBetweenPoints = 0.1
meshRadius = 0.4
mesh = M.NDGridMesh(dimension, minDistanceBetweenPoints, meshRadius, UseNoise = False)

Del = Delaunay(mesh) # Form triangulation

Vertices,Edges,Triangles = alpha_shape_3D(mesh, minDistanceBetweenPoints)

if dimension ==3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(mesh[:,0], mesh[:,1], mesh[:,2], c='k', marker='.')
    for e in Vertices:
            x,y,z = mesh[e]
            # plt.plot(x,y, 'ok')
            ax.scatter(x, y, z, c='r', marker='o')
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()