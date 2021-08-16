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

dimension = 4
minDistanceBetweenPoints = 0.1
meshRadius = 0.4
mesh = M.NDGridMesh(dimension, minDistanceBetweenPoints, meshRadius, UseNoise = False)

# def add_edge(edges, i, j, only_outer=True):
#        """
#        Add an edge between the i-th and j-th points,
#        if not in the list already
#        """
#        if (i, j) in edges or (j, i) in edges:
#            # already added
#            assert (j, i) in edges, "Can't go twice over same directed edge right?"
#            if only_outer:
#                # if both neighboring triangles are in shape, it's not a boundary edge
#                edges.remove((j, i))
#            return
#        edges.add((i, j))

Del = Delaunay(mesh) # Form triangulation
radii = []
for verts in Del.simplices:
    c, r = CS.circumsphere(mesh[verts])
    radii.append(r)
  
r = np.asarray(radii)
import matplotlib.pyplot as plt
alpha = minDistanceBetweenPoints
tetras = Del.vertices[r<alpha,:]

# edges = set()
# for rr in range(len(radii)):
#     if radii[rr] < alpha:
#         ia, ib, ic = tetras.simplices[rr]
#         add_edge(edges, ia, ib)
#         add_edge(edges, ib, ic)
#         add_edge(edges, ic, ia)
        
        
 # triangles
# TriComb = np.array([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
vals = np.asarray(list(range(0,dimension+1)))
TriComb = np.asarray(list(combinations(vals, dimension)))

Triangles = tetras[:,TriComb].reshape(-1,dimension)
Triangles = np.sort(Triangles,axis=1)
# Remove triangles that occurs twice, because they are within shapes
TrianglesDict = defaultdict(int)
for tri in Triangles:TrianglesDict[tuple(tri)] += 1
Triangles=np.array([tri for tri in TrianglesDict if TrianglesDict[tri] ==1])
#edges
# EdgeComb=np.array([(0, 1), (0, 2), (1, 2)])
vals = np.asarray(list(range(0,dimension)))
EdgeComb = np.asarray(list(combinations(vals, dimension-1)))

Edges=Triangles[:,EdgeComb].reshape(-1,dimension-1)
Edges=np.sort(Edges,axis=1)
Edges=np.unique(Edges,axis=0)

Vertices = np.unique(Edges)
        
if dimension ==2:
    plt.scatter(mesh[:,0], mesh[:,1])
    for i in Vertices:
        plt.scatter(mesh[i,0], mesh[i,1], c='r')
        
        
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
    
if dimension ==4:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(mesh[:,0], mesh[:,1], mesh[:,3], c="k", marker='.')
    for e in Vertices:
            x,y,z,w = mesh[e]
            # plt.plot(x,y, 'ok')
            ax.scatter(x, y, w, c="r", marker="o")
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()
    
    
dist = []
for e in Vertices:
    x,y,z,w = mesh[e]
    dist.append(np.sqrt(x**2+y**2+z**2+w**2))
    print(np.sqrt(x**2 + y**2 + z**2+w**2))
    
dist2 = []
for e in range(len(mesh)):
    x,y,z,w = mesh[e]
    dist2.append(np.sqrt(x**2+y**2+z**2+w**2))
    print(np.sqrt(x**2 + y**2 + z**2+w**2))
    