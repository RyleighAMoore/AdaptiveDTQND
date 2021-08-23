#https://stackoverflow.com/questions/26303878/alpha-shapes-in-3d
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

dimension = 3
minDistanceBetweenPoints = 0.1
meshRadius = 0.4
noise = False
mesh = M.NDGridMesh(dimension, minDistanceBetweenPoints, meshRadius, UseNoise = noise)

Del = Delaunay(mesh) # Form triangulation
radii = []
for verts in Del.simplices:
    c, r = CS.circumsphere(mesh[verts])
    radii.append(r)
  
r = np.asarray(radii)
r = np.nan_to_num(r)
import matplotlib.pyplot as plt
alpha = minDistanceBetweenPoints*2
tetras = Del.vertices[r<alpha,:]

vals = np.asarray(list(range(0,dimension+1)))
TriComb = np.asarray(list(combinations(vals, dimension)))

Triangles = tetras[:,TriComb].reshape(-1,dimension)
Triangles = np.sort(Triangles,axis=1)
# Remove triangles that occurs twice, because they are within shapes
TrianglesDict = defaultdict(int)
for tri in Triangles:
    TrianglesDict[tuple(tri)] += 1
    
Triangles=np.array([tri for tri in TrianglesDict if TrianglesDict[tri] ==1])
#edges
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
    
    
# dist = []
# for e in Vertices:
#     x,y,z,w = mesh[e]
#     dist.append(np.sqrt(x**2+y**2+z**2+w**2))
#     print(np.sqrt(x**2 + y**2 + z**2+w**2))
    
# dist2 = []
# for e in range(len(mesh)):
#     x,y,z,w = mesh[e]
#     dist2.append(np.sqrt(x**2+y**2+z**2+w**2))
#     print(np.sqrt(x**2 + y**2 + z**2+w**2))
    