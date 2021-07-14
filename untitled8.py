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
import Circumsphere

dimension = 2
minDistanceBetweenPoints = 0.1
meshRadius = 0.3
mesh = M.NDGridMesh(dimension, minDistanceBetweenPoints, meshRadius, UseNoise = False)

def add_edge(edges, i, j, only_outer=True):
       """
       Add an edge between the i-th and j-th points,
       if not in the list already
       """
       if (i, j) in edges or (j, i) in edges:
           # already added
           assert (j, i) in edges, "Can't go twice over same directed edge right?"
           if only_outer:
               # if both neighboring triangles are in shape, it's not a boundary edge
               edges.remove((j, i))
           return
       edges.add((i, j))

tri = Delaunay(mesh) # Form triangulation
r = []
for verts in tri.simplices:
    print(verts)
    A, B, C = mesh[verts]
    a = A-C
    b = B-C   
    
    num = np.linalg.norm(a)*np.linalg.norm(b)*np.linalg.norm(a-b)
    den = 2*np.sqrt(np.linalg.norm(a)**2*np.linalg.norm(b)**2-(np.dot(a,b)**2))

    r.append(num/den)
    
r = np.asarray(r)
import matplotlib.pyplot as plt
alpha = minDistanceBetweenPoints
edges = set()
for rr in range(len(r)):
    if r[rr] < alpha:
        ia, ib, ic = tri.simplices[rr]
        add_edge(edges, ia, ib)
        add_edge(edges, ib, ic)
        add_edge(edges, ic, ia)
        
aa = list(chain(edges))
out = [item for t in aa for item in t]
pointsOnBoundary = np.sort(out)
pointsOnBoundary = pointsOnBoundary[1::2]  # Skip every other element to remove repeated elements

plt.triplot(mesh[:,0], mesh[:,1], tri.simplices)
plt.plot(mesh[:,0], mesh[:,1], 'o') 
for e in pointsOnBoundary:
        x,y = mesh[e]
        plt.plot(x,y, 'ok')


plt.show()


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mesh[:,0], mesh[:,1], mesh[:,2], c='k', marker='o')
for e in pointsOnBoundary:
        x,y,z = mesh[e]
        # plt.plot(x,y, 'ok')
        ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
    
    
    