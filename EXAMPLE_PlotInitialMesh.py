import matplotlib.pyplot as plt
import matplotlib.animation as animation


from Functions import nDGridMeshCenteredAtOrigin

mesh1D = nDGridMeshCenteredAtOrigin(1, 2, 0.2)
mesh2D = nDGridMeshCenteredAtOrigin(2, 2, 0.2)
mesh3D = nDGridMeshCenteredAtOrigin(3, 2, 0.5)


import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from mpl_toolkits.mplot3d.axes3d import get_test_data
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure(figsize=plt.figaspect(0.5))

ax = fig.add_subplot(1, 3, 1, aspect='equal')
ax.scatter(0*mesh1D, mesh1D, c='w')
ax.scatter(mesh1D, 0*mesh1D)


ax = fig.add_subplot(1, 3, 2, aspect='equal')
ax.scatter(mesh2D[:,0], mesh2D[:,1])

ax = fig.add_subplot(1, 3, 3, projection='3d')
ax.scatter(mesh3D[:,0], mesh3D[:,1], mesh3D[:,2])
