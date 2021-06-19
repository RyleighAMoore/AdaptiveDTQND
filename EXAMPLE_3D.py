from DTQAdaptive import DTQ
import numpy as np
from DriftDiffFunctionBank import FourHillDrift, DiagDiffptSevenFive
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import ParametersClass as Param
from Errors import ErrorValsExact
from exactSolutions import TwoDdiffusionEquation


def MovingHillDrift(mesh):
    return np.asarray([np.ones((np.size(mesh,0))), np.zeros((np.size(mesh,0))), np.zeros((np.size(mesh,0)))]).T
    # return mesh*(4-mesh**2)
    
def DiagDiffOne(mesh):
    return np.diag([0.2,0.2, 0.2])
    # return np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
    # return np.expand_dims(np.asarray(0.5*np.asarray(np.ones((np.size(mesh))))),1)


mydrift = MovingHillDrift
mydiff = DiagDiffOne

'''Initialization Parameters'''
NumSteps = 5
'''Discretization Parameters'''
a = 1
h=0.01
#kstepMin = np.round(min(0.15, 0.144*mydiff(np.asarray([0,0]))[0,0]+0.0056),2)
kstepMin = 0.03 # lambda
kstepMax = 0.03 # Lambda
beta = 5
radius = 0.2 # R
dimension = 3
SpatialDiff = False
conditionNumForAltMethod = 10
NumLejas = 10
numPointsForLejaCandidates =50
numQuadFit = 50
par = Param.Parameters(conditionNumForAltMethod, NumLejas, numPointsForLejaCandidates, numQuadFit)

Meshes, PdfTraj, LPReuseArr, AltMethod= DTQ(NumSteps, kstepMin, kstepMax, h, beta, radius, mydrift, mydiff, dimension, SpatialDiff, par, PrintStuff=True)

pc = []
for i in range(len(Meshes)-1):
    l = len(Meshes[i])
    pc.append(LPReuseArr[i]/l)
    
mean = np.mean(pc)
print("Leja Reuse: ", mean*100, "%")

pc = []
for i in range(len(Meshes)-1):
    l = len(Meshes[i])
    pc.append(AltMethod[i]/l)
    
mean2 = np.mean(pc)
print("Alt Method: ", mean2*100, "%")



# def update_graph(num):
#     graph.set_data(Meshes[num], PdfTraj[num])
#     return title, graph
# fig = plt.figure()
# ax = fig.add_subplot(111)
# title = ax.set_title('2D Test')
    
# graph, = ax.plot(Meshes[-1], PdfTraj[-1], linestyle="", marker=".")
# ax.set_xlim(-4, 4)
# ax.set_ylim(0, np.max(PdfTraj[4]))


# ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj), interval=100, blit=False)
# plt.show()

# def solution(mesh, A, t):
#     D = 1*0.5
#     r = (mesh[:,0]-A*t)**2
#     vals = np.exp(-r/(np.sqrt(4*D*t)))*(1/(4*np.sqrt(np.pi*D*t)))
#     return vals

trueSoln = []
for i in range(len(Meshes)):
    xvec = Meshes[i]
    T=(i+1)*h
    truepdf = np.exp(-xvec**2/(1 - np.exp(-2*T)))/np.sqrt(np.pi*(1-np.exp(-2*T)))
    # truepdf = solution(xvec,-1,T)
    trueSoln.append(np.squeeze(truepdf))
        
    
# LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(Meshes, PdfTraj, trueSoln,  plot=True)

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')
ii=4
x = Meshes[ii][:,0]
y = Meshes[ii][:,1]
z = Meshes[ii][:,2]
ax.scatter3D(x, y, z, c=PdfTraj[ii], cmap='binary');

from mpl_toolkits.mplot3d.art3d import juggle_axes

def update_graph(num):
    print(num)
    # graph._offsets3d=(Meshes[num][:,0], Meshes[num][:,1],  Meshes[num][:,2])
    # graph.set_array(PdfTraj[num])
    ax.clear()
    graph = ax.scatter3D(Meshes[num][:,0], Meshes[0][:,1],  Meshes[num][:,2], c=np.log(PdfTraj[num]), cmap='binary', vmax=max(np.log(PdfTraj[0])), vmin=0, marker=".")

    # graph.set_data(Meshes[num][:,0], Meshes[num][:,1])
    # graph.set_3d_properties(Meshes[num][:,2], color=PdfTraj[num], cmap='binary')
    # title.set_text('3D Test, time={}'.format(num))
    return graph

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')
    
# graph = ax.scatter3D(Meshes[0][:,0], Meshes[0][:,1],  Meshes[0][:,2], c=PdfTraj[0], cmap='binary', marker=".")
# ax.set_zlim(0, 4.5)
ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj), interval=1000, blit=False)
plt.show()

