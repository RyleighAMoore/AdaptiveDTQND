from DTQAdaptive import DTQ
import numpy as np
from DriftDiffFunctionBank import FourHillDrift, DiagDiffptSevenFive
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import ParametersClass as Param
from Errors import ErrorValsExact
from exactSolutions import TwoDdiffusionEquation
from scipy.special import erf



def MovingHillDrift(mesh):
    if mesh.ndim ==1:
        mesh = np.expand_dims(mesh, axis=0)
    return np.asarray([np.zeros((np.size(mesh,0))), np.zeros((np.size(mesh,0))), np.zeros((np.size(mesh,0)))]).T
    return np.asarray([3*np.ones((np.size(mesh,0))), np.zeros((np.size(mesh,0))), np.zeros((np.size(mesh,0)))]).T
    # return np.asarray([mesh[:,0]*(4-mesh[:,0]**2), np.zeros((np.size(mesh,0))), np.zeros((np.size(mesh,0)))]).T
    x = mesh[:,0]
    # return np.asarray([3*erf(10*x), np.zeros((np.size(mesh,0))), np.zeros((np.size(mesh,0)))]).T
# return mesh*(4-mesh**2)
    
def DiagDiffOne(mesh):
    # return np.diag([1,1, 1])
    return np.diag([0.5, 0.5, 0.5])
    # return np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
    # return np.expand_dims(np.asarray(0.5*np.asarray(np.ones((np.size(mesh))))),1)


mydrift = MovingHillDrift
mydiff = DiagDiffOne

'''Initialization Parameters'''
NumSteps = 15
'''Discretization Parameters'''
# a = 1
h=0.01
#kstepMin = np.round(min(0.15, 0.144*mydiff(np.asarray([0,0]))[0,0]+0.0056),2)
kstepMin = 0.1 # lambda
kstepMax = 0.15 # Lambda
beta = 3
radius = 0.55 # R
dimension = 3
SpatialDiff = False
conditionNumForAltMethod = 8
NumLejas = 30
numPointsForLejaCandidates = 350
numQuadFit = 350

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



trueSoln = []
from exactSolutions import ThreeDdiffusionEquation
for i in range(len(Meshes)):
    truepdf = ThreeDdiffusionEquation(Meshes[i], DiagDiffOne(np.asarray([0,0,0]))[0,0], (i+1)*h, MovingHillDrift(np.asarray([0,0,0]))[0,0])
    # truepdf = solution(xvec,-1,T)
    trueSoln.append(np.squeeze(np.copy(truepdf)))
    
    
LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(Meshes, PdfTraj, trueSoln,  plot=True)


from mpl_toolkits.mplot3d.art3d import juggle_axes
def update_graph(num):
    # print(num)
    # graph._offsets3d=(Meshes[num][:,0], Meshes[num][:,1],  Meshes[num][:,2])
    # graph.set_array(PdfTraj[num])
    ax.clear()
    ax.set_zlim(np.min(Meshes[-1][:,2]),np.max(Meshes[-1][:,2]))
    ax.set_xlim(np.min(Meshes[-1][:,0]),np.max(Meshes[-1][:,0]))
    ax.set_ylim(np.min(Meshes[-1][:,1]),np.max(Meshes[-1][:,1]))
    graph = ax.scatter3D(Meshes[num][:,0], Meshes[num][:,1],  Meshes[num][:,2], c=np.log(PdfTraj[num]), cmap='bone_r', vmax=max(np.log(PdfTraj[0])), vmin=0, marker=".")

    # graph.set_data(Meshes[num][:,0], Meshes[num][:,1])
    # graph.set_3d_properties(Meshes[num][:,2], color=PdfTraj[num], cmap='binary')
    # title.set_text('3D Test, time={}'.format(num))
    return graph

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')
ax.set_zlim(np.min(Meshes[-1][:,2]),np.max(Meshes[-1][:,2]))
ax.set_xlim(np.min(Meshes[-1][:,0]),np.max(Meshes[-1][:,0]))
ax.set_ylim(np.min(Meshes[-1][:,1]),np.max(Meshes[-1][:,1]))


ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj), interval=1000, blit=False)
plt.show()








