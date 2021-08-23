from DTQAdaptive import DTQ
import numpy as np
from DriftDiffFunctionBank import FourHillDrift, DiagDiffptSevenFive
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import ParametersClass as Param
from Errors import ErrorValsExact
from exactSolutions import TwoDdiffusionEquation
from scipy.special import erf
from NDFunctionBank import SimpleDriftSDE

dimension =4
sde = SimpleDriftSDE(0,0.5,dimension)
mydrift = sde.Drift
mydiff = sde.Diff

# def MovingHillDrift(mesh):
#     if mesh.ndim ==1:
#         mesh = np.expand_dims(mesh, axis=0)
#     return np.asarray([np.zeros((np.size(mesh,0))), np.zeros((np.size(mesh,0))), np.zeros((np.size(mesh,0)))]).T
#     return np.asarray([3*np.ones((np.size(mesh,0))), np.zeros((np.size(mesh,0))), np.zeros((np.size(mesh,0)))]).T
#     # return np.asarray([mesh[:,0]*(4-mesh[:,0]**2), np.zeros((np.size(mesh,0))), np.zeros((np.size(mesh,0)))]).T
#     x = mesh[:,0]
#     # return np.asarray([3*erf(10*x), np.zeros((np.size(mesh,0))), np.zeros((np.size(mesh,0)))]).T
# # return mesh*(4-mesh**2)
    
# def DiagDiffOne(mesh):
#     # return np.diag([1,1, 1])
#     return np.diag([0.5, 0.5, 0.5])
#     # return np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
#     # return np.expand_dims(np.asarray(0.5*np.asarray(np.ones((np.size(mesh))))),1)


# mydrift = MovingHillDrift
# mydiff = DiagDiffOne

'''Initialization Parameters'''
NumSteps = 9
'''Discretization Parameters'''
h=0.01
kstepMin = 0.1 # lambda
kstepMax = kstepMin+0.01 # Lambda
beta = 4
radius = 0.5 # R
SpatialDiff = False
conditionNumForAltMethod = 8
NumLejas = 15
numPointsForLejaCandidates = 150
numQuadFit = 150

par = Param.Parameters(sde, h, conditionNumForAltMethod, beta)
par.kstepMin = kstepMin
par.kstepMax = kstepMax
par.radius = radius
par.numPointsForLejaCandidates = numPointsForLejaCandidates
par.NumLejas = NumLejas
par.numQuadFit = numQuadFit
TS = "EM"

Meshes, PdfTraj, LPReuseArr, AltMethod= DTQ(NumSteps, kstepMin, kstepMax, h, beta, radius, mydrift, mydiff, dimension, SpatialDiff, par, PrintStuff=True, TimeStepType = TS)

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
    truepdf = sde.Solution(Meshes[i], (i+1)*h)
    # truepdf = ThreeDdiffusionEquation(Meshes[i], mydiff(np.asarray([0,0,0]))[0,0], (i+1)*h, mydrift(np.asarray([0,0,0]))[0,0])
    # truepdf = solution(xvec,-1,T)
    trueSoln.append(np.squeeze(np.copy(truepdf)))
    
    
LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(Meshes, PdfTraj, trueSoln, h, plot=True)


if dimension ==2:
    def update_graph(num):
        graph.set_data (Meshes[num][:,0], Meshes[num][:,1])
        graph.set_3d_properties(PdfTraj[num])
        title.set_text('3D Test, time={}'.format(num))
        return title, graph
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title = ax.set_title('3D Test')
        
    graph, = ax.plot(Meshes[-1][:,0], Meshes[-1][:,1], PdfTraj[-1], linestyle="", marker=".")
    ax.set_zlim(0, 4.5)
    ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj), interval=100, blit=False)
    plt.show()
    
    
    
if dimension ==3:
    from mpl_toolkits.mplot3d.art3d import juggle_axes
    def update_graph(num):
        # print(num)
        # graph._offsets3d=(Meshes[num][:,0], Meshes[num][:,1],  Meshes[num][:,2])
        # graph.set_array(PdfTraj[num])
        indx = 0
        indy = 1
        indz = 2
        ax.clear()
        ax.set_zlim(np.min(Meshes[-1][:,indz]),np.max(Meshes[-1][:,indz]))
        ax.set_xlim(np.min(Meshes[-1][:,indx]),np.max(Meshes[-1][:,indx]))
        ax.set_ylim(np.min(Meshes[-1][:,indy]),np.max(Meshes[-1][:,indy]))
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
    
    
    
    
    
    
    
