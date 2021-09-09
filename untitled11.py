from DTQAdaptive import DTQ
import numpy as np
from DriftDiffFunctionBank import FourHillDrift, DiagDiffptSevenFive
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import Class_Parameters as Param
from Errors import ErrorValsExact
from exactSolutions import TwoDdiffusionEquation
from scipy.special import erf
from NDFunctionBank import SimpleDriftSDE

dimension = 2
fun = SimpleDriftSDE(0,0.5,dimension)
mydrift = fun.Drift
mydiff = fun.Diff

'''Initialization Parameters'''
NumSteps = 5
'''Discretization Parameters'''
# a = 1
h=0.01
#kstepMin = np.round(min(0.15, 0.144*mydiff(np.asarray([0,0]))[0,0]+0.0056),2)
beta = 3
# radius = 0.55 # R
SpatialDiff = False
conditionNumForAltMethod = 8
# NumLejas = 30
# numPointsForLejaCandidates = 350
# numQuadFit = 350

par = Param.Parameters(fun, h, conditionNumForAltMethod, beta)
# par.radius = 0.5

Meshes, PdfTraj, LPReuseArr, AltMethod, GMat = DTQ(NumSteps, par.kstepMin, par.kstepMax, par.h, par.beta, par.radius, mydrift, mydiff, dimension, SpatialDiff, par, RetG=True)

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
    truepdf = fun.Solution(Meshes[i], (i+1)*h)
    # truepdf = ThreeDdiffusionEquation(Meshes[i], mydiff(np.asarray([0,0,0]))[0,0], (i+1)*h, mydrift(np.asarray([0,0,0]))[0,0])
    # truepdf = solution(xvec,-1,T)
    trueSoln.append(np.squeeze(np.copy(truepdf)))
    
    
LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(Meshes, PdfTraj, trueSoln,  plot=True)


# from mpl_toolkits.mplot3d.art3d import juggle_axes
# def update_graph(num):
#     # print(num)
#     # graph._offsets3d=(Meshes[num][:,0], Meshes[num][:,1],  Meshes[num][:,2])
#     # graph.set_array(PdfTraj[num])
#     ax.clear()
#     ax.set_zlim(np.min(Meshes[-1][:,2]),np.max(Meshes[-1][:,2]))
#     ax.set_xlim(np.min(Meshes[-1][:,0]),np.max(Meshes[-1][:,0]))
#     ax.set_ylim(np.min(Meshes[-1][:,1]),np.max(Meshes[-1][:,1]))
#     graph = ax.scatter3D(Meshes[num][:,0], Meshes[num][:,1],  Meshes[num][:,2], c=np.log(PdfTraj[num]), cmap='bone_r', vmax=max(np.log(PdfTraj[0])), vmin=0, marker=".")

#     # graph.set_data(Meshes[num][:,0], Meshes[num][:,1])
#     # graph.set_3d_properties(Meshes[num][:,2], color=PdfTraj[num], cmap='binary')
#     # title.set_text('3D Test, time={}'.format(num))
#     return graph

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# title = ax.set_title('3D Test')
# ax.set_zlim(np.min(Meshes[-1][:,2]),np.max(Meshes[-1][:,2]))
# ax.set_xlim(np.min(Meshes[-1][:,0]),np.max(Meshes[-1][:,0]))
# ax.set_ylim(np.min(Meshes[-1][:,1]),np.max(Meshes[-1][:,1]))


# ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj), interval=1000, blit=False)
# plt.show()








