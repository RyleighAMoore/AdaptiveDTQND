from DTQAdaptive import DTQ
import numpy as np
from DriftDiffFunctionBank import FourHillDrift, DiagDiffptSevenFive
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import ParametersClass as Param
from Errors import ErrorValsExact
from exactSolutions import TwoDdiffusionEquation


def MovingHillDrift(mesh):
    return 0*np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
    # return -1*mesh
    return mesh*(4-mesh**2)
    
def DiagDiffOne(mesh):
    return np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
    return np.expand_dims(np.asarray(np.ones((np.size(mesh)))),1)
    # return np.expand_dims(np.asarray(0.5*np.asarray(np.ones((np.size(mesh))))),1)


mydrift = MovingHillDrift
mydiff = DiagDiffOne

'''Initialization Parameters'''
NumSteps = 10
'''Discretization Parameters'''
a = 1
h=0.1
#kstepMin = np.round(min(0.15, 0.144*mydiff(np.asarray([0,0]))[0,0]+0.0056),2)
kstepMin = 0.051 # lambda
kstepMax = 0.055 # Lambda
kstepMin = 0.1 # lambda
kstepMax = 0.12 # Lambda
beta = 8
radius = 4 # R
dimension = 1
SpatialDiff = False
conditionNumForAltMethod = 10
NumLejas = 5
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

trueSoln = []
from exactSolutions import OneDdiffusionEquation
for i in range(len(Meshes)):
    truepdf = OneDdiffusionEquation(Meshes[i], DiagDiffOne(Meshes[i]), (i+1)*h, 0)
    # truepdf = solution(xvec,-1,T)
    trueSoln.append(np.squeeze(np.copy(truepdf)))
    
from Errors import ErrorValsExact
LinfErrors, L2Errors, L1Errors, L2wErrors = ErrorValsExact(Meshes, PdfTraj, trueSoln, plot=True)


# trueSoln = []
# from exactSolutions import OneDdiffusionEquation
# # for i in range(len(Meshes)):
# for i in range(1,100):
#     truepdf = OneDdiffusionEquation(Meshes[i], 1, h*i, 0)
#     # truepdf = solution(xvec,-1,T)
#     trueSoln.append(np.squeeze(np.copy(truepdf)))


def update_graph(num):
    graph.set_data(Meshes[num], PdfTraj[num])
    return title, graph
fig = plt.figure()
ax = fig.add_subplot(111)
title = ax.set_title('2D Test')
    
graph, = ax.plot(Meshes[-1], PdfTraj[-1], linestyle="", marker=".")
ax.set_xlim(-4, 4)
ax.set_ylim(0, np.max(PdfTraj[4]))


ani = animation.FuncAnimation(fig, update_graph, frames=len(PdfTraj), interval=100, blit=False)
plt.show()


# plt.figure()
# ii =0
# plt.plot(Meshes[ii], trueSoln[ii], 'or')
# plt.plot(Meshes[ii], PdfTraj[ii], '.k')
    


